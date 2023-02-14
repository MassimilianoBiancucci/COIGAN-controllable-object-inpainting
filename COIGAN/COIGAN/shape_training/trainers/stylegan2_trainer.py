import math
import random
import os

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from omegaconf import OmegaConf, read_write

try:
    import wandb

except ImportError:
    wandb = None

from COIGAN.utils.ddp_utils import (
    get_rank,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
    data_sampler
)

from COIGAN.modules.stylegan2.op import conv2d_gradfix
from COIGAN.modules.stylegan2.swagan import Generator, Discriminator

class stylegan2_trainer:

    def __init__(self, rank, config: OmegaConf, dataset):
        """ 
            Initialize the trainer

            Args:
                rank (int): rank of the process
                config (OmegaConf): configuration
                dataset (torch.utils.data.Dataset): dataset
            
        """

        self.config = config
        self.device = rank

        self.generator = Generator(
            self.config.size, 
            self.config.latent, 
            self.config.n_mlp, 
            channel_multiplier=self.config.channel_multiplier,
            out_channels=self.config.channels
        ).to(self.device)

        self.discriminator = Discriminator(
            self.config.size, 
            channel_multiplier=self.config.channel_multiplier,
            input_channels=self.config.channels
        ).to(self.device)

        # perche un g_ema in ogni thread invece di tenerene uno solo nel primo thread?
        self.g_ema = Generator(
            self.config.size, 
            self.config.latent, 
            self.config.n_mlp, 
            channel_multiplier=self.config.channel_multiplier,
            out_channels=self.config.channels
        ).to(self.device)

        self.g_ema.eval()
        self.accumulate(self.g_ema, self.generator, 0)

        g_reg_ratio = self.config.g_reg_every / (self.config.g_reg_every + 1)
        d_reg_ratio = self.config.d_reg_every / (self.config.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )

        self.d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        # loading checkpoint
        if self.config.ckpt is not None:
            print("load model:", self.config.ckpt)

            ckpt = torch.load(self.config.ckpt, map_location=lambda storage, loc: storage)

            try:
                ckpt_name = os.path.basename(self.config.ckpt)
                with read_write(self.config):
                    self.config.start_iter = int(os.path.splitext(ckpt_name)[0])

            except ValueError:
                pass

            self.generator.load_state_dict(ckpt["g"])
            self.discriminator.load_state_dict(ckpt["d"])
            self.g_ema.load_state_dict(ckpt["g_ema"])

            self.g_optim.load_state_dict(ckpt["g_optim"])
            self.d_optim.load_state_dict(ckpt["d_optim"])

        if self.config.distributed:
            # if using multiple GPUs, wrap the models with DistributedDataParallel
            self.generator = nn.parallel.DistributedDataParallel(
                self.generator,
                device_ids=[rank],
                output_device=rank,
                broadcast_buffers=False,
            )

            self.discriminator = nn.parallel.DistributedDataParallel(
                self.discriminator,
                device_ids=[rank],
                output_device=rank,
                broadcast_buffers=False,
            )

        # check if the dataset object has the method on_worker_init
        # if so, use it to initialize the workers
        worker_init_fn = None
        if hasattr(dataset, "on_worker_init"):
            worker_init_fn = dataset.on_worker_init

        # define the dataloader
        self.loader = data.DataLoader(
            dataset,
            batch_size=self.config.batch,
            sampler=data_sampler(dataset, shuffle=True, distributed=self.config.distributed),
            drop_last=True,
            num_workers=self.config.num_workers,
            worker_init_fn=worker_init_fn
        )

        # initialize wandb
        if get_rank() == 0 and wandb is not None and self.config.wandb:
            wandb.init(
                project=self.config.wandb_project, 
                entity=self.config.wandb_entity,
                mode=self.config.wandb_mode
            )

            wandb.config.update(
                OmegaConf.to_container(
                    self.config
                )
            )
        

    def train(self):

        loader = self.sample_data(self.loader)

        pbar = range(self.config.iter)

        if get_rank() == 0:
            pbar = tqdm(pbar, initial=self.config.start_iter, dynamic_ncols=True, smoothing=0.01)

        mean_path_length = 0

        d_loss_val = 0
        r1_loss = torch.tensor(0.0, device=self.device)
        g_loss_val = 0
        path_loss = torch.tensor(0.0, device=self.device)
        path_lengths = torch.tensor(0.0, device=self.device)
        mean_path_length_avg = 0
        loss_dict = {}

        if self.config.distributed:
            self.g_module = self.generator.module
            self.d_module = self.discriminator.module

        else:
            self.g_module = self.generator
            self.d_module = self.discriminator

        # exponential moving average of the generator weights
        accum = 0.5 ** (32 / (10 * 1000)) # = 0.9977843871
        r_t_stat = 0

        sample_z = torch.randn(self.config.n_sample, self.config.latent, device=self.device)

        # if the dataset is ImageFolder, the first element of the tuple are the images
        imgFolderDs = False
        if isinstance(self.loader.dataset, ImageFolder):
            imgFolderDs = True

        for idx in pbar:
            i = idx + self.config.start_iter

            if i > self.config.iter:
                print("Done!")

                break

            # get the data
            real_img = next(loader)

            # if the dataset is ImageFolder, the first element of the tuple are the images
            if imgFolderDs:
                real_img = real_img[0]

            #DEBUG
            #one_img = real_img[0]
            real_img = real_img.to(self.device)

            # train the discriminator
            self.requires_grad(self.generator, False)
            self.requires_grad(self.discriminator, True)

            noise = self.mixing_noise(self.config.batch, self.config.latent, self.config.mixing, self.device)
            fake_img, _ = self.generator(noise)

            real_img_aug = real_img

            fake_pred = self.discriminator(fake_img)
            real_pred = self.discriminator(real_img_aug)

            # calculate the discriminator loss
            d_loss = self.d_logistic_loss(real_pred, fake_pred)

            loss_dict["d"] = d_loss
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            # update the discriminator
            self.discriminator.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            # apply the discriminator regularizations
            d_regularize = i % self.config.d_reg_every == 0
            if d_regularize:
                real_img.requires_grad = True

                real_img_aug = real_img

                real_pred = self.discriminator(real_img_aug)
                r1_loss = self.d_r1_loss(real_pred, real_img)

                self.discriminator.zero_grad()
                r1_weighted_loss = (self.config.r1 / 2) * r1_loss * self.config.d_reg_every
                r1_weighted_loss.backward()

                self.d_optim.step()

            loss_dict["r1"] = r1_loss

            # train the generator
            self.requires_grad(self.generator, True)
            self.requires_grad(self.discriminator, False)

            noise = self.mixing_noise(self.config.batch, self.config.latent, self.config.mixing, self.device)
            fake_img, _ = self.generator(noise)

            fake_pred = self.discriminator(fake_img)

            # calculate the generator loss
            g_loss = self.g_nonsaturating_loss(fake_pred)

            loss_dict["g"] = g_loss

            self.generator.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            # apply the generator regularizations
            g_regularize = i % self.config.g_reg_every == 0

            if g_regularize:
                path_batch_size = max(1, self.config.batch // self.config.path_batch_shrink)
                noise = self.mixing_noise(path_batch_size, self.config.latent, self.config.mixing, self.device)
                fake_img, latents = self.generator(noise, return_latents=True)

                path_loss, mean_path_length, path_lengths = self.g_path_regularize(
                    fake_img, latents, mean_path_length
                )

                self.generator.zero_grad()
                weighted_path_loss = self.config.path_regularize * self.config.g_reg_every * path_loss

                if self.config.path_batch_shrink:
                    weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                weighted_path_loss.backward()

                self.g_optim.step()

                mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
                )

            loss_dict["path"] = path_loss
            loss_dict["path_length"] = path_lengths.mean()

            self.accumulate(self.g_ema, self.g_module, accum)

            loss_reduced = reduce_loss_dict(loss_dict)
            
            d_loss_val = loss_reduced["d"].mean().item()
            g_loss_val = loss_reduced["g"].mean().item()
            r1_val = loss_reduced["r1"].mean().item()
            path_loss_val = loss_reduced["path"].mean().item()
            real_score_val = loss_reduced["real_score"].mean().item()
            fake_score_val = loss_reduced["fake_score"].mean().item()
            path_length_val = loss_reduced["path_length"].mean().item()

            if get_rank() == 0:
                pbar.set_description(
                    (
                        f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                        f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    )
                )

                if i % 100 == 0:
                    with torch.no_grad():
                        self.g_ema.eval()
                        sample, _ = self.g_ema([sample_z])
                        grid_sample = utils.make_grid(
                            sample, 
                            nrow=int(self.config.n_sample ** 0.5), 
                            normalize=True, 
                            value_range=(0, 1)
                        )
                        utils.save_image(
                            grid_sample,
                            f"{self.config.sampl_dir}/{str(i).zfill(6)}.png"
                        )
                        wandb.log({"Samples": wandb.Image(grid_sample)})

                if wandb and self.config.wandb:
                    wandb.log(
                        {
                            "Generator": g_loss_val,
                            "Discriminator": d_loss_val,
                            "Rt": r_t_stat,
                            "R1": r1_val,
                            "Path Length Regularization": path_loss_val,
                            "Mean Path Length": mean_path_length,
                            "Real Score": real_score_val,
                            "Fake Score": fake_score_val,
                            "Path Length": path_length_val,
                        }
                    )

                if i % 10000 == 0:
                    torch.save(
                        {
                            "g": self.g_module.state_dict(),
                            "d": self.d_module.state_dict(), 
                            "g_ema": self.g_ema.state_dict(),
                            "g_optim": self.g_optim.state_dict(),
                            "d_optim": self.d_optim.state_dict(),
                            "self.config": self.config,
                        },
                        f"{self.config.ckpt_dir}/{str(i).zfill(6)}.pt",
                    )

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    @staticmethod
    def accumulate(model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    @staticmethod
    def sample_data(loader):
        while True:
            for batch in loader:
                yield batch

    @staticmethod
    def d_logistic_loss(real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    @staticmethod
    def d_r1_loss(real_pred, real_img):
        with conv2d_gradfix.no_weight_gradients():
            grad_real, = autograd.grad(
                outputs=real_pred.sum(), inputs=real_img, create_graph=True
            )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def g_nonsaturating_loss(fake_pred):
        loss = F.softplus(-fake_pred).mean()

        return loss

    @staticmethod
    def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):

        noise = torch.randn_like(fake_img) / math.sqrt(
            fake_img.shape[2] * fake_img.shape[3]
        )

        grad, = autograd.grad(
            outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
        )

        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

        path_penalty = (path_lengths - path_mean).pow(2).mean()

        return path_penalty, path_mean.detach(), path_lengths

    @staticmethod
    def make_noise(batch, latent_dim, n_noise, device):
        if n_noise == 1:
            return torch.randn(batch, latent_dim, device=device)

        noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

        return noises

    @staticmethod
    def mixing_noise(batch, latent_dim, prob, device):
        if prob > 0 and random.random() < prob:
            return stylegan2_trainer.make_noise(batch, latent_dim, 2, device)

        else:
            return [stylegan2_trainer.make_noise(batch, latent_dim, 1, device)]

    @staticmethod
    def set_grad_none(model, targets):
        for n, p in model.named_parameters():
            if n in targets:
                p.grad = None