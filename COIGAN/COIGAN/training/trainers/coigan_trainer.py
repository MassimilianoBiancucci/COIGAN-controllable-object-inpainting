import math
import random
import os
import logging

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from omegaconf import OmegaConf, read_write

from COIGAN.modules import make_generator, make_discriminator
from COIGAN.training.losses import CoiganLossManager
from COIGAN.training.logger.data_logger import DataLogger

from COIGAN.utils.ddp_utils import (
    get_rank,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
    data_sampler
)

from COIGAN.utils.common_utils import (
    requires_grad,
    accumulate,
    make_optimizer,
    sample_data
)


LOGGER = logging.getLogger(__name__)


class COIGANtrainer:

    def __init__(self, rank, config: OmegaConf, dataset):
        """ 
            Initialize the trainer

            Args:
                rank (int): rank of the process
                config (OmegaConf): configuration
                dataset (): dataset
            
        """

        self.config = config
        self.device = rank

        # create the generator, discriminator and EMA generator
        self.generator = make_generator(**config.generator).to(rank)
        self.discriminator = make_discriminator(**config.discriminator).to(rank)

        # if in the first process, create the moving average generator
        if self.device == 0:
            self.g_ema = make_generator(**config.generator).to(rank)
            self.g_ema.eval()
            # Initialize the generator moving average to be the same as the generator
            accumulate(self.g_ema, self.generator, 0)

        # create the optimizers
        self.g_optim = make_optimizer(self.generator, **config.optimizers.generator)
        self.d_optim = make_optimizer(self.discriminator, **config.optimizers.discriminator)

        # loading checkpoint
        if config.checkpoint is not None:
            self.load_checkpoint(config.checkpoint)

        # load the loss manager
        self.loss_mng = CoiganLossManager(config.losses)

        if self.config.distributed:
            self.generator = nn.parallel.DistributedDataParallel(
                self.generator,
                device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False
            )

            self.discriminator = nn.parallel.DistributedDataParallel(
                self.discriminator,
                device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False
            )
        
        # create the dataloader
        self.dataloader = data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=data_sampler(
                dataset,
                shuffle=self.config.dataloader_shuffle,
                distributed=self.config.distributed
            )
        )

        # initialize wandb
        if self.device == 0:
            self.datalogger = DataLogger(
                **self.config.logger,
                config=self.config
            )
        
    
    def load_checkpoint(self, checkpoint_path):
        """
            Method that load the checkpoint.

            Args:
                checkpoint_path (str): path to the checkpoint
        """
        LOGGER.info(f"Loading checkpoint from {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(self.config.ckpt)
            with read_write(self.config):
                self.config.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass

        self.generator.load_state_dict(ckpt["g"])
        self.discriminator.load_state_dict(ckpt["d"])
        #self.g_ema.load_state_dict(ckpt["g_ema"])

        self.g_optim.load_state_dict(ckpt["g_optim"])
        self.d_optim.load_state_dict(ckpt["d_optim"])


    def save_checkpoint(self, step_idx):
        """
            Method that save the checkpoint.

            Args:
                path (str): path to the checkpoint
                step_idx (int): current step
        """
        LOGGER.info(f"Saving checkpoint to {path}")
        ckpt = {
            "g": self.generator.state_dict(),
            "d": self.discriminator.state_dict(),
            #"g_ema": self.g_ema.state_dict(),
            "g_optim": self.g_optim.state_dict(),
            "d_optim": self.d_optim.state_dict(),
        }

        path = os.path.join(self.config.checkpoint_dir, f"{step_idx}.pt")
        torch.save(ckpt, path)


    def train(self):
        """

        """

        # create iterator with the dataloader
        loader = sample_data(self.dataloader)

        # create the progress bar
        if get_rank() == 0:
            pbar = tqdm(range(self.config.max_iter), initial=self.config.start_iter, dynamic_ncols=True, smoothing=0.01)

        # if distributed, get the module of the generator and discriminator from the DataParallel wrapper
        if self.config.distributed:
            self.g_module = self.generator.module
            self.d_module = self.discriminator.module
        else:
            self.g_module = self.generator
            self.d_module = self.discriminator
        

        for step_idx in pbar:
            i = step_idx + self.config.start_iter

            # check if done
            if i > self.config.max_iter:
                LOGGER.info("training finished!")
                break
            
            #get the data
            sample = next(loader)

            #unpack the data
            image = sample["base"] # [base_r, base_g, base_b] the original image without any masking
            gen_in = sample["gen_input"].to(self.device) # [base_r, base_g, base_b, mask_0, mask_1, mask_2, mask_3]
            gen_in_orig_masks = sample["orig_gen_input_masks"] # [mask_0, mask_1, mask_2, mask_3] the original masks without the noise
            disc_in_true = sample["disc_input"].to(self.device) # [defect_0_r, defect_0_g, defect_0_b, defect_1_r, defect_1_g, defect_1_b, defect_2_r, defect_2_g, defect_2_b, defect_3_r, defect_3_g, defect_3_b]    
            
            # train the discriminator
            requires_grad(self.discriminator, True)
            requires_grad(self.generator, False)

            #----> generate the fake images
            fake_image = self.generator(gen_in)
            
            #----> extract defects from generated
            disc_in_fake = self.extract_defects(fake_image, gen_in_orig_masks)

            #----> compute the discriminator outputs for the real and fake images
            disc_out_true = self.discriminator(disc_in_true)
            disc_out_fake = self.discriminator(disc_in_fake)

            # compute the discriminator losses
            #disc_loss = self.loss_mng.compute_discriminator_loss(disc_out_true, disc_out_fake)

            # apply regularization to the discriminator


            # train the generator
            requires_grad(self.discriminator, False)
            requires_grad(self.generator, True)

            #----> generate the fake images
            fake_image = self.generator(gen_in)

            #----> extract defects from generated
            disc_in_fake = self.extract_defects(fake_image, gen_in)

            #----> compute the discriminator outputs for the fake images
            disc_out_fake = self.discriminator(disc_in_fake)

            # compute the generator losses
            #gen_loss = self.loss_mng.compute_generator_loss(disc_out_fake)

            # apply regularization to the generator


            # update the moving average generator
            # self.update_average_generator()

            # logging, checkpointing and visualization
            if self.device == 0:

                # log the results locally and on wandb
                self.datalogger.log_step_results(
                    i, 
                    {
                    #    "disc_loss": disc_loss,
                    #    "gen_loss": gen_loss
                    }
                )

                # log the outputs 
                if i % self.visualize_interval == 0:
                    self.datalogger.log_visual_results(
                        i,
                        {

                        }
                    )

                # save the checkpoint
                if i % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(i)


    def extract_defects(self, fake_image, defect_masks):
        """
        Method that extract the generated defects from an image.

        Args:
            fake_image (torch.Tensor): fake image with dfects, generated by the generator (shape: [batch_size, 3, 256, 256])
            defect_masks (torch.Tensor): a tensor containing the masks of the defects
                                            used to extract the generated defects from the image. (shape: [batch_size, n, 256, 256])
        
        Returns:
            torch.Tensor: a tensor containing the extracted defects. (shape: [batch_size, nx3, 256, 256])
        """

        # get the number of defects
        n = defect_masks.shape[1]

        # create a tensor to store the extracted defects
        extracted_defects = torch.zeros((fake_image.shape[0], n*3, fake_image.shape[2], fake_image.shape[3]), device=fake_image.device)

        # for each defect, extract it from the generated image
        for i in range(n):
            target_areas = defect_masks[:, i] > 0
            extracted_defects[:, i*3:(i*3)+3, target_areas] = fake_image[:, :, target_areas]

        return extracted_defects
        
        



