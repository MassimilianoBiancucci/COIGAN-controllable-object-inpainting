import math
import random
import os
import logging

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from omegaconf import OmegaConf, read_write

from COIGAN.modules import make_generator, make_discriminator
from COIGAN.training.losses import CoiganLossManager
from COIGAN.training.logger import DataLogger

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

from COIGAN.utils.debug_utils import (
    find_null_grads,
    there_is_nan,
    check_nan
)

LOGGER = logging.getLogger(__name__)


class COIGANtrainer:

    def __init__(self, rank, config: OmegaConf, dataloader: data.DataLoader):
        """ 
            Initialize the trainer

            Args:
                rank (int): rank of the process
                config (OmegaConf): configuration
                dataloader (torch.utils.data.DataLoader): torch dataloader wrapping the dataset
        """

        self.config = config
        self.device = rank

        # store the dataloader in the trainer
        self.dataloader = dataloader

        # load the data logger variables
        self.log_img_interval = self.config.log_img_interval
        self.log_shapes_and_defects = self.config.log_shapes_and_defects

        # load checkpoint out path
        self.checkpoint_path = self.config.location.checkpoint_dir

        # other process variables
        self.mask_base_img = self.config.mask_base_img
        self.use_g_ema = self.config.use_g_ema
        self.use_ref_disc = self.config.use_ref_disc

        # create the generator, discriminator and EMA generator
        self.generator = make_generator(**config.generator).to(rank)
        self.discriminator = make_discriminator(**config.discriminator).to(rank)
        self.ref_discriminator = make_discriminator(**config.ref_discriminator).to(rank) if self.use_ref_disc else None

        # if in the first process, create the moving average generator
        if self.device == 0 and self.use_g_ema:
            self.g_ema = make_generator(**config.generator).to(rank)
            self.g_ema.eval()
            # Initialize the generator moving average to be the same as the generator
            accumulate(self.g_ema, self.generator, 0)

        # create the optimizers
        self.g_optim = make_optimizer(self.generator, **config.optimizers.generator)
        self.d_optim = make_optimizer(self.discriminator, **config.optimizers.discriminator)
        self.ref_d_optim = make_optimizer(self.ref_discriminator, **config.optimizers.ref_discriminator) if self.use_ref_disc else None

        # loading checkpoint
        if config.checkpoint is not None:
            self.load_checkpoint(config.checkpoint)
        
        # create the checkpoint dir
        os.makedirs(config.location.checkpoint_dir, exist_ok=True)

        if self.config.distributed:
            # if distributed, wrap the generator and discriminator in a data parallel wrapper
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
            self.ref_discriminator = nn.parallel.DistributedDataParallel(
                self.ref_discriminator,
                device_ids=[self.device],
                output_device=self.device,
                broadcast_buffers=False
            ) if self.use_ref_disc else None

        # save a reference to the unwrapped modules
        if self.config.distributed:
            # if distributed, save a reference of the modules of the generator and discriminator
            # outside the data parallel wrapper, will be used for save the checkpoint
            self.g_module = self.generator.module
            self.d_module = self.discriminator.module
            self.ref_d_module = self.ref_discriminator.module if self.use_ref_disc else None

        else:
            # if not distributed, save a reference of the modules of the generator and discriminator
            # anyway will be used for save the checkpoint
            self.g_module = self.generator
            self.d_module = self.discriminator
            self.ref_d_module = self.ref_discriminator
        

        # load the loss manager
        self.loss_mng = CoiganLossManager(
            **config.losses,
            generator =         self.generator,
            discriminator =     self.discriminator,
            ref_discriminator=  self.ref_discriminator,
            g_optimizer =       self.g_optim,
            d_optimizer =       self.d_optim,
            ref_d_optimizer=    self.ref_d_optim,
            device =            self.device
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
            ckpt_name = os.path.basename(checkpoint_path)
            with read_write(self.config):
                self.config.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass

        self.generator.load_state_dict(ckpt["g"])
        self.discriminator.load_state_dict(ckpt["d"])
        if self.use_g_ema: self.g_ema.load_state_dict(ckpt["g_ema"])
        if self.use_ref_disc: self.ref_discriminator.load_state_dict(ckpt["ref_d"])

        self.g_optim.load_state_dict(ckpt["g_optim"])
        self.d_optim.load_state_dict(ckpt["d_optim"])
        if self.use_ref_disc: self.ref_d_optim.load_state_dict(ckpt["ref_d_optim"])


    def save_checkpoint(self, step_idx):
        """
            Method that save the checkpoint.

            Args:
                path (str): path to the checkpoint
                step_idx (int): current step
        """
        LOGGER.info(f"Saving checkpoint: {self.checkpoint_path}/{step_idx}.pt")
        ckpt = {
            "g": self.g_module.state_dict(),
            "d": self.d_module.state_dict(),
            "g_optim": self.g_optim.state_dict(),
            "d_optim": self.d_optim.state_dict(),
        }
        
        # if used save the g_ema
        if self.use_g_ema: ckpt["g_ema"] = self.g_ema.state_dict()

        # if used save the ref_discriminator
        if self.use_ref_disc: 
            ckpt["ref_d"] = self.ref_d_module.state_dict()
            ckpt["ref_d_optim"] = self.ref_d_optim.state_dict()

        path = os.path.join(self.checkpoint_path, f"{step_idx}.pt")
        torch.save(ckpt, path)


    def train(self):
        """

        """

        # create iterator with the dataloader
        loader = sample_data(self.dataloader)

        # create the progress bar
        if get_rank() == 0:
            pbar = tqdm(range(self.config.max_iter), initial=self.config.start_iter, dynamic_ncols=True, smoothing=0.01)
        
        # setup_training training pipeline variables
        # used to alternate between generator and discriminator training
        self.d_limit_step = self.config.disc_steps
        self.d_step = 0

        self.g_limit_step = self.config.gen_steps
        self.g_step = 0

        self.turn = True # True -> discriminator turn, False -> generator turn
        
        last_d_loss = 0
        last_ref_d_loss = 0
        last_g_loss = 0
        last_real_score = 0
        last_fake_score = 0
        last_ref_real_score = 0
        last_ref_fake_score = 0

        for i in range(self.config.start_iter, self.config.max_iter):
            
            if get_rank() == 0:
                pbar.update()

            # check if done
            if i > self.config.max_iter:
                LOGGER.info("training finished!")
                break
            
            #get the data
            sample = next(loader)

            #check_nan(sample)

            #unpack the data
            base_image = sample["base"] # [base_r, base_g, base_b] the original image without any masking
            if "ref" in sample: ref_image = sample["ref"].to(self.device) # [ref_r, ref_g, ref_b] the reference image (if config.use_ref_disc = False then ref = None)
            gen_in = sample["gen_input"].to(self.device) # [base_r, base_g, base_b, mask_0, mask_1, mask_2, mask_3]
            gen_in_orig_masks = sample["orig_gen_input_masks"].to(self.device) # [mask_0, mask_1, mask_2, mask_3] the original masks without the noise
            disc_in_true = sample["disc_input"].to(self.device) # [defect_0_r, defect_0_g, defect_0_b, defect_1_r, defect_1_g, defect_1_b, defect_2_r, defect_2_g, defect_2_b, defect_3_r, defect_3_g, defect_3_b]    
            union_mask = sample["gen_input_union_mask"] # [union_mask] the union mask of all the masks used in the generator input
            
            #######################################################################################
            #######################################################################################
            # train the discriminator #############################################################
            if self.turn:
                requires_grad(self.discriminator, True)
                requires_grad(self.generator, False)
                if self.use_ref_disc: requires_grad(self.ref_discriminator, True)
                
                #----> generate the fake images
                fake_image = self.generator(gen_in)
                
                #----> extract defects from generated
                # TODO add setting to manage if extract the defects or not before passing it to the discriminator
                disc_in_fake = self.extract_defects(fake_image, gen_in_orig_masks)
                
                #----> compute the discriminator outputs for the real and fake images
                disc_out_true, _ = self.discriminator(disc_in_true)
                disc_out_fake, _ = self.discriminator(disc_in_fake)

                self.loss_mng.metrics.update({
                    "real_score": disc_out_true.mean(),
                    "fake_score": disc_out_fake.mean(),
                })

                if self.use_ref_disc:
                    #----> compute the reference discriminator outputs for the real and fake images
                    ref_disc_out_true, _ = self.ref_discriminator(ref_image)
                    ref_disc_out_fake, _ = self.ref_discriminator(fake_image)

                    self.loss_mng.metrics.update({
                        "ref_real_score": ref_disc_out_true.mean(),
                        "ref_fake_score": ref_disc_out_fake.mean(),
                    })
                
                # compute the discriminator losses
                self.loss_mng.discriminator_loss(disc_out_fake, disc_out_true)

                # apply regularization to the discriminator
                self.loss_mng.discriminator_regularization(disc_in_true)

                if self.use_ref_disc:
                    # compute the reference discriminator losses
                    self.loss_mng.ref_discriminator_loss(ref_disc_out_fake, ref_disc_out_true)

                    # apply regularization to the reference discriminator
                    self.loss_mng.ref_discriminator_regularization(ref_image)

                # determine the next turn owner
                self.d_step += 1
                if self.d_step >= self.d_limit_step:
                    self.d_step = 0
                    self.turn = False

            #######################################################################################
            #######################################################################################
            # train the generator #################################################################
            else:
                requires_grad(self.discriminator, False)
                requires_grad(self.generator, True)
                if self.use_ref_disc: requires_grad(self.ref_discriminator, False)

                #----> generate the fake images
                fake_image = self.generator(gen_in)

                #----> extract defects from generated
                # TODO add setting to manage if extract the defects or not before passing it to the discriminator
                disc_in_fake = self.extract_defects(fake_image, gen_in_orig_masks)

                #----> compute the discriminator outputs for the fake and real images
                #disc_out_true, _ = self.discriminator(disc_in_true)
                disc_out_fake, _ = self.discriminator(disc_in_fake)
                
                self.loss_mng.metrics.update({
                    #"real_score": disc_out_true.mean(),
                    "fake_score": disc_out_fake.mean(),
                })

                ref_disc_out_fake = None
                if self.use_ref_disc:
                    #----> compute the reference discriminator outputs for the real and fake images
                    #ref_disc_out_true, _ = self.ref_discriminator(ref_image)
                    ref_disc_out_fake, _ = self.ref_discriminator(fake_image)

                    self.loss_mng.metrics.update({
                        #"ref_real_score": ref_disc_out_true.mean(),
                        "ref_fake_score": ref_disc_out_fake.mean(),
                    })

                # prepare the base image and the generated base image for the generator loss
                base_image_4loss = gen_in[:, :3, :, :] # load the image from the gen_in tensor, so if there are masked defects, the mask is already applied.
                fake_image_4loss = fake_image.clone() # TODO verify if a copy is needed
                if self.mask_base_img:
                    # if the base img is masked remove the defects from the generated img
                    # otherwise the generated defects will be present in the preceptual, l1, mse, etc losses
                    fake_image_4loss[union_mask.unsqueeze(1).repeat(1, 3, 1, 1) > 0] = 0
                    

                # compute the generator losses
                self.loss_mng.generator_loss(
                    fake_image_4loss,
                    base_image_4loss,
                    disc_out_fake,
                    ref_disc_out_fake,
                    gen_in_orig_masks
                )

                if self.device == 0 :
                    self.datalogger.log_weights_and_gradients(self.generator)

                # apply regularization to the generator, and saving the results in the gen_loss dict if present
                self.loss_mng.generator_regularization(gen_in)

                # update the moving average generator
                # self.update_average_generator()

                # determine the next turn owner
                self.g_step += 1
                if self.g_step >= self.g_limit_step:
                    self.g_step = 0
                    self.turn = True

            #if there_is_nan(metrics):
            #    raise ValueError("NaN detected in the metrics")

            reduced_metrics = reduce_loss_dict(self.loss_mng.metrics)

            #######################################################################################
            #######################################################################################
            # logging, checkpointing and visualization ############################################
            if self.device == 0:
                # update the logging bar
                last_d_loss = reduced_metrics["d_logistic_loss"] if "d_logistic_loss" in reduced_metrics else last_d_loss
                last_g_loss = reduced_metrics["g_loss"] if "g_loss" in reduced_metrics else last_g_loss
                last_real_score = reduced_metrics["real_score"] if "real_score" in reduced_metrics else last_real_score
                last_fake_score = reduced_metrics["fake_score"] if "fake_score" in reduced_metrics else last_fake_score

                if self.use_ref_disc:
                    last_ref_d_loss = reduced_metrics["ref_d_logistic_loss"] if "ref_d_logistic_loss" in reduced_metrics else last_ref_d_loss
                    last_ref_real_score = reduced_metrics["ref_real_score"] if "ref_real_score" in reduced_metrics else last_ref_real_score
                    last_ref_fake_score = reduced_metrics["ref_fake_score"] if "ref_fake_score" in reduced_metrics else last_ref_fake_score
                    pbar.set_description(
                        (f"d: {last_d_loss:.4f}; ref_d: {last_ref_d_loss}; g: {last_g_loss:.4f}; real_s: {last_real_score:.4f}; fake_s: {last_fake_score:.4f}; ref_real_s: {last_ref_real_score:.4f}; ref_fake_s: {last_ref_fake_score:.4f}")
                    )
                else:
                    pbar.set_description(
                        (f"d: {last_d_loss:.4f}; g: {last_g_loss:.4f}; real_s: {last_real_score:.4f}; fake_s: {last_fake_score:.4f}")
                    )

                # log the results locally and on wandb
                self.datalogger.log_step_results(i, reduced_metrics)

                # log weights and gradients
                #if i % self.log_weights_interval == 0:
                    

                # log the outputs 
                if i % self.log_img_interval == 0:
                    # prepare the images for the visualization
                    visual_results = {
                        "base_image": self.make_grid(base_image),
                        "fake_image": self.make_grid(fake_image),
                        "union_shapes": self.make_grid(union_mask.unsqueeze(1))
                    }

                    # add the shapes for each class
                    if self.log_shapes_and_defects:
                        for j in range(gen_in_orig_masks.shape[1]):
                            gen_in_orig_masks.to("cpu")
                            visual_results[f"shape_{j}"] = self.make_grid(gen_in_orig_masks[:, j].unsqueeze(1))
                            visual_results[f"defect_{j}"] = self.make_grid(disc_in_true[:, j*3:(j+1)*3])


                    self.datalogger.log_visual_results(i, visual_results)

                # save the checkpoint
                if i > 0 and i % self.config.checkpoint_interval == 0:
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

        # create a tensor to store the extracted defects # TODO remove the fill with 0.5 if not needed
        extracted_defects = torch.zeros((fake_image.shape[0], n*3, fake_image.shape[2], fake_image.shape[3]), device=fake_image.device)

        # for each defect, extract it from the generated image
        for i in range(n):
            target_areas = (defect_masks[:, i] > 0).unsqueeze(1).repeat(1, 3, 1, 1)
            extracted_defects[:, i*3:(i*3)+3][target_areas] = fake_image[target_areas]

        return extracted_defects
    
    
    def make_grid(self, sample):
        """
        Method that wrap the torch make_grid function to create a grid of images.
        with predefined parameters.

        Args:
            sample (torch.Tensor): a tensor containing the images to be plotted (shape: [batch_size, 3, H, W] or [batch_size, 1,  H, W])
        
        Returns:
            torch.Tensor: a tensor containing the grid of images (shape: [3, H, W] or [1, H, W])
        """
        
        return make_grid(
            sample,
            nrow= int(np.sqrt(sample.shape[0])),
            normalize=True,
            range=(0, 1)
        )


