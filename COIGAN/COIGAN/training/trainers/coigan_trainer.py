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

try:
    import wandb
except ImportError:
    wandb = None

from COIGAN.modules import make_generator, make_discriminator
from COIGAN.training.losses import CoiganLossManager


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
        if self.device == 0 and wandb is not None:
            wandb.init(**self.config.wandb)
            wandb.config.update(
                OmegaConf.to_container(self.config, resolve=True)
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
        self.g_ema.load_state_dict(ckpt["g_ema"])

        self.g_optim.load_state_dict(ckpt["g_optim"])
        self.d_optim.load_state_dict(ckpt["d_optim"])


    def train(self):
        """

        """
        # create iterator with the dataloader
        loader = sample_data(self.dataloader)

        # create the progress bar
        if get_rank() == 0:
            pbar = tqdm(range(self.config.max_iter), initial=self.config.start_iter, dynamic_ncols=True, smoothing=0.01)

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
            base = sample["base"].to(self.device)
            
            # train the discriminator
            

            # compute the discriminator losses


            # appply regularization to the discriminator


            # train the generator


            # compute the generator losses


            # apply regularization to the generator


            # update the moving average generator


            # log the results
            

            # save the checkpoint



        
        



