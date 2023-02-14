#!/usr/bin/env python3

import logging
import os

os.environ["HYDRA_FULL_ERROR"] = "1"

import torch
import torch.multiprocessing as mp

import hydra
from omegaconf import OmegaConf

from COIGAN.training.data.datasets_loaders import make_dataloader
from COIGAN.training.trainers.coigan_trainer import COIGANtrainer

from COIGAN.utils.ddp_utils import ddp_setup

LOGGER = logging.getLogger(__name__)

@hydra.main(config_path="../configs/training/", config_name="test_train.yaml")
def main(config: OmegaConf):
    
    #resolve the config inplace
    OmegaConf.resolve(config)

    LOGGER.info(f'Config: {OmegaConf.to_yaml(config)}')

    OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml')) # saving the configs to config.hydra.run.dir

    if config.distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size, config), nprocs=world_size)

    else:
        train(0, 1, config)
    

def train(rank: int, world_size: int, config):

    torch.cuda.set_device(rank)

    if config.distributed:
        ddp_setup(rank, world_size)

    # generate the dataset and wrap it in a dataloader
    dataloader = make_dataloader(config, rank=rank)

    trainer = COIGANtrainer(rank, config, dataloader)
    trainer.train()


if __name__ == "__main__":
    main()