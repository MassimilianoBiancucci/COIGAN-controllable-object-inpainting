#!/usr/bin/env python3

import logging
import os
import sys
import traceback

os.environ["HYDRA_FULL_ERROR"] = "1"

import torch
import torch.multiprocessing as mp

import hydra
from omegaconf import OmegaConf

from COIGAN.shape_training.trainers.stylegan2_trainer import stylegan2_trainer
from COIGAN.utils.stylegan2_ddp_utils import ddp_setup

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="../configs/shape_training", config_name="test_shape_train.yaml")
def main(config: OmegaConf):

    #resolve the config inplace
    OmegaConf.resolve(config)

    LOGGER.info(f'Config: {OmegaConf.to_yaml(config)}')

    OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml')) # saving the configs to config.hydra.run.dir

    # create the checkpoints dirls
    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs(config.sampl_dir, exist_ok=True)

    if config.distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size, config), nprocs=world_size)

    else:
        train(0, 1, config)


def train(rank: int, world_size: int, config):
    
    torch.cuda.set_device(rank)

    if config.distributed:
        ddp_setup(rank, world_size)

    trainer = stylegan2_trainer(rank, config)
    trainer.train()


if __name__ == '__main__':
    main()
