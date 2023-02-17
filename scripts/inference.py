#!/usr/bin/env python3

import os
import logging

os.environ["HYDRA_FULL_ERROR"] = "1"

import torch

import hydra
from omegaconf import OmegaConf

from COIGAN.training.data.datasets_loaders import make_dataloader
from COIGAN.inference import COIGANinference

from COIGAN.utils.ddp_utils import ddp_setup

LOGGER = logging.getLogger(__name__)

@hydra.main(config_path="../configs/inference/", config_name="test_inference.yaml", version_base="1.1")
def main(config: OmegaConf):
    
    #resolve the config inplace
    OmegaConf.resolve(config)

    LOGGER.info(f'Config: {OmegaConf.to_yaml(config)}')

    OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml')) # saving the configs to config.hydra.run.dir

    


if __name__ == "__main__":
    main()