#!/usr/bin/env python3

import logging
import os
import sys
import traceback

import hydra
from omegaconf import OmegaConf

from MaskGAN.utils.ddp_utils import handle_ddp_parent_process, handle_ddp_subprocess

LOGGER = logging.getLogger(__name__)

@handle_ddp_subprocess()
@hydra.main(config_path='..MaskGAN/configs/training', config_name='test_train.yaml')
def main(config: OmegaConf):

    try:
        LOGGER.info(f'Config: {OmegaConf.to_yaml(config)}')

        # check if that one is the 
        is_ddp_subprocess = handle_ddp_parent_process()

        if not is_ddp_subprocess:
            LOGGER.info(f"training configs: \n{config.pretty()}")

        checkpoints_dir = os.path.join(config.training.checkpoints_dir, config.training.name)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Training failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()