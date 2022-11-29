#!/usr/bin/env python3

import logging
import os
import sys
import traceback
import datetime as dt

import hydra
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


from COIGAN.utils.ddp_utils import handle_ddp_parent_process, handle_ddp_subprocess

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
            OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml')) # saving the configs to config.hydra.run.dir

        # create the checkpoints dir
        checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # create the model checkpoint callback and the wandb logger
        checkpointer = ModelCheckpoint(dirpath=checkpoints_dir, **config.trainer.checkpoint_kwargs)
        logger = WandbLogger(**config.wandb)

        # create the trainer
        trainer = Trainer(
            callbacks=checkpointer,
            logger=logger,
            default_root_dir=os.getcwd(),
            **config.trainer.kwargs
        )

        # train the model
        

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Training failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()