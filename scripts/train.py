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

        wandb_logger = WandbLogger(
            project=config.wandb.project,
            name=config.wandb.name,
            config=config,
            save_dir=os.getcwd(),
            offline=config.wandb.offline,
            id=config.wandb.id,
            anonymous=config.wandb.anonymous,
            log_model=config.wandb.log_model,
            version=config.wandb.version,
            entity=config.wandb.entity,
            group=config.wandb.group,
            tags=config.wandb.tags,
            notes=config.wandb.notes,
            save_code=config.wandb.save_code,
            job_type=config.wandb.job_type,
            dir=os.getcwd(),
        )

        trainer_kwargs = OmegaConf.to_container(config.trainer.kwargs, resolve=True)

        trainer = Trainer(
            callbacks=ModelCheckpoint(dirpath=checkpoints_dir, **config.trainer.checkpoint_kwargs),
            logger=metrics_logger,
            default_root_dir=os.getcwd(),
            **trainer_kwargs
        )



    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Training failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()