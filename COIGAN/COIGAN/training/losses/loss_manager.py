import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from typing import Dict, List, Tuple, Union


from COIGAN.training.losses .perceptual import PerceptualLoss, ResNetPL

LOGGER = logging.getLogger(__name__)


class CoiganLossManager:

    def __init__(self, config: OmegaConf):
        """
        Init method of the CoiganLossManager class.

        Args:
            config (OmegaConf): configuration
        """

        self.config = config

        self.loss_perceptual = None
        if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
            self.loss_l1 = nn.L1Loss(reduction='none')

        self.loss_mse = None
        if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
            self.loss_mse = nn.MSELoss(reduction='none')

        self.loss_resnet_pl = None
        if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
            self.loss_resnet_pl = ResNetPL(**self.config.losses.resnet_pl)
    
    def __call__(self, ):
        pass