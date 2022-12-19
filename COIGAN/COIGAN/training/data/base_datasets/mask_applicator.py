import os
import logging

from omegaconf import OmegaConf

from typing import List, Tuple, Dict, Union

LOGGER = logging.getLogger(__name__)


class MaskApplicator:
    """
        This class is used to apply the target masks on the images.
        The target masks are rescaled and translated based on the segmentation masks.
        The target masks are applied on the images accordingly with the config.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        application_config: OmegaConf
    ):
        """
            Args:
                size (Union[int, Tuple[int, int]]): size of the images
                application_config (OmegaConf): config for the mask applicator

        """
        self.size = size
        self.application_config = application_config


    def apply_mask(self, base_masks, target_masks):
        """
            This method modify the target masks based on the base masks.
            To make it more clear, the target masks are rescaled and translated
            based on the config to match the base masks, as requested in the config.
            
            Args:
                base_masks (torch.Tensor): the base masks
                target_masks (torch.Tensor): the target masks
            
            Returns:
                torch.Tensor: the modified target masks
        """
        pass


