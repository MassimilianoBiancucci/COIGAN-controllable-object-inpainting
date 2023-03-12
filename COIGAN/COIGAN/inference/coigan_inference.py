import os
import logging

import numpy as np
import torch

from omegaconf import DictConfig

from COIGAN.modules import make_generator, make_discriminator
from COIGAN.training.data.augmentation.noise_generators import make_noise_generator

LOGGER = logging.getLogger(__name__)

class COIGANinference:

    def __init__(
        self,
        config: DictConfig
    ):
        
        # save the config
        self.config = config

        # set device
        self.device = config.device \
              if config.device is not None else \
                 torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # other process variables
        self.mask_base_img = self.config.mask_base_img
        self.use_g_ema = self.config.use_g_ema

        # load the generator
        self.generator = make_generator(**config.generator).to(self.device)
        self.load_checkpoint(self.config.checkpoint_path)

        # load the mask noise generator
        self.mask_noise_generator = make_noise_generator(**self.config.data.mask_noise_generator_kwargs) \
             if self.config.data.mask_noise_generator_kwargs is not None else None
        

    def load_checkpoint(
            self, 
            checkpoint_path
        ):
        """
            Method that load the checkpoint.

            Args:
                checkpoint_path (str): path to the checkpoint
        """
        LOGGER.info(f"Loading checkpoint from {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        if self.use_g_ema:
            self.generator.load_state_dict(ckpt["g_ema"])
        else:
            self.generator.load_state_dict(ckpt["g"])


    def __call__(
            self, 
            img,
            masks
        ):
        """
            Method that perform the inference on the input image.

            Args:
                img (numpy.): the input image
                mask (torch.Tensor): the mask tensor
        """

        # if needed mask the img to remove the background where 
        # will be generated the new content
        if self.mask_base_img:
            # collapse all the channels of the mask
            bool_mask = masks.sum(dim=2, keepdim=True).astype(np.bool)
            img[bool_mask] = 0

        # convert the input image to a tensor
        img = self.img2tensor(img)
        masks = self.masks2Tensor(masks)

        input = torch.cat([img, masks], dim=1)

        # generate the output
        with torch.no_grad():
            output = self.generator(input)
        
        # convert the output to a numpy array
        output = self.tensor2img(output)

        return output


    def img2tensor(
            self,
            img
        ):
        """
            Method that convert the input image to a tensor.

            Args:
                img (np.ndarray): the input image with shape (H, W, C)  * and format rgb not bgr *
                    with values in range [0, 255]

            Returns:
                torch.Tensor: the tensor with format (1, C, H, W) and values in range [0, 1]
        """
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().to(self.device) / 255.0
        return img.unsqueeze(0)


    @staticmethod
    def tensor2img(
            tensor
        ):
        """
            Method that convert the input tensor to a numpy array.

            Args:
                tensor (torch.Tensor): the input tensor with format (1, C, H, W) and values in range [0, 1]

            Returns:
                np.ndarray: the numpy array with shape (H, W, C) and values in range [0, 255]
        """

        return (tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


    def masks2Tensor(
            self,
            masks
        ):
        """
            Method that convert the input mask to a tensor.

            Args:
                mask (np.ndarray): the input mask with shape (H, W, C)
                    with values in range [0, 1]

            Returns:
                torch.Tensor: the tensor with format (1, C, H, W) and values in range [0, 1]
        """
        # convert the mask to a tensor
        masks = torch.from_numpy(masks.transpose(2, 0, 1)).float()

        # apply the mask noise generator
        if self.mask_noise_generator is not None:
            masks = self.mask_noise_generator(masks)
        
        # unsqueeze the batch dim and return
        return masks.to(self.device).unsqueeze(0)