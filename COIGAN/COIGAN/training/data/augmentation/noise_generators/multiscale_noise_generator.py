import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from omegaconf import DictConfig, ListConfig
from typing import Tuple

from COIGAN.training.data.augmentation.noise_generators import make_noise_generator
from COIGAN.training.data.augmentation.noise_generators.base_noise_generator import BaseNoiseGenerator

from COIGAN.utils.debug_utils import check_nan

class MultiscaleNoiseGenerator(BaseNoiseGenerator):
    """
    Method to generate multiscale noise.
    This method accept as input the type of noise generator and its parameters.
    """

    # define enum for the strategy
    strategy_dict = {
        "additive": 0,
        "multiplicative": 1,
        "replace": 2
    }

    def __init__(
        self,
        base_generator_kwargs: DictConfig,
        interpolation: str = "bilinear",
        scales: list = [1, 2, 4, 8, 16],
        strategy: str = "additive",
        normalize: bool = True,
        smooth: int = 3
    ):
        """
        Args:
            base_generator_kwargs (DictConfig): parameters of the noise generator to use
            scales (list): scales to use
            strategy (str): strategy to apply the noise:
                - additive: add the noise to the original masks (default).      es. (masks + noise)*(masks > 0)
                - multiplicative: multiply the original masks by the noise.     es. masks * noise
                - replace: replace the original masks with the noise.           es. noise*(masks > 0)
            normalize (bool): if True, normalize the noise to the range [0, 1]
            smooth (int): if > 0, smooth the noise with a gaussian filter with the given sigma
        """
        # check if the strategy is valid
        if strategy not in ["additive", "multiplicative", "replace"]:
            raise ValueError(f"Invalid strategy {strategy}. Valid strategies are: additive, multiplicative, replace.")

        # check if the interpolation is valid
        if interpolation not in ["bilinear", "bicubic", "nearest"]:
            raise ValueError(f"Invalid interpolation {interpolation}. Valid interpolations are: bilinear, bicubic, nearest.")

        if not isinstance(scales, (list, ListConfig)):
            raise ValueError(f"scales must be a list, got {type(scales)}")

        # check that the scales values are equal or higer than 1
        for scale in scales:
            if scale < 1:
                raise ValueError("The scale values must be greater or equal than 1. found scale: {}".format(scale))


        self.noise_generator = make_noise_generator(**base_generator_kwargs)
        self.scales = scales
        self.strategy = self.strategy_dict[strategy]
        self.interpolation = interpolation
        self.align_corners = self.interpolation in {"bilinear", "bicubic"}

        self.normalize = normalize
        self.smooth = smooth if smooth%2 == 1 else smooth+1

    def __call__(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Add multiscale noise to the masks, only where the original masks are > 0.

        Args:
            masks (torch.Tensor): a new tensor with noise added

        Returns:
            torch.Tensor: masks with noise, only where the original masks are > 0
        """

        mask_target = (masks > 0)
        if mask_target.any():
            noise = self.get_noise(masks.shape)

            # smooth the noise
            if self.smooth > 0:
                mask_target = gaussian_blur(masks, self.smooth)

            if self.strategy == 0: # additive
                noise_mask = (masks + noise) * mask_target

            elif self.strategy == 1: # multiplicative
                noise_mask = masks * noise

            elif self.strategy == 2: # replace
                noise_mask = noise * mask_target

            # normalize the noise to the range [0, 1]
            if self.normalize:
                noise_mask_min = min(0, noise_mask.min())
                noise_mask_max = max(1, noise_mask.max())
                noise_mask = ((noise_mask - noise_mask_min) / (noise_mask_max - noise_mask_min)) * mask_target

            return noise_mask

        else:
            return masks


    def get_noise(self, shape) -> torch.Tensor:
        """
        Get a noise tensor of the given shape

        Args:
            shape (tuple or list): shape of the noise tensor, in the form (n_classes, h, w)

        Returns:
            torch.Tensor: noise tensor
        """
        noise = torch.zeros(shape)
        n_classes, h, w = shape
        for scale in self.scales:
            cur_h, cur_w = int(h / scale), int(w / scale)
            scale_noise = self.noise_generator.get_noise((1, n_classes, cur_h, cur_w))
            scale_noise = F.interpolate(scale_noise, size=(h, w), mode=self.interpolation, align_corners=self.align_corners)
            noise += scale_noise.squeeze(0)

        return noise


##################################################
### DEBUG SECTION

def test0():
    from time import time

    noise_gen = MultiscaleNoiseGenerator(
        base_generator_kwargs = {
            "kind": "gaussian",
            "kind_kwargs": {
                "mean": 0.5,
                "std": 0.08
            }
        },
        scales = [1, 2, 4],
        strategy = "replace"
    )

    masks = torch.zeros((1, 256, 256))
    masks[:, 128:, :] = 1

    noise_gen.scales = [1, 2, 4, 8, 16]
    t0 = time()
    test_0 = noise_gen(masks)[0]
    t1 = time()
    print(f"Time to generate noise 0: {t1 - t0:.3f} seconds")

    noise_gen.scales = [1, 3, 6, 12, 24]
    t0 = time()
    test_1 = noise_gen(masks)[0]
    t1 = time()
    print(f"Time to generate noise 1: {t1 - t0:.3f} seconds")

    noise_gen.scales = [6, 9, 18]
    t0 = time()
    test_2 = noise_gen(masks)[0]
    t1 = time()
    print(f"Time to generate noise 2: {t1 - t0:.3f} seconds")

    print("DONE!")

def test1():
    from time import time
    noise_gen = MultiscaleNoiseGenerator(
        base_generator_kwargs = {
            "kind": "gaussian",
            "kind_kwargs": {
                "mean": 0.5,
                "std": 0.08
            }
        },
        scales = [1, 2, 4],
        strategy = "replace",
        smooth = 41
    )

    masks = torch.zeros((1, 256, 256))
    masks[:, 128:, :] = 1

    noise_gen.smooth = 41
    t0 = time()
    test_0 = noise_gen(masks)[0]
    t1 = time()
    print(f"Time to generate noise 0: {t1 - t0:.3f} seconds")

    noise_gen.smooth = 21
    t0 = time()
    test_1 = noise_gen(masks)[0]
    t1 = time()
    print(f"Time to generate noise 1: {t1 - t0:.3f} seconds")

    noise_gen.smooth = 11
    t0 = time()
    test_2 = noise_gen(masks)[0]
    t1 = time()
    print(f"Time to generate noise 2: {t1 - t0:.3f} seconds")

    print("DONE!")



if __name__ == "__main__":
    #test0()
    test1()
    
