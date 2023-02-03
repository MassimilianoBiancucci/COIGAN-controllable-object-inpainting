import torch

from COIGAN.training.data.augmentation.noise_generators.base_noise_generator import BaseNoiseGenerator

class GaussianNoiseGenerator(BaseNoiseGenerator):
    """
    Applicator of gaussian noise to masks.
    """

    strategy_dict = {
        "additive": 0,
        "multiplicative": 1,
        "replace": 2
    }

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        strategy: str = "additive"
    ):
        """
        Args:
            mean (float): mean of the gaussian noise
            std (float): standard deviation of the gaussian noise
            strategy (str): strategy to apply the noise:
                - additive: add the noise to the original masks (default).      es. (masks + noise)[masks == 0] = 0
                - multiplicative: multiply the original masks by the noise.     es. masks * noise
                - replace: replace the original masks with the noise.           es. noise[masks == 0] = 0
        """
        # check if the strategy is valid
        if strategy not in ["additive", "multiplicative", "replace"]:
            raise ValueError(f"Invalid strategy {strategy}. Valid strategies are: additive, multiplicative, replace.")

        self.mean = mean
        self.std = std
        self.strategy = self.strategy_dict[strategy]
    

    def __call__(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Add gaussian noise to the masks, only where the original masks are > 0.

        Args:
            masks (torch.Tensor): a new tensor with noise added

        Returns:
            torch.Tensor: masks with noise, only where the original masks are > 0
        """
        noise = torch.normal(mean=self.mean, std=self.std, size=masks.shape, device=masks.device)

        if self.strategy == 0: # additive
            return (masks + noise) * (masks > 0)
        
        elif self.strategy == 1: # multiplicative
            return masks * noise
        
        elif self.strategy == 2: # replace
            #noise[masks == 0] = 0
            return noise * (masks > 0)
    

    def get_noise(self, shape: tuple, device = "cpu") -> torch.Tensor:
        """
        Get the noise tensor.

        Args:
            shape (tuple): shape of the noise tensor

        Returns:
            torch.Tensor: noise tensor
        """
        return torch.normal(mean=self.mean, std=self.std, size=shape, device=device)


##########################################
### DEBUG SECTION

if __name__ == "__main__":

    from time import time

    masks = torch.zeros((1, 4, 256, 256)).to("cuda")

    # create a cube of 1s
    masks[:, :, 100:150, 100:150] = 1

    orig = masks[0, 0].cpu().numpy()

    # add noise and time it
    noise_gen = GaussianNoiseGenerator(strategy="additive")
    t0 = time()
    noise_masks = noise_gen(masks)
    tadd = time() - t0

    test_0 = noise_masks[0, 0].cpu().numpy()

    # multiply noise and time it
    noise_gen = GaussianNoiseGenerator(strategy="multiplicative")
    t0 = time()
    noise_masks = noise_gen(masks)
    tmul = time() - t0

    test_1 = noise_masks[0, 0].cpu().numpy()

    # replace noise and time it
    noise_gen = GaussianNoiseGenerator(strategy="replace")
    t0 = time()
    noise_masks = noise_gen(masks)
    trep = time() - t0

    test_2 = noise_masks[0, 0].cpu().numpy()

    print(f"Additive noise: {tadd:.6f}s")
    print(f"Multiplicative noise: {tmul:.6f}s")
    print(f"Replace noise: {trep:.6f}s")


