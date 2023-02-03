import torch
class BaseNoiseGenerator:
    """
    Base class for noise generators.
    This kind of objects will take an input tensor of masks
    and return a new tensor of masks with noise applied.
    """

    def __init__(
        self
    ):
        """
        """
        pass


    def __call__(self, masks: torch.Tensor) -> torch.Tensor:
        """
        """
        pass