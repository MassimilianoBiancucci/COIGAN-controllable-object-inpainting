import torch
import torch.nn.functional as F

class SmoothMaskedL1:
    """
    Method that applies a mask to the L1 loss,
    this method accept masks with values {0, 1},
    and generate a smoothed version of the mask with values
    between [obj_weight, bg_weight] that change smoothly.
    This method is used to apply a smoothed l1 loss
    near the object boundaries.
    """

    def __init__(
            self,
            obj_weight=0.0,
            bg_weight=1.0,
            smoothness=0.2
        ):
        """

        Args:
            obj_weight (float, optional): L1 loss weight for object areas. Defaults to 0.0.
            bg_weight (float, optional): L1 loss weight for background areas. Defaults to 1.0.
            smoothness (float, optional): define the smooth variation between the obj_weight and the bg_weight. Defaults to 0.2.
                es: smoothness = 0.0 mean a binary mask with obj_weight where the mask is 1 and bg_weight where the mask is 0
                    if smoothness is greater than 0 is used to compute the gaussian kernel that is used to smooth the mask
        """
        self.obj_weight = obj_weight
        self.bg_weight = bg_weight

        # convert a 



