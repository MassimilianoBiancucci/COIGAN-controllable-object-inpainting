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
            channels,
            device,
            obj_weight=0.0,
            bg_weight=1.0,
            kernel_size=51
        ):
        """
        Args:
            channels (int): number of channels of the input tensor
            obj_weight (float, optional): L1 loss weight for object areas. Defaults to 0.0.
            bg_weight (float, optional): L1 loss weight for background areas. Defaults to 1.0.
            kernel_size (int, optional): define the kernel size used for the mask 
        """
        self.channels = channels
        self.obj_weight = obj_weight
        self.bg_weight = bg_weight

        self.device = device

        # convert a smoothness value in a gaussian kernel
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.kernel = torch.ones(1, self.channels, self.kernel_size, self.kernel_size, device=self.device) / (self.kernel_size ** 2)


    def __call__(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): prediction tensor
            target (torch.Tensor): target tensor
            mask (torch.Tensor): mask tensor
        """

        # create the smoothed mask
        if self.kernel_size > 0:
            mask = F.conv2d(mask, self.kernel, padding=self.kernel_size // 2)

        # convert the mask in a weight mask
        weight_mask = self.obj_weight * mask + self.bg_weight * (1 - mask)

        # apply the masked L1 loss
        loss = F.l1_loss(pred, target, reduction='none')
        loss = loss * weight_mask

        return loss.sum() / weight_mask.sum()
    

#####################################
### TEST SmoothMaskedL1

if __name__ == "__main__":

    import numpy as np
    import cv2

    batch_size = 2

    # create a mask with a circle
    mask = np.zeros((batch_size, 256, 256), dtype=np.float32)
    for i in range(batch_size):
        x = np.random.randint(50, 256-50)
        y = np.random.randint(50, 256-50)
        cv2.circle(mask[i], (x, y), 30, 1, -1)

    # load a test image
    img = cv2.imread('/home/max/thesis/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/base_dataset/data/3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    batch_img = np.stack([img] * batch_size, axis=0)

    # create a noise mask to add to the image in the circle
    noise_mask = mask.copy()
    noise_mask = np.expand_dims(noise_mask, axis=3)
    noise_mask = np.repeat(noise_mask, 3, axis=3)
    noise_mask = np.random.rand(*noise_mask.shape) * 20 * noise_mask
    
    # add the noise to the image
    noised_img = batch_img + noise_mask

    # convert to tensor
    mask = torch.from_numpy(mask).unsqueeze(1)
    img = torch.from_numpy(batch_img).permute(0, 3, 1, 2).float()
    noised_img = torch.from_numpy(noised_img).permute(0, 3, 1, 2).float()

    # create the loss
    sml1 = SmoothMaskedL1(obj_weight=0.0, bg_weight=1.0, kernel_size=51)
    
    # compute the loss
    loss = sml1(noised_img, img, mask)

    print(loss)
