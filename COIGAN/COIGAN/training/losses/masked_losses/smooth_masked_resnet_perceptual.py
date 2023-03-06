import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from COIGAN.modules.ade20k import ModelBuilder

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

class ResNetPLSmoothMasked(nn.Module):

    def __init__(
        self,
        channels,
        device,
        weights_path=None,
        arch_encoder='resnet50dilated',
        segmentation=True,
        obj_weight=0.0,
        bg_weight=1.0,
        kernel_size=51
    ):
        """
        Initialize the ResNet perceptual loss.

        Args:
            channels (int): number of channels of the input tensor
            device: device where the kernel and the model are located
            weights_path: path to the weights of the model
            arch_encoder: encoder architecture
            segmentation: whether to use segmentation
            obj_weight: weight of the object
            bg_weight: weight of the background
            kernel_size: size of the kernel for the smoothing
        """
        super().__init__()
        self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
                                             arch_encoder=arch_encoder,
                                             arch_decoder='ppm_deepsup',
                                             fc_dim=2048,
                                             segmentation=segmentation)
        self.impl.to(device)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.channels = channels
        self.obj_weight = obj_weight
        self.bg_weight = bg_weight

        self.device = device

        self.kernel_size = kernel_size
        self.kernel = torch.ones(1, self.channels, self.kernel_size, self.kernel_size, device=self.device) / (self.kernel_size ** 2)
        self.interpolation_mode = 'bilinear' if self.kernel_size > 1 else 'nearest'
        self.allign_corners = True if self.kernel_size > 1 else None

    def forward(self, pred, target, input_mask):
        """
        Compute the ResNet perceptual loss for the input and target.

        Args:
            pred: predicted tensor
            target: target tensor
            input_mask: input mask tensor
        
        Returns:
            ResNet perceptual loss
        """
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        # Compute the mask
        if self.kernel_size > 0:
            mask = F.conv2d(input_mask, self.kernel, padding=self.kernel_size // 2)
            #mask = mask / mask.max()

        # convert the mask in a weight mask
        weight_mask = self.obj_weight * mask + self.bg_weight * (1 - mask)

        # create a weight mask for each feature layer,
        # reducing the mask to match the feature map size
        resized_weight_masks = [
            F.interpolate(weight_mask, size=feat.shape[2:], mode=self.interpolation_mode, align_corners=self.allign_corners)
            for feat in pred_feats
        ]

        # Compute the loss
        layer_losses = []
        for cur_pred, cur_target, w_mask in zip(pred_feats, target_feats, resized_weight_masks):
            layer_loss = F.mse_loss(cur_pred, cur_target, reduction='none')
            #summed_layer_loss = layer_loss.sum(dim=1, keepdim=True)
            #masked_layer_loss = summed_layer_loss * w_mask
            #summed_masked_layer_loss = masked_layer_loss.sum()
            masked_layer_loss = layer_loss.sum(dim=1, keepdim=True) * w_mask
            layer_losses.append(masked_layer_loss.sum() / w_mask.sum())

        return torch.stack(layer_losses).mean()

#############################################
### TEST ResNetPLSmoothMasked

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
    np.random.seed(0)
    noise_mask = np.random.rand(*noise_mask.shape) * 40 * noise_mask
    
    # add the noise to the image
    noised_img = batch_img + noise_mask

    #z_n_mask = noise_mask * (1 - mask).reshape(batch_size, 256, 256, 1)
    #sum_z = z_n_mask.sum() # 0.0

    # convert to tensor
    mask = torch.from_numpy(mask).unsqueeze(1).float()
    img = torch.from_numpy(batch_img).permute(0, 3, 1, 2).float() / 255.0
    noised_img = torch.from_numpy(noised_img).permute(0, 3, 1, 2).float() / 255.0

    for i in range(10):
        
        k = i*10 + 1

        # create the loss
        resnetPL = ResNetPLSmoothMasked(
            weights_path="/home/max/thesis/COIGAN-controllable-object-inpainting/models_loss", 
            arch_encoder='resnet50dilated', 
            segmentation=True,
            obj_weight=0.0,
            bg_weight=1.0,
            kernel_size=k
        )
        
        # compute the loss
        loss = resnetPL(noised_img, img, mask)

        print(f"k = {k}  loss = {loss}")


    