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
        weight=1,
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
            weight: weight of the loss
            weights_path: path to the weights of the model
            arch_encoder: encoder architecture
            segmentation: whether to use segmentation
        """
        super().__init__()
        self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
                                             arch_encoder=arch_encoder,
                                             arch_decoder='ppm_deepsup',
                                             fc_dim=2048,
                                             segmentation=segmentation)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

        self.obj_weight = obj_weight
        self.bg_weight = bg_weight

        self.kernel_size = kernel_size
        self.kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)


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
            self.kernel.to(mask.device)
            mask = F.conv2d(mask, self.kernel, padding=self.kernel_size // 2)
            mask = mask / mask.max()

        # convert the mask in a weight mask
        weight_mask = self.obj_weight * mask + self.bg_weight * (1 - mask)

        # create a weight mask for each feature layer,
        # reducing the mask to match the feature map size
        resized_weight_masks = [
            F.interpolate(weight_mask, size=feat.shape[2:], mode='bilinear', align_corners=False)
            for feat in pred_feats
        ]

        # Compute the loss
        layer_losses = []
        for cur_pred, cur_target, w_mask in zip(pred_feats, target_feats, resized_weight_masks):
            layer_losses.append(F.mse_loss(cur_pred, cur_target, reduction='none') * w_mask)

        return torch.stack(layer_losses).sum() * self.weight

#############################################
### TEST ResNetPLSmoothMasked

if __name__ == "__main__":
    
    import cv2
    import numpy as np

    