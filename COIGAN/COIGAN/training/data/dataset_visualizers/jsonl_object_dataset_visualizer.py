import os
import numpy as np
import cv2

from typing import Tuple, Union, Dict, List
from torchvision.transforms.transforms import Normalize

from COIGAN.training.data.datasets_loaders.jsonl_object_dataset import JsonLineObjectDataset, JsonLineMaskObjectdataset
from COIGAN.training.data.augmentation.augmentor import Augmentor
from COIGAN.training.data.augmentation.augmentation_presets import mask_inpainting_preset, imgs_inpainting_preset


class JsonLineObjectDatasetVisualizer(JsonLineObjectDataset):

    """
        Dataset visualizer.
        Used to inspect the Object dataset.
    """

    def __init__(
        self,
        dataset_path: str,
        binary: bool = True
    ):

        super(JsonLineObjectDatasetVisualizer, self).__init__(
            dataset_path=dataset_path,
            binary=binary
        )


    def visualize(self):
        """
            Inspect the dataset.
        """
        alpha = 0.8 # 
        color = np.array([0, 0, 255])

        # create the windows
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("appllied_mask", cv2.WINDOW_NORMAL)

        for (img, mask) in self:

            cv2.imshow("img", img)
            cv2.imshow("mask", mask*255)

            # apply the mask to the image with a random color
            color_mask = np.zeros_like(img)
            for i in range(3):
                color_mask[:, :, i] = mask * color[i]
            applied_mask = cv2.addWeighted(img, alpha, color_mask, 1-alpha, 0)

            cv2.imshow("appllied_mask", applied_mask)

            k = cv2.waitKey(0)
            if k == ord('q'):
                break
        
        cv2.destroyAllWindows()


class JsonLineObjectTrainDatasetVisualizer(JsonLineObjectDataset):

    """
        Dataset visualizer.
        Used to inspect the Object dataset.
    """

    def __init__(
        self,
        dataset_path: str,
        binary: bool = True,
        augmentor: Augmentor = None,
        remove_bg: bool = False
    ):

        super(JsonLineObjectTrainDatasetVisualizer, self).__init__(
            dataset_path = dataset_path,
            binary = binary,
            train = True,
            augmentor = augmentor,
            remove_bg = remove_bg
        )


    def visualize(self):
        """
            Inspect the dataset.
        """
        alpha = 0.8 # 
        color = np.array([0, 0, 255])

        # create the windows
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("appllied_mask", cv2.WINDOW_NORMAL)

        for (img, mask) in self:

            # add mean and std to the image of the ImageNet dataset
            #img = Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225])(img)

            # decode the image from torch tensor to numpy array
            img = img[0].numpy().transpose(1, 2, 0) * 255
            img = img.astype(np.uint8)
            
            # decode the mask from torch tensor to numpy array
            mask = mask[0].numpy()
            mask = mask.astype(np.uint8)

            cv2.imshow("img", img)
            cv2.imshow("mask", mask*255)

            # apply the mask to the image with a random color
            color_mask = np.zeros_like(img)
            for i in range(3):
                color_mask[:, :, i] = mask * color[i]
            applied_mask = cv2.addWeighted(img, alpha, color_mask, 1-alpha, 0)

            cv2.imshow("appllied_mask", applied_mask)

            k = cv2.waitKey(0)
            if k == ord('q'):
                break
        
        cv2.destroyAllWindows()


class JsonLineObjectMaskOnlyTrainDatasetVisualizer(JsonLineMaskObjectdataset):

    """
        Dataset visualizer.
        Used to inspect the Object dataset.
    """

    def __init__(
        self,
        dataset_path: str,
        binary: bool = True,
        augmentor: Augmentor = None
    ):

        super(JsonLineObjectMaskOnlyTrainDatasetVisualizer, self).__init__(
            dataset_path = dataset_path,
            binary = binary,
            train = True,
            augmentor = augmentor
        )


    def visualize(self):
        """
            Inspect the dataset.
        """

        # create the windows
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

        for mask in self:
            
            # decode the mask from torch tensor to numpy array
            mask = mask[0].numpy()
            mask = mask.astype(np.uint8)

            cv2.imshow("mask", mask*255)

            k = cv2.waitKey(0)
            if k == ord('q'):
                break
        
        cv2.destroyAllWindows()


if __name__ == "__main__":

    augmentor = Augmentor(
        transforms=mask_inpainting_preset,
        only_imgs_transforms=imgs_inpainting_preset
    )
    
    # create the dataset visualizer
    dataset_path = "/home/max/thesis/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/object_datasets/object_dataset_2"
    
    #dataset_visualizer = JsonLineObjectDatasetVisualizer(dataset_path=dataset_path)
    dataset_visualizer = JsonLineObjectTrainDatasetVisualizer(dataset_path=dataset_path, augmentor=augmentor, remove_bg=False)
    #dataset_visualizer = JsonLineObjectMaskOnlyTrainDatasetVisualizer(dataset_path=dataset_path, augmentor=augmentor)
    
    dataset_visualizer.on_worker_init()

    # visualize the dataset
    dataset_visualizer.visualize()
