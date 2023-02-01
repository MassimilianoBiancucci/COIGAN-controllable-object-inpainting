import os
import logging
import cv2
import numpy as np
import torch

from torchvision.io import read_image
from torchvision.transforms.transforms import Normalize

from typing import Tuple, Union, Dict, List

from COIGAN.training.data.datasets_loaders.jsonl_dataset import JsonLineDatasetBase
from COIGAN.training.data.augmentation.augmentor import Augmentor

logger = logging.getLogger(__name__)


class JsonLineObjectDataset(JsonLineDatasetBase):

    """
        Dataset loader for object datasets in jsonl format.
        Those datasets contains only the onteresting objects extracted from the original images.
        Each sample in the dataset is a jsonl line with the following fields:
            - "img": name of the image file inside the data folder of the dataset
            - "points": list of points that define the object mask
            - "shape": shape of the mask (height, width) that matches the object image
        
        NOTE: in each sample is considered only the main object of the image, if other objects are present
        they are ignored, in each sample only one object is present.

    """

    def __init__(
        self,
        dataset_path: str,
        binary: bool = True,
        train: bool = False,
        augmentor: Augmentor = None,
        remove_bg: bool = False,
    ):
        """
            Init method of the object dataset loader.

            Args:
                dataset_path: path to the dataset folder
                binary: specify if the input dataset is in binary format.
                train: specify if the dataset is used for training or not. if True the dataset will return the data in torch.Tensor format.
                augmentor: specify the augmentor to use for the dataset. if None no augmentation is applied.
                remove_bg: specify if the background of the image should be removed. if True the background is removed using the mask.
        """

        self.train = train
        self.augmentor = augmentor
        self.remove_bg = remove_bg

        self.dataset_path = dataset_path
        metadata_path = os.path.join(dataset_path, "dataset.jsonl")
        index_path = os.path.join(dataset_path, "index")

        super(JsonLineObjectDataset, self).__init__(
            metadata_file_path=metadata_path,
            index_file_path=index_path,
            binary=binary,
        )


    def _get_sample(self, idx: int) -> \
        Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return the sample with the given index

        Args:
            idx: index of the sample to return
        
        Returns:
            img (np.ndarray or torch.Tensor): image of the object
            mask (np.ndarray or torch.Tensor): mask of the object
        """

        metadata = self._get_metadata(idx)
        
        # load the image path
        img_path = os.path.join(self.dataset_path, "data", metadata["img"])

        # load the mask
        mask = self.poly2mask(metadata["points"], metadata["shape"]) # NOTE: load the mask with values in {0, 1}

        # preprocess the image and the mask if used for training
        if self.train:
            img = read_image(img_path).unsqueeze(0)
            mask = torch.as_tensor(mask).unsqueeze(0)

            if self.augmentor:
                img, mask = self.augmentor(img=img, mask=mask)

            # apply normalization to img and remove mean and variance of ImageNet
            img = img.float() / 255.0
            mask = mask.float()
            
            if self.remove_bg:
                img = img * mask

        # otherwise just load the image in cv2 format
        else:
            img = cv2.imread(img_path)

            if self.remove_bg:
                img = cv2.bitwise_and(img, img, mask=mask)

        return img, mask


    def __getitem__(self, idx: Union[slice, int, List[int]]):
        """
        Return the sample with the given index
        """
        if isinstance(idx, slice):
            return [self._get_sample(i) for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, (list, np.ndarray)):
            return [self._get_sample(i) for i in idx]
        else:
            return self._get_sample(idx)
    

    def __iter__(self):
        """
        Return an iterator over the dataset
        """
        for idx in range(len(self)):
            yield self._get_sample(idx)
    

    @staticmethod
    def poly2mask(
        polygon: List[List[List[int]]],
        shape: tuple,
        normalized_points: bool = False,
        mask_value: int = 1,
    ) -> np.ndarray:

        """
            Convert a list of polygons to a mask.

            Args:
                points: list of polygons
                shape: shape of the mask
                normalized_points: specify if the points are normalized
                mask_value: value of the mask

            Returns:
                mask: mask of the object

        """ 

        mask = np.zeros(shape, dtype=np.uint8)

        for sub_poly in polygon:
            points = np.asarray(sub_poly)
            points = points.squeeze()

            # TODO workaround for bugs in some datasets
            # polygons with less than 3 points can bring to other bugs
            if points.shape[0] < 3:
                continue

            if normalized_points:
                points[:, 0] = points[:, 0] * shape[1]
                points[:, 1] = points[:, 1] * shape[0]
                cv2.fillPoly(mask, [points.astype(np.int32)], mask_value)
            else:
                cv2.fillPoly(mask, [points], mask_value)

        return mask

    

class JsonLineMaskObjectDataset(JsonLineObjectDataset):

    """
        Extension of the JsonLineObjectDataset for datasets that load only
        the mask of each object.
    """

    def _get_sample(self, idx: int):
        """
        Return the sample with the given index
        """
        metadata = self._get_metadata(idx)
        
        #load mask
        mask = self.poly2mask(metadata["points"], metadata["shape"])

        if self.train:
            mask = torch.as_tensor(mask).unsqueeze(0)

            if self.augmentor:
               _, mask = self.augmentor(mask=mask)
            
            mask = mask.float()

        return mask