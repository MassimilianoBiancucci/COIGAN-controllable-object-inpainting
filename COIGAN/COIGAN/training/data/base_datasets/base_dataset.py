import os
import logging

from typing import List, Tuple, Dict, Union

from COIGAN.training.data.jsonl_dataset import JsonLineDataset, JsonLineDatasetMasksOnly

LOGGER = logging.getLogger(__name__)


class BaseDataset:

    """
        This class is used as base class for all the base datasets.
        With base datasets is intended the dataset used as base for the
        inpainting process.
        In other words this dataset contains the images that will be
        inpainted, the segmentation masks of the image used to choice 
        where to apply the inpainting target mask and the target masks.

        So this object has 3 principal inputs:
            - images
            - segmentation masks of the images
            - target masks with the target object shape
        
        This object accomplish the importan function of rescaling and translate the
        target mask based on the segmentation masks, so in base at the dataset the 
        object can be placed in resonable positions, for example people on the ground 
        and not in the sky.

        This class take 2 type of datasets as input: 
        1 - only one JsonLineDataset that contains the images and the segmentation masks that identifies where
            the target masks will be placed.
        2 - one or more JsonLineDatasetMasksOnly that contains the target masks, with the shape of the object to inpaint.
            more than one dataset or a dataset with multiple labels can be used.

        For this class it's needed a datastructure that define how the target masks 
        should be placed on the images, this datastructure is a json file that contains relations
        betwen the target masks labels and the base image segmentations labels.
    """

    def __init__(
        self, 
        base_dataset: JsonLineDataset,
        masks_datasets: list[JsonLineDatasetMasksOnly],
        size: int,
        mask_applicator_kwargs: dict
    ):
        """
            Args:
                indir (str): path to the directory containing the images
                size (int): size of the images
                mask_applicator_kwargs (dict): kwargs for the mask applicator
        """
        self.base_dataset = base_dataset
        self.masks_datasets = masks_datasets
        self.size = size
        self.mask_applicator_kwargs = mask_applicator_kwargs

        