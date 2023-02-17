import os
import json
import numpy as np
import cv2

import logging

from tqdm import tqdm
from typing import Union, Tuple, List
from omegaconf.listconfig import ListConfig

from COIGAN.training.data.datasets_loaders.jsonl_dataset import JsonLineDataset
from COIGAN.training.data.image_evaluators.image_evaluator_base import ImageEvaluatorBase

LOGGER = logging.getLogger(__name__)

class BaseDatasetGenerator:

    """
        This object is used to generate only a collection of images 
        extracted from one source dataset.
        In the result collection of images there will be only immages
        that haven't any annotation in the source dataset, so can be used to
        train a generator for create new base images where will be applied
        the objects trhough the inpainting process.
    """

    def __init__(
        self, 
        input_dataset: JsonLineDataset,
        image_dir: str,
        output_dir: str,
        tile_size: Union[int, Tuple[int, int]],
        img_evaluator: ImageEvaluatorBase,
        fields_to_avoid: List[str],
        classes_to_avoid: List[str]
    ):

        self.input_dataset = input_dataset

        self.image_dir = image_dir
        self.output_dir = output_dir
        self.output_data_dir = os.path.join(self.output_dir, "data")
        
        # create the output directory
        os.makedirs(self.output_data_dir, exist_ok=True)

        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        elif isinstance(tile_size, (list, Tuple, ListConfig)):
            tile_size = tuple(tile_size)
        self.tile_size = tile_size
        
        self.img_evaluator = img_evaluator
        
        self.fields_to_avoid = fields_to_avoid
        self.classes_to_avoid = classes_to_avoid

        # process variables
        self.n_samples = 0

    
    def convert(self):
        """
            Method that start the conversion process from the input dataset
            to a dataset of images that respect the requirements for 
            a base dataset.

            divide all the images in tiles and check if there are any
            annotation in the tile, if there are no annotations, and the tile
            is valid (not black), the tile is saved in the output dataset.
        """

        for sample in tqdm(self.input_dataset):
            
            image = sample[0]
            group_masks = sample[1]
            bad_tile_idxs = []
            # check if there are masks in the sample
            for field in self.fields_to_avoid:
                for label, mask in group_masks[field].items():
                    if label in self.classes_to_avoid:
                        mask_tiles = self._generate_tiles(mask, self.tile_size)
                        for tile_idx, tile in enumerate(mask_tiles):
                            if np.sum(tile) > 0:
                                bad_tile_idxs.append(tile_idx)
            # remove the duplicates
            bad_tile_idxs = list(set(bad_tile_idxs))

            tiles = self._generate_tiles(image, self.tile_size)

            good_tiles = []
            for tile_idx, tile in enumerate(tiles):
                if tile_idx not in bad_tile_idxs:
                    if self.img_evaluator(tile):
                        good_tiles.append(tile)

            for tile in good_tiles:
                img_name = f"{self.n_samples}.jpg"
                cv2.imwrite(os.path.join(self.output_data_dir, img_name), tile)
                self.n_samples += 1
            

    @staticmethod
    def _generate_tiles(image: np.ndarray, tile_size: List[int]) -> List[np.ndarray]:

        """
            Split the image and the masks in tiles.
            the number of tiles is determined by the tile_size as
            w_tiles = (w // tile_size[1]) +1
            h_tiles = (h // tile_size[0]) +1

            the tiles normaly have a litle overlap, it depends on the tile_size and the image size.

            Args:
                image (np.ndarray): input image
                tile_size (tuple, optional): tile size. Defaults to (256, 256).
            
            Returns:
                list[np.ndarray]: return a list of image's tiles
        """

        h, w = image.shape[:2]

        # if the tile size is equal to the image size, return the image and the masks
        # just jump the process
        if h == tile_size[0] and w == tile_size[1]:
            return [image]

        images = []

        nh_tiles = np.ceil(tile_size[0])
        h_offset = np.floor((h-tile_size[0])/(nh_tiles-1)).astype(np.int32) \
            if h - tile_size[0] > 0 else tile_size[0]

        nw_tiles = np.ceil(w/tile_size[1])
        w_offset = np.floor((w-tile_size[1])/(nw_tiles-1)).astype(np.int32) \
            if w - tile_size[1] > 0 else tile_size[1]

        for i in range(0, h, h_offset):
            for j in range(0, w, w_offset):
                if i+tile_size[0] <= h and j+tile_size[1] <= w:
                    images.append(image[i:i+tile_size[0], j:j+tile_size[1]])

        return images


                
    

            