import os
import json
import numpy as np
import cv2

import logging

from tqdm import tqdm
from typing import Union, Tuple
from omegaconf.listconfig import ListConfig

from COIGAN.training.data.dataset_generators.jsonl_dataset_generator import JsonLineDatasetBaseGenerator
from COIGAN.training.data.datasets_loaders.jsonl_dataset import JsonLineDatasetBase

LOGGER = logging.getLogger(__name__)

class ObjectDatasetGenerator(JsonLineDatasetBaseGenerator):

    """
        Class that generate the jsonl dataset for the mask dataset.
        this type of dataset do not have a complex structure.
        It's a simple collection of json in which each json file 
        contains the information about one mask, and the the image 
        box that contains the mask.
        This dataset can be used to train a mask generator or 
        an object inpainter, due to the presence of the object image and 
        the mask of the object.
    """

    def __init__(
        self,
        input_dataset: JsonLineDatasetBase,
        image_dir: str,
        output_dir: str,
        target_field: str,
        target_class: str,
        tile_size: Union[int, Tuple[int, int]],
        rst_origin: bool = True,
        normalized: bool = False,
        dump_every=1000,
        binary=False,
    ):
        """
            Init method for the MaskDatasetGenerator class.

            Args:
                input_dataset (JsonLineDatasetBase): The dataset from which the masks will be extracted.
                image_dir (str): The directory where the input images are stored.
                output_dir (str): The output directory where the folder of the jsonl dataset will be created. the generator add another folder to the output_dir with the name of the target_class.
                target_field (str): The field of the input dataset that contains the polygons.
                target_class (str): The class of the polygons that will be extracted.
                tile_size (int): The size of the tile that will be extracted from the original image.
                rst_origin (bool, optional): If True the polygons will be translated to the origin, so the smallest x and y value are 0. Defaults to True.
                normalized (bool, optional): This flag is used to tell the generator if the polygons are normalized or not in the origin dataset. Defaults to False.
                dump_every (int, optional): The number of samples that will be dumped in the jsonl dataset before saving it. Defaults to 1000.
                binary (bool, optional): If True the masks will be saved as binary files. Defaults to False.
        """

        self.output_dir = os.path.join(output_dir, f"object_dataset_{target_class}")
        self.image_dir = image_dir

        super(ObjectDatasetGenerator, self).__init__(
            self.output_dir,
            dump_every,
            binary
        )

        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        elif isinstance(tile_size, ListConfig):
            tile_size = [tile_size[0], tile_size[1]]

        self.input_dataset = input_dataset

        self.target_field = target_field
        self.target_class = target_class
        self.tile_size = tile_size
        self.rst_origin = rst_origin
        self.normalized = normalized

        self.n_samples = 0

        # create the output data directory
        os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)

    
    def convert(self):
        """
            Conversion method, start the conversion from the original dataset
            to the jsonl dataset of masks.
        """

        for sample in tqdm(self.input_dataset):

            # load the image if the sample has at least one polygon
            if sample[self.target_field]:
                image = cv2.imread(os.path.join(self.image_dir, sample["img"]))

            for poly in sample[self.target_field]:
                if poly["label"] == self.target_class:
                    
                    points, shape, bbox = self.process_polygon(poly["points"])
                    
                    img_name = f"{self.n_samples}.jpg"
                    cv2.imwrite(
                        os.path.join(self.output_dir, "data", img_name),
                        image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    )
                    
                    new_sample = {
                        "img": img_name,
                        "points": points,
                        "shape": shape
                    }

                    self.insert(new_sample)
                    self.n_samples += 1

        self.close()
        self.generate_params_brief()


    def process_polygon(self, points):
        """
            Get the polygon from the points,
            if the rst_origin value is True, the polygon will be
            translated to the origin, so the smallest x and y value are 0.

            Args:
                points (list[list[list[int]]]): List of points of the polygon
                    each polygons has the following structure:
                    [
                        [
                            [x1, y1],
                            [x2, y2],
                            ...
                        ],
                        [
                            [x1, y1],
                            [x2, y2],
                            ...
                        ],
                        ...
                    ]
                NOTE: potentially there are more than one closed polygon in one mask
        
            Returns:
                list[list[list[int]]]: polygon reformatted, with normalization removed and moved to the origin
                tuple[int, int]: shape of the polygon (height, width)
                tuple[int, int, int, int]: bounding box of the polygon (x1, y1, x2, y2)

        """
        sub_poly_max_x = 0
        sub_poly_max_y = 0
        sub_poly_min_x = 10000
        sub_poly_min_y = 10000

        sub_polygons = []

        for sub_poly in points:
            sub_poly = np.array(sub_poly)

            if self.normalized:
                #rescale the points to the tile size
                sub_poly[:, 0] *= self.tile_size[1]
                sub_poly[:, 1] *= self.tile_size[0]
                sub_poly = sub_poly.astype(np.int32)
            
            sub_polygons.append(sub_poly)

            #get the min x and y
            sub_poly_min_x = min(np.min(sub_poly[:, 0]), sub_poly_min_x)
            sub_poly_min_y = min(np.min(sub_poly[:, 1]), sub_poly_min_y)

            # get the max x and y
            sub_poly_max_x = max(np.max(sub_poly[:, 0]), sub_poly_max_x)
            sub_poly_max_y = max(np.max(sub_poly[:, 1]), sub_poly_max_y)

        lst_sub_polygons = []
        for sub_poly in sub_polygons:
            
            if self.rst_origin:
                # translate the polygon to the origin
                sub_poly[:, 0] -= sub_poly_min_x
                sub_poly[:, 1] -= sub_poly_min_y
            
            lst_sub_polygons.append(sub_poly.tolist())

        # compute the bounding box shapes
        h = sub_poly_max_y - sub_poly_min_y
        w = sub_poly_max_x - sub_poly_min_x

        shape = [h.item(), w.item()]
        bbox = [
            sub_poly_min_x.item(), 
            sub_poly_min_y.item(), 
            sub_poly_max_x.item(), 
            sub_poly_max_y.item()
        ]

        return lst_sub_polygons, shape, bbox


    def generate_params_brief(self):
        """
            Generate a brief description of the parameters of the generator.
        """

        params = {
            "n_samples": self.n_samples,
            "target_field": self.target_field,
            "target_class": self.target_class,
            "tile_size": self.tile_size,
            "rst_origin": self.rst_origin,
            "normalized": self.normalized,
            "binary": self.binary
        }

        with open(os.path.join(self.output_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

