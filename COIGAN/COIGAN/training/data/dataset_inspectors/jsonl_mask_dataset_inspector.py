import os
import json
import cv2
import numpy as np
import logging

from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import Union, Tuple, List

from COIGAN.training.data.datasets_loaders.jsonl_dataset import JsonLineDatasetBase

LOGGER = logging.getLogger(__name__)

class JsonLineMaskDatasetInspector:

    """
        Class aimed to inspect the jsonl mask dataset.
        generate a simple report and a set of graphs about the masks distributions.
    """

    def __init__(
        self,
        dataset_path: str,
        report_output_path: str = None,
        binary: bool = False,
    ):

        """
        Init the JsonLineMaskDatasetInspector
        Object that analyze the dataset and produce a report as a set of graphs
        and a json file with the statistics, used in the split process.

        Args:
            dataset_path (str): _description_
            fields_to_inspect (list[str]): _description_
            report_output_path (str): _description_
            binary (bool, optional): define if the input jsonl dataset is binary or not. Defaults to False.
        """
        
        self.dataset_path = dataset_path
        self.report_output_path = report_output_path if report_output_path is not None else self.dataset_path
        self.binary = binary

        self.metadata_file_path = os.path.join(self.dataset_path, "dataset.jsonl")
        self.index_file_path = os.path.join(self.dataset_path, "index")

        self.dataset = None

        # loading the generation parameters of the dataset
        self.dataset_params = json.load(open(os.path.join(self.dataset_path, "params.json"), "r"))

        self.dataset = JsonLineDatasetBase(
            self.metadata_file_path,
            self.index_file_path,
            binary = self.binary
        )

        self.dataset.on_worker_init()

        # inspection variables
        self.n_polygons = len(self.dataset)
        self.area_of_polygons = [] # list of the area of polygons
        self.shape_of_polygons = [] # list of the shape of polygons

    
    def inspect(self):
        """
        Inspect the dataset and generate the report.
        """
        LOGGER.info("Inspecting the dataset...")
        for sample in tqdm(self.dataset):
            shape = sample["shape"]
            mask = self.poly2mask(sample["points"], shape)
            self.area_of_polygons.append(np.sum(mask))
            self.shape_of_polygons.append(shape)
    

    def dump_report(self):
        """
        Dump the report in the report_output_path.
        """
        LOGGER.info("Dumping the report...")
        os.makedirs(self.report_output_path, exist_ok=True)
        
        np_area_of_polygons = np.array(self.area_of_polygons)
        np_shape_of_polygons = np.array(self.shape_of_polygons)

        brief_report = {
            "n_polygons": self.n_polygons,
            "shapes": {
                "min": [np.min(np_shape_of_polygons[:, 0]), np.min(np_shape_of_polygons[:, 1])],
                "max": [np.max(np_shape_of_polygons[:, 0]), np.max(np_shape_of_polygons[:, 1])],
                "mean": [np.mean(np_shape_of_polygons[:, 0]), np.mean(np_shape_of_polygons[:, 1])],
                "std": [np.std(np_shape_of_polygons[:, 0]), np.std(np_shape_of_polygons[:, 1])],
            },
            "area":{
                "min": np.min(np_area_of_polygons),
                "max": np.max(np_area_of_polygons),
                "mean": np.mean(np_area_of_polygons),
                "std": np.std(np_area_of_polygons),
            }
        }

        with open(os.path.join(self.report_output_path, "brief_report.json"), "w") as f:
            json.dump(brief_report, f, cls=NpEncoder, indent=4)
    

    def dump_raw_report(self):
        """
        Dump the raw report in the report_output_path.
        """
        LOGGER.info("Dumping the raw report...")
        os.makedirs(self.report_output_path, exist_ok=True)
        
        raw_report = {
            "n_polygons": self.n_polygons,
            "area_of_polygons": self.area_of_polygons,
            "shape_of_polygons": self.shape_of_polygons
        }

        with open(os.path.join(self.report_output_path, "raw_report.json"), "w") as f:
            json.dump(raw_report, f, cls=NpEncoder)

    
    def generate_graphs(self):
        """
        Generate the graphs of the report.
        """
        LOGGER.info("Generating the graphs...")
        
        np_area_of_polygons = np.array(self.area_of_polygons)
        np_shape_of_polygons = np.array(self.shape_of_polygons)

        # shape graphs #############

        # scatter plot of the shape of the polygons
        plt.figure()
        plt.scatter(np_shape_of_polygons[:, 0], np_shape_of_polygons[:, 1])
        plt.xlabel("height")
        plt.ylabel("width")
        plt.savefig(os.path.join(self.report_output_path, "shape_scatter.png"))

        # area graphs #############

        #hist of the area of the polygons
        plt.figure()
        plt.hist(np_area_of_polygons, bins=100)
        plt.xlabel("area")
        plt.ylabel("count")
        plt.savefig(os.path.join(self.report_output_path, "area_hist.png"))


    @staticmethod
    def poly2mask(
        polygon: List[List[List[int]]],
        shape: Tuple[int],
        normalized_points: bool = False,
        mask_value: int = 1,
    ) -> "dict[str, np.ndarray]":
        """
        Load all the masks in a given field, groupping all the masks with the same label
        in a single mask.

        Args:
            polygon (list[list[list[int]]]): the polygon to convert in a mask
            shape (tuple): the shape of the output mask
            normalized_points (bool, optional): if the points are normalized. Defaults to False.
            mask_value (int, optional): the value to use in the mask. Defaults to 1.

        Returns:
            dict[str, np.ndarray]: a dict mapping the class to the mask
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
        


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

