import os
import cv2
import numpy as np
import json

import logging

from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import Union, Tuple, List

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

from COIGAN.training.data.datasets_loaders.jsonl_dataset import JsonLineDatasetSeparatedMasksOnly

class JsonLineDatasetInspector:

    def __init__(
        self,
        dataset_path: str,
        fields_to_inspect: List[str],
        report_output_path: str = None,
        binary: bool = False,
    ):
        """
        Init the JsonLineDatasetInspector
        Object that analyze the dataset and produce a report as a set of graphs
        and a json file with the statistics, used in the split process.

        Args:
            dataset_path (str): _description_
            fields_to_inspect (list[str]): _description_
            report_output_path (str): _description_
            binary (bool, optional): define if the input jsonl dataset is binary or not. Defaults to False.
        """
        
        self.dataset_path = dataset_path
        self.fields_to_inspect = fields_to_inspect
        self.report_output_path = report_output_path if report_output_path is not None else self.dataset_path
        self.binary = binary

        self.metadata_file_path = os.path.join(self.dataset_path, "dataset.jsonl")
        self.index_file_path = os.path.join(self.dataset_path, "index")

        self.dataset = None

        # loading the generation parameters of the dataset
        self.dataset_params = json.load(open(os.path.join(self.dataset_path, "params.json"), "r"))

        self.dataset = JsonLineDatasetSeparatedMasksOnly(
            self.metadata_file_path,
            self.index_file_path,
            self.fields_to_inspect,
            size = self.dataset_params["tile_size"],
            binary = self.binary
        )

        self.dataset.on_worker_init()

        #extract fields-class mapping
        self.fields_class_map = self.dataset.map_field_classes()

        # inspection variables
        self.n_polygons = 0
        self.n_polygons_per_image = [] # list of the number of polygons per image
        self.n_polygons_per_class = {  # dict of the number of polygons per class
            field: {
                class_name: []
                for class_name in self.fields_class_map[field]
            }
            for field in self.fields_class_map
        }

        self.area_of_polygons = [] # list of the area of polygons (without sample index)
        self.area_of_polygons_per_image = [] # list of the area of polygons per image (without info on class)
        self.area_of_polygons_per_class = { # dict of the area of polygons per class
            field: {
                class_name: []
                for class_name in self.fields_class_map[field]
            }
            for field in self.fields_class_map
        }

        # overlap area between classes
        self.overlap_area_per_class = {}
        for field in self.fields_class_map:
            
            # create a list of unique couples of classes
            # es. [0, 1, 2] -> [(0, 1), (0, 2), (1, 2)]
            field_classes = self.fields_class_map[field]
            field_classes_unique_couples = \
                [[field_classes[i], field_classes[j]] for i in range(len(field_classes)) for j in range(i+1, len(field_classes))]

            self.overlap_area_per_class[field] = [
                {
                    "classes": [class_1, class_2],
                    "overlap_areas": [],
                    "total_overlap_area": 0
                } for class_1, class_2 in field_classes_unique_couples
            ]

        self.sample_polygons_class_map = {
            field: []
            for field in self.fields_class_map
        }


      
    def inspect(self):
        """
        Inspect the dataset.
        
        """

        for smpl_idx in tqdm(range(len(self.dataset))):
            
            sample = self.dataset[smpl_idx]

            img_polygons = 0
            union_mask = np.zeros((self.dataset_params["tile_size"][0], self.dataset_params["tile_size"][1]), dtype = np.uint8)

            # compute the number of masks
            for field, val in sample.items():

                class_masks = {}
                self.sample_polygons_class_map[field] += [[0]*len(self.fields_class_map[field])]
                for class_name, masks in val.items():
                    img_polygons += len(masks)
                    self.n_polygons_per_class[field][class_name] += [len(masks)]
                    self.sample_polygons_class_map[field][-1][self.fields_class_map[field].index(class_name)] = len(masks)

                    class_masks[class_name] = class_masks.get(class_name, np.zeros_like(union_mask))
                    # iter each mask and compute the area
                    for mask in masks:
                        self.area_of_polygons += [np.sum(mask)]
                        union_mask = np.bitwise_or(union_mask, mask)
                        class_masks[class_name] = np.bitwise_or(class_masks[class_name], mask)

                    # compute the area of the mask
                    self.area_of_polygons_per_class[field][class_name] += [np.sum(class_masks[class_name])]
            
                # check if there is overlap between classes
                for classes_overlap in self.overlap_area_per_class[field]:
                    class_1, class_2 = classes_overlap["classes"]
                    if class_1 in class_masks and class_2 in class_masks:
                        classes_overlap["overlap_areas"] += [np.sum(np.bitwise_and(class_masks[class_1], class_masks[class_2]))]

            self.area_of_polygons_per_image.append(np.sum(union_mask))
            self.n_polygons_per_image.append(img_polygons)
            self.n_polygons += img_polygons

        # compute the sum of the overlap areas
        for classes_overlap in self.overlap_area_per_class[field]:
            classes_overlap["total_overlap_area"] += np.sum(classes_overlap["overlap_areas"], dtype=np.int32)
    

    def dump_report(self):
        """
        Method that dumps the report of the inspection.
        The report is made by 2 files:
            - a brief report in json format:
                - total number of plygons
                - total number of polygons for each class
                - min, max, mean, of number of polygons per image
                - min, max, mean, of area of polygons per image
                - min, max, mean, of area of polygons for each class
                - number of overlapps between classes
                - total area of all polygons
                - total area of polygons for each class
                - number of images with n polygons
                - number of images with n polygons for each class

        Args:
            generate_graphs (bool, optional): If True, it generates the graphs of the report. Defaults to False.

        """

        LOGGER.info("Generating the brief report...")

        # create report folder
        os.makedirs(os.path.join(self.report_output_path, "reports"), exist_ok=True)

        # generate the brief report
        brief_report = {
            "n_samples": len(self.dataset),
            "n_samples_with_polygons": np.sum(np.array(self.n_polygons_per_image) > 0).item(),
            "n_samples_without_polygons": np.sum(np.array(self.n_polygons_per_image) == 0).item(),
            "n_polygons": self.n_polygons,
            "n_polygons_per_class": {
                field: {
                    class_name: np.sum(self.n_polygons_per_class[field][class_name]).item()
                    for class_name in self.n_polygons_per_class[field]
                }
                for field in self.n_polygons_per_class
            },
            "n_polygons_per_image": {
                "min": np.min(self.n_polygons_per_image).item(),
                "max": np.max(self.n_polygons_per_image).item(),
                "mean": np.mean(self.n_polygons_per_image).item()
            },
            "area_of_polygons_per_image": {
                "min": np.min(self.area_of_polygons_per_image).item(),
                "max": np.max(self.area_of_polygons_per_image).item(),
                "mean": np.mean(self.area_of_polygons_per_image).item()
            },
            "area_of_polygons_per_class": {
                field: {
                    class_name: np.sum(self.area_of_polygons_per_class[field][class_name]).item()
                    for class_name in self.area_of_polygons_per_class[field]
                }
                for field in self.area_of_polygons_per_class
            },
            "overlap_area_per_class": {
                field: {
                    f"{class_1}_{class_2}": classes_overlap["total_overlap_area"]
                    for classes_overlap in self.overlap_area_per_class[field]
                    for class_1, class_2 in [classes_overlap["classes"]]
                }
                for field in self.overlap_area_per_class
            },
            "area_of_polygons": np.sum(self.area_of_polygons).item(),
            "n_polygons_per_image_per_class_histogram": {
                field: {
                    class_name: {
                        str(n_polygons): np.sum(self.n_polygons_per_class[field][class_name] == n_polygons, dtype=np.int32).item()
                        for n_polygons in np.unique(self.n_polygons_per_class[field][class_name])
                    }
                    for class_name in self.n_polygons_per_class[field]
                }
                for field in self.n_polygons_per_class
            },
            "n_polygons_per_image_histogram": {
                str(n_polygons): np.sum(self.n_polygons_per_image == n_polygons, dtype=np.int32).item()
                for n_polygons in np.unique(self.n_polygons_per_image)
            }
        }

        # dump the brief report
        with open(f"{self.report_output_path}/reports/brief_report.json", "w") as f:
            json.dump(brief_report, f, indent=4, cls=NpEncoder)
    

    def dump_raw_report(self):
        """
        Method that dumps the raw report of the inspection.
        The raw report is made by 2 files:
            - a json file with the number of polygons per image for each class
            - a json file with the area of polygons per image for each class
        """

        LOGGER.info("Generating the raw report...")

        # create report folder
        os.makedirs(os.path.join(self.report_output_path, "reports"), exist_ok=True)

        # dump the raw report
        with open(f"{self.report_output_path}/reports/raw_report.json", "w") as f:
            json.dump({
                "fields_class_map": self.fields_class_map,
                "sample_polygons_class_map": self.sample_polygons_class_map,
                "area_of_polygons_per_class": self.area_of_polygons_per_class
            }, f, cls=NpEncoder)


    def generate_graphs(self, bins: int = 200):
        """
        Method that generates the graphs of the report.

        Grphs generated:
            - histogram of number of polygons per image
            - histogram of number of polygons per image for each class
            - histogram of area of polygons per image
            - histogram of area of polygons per image for each class
            - histogram of number polygons class total distribution
            - histogram of samples with and without polygons

        Args:
            brief_report (dict): The brief report of the inspection.
        """

        LOGGER.info("Generating the graphs...")

        #check if the graphs folder exists
        os.makedirs(f"{self.report_output_path}/reports/graphs", exist_ok=True)

        # histogram of number of polygons per image
        plt.figure(figsize=(10, 10))
        plt.hist(self.n_polygons_per_image, bins=10)
        plt.title(f"Number of polygons per image")
        plt.savefig(f"{self.report_output_path}/reports/graphs/n_polygons_per_image_histogram.png")
        plt.close()

        # histogram of number of polygons per image for each class
        for field in self.n_polygons_per_class:
            for class_name in self.n_polygons_per_class[field]:
                plt.figure(figsize=(10, 10))
                plt.hist(self.n_polygons_per_class[field][class_name], bins=10)
                plt.title(f"Number of polygons per image, field: {field}, class:{class_name}")
                plt.savefig(f"{self.report_output_path}/reports/graphs/n_polygons_per_image_per_class_histogram_{field}_{class_name}.png")
                plt.close()
        
        # histogram of area of polygons per image
        plt.figure(figsize=(10, 10))
        plt.hist(self.area_of_polygons_per_image, bins=bins)
        plt.title(f"Area of polygons per image")
        plt.savefig(f"{self.report_output_path}/reports/graphs/area_of_polygons_per_image_histogram.png")
        plt.close()

        # histogram of area of polygons per image for each class
        for field in self.area_of_polygons_per_class:
            for class_name in self.area_of_polygons_per_class[field]:
                plt.figure(figsize=(10, 10))
                plt.hist(self.area_of_polygons_per_class[field][class_name], bins=bins)
                plt.title(f"Area of polygons per image, field: {field}, class:{class_name}")
                plt.savefig(f"{self.report_output_path}/reports/graphs/area_of_polygons_per_image_per_class_histogram_{field}_{class_name}.png")
                plt.close()

        # histogram of number polygons class total distribution
        for field in self.n_polygons_per_class:
            plt.figure(figsize=(10, 10))
            plt.bar(
                [f"class: {class_name}" for class_name in self.n_polygons_per_class[field]],
                [np.sum(self.n_polygons_per_class[field][class_name]) for class_name in self.n_polygons_per_class[field]]
            )
            plt.title(f"Number of polygons per class total distribution")
            plt.savefig(f"{self.report_output_path}/reports/graphs/n_polygons_per_class_total_histogram.png")
            plt.close()
        
        # histogram of samples with and without polygons
        np_n_polygons_per_image = np.array(self.n_polygons_per_image)
        plt.figure(figsize=(10, 10))
        plt.bar(
            ["with_polygons", "without_polygons"],
            [np.sum(np_n_polygons_per_image > 0), np.sum(np_n_polygons_per_image == 0)]
        )
        plt.title(f"Samples with and without polygons")
        plt.savefig(f"{self.report_output_path}/reports/graphs/samples_with_and_without_polygons_histogram.png")
        plt.close()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":

    #target_dataset = "/home/ubuntu/hdd/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/jsonl_all_samples"
    target_dataset = "/home/ubuntu/hdd/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/train_set"
    #target_dataset = "/home/ubuntu/hdd/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/test_set"

    inspector = JsonLineDatasetInspector(
        dataset_path = target_dataset,
        fields_to_inspect = ["polygons"],
        binary = True
    )

    inspector.inspect()
    inspector.dump_report()
    inspector.dump_raw_report()
    inspector.generate_graphs()
