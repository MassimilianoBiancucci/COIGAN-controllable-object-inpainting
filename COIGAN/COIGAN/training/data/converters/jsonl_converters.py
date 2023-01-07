import os
import json, pbjson
import logging

from tqdm import tqdm
from typing import Tuple, Union, Dict, List

from COIGAN.training.data.datasets_loaders.jsonl_dataset import JsonLineDatasetBase
from COIGAN.training.data.dataset_generators.jsonl_dataset_generator import JsonLineDatasetBaseGenerator

logger = logging.getLogger(__name__)

class Bin2JsonLineDatasetConverter:

    """
        Object that convert a Jsonl dataset from binary to jsonl format.
    """

    def __init__(
        self, 
        jsonl_dataset: JsonLineDatasetBase,
        output_path: str
    ):
        """
            Initialize the converter.

            Args:
                jsonl_dataset: dataset to convert
                output_path: if not None, the converter will convert the whole dataset and copy the result in the given path, 
                        otherwise it will convert only the dataset.jsonl file and add to the dataset a new file 
                        called decoded_dataset.jsonl with the decoded dataset.
        """

        self.jsonl_dataset = jsonl_dataset
        self.output_path = output_path

        self.jsonl_dataset_gen = JsonLineDatasetBaseGenerator(
            out_path=self.output_path,
            dump_every=1000,
            binary=False
        )

    
    def convert(self):
        """
            Convert the dataset.
        """

        for sample in tqdm(self.jsonl_dataset):
            self.jsonl_dataset_gen.insert(sample)

if __name__ == "__main__":

    #
    orig_dataset_path = "/home/max/thesis/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/object_datasets/object_dataset_0"
    output_path = "/home/max/thesis/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/object_datasets/decoded_object_dataset_0"

    # Load the dataset
    dataset = JsonLineDatasetBase(
        metadata_file_path=os.path.join(orig_dataset_path, "dataset.jsonl"),
        index_file_path=os.path.join(orig_dataset_path, "index"),
        binary=True
    )
    dataset.on_worker_init()

    # Convert the dataset
    converter = Bin2JsonLineDatasetConverter(
        jsonl_dataset=dataset,
        output_path=output_path
    ).convert()