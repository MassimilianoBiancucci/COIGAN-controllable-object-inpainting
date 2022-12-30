import os
import json
import numpy as np
import logging

from tqdm import tqdm
from typing import Union, Tuple

from COIGAN.training.data.datasets_loaders.jsonl_dataset import JsonLineDatasetBase
from COIGAN.training.data.dataset_generators.jsonl_dataset_generator import JsonLineDatasetBaseGenerator

LOGGER = logging.getLogger(__name__)

class BaseSplitter:

    """
        Base class for dataset splitters.
        Implement basic functionality for dataset splitters.
        this apply a random split of the dataset based only on
        the number of samples and the train and test ratio.
    """

    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        train_ratio = None,
        val_ratio = None,
        test_ratio = None,
        seed: int = 42,
        binary: bool = True,
        max_chunks: int = 1000,
        tile_size: Union[int, Tuple[int]] = 256,
        **kwargs
    ):
        """
            Init method for the BaseSplitter class.

        Args:
            dataset_path (str): path to the source dataset
            output_dir (str): path to the output directory, in this directory will be created a folder for each split [train, val, test]
            train_ratio (_type_, optional): train ratio, define the percentage of the dataset that will be added to the train set. Defaults to None.
            val_ratio (_type_, optional): val ratio, define the percentage of the dataset that will be added to the val set. Defaults to None.
            test_ratio (_type_, optional): test ratio, define the percentage of the dataset that will be added to the test set. Defaults to None.
            seed (int, optional): random seed. Defaults to 42.
            binary (bool, optional): define if the source dataset is in binary mode, if true keep the splits in the same format. Defaults to True.
            max_chunks (int, optional): define how many samples should be readed in one step, usefull if the dataset is big. Defaults to 10000.
        """

        self.dataset_path = dataset_path
        self.dataset_images_path = os.path.join(dataset_path, "data")
        self.output_dir = output_dir

        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        self.tile_size = tile_size

        # check if train + val + test = 1 otherwise normalize it
        train_ratio = 0.0 if train_ratio is None else train_ratio
        val_ratio = 0.0 if val_ratio is None else val_ratio
        test_ratio = 0.0 if test_ratio is None else test_ratio

        if train_ratio + val_ratio + test_ratio != 1.0:
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total

        # store the ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # store other parameters
        self.seed = seed
        self.binary = binary
        self.max_chunks = max_chunks

        # load the input dataset
        self.dataset = JsonLineDatasetBase(
            os.path.join(self.dataset_path, "dataset.jsonl"),
            os.path.join(self.dataset_path, "index"),
            self.binary
        )
        self.dataset.on_worker_init()
        self.dataset_size = len(self.dataset)

        self.train_set_len = int(self.dataset_size * self.train_ratio)
        self.val_set_len = int(self.dataset_size * self.val_ratio)
        self.test_set_len = int(self.dataset_size * self.test_ratio)

        # solve eventual rounding errors
        self.train_set_len += self.dataset_size - (self.train_set_len + self.val_set_len + self.test_set_len)

        # logging info
        LOGGER.info("Dataset size: %d", self.dataset_size)
        if self.train_ratio > 0.0:
            LOGGER.info("Train set size: %d", self.train_set_len) 
        if self.val_ratio > 0.0:
            LOGGER.info("Val set size: %d", self.val_set_len)
        if self.test_ratio > 0.0:
            LOGGER.info("Test set size: %d", self.test_set_len)

        # create the output datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_dataset_path = os.path.join(self.output_dir, "train_set")
        self.train_images_path = os.path.join(self.train_dataset_path, "data")

        self.val_dataset_path = os.path.join(self.output_dir, "val_set")
        self.val_images_path = os.path.join(self.val_dataset_path, "data")

        self.test_dataset_path = os.path.join(self.output_dir, "test_set")
        self.test_images_path = os.path.join(self.test_dataset_path, "data")

        if self.train_ratio > 0.0:
            self.train_dataset = JsonLineDatasetBaseGenerator(
                self.train_dataset_path,
                dump_every = 10000,
                binary = self.binary
            )
            os.makedirs(self.train_images_path, exist_ok=True) # create the data folder
        
        if self.val_ratio > 0.0:
            self.val_dataset = JsonLineDatasetBaseGenerator(
                self.val_dataset_path,
                dump_every = 10000,
                binary = self.binary
            )
            os.makedirs(self.val_images_path, exist_ok=True) # create the data folder

        if self.test_ratio > 0.0:
            self.test_dataset = JsonLineDatasetBaseGenerator(
                self.test_dataset_path,
                dump_every = 10000,
                binary = self.binary
            )
            os.makedirs(self.test_images_path, exist_ok=True) # create the data folder


    def split(self):
        """
            Split the dataset in train, val and test set.
        """
        
        idxs = list(range(self.dataset_size))
        np.random.seed(self.seed)
        np.random.shuffle(idxs)

        # split the dataset
        if self.train_dataset is not None:
            train_idxs = idxs[:self.train_set_len]
            train_idxs = np.array_split(train_idxs, len(train_idxs)//self.max_chunks)

        if self.val_dataset is not None:
            val_idxs = idxs[self.train_set_len:self.train_set_len + self.val_set_len]
            val_idxs = np.array_split(val_idxs, len(val_idxs)//self.max_chunks)

        if self.test_dataset is not None:
            test_idxs = idxs[self.train_set_len + self.val_set_len:]
            test_idxs = np.array_split(test_idxs, len(test_idxs)//self.max_chunks)

        # add the samples to the datasets
        if self.train_dataset is not None:
            LOGGER.info("splitting the train set...")
            for train_idxs_chunk in tqdm(train_idxs):
                train_samples = self.dataset[train_idxs_chunk]
                self.train_dataset.insert(train_samples)
                
                self.copy_images(
                    train_samples, 
                    self.dataset_images_path, 
                    self.train_images_path
                )
            self.train_dataset.close()
        
        if self.val_dataset is not None:
            LOGGER.info("splitting the val set...")
            for val_idxs_chunk in tqdm(val_idxs):
                val_samples = self.dataset[val_idxs_chunk]
                self.val_dataset.insert(val_samples)
                
                self.copy_images(
                    val_samples,
                    self.dataset_images_path,
                    self.val_images_path
                )
            self.val_dataset.close()
        
        if self.test_dataset is not None:
            LOGGER.info("splitting the test set...")
            for test_idxs_chunk in tqdm(test_idxs):
                test_samples = self.dataset[test_idxs_chunk]
                self.test_dataset.insert(test_samples)
                
                self.copy_images(
                    test_samples,
                    self.dataset_images_path,
                    self.test_images_path
                )
            self.test_dataset.close()
        
        self.generate_split_briefs()

        LOGGER.info("Done!")
    

    def copy_images(
        self, 
        samples, 
        source_dir: str, 
        target_dir: str, 
        img_field: str = "img"
    ):
        """
            Copy the images from the input dataset to the output dataset.
        """

        for sample in samples:
            img_name = sample[img_field]
            img_source_path = os.path.join(source_dir, img_name)
            img_target_path = os.path.join(target_dir, img_name)
            os.system("cp {} {}".format(img_source_path, img_target_path))
    

    def generate_split_briefs(self):
        """
            Generate a brief of the dataset parameters.
        """
        if self.train_dataset is not None:
            self.genrate_params_brief("train", self.train_set_len, self.train_dataset_path)
        
        if self.val_dataset is not None:
            self.genrate_params_brief("val", self.val_set_len, self.val_dataset_path)
        
        if self.test_dataset is not None:
            self.genrate_params_brief("test", self.test_set_len, self.test_dataset_path)


    def genrate_params_brief(self, split: str, split_size: int, output_dir: str):
        """
            Generate a brief of the dataset parameters.

            Args:
                split (str): the split name. (train, val, test)W
                split_size (int): the split size. (number of samples)
                output_dir (str): the output directory.
        """
        # generate a file that specify the dataset generation parameters
        params = {
            "original_dataset": self.dataset_path,
            "dataset_size": self.dataset_size,
            "split": split,
            "split_size": split_size,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "tile_size": [
                self.tile_size[0],
                self.tile_size[1]
            ],
            "seed": self.seed,
        }

        with open(os.path.join(output_dir, "params.json"), "w") as f:
            json.dump(params, f)



        

        
