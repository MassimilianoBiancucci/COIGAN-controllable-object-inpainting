import os
import json, pbjson
import logging
import numpy as np

from typing import Tuple, Union, Dict

LOGGER = logging.getLogger(__name__)


class JsonLineDatasetBaseGenerator:

    """
    Object that aim to give all the tools to generate a jsonl dataset usable with the 
    JsonLineDataset class in COIGAN/training/data/jsonl_dataset.py

    this class has methods to feed polygons or masks as numpy arrays and generate a jsonl file
    with the relative index file.
    """

    def __init__(
        self,
        out_path,
        dump_every=1000,
        binary=False
    ):
        """
        Init the dataset generator, creating the dataset file and the index file.

        Args:
            out_path (str): path of the output dataset
            dump_every (int): number of samples to dump in the dataset file before flushing
            binary (bool): if True, the dataset file will be saved in binary mode
                NOTE: if binary flag is True, the dataset file will be saved in binary mode
                improving reading and writing speed, but the file will be unreadable and
                numpy arrays will be not supported in the jsonl file!
        """

        # creating the output dataset paths
        self.out_path = out_path
        self.dataset_path = os.path.join(self.out_path, "dataset.jsonl")
        self.index_path = os.path.join(self.out_path, "index")

        self.dump_every = dump_every
        self.binary = binary

        self.cache = []

        # creating the output dataset
        os.makedirs(self.out_path, exist_ok=True)
        open(self.dataset_path, "wb" if self.binary else "w").close()
        open(self.index_path, "wb" if self.binary else "w").close()


    def convert(self):
        """
        Method that convert the input dataset in the output dataset.
        """
        raise NotImplementedError 


    def preprocess(self, sample):
        """
        Method that preprocess a sample from the input dataset
        and return a json sample.

        Args:
            sample: the sample to preprocess
        
        Returns:
            the json sample
        """
        raise NotImplementedError


    def _dump_block(self, data_block):
        """
        Method that given a dump them in the dataset file.
        storing the index of the line in the index file.
        """
        LOGGER.debug("dumping block..")
        with open(self.index_path, "a") as index_f, open(self.dataset_path, "a") as f:
            for data in data_block:
                index_f.write(f"{f.tell()}\n")    

                json.dump(data, f, cls=NpEncoder)
                f.write("\n")


    def _dump_binary_block(self, data_block):
        """
        Method that given a dump them in the dataset file.
        storing the index of the line in the index file.
        """
        LOGGER.debug("dumping block..")
        with open(self.index_path, "ab") as index_f, open(self.dataset_path, "ab") as f:
            for data in data_block:
                f.write(pbjson.dumps(data))

                endpos = f.tell()
                byte_endpos = endpos.to_bytes(4, byteorder="little")
                index_f.write(byte_endpos)

                #index_f.write(f.tell().to_bytes(4, byteorder="little")) # add the pos of the last byte of the line in the dataset file

    def _dump(self):
        """
        Method that given a dump them in the dataset file.
        storing the index of the line in the index file.
        """
        if self.binary:
            self._dump_binary_block(self.cache)
        else:
            self._dump_block(self.cache)
        
        self.cache = []


    def insert(self, samples):
        """
        Method that add a sample to the cache list,
        if the cache list is full, it dump the cache list
        in the dataset file.
        """

        if isinstance(samples, list):
            self.cache.extend(samples)
        elif isinstance(samples, dict):
            self.cache.append(samples)
        else:
            raise TypeError("samples must be a list[dict] or a dict")

        if len(self.cache) >= self.dump_every:
            self._dump()


    def close(self):
        """
        Close the dataexporter, dumping the remaining samples.
        clone the dataset images into the dataset folder.
        """
        self._dump()


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


##################################################################################################

##################################################################################################

##################################################################################################

##################################################################################################

if __name__ == "__main__":

    ###########################################
    # Test and example of usage of the class
    # comparison between the reading and writing speed using and not the binary mode
    # almost depend by the type of json passed but 

    int_poly_test_json = {
        "filename": "test.png",
        "field_1": [
            {
                "label": "test",
                "polygon": [
                    [
                        [[1000, 1000]] * 100
                    ] *10
                ]
            }
        ]
    }

    float_poly_test_json = {
        "filename": "test.png",
        "field_1": [
            {
                "label": "test",
                "polygon": [
                    [
                        [[0.7654, 0.2956]] * 100
                    ] *10
                ]
            }
        ]
    }

    import time
    import datetime
    out_path = "/home/ubuntu/hdd/COIGAN-controllable-object-inpainting/experiments_data"
    out_path = os.path.join(out_path, f"test_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"test out path: {out_path}")

    n = 10

    ##############################################################################
    ##### WRITING TESTS ON INT POLYGONS

    # creating the dataset generators
    # creating it with more cache than intserted elements so they will be loaded in the cache
    # and then dumped in the dataset file only at the close, so can be timed.
    dataset_generator = JsonLineDatasetBaseGenerator(out_path, dump_every=1001)
    binary_dataset_generator = JsonLineDatasetBaseGenerator(out_path+"_bin", dump_every=1001, binary=True)

    # inserting the same number of elements in the two dataset generators
    dataset_generator.insert([int_poly_test_json] * n)
    binary_dataset_generator.insert([int_poly_test_json] * n)

    print("\n\nstarting int dataset generation dump..")
    t = time.time()
    dataset_generator.close()
    t = time.time() - t
    print(f"int dataset generation dump took: {t} seconds")
    
    print("\nstarting float dataset generation dump..")
    t = time.time()
    binary_dataset_generator.close()
    t = time.time() - t
    print(f"int bin dataset generation dump took: {t} seconds")

    ##############################################################################
    ##### READING TESTS ON INT POLYGONS

    from COIGAN.training.data.datasets_loaders.jsonl_dataset import JsonLineDatasetBase

    dataset = JsonLineDatasetBase(out_path+"/dataset.jsonl", out_path+"/index")
    dataset.on_worker_init()
    bin_dataset = JsonLineDatasetBase(out_path+"_bin/dataset.jsonl", out_path+"_bin/index", binary=True)
    bin_dataset.on_worker_init()

    print("\n\nstarting int dataset reading..")
    t = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]
    t = time.time() - t
    print(f"int dataset reading took: {t} seconds")
    
    # checking integrity of the dataset
    for i in range(len(dataset)):
        assert dataset[i] == int_poly_test_json , "int dataset integrity check failed!"
    print("int dataset integrity check passed!")

    print("\nstarting int bin dataset reading..")
    t = time.time()
    for i in range(len(bin_dataset)):
        sample = bin_dataset[i]
    t = time.time() - t
    print(f"int bin dataset reading took: {t} seconds")

    # checking integrity of the dataset
    for i in range(len(bin_dataset)):
        assert bin_dataset[i] == int_poly_test_json , "int bin dataset integrity check failed!"
    print("int bin dataset integrity check passed!")


    ##############################################################################
    ##### WRITING TESTS ON FLOAT POLYGONS

    # doing the same with float polygons
    dataset_generator = JsonLineDatasetBaseGenerator(out_path, dump_every=1001)
    binary_dataset_generator = JsonLineDatasetBaseGenerator(out_path+"_bin", dump_every=1001, binary=True)

    # inserting the same number of elements in the two dataset generators
    dataset_generator.insert([float_poly_test_json] * n)
    binary_dataset_generator.insert([float_poly_test_json] * n)

    print("\n\nstarting float dataset generation dump..")
    t = time.time()
    dataset_generator.close()
    t = time.time() - t
    print(f"float dataset generation dump took: {t} seconds")

    print("\nstarting float bin dataset generation dump..")
    t = time.time()
    binary_dataset_generator.close()
    t = time.time() - t
    print(f"float bin dataset generation dump took: {t} seconds")

    ##############################################################################
    ##### READING TESTS ON FLOAT POLYGONS

    dataset = JsonLineDatasetBase(out_path+"/dataset.jsonl", out_path+"/index")
    dataset.on_worker_init()
    bin_dataset = JsonLineDatasetBase(out_path+"_bin/dataset.jsonl", out_path+"_bin/index", binary=True)
    bin_dataset.on_worker_init()

    print("\n\nstarting float dataset reading..")
    t = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]
    t = time.time() - t
    print(f"float dataset reading took: {t} seconds")

    # checking integrity of the dataset
    for i in range(len(dataset)):
        assert dataset[i] == float_poly_test_json , "float dataset integrity check failed!"
    print("float dataset integrity check passed!")

    print("\nstarting float bin dataset reading..")
    t = time.time()
    for i in range(len(bin_dataset)):
        sample = bin_dataset[i]
    t = time.time() - t
    print(f"float bin dataset reading took: {t} seconds")

    # checking integrity of the dataset
    for i in range(len(bin_dataset)):
        assert bin_dataset[i] == float_poly_test_json , "float bin dataset integrity check failed!"
    print("float bin dataset integrity check passed!")

    ##############################################################################

    print("\n\nALL TESTS PASSED!")
    