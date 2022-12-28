import os
import hydra
import logging

from omegaconf import OmegaConf

from COIGAN.training.data.datasets_loaders.severstal_steel_defect import SeverstalSteelDefectDataset
from COIGAN.training.data.dataset_generators.severstal_steel_defect_dataset_preprcessor import SeverstalSteelDefectPreprcessor
from COIGAN.training.data.dataset_inspectors.jsonl_dataset_inspector import JsonLineDatasetInspector

LOGGER = logging.getLogger(__name__)


def check_severstal_jsonl_dataset_health(config: OmegaConf):
    """
        Method that check if the jsonl_all dataset
        is correctly created, and if need to be recreated.
    """
    def delete_jsonl_all_dataset():
        LOGGER.info("Severstal jsonl dataset it's damaged, deleting it...")
        os.system(f"rm -r {config.jsonl_dataset_dir}")

    if os.path.exists(config.jsonl_dataset_dir):
        # if the folder already exist do the check
        # if the check is ok return True else
        # delete the folder and return False

        #check if the data folder is present
        if not os.path.exists(
            os.path.join(config.jsonl_dataset_dir, "data")
        ):
            delete_jsonl_all_dataset()
            return False

        # check if the index file is present and not empty
        if not os.path.exists(os.path.join(config.jsonl_dataset_dir, "index")) or \
            not os.path.exists(os.path.join(config.jsonl_dataset_dir, "dataset.jsonl")):
            if os.stat(os.path.join(config.jsonl_dataset_dir, "index")).st_size == 0 or \
                os.stat(os.path.join(config.jsonl_dataset_dir, "dataset.jsonl")).st_size == 0:

                delete_jsonl_all_dataset()
                return False
        
        # check if the image folder is empty
        if len(os.listdir(os.path.join(config.jsonl_dataset_dir, "data"))) == 0:
            delete_jsonl_all_dataset()
            return False

        return True
    else:
        # else return False, because the dataset is not created
        return False


def download_and_extract_severstal_dataset(config: OmegaConf):
    """
        Download and extract the Severstal Steel Defect Dataset from kaggle
        only if needed!

    Args:
        dataset_path (str): Path to the dataset
    """

    # check if the extracted dataset is already present if not extract the dataset
    if not os.path.exists(config.raw_dataset_dir):

        # check if the dataset is already downloaded, if not download it
        if not os.path.exists(config.zip_file):
            LOGGER.info(f"Downloading the Steel Defect Dataset in {config.dataset_dir}")
            os.system(f"kaggle competitions download -c severstal-steel-defect-detection -p {config.dataset_dir}")
        else:
            LOGGER.info("The dataset is already downloaded")

        # unzip the dataset
        LOGGER.info(f"Unzipping the dataset in {config.raw_dataset_dir}")
        os.system(f"unzip -q {config.zip_file} -d {config.raw_dataset_dir}")
        # delete the zip file
        os.system(f"rm {config.zip_file}")
    else:
        LOGGER.info("The dataset from kaggle is already downloaded and extracted")


def convert_severstal_dataset_to_jsonl(config: OmegaConf):
    """
        Convert the Severstal Steel Defect Dataset into COIGAN compatible dataset.

    Args:
        dataset_path (str): Path to the dataset
        output_dir (str): Path to the output directory
    """

    # check if the dataset is already converted and if is converted correctly
    if not check_severstal_jsonl_dataset_health(config):
        
        # load the dataset and prepare it for the training pipeline
        LOGGER.info("Converting the kaggle dataset in json format...")
        SeverstalSteelDefectPreprcessor(
            SeverstalSteelDefectDataset(config.raw_dataset_dir, mode="all"),
            **config.dataset_preprocessor_kwargs
        ).convert()

    else:
        LOGGER.info("The dataset is already converted in jsonl format!")


def create_severstal_dataset_report(config: OmegaConf):
    """
        Create a report of the dataset
    """

    LOGGER.info("Creating the report of the dataset...")

    inspector = JsonLineDatasetInspector(
        **config.dataset_inspector_kwargs
    )

    inspector.inspect()
    inspector.dump_report()
    inspector.dump_raw_report()
    inspector.generate_graphs()


def split_dataset(config: OmegaConf):
    """
        Split the dataset in train and test set, keeping the same distribution of the classes
    """
    pass




@hydra.main(config_path='../configs/data_preparation', config_name='severstal_dataset_preparation.yaml')
def main(config: OmegaConf):
    """
        Script that load the Severstal steel defect dataset and prepare it for the trianing pipeline.

        Operations:
            1 - Download the dataset from kaggle
            2 - Extract the dataset and delete the zip file
            3 - Convert the dataset in jsonl format
                3.1 - create a report of the dataset
            4 - Split the dataset in train, and test (keeping the same distribution of the classes)
            5 - Create the mask dataset to train stylegan to generate new masks, from the train dataset
            6 - Create the none defected img dataset to train another stylegan, to generate new none defected images
            7 - Create the defects datasets to train the COIGAN model to inpaint the defects

    """

    LOGGER.info(f"Current folder: {os.getcwd()}")

    # 1 & 2 - download and extract the dataset if necessary
    download_and_extract_severstal_dataset(config)
    
    # 3 - convert the dataset in jsonl format
    convert_severstal_dataset_to_jsonl(config)

    # create report of the dataset
    create_severstal_dataset_report(config)

    # split the dataset in train and test set



if __name__ == "__main__":
    main()