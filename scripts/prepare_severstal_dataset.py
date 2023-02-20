import os
import hydra
import logging

from omegaconf import OmegaConf, DictConfig

# import the dataset loaders
from COIGAN.training.data.datasets_loaders import JsonLineDatasetBase, JsonLineDataset
from COIGAN.training.data.datasets_loaders import SeverstalSteelDefectDataset

# import the dataset generators
from COIGAN.training.data.dataset_generators import SeverstalSteelDefectPreprcessor
from COIGAN.training.data.dataset_generators import ObjectDatasetGenerator
from COIGAN.training.data.dataset_generators import BaseDatasetGenerator
from COIGAN.training.data.dataset_generators import TileDatasetPreprocessor


# import the dataset inspectors
from COIGAN.training.data.dataset_inspectors import JsonLineDatasetInspector
from COIGAN.training.data.dataset_inspectors import JsonLineMaskDatasetInspector

# import the dataset splitters
from COIGAN.training.data.dataset_splitters import BaseSplitter
from COIGAN.training.data.dataset_splitters import FairSplitter

# import the image evaluators
from COIGAN.training.data.image_evaluators import SeverstalBaseEvaluator


# setting up the logger and enable warnings
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def check_severstal_jsonl_dataset_health(config: DictConfig):
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
        
        #check if the force flag is set
        if config.force_dataset_preprocessor:
            LOGGER.warning("Force dataset preprocessor flag is set, recreating the dataset...")
            delete_jsonl_all_dataset()
            return False

        return True
    else:
        # else return False, because the dataset is not created
        return False


def download_and_extract_severstal_dataset(config: DictConfig):
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
            # check if the download is completed correctly
            if not os.path.exists(config.zip_file):
                LOGGER.error(f"Dataset can't be downloaded from kaggle!\n \
                    double check the the kaggle api is installed and the kaggle.json file is present in the home directory, \n \
                    if the problem persist downlod the dataset from kaggle and put the zip file in: \n \
                    {config.dataset_dir}")
                raise FileNotFoundError(f"Unable to download the dataset from kaggle in {config.dataset_dir}")
        else:
            LOGGER.info("The dataset is already downloaded")

        # unzip the dataset
        LOGGER.info(f"Unzipping the dataset in {config.raw_dataset_dir}")
        os.system(f"unzip -q {config.zip_file} -d {config.raw_dataset_dir}")
        # delete the zip file
        os.system(f"rm {config.zip_file}")
    else:
        LOGGER.info("The dataset from kaggle is already downloaded and extracted")


def convert_severstal_dataset_to_jsonl(config: DictConfig):
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


def create_severstal_dataset_report(config: DictConfig):
    """
        Create a report of the dataset
    """
    
    # checking if the report is already generated
    report_dir = os.path.join(config.jsonl_dataset_dir, "reports")
    if os.path.exists(report_dir):
        # check if the force flag is set
        if config.force_dataset_inspector:
            LOGGER.warning("Force dataset inspector flag is set, recreating the dataset report...")

        #check if the raw_report is present and not empty
        elif os.path.exists(os.path.join(report_dir, "raw_report.json")) and \
            os.stat(os.path.join(report_dir, "raw_report.json")).st_size != 0:
                # the report for the next step is preset
                LOGGER.info("The dataset report is already generated!")
                return

        # if the report is present but incomplete or the force flag is set
        # delete it, and create a new one
        os.system(f"rm -r {report_dir}")
    
    

    LOGGER.info("Creating the report of the dataset...")

    inspector = JsonLineDatasetInspector(
        **config.dataset_inspector_kwargs
    )

    inspector.inspect()
    inspector.dump_report()
    inspector.dump_raw_report()
    inspector.generate_graphs()

        
def split_dataset(config: DictConfig):
    """
        Split the dataset in train and test set, keeping the same distribution of the classes
    """

    # checking if the splits are already created or the force flag is set

    train_split_check = os.path.exists(os.path.join(config.train_set_dir))
    test_split_check = os.path.exists(os.path.join(config.test_set_dir))

    if train_split_check and test_split_check:
        # if both are present
        # check if the force flag is set
        if config.force_dataset_splitter:
            LOGGER.warning("Force dataset splitter flag is set, recreating the dataset splits...")
            os.system(f"rm -r {config.dataset_splitter_kwargs.output_dir}")
        else:
            LOGGER.info(f"The train and test splits are already created under the folder {config.dataset_name}! skipping the splitting step...")
            return

    elif train_split_check or test_split_check:
        # if only one is present
        # delete both and recreate them
        LOGGER.info("only one of the splits are present, recreating the splits...")
        os.system(f"rm -r {config.dataset_splitter_kwargs.output_dir}")
        

    LOGGER.info("Splitting the dataset in train and test set...")

    if config.split_mode == "random":
        LOGGER.info("Splitting the dataset in random mode..")
        BaseSplitter(
            **config.dataset_splitter_kwargs
        ).split()

    elif config.split_mode == "fair":
        LOGGER.info("Splitting the dataset in fair mode..")
        FairSplitter(
            **config.dataset_splitter_kwargs
        ).split()
    else:
        raise ValueError(f"Split mode {config.split_mode} not supported!")


def create_splits_reports(config: DictConfig):

    # checking if the report is already generated
    train_report_dir = os.path.join(config.train_set_dir, "reports")
    test_report_dir = os.path.join(config.test_set_dir, "reports")

    if os.path.exists(train_report_dir) and os.path.exists(test_report_dir):
        # if the report is present skip the process and the force flag is not set
        if config.force_dataset_split_inspector:
            LOGGER.warning("Force dataset split inspector flag is set, deletting the train and test splits report...")
            os.system(f"rm -r {train_report_dir}")
            os.system(f"rm -r {test_report_dir}")

        else:
            LOGGER.info("The train split report is already generated! skipping the report generation step...")
            return

    LOGGER.info("Creating the report for the train and test splits...")

    inspector = JsonLineDatasetInspector(
        **config.train_split_inspector_kwargs
    )

    inspector.inspect()
    inspector.dump_report()
    inspector.dump_raw_report()
    inspector.generate_graphs()

    inspector = JsonLineDatasetInspector(
        **config.test_split_inspector_kwargs
    )

    inspector.inspect()
    inspector.dump_report()
    inspector.dump_raw_report()
    inspector.generate_graphs()


def create_object_datasets(config: DictConfig):
    
    # checking if the object datasets are already created
    if config.force_object_dataset_generator:
        LOGGER.warning("Force object dataset generator flag is set, deletting the object datasets...")
        if os.path.exists(config.object_datasets_dir):
            os.system(f"rm -r {config.object_datasets_dir}")

    else:
        count = 0
        for _class in config.object_target_classes:
            if os.path.exists(os.path.join(config.object_datasets_dir, f"object_dataset_{_class}")):
                count += 1
        if count == len(config.object_target_classes):
            LOGGER.info("All the object datasets are already created!")
            return
        elif count > 0:
            LOGGER.info("Some object datasets are already created! deleting them...")
            os.system(f"rm -r {config.object_datasets_dir}")
        # if count == 0 the object datasets are not created, so we can continue

    # loading the input dataset
    dataset = JsonLineDatasetBase(
        os.path.join(config.train_set_dir, "dataset.jsonl"),
        os.path.join(config.train_set_dir, "index"),
        config.binary
    )
    dataset.on_worker_init()

    # creating one object dataset for each class
    for _class in config.object_target_classes:
        LOGGER.info(f"Creating the object dataset for the class {_class}...")
        ObjectDatasetGenerator(
            input_dataset=dataset,
            image_dir=os.path.join(config.train_set_dir, "data"),
            target_class=_class,
            **config.object_dataset_generator_kwargs,
        ).convert()


def create_object_datasets_reports(config: DictConfig):
    """
        Create the reports for the object datasets
    """
    if config.force_object_dataset_inspector:
        LOGGER.warning("Force object dataset inspector flag is set, deletting the object datasets reports...")
        for _class in config.object_target_classes:
            report_dir = os.path.join(config.object_datasets_dir, f"object_dataset_{_class}", "reports")
            if os.path.exists(report_dir):
                os.system(f"rm -r {report_dir}")

    else:
        # checking if the report is already generated
        reports_found = 0
        for _class in config.object_target_classes:
            report_dir = os.path.join(config.object_datasets_dir, f"object_dataset_{_class}", "reports")
            if os.path.exists(report_dir) and \
                os.path.exists(os.path.join(report_dir, "brief_report.json")):
                    reports_found += 1

        if reports_found == len(config.object_target_classes):
            # if the report is present skip the process
            LOGGER.info("The object datasets reports are already generated! skipping the report generation step...")
            return

        elif reports_found > 0:
            LOGGER.info("Some object datasets reports are already generated! deleting them...")
            for _class in config.object_target_classes:
                report_dir = os.path.join(config.object_datasets_dir, f"object_dataset_{_class}", "reports")
                if os.path.exists(report_dir):
                    os.system(f"rm -r {report_dir}")

        # if reports_found == 0 the object datasets reports are not created, so we can continue

    LOGGER.info("Creating the reports for the object datasets...")
    for _class in config.object_target_classes:
        LOGGER.info(f"Creating the report for the object dataset {_class}...")
        inspector = JsonLineMaskDatasetInspector(
            os.path.join(config.object_datasets_dir, f"object_dataset_{_class}"),
            os.path.join(config.object_datasets_dir, f"object_dataset_{_class}", "reports"),
            config.binary
        )

        inspector.inspect()
        inspector.dump_report()
        inspector.dump_raw_report()
        inspector.generate_graphs()


def create_base_dataset(config: DictConfig):
    """
        Create the base dataset
    """
    if config.force_base_dataset_generator:
        LOGGER.warning("Force base dataset generator flag is set, deletting the base dataset...")
        if os.path.exists(config.base_dataset_dir):
            os.system(f"rm -r {config.base_dataset_dir}")

    elif os.path.exists(config.base_dataset_dir):
        # checking if the base dataset is empty
        if len(os.listdir(os.path.join(config.base_dataset_dir, "data"))) == 0:
            LOGGER.info("The base dataset is empty! deleting it...")
            os.system(f"rm -r {config.base_dataset_dir}")
        else:
            LOGGER.info("The base dataset is already created! skipping the base dataset creation step...")
            return

    # load the input dataset
    input_dataset = JsonLineDataset(
        image_folder_path =     os.path.join(config.train_set_dir, "data"),
        metadata_file_path =    os.path.join(config.train_set_dir, "dataset.jsonl"),
        index_file_path =       os.path.join(config.train_set_dir, "index"),
        masks_fields =          ["polygons"],
        classes =               config.object_target_classes,
        size =                  config.original_tile_size,
        binary =                config.binary
    )
    
    # initialize the input dataset
    input_dataset.on_worker_init()

    # load the image evaluator
    img_evaluator = SeverstalBaseEvaluator(
        **config.base_evaluator_kwargs
    )

    LOGGER.info("Creating the base dataset...")
    BaseDatasetGenerator(
        input_dataset =     input_dataset,
        image_dir=          os.path.join(config.train_set_dir, "data"),
        img_evaluator=      img_evaluator,
        **config.base_dataset_generator_kwargs
    ).convert()


def create_tile_datasets(config: DictConfig):
    """
    Method that create the tile datasets

    Args:
        config (DictConfig): _description_
    """
    def delete_tile_datasets(config):
        if os.path.exists(config.tile_train_set_dir):
            LOGGER.info("Deleting the tile train set...")
            os.system(f"rm -r {config.tile_train_set_dir}")
        if os.path.exists(config.tile_test_set_dir):
            LOGGER.info("Deleting the tile test set...")
            os.system(f"rm -r {config.tile_test_set_dir}")

    if config.force_tile_dataset_generator:
        LOGGER.warning("Force tile dataset generator flag is set, deletting the tile datasets...")
        delete_tile_datasets(config)
    elif (not os.path.exists(config.tile_train_set_dir)) or (not os.path.exists(config.tile_test_set_dir)):
        LOGGER.info("The tile datasets are not created! creating them...")
        delete_tile_datasets(config)
    elif os.path.exists(config.tile_train_set_dir) and os.path.exists(config.tile_test_set_dir):
        LOGGER.info("The tile datasets are already created! skipping the tile dataset creation step...")
        return

    # load the train dataset
    train_dataset = JsonLineDataset(
        image_folder_path =     os.path.join(config.train_set_dir, "data"),
        metadata_file_path =    os.path.join(config.train_set_dir, "dataset.jsonl"),
        index_file_path =       os.path.join(config.train_set_dir, "index"),
        masks_fields =          ["polygons"],
        classes =               config.object_target_classes,
        size =                  config.original_tile_size,
        binary =                config.binary
    )

    # load the test dataset
    test_dataset = JsonLineDataset(
        image_folder_path =     os.path.join(config.test_set_dir, "data"),
        metadata_file_path =    os.path.join(config.test_set_dir, "dataset.jsonl"),
        index_file_path =       os.path.join(config.test_set_dir, "index"),
        masks_fields =          ["polygons"],
        classes =               config.object_target_classes,
        size =                  config.original_tile_size,
        binary =                config.binary
    )

    # initialize the datasets
    train_dataset.on_worker_init()
    test_dataset.on_worker_init()

        # load the image evaluator
    image_evaluator = SeverstalBaseEvaluator(
        **config.base_evaluator_kwargs
    )

    LOGGER.info("Creating the tile train dataset...")
    TileDatasetPreprocessor(
        input_dataset = train_dataset,
        image_evaluator = image_evaluator,
        **config.tile_train_dataset_generator_kwargs
    ).convert()

    LOGGER.info("Creating the tile test dataset...")
    TileDatasetPreprocessor(
        input_dataset = test_dataset,
        **config.tile_test_dataset_generator_kwargs
    ).convert()


@hydra.main(config_path='../configs/data_preparation', config_name='severstal_dataset_preparation.yaml', version_base="1.1")
def main(config: DictConfig):
    """
        Script that load the Severstal steel defect dataset and prepare it for the trianing pipeline.

        Operations:
            V 1 - Download the dataset from kaggle
            V 2 - Extract the dataset and delete the zip file
            V 3 - Convert the dataset in jsonl format
                V 3.1 - create a report of the dataset
            V 4 - Split the dataset in train, and test (keeping the same distribution of the classes)
                V 4.1 - create a report of the train and test splits
            V 5 - Create the defects datasets to train the COIGAN model to inpaint the defects and stylegan to generate new masks, from the train dataset
                V 5.1 - Create the reports for the object datasets
            V 6 - Create the none defected img dataset to train another stylegan, to generate new none defected images, or to use directly in the inpainting process.
            7 - Convert in tile datasets the train and the test dataset, with the same size of the tiles in the precedenet processes

        TODO: list of bugs that need to be fixed or improvements:

    """

    LOGGER.info(f"Current folder: {os.getcwd()}")

    # 1 & 2 - download and extract the dataset if necessary
    download_and_extract_severstal_dataset(config)
    
    # 3 - convert the dataset in jsonl format
    convert_severstal_dataset_to_jsonl(config)

    # 3.1 - create report of the dataset
    create_severstal_dataset_report(config)

    # 4 - split the dataset in train and test set
    split_dataset(config)

    # 4.1 - create reports for the splitted datasets
    create_splits_reports(config)

    # 5 - Create the defects datasets to train the COIGAN model to inpaint the defects and stylegan to generate new masks, from the train dataset
    create_object_datasets(config)

    # 5.1 - create the reports for the object datasets
    create_object_datasets_reports(config)

    # 6 - create the none defected img dataset to train another stylegan, to generate new none defected images
    create_base_dataset(config)
    
    # 7 - convert in tile datasets the train and the test dataset, with the same size of the tiles used in the COIGAN model
    create_tile_datasets(config)



if __name__ == "__main__":
    main()