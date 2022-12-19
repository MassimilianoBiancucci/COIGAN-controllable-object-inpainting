import os
import logging

from omegaconf import OmegaConf

from torch.utils.data import DataLoader

#from COIGAN.training.data.defect_dataset import DefectDataset

LOGGER = logging.getLogger(__name__)


def make_base_dataloader(kind, dataset_kwargs, dataloader_kwargs):
    """
    Method that create and return a dataloader for training, based on the configuration file.

    Args:
        kind (str): kind of dataset to use
        dataset_kwargs (dict): kwargs for the dataset
        dataloader_kwargs (dict): kwargs for the dataloader
    
    Returns:
        dataloader: dataloader for training
    """

    LOGGER.info(f"Creating {kind} dataloader for training")

    on_worker_init = None
    
    if kind == "defect_dataset":
        dataset = {} # DefectsDataset(**dataset_kwargs)
        #NOTE: the dataset keep open a jsonl file, so it need to be opened
        # when the dataloader thread is created. otherwise the dataloader rise an error.
        on_worker_init = dataset.on_worker_init

    elif kind == "car_dataset":
        dataset = {} # CarsDataset(**dataset_kwargs)
        #NOTE: the dataset keep open a jsonl file, so it need to be opened
        # when the dataloader thread is created. otherwise the dataloader rise an error.
        on_worker_init = dataset.on_worker_init
    
    else:
        raise ValueError(f"Unknown data kind: {kind}")

    dataloader = DataLoader(
        dataset,
        **dataloader_kwargs,
        worker_init_fn=on_worker_init
    )

    return dataloader


def make_objects_dataloader(kind, dataset_kwargs, dataloader_kwargs):
    """
    Method that create and return a dataloader for validation, based on the configuration file.

    Args:
        kind (str): kind of dataset to use
        dataset_kwargs (dict): kwargs for the dataset
        dataloader_kwargs (dict): kwargs for the dataloader
    
    Returns:
        dataloader: dataloader for validation
    """

    LOGGER.info(f"Creating {kind} dataloader for validation")

    if kind == "defect_dataset":
        dataset = DefectDataset(**dataset_kwargs)
    
    else:
        raise ValueError(f"Unknown data kind: {kind}")

    dataloader = DataLoader(
        dataset,
        **dataloader_kwargs
    )

    return dataloader