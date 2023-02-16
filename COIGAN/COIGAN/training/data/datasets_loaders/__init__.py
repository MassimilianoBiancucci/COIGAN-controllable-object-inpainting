import os
import logging
from omegaconf import OmegaConf, DictConfig

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from COIGAN.training.data.datasets_loaders.shape_dataloader import ShapeObjectDataloader
from COIGAN.training.data.datasets_loaders.object_dataloader import ObjectDataloader

from COIGAN.training.data.augmentation.augmentor import Augmentor
from COIGAN.training.data.augmentation.augmentation_presets import augmentation_presets_dict
from COIGAN.training.data.datasets_loaders.jsonl_object_dataset import JsonLineObjectDataset, JsonLineMaskObjectDataset
from COIGAN.training.data.datasets_loaders.coigan_severstal_steel_defects_dataset import CoiganSeverstalSteelDefectsDataset

from COIGAN.utils.ddp_utils import data_sampler

LOGGER = logging.getLogger(__name__)


def make_dataloader(config: OmegaConf, rank=None):
    """
    Method to create the dataset by params in config

    Args:
        config (OmegaConf): the config object with the data params used to build the dataset
        rank (int): the rank of the process (if passed is used to set the seed of the dataset)
    """
    rank = rank if rank is not None else 0 # if no rank is passed, set it to 0
    seed = config.data.seed * (rank + 1) # multiply the main seed by the rank to get a different seed for each process

    # create the dataset
    if config.data.kind == "severstal-steel-defect":
        dataset = make_severstal_steel_defect(config.data, seed=seed)
    else:
        raise "Dataset kind not supported"

    # extracting the worker_init_fn from the dataset if it exists
    worker_init_fn = dataset.on_worker_init if hasattr(dataset, "on_worker_init") else None

    # encapsulate the dataset in the torch dataloader
    dataloader = DataLoader(
        dataset=dataset,
        sampler=data_sampler(
            dataset, 
            shuffle=config.data.dataloader_shuffle,
            distributed=config.distributed
        ),
        worker_init_fn=worker_init_fn,
        **config.data.torch_dataloader_kwargs
    )
    
    return dataloader


def make_severstal_steel_defect(config: DictConfig, seed: int = None):
    """
    Method to preparare the dataset object 
    for the severstal steel defect dataset.

    Args:
        config (OmegaConf): the data config object
    """

    # load the base dataset
    base_dataset = ImageFolder(
        **config.base_dataset_kwargs,
        transform=augmentation_presets_dict["base_imgs_preset"]
    )


    # create the augmentor object
    augmentor = Augmentor(
        transforms=augmentation_presets_dict[config.augmentation_sets.mask_aug],
        only_imgs_transforms=augmentation_presets_dict[config.augmentation_sets.img_aug]
    )

    # generate the paths for the object datasets
    object_datasets_paths = [
        os.path.join(config.object_datasets.base_path, object_dataset_name)
        for object_dataset_name in config.object_datasets.names
    ]

    # create the shape dataset
    shape_dataloaders = [
        ShapeObjectDataloader(
            JsonLineMaskObjectDataset(
                object_dataset_path,
                binary=config.object_datasets.binary,
                augmentor=augmentor
            ),
            seed=seed,
            **config.shape_dataloader_kwargs
        )
        for object_dataset_path in object_datasets_paths
    ] if config.differet_shapes else None

    # create the object dataset
    object_dataloaders = [
        ObjectDataloader(
            JsonLineObjectDataset(
                object_dataset_path,
                binary=config.object_datasets.binary,
                augmentor=augmentor
            ),
            seed=seed,
            **config.object_dataloader_kwargs
        )
        for object_dataset_path in object_datasets_paths
    ]

    # create the COIGAN dataloader
    dataset = CoiganSeverstalSteelDefectsDataset(
        base_dataset,
        config.object_datasets.classes,
        object_dataloaders,
        shape_dataloaders,
        seed=seed,
        **config.coigan_dataset_kwargs
    )

    return dataset
        

#####################
# Debugging section #
if __name__ == "__main__":
    import hydra
    from tqdm import tqdm
    from COIGAN.utils.common_utils import sample_data
    from COIGAN.utils.debug_utils import check_nan

    @hydra.main(config_path="/home/ubuntu/hdd/COIGAN-controllable-object-inpainting/configs/training/", config_name="test_train.yaml")
    def main_debug(cfg: OmegaConf):
        dataloader = make_dataloader(cfg)
        loader = sample_data(dataloader)

        for sample in tqdm(dataloader):
            base_image =        check_nan(sample["base"]) # [base_r, base_g, base_b] the original image without any masking
            gen_in =            check_nan(sample["gen_input"]) # [base_r, base_g, base_b, mask_0, mask_1, mask_2, mask_3]
            gen_in_orig_masks = check_nan(sample["orig_gen_input_masks"]) # [mask_0, mask_1, mask_2, mask_3] the original masks without the noise
            disc_in_true =      check_nan(sample["disc_input"])# [defect_0_r, defect_0_g, defect_0_b, defect_1_r, defect_1_g, defect_1_b, defect_2_r, defect_2_g, defect_2_b, defect_3_r, defect_3_g, defect_3_b]    
            union_mask =        check_nan(sample["gen_input_union_mask"]) # [union_mask] the union mask of all the masks used in the generator input


    main_debug()





    