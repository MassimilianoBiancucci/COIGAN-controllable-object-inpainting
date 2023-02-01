import os
import logging
from random import Random
import torch
from torchvision.datasets import ImageFolder

from typing import List, Tuple, Union, Dict

from COIGAN.training.data.datasets_loaders.shape_dataloader import ShapeObjectDataloader
from COIGAN.training.data.datasets_loaders.object_dataloader import ObjectDataloader

LOGGER = logging.getLogger(__name__)

class CoiganSeverstalSteelDefectsDataset:

    """
        Main dataset used to load the severstal files,
        for the COIGAN training.
    """

    def __init__(
        self,
        base_dataset: ImageFolder,
        shape_dataloaders: List[ShapeObjectDataloader],
        defect_dataloaders: List[ObjectDataloader],
        defect_classes: List[str],
        allow_overlap: bool = False,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Init method of the CoiganSeverstalSteelDefectsDataset class.

        This dataset object require the following input datasets:
            - base dataset: the dataset containing the images without any defects
            - shape_dataloaders: a list of dataloaders, one for each type of defect.
                    Those dataloaders are used to create the masks for the defect generation.
            - defect dataloaders: a list of dataloaders, one for each type of defect.
                    Those dataloaders are used to create black images with the defects on them,
                    and to train the discriminator to detect the real defects.
            
        The output of this dataset is a tuple containing the following elements:
            - ["base"] : the original image in torch.Tensor format.
            - ["masks"] : a torch.Tensor of masks, one channel for each type of defect.
            - ["defects"] : a list of images, one for each type of defect.
            - ["defects_union"] : a single image containing all the defects.
        
        NOTE:
            - The dataset take as lenght the lenght of the base dataset.

            
        Args:
            base_dataset: the dataset containing the images without any defects
            shape_dataloaders: a list of dataloaders, one for each type of defect.
            defect dataloaders: a list of dataloaders, one for each type of defect.
            defect_classes: a list of strings, one for each type of defect.
            allow_defect_masks_overlap: specify if the defects masks can overlap between different classes or not.
            shuffle: specify if the dataset should be shuffled or not.
            seed: seed used to shuffle the dataset.
        """

        # check the input
        if len(shape_dataloaders) != len(defect_dataloaders):
            raise ValueError("The number of shape dataloaders and defect dataloaders must be the same.")

        if len(defect_dataloaders) != len(defect_classes):
            raise ValueError("The number of shape dataloaders and defect classes must be the same.")

        # datasets
        self.base_dataset = base_dataset
        self.shape_dataloaders = shape_dataloaders
        self.defect_dataloaders = defect_dataloaders

        # dataset variables
        # base dataset vars
        self.base_regenerations = 0
        self.base_dataset_len = len(base_dataset)
        self.base_dataset_idxs = []

        # shape dataloaders vars
        self.shape_regenerations = [0] * len(shape_dataloaders)
        self.shape_dataloaders_len = [len(shape_dataloader) for shape_dataloader in shape_dataloaders]
        self.shape_dataloaders_idxs = [[] for _ in range(len(shape_dataloaders))]

        # defect dataloaders vars
        self.defect_regenerations = [0] * len(defect_dataloaders)
        self.defect_dataloaders_len = [len(defect_dataloader) for defect_dataloader in defect_dataloaders]
        self.defect_dataloaders_idxs = [[] for _ in range(len(defect_dataloaders))]

        # labels
        self.defect_classes = defect_classes
        
        # other settings
        self.allow_overlap = allow_overlap
        self.shuffle = shuffle
        self.random = Random(seed)

        # regenerate the idxs
        self._regenerate_idxs()


    def on_worker_init(self,  *args, **kwargs):
        """
            Init method for the workers.
        """
        for shape_dataloader in self.shape_dataloaders: shape_dataloader.on_worker_init()
        for defect_dataloader in self.defect_dataloaders: defect_dataloader.on_worker_init()


    def _regenerate_idxs(self):
        """
        Method that regenerate the idxs lists for each sub dataset.
        """
        self._regenerate_base_idxs()
        self._regenerate_shape_idxs()
        self._regenerate_defect_idxs()


    def _regenerate_base_idxs(self):
        """
        Method that regenerate the idxs lists for the base dataset.
        """
        # regenerate the base dataset idxs
        self.base_regenerations += 1
        self.base_dataset_idxs = list(range(self.base_dataset_len))
        if self.shuffle: self.random.shuffle(self.base_dataset_idxs)


    def _regenerate_shape_idxs(self, idx: int = -1):
        """
        Method that regenerate the idxs lists for each shape dataset,
        or only the requested one.

        Args:   
            idx: the index of the shape dataset to regenerate.
                If -1, regenerate all the shape datasets.
        """
        if idx == -1:
            # regenerate the shape dataloaders idxs
            for i, shape_dataloader_len in enumerate(self.shape_dataloaders_len):
                if len(self.shape_dataloaders_idxs[i]) == 0:
                    self.shape_regenerations[i] += 1
                    self.shape_dataloaders_idxs[i] = list(range(shape_dataloader_len))
                    if self.shuffle: self.random.shuffle(self.shape_dataloaders_idxs[i])
        else:
            self.shape_regenerations[idx] += 1
            self.shape_dataloaders_idxs[idx] = list(range(self.shape_dataloaders_len[idx]))
            if self.shuffle: self.random.shuffle(self.shape_dataloaders_idxs[idx])


    def _regenerate_defect_idxs(self, idx: int = -1):
        """
        Method that regenerate the idxs lists for each defect dataset,
        or only the requested one.

        Args:
            idx: the index of the defect dataset to regenerate.
                If -1, regenerate all the defect datasets.
        """
        if idx == -1:
            # regenerate the defect dataloaders idxs
            for i, defect_dataloader_len in enumerate(self.defect_dataloaders_len):
                if len(self.defect_dataloaders_idxs[i]) == 0:
                    self.defect_regenerations[i] += 1
                    self.defect_dataloaders_idxs[i] = list(range(defect_dataloader_len))
                    if self.shuffle: self.random.shuffle(self.defect_dataloaders_idxs[i])
        else:
            self.defect_regenerations[idx] += 1
            self.defect_dataloaders_idxs[idx] = list(range(self.defect_dataloaders_len[idx]))
            if self.shuffle: self.random.shuffle(self.defect_dataloaders_idxs[idx])


    def _generate_sample_idxs(self, only_base: bool = False):
        """
        This method generate a collection of idxs for each sub dataset.
        and think about the indexes lists regeneration if necessary.
        """
        # get the base idx
        if len(self.base_dataset_idxs) == 0: self._regenerate_base_idxs()
        base_idx = self.base_dataset_idxs.pop()

        if not only_base:
            # get the shape idxs
            shape_idxs = []
            for i, shape_dataloader_idxs in enumerate(self.shape_dataloaders_idxs):
                if len(shape_dataloader_idxs) == 0: self._regenerate_shape_idxs(i)
                shape_idxs.append(self.shape_dataloaders_idxs[i].pop())
            
            # get the defect idxs
            defect_idxs = []
            for i, defect_dataloader_idxs in enumerate(self.defect_dataloaders_idxs):
                if len(defect_dataloader_idxs) == 0: self._regenerate_defect_idxs(i)
                defect_idxs.append(self.defect_dataloaders_idxs[i].pop())
        
            return base_idx, shape_idxs, defect_idxs
        
        return base_idx


    def get_sample(self):
        if self.allow_overlap:
            raise NotImplementedError("Overlap is not implemented yet.")
        else:
            return self.get_sample_no_overlap()


    def get_sample_no_overlap(self):
        """
            Get sample method of the dataset.
            This method is used to get a sample from the dataset.
        """

        # Get the indexes for each sub dataset
        base_idx = self._generate_sample_idxs(only_base=True)

        # Get the base image
        base = self.base_dataset[base_idx][0]

        # Get the masks for each defect
        # used for the inpainting process
        # input of generator
        masks = [None] * len(self.shape_dataloaders)
        union_mask = torch.zeros(base.shape[-2:])
        shape_dataloaders_idxs = list(range(len(self.shape_dataloaders)))
        self.random.shuffle(shape_dataloaders_idxs)
        for idx in shape_dataloaders_idxs:
            mask, union_mask = self.shape_dataloaders[idx].generate_random_sample(union_mask)
            masks[idx] = mask.unsqueeze(0)
        
        # Get the defects images
        # used for train the discriminator
        defects = [None] * len(self.defect_dataloaders)
        defects_maks = [None] * len(self.defect_dataloaders)
        union_defect = torch.zeros(base.shape[-2:])
        defect_dataloaders_idxs = list(range(len(self.defect_dataloaders)))
        self.random.shuffle(defect_dataloaders_idxs)
        for idx in defect_dataloaders_idxs:
            defect, dafect_mask, union_defect = self.defect_dataloaders[idx].generate_random_sample(union_defect)
            defects[idx] = defect
            defects_maks[idx] = dafect_mask

        # Store the union of the defects
        #sample["defects_masks_union"] = union_defect

        # Return the sample
        return {
            "gen_input": torch.cat([base, *masks], dim=0).contiguous(),
            "disc_input": torch.cat(defects, dim=0).contiguous(),
            #"defects_masks": torch.cat(defects_maks, dim=0).contiguous()
        }


    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
            Get item method of the dataset.
            This method is used to get a sample from the dataset.
        """
        return self.get_sample()


    def __iter__(self):
        for i in range(len(self)):
            yield self.get_sample()
    

    def __next__(self):
        return self.get_sample()


    def __len__(self):
        return len(self.base_dataset)


def timeit(dataloader):
    """
        Time the dataloader.
    """
    dt = []

    for i in tqdm(range(10000)):
        t0 = time.time()
        sample = dataloader.get_sample_no_overlap()
        t1 = time.time()
        dt.append(t1 - t0)
    
    print("Mean time: {}".format(np.mean(dt)))
    print("Std time: {}".format(np.std(dt)))
    print("Max time: {}".format(np.max(dt)))
    print("Min time: {}".format(np.min(dt)))
    print("Elapsed time: {}".format(np.sum(dt)))


def visualize(COIGAN_dataloader: CoiganSeverstalSteelDefectsDataset):
    """
        Visualize the dataset.
        TODO: need a refactor
    """
    raise NotImplementedError("Visualize need a refactor to work with the new output format")

    n_classes = len(COIGAN_dataloader.defect_classes)

    cv2.namedWindow("base", cv2.WINDOW_NORMAL)

    for i in range(n_classes):
        cv2.namedWindow("defect_{}".format(i), cv2.WINDOW_NORMAL)
        cv2.namedWindow("defect_mask_{}".format(i), cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask_{}".format(i), cv2.WINDOW_NORMAL)

    cv2.namedWindow("defects_masks_union", cv2.WINDOW_NORMAL)

    while True:
        sample = COIGAN_dataloader.get_sample_no_overlap()

        # convert the base image
        img = (sample["base"].numpy()*255).transpose(1, 2, 0).astype(np.uint8)

        # convert the input generator masks
        shapes = [
            (sample["masks"][i].numpy()*255).astype(np.uint8)
            for i in range(n_classes)
        ]

        # convert the defects used as input for the discriminator
        defects = [
            (sample["defects"][i].numpy()*255).transpose(1, 2, 0).astype(np.uint8)
            for i in range(n_classes)
        ]

        # convert the defects masks used as input for the discriminator
        defects_masks = [
            (sample["defects_masks"][i].numpy()*255).astype(np.uint8)
            for i in range(n_classes)
        ]

        defects_masks_union = (sample["defects_masks_union"].numpy()*255).astype(np.uint8)

        # visualize the base image
        cv2.imshow("base", img)

        # visualize the shapes
        for i in range(n_classes):
            cv2.imshow("mask_{}".format(i), shapes[i])
            cv2.imshow("defect_{}".format(i), defects[i])
            cv2.imshow("defect_mask_{}".format(i), defects_masks[i])
        
        cv2.imshow("defects_masks_union", defects_masks_union)

        if cv2.waitKey(0) == ord("q"):
            break
    
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    import cv2
    import numpy as np
    import time
    from tqdm import tqdm

    from COIGAN.training.data.augmentation.augmentor import Augmentor
    from COIGAN.training.data.augmentation.augmentation_presets import base_imgs_preset
    from COIGAN.training.data.augmentation.augmentation_presets import mask_inpainting_preset, imgs_inpainting_preset
    
    from COIGAN.training.data.datasets_loaders.jsonl_object_dataset import JsonLineObjectDataset, JsonLineMaskObjectDataset

    # load the base dataset
    base_dataset = ImageFolder(
        root="/home/max/thesis/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/base_dataset",
        transform=base_imgs_preset
    )

    # generate the paths for the object datasets
    object_datasets_base_name = "object_dataset_"
    n_object_datasets = 4
    object_datasets_names = [object_datasets_base_name + str(i) for i in range(n_object_datasets)]
    object_datasets_base_path = "/home/max/thesis/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/object_datasets"
    object_datasets_paths = [os.path.join(object_datasets_base_path, object_dataset_name) for object_dataset_name in object_datasets_names]

    # create the shape augmentor
    shape_augmentor = Augmentor(
        transforms=mask_inpainting_preset,
        only_imgs_transforms=imgs_inpainting_preset
    )
    
    # create the shape dataloaders
    shape_dataloaders = [
        ShapeObjectDataloader(
            JsonLineMaskObjectDataset(
                object_dataset_path,
                binary = True,
                augmentor=shape_augmentor
            ),
            sample_shapes=[0, 1, 2, 3],
            shape_probs=[0.2, 0.5, 0.2, 0.1]
        ) for object_dataset_path in object_datasets_paths
    ]

    # create the defect augmentor
    object_dataloaders = [
        ObjectDataloader(
            JsonLineObjectDataset(
                object_dataset_path,
                binary = True,
                augmentor=shape_augmentor,
            ),
            sample_defects=[0, 1, 2, 3],
            defects_probs=[0.2, 0.5, 0.2, 0.1]
        ) for object_dataset_path in object_datasets_paths
    ]

    # create the COIGAN dataloader
    coigan_dataloader = CoiganSeverstalSteelDefectsDataset(
        base_dataset,
        shape_dataloaders,
        object_dataloaders,
        ["defect_1", "defect_2", "defect_3", "defect_4"],
        allow_overlap=False
    )
    coigan_dataloader.on_worker_init()

    timeit(coigan_dataloader)
    #visualize(coigan_dataloader)




        
