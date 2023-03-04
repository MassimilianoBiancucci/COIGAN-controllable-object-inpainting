import os
import logging
from random import Random
import torch
from torchvision.datasets import ImageFolder
from omegaconf import DictConfig

from typing import List, Tuple, Union, Dict

from COIGAN.training.data.datasets_loaders.shape_dataloader import ShapeObjectDataloader
from COIGAN.training.data.datasets_loaders.object_dataloader import ObjectDataloader

from COIGAN.training.data.augmentation.noise_generators import make_noise_generator

from COIGAN.utils.debug_utils import check_nan

LOGGER = logging.getLogger(__name__)

class CoiganSeverstalSteelDefectsDataset:

    """
        Main dataset used to load the severstal files,
        for the COIGAN training.
    """

    def __init__(
        self,
        base_dataset: ImageFolder,
        defect_classes: List[str],
        defect_dataloaders: List[ObjectDataloader],
        shape_dataloaders: List[ShapeObjectDataloader] = None,
        ref_dataset: ImageFolder = None,
        mask_noise_generator_kwargs: DictConfig = None,
        mask_base_img: bool = False,
        allow_overlap: bool = False,
        force_non_empty: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        length: int = None
    ):
        """
        Init method of the CoiganSeverstalSteelDefectsDataset class.

        This dataset object require the following input datasets:
            - base dataset: the dataset containing the images without any defects
            - ref_dataset: the dataset containing all the images of the training set, with and without defects.
            - shape_dataloaders: a list of dataloaders, one for each type of defect.
                    Those dataloaders are used to create the masks for the defect generation.
            - defect dataloaders: a list of dataloaders, one for each type of defect.
                    Those dataloaders are used to create black images with the defects on them,
                    and to train the discriminator to detect the real defects.
            
        The output of this dataset is a tuple containing the following elements:
            - ["base"]: The base image, that it's not masked even if the mask_base_img flag is True.
            - ["ref"]: The reference image, used to train a discriminator that push the inpainted image to seems realistic.
            - ["gen_input"]: the input tensor of the generator, containing the base image that can be masked with the input defect masks
                        and the input defects masks (or the noise masks if the noise generator parameters are passed) concatenated. 
                        shape: (3+n, H, W) where n is the number of defect classes.
            - ["disc_input"]: the input tensor of the discriminator, containing the masked images of some defects of each class. 
                        shape: (3*n, H, W) where n is the number of defect classes.
            - ["orig_gen_input_masks"]: A tensor containing the original masks of the defects, without noise. 
                        shape: (n, H, W) where n is the number of defect classes.
        
        Args:
            base_dataset: the dataset containing the images without any defects
            shape_dataloaders: a list of dataloaders, one for each type of defect. Default None, if None the same defects are used for the generator and the discriminator.
                to the generator only the defects masks are passed, while to the discriminator the defects images are passed.
            defect dataloaders: a list of dataloaders, one for each type of defect.
            defect_classes: a list of strings, one for each type of defect.
            mask_noise_generator: a function that takes as input a mask and return a noisy mask.
                    if passed the getitem method will return a noisy mask in the gen_in field of the output, 
                    and another field for the original masks.
            mask_noise_generator_kwargs: a dictionary containing the kwargs for the mask_noise_generator.
            mask_base_img: if True, the base image will be masked with the defects masks.
            allow_overlap: specify if the defects masks can overlap between different classes or not.
            force_non_empty: if True force the shapes and objects to be non empty, at least one shape and one object of one class will be in the sample.
            shuffle: specify if the dataset should be shuffled or not.
            seed: seed used to shuffle the dataset.
            lenght: specify the lenght of the dataset. If None, the lenght of the base dataset is used.
        """

        # check the input
        if shape_dataloaders is not None:
            if len(shape_dataloaders) != len(defect_dataloaders):
                raise ValueError("The number of shape dataloaders and defect dataloaders must be the same.")

        if len(defect_dataloaders) != len(defect_classes):
            raise ValueError("The number of shape dataloaders and defect classes must be the same.")

        # datasets
        self.base_dataset = base_dataset
        self.ref_dataset = ref_dataset
        self.shape_dataloaders = shape_dataloaders
        self.defect_dataloaders = defect_dataloaders

        # dataset variables
        # base dataset vars
        self.base_regenerations = 0
        self.base_dataset_len = len(base_dataset)
        self.base_dataset_idxs = []

        # ref dataset vars
        self.ref_regenerations = 0
        self.ref_dataset_len = len(ref_dataset) if ref_dataset is not None else 0
        self.ref_dataset_idxs = []

        # shape dataloaders vars
        if shape_dataloaders is not None:
            self.shape_regenerations = [0] * len(shape_dataloaders)
            self.shape_dataloaders_len = [len(shape_dataloader) for shape_dataloader in shape_dataloaders]
            self.shape_dataloaders_idxs = [[] for _ in range(len(shape_dataloaders))]

        # defect dataloaders vars
        self.defect_regenerations = [0] * len(defect_dataloaders)
        self.defect_dataloaders_len = [len(defect_dataloader) for defect_dataloader in defect_dataloaders]
        self.defect_dataloaders_idxs = [[] for _ in range(len(defect_dataloaders))]

        # labels
        self.defect_classes = defect_classes
        
        # load the noise generator
        self.mask_noise_generator = make_noise_generator(**mask_noise_generator_kwargs) \
             if mask_noise_generator_kwargs is not None else None

        # other settings
        self.mask_base_img = mask_base_img
        self.allow_overlap = allow_overlap
        self.force_non_empty = force_non_empty
        self.shuffle = shuffle
        self.random = Random(seed)

        # set the length, if None use the base dataset length
        self.length = length if length is not None else self.base_dataset_len

        # regenerate the idxs
        self._regenerate_idxs()


    def on_worker_init(self,  *args, **kwargs):
        """
            Init method for the workers.
        """
        if self.shape_dataloaders is not None:
            for shape_dataloader in self.shape_dataloaders: shape_dataloader.on_worker_init()
        for defect_dataloader in self.defect_dataloaders: defect_dataloader.on_worker_init()


    def _regenerate_idxs(self):
        """
        Method that regenerate the idxs lists for each sub dataset.
        """
        self._regenerate_base_idxs()
        self._regenerate_ref_idxs()
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


    def _regenerate_ref_idxs(self):
        """
        Method that regenerate the idxs lists for the ref dataset.
        """
        # regenerate the ref dataset idxs
        if self.ref_dataset is None:
            return
        self.ref_regenerations += 1
        self.ref_dataset_idxs = list(range(self.ref_dataset_len))
        if self.shuffle: self.random.shuffle(self.ref_dataset_idxs)


    def _regenerate_shape_idxs(self, idx: int = -1):
        """
        Method that regenerate the idxs lists for each shape dataset,
        or only the requested one.

        Args:   
            idx: the index of the shape dataset to regenerate.
                If -1, regenerate all the shape datasets.
        """
        if self.shape_dataloaders is None:
            return
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

        # get the ref idx
        if self.ref_dataset is not None:
            if len(self.ref_dataset_idxs) == 0: self._regenerate_ref_idxs()
            ref_idx = self.ref_dataset_idxs.pop()
        else:
            ref_idx = None

        if only_base:
            return base_idx, ref_idx

        # get the shape idxs
        if self.shape_dataloaders is not None:
            shape_idxs = []
            for i, shape_dataloader_idxs in enumerate(self.shape_dataloaders_idxs):
                if len(shape_dataloader_idxs) == 0: self._regenerate_shape_idxs(i)
                shape_idxs.append(self.shape_dataloaders_idxs[i].pop())
        
        # get the defect idxs
        defect_idxs = []
        for i, defect_dataloader_idxs in enumerate(self.defect_dataloaders_idxs):
            if len(defect_dataloader_idxs) == 0: self._regenerate_defect_idxs(i)
            defect_idxs.append(self.defect_dataloaders_idxs[i].pop())
    
        return base_idx, ref_idx, shape_idxs, defect_idxs


    def get_sample(self):
        if self.allow_overlap:
            raise NotImplementedError("Overlap is not implemented yet.")
        else:
            return self.get_sample_no_overlap()


    def get_sample_no_overlap(self):
        """
            Get sample method of the dataset.
            This method is used to get a sample from the dataset.

            Returns:
                A dict with the following keys:
                - base: image of the base dataset without any modification.
                - gen_input: tensor of the base image (masked) concatenated with the masks of defects to inpaint (noised). channels: (br, bg, bb, mask0, mask1, mask2, mask3)
                - disc_input: tensor of concatenated masked images of defects. channels: (d0r, d0g, d0b, d1r, d1g, d1b, d2r, d2g, d2b, d3r, d3g, d3b)
                - orig_gen_input_masks:

        """

        # Get the indexes for each sub dataset
        base_idx, ref_idx = self._generate_sample_idxs(only_base=True)

        # Get the base image
        base = self.base_dataset[base_idx][0]

        # Get the reference image
        ref = self.ref_dataset[ref_idx][0] if ref_idx is not None else None
        
        # Get the defects images
        # used for train the discriminator
        defects = [None] * len(self.defect_dataloaders)
        defects_maks = [None] * len(self.defect_dataloaders)
        union_defect = torch.zeros(base.shape[-2:])
        defect_dataloaders_idxs = list(range(len(self.defect_dataloaders)))
        self.random.shuffle(defect_dataloaders_idxs)
        force_defect = False
        for idx in defect_dataloaders_idxs:
            if self.force_non_empty and defect_dataloaders_idxs[-1] == idx:
                # if this is the last defect added, check if there is at least another defect in union_defect
                # otherwise force the last generator to add a defect
                if torch.sum(union_defect) == 0:
                    force_defect = True
            defect, dafect_mask, union_defect = self.defect_dataloaders[idx].generate_random_sample(union_defect, force_defect)
            defects[idx] = defect
            defects_maks[idx] = dafect_mask.unsqueeze(0)


        if self.shape_dataloaders is not None:
            # Get the masks for each defect
            # used for the inpainting process
            # input of generator
            masks = [None] * len(self.shape_dataloaders)
            union_mask = torch.zeros(base.shape[-2:])
            shape_dataloaders_idxs = list(range(len(self.shape_dataloaders)))
            self.random.shuffle(shape_dataloaders_idxs)
            force_shape = False
            for idx in shape_dataloaders_idxs:
                if self.force_non_empty and shape_dataloaders_idxs[-1] == idx:
                    # if this is the last shape added, check if there is at least another shape in union_mask
                    # otherwise force the last generator to add a shape
                    if torch.sum(union_mask) == 0:
                        force_shape = True
                mask, union_mask = self.shape_dataloaders[idx].generate_random_sample(union_mask, force_shape)
                masks[idx] = mask.unsqueeze(0)
        else:
            # if no shape dataloader are provided, as masks and union_mask
            # use the union_defect tensor. This way the defects passed to 
            # the generator and the discriminator are the same.
            masks = defects_maks
            union_mask = union_defect

        sample =  {}
        
        sample["base"] = base
        base_masked = base.clone()

        if ref is not None: sample["ref"] = ref

        # Store the union of the shapes used as input of the generator
        sample["gen_input_union_mask"] = union_mask

        # masking the base image
        if self.mask_base_img:
            # if mask_base_img is True, the base image is masked with the union mask
            base_masked[:, union_mask > 0] = 0
        
        masks = torch.cat(masks, dim=0)
        if self.mask_noise_generator is not None:
            noise_masks = self.mask_noise_generator(masks)
            sample["gen_input"] = torch.cat([base_masked, noise_masks], dim=0).contiguous()
            
        else:
            sample["gen_input"] = torch.cat([base_masked, masks], dim=0).contiguous()

        sample["disc_input"] = torch.cat(defects, dim=0).contiguous()
        sample["orig_gen_input_masks"] = masks.contiguous()

        return sample


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
        return self.length


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

    n_classes = len(COIGAN_dataloader.defect_classes)

    cv2.namedWindow("base", cv2.WINDOW_NORMAL)
    cv2.namedWindow("base_masked", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ref", cv2.WINDOW_NORMAL)

    for i in range(n_classes):
        cv2.namedWindow("defect_{}".format(i), cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask_orig_{}".format(i), cv2.WINDOW_NORMAL)
        cv2.namedWindow("mask_noise_{}".format(i), cv2.WINDOW_NORMAL)

    #cv2.namedWindow("defects_masks_union", cv2.WINDOW_NORMAL)

    while True:
        sample = COIGAN_dataloader.get_sample_no_overlap()

        img = (sample["base"].numpy()*255).transpose(1, 2, 0).astype(np.uint8)
        ref_img = (sample["ref"].numpy()*255).transpose(1, 2, 0).astype(np.uint8)

        # unwrap the generator input
        gen_input = sample["gen_input"]
        masked_img = (gen_input[:3].numpy()*255).transpose(1, 2, 0).astype(np.uint8)
        masks = gen_input[3:]
        noise_masks = [
            (masks[i].numpy()*255).astype(np.uint8)
            for i in range(n_classes)
        ]

        orig_masks = sample["orig_gen_input_masks"]
        orig_masks = [
            (orig_masks[i].numpy()*255).astype(np.uint8)
            for i in range(n_classes)
        ]

        # convert the defects used as input for the discriminator
        defects = [
            (sample["disc_input"][i*3:(i*3)+3].numpy()*255).transpose(1, 2, 0).astype(np.uint8)
            for i in range(n_classes)
        ]

        # visualize the base image
        cv2.imshow("base", img)
        cv2.imshow("base_masked", masked_img)
        cv2.imshow("ref", ref_img)

        # visualize the shapes
        for i in range(n_classes):
            cv2.imshow("mask_orig_{}".format(i), orig_masks[i])
            cv2.imshow("mask_noise_{}".format(i), noise_masks[i])
            cv2.imshow("defect_{}".format(i), defects[i])

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

    ref_dataset = ImageFolder(
        root="/home/max/thesis/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/tile_train_set",
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
            shape_probs=[0.4, 0.45, 0.1, 0.05]
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
            defects_probs=[0.4, 0.45, 0.1, 0.05]
        ) for object_dataset_path in object_datasets_paths
    ]

    # create the COIGAN dataloader
    coigan_dataloader = CoiganSeverstalSteelDefectsDataset(
        base_dataset,
        ["defect_1", "defect_2", "defect_3", "defect_4"],
        object_dataloaders,
        shape_dataloaders,
        ref_dataset=ref_dataset,
        mask_base_img = True,
        allow_overlap = False,
        force_non_empty=True,
        length = 100000,
        mask_noise_generator_kwargs = {
            "kind": "multiscale",
            "kind_kwargs": {
                "base_generator_kwargs": {
                    "kind": "gaussian",
                    "kind_kwargs": {
                        "mean": 0.5,
                        "std": 0.08
                    }
                },
                "scales": [1, 2, 4],
                "strategy": "replace"
            }
        }
    )
    coigan_dataloader.on_worker_init()

    #timeit(coigan_dataloader)
    visualize(coigan_dataloader)




        
