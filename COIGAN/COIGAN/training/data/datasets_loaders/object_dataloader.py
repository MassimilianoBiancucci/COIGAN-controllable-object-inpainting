import numpy as np
import torch
from random import Random

import logging

from typing import Tuple, Union, Dict, List, Optional
from omegaconf import ListConfig

from COIGAN.training.data.datasets_loaders.jsonl_object_dataset import JsonLineObjectDataset

LOGGER = logging.getLogger(__name__)


class ObjectDataloader:

    """
        Object that prepare the defect samples for the disrciminator.

        This object is used to load different numbers of defects of the same type and monut
        them together in a single image.
    """

    def __init__(
        self,
        input_dataset: JsonLineObjectDataset,
        sample_defects: List[int] = [0, 1, 2, 3],
        defects_probs: Optional[List[float]] = None,
        tile_size: Union[int, List[int], Tuple[int], ListConfig] = 256,
        seed: int = 42
    ):

        """
            Init method of the shape object dataloader.

            Args:
                input_dataset: dataset object that contains the input data, and return the mask as normalized torch.Tensor.
                sample_defects (Union[int, List[int], Tuple[int]]): number of shapes to mount in the image.
                    - int: the same number of shapes is used for each sample.
                    - List[int, int]: the number of shapes is sampled from a uniform distribution between the two values.
                defects_probs: probability of each defect to be sampled. if None the probability is uniform.
                    - List[Float]: the probability of each defect to be sampled.
                    - None: the probability is uniform. so each defect has 1/len(sample_defects) probability to be sampled.
                tile_size: size of the tile where the shapes are mounted.
                seed: seed to use for the random number generator.
        """

        self.input_dataset = input_dataset
        # enable the train mode from the input dataset if it is disabled
        self.input_dataset.train = True

        if isinstance(sample_defects, (list, tuple)):
            self.sample_defects = sample_defects
        elif isinstance(sample_defects, ListConfig):
            self.sample_defects = [shape for shape in sample_defects]
        elif isinstance(sample_defects, int):
            self.sample_defects = list(range(sample_defects+1))
        else:
            raise ValueError("sample_shapes must be a list or a tuple of int values.")

        # load the defects probabilities
        if defects_probs is None:
            self.defects_probs = [1 / len(self.sample_defects)] * len(self.sample_defects)
        else:
            if len(defects_probs) != len(self.sample_defects):
                raise ValueError(f"defects_probs must have the same length of sample_defects = {len(self.sample_defects)}.")
            elif sum(defects_probs) != 1:
                # normalize the defects probabilities
                defects_probs = np.array(defects_probs) / sum(defects_probs)
            self.defects_probs = defects_probs
        
        # define a list of defects numbers that exclude the 0 defects
        # and a second list of probs that exclude the prob of the o defects
        # with probs normalized
        self.sample_defects_nz = []
        self.defects_probs_nz = []
        for defects_n, defects_prob in zip(self.sample_defects, self.defects_probs):
            if defects_n != 0:
                self.sample_defects_nz.append(defects_n)
                self.defects_probs_nz.append(defects_prob)
        self.defects_probs_nz = np.array(self.defects_probs_nz) / sum(self.defects_probs_nz)


        self.tile_size = tile_size if isinstance(tile_size, (list, tuple)) else (tile_size, tile_size)

        # random index variables
        # variables used to retrive shapes from the dataset with the random strategy
        self.random = Random(seed)
        self.regenerate_random_idxs()


    def on_worker_init(self,  *args, **kwargs):
        """
            Init method for the workers.
        """
        self.input_dataset.on_worker_init()


    def regenerate_random_idxs(self, shuffle: bool = True):
        """
            Regenerate the random indexes used to sample the shapes from the dataset.
        """
        self.random_idxs = list(range(len(self.input_dataset)))

        if shuffle:
            self.random.shuffle(self.random_idxs)
    

    def generate_random_sample(self, avoid_mask: torch.Tensor = None, force_non_empty: bool = False):
        """
            Generate a random sample of the dataset.

            Args:
                avoid_mask: mask to avoid when sampling the shapes.
                    used in case the shapes should avoid to overlap with other objects in other layers.
                force_non_empty (bool) : if True, the sampler will add at least one object to the image.
        """

        # if the avoid mask is not None, the shapes should match the tile size
        if avoid_mask is not None:
            assert avoid_mask.shape == (self.tile_size[0], self.tile_size[1]), \
                f"The avoid mask must have the same shape of the tile size: {self.tile_size}, got {avoid_mask.shape} instead."

        # generate the img and mask tile
        img_tile = torch.zeros((3, self.tile_size[0], self.tile_size[1]), dtype=torch.float32)
        mask_tile = torch.zeros((self.tile_size[0], self.tile_size[1]), dtype=torch.float32)
        
        # sample the number of shapes to place in the image
        # considering the shapes probabilities
        if not force_non_empty:
            num_defects = self.random.choices(self.sample_defects, weights=self.defects_probs)[0]
        else:
            num_defects = self.random.choices(self.sample_defects_nz, weights=self.defects_probs_nz)[0]

        # extract from random_idxs num_shapes random indexes, removing it from the list
        if len(self.random_idxs) < num_defects:
            self.regenerate_random_idxs()
        idxs = [self.random_idxs.pop() for _ in range(num_defects)]

        # retrive shapes from the dataset
        defects = self.input_dataset[idxs]

        # mount the defects in the tile
        for i in range(num_defects):
            img_tile, mask_tile, avoid_mask = self.apply_defect(
                img_tile, 
                mask_tile, 
                defects[i][0].squeeze(0), # defect img
                defects[i][1].squeeze(0), # defect mask
                self.random,
                avoid_mask
            )
        
        return img_tile, mask_tile, avoid_mask


    def __getitem__(self, idx):
        """
            Get the sample at the index idx.
        """
        img, mask = self.generate_random_sample()
        return img, mask
    

    def __iter__(self):
        """
            Iterator of the dataset.
        """
        while True:
            yield self.__getitem__(0)


    def __len__(self):
        return len(self.input_dataset)


    @staticmethod
    def apply_defect(
        img: torch.Tensor, 
        mask: torch.Tensor, 
        defect_img: torch.Tensor,
        defect_mask: torch.Tensor,
        random: Random,
        avoid_mask: Optional[torch.Tensor] = None
    ):
        """
            Apply the defect to the img.

            this method try to apply a new shape on an existing mask,
            without overlapping the shapes, and try to reduce the amount of defect placed outside the img.

            Args:
                img: img where to apply the defect.
                mask: mask where to apply the defect.
                defect_img: defect img to apply.
                defect_mask: defect mask to apply.
                avoid_mask: mask to avoid when applying the defects.
            
            NOTE: if the avoid_mask is not None, the defects will be placed only where the avoid_mask is 0,
                otherwise the mask will be used for the check.
            
            Returns:
                img: img with the defect applied.
                mask: mask with the defect applied.
                avoid_mask: avoid_mask with the defect applied.
        """

        check_mask = mask if avoid_mask is None else avoid_mask

        # retrice the shapes of the mask and the shape
        mh, mw = mask.shape # mask dims
        sh, sw = defect_mask.shape # defect mask dims

        n_try = 3

        # check if the shape fit in the mask
        # if not try to place it in a random position
        # croping the shape where necessary
        if sh > mh or sw > mw:
            for i in range(n_try):
                # generate a random position for the shape
                if sh > mh:
                    dh = sh - mh
                    th = random.randint(-dh, 0)
                else:
                    th = random.randint(0, mh - sh)
                
                if sw > mw:
                    dw = sw - mw
                    tw = random.randint(-dw, 0)
                else:
                    tw = random.randint(0, mw - sw)

                # check if position is already occupied by another defect
                ch = max(th, 0) # crop height
                cw = max(tw, 0) # crop width
                che = min(th + sh, mh) # crop height end
                cwe = min(tw + sw, mw) # crop width end

                # extract the defect that will be inside the img
                sch = 0 if ch == th else -th
                sche = sh if che == th + sh else che - th           
                scw = 0 if cw == tw else -tw
                scwe = sw if cwe == tw + sw else cwe - tw

                cdefect_mask = defect_mask[sch:sche, scw:scwe]

                if torch.sum( # if the sum is 0, there is no intersection between the mask to avoid and the defect mask (cutted)
                    torch.bitwise_and( # return the intersection between the mask to avoid and the defect mask (cutted)
                        check_mask[ch:che, cw:cwe].bool(),
                        cdefect_mask.bool()
                    )
                ) == 0:

                    cdefect_img = defect_img[:, sch:sche, scw:scwe]
                    
                    # place the defect in the image
                    apply_coords = cdefect_mask > 0 #  coords where the defect must be applied
                    img[:, ch:che, cw:cwe][:, apply_coords] = cdefect_img[:, apply_coords]
                    mask[ch:che, cw:cwe][apply_coords] = 1.0

                    if avoid_mask is not None:
                        avoid_mask[ch:che, cw:cwe][apply_coords] = 1.0
                    else:
                        avoid_mask = mask.clone()

                    return img, mask, avoid_mask
                
                # if more than n try are made, return the image in the current state without
                # placing other defects
                if i == n_try - 1:
                    return img, mask, avoid_mask
        
        # if the defect fit in the img, try to place it in a random position 
        else:
            for i in range(n_try):
                # generate a random position for the shape
                th = random.randint(0, mh - sh)
                tw = random.randint(0, mw - sw)

                if torch.sum(
                    torch.bitwise_and(
                        check_mask[th:th + sh, tw:tw + sw].bool(),
                        defect_mask.bool()
                    )
                ) == 0:

                    # place the shape in the mask
                    apply_coords = defect_mask > 0 #  coords where the defect must be applied
                    img[:, th:th + sh, tw:tw + sw][:, apply_coords] = defect_img[:, apply_coords]
                    mask[th:th + sh, tw:tw + sw][apply_coords] = 1.0

                    if avoid_mask is not None:
                        avoid_mask[th:th + sh, tw:tw + sw][apply_coords] = 1.0
                    else:
                        avoid_mask = mask.clone()

                    return img, mask, avoid_mask
                
                # if more than n try are made, return the image in the current state without
                # placing other defects
                if i == n_try - 1:
                    return img, mask, avoid_mask



#####################################################
### DEBUG FUNCTIONS ###

def timeit(shape_dataloader):
    """
        Decorator to time a function.
    """

    for sample in tqdm(shape_dataloader):
        pass


def visualize(shape_dataloader):
    """
        Visualize the dataset.
    """
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    for sample in shape_dataloader:

        # convert a torch tensor to a numpy array
        img, mask = sample
        img = img.numpy()*255
        mask = mask.numpy()*255
        img = img.transpose(1, 2, 0).astype(np.uint8)
        mask = mask.astype(np.uint8)

        cv2.imshow("img", img)
        cv2.imshow("mask", mask)

        if cv2.waitKey(0) == ord("q"):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":

    import cv2
    from tqdm import tqdm
    from COIGAN.training.data.augmentation.augmentor import Augmentor
    from COIGAN.training.data.augmentation.augmentation_presets import mask_inpainting_preset, imgs_inpainting_preset

    augmentor = Augmentor(
        transforms=mask_inpainting_preset,
        only_imgs_transforms=imgs_inpainting_preset
    )

    object_dataset = JsonLineObjectDataset(
        "/home/max/thesis/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/object_datasets/object_dataset_2",
        binary = True,
        augmentor=augmentor
    )

    object_dataloader = ObjectDataloader(
        input_dataset = object_dataset,
        sample_defects=[0, 1, 2, 3],
        defects_probs=[0.4, 0.3, 0.2, 0.1]
    )

    object_dataloader.on_worker_init()

    timeit(object_dataloader)
    visualize(object_dataloader)



