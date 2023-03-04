import numpy as np
import torch
from random import Random

import logging

from typing import Tuple, Union, Dict, List, Optional
from omegaconf import ListConfig

from COIGAN.training.data.datasets_loaders.jsonl_object_dataset import JsonLineMaskObjectDataset

LOGGER = logging.getLogger(__name__)


class ShapeObjectDataloader:

    """
        Object that prepare the samples for the shape gan generator.

        This object is used to load different numbers of shapes of the same type and mout
        them together in a single image.
    """

    def __init__(
        self,
        input_dataset: JsonLineMaskObjectDataset,
        sample_shapes: Union[int, List[int], Tuple[int]] = [0, 1, 2, 3],
        shape_probs: Optional[List[float]] = None,
        tile_size: Union[int, List[int], Tuple[int], ListConfig] = 256,
        out_channels: int = 1,
        seed: int = 42
    ):

        """
            Init method of the shape object dataloader.

            Args:
                input_dataset: dataset object that contains the input data, and return the mask as normalized torch.Tensor.

                sample_shapes (Union[List[int], Tuple[int]]): number of shapes to mount in the image.
                    - List[int]: the number of shapes is sampled from the numbers passed in the list.
                    - Tuple[int]: the number of shapes is sampled from the range of numbers passed in the tuple.
                    - int: the number of shapes is sampled from the range [0, sample_shapes].

                shapes_probs (Optional[List[float]]): probability of each shape to be sampled.
                    - List[float]: the probability of each shape to be sampled is passed in the list. Must match the number of shapes.
                    - None: the probability of each shape is equal to 1 / number of shapes. (default: None)

                tile_size: size of the tile where the shapes are mounted.
                
                out_channels: number of channels of the output mask.
                seed: seed to use for the random number generator.

        """

        self.input_dataset = input_dataset
        # enable the train mode from the input dataset if it is disabled
        self.input_dataset.train = True

        if isinstance(sample_shapes, (list, tuple)):
            self.sample_shapes = sample_shapes
        elif isinstance(sample_shapes, ListConfig):
            self.sample_shapes = [shape for shape in sample_shapes]
        elif isinstance(sample_shapes, int):
            self.sample_shapes = list(range(sample_shapes+1))
        else:
            raise ValueError("sample_shapes must be a list or a tuple of int values.")

        # load the shapes probabilities
        if shape_probs is None:
            self.shapes_probs = [1 / len(self.sample_shapes)] * len(self.sample_shapes)
        else:
            if len(shape_probs) != len(self.sample_shapes):
                raise ValueError(f"shape_probs must have the same length of sample_shapes = {len(self.sample_shapes)}.")
            elif np.sum(shape_probs) != 1:
                shape_probs = np.array(shape_probs) / np.sum(shape_probs)
            self.shapes_probs = shape_probs

        # define a list of defects numbers that exclude the 0
        # and a second list of probs that exclude the prob of 0 shapes
        # with probs normalized
        self.sample_shapes_nz = []
        self.shapes_probs_nz = []
        for shape_n, shape_prob in zip(self.sample_shapes, self.shapes_probs):
            if shape_n != 0:
                self.sample_shapes_nz.append(shape_n)
                self.shapes_probs_nz.append(shape_prob)
        self.shapes_probs_nz = np.array(self.shapes_probs_nz) / sum(self.shapes_probs_nz)

        self.tile_size = tile_size if isinstance(tile_size, (list, tuple)) else (tile_size, tile_size)

        # set the number of channels 
        self.out_channels = out_channels

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
    

    def generate_random_sample(self, avoid_mask: Optional[torch.Tensor] = None, force_non_empty: bool = False):
        """
            Generate a random sample of the dataset.
        """
        # generate the tile
        tile = torch.zeros(self.tile_size, dtype=torch.float32)

        # sample the number of shapes to place in the image
        # considering the shapes probabilities
        if not force_non_empty:
            num_shapes = self.random.choices(self.sample_shapes, weights=self.shapes_probs)[0]
        else:
            num_shapes = self.random.choices(self.sample_shapes_nz, weights=self.shapes_probs_nz)[0]

        # extract from random_idxs num_shapes random indexes, removing it from the list
        if len(self.random_idxs) < num_shapes:
            self.regenerate_random_idxs()
        idxs = [self.random_idxs.pop() for _ in range(num_shapes)]

        # retrive shapes from the dataset
        shapes = self.input_dataset[idxs]

        # mount the shapes in the tile
        _avoid_mask = avoid_mask
        for i in range(num_shapes):
            tile, _avoid_mask = self.apply_shape(
                tile,
                shapes[i].squeeze(0),
                self.random,
                avoid_mask=_avoid_mask
            )
        
        if avoid_mask is None:
            return tile
        else:
            return tile, _avoid_mask


    def __getitem__(self, idx):
        """
            Get the sample at the index idx.
        """
        mask = self.generate_random_sample()

        if self.out_channels > 1:
            mask = torch.stack([mask] * self.out_channels, dim=0)

        return mask


    def __iter__(self):
        """
            Iterator of the dataset.
        """
        while True:
            yield self.__getitem__(0)


    def __len__(self):
        return len(self.input_dataset)


    @staticmethod
    def apply_shape(
        mask: torch.Tensor, 
        shape: torch.Tensor,
        random: Random,
        avoid_mask: torch.Tensor = None
    ):
        """
            Apply the shape to the mask.

            this method try to apply a new shape on an existing mask,
            without overlapping the shapes, and try to reduce the amount of shape placed outside the mask.

            Args:
                mask: mask where the shape is applied. contain other shapes already placed from the same class.
                shape: shape to apply to the mask.
                avoid_mask: reference mask, signal where the shape can't be placed, because there are other shapes from ther. Default: None
            
            NOTE: if the avoid_mask is not None, the shape will be placed only where the avoid_mask is 0,
                otherwise the mask will be used for the check.
            
            Returns:
                mask: the mask with the shape applied.
                avoid_mask: the avoid_mask with the shape applied.
        """

        check_mask = mask if avoid_mask is None else avoid_mask

        # retrice the shapes of the mask and the shape
        mh, mw = mask.shape # mask dims
        sh, sw = shape.shape # shape dims

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

                # check if position is already occupied by another shape
                ch = max(th, 0) # crop height
                cw = max(tw, 0) # crop width
                che = min(th + sh, mh) # crop height end
                cwe = min(tw + sw, mw) # crop width end

                # extract the shape that will be inside the mask
                sch = 0 if ch == th else -th
                sche = sh if che == th + sh else che - th                
                scw = 0 if cw == tw else -tw
                scwe = sw if cwe == tw + sw else cwe - tw

                cshape = shape[sch:sche, scw:scwe]

                if torch.sum(
                    torch.bitwise_and( # return the intersection btween the check_mask and the shape mask
                        check_mask[ch:che, cw:cwe].bool(),
                        cshape.bool()
                    )
                ) == 0:
                    
                    # place the shape in the mask
                    mask[ch:che, cw:cwe] = cshape

                    if avoid_mask is not None:
                        avoid_mask[ch:che, cw:cwe][cshape > 0] = 1.0
                    else:
                        avoid_mask = mask.clone()

                    return mask, avoid_mask
                
                if i == n_try - 1:
                    # the last try, check if the shape overlap the avoid_mask, in that case
                    # the shape overlap other classes, so don't place it.
                    
                    cshape = shape[sch:sche, scw:scwe]

                    if avoid_mask is None or torch.sum(
                        torch.bitwise_and(
                            torch.bitwise_and( # return the avoid_mask with the current mask subtracted
                                avoid_mask[ch:che, cw:cwe].bool(),
                                torch.bitwise_not(mask[ch:che, cw:cwe].bool())
                            ),
                            cshape.bool()
                        )
                    ) == 0:

                        apply_coords = cshape > 0

                        # set the mask values to 1 where the shape is placed
                        mask[ch:che, cw:cwe][apply_coords] = 1.0
                    
                        if avoid_mask is not None:
                            avoid_mask[ch:che, cw:cwe][apply_coords] = 1.0
                    
                    if avoid_mask is None:
                        avoid_mask = mask.clone()

                    return mask, avoid_mask
        
        # if the shape fit in the mask, try to place it in a random position 
        else:
            for i in range(n_try):
                # generate a random position for the shape
                th = random.randint(0, mh - sh)
                tw = random.randint(0, mw - sw)

                if torch.sum( # return the area of the intersection between the check_mask and the shape mask
                    torch.bitwise_and( # return the intersection btween the check_mask and the shape mask
                        check_mask[th:th + sh, tw:tw + sw].bool(),
                        shape.bool()
                    )
                ) == 0:

                    # place the shape in the mask
                    apply_coords = shape > 0
                    mask[th:th + sh, tw:tw + sw][apply_coords] = 1.0

                    if avoid_mask is not None:
                        avoid_mask[th:th + sh, tw:tw + sw][apply_coords] = 1.0
                    else:
                        avoid_mask = mask.clone()

                    return mask, avoid_mask
                
                if i == n_try - 1:
                    #the last try, check if the shape overlap the, in that case
                    # the shape overlap other classes, so don't place it.
                    if avoid_mask is None or torch.sum(
                        torch.bitwise_and( # return the overlap between the avoid_mask (without the current mask) and the shape that should be placed
                            torch.bitwise_and( # return the avoid_mask with the current mask subtracted
                                avoid_mask[th:th + sh, tw:tw + sw].bool(),
                                torch.bitwise_not(mask[th:th + sh, tw:tw + sw].bool())
                            ),
                            shape.bool()
                        )
                    ) == 0:
                        mask[th:th + sh, tw:tw + sw][shape > 0] = 1.0

                        if avoid_mask is not None:
                            avoid_mask[th:th + sh, tw:tw + sw][shape > 0] = 1.0

                    if avoid_mask is None:
                        avoid_mask = mask.clone()

                    return mask, avoid_mask


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

    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    for sample in shape_dataloader:

        # convert a torch tensor to a numpy array
        sample = sample.numpy().astype(np.uint8)
        cv2.imshow("mask", sample*255)
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

    masks_dataset = JsonLineMaskObjectDataset(
        "/home/max/thesis/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/object_datasets/object_dataset_0",
        binary = True,
        augmentor=augmentor
    )

    shape_dataloader = ShapeObjectDataloader(
        input_dataset = masks_dataset,
        sample_shapes=(1, 3)
    )

    shape_dataloader.on_worker_init()

    #timeit(shape_dataloader)
    visualize(shape_dataloader)



