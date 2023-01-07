import numpy as np
import torch
import random

import logging

from typing import Tuple, Union, Dict, List

from COIGAN.training.data.datasets_loaders.jsonl_object_dataset import JsonLineMaskObjectdataset

logger = logging.getLogger(__name__)


class ShapeObjectDataloader:

    """
        Object that prepare the samples for the shape gan generator.

        This object is used to load different numbers of shapes of the same type and mout
        them together in a single image.
    """

    def __init__(
        self,
        input_dataset: JsonLineMaskObjectdataset,
        sample_shapes: List[int] = [1, 3],
        tile_size: Union[int, List[int], Tuple[int]] = 256,
        strategy: str = "random",
        out_channels: int = 1,
        seed: int = 42,
    ):

        """
            Init method of the shape object dataloader.

            Args:
                input_dataset: dataset object that contains the input data, and return the mask as normalized torch.Tensor.

                sample_shapes (Union[int, List[int], Tuple[int]]): number of shapes to mount in the image.
                    - int: the same number of shapes is used for each sample.
                    - List[int, int]: the number of shapes is sampled from a uniform distribution between the two values.

                tile_size: size of the tile where the shapes are mounted.

                strategy: strategy to use to mount the shapes in the image.
                    - "random": indipendent from the idx passed to the __getitem__ method, the shapes are retrived from the dataset randomly,
                                with a random position.
                    - "single": the shapes are retrived from the dataset correspondignly to the idx passed to the __getitem__ method.
                
                out_channels: number of channels of the output mask.
                seed: seed to use for the random number generator.

        """

        self.input_dataset = input_dataset
        # enable the train mode from the input dataset if it is disabled
        self.input_dataset.train = True

        if isinstance(sample_shapes, (list, tuple)):
            self.sample_shapes = sample_shapes
        elif isinstance(sample_shapes, int):
            self.sample_shapes = [0, sample_shapes]
        else:
            raise ValueError("sample_shapes must be a list or a tuple of 2 int values, or at least an int value.")

        # check the elements passed as num_shapes are correctly defined
        if len(self.sample_shapes) != 2 or \
            not isinstance(self.sample_shapes[0], int) or \
            not isinstance(self.sample_shapes[1], int) or \
            self.sample_shapes[0] > self.sample_shapes[1] or \
            self.sample_shapes[0] < 0 or \
            self.sample_shapes[1] < 0:
            raise ValueError("sample_shapes must be a list or a tuple of 2 int values, that define a non-null interval of positive values.")

        self.sample_shapes = list(range(self.sample_shapes[0], self.sample_shapes[1] + 1))

        self.tile_size = tile_size if isinstance(tile_size, (list, tuple)) else (tile_size, tile_size)

        if strategy in ["random", "single"]:
            self.strategy = strategy
        else:
            raise ValueError("strategy must be 'random' or 'single'.")

        # set the number of channels 
        self.out_channels = out_channels

        # random index variables
        # variables used to retrive shapes from the dataset with the random strategy
        random.seed(seed)
        self.regenerate_random_idxs()


    def on_worker_init(self):
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
            random.shuffle(self.random_idxs)
    

    def generate_random_sample(self):
        """
            Generate a random sample of the dataset.
        """
        # generate the tile
        tile = torch.zeros(self.tile_size, dtype=torch.float32)

        # sample the number of shapes to place in the image
        num_shapes = random.choice(self.sample_shapes)
        
        # extract from random_idxs num_shapes random indexes, removing it from the list
        if len(self.random_idxs) < num_shapes:
            self.regenerate_random_idxs()
        idxs = [self.random_idxs.pop() for _ in range(num_shapes)]

        # retrive shapes from the dataset
        shapes = self.input_dataset[idxs]

        # mount the shapes in the tile
        for i in range(num_shapes):
            tile = self.apply_shape(tile, shapes[i].squeeze(0))
        
        return tile


    def generate_simple_sample(self, idx):
        """
            Generate a sample of the dataset using the idx passed to the __getitem__ method.
        """
        pass


    def __getitem__(self, idx):
            
        """
            Get the sample at the index idx.
        """

        if self.strategy == "random":
            mask = self.generate_random_sample()

        elif self.strategy == "single":
            mask = self.generate_simple_sample(idx)

        if self.out_channels > 1:
            mask = torch.stack([mask] * self.out_channels, dim=0)

        return mask
    

    def __iter__(self):
        """
            Iterator of the dataset.
        """

        for i in range(len(self)):
            yield self.__getitem__(i)


    def __len__(self):
        return len(self.input_dataset)


    @staticmethod
    def apply_shape(mask: torch.Tensor, shape: torch.Tensor):
        """
            Apply the shape to the mask.

            this method try to apply a new shape on an existing mask,
            without overlapping the shapes, and try to reduce the amount of shape placed outside the mask.

            Args:
                mask: mask where the shape is applied.
                shape: shape to apply to the mask.
        """
        
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

                if torch.sum(mask[ch:che, cw:cwe]) == 0:
                    # extract the shape that will be inside the mask
                    sch = 0 if ch == th else -th
                    sche = sh if che == th + sh else che - th                
                    scw = 0 if cw == tw else -tw
                    scwe = sw if cwe == tw + sw else cwe - tw

                    cshape = shape[sch:sche, scw:scwe]

                    # place the shape in the mask
                    mask[ch:che, cw:cwe] = cshape

                    return mask.contiguous()
                
                if i == n_try - 1:
                    # if the shape is not placed, and is the last try, place it anyway
                    # extract the shape that will be inside the mask
                    sch = 0 if ch == th else -th
                    sche = sh if che == th + sh else che - th                
                    scw = 0 if cw == tw else -tw
                    scwe = sw if cwe == tw + sw else cwe - tw
                    cshape = shape[sch:sche, scw:scwe]

                    # set the mask values to 1 where the shape is placed
                    mask[ch:che, cw:cwe][cshape > 0] = 1.0

                    return mask.contiguous()
        
        # if the shape fit in the mask, try to place it in a random position 
        else:
            for i in range(n_try):
                # generate a random position for the shape
                th = random.randint(0, mh - sh)
                tw = random.randint(0, mw - sw)

                if torch.sum(mask[th:th + sh, tw:tw + sw]) == 0:
                    # place the shape in the mask
                    mask[th:th + sh, tw:tw + sw] = shape

                    return mask.contiguous()
                
                if i == n_try - 1:
                    # if the shape is not placed, and is the last try, place it anyway
                    # set the mask values to 1 where the shape is placed
                    mask[th:th + sh, tw:tw + sw][shape > 0] = 1.0

                    return mask.contiguous()



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

    masks_dataset = JsonLineMaskObjectdataset(
        "/home/max/thesis/COIGAN-controllable-object-inpainting/datasets/severstal_steel_defect_dataset/test_1/object_datasets/object_dataset_0",
        binary = True,
        augmentor=augmentor
    )

    shape_dataloader = ShapeObjectDataloader(
        input_dataset = masks_dataset,
        sample_shapes=(1, 3),
        strategy="random"
    )

    shape_dataloader.on_worker_init()

    timeit(shape_dataloader)
    #visualize(shape_dataloader)



