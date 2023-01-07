from torchvision.transforms import *
from torchvision.io import read_image
import torch.nn as nn

import numpy as np
import cv2

from COIGAN.training.data.augmentation.custom_transformations import *
from torchvision.transforms.transforms import Pad


class Augmentor:
    """
    Class that manage the augmentation process and the images standardization.
    For a multibatch training the augmentation need to respect an H, W standard size.

    """

    def __init__(
        self,
        transforms=None,
        only_imgs_transforms=None,
        device=None,
        mode=None,
        output_dims='keep',
    ):
        """
        Initialize the augmentor object.
        
        Args:
            transforms (transformation): The preset of transformations to use on images and masks.
            only_imgs_transforms (transformation): The preset of transformations to use on images only.
            device (str, optional): The device to use for the augmentation. Defaults to 'None'.
                - None: keep the tensors on the same device
                - 'cuda': use the cuda device
                - 'cpu': use the cpu device
            mode (str, optional): The mode of the augmentor. Defaults to 'pad'.
                - None: keep the tensors as they are after the augmentation.
                - 'pad': pad the images with zeros to have as size output_dims, if the images are bigger than resize it to the output dims.
                         The pad are applied equally in the 2 dimensions, keeping the image in the center.
                - 'rnd_pad': pad the images with zeros to have as size output_dims, if the images are bigger than resize it to the output dims.
                         The pad are applied randomly on the 4 edges, placing the image in a random position.
                - 'resize': resize the images to have at least one edge that match the output_dims and pad them with zeros for the other edges.
            output_dims (int or tuple[int], optional): The output dimension of the images. Defaults to 512.
                - int: the output dimension of the images are interpreted as (output_dims, output_dims) (h, w)
                - tuple[int]: the output dimension of the images are interpreted as (output_dims[0], output_dims[1]) (h, w)
                - string: 
                    - 'keep': keep the image dims as they are before the augmentation.
        
        NOTE: all the augmentation passed to transforms and only_imgs_transforms must support the None input,
            otherwise can't be garanted the correct behavior of the augmentor.

        """

        self.device = device

        self.transforms = transforms
        self.only_imgs_transforms = only_imgs_transforms

        self.mode = mode

        if isinstance(output_dims, int):
            self.output_dims = (output_dims, output_dims)
        elif isinstance(output_dims, tuple) and len(output_dims) == 2:
            self.output_dims = output_dims
        elif output_dims == 'keep':
            self.output_dims = None
        else:
            raise ValueError("output_dims must be an int or a tuple of two ints")



    def _pad(self, img, mask, rand=False):
        """Pad the image with zeros to have as size output_dims.

        Args:
            img (tensor): The image to pad.
            mask (tensor): The mask to pad.
            rand (bool, optional): If True, the padding is placed random on the 4 edges, otherwise is equaly distributed. Defaults to False.
        """
        # get the size of the image
        h, w = img.shape[-2:] if img is not None else mask.shape[-2:]

        if rand:

            # get the padding for the 4 edges
            h_distr = np.random.uniform()
            pad_top = int((self.output_dims[0] - h) * h_distr)
            pad_bot = int((self.output_dims[0] - h) * (1 - h_distr))

            w_distr = np.random.uniform()
            pad_left = int((self.output_dims[1] - w) * w_distr)
            pad_right = int((self.output_dims[1] - w) * (1 - w_distr))

            # fill the approssimation gap to reach the output dims
            d_h = self.output_dims[0] - (h + pad_top + pad_bot)
            d_w = self.output_dims[1] - (w + pad_left + pad_right)

            pad_dims = (pad_left, pad_top, pad_right + d_w, pad_bot + d_h)

        else:
            # get the padding size
            pad_h = int((self.output_dims[0] - h) / 2)
            pad_w = int((self.output_dims[1] - w) / 2)

            # fill the approssimation gap to reach the output_dims
            d_h = self.output_dims[0] - ((2 * pad_h) + h)
            d_w = self.output_dims[1] - ((2 * pad_w) + w)

            pad_dims = (pad_w, pad_h, pad_w + d_w, pad_h + d_h)

        # pad the image and the mask
        if any(pad > 0 for pad in pad_dims):
            img = Pad(pad_dims)(img) if img is not None else None
            mask = Pad(pad_dims)(mask) if mask is not None else None
 
        return img, mask

    def _resize(self, img, mask):
        """Resize the image to have at least one edge that match the output_dims and pad them with zeros for the other edges.
            The resize keep the aspect ratio and only the longest edge match the output_dims.

        Args:
            img (tensor): The image to resize, and to pad for a perfect out_dims match.
            mask (tensor): The mask to resize, and to pad for a perfect out_dims match.
        """
        # get the size of the image
        h, w = img.shape[-2:] if img is not None else mask.shape[-2:]

        # get the resize factor, from the bigger image dimension
        if h > w:
            resize_factor = self.output_dims[0] / h
        else:
            resize_factor = self.output_dims[1] / w

        new_h = int(h * resize_factor)
        new_w = int(w * resize_factor)

        if new_h > self.output_dims[0] or new_w > self.output_dims[1]:
            raise ValueError("The resize factor is too high")

        # resize the image and the mask
        img = Resize((new_h, new_w))(img) if img is not None else None
        mask = Resize((new_h, new_w))(mask) if mask is not None else None

        return img, mask

    def applay_pad(self, img, mask, rand=False):
        """Pad the image to fit the output_dims. if the image is bigger than the output_dims,
            resize it to fit in it keeping the aspect ratio.

        Args:
            img (tensor): The image to resize if too bigger and to pad.
            mask (tensor): The mask to resize if too bigger and to pad.
        """

        # get the size of the image
        h, w = img.shape[-2:] if img is not None else mask.shape[-2:]
        
        # check if a resize is needed
        if h > self.output_dims[0] or w > self.output_dims[1]:
            img, mask = self._resize(img, mask)

        # pad the image and the mask
        img, mask = self._pad(img, mask, rand)

        return img, mask

    def applay_resize(self, img, mask):
        """Resize the image to fit the output_dims for at least one edge keeping the aspect ratio,
            and pad them with zeros for the other edges, for a perfet match.

        Args:
            img (_type_): _description_
            mask (_type_): _description_
        """

        # resize the images
        img, mask = self._resize(img, mask)

        # pad the images
        img, mask = self._pad(img, mask)

        return img, mask

    def __call__(self, img = None, mask = None, out_dims=None):
        """
            Call method of the class.
            allow to apply the augmentations passed in the constructor to the image and the mask.
            allow to specify the output_dims of the image and the mask, if passed 
            override the output_dims passed in the constructor.

            Args:
                img (tensor): The image to transform. Defaults to None.
                mask (tensor): The mask to transform. Defaults to None.
                out_dims (tuple, optional): The output_dims of the image and the mask. Defaults to None. 
                                            If not None, override the output_dims passed in the constructor.
        """
        # change the output_dims if needed
        if out_dims is not None:
            if isinstance(out_dims, int):
                self.output_dims = (out_dims, out_dims)
            elif (isinstance(out_dims, tuple, list)) and len(out_dims) == 2:
                self.output_dims = out_dims

        # change the device of the image and mask
        if self.device != None:
            img = img.to(self.device) if img is not None else img
            mask = mask.to(self.device) if mask is not None else mask

        # apply the transformations on the image and mask
        if self.transforms != None:
            img, mask = self.transforms(img, mask)

        if self.only_imgs_transforms != None:
            img = self.only_imgs_transforms(img)

        if self.mode == "pad":
            img, mask = self.applay_pad(img, mask)

        elif self.mode == "rnd_pad":
            img, mask = self.applay_pad(img, mask, True)

        elif self.mode == "resize":
            img, mask = self.applay_resize(img, mask)
            

        return img, mask

    def __repr__(self):
        """Print the representation of the object.

        Returns:
            String: Return a string representation of the augmentations present in the object
        """
        return (
            "transforms: "
            + self.transforms.__repr__()
            + "\n"
            + "only images transforms: "
            + self.only_imgs_transforms.__repr__()
        )


if __name__ == "__main__":

    img_path = "/home/max/cloe/synthetic_datasets/dent_dataset_1/data/sample__00007.jpg"
    mask_path = (
        "/home/max/cloe/synthetic_datasets/dent_dataset_1/masks/sample__00007.png"
    )

    # img = Image.open(img_path)

    transform = T.Compose(
        [
            ToTensor(),
            # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            RandomCrop(224),
            # T.RandomHorizontalFlip(p=0.3),
            # T.RandomVerticalFlip(p=0.3),
            # T.RandomRotation(180),
            # T.RandomPerspective(p=0.3),
        ]
    )

    script_transforms = nn.Sequential(
        # RandomCrop(224),
        RandomHorizontalFlip(p=0.3),
        RandomVerticalFlip(p=0.3),
        RandomRotation(180),
        RandomPerspective(p=0.3),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    )

    mask_transforms = dChoice(
        [
            dCrop((224, 512)),
            dResize((300, 1000), max_size=1500),
            dRotations(120),
            dAffine(degrees=0.0, translate=0.3, scale=(0.6, 1.2), shear=25.0),
        ],
        n_t=[0, 3],
    )

    mask_test = dRotations(120)

    imgs_transforms = sChoice(
        [
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            RandomErasing(p=0.3, scale=(0.02, 0.13), ratio=(0.3, 3.3), value="random"),
            GaussianBlur(kernel_size=5),
            RandomPosterize(p=0.3, bits=4),
            RandomAutocontrast(p=0.3),
            RandomSolarize(p=0.3, threshold=128),
            RandomAdjustSharpness(p=0.3, sharpness_factor=2.0),
            RandomEqualize(p=0.3),
        ],
        n_t=[0, 4],
    )

    from augmentation_presets import mask_preset_1, imgs_preset_1

    """     
    augmentor = Augmentor(
        transforms=mask_preset_1,
        only_imgs_transforms=imgs_preset_1,
        device="cuda",
        mode="rnd_pad",
        output_dims=(512, 512)
    ) 
    """

    augmentor = Augmentor(
        transforms=mask_preset_1,
        only_imgs_transforms=imgs_preset_1,
        mode = "resize",
        output_dims='keep'
    )


    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    while True:

        # img_a = img.copy()
        # img_a = transform(img_a)

        # img_a = torch.clone(img)

        ### with Benchmark("augmentation"): # start benchmark

        img_a = read_image(img_path)
        mask_a = read_image(mask_path)

        """
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        img_a = img_a.to(device)
        mask_a = mask_a.to(device) 

        img_a = imgs_transforms(img_a)
        img_a, mask_a = mask_transforms(img_a, mask_a)
        """

        print("----------------")
        img_a, mask_a = augmentor(img_a, mask_a)

        print(f"img_a shape: {img_a.shape}")
        print(f"mask_a shape: {mask_a.shape}")

        img_a = img_a.to("cpu")
        mask_a = mask_a.to("cpu")

        ### end benchmark

        # show torch tensor on cv2
        img_a = img_a.numpy()
        mask_a = mask_a.numpy()
        img_a = np.transpose(img_a, (1, 2, 0))
        mask_a = np.transpose(mask_a, (1, 2, 0))

        img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)
        cv2.imshow("img", img_a)
        cv2.imshow("mask", mask_a)

        key = cv2.waitKey(0)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    #print(Benchmark())
