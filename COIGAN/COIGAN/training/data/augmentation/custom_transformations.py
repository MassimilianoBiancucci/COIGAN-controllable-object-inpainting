import torch
import torch.nn as nn
from numpy import number
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

ct_dbg = False

"""IMAGES AND MASKS TRANSFORMATIONS"""


class dHorizontalFlip(nn.Module):
    """Flip the image and mask horizontally with a probability of 0.5."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, mask):
        if torch.rand(1) < self.p:
            img = F.hflip(img) if img is not None else None
            mask = F.hflip(mask) if mask is not None else None
        return img, mask


class dVerticalFlip(nn.Module):
    """Flip the image and mask vertically with a probability of 0.5."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, mask):
        if torch.rand(1) < self.p:
            img = F.vflip(img) if img is not None else None
            mask = F.vflip(mask) if mask is not None else None
        return img, mask


class dCrop(nn.Module):
    def __init__(self, crop_size):
        """Custom crop augmentation for the masks and images.

        Args:
            crop_size (tuple[int]): Tupla di int che specifica l'intervallo di dimensioni del crop.
        """

        super(dCrop, self).__init__()
        self.crop_size = crop_size
        self.name = "dCrop"

    def forward(self, x, mask):

        if ct_dbg:
            print(self.name)

        size = torch.randint(self.crop_size[0], self.crop_size[1], size=(1,)).item()

        # check that che crop_size il less than the torch tensor size
        if size < x.shape[1] or size < x.shape[2]:
            i, j, h, w = T.RandomCrop.get_params(x, (size, size))

            x = F.crop(x, i, j, h, w) if x is not None else None
            mask = F.crop(mask, i, j, h, w) if mask is not None else None

        return x, mask

    def __repr__(self):
        return self.name + "(" + str(self.crop_size) + ")"


class dResize(nn.Module):
    def __init__(self, size, max_size=2000):
        """Function that apply a random resize to the image and the mask.

        Args:
            size (tuple[int]): (min_edge, max_edge) tuple that contain the interval of seizes that the image should have
            max_size (int, optional): the maximum size of the biggest edge for a resized image. Defaults to 2000.
        """
        super(dResize, self).__init__()
        self.size = size
        self.max_size = max_size
        self.name = "dResize"

    def forward(self, x, mask):

        if ct_dbg:
            print(self.name)

        # get randomly the size in the interval
        size = torch.randint(self.size[0], self.size[1], size=(1,)).item()

        x = F.resize(
            x, size, interpolation=InterpolationMode.BILINEAR, max_size=self.max_size
        ) if x is not None else None
        mask = F.resize(
            mask, size, interpolation=InterpolationMode.NEAREST, max_size=self.max_size
        ) if x is not None else None

        return x, mask

    def __repr__(self):
        return self.name + "(" + str(self.size) + ")"


class dCropUpscale(nn.Module):
    def __init__(self, crop_size, up_size):
        """Function that crop and than upscale a certain box in the image and the mask.

        Args:
            crop_size (tuple[int]): Tupla di int che specifica l'intervallo di dimensioni del crop
            up_size (int): Maximum size of the biggest edge for a resized image.

        """
        super(dCropUpscale, self).__init__()
        self.crop_size = crop_size
        self.up_size = up_size
        self.name = "dCropUpscale"

    def forward(self, x, mask):

        if ct_dbg:
            print(self.name)

        h, w = x.shape[-2:]

        box_h_max = min(h, self.crop_size[1])
        box_w_max = min(w, self.crop_size[1])

        # get randomly the size in the interval
        size_h = torch.randint(self.crop_size[0], box_h_max, size=(1,)).item()
        size_w = torch.randint(self.crop_size[0], box_w_max, size=(1,)).item()

        # check that che crop_size il less than the torch tensor size
        i, j, h, w = T.RandomCrop.get_params(x, (size_h, size_w))

        x = F.crop(x, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        min_crop_edge = min(h, w)
        max_crop_edge = max(h, w)

        # compute the scale factor for the upscaling to up_size for the biggest edge
        scale_factor = self.up_size / max_crop_edge

        # compute the new up size for getting the right size for the upscaling
        # Note that the resize function scale the image keeping the aspect ratio
        # and bring the smallest edge to the size passed so the targt value.
        min_edge_up_size = int(min_crop_edge * scale_factor)

        # upscaling the image and the mask
        # get randomly the size in the interval
        size = torch.randint(min_crop_edge, min_edge_up_size, size=(1,)).item()

        x = F.resize(
            x, size, interpolation=InterpolationMode.BILINEAR, max_size=self.up_size
        ) if x is not None else None
        mask = F.resize(
            mask, size, interpolation=InterpolationMode.NEAREST, max_size=self.up_size
        ) if mask is not None else None

        return x, mask


class dRotations(nn.Module):
    def __init__(self, degrees, expand=False):
        """Rotate image and mask

        Args:
            degrees (Tuple[float or int]): single value int or float, or tuple of int or float,
                                if single values the interval of rotation is [-degrees, degrees].

            expand (bool, optional): Expand the image for containing the entire rotated image. Defaults to False.
        """

        super(dRotations, self).__init__()

        if isinstance(degrees, int) or isinstance(degrees, float):
            degrees = (-degrees, degrees)
        self.degrees = degrees
        self.expand = expand
        self.name = "dRotations"

    def forward(self, x, mask):
        if ct_dbg:
            print(self.name)

        degrees = T.RandomRotation.get_params(self.degrees)
        x = F.rotate(
            x, degrees, interpolation=InterpolationMode.BILINEAR, expand=self.expand
        ) if x is not None else None
        mask = F.rotate(
            mask, degrees, interpolation=InterpolationMode.NEAREST, expand=self.expand
        ) if mask is not None else None
        return x, mask

    def __repr__(self):
        return self.name + "(" + str(self.degrees) + ")"


class dAffine(nn.Module):
    def __init__(self, degrees, translate, scale, shear):
        super(dAffine, self).__init__()

        self.degrees = (-degrees, degrees) if isinstance(degrees, float) else degrees
        self.translate = (
            (translate, translate) if isinstance(translate, float) else translate
        )
        self.scale = (1 - (scale - 1), scale) if isinstance(scale, float) else scale
        self.shear = (
            (-shear, shear, -shear, shear) if isinstance(shear, float) else shear
        )
        self.name = "dAffine"

    def forward(self, x, mask):

        if ct_dbg:
            print(self.name)

        angle, translations, scale, shear = T.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, x.shape if x is not None else mask.shape
        )

        x = F.affine(
            x,
            angle,
            translations,
            scale,
            shear,
            interpolation=InterpolationMode.BILINEAR,
        ) if x is not None else None

        mask = F.affine(
            mask,
            angle,
            translations,
            scale,
            shear,
            interpolation=InterpolationMode.NEAREST,
        ) if mask is not None else None

        return x, mask

    def __repr__(self):
        return self.name + "(" + str(self.degrees) + ")"


class dApply(T.RandomApply):
    """Wrapper for the RandomApply transform, that enable the usage of
    the same trasformation even in the image as in the mask.
    """

    def forward(self, img, mask):
        if self.p < torch.rand(1):
            return img, mask
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class dChoice:
    """Random choice Transform that permit to use Transformations in parallel on
    images and masks.
    This object permit to chose the maximum and minium number of transformations
    that can be applied to the image and mask.
    """

    def __init__(self, transforms, n_t=[0, 2]):
        """
        Args:
            transforms (sequence): List of transformations that supports image and mask
            n_t (tuple, optional): interval of transformations that should be applied. Defaults to (0, 2).
        """
        self.transforms = transforms
        n_t[1] = min(n_t[1], len(transforms))
        self.n_t = n_t

    def _choice_transforms(self):
        # chose the number of transformations
        n_idxs = torch.randint(self.n_t[0], self.n_t[1], size=(1,)).item()
        # define which transformations should be applied
        idxs = torch.arange(0, len(self.transforms))
        idxs = torch.randperm(len(idxs))[:n_idxs].tolist()

        return idxs

    def __call__(self, img, mask):
        for idx in self._choice_transforms():
            img, mask = self.transforms[idx](img, mask)
        return img, mask

    def __repr__(self):
        str_transforms = ""
        for transform in self.transforms:
            str_transforms += transform + "\n"
        return "dChoice(\n" + str_transforms + ")"


class dCombine(nn.Module):
    def __init__(self, transforms):
        """
        Args:
            transforms (sequence): List of transformations that supports image and mask
        """
        super(dCombine, self).__init__()
        self.transforms = transforms

    def forward(self, img, mask):
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask

    def __repr__(self):
        str_transforms = ""
        for transform in self.transforms:
            str_transforms += transform + "\n"
        return "dCombine(\n" + str_transforms + ")"


"""ONLY IMAGES TRANSFORMATIONS"""


class sChoice(dChoice):
    """Extension of dChoice for transformations on the imges without masks."""

    def __call__(self, img):
        if img is not None:
            for idx in self._choice_transforms():
                img = self.transforms[idx](img)
        return img


class sNoOp(nn.Module):
    def __init__(self):
        super(sNoOp, self).__init__()

    def forward(self, x):
        return x

    def __repr__(self):
        return "noOp"