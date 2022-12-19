from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionMasksDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        """Multi resolution dataset allow to load images in lmdb format

        Args:
            path (str): path to the lmdb dataset
            transform (torchvision.transforms): transformation to apply to the images
            resolution (int, optional): resolution of the images. Defaults to 256.
            mask (bool, optional): if True, the dataset enter in mask mode and return an image single channel. Defaults to False.

        Raises:
            IOError: _description_


        """

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img.contiguous()
