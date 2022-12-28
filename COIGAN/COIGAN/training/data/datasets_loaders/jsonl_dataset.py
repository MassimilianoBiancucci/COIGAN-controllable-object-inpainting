import os
import json, pbjson
import logging
import cv2
import numpy as np

from tqdm import tqdm
from typing import Tuple, Union, Dict

LOGGER = logging.getLogger(__name__)


class JsonLineDatasetBase:
    
    def __init__(
        self,
        metadata_file_path: str,
        index_file_path: str,
        binary: bool = False
    ):
        """
        Object that load a dataset from a jsonl file
        and allow a random access to the samples using an index file.
        This class does not have any logic to load masks or images, so the json structure is not defined.

        Args:
            metadata_path (str): metadata file containing the masks of each image
            index_path (str): index file containing the start position of each sample in the metadata file

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
        """

        self.binary = binary

        if not os.path.isfile(metadata_file_path):
            raise RuntimeError(f"Metadata file {metadata_file_path} does not exist")
        if not os.path.isfile(index_file_path):
            raise RuntimeError(f"Index file {index_file_path} does not exist")

        # read the index file and fill the ids list
        self.index_file = index_file_path
        self.metadata_file_path = metadata_file_path

        if self.binary:
            with open(self.index_file, "rb") as f:
                #self.ids = [int.from_bytes(i.strip(), byteorder="big", signed=False) for i in f.readlines()]
                self.ids = []
                while True:
                    b = f.read(4)
                    if b == b'':
                        break
                    self.ids.append(int.from_bytes(b, byteorder="little"))
        else:
            with open(self.index_file, "r") as f:
                self.ids = [int(i) for i in f.readlines()]

        if not self.ids:
            raise RuntimeError(f"index file {self.index_file} is empty")

        LOGGER.info(f"Creating dataset with {len(self.ids)} examples")


    def on_worker_init(self, *args, **kwargs):
        """
        Method called on each dataloader worker when initialized.
        It open the jsonl file and seek to the start position of the first sample.
        It's needed because the file must be left open, and opening it before 
        the dataloader worker is created cause an error.
        NOTE: it should be passed to the dataloader as worker_init_fn parameter.
        """
        self.jsonl = open(self.metadata_file_path, "br" if self.binary else "r")


    def _get_meta(self, idx: int) -> dict:
        """
        Method that return a sample with the given index
        from the dataset.

        Args:
            idx (int): index of the sample to return

        Returns:
            dict: json as dict of the sample
        """
        index = self.ids[idx]
        self.jsonl.seek(index)
        line = self.jsonl.readline()
        metadata = json.loads(line)

        return metadata


    def _get_bin_meta(self, idx: int) -> dict:
        """
        Method that return a sample with the given index
        from the dataset.

        Args:
            idx (int): index of the sample to return

        Returns:
            dict: json as dict of the sample
        """
        index = self.ids[idx]
        self.jsonl.seek(index)
        line = self.jsonl.readline()
        metadata = pbjson.loads(line)

        return metadata


    def _get_metadata(self, idx: int) -> dict:
        """
        Method that return a sample with the given index
        from the dataset.

        Args:
            idx (int): index of the sample to return

        Returns:
            dict: json as dict of the sample
        """
        if self.binary:
            return self._get_bin_meta(idx)
        else:
            return self._get_meta(idx)


    def __getitem__(self, idx: int):
        """
        Return the sample with the given index
        """
        if isinstance(idx, slice):
            return [self._get_metadata(i) for i in range(*idx.indices(len(self)))]
        else:
            return self._get_metadata(idx)


    def __len__(self):
        """
        Return the number of samples in the dataset
        """
        return len(self.ids)


    def close(self):
        """
        Close the jsonl file
        """
        self.jsonl.close()


    def map_field_classes(self, excluded_fields: list[str] = ["img"]):
        """
        Map the classes of a field to a list of classes.
        The classes are mapped to the index of the class in the list.
        If a class is not in the list, it is mapped to -1.

        Args:
            excluded_fields (list[str], optional): list of fields to exclude from the mapping. Defaults to ["img"].

        Returns:
            dict: dictionary with the mapping of each field
                es: 
                {
                    "field1": [
                        "class1",
                        "class2",
                        "class3"
                    ],
                    "field2": [
                        ...
                    ]
                }
        """

        LOGGER.info("Mapping fields and classes...")

        fields_map = {}
        for i in tqdm(range(len(self.ids))):
            metadata = self._get_metadata(i)
            for field in metadata:
                if field not in excluded_fields:
                    if field not in fields_map:
                        fields_map[field] = set()
                    for obj in metadata[field]:
                        fields_map[field].add(obj["label"])

        for field in fields_map:
            fields_map[field] = sorted(list(fields_map[field]))

        return fields_map



class JsonLineDatasetMasksOnly(JsonLineDatasetBase):

    def __init__(
        self,
        metadata_file_path: str,
        index_file_path: str,
        masks_fields: list[str],
        classes: Union[list[str],None] = None,
        size: Union[Tuple[int, int], int] = (256, 256),
        points_normalized: bool = False,
        binary: bool = False
    ):
        """
        Extension of the JsonLineDatasetBase class, for datasets with only masks and no images.

        This object require a parameter named masks_fields, 
        that is a list of the fields containing the polygons in the json dataset.

        Each field will be returned as a separate list of masks, 
        with all the polygons with a subfield label returned as a single mask.

        the expected Json structure for each sub json line is:
        {
            "field1": [
                {
                    "label": "class1",
                    "points": [
                        [
                            [x1, y1],
                            [x2, y2],
                            ...
                        ],
                        [
                            [x1, y1],
                            [x2, y2],
                            ...
                        ],
                        ...
                    ]
                },
                {
                    "label": "class2",
                    "points": [
                        ...
                    ]
                },
                ...
            ],
            "field2": [
                ...
            ],
            ...
        }

        Args:
            metadata_path (str): metadata file containing the masks of each image
            index_path (str): index file containing the start position of each sample in the metadata file
            masks_fields (list[str]): list of the fields containing the masks, used to group the masks in the output dict, and to read the polygons.
            classes (Union[list[str],None], optional): list of the classes to load. Defaults to None. if None, all the classes will be loaded.
            size (Union[Tuple[int, int], int], optional): size of the output masks. Defaults to (256, 256).
            points_normalized (bool, optional): if the points are normalized. Defaults to False. if the are in the range [0, 1], the points_normalized parameter must be set to True.

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
        """

        super(JsonLineDatasetMasksOnly, self).__init__(
            metadata_file_path, 
            index_file_path,
            binary
        )
        
        self.masks_fields = masks_fields
        self.classes = classes

        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        self.points_normalized = points_normalized

    
    def load_masks(
        self,
        polygon_field: str,
        metadata: dict,
        shape: tuple,
        normalized_points: bool = False,
        mask_value: int = 1,
    ) -> "dict[str, np.ndarray]":
        """
        Load all the masks in a given field, groupping all the masks with the same label
        in a single mask.

        Args:
            polygon_field (str): the name of the field containing the polygons
            classes (List[str]): the list of classes to load
            metadata (dict): the metadata of the sample
            shape (tuple): the shape of the output mask
            normalized_points (bool, optional): if the points are normalized. Defaults to False.

        Returns:
            dict[str, np.ndarray]: a dict mapping the class to the mask
        """

        masks = {}

        for polygons in metadata[polygon_field]:
            
            # if the class is in the list of classes to load or if the list is None
            if polygons["label"] in self.classes or self.classes is None:

                mask = masks.get(polygons["label"], np.zeros(shape, dtype=np.uint8))

                for sub_poly in polygons["points"]:
                    points = np.asarray(sub_poly)
                    points = points.squeeze()

                    # TODO workaround for bugs in some datasets
                    # polygons with less than 3 points can bring to other bugs
                    if points.shape[0] < 3:
                        continue

                    if normalized_points:
                        points[:, 0] = points[:, 0] * shape[1]
                        points[:, 1] = points[:, 1] * shape[0]
                        cv2.fillPoly(mask, [points.astype(np.int32)], mask_value)
                    else:
                        cv2.fillPoly(mask, [points], mask_value)

                masks[polygons["label"]] = mask

        return masks


    def _get_masks(self, idx: int, ret_meta: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Method that return the masks of the sample at the given index
        and eventually the metadata of the sample, if ret_meta is True.

        Args:
            idx (int): the index of the sample
            ret_meta (bool, optional): Flag that active the return of the metadata. Defaults to False.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: the masks of the sample, as dict with as keys the passed list of fields and as values other dicts with as keys the classes and as values the masks as numpy arrays.
            Tuple[dict, dict]: the masks of the sample in the same format as above and the raw metadata of the sample.
        """
        metadata = self._get_metadata(idx)

        # load masks from the sample
        masks = {
            field: self.load_masks(field, metadata, self.size, self.points_normalized)
            for field in self.masks_fields
        }
        
        return (masks, metadata) if ret_meta else masks


    def __getitem__(self, idx: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Method that return the masks of the sample at the given index.

        Args:
            idx (int): the index of the sample

        Returns:
            Dict[str, Dict[str, np.ndarray]]: the masks of the sample, as dict with as keys the passed list of fields and as values other dicts with as keys the classes and as values the masks as numpy arrays.
        """
        return self._get_masks(idx)


class JsonLineDatasetSeparatedMasksOnly(JsonLineDatasetMasksOnly):
    
    """
    Variant of the JsonLineDatasetMasksOnly that returns the masks in a different format.
    In this case each polygon is returned as a separate mask, instead of being grouped into a single mask.
    Instead of a single mask, under a class key there will be a list of masks.

    NOTE: the __getitem__ method returns a dict[str, dict[str, list[np.ndarray]] !!
    """

    def load_masks(
        self,
        polygon_field: str,
        metadata: dict,
        shape: tuple,
        normalized_points: bool = False,
        mask_value: int = 1,
    ) -> "dict[str, list[np.ndarray]]":
        """
        Load all the masks in a given field, groupping all the masks with the same label
        in a single mask.

        Args:
            polygon_field (str): the name of the field containing the polygons
            classes (List[str]): the list of classes to load
            metadata (dict): the metadata of the sample
            shape (tuple): the shape of the output mask
            normalized_points (bool, optional): if the points are normalized. Defaults to False.

        Returns:
            dict[str, list[np.ndarray]]: a dict mapping the class to the mask
        """

        masks = {}

        for polygon in metadata[polygon_field]:
            
            # if the class is in the list of classes to load or if the list is None
            if self.classes is None or polygon["label"] in self.classes:

                mask = np.zeros(shape, dtype=np.uint8)

                for sub_poly in polygon["points"]:
                    points = np.asarray(sub_poly)
                    points = points.squeeze()

                    # TODO workaround for bugs in some datasets
                    # polygons with less than 3 points can bring to other bugs
                    if points.shape[0] < 3:
                        continue

                    if normalized_points:
                        points[:, 0] = points[:, 0] * shape[1]
                        points[:, 1] = points[:, 1] * shape[0]
                        cv2.fillPoly(mask, [points.astype(np.int32)], mask_value)
                    else:
                        cv2.fillPoly(mask, [points], mask_value)

                masks[polygon["label"]] = masks.get(polygon["label"], []) + [mask]

        return masks


class JsonLineDataset(JsonLineDatasetMasksOnly):

    def __init__(
        self,
        image_folder_path: str,
        metadata_file_path: str,
        index_file_path: str,
        masks_fields: list[str],
        classes: Union[list[str],None] = None,
        size: Union[Tuple[int, int], int] = (256, 256),
        points_normalized: bool = False,
        binary: bool = False
    ):
        """
        Extension of the JsonLineDatasetMasksOnly class, 
        for datasets with images and associated masks.

        This object require a parameter named masks_fields, 
        that is a list of the fields containing the masks.

        Each field will be returned as a separate list of masks, 
        with all the polygons with a subfield label returned as a single mask.

        the expected Json structure for each sub json line is:
        {
            "file_name": "image_example_0001.jpg",
            "field1": [
                {
                    "label": "class1",
                    "points": [
                        [
                            [x1, y1],
                            [x2, y2],
                            ...
                        ],
                        [
                            [x1, y1],
                            [x2, y2],
                            ...
                        ],
                        ...
                    ]
                },
                {
                    "label": "class2",
                    "points": [
                        ...
                    ]
                },
                ...
            ],
            "field2": [
                ...
            ],
            ...
        }

        Args:
            image_folder_path (str): path to the folder containing the images
            metadata_path (str): metadata file containing the masks of each image
            index_path (str): index file containing the start position of each sample in the metadata file
            masks_fields (list[str]): list of the fields containing the masks, used to group the masks in the output dict, and to read the polygons.
            classes (Union[list[str],None], optional): list of the classes to load. Defaults to None. if None, all the classes will be loaded.
            size (Union[Tuple[int, int], int], optional): size of the output masks. Defaults to (256, 256).
            points_normalized (bool, optional): if the points are normalized. Defaults to False. if the are in the range [0, 1], the points_normalized parameter must be set to True.

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
        """

        super(JsonLineDataset, self).__init__(
            metadata_file_path, 
            index_file_path,
            masks_fields,
            classes,
            size,
            points_normalized,
            binary
        )

        self.image_folder_path = image_folder_path

        # check if the image folder exists
        if not os.path.isdir(self.image_folder_path):
            raise RuntimeError(f"Image folder path {self.image_folder_path} does not exist")


    def _get_image_and_masks(self, idx: int, ret_meta: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Method that return the image, the masks of the sample at the given index
        and eventually the metadata of the sample, if ret_meta is True.

        Args:
            idx (int): the index of the sample
            ret_meta (bool, optional): Flag that active the return of the metadata. Defaults to False.

        Returns:
            Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]: the image and the masks of the sample, 
                as dict with as keys the passed list of fields and as values other dicts with as keys 
                the classes and as values the masks as numpy arrays.

            Tuple[np.ndarray, dict, dict]: the image, the masks of the sample in the same format as 
                above and the raw metadata of the sample.
        """

        masks, metadata = self._get_masks(idx, ret_meta=True)

        # load the image
        img = cv2.imread(os.path.join(self.image_folder_path, metadata["file_name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            raise RuntimeError(f"Image {metadata['file_name']} not found")
        
        return img, masks, metadata if ret_meta else img, masks


    def __getitem__(self, idx: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Method that return the image and the masks of the sample at the given index.

        Args:
            idx (int): the index of the sample

        Returns:
            Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]: the image and the masks of the sample, 
                as dict with as keys the passed list of fields and as values other dicts with as keys 
                the classes and as values the masks as numpy arrays.
        """
        if isinstance(idx, slice):
            return [self._get_image_and_masks(i) for i in range(*idx.indices(len(self)))]
        else:
            return self._get_image_and_masks(idx)


if __name__ == "__main__":

    ######################################
    # Test JsonLineDatasetBase class
    
    ######################################
    # Test JsonLineDatasetMasksOnly class

    ######################################
    # Test JsonLineDataset class
    pass