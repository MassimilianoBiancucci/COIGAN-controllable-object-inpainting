import os
import csv
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import logging

LOGGER = logging.getLogger(__name__)

class SeverstalSteelDefectDataset(object):
    """
        This class allow to load the Severstal Steel Defect Dataset from its 
        original format.
    """

    n_classes = 4 # Number of classes presents in the dataset

    def __init__(
        self,
        dataset_path: str,
        mode: str = "all"
    ):
        """
            Initialize the Severstal Steel Defect Dataset reader

            Args:
                dataset_path (str): Path to the dataset
                mode (str): Mode of the dataset. Can be "all", "defected" or "none_defected"
                    - all (default): All the images
                    - defected: Only the images with defects
                    - none_defected: Only the images without defects
                
        """

        # Check if the dataset path exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("The dataset path does not exist")
        self.dataset_path = dataset_path
        
        if mode not in ["all", "defected", "none_defected"]:
            raise ValueError(f"The mode must be 'all', 'defected' or 'none_defected', got {mode}")
        self.mode = mode

        # Load the images
        self.train_images_path = os.path.join(self.dataset_path, "train_images")

        # load the segmentation masks in the csv file
        self.train_csv_path = os.path.join(self.dataset_path, "train.csv")
        self.train_csv = pd.read_csv(self.train_csv_path)

        # Group the images metadata by image name
        self.train_csv = self.train_csv.groupby("ImageId").agg(lambda x: list(x))
        self.defected_imgs = self.train_csv.index.values.tolist() # list of all defected images
        self.all_images = os.listdir(self.train_images_path) # list of all images
        self.none_defected_imgs = list(set(self.all_images) - set(self.defected_imgs)) # list of all non-defected images
        
        #changing the data returned based on the mode
        if self.mode == "all":
            self.used_images = self.all_images
            self.n_images = self.all_images.__len__()

        elif self.mode == "defected":
            self.used_images = self.defected_imgs
            self.n_images = self.defected_imgs.__len__()

        elif self.mode == "none_defected":
            self.used_images = self.none_defected_imgs
            self.n_images = self.none_defected_imgs.__len__()
        

    def get_metadata(self, index: int):
        """
            Get the metadata of one sample

            Args:
                index (int): _description_
            
            Returns:
                img_name (str): Name of the image
                classes (Union[list, None]): List of the classes of the defects if any, None otherwise
                encoded_masks (Union[list, None]): List of the encoded masks of the defects if any, None otherwise
        """

        if self.mode != "none_defected":
            img_name = self.used_images[index]

            try:
                pd_sample = self.train_csv.loc[img_name]
            except KeyError:
                return img_name, None, None

            classes = pd_sample["ClassId"]
            encoded_masks = pd_sample["EncodedPixels"]
        else:
            img_name = self.used_images[index]
            classes = None
            encoded_masks = None

        return img_name, classes, encoded_masks


    def __getitem__(self, index: int):
        """
            Get an item from the dataset

            Args:
                index (int): Index of the item to get
        
            Returns:
                image (np.array): Image
                mask (Union[np.array, None]): Masks of defects if any, None otherwise
        """

        img_name, classes, encoded_masks = self.get_metadata(index)

        # Get the image path
        image_path = os.path.join(self.train_images_path, img_name)
        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        if classes is not None:
            # Create the masks
            masks = np.zeros((h, w, self.n_classes), dtype=np.uint8)
            for i, (mask_i, _class) in enumerate(zip(encoded_masks, classes)):
                if mask_i is not np.nan:
                    masks[:,:,_class-1] += self.rle2mask(mask_i, h, w)
        else:
            masks = None
        
        return image, masks


    def __iter__(self):
        """
            Iterator over the dataset

            Returns:
                image (np.array): Image
                mask (Union[np.array, None]): Masks of defects if any, None otherwise
        """

        for i in range(self.n_images):
            yield self.__getitem__(i)


    def __len__(self):
        """
            Get the length of the dataset

            Returns:
                length (int): Length of the dataset
        """

        return self.n_images
    

    @staticmethod
    def rle2mask(rle, h, w):
        """
            Convert a run length encoding to a mask

            Args:
                rle (str): Run length encoding
                h (int): Height of the mask
                w (int): Width of the mask
            
            Returns:
                mask (np.array): Mask
        """

        mask = np.zeros(h*w, dtype=np.uint8)
        rle = rle.split()
        starts = np.asarray(rle[0::2], dtype=int)
        lengths = np.asarray(rle[1::2], dtype=int)
        
        for i in range(len(starts)):
            mask[starts[i]:(starts[i]+lengths[i])] = 1
        mask = mask.reshape((w, h)).T

        return mask


    def dataset_analysis_report(self, out_path: str):
        """
            Analyze the dataset

            this method return a set of statistics about the dataset:
            - a report with:
                - the number of images
                - the number of images with defects
                - the number of images without defects
                - the number of defects
                - the number of defects per class
            - a set of plots:
                - class distribution istogram
                - class area distribution istogram
                - defect and non defected istogram
        """

        report_file = os.path.join(out_path, "dataset_analysis_report.txt")
        report = open(report_file, "w")

        # Get the number of images
        report.write("Number of images: {}\n".format(self.n_images))
        report.write("Number of images with defects: {}\n".format(self.defected_imgs.__len__()))
        report.write("Number of images without defects: {}\n".format(self.none_defected_imgs.__len__()))
        



if __name__ == "__main__":

    # Test the dataset class
    out_path = "/home/ubuntu/hdd/COIGAN-controllable-object-inpainting/experiments_data/tests_severstal_steel_1"
    dataset_path = "/home/ubuntu/hdd/Datasets/severstal-steel-defect-detection"
    dataset = SeverstalSteelDefectDataset(dataset_path)

    print("Number of total csv images: {}".format(dataset.n_images))
    print("Number of total images {}".format(len(dataset.images_paths)))

    for i in range(10):
        img, masks = dataset[i]

        cv2.imwrite(os.path.join(out_path, f"image_{i}.png"), img)
        for j in range(dataset.n_classes):
            cv2.imwrite(os.path.join(out_path, f"mask_{i}_{j}.png"), masks[:,:,j]*255)

    print("Done")


