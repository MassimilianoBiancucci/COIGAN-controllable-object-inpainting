import os
import numpy as np
import cv2

import logging

from COIGAN.training.data.image_evaluators.image_evaluator_base import ImageEvaluatorBase

LOGGER = logging.getLogger(__name__)

class SeverstalBaseEvaluator(ImageEvaluatorBase):

    """
        The severstal base evaluator is used to evaluate if an image
        is valid as base for the defects inpainting process.
        Many images in the severstal dataset are partially or totally
        black, and that areas can't be used as base for the inpainting.

        This evaluator check if the image is black over a certain percentage.
    """

    def __init__(
        self,
        black_threshold: int = 10,
        black_area_max_coverage: float = 0.1,
        debug: bool = False
    ):
        """
            Init method for the SeverstalBaseEvaluator class.

            Args:
                black_threshold (int): threshold value for the black color.
                black_area_max_coverage (float): maximum percentage of the image that can be black.
        """

        self.black_threshold = black_threshold
        self.black_area_max_coverage = black_area_max_coverage
        self.debug = debug
    

    def __call__(self, image: np.ndarray) -> bool:

        """
            Call method for the SeverstalBaseEvaluator class.
            This method evaluate how much area of the image is below the
            black threshold and compare it with the maximum coverage allowed.
            if the percentage of black area is greater than the maximum coverage
            allowed, the image is considered invalid.

            Args:
                image (np.ndarray): image in numpy array format that will be evaluated.

            Returns:
                bool: result of the evaluation. True if the image is valid, False otherwise.
        """

        # convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.debug:
            thres_img = cv2.threshold(gray_image, self.black_threshold, 1, cv2.THRESH_BINARY_INV)[1]

        # get the number of pixels that are black
        black_pixels = np.sum(gray_image < self.black_threshold)

        # get the total number of pixels
        total_pixels = gray_image.shape[0] * gray_image.shape[1]

        # get the percentage of black pixels
        black_percentage = black_pixels / total_pixels

        # check if the percentage of black pixels is greater than the maximum coverage allowed
        if black_percentage > self.black_area_max_coverage:
            return False
        
        return True