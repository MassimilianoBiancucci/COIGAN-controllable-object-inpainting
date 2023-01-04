import numpy as np

import logging

LOGGER = logging.getLogger(__name__)

class ImageEvaluatorBase:

    """
        Base class for image evaluator objects.
        This class of object should be used to evaluate if an image
        is valid or not for a certain task.
    """


    def __call__(self, image: np.ndarray) -> bool:
        """

        Args:
            image (np.ndarray): image in numpy array format that will be evaluated.

        Returns:
            bool: result of the evaluation. True if the image is valid, False otherwise.
        """
        raise NotImplementedError
        