
from COIGAN.training.data.base_datasets.base_dataset import BaseDataset
from COIGAN.training.data.object_datasets.object_dataset import ObjectDataset

class Controllable_object_inpainting_dataset:

    """
        This class is the main interface between the data and the training process.

    """

    def __init__(
        self, 
        BaseDataset: BaseDataset,
        ObjectDataset: ObjectDataset
        ):
        """
        Init the class
        ...

        Args:
            BaseDataset (BaseDataset): _description_
            ObjectDataset (ObjectDataset): _description_
        """

        self.base_dataset = BaseDataset
        self.object_dataset = ObjectDataset