
class DefectDataset:

    def __init__(self, indir, size):
        """
        Object that contains the dataset for training the model.
        This dataset is designed for loading images of cars,
        images of defects.
        
        Args:
            indir (str): path to the directory containing the images
            size (int): size of the images
            mask_geenrator_kwargs (dict): kwargs for the mask generator
        
        """

        

        