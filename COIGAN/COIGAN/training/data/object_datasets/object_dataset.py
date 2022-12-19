
class ObjectDataset:
    
    """ 
        Object used as base for the target objects dataloader feeded to the discriminator.
        this object need a dataset with a list of images of the target objects with their segmentation masks.
        This object load the images of the target objects and their respective masks.

        

        legend:
            -images: directory containing the base images
            -images_segm.jsonl: jsonl file containing the segmentation masks of the images, the segmentations must be provided as a list of polygons associated to each image
            -index_images_segm: index of the images_segm.jsonl file, contain a list of the start positions of each json, to allow a random access
    """

    def __init__(
        self,
        indir,
        size
    ):
        """
            Args:
                indir (str): path to the directory containing the images
                size (int): size of the images
                mask_applicator_kwargs (dict): kwargs for the mask applicator
        """