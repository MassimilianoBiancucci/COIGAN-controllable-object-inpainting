import os
import cv2
import numpy as np
import json

import logging
import traceback

from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count
from omegaconf.listconfig import ListConfig
from typing import Union, Tuple, List

LOGGER = logging.getLogger(__name__)

from COIGAN.training.data.dataset_generators.jsonl_dataset_generator import JsonLineDatasetBaseGenerator
from COIGAN.training.data.datasets_loaders.severstal_steel_defect import SeverstalSteelDefectDataset

class SeverstalSteelDefectPreprcessor(JsonLineDatasetBaseGenerator):
    """
    That module convert the Severstal Steel Defect Dataset into a COIGAN compatible dataset.

    Operations:
        - convert the masks into polygons
        - split each sample in squared tiles
        - save each tile as a separated image and add the corresponding masks as polygons in a jsonl file
    """

    def __init__(
        self,
        raw_dataset: SeverstalSteelDefectDataset,
        output_dir: str,
        tile_size: Union[int, Tuple[int, int], List[int], ListConfig],
        dump_every: int = 1000,
        binary: bool = True,
        n_workers: int = 1,
        q_size: int = 10
    ):
        """
        Init the SeverstalSteelDefectPreprcessor

        Args:
            raw_dataset (SeverstalSteelDefectDataset): The raw dataset
            output_dir (str): Path to the output directory
            tile_size (int, Tuple[int, int], list[int], ListConfig): The size of the tile
            dump_every (int, optional): Dump the metadata every n samples. Defaults to 1000.
            binary (bool, optional): If True the masks are binarized. Defaults to True.
            n_workers (int, optional): Number of workers to use. Defaults to 1.

        """
        super().__init__(
            output_dir, 
            dump_every=dump_every, 
            binary=binary
        )

        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, "data")
        self.raw_dataset = raw_dataset
        
        if isinstance(tile_size, (tuple, list, ListConfig)):
            self.tile_size = tile_size
        elif isinstance(tile_size, int):
            self.tile_size = (tile_size, tile_size)

        self.n_workers = cpu_count() if n_workers == -1 else n_workers
        self.q_size = q_size

        # create the data directory
        os.makedirs(self.data_dir, exist_ok=True)

        self.genrate_params_brief()


    def convert(self):
        """
            Convert the dataset into COIGAN compatible dataset.
        """

        if self.n_workers != 1:
            self.convert_parallel()
            return

        LOGGER.info("Converting the dataset into COIGAN compatible dataset")
        idx = 0
        for image, masks in tqdm(self.raw_dataset):
            images, metadata_lst = self.preprocess(image, masks, self.tile_size)
            
            for image, metadata in zip(images, metadata_lst):
                img_name = f"{idx}.jpg"
                idx += 1

                metadata["img"] = img_name
                cv2.imwrite(os.path.join(self.data_dir, img_name), image)
            
            self.insert(metadata_lst)

        self.close()


    def convert_parallel(self):
        """
            Convert the dataset into COIGAN compatible dataset using multiple workers.
        """
        LOGGER.info(f"Converting the dataset into COIGAN compatible dataset using {self.n_workers} workers")
        
        # create a tqdm bar
        pbar = tqdm(total=len(self.raw_dataset))

        # create input output queues
        sample_queues = [Queue(self.q_size) for _ in range(self.n_workers)]
        output_queue = Queue()

        # create the processes
        process_objects = []
        for i in range(self.n_workers):
            process_objects.append(
                PreprocessTask(
                    sample_queues[i], 
                    output_queue, 
                    self.data_dir,
                    self.tile_size
                )
            )

        # start the processes
        processes = []
        for i, proc in enumerate(process_objects):
            processes.append(
                Process(target=proc.run, name=f"preprocessor_{i}"))
            processes[-1].start()
        
        idx = 0
        for image, masks in self.raw_dataset:

            # load the sample in the first available queue
            found_free_proc = False
            while not found_free_proc:
                q_stats = [q.qsize() for q in sample_queues]
                if min(q_stats) < self.q_size:
                    sample_queues[q_stats.index(min(q_stats))].put([idx, image, masks])
                    idx += 1
                    found_free_proc = True
                    break
            
            pbar.update(1)
        
            # check for results in the output queues
            while not output_queue.empty():
                metadata_lst = output_queue.get()
                self.insert(metadata_lst)
        
        # closing the pbar before sending the stop signal
        pbar.close()

        # finalization of the process, sending the stop signal
        print("sending the stop signal to the processes")
        for queue in sample_queues:
            queue.put("END")
        
        n_end = 0
        while n_end < self.n_workers:
            if not output_queue.empty():
                msg = output_queue.get()
                if msg == "END":
                    n_end += 1
                else:

                    self.insert(msg)

        # wait for all the processes to finish
        for proc in processes:
            proc.join()

        pbar.close()
        self.close()


    def genrate_params_brief(self):
        """
            Generate a brief of the dataset parameters.
        """
        # generate a file that specify the dataset generation parameters
        params = {
            "tile_size": [
                self.tile_size[0],
                self.tile_size[1]
            ]
        }

        with open(os.path.join(self.output_dir, "params.json"), "w") as f:
            json.dump(params, f)


    @staticmethod
    def preprocess(image, masks, tile_size=(256, 256)):
        """
            Preprocess the image and the masks.
            operations:
                - split the image and the masks in tiles
                - convert each mask into a polygon
                - create a  dict for each tile
                - return a list of images and a list of dict

            Args:
                image (np.ndarray): input image
                masks (np.ndarray): input masks
                tile_size (tuple, optional): tile size. Defaults to (256, 256).
        """

        # split the image and masks in tiles
        images, masks = SeverstalSteelDefectPreprcessor.split_image_and_masks(image, masks, tile_size)

        # convert the masks into polygons
        polygons = SeverstalSteelDefectPreprcessor.masks_to_polygons(masks)

        return images, polygons


    @staticmethod
    def split_image_and_masks(image, _masks, tile_size=(256, 256)):
        """
            Split the image and the masks in tiles.
            the number of tiles is determined by the tile_size as
            w_tiles = (w // tile_size[1]) +1
            h_tiles = (h // tile_size[0]) +1

            the tiles normaly have a litle overlap, it depends on the tile_size and the image size.

            Args:
                image (np.ndarray): input image
                masks (np.ndarray): input masks
                tile_size (tuple, optional): tile size. Defaults to (256, 256).
            
            Returns:
                list[np.ndarray]: return a list of image's tiles
                list[Union[np.ndarray, None]]: return a list of masks' tiles or None if there aren't masks
            
            TODO: debug the function!
        """

        h, w = image.shape[:2]

        # if the tile size is equal to the image size, return the image and the masks
        # just jump the process
        if h == tile_size[0] and w == tile_size[1]:
            return [image], [_masks]

        images = []
        masks = []

        nh_tiles = np.ceil(tile_size[0])
        h_offset = np.floor((h-tile_size[0])/(nh_tiles-1)).astype(np.int32) \
            if h - tile_size[0] > 0 else tile_size[0]

        nw_tiles = np.ceil(w/tile_size[1])
        w_offset = np.floor((w-tile_size[1])/(nw_tiles-1)).astype(np.int32) \
            if w - tile_size[1] > 0 else tile_size[1]

        for i in range(0, h, h_offset):
            for j in range(0, w, w_offset):
                if i+tile_size[0] <= h and j+tile_size[1] <= w:
                    images.append(image[i:i+tile_size[0], j:j+tile_size[1]])

                    # the masks are optional, if there aren't defects in the image
                    # the masks will be None
                    if _masks is not None:
                        masks.append(_masks[i:i+tile_size[0], j:j+tile_size[1]])
                    else:
                        masks.append(None)

        return images, masks


    @staticmethod
    def masks_to_polygons(
        masks
    ):
        """
            Convert the masks into polygons.

            Args:
                masks (np.ndarray): input masks
            
            Returns:
                list[dict]: return a list of dict with the following structure:
                    {
                        "img": "",
                        polygons: [
                            {
                                "label": "1",
                                "points": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]]
                            },
                            ...
                        ]
                    }
                    NOTE: the image name isn't specified here, it will be added in the writer.
                        it's returned only as an ampty string.
        """

        samples = []
        for mask in masks: # iter over many tiles from one original sample
            sml_polygons = []
            if mask is not None: # if there are masks
                for class_id in range(mask.shape[-1]): # iter over the masks classes of one tile
                    if np.sum(mask[..., class_id]) != 0: # if the mask is empty
                        polygons = SeverstalSteelDefectPreprcessor.mask2poly(mask[..., class_id]) # convert the mask into polygons
                        for poly in polygons: # iter over the polygons of one mask
                            sml_polygons.append(
                                {
                                    "label": str(class_id),
                                    "points": [poly.squeeze().tolist()]
                                }
                            )

            samples.append({
                "img": "", 
                "polygons": sml_polygons
            })

        return samples


    @staticmethod
    def mask2poly(
        mask, 
        reduction_factor=4, 
        normalize=False, 
        min_poly_points=25,
        min_allowed_poly_points=5,
        check_validity=True
    ):
        """
        Method that return a list of poligons
        given a uint8 mask, and reduce the number of points by a
        factor of reduction_factor.

        Args:
            mask (np.array): mask of the segmentation
            reduction_factor (int): factor of reduction of the number of points, default 10
                                    for a reduction of 10, the number of points is divided by 10.
            normalize (bool): normalize the points between 0 and 1, default False
            min_poly_points (int): minimum number of points to reduce the number of points, default 25
            min_allowed_poly_points (int): minimum number of points allowed, default 5
            check_validity (bool): check the validity of the polygon, default False, if True the function
                                    will return only the valid polygons, and the number of skipped polygons.
        """

        h, w = mask.shape[:2]

        raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # remove the contours with less than "min_allowed_poly_points"
        contours = []
        for c in raw_contours:
            if len(c) >= min_allowed_poly_points:
                contours.append(c)
        
        # foreach contour, reduce the number of point by the reduction factor
        if reduction_factor > 1:
            reduced_contours = []
            for c in contours:
                # reduce the number of points if the number of points is greater than "min_poly_points"
                if len(c) > min_poly_points:
                    reduced_contours.append(c[::reduction_factor])
                else:
                    reduced_contours.append(c)
        else:
            reduced_contours = contours

        # normalize the contours between 0 and 1 using the shape of the mask
        if normalize:
            contours_norm = []
            for contour in reduced_contours:
                c = []
                # contour = np.squeeze(contour, axis=1)
                for p in contour:
                    c.append([(p[0, 0]) / w, (p[0, 1]) / h])
                    contours_norm.append(c)
        else:
            contours_norm = reduced_contours
        
        # check if the polygon is valid
        if check_validity:
            valid_contours = []
            for c in contours_norm:
                if SeverstalSteelDefectPreprcessor.check_polygon_healt(c):
                    valid_contours.append(c)
            return valid_contours

        return contours_norm


    @staticmethod
    def check_polygon_healt(
        polygon
    ):
        """
        Check if the polygon is valid, if it's not valid, return False.
        A valid polygon is a polygon with at least 3 points and with
        at least 2 points in each axis different.

        one polygon with all the x or y coordinates equal is not valid!

        Args:
            polygon (list[list[list[int]]]): list of points of the polygon
            exception (bool): if True, raise an exception if the polygon is not valid, default False.

        Returns:
            bool: True if the polygon is valid, False otherwise.
        """
        # check the number of points
        if polygon.shape[0] < 3:
            LOGGER.warning("Found one polygon with less than 3 points!")
            return False
        
        # check if there are at least 2 points in each axis
        polygon = polygon.squeeze()
        if len(set(polygon[:, 0])) < 2 or len(set(polygon[:, 1])) < 2:
            LOGGER.warning("Found one polygon with all the points on the same line!")
            return False

        return True


class PreprocessTask:

    def __init__(
        self, 
        in_q, 
        out_q, 
        output_data_dir,
        tile_size=(256, 256),
        verbose=False
    ):

        self.in_q = in_q
        self.out_q = out_q
        self.output_data_dir = output_data_dir
        self.tile_size = tile_size
        self.verbose = verbose
    
    def run(self):
        
        while (sample := self.in_q.get()) != "END":

            try:
                # apply the preprocess to the sample
                idx, image, masks = sample
                (images, metadata_lst) = SeverstalSteelDefectPreprcessor.preprocess(image, masks, self.tile_size)

                for i, (img, metadata) in enumerate(zip(images, metadata_lst)):
                    img_name = f"{idx}_{i}.jpg"
                    metadata["img"] = img_name
                    cv2.imwrite(os.path.join(self.output_data_dir, img_name), img)

                # put the sample in the output queue
                self.out_q.put(metadata_lst)

            except Exception as e:
                if self.verbose:
                    print(e)
                    # print stack trace
                    traceback.print_exc()
                    #print("Error in the sample: ", sample["filename"])
                continue

        # end the process
        self.out_q.put("END")

