import os
import json
import numpy as np

import logging

from tqdm import tqdm

from COIGAN.training.data.dataset_splitters.base_splitter import BaseSplitter


LOGGER = logging.getLogger(__name__)


class FairSplitter(BaseSplitter):

    """
        FairSplitter class that split the dataset in a fair way.
        This class try to divide the samples in the dataset keeping the 
        same distribution of the labels.
    """

    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        train_ratio = None,
        val_ratio = None,
        test_ratio = None,
        seed: int = 42,
        binary: bool = True,
        max_chunks: int = 1000,
        target_field: str = "polygons",
        tile_size: int = 256
    ):
        """
            Init method for the FairSplitter class.

        Args:
            dataset_path (str): path to the source dataset
            output_dir (str): path to the output directory, in this directory will be created a folder for each split [train, val, test]
            train_ratio (_type_, optional): train ratio, define the percentage of the dataset that will be added to the train set. Defaults to None.
            val_ratio (_type_, optional): val ratio, define the percentage of the dataset that will be added to the val set. Defaults to None.
            test_ratio (_type_, optional): test ratio, define the percentage of the dataset that will be added to the test set. Defaults to None.
            seed (int, optional): random seed. Defaults to 42.
            binary (bool, optional): define if the source dataset is in binary mode, if true keep the splits in the same format. Defaults to True.
            max_chunks (int, optional): define how many samples should be readed in one step, usefull if the dataset is big. Defaults to 10000.
            target_field (str, optional): field that will be used as reference for th distribution. Defaults to "polygons".
            tile_size (int, optional): tile size, needed for the dataset params file. Defaults to 256.

        """
        super(FairSplitter, self).__init__(
            dataset_path,
            output_dir,
            train_ratio,
            val_ratio,
            test_ratio,
            seed,
            binary,
            max_chunks,
            tile_size
        )

        # field that will be used as reference for th distribution
        self.target_field = target_field

        self.labels_map = [] # ["class1", "class2", "class3", "class4"] classes in ordered as in the other lists
        self.labels_count_raw = [] # [[0, 0, 1, 0], [0, 1, 0, 0], ...] labels distribution in the dataset, for each sample
        self.labels_count = [] # [350, 300, 150, 200] amount of labels per class
        self.labels_distr = [] # [0.35, 0.30, 0.15, 0.20] labels distribution in the dataset

        self.train_labels_count = [] # [350, 300, 150, 200] amount of labels per class in the train set
        self.val_labels_count = [] # [350, 300, 150, 200] amount of labels per class in the val set
        self.test_labels_count = [] # [350, 300, 150, 200] amount of labels per class in the test set

        # load the dataset raw report, it will be used to
        # compute the distribution of the labels
        # NOTE: it must be already present in the dataset under the reports folder
        self.dataset_labels_distr = self._load_dataset_labels_distr()

        # log the dataset labels distribution
        msg = "dataset labels count: \n"
        for label, count in zip(self.labels_map, self.labels_count):
            msg += f"{label}: {count} "
        LOGGER.info(msg)

        if self.train_dataset is not None:
            msg = "train split labels target count: \n"
            for label, count in zip(self.labels_map, self.train_labels_count):
                msg += f"{label}: {count} "
            LOGGER.info(msg)

        if self.val_dataset is not None:
            msg = "val split labels target count: \n"
            for label, count in zip(self.labels_map, self.val_labels_count):
                msg += f"{label}: {count} "
            LOGGER.info(msg)

        if self.test_dataset is not None:
            msg = "test split labels target count: \n"
            for label, count in zip(self.labels_map, self.test_labels_count):
                msg += f"{label}: {count} "
            LOGGER.info(msg)




    def _load_dataset_labels_distr(self):
        """
            Load the dataset labels distribution from the raw report.
            This method load the dataset raw report and extract the labels distribution.
        """
        # load the dataset raw report
        raw_report_path = os.path.join(self.dataset_path, "reports", "raw_report.json")
        raw_report = json.load(open(raw_report_path, "r"))

        # extract the labels distribution
        self.labels_map = raw_report["fields_class_map"][self.target_field]
        self.labels_count_raw = raw_report["sample_polygons_class_map"][self.target_field]

        # load the amount of polygons per class
        brief_report_path = os.path.join(self.dataset_path, "reports", "brief_report.json")
        brief_report = json.load(open(brief_report_path, "r"))

        # extract the number of labes for each class
        # this amount will be used as reference for the splits
        self.labels_count= [0] * len(self.labels_map)
        for _class, val in brief_report["n_polygons_per_class"][self.target_field].items():
            self.labels_count[self.labels_map.index(_class)] = val

        # compute the labels distribution
        self.labels_distr = [x / sum(self.labels_count) for x in self.labels_count]

        # compute the target labels count for each class in each set
        self.train_labels_count = [int(x * self.train_ratio) for x in self.labels_count]
        self.val_labels_count = [int(x * self.val_ratio) for x in self.labels_count]
        self.test_labels_count = [int(x * self.test_ratio) for x in self.labels_count]


    def split(self):
        """
            Split the dataset in a fair way.
            tring to divide the samples in the dataset keeping the
            same distribution of the labels in each subset.
        """

        idxs = list(range(self.dataset_size))

        # shuffle the idxs and labels_count_raw togheter
        np.random.seed(self.seed)
        np.random.shuffle(idxs)
        np.random.seed(self.seed)
        np.random.shuffle(self.labels_count_raw)

        # preparation for the gradient selection

        n_labels = len(self.labels_map)
        n_sets = 3 # at this time fixed for train val and test sets

        target_mat = np.zeros((n_sets, n_labels), np.int32)
        state_mat =  np.zeros((n_sets, n_labels), np.int32)
        diff_mat =   np.zeros((n_sets, n_labels), np.int32)

        # output lists for sample ids ordered in sets
        idxs_lists = [[], [], []] # [[train_idxs], [val_idxs], [test_idxs]]

        # load the target values inside the matrix
        for i, (train_label_count, val_label_count, test_label_count) in enumerate(zip(self.train_labels_count, self.val_labels_count, self.test_labels_count)):
            target_mat[0][i] = train_label_count
            target_mat[1][i] = val_label_count
            target_mat[2][i] = test_label_count

        # diff_matt is equal to target_mat due to state_mat with all zeros
        diff_mat = target_mat
        
        # place holder for sets points calculation
        pts = np.zeros((n_sets,), np.float32)

        samples_sets_distr = [self.train_set_len, self.val_set_len, self.test_set_len]

        LOGGER.info("Split procedure started...")
        for j, smpl_labels in tqdm(enumerate(self.labels_count_raw), total=len(self.labels_count_raw)):
            
            # sample labels count as np array
            np_smpl_labels = np.asarray(smpl_labels)

            # iter over samples
            for i in range(n_sets):
                # iter datasets
                pts[i] = (float(np.sum((diff_mat[i]) - np_smpl_labels))/float(samples_sets_distr[i])) if samples_sets_distr[i] > 0 else 0

            # get the datasets ordered by affinity with the current sample
            # argsort return index of the lowest value first
            # flip put the index of the bigget value first
            ordered_idxs = np.flip(np.argsort(pts, axis=0))

            # check which dataset has space for the new sample
            for i in ordered_idxs:
                # if the current selected set has space for a new example
                if idxs_lists[i].__len__() < samples_sets_distr[i]:
                    idxs_lists[i].append(idxs[j])
                    state_mat[i] += np_smpl_labels
                    diff_mat[i] -= np_smpl_labels
                    break
            
        # save the splits
        train_idxs = idxs_lists[0]
        val_idxs = idxs_lists[1]
        test_idxs = idxs_lists[2]

        # add the samples to the datasets
        if self.train_dataset is not None:
            LOGGER.info("Saving the train set...")
            train_idxs = np.array_split(train_idxs, len(train_idxs)//self.max_chunks)
            for train_idxs_chunk in tqdm(train_idxs):
                train_samples = self.dataset[train_idxs_chunk]
                self.train_dataset.insert(train_samples)
                
                self.copy_images(
                    train_samples, 
                    self.dataset_images_path, 
                    self.train_images_path
                )
            self.train_dataset.close()
        
        if self.val_dataset is not None:
            LOGGER.info("saving the val set...")
            val_idxs = np.array_split(val_idxs, len(val_idxs)//self.max_chunks)
            for val_idxs_chunk in tqdm(val_idxs):
                val_samples = self.dataset[val_idxs_chunk]
                self.val_dataset.insert(val_samples)
                
                self.copy_images(
                    val_samples,
                    self.dataset_images_path,
                    self.val_images_path
                )
            self.val_dataset.close()
        
        if self.test_dataset is not None:
            LOGGER.info("saving the test set...")
            test_idxs = np.array_split(test_idxs, len(test_idxs)//self.max_chunks)
            for test_idxs_chunk in tqdm(test_idxs):
                test_samples = self.dataset[test_idxs_chunk]
                self.test_dataset.insert(test_samples)
                
                self.copy_images(
                    test_samples,
                    self.dataset_images_path,
                    self.test_images_path
                )
            self.test_dataset.close()
        
        self.generate_split_briefs()
        
        LOGGER.info("Done!")

        

