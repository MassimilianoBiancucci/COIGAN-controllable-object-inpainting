#!/usr/bin/env python3

import os
import logging
import json

os.environ["HYDRA_FULL_ERROR"] = "1"

import cv2
import torch

import hydra

from tqdm import tqdm
from omegaconf import OmegaConf

from COIGAN.utils.common_utils import sample_data
from COIGAN.training.data.datasets_loaders import make_dataloader
from COIGAN.inference.coigan_inference import COIGANinference
from COIGAN.evaluation.losses.fid.fid_score import calculate_fid_given_paths

LOGGER = logging.getLogger(__name__)


def generate_inference_dataset(config):
    """
    Generate the dataset from the trained model.
    """

    # create the folder for the generated images
    out_path = config.generated_imgs_path
    os.makedirs(out_path, exist_ok=True)

    n_samples = config.n_samples
    dataloader = sample_data(make_dataloader(config))
    model = COIGANinference(config)
    idx = 0
    pbar = tqdm(total=n_samples)

    while True:
        # inference on the next sample
        sample = next(dataloader)
        inpainted_img = model(sample["gen_input"])

        # save the inpainted image in the target folder
        for img in inpainted_img:
            cv2.imwrite(os.path.join(out_path, f"{idx}.png"), img)
            pbar.update()
            idx += 1

            if idx >= n_samples:
                return


@hydra.main(config_path="../configs/evaluation/", config_name="test_eval.yaml", version_base="1.1")
def main(config: OmegaConf):

    #resolve the config inplace
    OmegaConf.resolve(config)

    LOGGER.info(f'Config: {OmegaConf.to_yaml(config)}')

    # save ghe config in the output folder
    OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml')) # saving the configs to config.hydra.run.dir

    # generate the dataset for the evaluation step with the FID metric
    LOGGER.info("Generating the dataset for the evaluation step...")
    generate_inference_dataset(config)

    ref_fid = calculate_fid_given_paths(
        [config.train_imgs_path, config.test_imgs_path],
        config.inc_batch_size,
        config.device,
        config.inception_dims,
        n_imgs=config.n_samples
    )
    LOGGER.info(f"Ref FID: {ref_fid}")

    # evaluate the generated dataset with the FID metric
    fid = calculate_fid_given_paths(
        [config.generated_imgs_path, config.test_imgs_path],
        config.inc_batch_size,
        config.device,
        config.inception_dims,
        n_imgs=config.n_samples
    )
    LOGGER.info(f"FID: {fid}")

    # create a report of the evaluation
    with open(os.path.join(os.getcwd(), "report.json"), "w") as f:
        json.dump({
            "ref FID": ref_fid,
            "FID": fid
        }, f)


if __name__ == "__main__":
    main()