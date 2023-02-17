import os
import torch
import logging

from omegaconf import OmegaConf
from typing import Union

try:
    import wandb
except ImportError:
    wandb = None

LOGGER = logging.getLogger(__name__)

class DataLogger:
    """
    Class that handles the training data and log them localy and on wandb.
    This class is used even for save the data to visualize, in local and on wandb.
    """

    def __init__(
        self,
        logs_dir: str = None,
        log_weights_interval: int = 1000,
        enable_wandb: bool = False,
        wandb_kwargs: dict = None,
        config: OmegaConf = None,
    ):
        """
            Init the DataLogger class.

            Args:
                logs_dir (str): the directory where the logs are saved.
                log_weights_interval (int): the interval between two weights and gradients logging,
                    NOTE: this value only state every how many calls of "log_weights_and_gradients()" it actually log it.
                    This solve a little desincronization that can happen in the training pipeline.
                enable_wandb (bool): if True wandb is used. Otherwise the wandb initialization is skipped.
                wandb_kwargs (dict): the wandb kwargs. If None, wandb is not used.
        """

        self.log_weights_interval = log_weights_interval
        self.log_weights_counter = 0

        # init wandb
        self.wandb = False # flag that indicates if wandb is used
        if enable_wandb and wandb_kwargs is not None:
            if wandb is None:
                LOGGER.warning("wandb is not installed, wandb_kwargs will be ignored!")
            else:
                # check that config is not None
                if config is None:
                    raise ValueError("config must be passed if wandb is used!")

                wandb.init(**wandb_kwargs)
                wandb.config.update(
                    OmegaConf.to_container(config, resolve=True)
                )
                self.wandb = True
                self.distributed = config.distributed

        # init local 
        self.local_log = False # flag that indicates if local is used
        if logs_dir is not None:
            self.logs_dir = logs_dir
            self.local_log = True

            # create a csv file containing the training metrics
            #self.log_file = os.path.join(self.logs_dir, "training_metrics.csv")
            #with open(self.log_file, "w") as f:
            #    f.write("step_idx,loss\n")

    
    def log_step_results(self, step_idx: int, results: dict):
        """
            Method that log the results of the step.

            Args:
                step_idx (int): current step
                results (dict): the results of the step
        """
        # log the results on wandb
        if self.wandb:
            # log the results to wandb
            wandb.log(results, step=step_idx)
        
        # log the results on local
        if self.local_log:
            # log the results to local
            pass
    

    def log_visual_results(self, step_idx: int, results: dict):
        """
            Method that log the results of the step.

            Args:
                step_idx (int): current step
                results (dict): the results of the step
        """
        # log the results on wandb
        if self.wandb:
            # wrap each image in a wandb.Image object
            wandb_img_results = {}
            for key, value in results.items():
                wandb_img_results[key] = wandb.Image(value)
            
            # log the results to wandb
            wandb.log(wandb_img_results, step=step_idx)
        
        # log the results on local
        if self.local_log:
            # TODO save the results on local
            pass
    
    
    def log_weights_and_gradients(self, model: Union[torch.nn.Module, torch.nn.DataParallel]):
        """
            Method that log the weights of the model.

            Args:
                step_idx (int): current step
                model (torch.nn.Module or torch.nn.DataParallel): the model
        """
        if self.wandb and \
            self.log_weights_counter % self.log_weights_interval == 0:
            
            if self.distributed:
                # if the model is wrapped in a DataParallel, unwrap it
                model = model.module

            histograms = {}
            model_name = model.__class__.__name__
            for tag, value in model.named_parameters():
                tag = tag.replace("/", ".")
                histograms[f"{model_name}_Weights/" + tag] = wandb.Histogram(
                    value.data.cpu()
                )
                histograms[f"{model_name}_Gradients/" + tag] = wandb.Histogram(
                    value.grad.data.cpu()
                )

            wandb.log(histograms)

        # increment the counter
        self.log_weights_counter += 1
