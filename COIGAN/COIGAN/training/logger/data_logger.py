import os
import torch
import logging

from omegaconf import OmegaConf

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
        local_log_path: str = None,
        wandb_kwargs: dict = None,
        config: OmegaConf = None,
    ):
        """
            Init the DataLogger class.

            Args:
                wandb_kwargs (dict): the wandb kwargs. If None, wandb is not used.
        """

        # init wandb
        self.wandb = False # flag that indicates if wandb is used
        if wandb_kwargs is not None:
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
        
        # init local 
        self.local_log = False # flag that indicates if local is used
        if local_log_path is not None:
            self.local_log_path = local_log_path
            self.local_log = True

            # create a csv file containing the training metrics
            self.local_log_file = os.path.join(self.local_log_path, "training_metrics.csv")
            #with open(self.local_log_file, "w") as f:
            #    f.write("step_idx,loss\n")
            # TODO add the header

    
    def log_step_results(self, step_idx: int, results: dict):
        """
            Method that log the results of the step.

            Args:
                step_idx (int): current step
                results (dict): the results of the step
        """
        # log the results on wandb
        if self.wandb:
            # TODO save the results on wandb
            pass
        
        # log the results on local
        if self.local_log:
            # TODO save the results on local
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
            # TODO save the results on wandb
            pass
        
        # log the results on local
        if self.local_log:
            # TODO save the results on local
            pass