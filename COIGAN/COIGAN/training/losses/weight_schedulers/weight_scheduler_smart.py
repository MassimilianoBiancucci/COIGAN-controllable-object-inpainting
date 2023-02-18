from COIGAN.training.losses.weight_schedulers import WeightSchedulerLinear

class WeightSchedulerSmart(WeightSchedulerLinear):

    def __init__(
        self, 
        max_weight: float, 
        min_weight: float, 
        n_steps: int, 
        descendant: bool = True
    ):

        super().__init__(max_weight, min_weight, n_steps, descendant)

        self.loss_avg = 0
        self.loss_count = 0
        self.loss2_avg = 0
        self.loss2_count = 0


    def __call__(
        self,
        step: int,
        loss: float,
        loss2: float
    ):
        """
        TODO: implement the smart scheduler.

        Args:
            step (int): _description_
            loss (float): _description_
            loss2 (float): _description_
        """
        NotImplementedError("This method is not implemented yet")
