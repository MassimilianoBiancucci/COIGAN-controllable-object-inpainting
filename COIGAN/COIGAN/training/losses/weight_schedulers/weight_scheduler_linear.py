from COIGAN.training.losses.weight_schedulers import WeightSchedulerBase

class WeightSchedulerLinear(WeightSchedulerBase):
    """
    The weight scheduler linear class,
    used to schedule losses' weights during training.
    This scheduler apply a linear function to the weights,
    descending from the max weight to the min weight, if descendent is True,
    or ascending from the min weight to the max weight, if descendent is False,
    in n_steps steps.
    """

    def __init__(
        self,
        max_weight: float,
        min_weight: float,
        n_steps: int,
        descendent: bool = True,
    ):
        """
        
        """
        super().__init__(max_weight, min_weight)

        self.n_steps = n_steps
        self.descendent = descendent

        # increment or decrement of the weight in each step
        self.delta = (self.max_weight - self.min_weight) / self.n_steps


    def __call__(self, step: int) -> float:
        """
        Returns the weight for the current step.

        Args:
            step (int): the current step
        
        Returns:
            float: the weight for the current step
        """
        if self.descendent:
            if step > self.n_steps:
                return self.min_weight # if reached the last step, return the min weight
            return self.max_weight - self.delta * step # decrement if not reached the last step
        else:
            if step > self.n_steps:
                return self.max_weight # if reached the last step, return the max weight
            return self.min_weight + self.delta * step # increment if not reached the last step
