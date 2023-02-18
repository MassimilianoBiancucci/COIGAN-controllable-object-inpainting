
class WeightSchedulerBase:
    """
    The weight scheduler base class,
    used to schedule losses' weights during training.
    """

    def __init__(
        self,
        max_weight: float,
        min_weight: float,
    ):
        """
        
        """
    
        self.max_weight = max_weight
        self.min_weight = min_weight


    def __call__(self, step: int) -> float:
        """
        Returns the weight for the current step.
        """
        raise NotImplementedError