import numpy as np


class AlphaScheduler:
    def __ini__(self, name: str, alpha: np.float16) -> None:
        self.name = name
        self.alpha = alpha

    def step(self, time_step: int) -> np.float16:
        raise NotImplementedError(f"{self.name} must implement the step function")

class InverseTimeDecay(AlphaScheduler):
    def __init__(self, alpha: np.float16, decay_rate: np.float16=0.01):
        super().__init__("InverseTimeDecay", alpha)
        self.decay_rate = decay_rate

    def step(self, time_step: int) -> np.float16:
        return self.alpha / (1 + self.decay_rate * time_step)
