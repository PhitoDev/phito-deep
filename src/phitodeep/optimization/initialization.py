from enum import Enum
import numpy as np


class InitType(Enum):
    NORMAL = 1
    UNIFORM = 2

class Initializer:
    def __init__(self, name: str, init_type: InitType):
        self.name = name
        self.init_type = init_type

    def get_scale(self, weights: np.ndarray) -> float:
        raise NotImplementedError(
            f"{self.name} initialization must implement get_weights."
        )

class Xavier(Initializer):
    def __init__(self, init_type: InitType=InitType.NORMAL):
        super().__init__("Xavier", init_type)

    def get_scale(self, weights: np.ndarray) -> float:
        divident = 2 if self.init_type == InitType.NORMAL else 6
        fan_in = weights.shape[0]
        fan_out = weights.shape[1]

        return np.sqrt(divident / (fan_in + fan_out))

class He(Initializer):
    def __init__(self, init_type: InitType=InitType.NORMAL):
        super().__init__("He", init_type)

    def get_scale(self, weights: np.ndarray) -> float:
        divident = 2 if self.init_type == InitType.NORMAL else 6
        fan_in = weights.shape[0]

        return np.sqrt(divident / fan_in)
