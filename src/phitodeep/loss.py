import numpy as np


class LossBase:
    def __init__(self, name) -> None:
        self.name = name

    def loss_func(self, y_pred, y_true):
        raise NotImplementedError(f"{self.name} must implement the loss_func method.")

    def loss_gradient(self, y_pred, y_true):
        raise NotImplementedError(
            f"{self.name} must implement the loss_gradient method."
        )


class MeanSquaredError(LossBase):
    def __init__(self) -> None:
        super().__init__("MeanSquaredError")

    def loss_func(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def loss_gradient(self, y_pred, y_true):
        m = len(y_true)
        return 2 * (y_pred - y_true) / m


class CategoricalCrossEntropy(LossBase):
    def __init__(self) -> None:
        super().__init__("CategoricalCrossEntropy")

    def loss_func(self, y_pred, y_true):
        N = len(y_true)
        correct = y_pred[np.arange(N), y_true]
        return -np.mean(np.log(correct + 1e-8))

    def loss_gradient(self, y_pred, y_true):
        N = len(y_true)
        correct = y_pred[np.arange(N), y_true]
        return -y_true / (correct + 1e-8)


class BinaryCrossEntropy(LossBase):
    def __init__(self) -> None:
        super().__init__("BinaryCrossEntropy")

    def loss_func(self, y_pred, y_true):
        return -np.mean(
            y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)
        )

    def loss_gradient(self, y_pred, y_true):
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-8)
