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
        # Fused Softmax + CCE gradient: (y_pred - one_hot(y_true)) / N
        N = len(y_true)
        grad = y_pred.copy()
        grad[np.arange(N), y_true] -= 1.0
        return grad / N


class BinaryCrossEntropy(LossBase):
    def __init__(self) -> None:
        super().__init__("BinaryCrossEntropy")

    def loss_func(self, y_pred, y_true):
        return -np.mean(
            y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)
        )

    def loss_gradient(self, y_pred, y_true):
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-8)

class Hinge(LossBase):
    def __init__(self) -> None:
        super().__init__("Hinge")

    def loss_func(self, y_pred, y_true):
        return np.maximum(0, 1 - y_pred * y_true)

    def loss_gradient(self, y_pred, y_true):
        yz = y_pred * y_true
        return np.where(yz < 1, -y_true, 0)

class Huber(LossBase):
    def __init__(self, delta=1.0):
        super().__init__("Huber")
        self.delta = delta

    def loss_func(self, y_pred, y_true):
        error = y_true - y_pred
        L1 = error ** 2 / 2
        L2 = self.delta * (np.abs(error) - (self.delta / 2))
        return np.where(np.abs(error) > self.delta, L1, L2)

    def loss_gradient(self, y_pred, y_true):
        error = y_pred - y_true
        return np.where(np.abs(error) > self.delta, self.delta * np.sign(error), error)
