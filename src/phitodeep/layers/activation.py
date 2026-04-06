import numpy as np
from layers.base import Layer


class ReLu(Layer):
    def __init__(self) -> None:
        super().__init__("relu")

    def forward(self, X):
        self.cache["X"] = X
        return np.maximum(0, X)

    def backward(self, dL_dZ):
        """
        Backpropagate through ReLU activation.
        ReLU derivative: 1 if X > 0, else 0
        """
        X = self.cache["X"]
        dL_dX = dL_dZ * (X > 0).astype(float)
        return dL_dX


class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__("sigmoid")

    def forward(self, X):
        self.cache["X"] = X
        self.cache["Z"] = 1 / (1 + np.exp(-X))
        return self.cache["Z"]

    def backward(self, dL_dZ):
        """
        Backpropagate through Sigmoid activation.
        Sigmoid derivative: sigmoid(Z) * (1 - sigmoid(Z))
        """
        Z = self.cache["Z"]
        dL_dX = dL_dZ * Z * (1 - Z)
        return dL_dX


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__("tanh")

    def forward(self, X):
        self.cache["X"] = X
        e_x = np.exp(X)
        e_neg_x = np.exp(-X)
        self.cache["Z"] = (e_x - e_neg_x) / (e_x + e_neg_x)
        return self.cache["Z"]

    def backward(self, dL_dZ):
        """
        Backpropagate through Tanh activation.
        Tanh derivative: 1 - tanh(Z)^2
        """
        Z = self.cache["Z"]
        dL_dX = dL_dZ * (1 - Z**2)
        return dL_dX


class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__("softmax")

    def forward(self, X):
        self.cache["X"] = X
        max_a = np.max(X)
        axis = None if X.ndim < 2 else 1

        dividend = np.exp(X - max_a)
        divisor = np.sum(np.exp(X - max_a), axis=axis, keepdims=True)

        self.cache["Z"] = dividend / divisor
        return self.cache["Z"]

    def backward(self, dL_dZ):
        """
        Backpropagate through Softmax activation.
        For softmax with cross-entropy loss, dL_dX = softmax(Z) - y_true
        For MSE loss: approximate derivative as softmax(Z) * (1 - softmax(Z))
        """
        Z = self.cache["Z"]
        # Element-wise product with jacobian diagonal approximation
        dL_dX = dL_dZ * Z * (1 - Z)
        return dL_dX


class ELU(Layer):
    def __init__(self, alpha=1.0) -> None:
        super().__init__("elu")
        self.alpha_activation = alpha

    def forward(self, X):
        self.cache["X"] = X
        self.cache["Z"] = np.where(X > 0, X, self.alpha_activation * (np.exp(X) - 1))
        return self.cache["Z"]

    def backward(self, dL_dZ):
        """
        Backpropagate through ELU activation.
        ELU derivative: 1 if X > 0, else alpha * exp(X)
        """
        X = self.cache["X"]
        dL_dX = dL_dZ * np.where(X > 0, 1.0, self.alpha_activation * np.exp(X))
        return dL_dX
