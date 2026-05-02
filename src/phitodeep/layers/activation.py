import numpy as np

from .base import Layer


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

    def copy(self):
        new_layer = ReLu()
        new_layer.cache = self.cache.copy()
        return new_layer

class LeakyReLu(Layer):
    def __init__(self, alpha=0.01) -> None:
        super().__init__("leaky_rely")
        self.alpha = alpha

    def forward(self, X):
        self.cache["X"] = X
        return np.where(X > 0, X, self.alpha * X)

    def backward(self, dL_dZ):
        X = self.cache["X"]
        dL_dX = dL_dZ * np.where(X > 0, 1, self.alpha)
        return dL_dX

    def copy(self):
        new_layer = LeakyReLu()
        new_layer.cache = self.cache.copy()
        return new_layer

class GELU(Layer):
    def __init__(self) -> None:
        super.__init__("gelu")

    def forward(self, X):
        self.cache["X"] = X
        constant = 0.044715
        inner = (np.sqrt(2.0 / np.pi) * (X + constant * X ** 3))
        t = Tanh().forward(inner)
        self.cache["t"] = t
        return 0.5 * X * (1 + t)

    def backward(self, dL_dz):
        t = self.cache["t"]
        X = self.cache["X"]
        dL_dX = 0.5 * (1 + t)
        dL_dX += 0.5 * x * (1 - t ** 2) * np.sqrt(2.0 / np.pi)
        dL_dX *= (1 + 3 * constant * X ** 3)
        dL_dX *= dL_dZ
        return dl_dX

    def copy(self):
        new_layer = GELU()
        new_layer.cache = self.cache.copy()
        return new_layer

class Swish(Layer):
    def __init__(self) -> None:
        super().__init__("swish")

    def forward(self, X):
        self.cache["X"] = X
        Z = 1 / (1 + np.exp(-X))
        self.cache["Z"] = Z
        return X * Z

    def backward(self, dL_dZ):
        X = self.cache["X"]
        Z = self.cache["Z"]
        dL_dX =  dL_dZ * (Z + X * Z * (1 - Z))
        return dL_dX

    def copy(self):
        new_layer = Swish()
        new_layer.cache = self.cache.copy()
        return new_layer

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

    def copy(self):
        new_layer = Sigmoid()
        new_layer.cache = self.cache.copy()
        return new_layer


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

    def copy(self):
        new_layer = Tanh()
        new_layer.cache = self.cache.copy()
        return new_layer


class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__("softmax")

    def forward(self, X):
        self.cache["X"] = X
        axis = None if X.ndim < 2 else 1
        max_a = np.max(X, axis=axis, keepdims=True)

        dividend = np.exp(X - max_a)
        divisor = np.sum(np.exp(X - max_a), axis=axis, keepdims=True)

        self.cache["Z"] = dividend / divisor
        return self.cache["Z"]

    def backward(self, dL_dZ):
        """
        Backpropagate through Softmax activation.
        When paired with CategoricalCrossEntropy, the combined gradient
        (y_pred - one_hot(y_true)) / N is computed entirely in the loss,
        so this layer is a straight pass-through.
        """
        return dL_dZ

    def copy(self):
        new_layer = Softmax()
        new_layer.cache = self.cache.copy()
        return new_layer


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

    def copy(self):
        new_layer = ELU(self.alpha_activation)
        new_layer.cache = self.cache.copy()
        return new_layer
