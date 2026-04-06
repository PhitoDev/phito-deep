import numpy as np


class Layer:
    def __init__(self, name) -> None:
        self.name = name
        self.cache = {}
        self.grads = {}

    def forward(self, X):
        raise NotImplementedError(f"Block '{self.name}' must implement forward method")

    def backward(self, dL_dZ):
        """
        Backward pass through the block.

        Args:
            dL_dZ: gradient of loss w.r.t. output of this block

        Returns:
            dL_dX: gradient of loss w.r.t. input (to pass to previous layer)
        """
        raise NotImplementedError(f"Block '{self.name}' must implement backward method")


class Flatten(Layer):
    def __init__(self):
        super().__init__("flatten")

    def forward(self, X):
        """
        X: (batch_size, ...) -> (batch_size, ...)
        """
        self.cache["X"] = X
        return X.reshape(X.shape[0], -1)

    def backward(self, dL_dZ):
        """
        dL_dZ: (batch_size, ...) -> (batch_size, ...)
        """
        X = self.cache["X"]
        return dL_dZ.reshape(X.shape)


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__("dense")
        self.grads = {}
        self.input_size = input_size
        self.output_size = output_size
        # Initialize weights and biases
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)

    def forward(self, X):
        """
        X: (batch_size, input_size) -> (batch_size, output_size)
        """
        self.cache["X"] = X
        Z = np.dot(X, self.W) + self.b
        return Z

    def backward(self, dL_dZ):
        """
        Backpropagate through Dense layer.

        Args:
            dL_dZ: (batch_size, output_size) - gradient of loss w.r.t. output

        Returns:
            dL_dX: (batch_size, input_size) - gradient to pass to previous layer
        """
        X = self.cache["X"]
        m = X.shape[0]  # batch size

        # Gradient w.r.t. weights: (1/m) * X^T @ dL_dZ
        self.grads["W"] = np.dot(X.T, dL_dZ) / m

        # Gradient w.r.t. bias: (1/m) * sum(dL_dZ)
        self.grads["b"] = np.sum(dL_dZ, axis=0) / m

        # Gradient w.r.t. input: dL_dZ @ W^T
        dL_dX = np.dot(dL_dZ, self.W.T)

        return dL_dX
