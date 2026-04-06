import loss as ls
import optimization
from layers import activation as a
from layers import base as b


class Sequential:
    def __init__(
        self,
        *layers,
        alpha=0.01,
        optimizer="adam",
        batch_size=1,
        epochs=1000,
        loss_class=ls.MeanSquaredError(),
    ) -> None:
        """
        Initialize with variable number of layers.

        Usage:
            model = Sequential(
                b.Dense(256, 128),
                a.ReLu(),
                b.Dense(128, 1),
                a.Sigmoid()
            )
        """
        self.layers = list(layers)
        self.alpha = alpha
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_class = loss_class

    def add(self, layer) -> None:
        """Add a layer to the network."""
        self.layers.append(layer)

    def setoptimizer(self, name):
        self.optimizer = name

    def setbatchsize(self, num):
        self.batch_size = num

    def setloss(self, loss_class):
        self.loss_class = loss_class

    def train(self, X, y):
        match self.optimizer:
            case "sgd":
                optimizer = optimization.SGD(alpha=self.alpha)
            case "adam":
                optimizer = optimization.Adam(alpha=self.alpha)
            case _:
                raise ValueError(f"{self.optimizer} is not a valid optimizer.")

        return optimization.train_loop(
            model=self,
            X=X,
            y=y,
            optimizer=optimizer,
            loss_class=self.loss_class,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

    def forward(self, X):
        """
        Forward pass through all layers.

        Args:
            X: input array

        Returns:
            output after passing through all layers
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, gradient):
        """
        Backward pass through all layers.

        Args:
            gradient: dL/dY from loss function (shape: batch_size x output_size)

        Propagates gradient backwards through all layers in reverse order.
        Each layer computes its parameter gradients, updates parameters,
        and returns the gradient for the previous layer.
        """
        # Start with gradient from loss and propagate backwards
        current_gradient = gradient

        # Iterate through layers in reverse order
        for layer in reversed(self.layers):
            # Pass gradient through layer and get gradient for previous layer
            current_gradient = layer.backward(current_gradient)

    def __call__(self, X):
        """Allow model(X) syntax."""
        return self.forward(X)

    def summary(self):
        """Print model architecture."""
        print("Model Summary:")
        print("-" * 60)
        print(
            f"Optimizer: {self.optimizer} | Learning Rate: {self.alpha} | Batch Size: {self.batch_size} \nEpochs: {self.epochs} | Loss: {self.loss_class.name}"
        )
        print("-" * 60)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, b.Dense):
                print(
                    f"Layer {i}: {layer.name.upper():<10} | Input: {layer.input_size:<5} Output: {layer.output_size:<5}"
                )
            else:
                print(f"Layer {i}: {layer.name.upper():<10}")
        print("-" * 60)


class SequentialBuilder:
    """Fluent API for building Sequential models."""

    def __init__(self):
        self.layers = []
        self.alpha_value = 1
        self.optimizer_name = "sgd"
        self.batch_size = 1
        self.epochs_value = 1000
        self.loss_class = ls.MeanSquaredError()

    def flatten(self):
        """Add a Flatten layer."""
        self.layers.append(b.Flatten())
        return self

    def dense(self, input_size, output_size):
        """Add a Dense layer."""
        self.layers.append(b.Dense(input_size, output_size))
        return self

    def relu(self):
        """Add a ReLU activation."""
        self.layers.append(a.ReLu())
        return self

    def sigmoid(self):
        """Add a Sigmoid activation."""
        self.layers.append(a.Sigmoid())
        return self

    def tanh(self):
        """Add a Tanh activation."""
        self.layers.append(a.Tanh())
        return self

    def softmax(self):
        """Add a Softmax activation."""
        self.layers.append(a.Softmax())
        return self

    def elu(self, alpha_activation=1.0):
        """Add an ELU activation."""
        self.layers.append(a.ELU(alpha_activation))
        return self

    def optimizer(self, name):
        """Set the optimizer."""
        self.optimizer_name = name
        return self

    def batch(self, num):
        """Set the batch size."""
        self.batch_size = num
        return self

    def alpha(self, num):
        """Set the learning rate."""
        self.alpha_value = num
        return self

    def epochs(self, num):
        """Set the number of epochs."""
        self.epochs_value = num
        return self

    def loss(self, loss_class):
        """Set the loss function."""
        self.loss_class = loss_class
        return self

    def build(self):
        """Build and return the Sequential model."""
        return Sequential(
            *self.layers,
            alpha=self.alpha_value,
            optimizer=self.optimizer_name,
            batch_size=self.batch_size,
            epochs=self.epochs_value,
            loss_class=self.loss_class,
        )
