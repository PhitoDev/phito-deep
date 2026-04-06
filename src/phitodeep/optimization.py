import numpy as np


class Optimizer:
    def step(self, layers):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def step(self, layers):
        for layer in layers:
            if layer.grads:
                layer.W -= self.alpha * layer.grads["W"]
                layer.b -= self.alpha * layer.grads["b"]


class Adam(Optimizer):
    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if layer.grads:
                for param_name, g in layer.grads.items():
                    key = (id(layer), param_name)

                    if key not in self.m:
                        self.m[key] = np.zeros_like(g)
                        self.v[key] = np.zeros_like(g)

                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g**2

                    m_hat = self.m[key] / (1 - self.beta1**self.t)
                    v_hat = self.v[key] / (1 - self.beta2**self.t)

                    param = getattr(layer, param_name)
                    param -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)


def train_loop(model, X, y, loss_class, optimizer, epochs=1000, batch_size=1):
    losses = []

    for epoch in range(epochs):
        for _ in range(len(X) // batch_size):
            indices = np.random.randint(0, len(X), batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # Forward pass
            y_pred = model.forward(X_batch)
            loss = loss_class.loss_func(y_pred, y_batch)

            # Compute gradient of loss w.r.t. predictions (dy)
            dy = loss_class.loss_gradient(y_pred, y_batch)

            # Backward pass with gradient of loss
            model.backward(dy)
            optimizer.step(model.layers)

        y_pred = model.forward(X)
        loss = loss_class.loss_func(y_pred, y)
        losses.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return losses
