import numpy as np


class Optimizer:
    def __init__(self, name: str) -> None:
        self.name = name

    def step(self, layers):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, alpha=0.01):
        super().__init__("SGD")
        self.alpha = alpha

    def step(self, layers):
        for layer in layers:
            if layer.grads:
                layer.W -= self.alpha * layer.grads["W"]
                layer.b -= self.alpha * layer.grads["b"]


class Adam(Optimizer):
    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__("Adam")
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
