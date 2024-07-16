import numpy as np

from IntelliMesh.base import func as F
from IntelliMesh.base.tensor import Tensor
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.x = None

        # init with xavier
        weight = np.random.randn(in_features, out_features) * np.sqrt(2 / (in_features + out_features))
        self.weights = Tensor(weight, requires_grad=True)

        bias = np.zeros(out_features)
        self.biases = Tensor(bias, requires_grad=True)

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.biases

    def __call__(self, x):
        return self.forward(x)


class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)


class Flatten(Module):
    def __init__(self):
        pass

    def forward(self, x):
        return F.flatten(x)
