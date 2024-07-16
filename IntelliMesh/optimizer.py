from IntelliMesh.base.tensor import Tensor
from IntelliMesh.nn.module import Module
from .nn import Sequential


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = []
        if isinstance(parameters, Sequential):
            for layer in parameters.layers:
                for attr in dir(layer):
                    if isinstance(getattr(layer, attr), Tensor) and getattr(layer, attr).requires_grad:
                        self.parameters.append(getattr(layer, attr))

        elif isinstance(parameters, Module):
            for attr in dir(parameters):
                if isinstance(getattr(parameters, attr), Tensor) and getattr(parameters, attr).requires_grad:
                    self.parameters.append(getattr(parameters, attr))

        elif isinstance(parameters, Tensor):
            if parameters.requires_grad:
                self.parameters.append(parameters)

        else:
            raise ValueError("Invalid parameter type.")

        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad:
                param.grad = None

    def __repr__(self):
        return f'Optimizer(lr={self.lr})'


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr)

    def step(self):
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                if param.grad.shape != param.data.shape:
                    grad_agg = param.grad.mean(axis=0)
                else:
                    grad_agg = param.grad

                # Update parameter
                param.data -= self.lr * grad_agg

    def __repr__(self):
        return f'SGD(lr={self.lr})'


class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = [0 for _ in range(len(self.parameters))]
        self.v = [0 for _ in range(len(self.parameters))]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                if param.grad.shape != param.data.shape:
                    grad_agg = param.grad.mean(axis=0)
                else:
                    grad_agg = param.grad

                self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grad_agg
                self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * grad_agg ** 2

                m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta_2 ** self.t)

                param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)

    def __repr__(self):
        return f'Adam(lr={self.lr}, beta_1={self.beta_1}, beta_2={self.beta_2}, eps={self.eps})'


class RMSProp(Optimizer):
    def __init__(self, parameters, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta = beta
        self.eps = eps
        self.v = [0 for _ in range(len(self.parameters))]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                # Check if gradient aggregation is needed
                if param.grad.shape != param.data.shape:
                    # Aggregate gradients along the batch dimension (axis=0)
                    grad_agg = param.grad.sum(axis=0)
                else:
                    grad_agg = param.grad

                self.v[i] = self.beta * self.v[i] + (1 - self.beta) * grad_agg ** 2
                param.data -= self.lr * grad_agg / (self.v[i] ** 0.5 + self.eps)

    def __repr__(self):
        return f'RMSProp(lr={self.lr}, beta={self.beta}, eps={self.eps})'
