from abc import abstractmethod

import numpy as np


class Operation:
    @abstractmethod
    def forward(self):
        raise NotImplemented

    @abstractmethod
    def backward(self, grad):
        raise NotImplemented


class MatMul(Operation):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def forward(self):
        return self.x.data @ self.y.data

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = grad @ self.y.data.T
            self.x.backward(grad_x)

        if self.y.requires_grad:
            grad_y = self.x.data.T @ grad
            self.y.backward(grad_y)


class Add(Operation):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def forward(self):
        return self.x.data + self.y.data

    def backward(self, grad):
        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = grad
            else:
                self.x.grad += grad
            if self.x.grad_fn:
                self.x.grad_fn.backward(grad)

        if self.y.requires_grad:
            if self.y.grad is None:
                self.y.grad = grad
            else:
                self.y.grad += grad
            if self.y.grad_fn:
                self.y.grad_fn.backward(grad)


class Sub(Operation):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def forward(self):
        return self.x.data - self.y.data

    def backward(self, grad):
        if self.x.requires_grad:
            if self.x.grad is None:
                self.x.grad = grad
            else:
                self.x.grad += grad
            if self.x.grad_fn:
                self.x.grad_fn.backward(grad)

        if self.y.requires_grad:
            if self.y.grad is None:
                self.y.grad = grad
            else:
                self.y.grad += grad
            if self.y.grad_fn:
                self.y.grad_fn.backward(-grad)


class Pow(Operation):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def forward(self):
        return self.x.data ** self.y.data

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = self.y.data * (self.x.data ** (self.y.data - 1)) * grad
            self.x.backward(grad_x)

        if self.y.requires_grad:
            grad_y = (self.x.data ** self.y.data) * np.log(self.x.data) * grad
            self.y.backward(grad_y)


class Slice(Operation):
    def __init__(self, input_tensor, slices):
        self.input_tensor = input_tensor
        self.slices = slices

    def forward(self):
        return self.input_tensor.data[self.slices]

    def backward(self, grad):
        if self.input_tensor.requires_grad:
            grad_input = np.zeros(self.input_tensor.shape)
            grad_input[self.slices] = grad
            self.input_tensor.backward(grad_input)


class Reshape(Operation):
    def __init__(self, x, shape):
        self.x = x
        self.shape = shape

    def forward(self):
        return np.reshape(self.x.data, self.shape)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = np.reshape(grad, self.x.data.shape)
            self.x.backward(grad_x)


class Transpose(Operation):
    def __init__(self, x, axes=None):
        self.x = x
        self.axes = axes
        self.inverse_axes = np.argsort(axes)

    def forward(self):
        return np.transpose(self.x.data, axes=self.axes)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = np.transpose(grad, axes=self.inverse_axes)
            self.x.backward(grad_x)
