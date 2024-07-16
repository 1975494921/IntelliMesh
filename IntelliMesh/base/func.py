from abc import abstractmethod

import numpy as np

from .tensor import Tensor


class Operation:
    @abstractmethod
    def forward(self):
        raise NotImplemented

    @abstractmethod
    def backward(self, *args, **kwargs):
        raise NotImplemented

    def __call__(self):
        return self.forward()


class sqrt_obj(Operation):
    def __init__(self, x):
        self.x = x

    def forward(self):
        return np.sqrt(self.x.data)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = 1 / (2 * np.sqrt(self.x.data))
            self.x.backward(grad_x)


def sqrt(x):
    obj_operation = sqrt_obj(x)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class mean_obj(Operation):
    def __init__(self, x, axis=None, keepdims=False):
        self.x = x
        self.axis = axis
        self.keepdims = keepdims

    def forward(self):
        return np.mean(self.x.data, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = np.full(self.x.data.shape, grad / np.prod(self.x.data.shape))
            self.x.backward(grad_x)


def mean(x, axis=None, keepdims=False):
    obj_operation = mean_obj(x, axis=axis, keepdims=keepdims)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class var_obj(Operation):
    def __init__(self, x, axis=None, keepdims=False):
        self.x = x
        self.axis = axis
        self.keepdims = keepdims

    def forward(self):
        return np.var(self.x.data, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = np.full(self.x.data.shape, grad / np.prod(self.x.data.shape))
            self.x.backward(grad_x)


def var(x, axis=None, keepdims=False):
    obj_operation = var_obj(x, axis=axis, keepdims=keepdims)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class sum_obj(Operation):
    def __init__(self, x, axis=None, keepdims=False):
        self.x = x
        self.axis = axis
        self.keepdims = keepdims

    def forward(self):
        return np.sum(self.x.data, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = np.full(self.x.data.shape, grad)
            self.x.backward(grad_x)


def sum(x, axis=None, keepdims=False):
    obj_operation = sum_obj(x, axis=axis, keepdims=keepdims)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class pad_obj(Operation):
    def __init__(self, x, pad_width, mode='constant', constant_values=0):
        super().__init__()  # Initialize the superclass if there's any
        self.x = x
        self.pad_width = pad_width
        self.mode = mode
        self.constant_values = constant_values

    def forward(self):
        self.output = np.pad(
            self.x.data,
            pad_width=self.pad_width,
            mode=self.mode,
            constant_values=self.constant_values
        )

        return self.output

    def backward(self, grad):
        if self.x.requires_grad:
            pad_top, pad_bottom = self.pad_width[2][0], self.pad_width[2][1]
            pad_left, pad_right = self.pad_width[3][0], self.pad_width[3][1]
            grad_x = grad[
                     :,
                     :,
                     pad_top: -pad_bottom if pad_bottom != 0 else None,
                     pad_left: -pad_right if pad_right != 0 else None]

            self.x.backward(grad_x)


def pad2d(x, pad_width, mode='constant', constant_values=0):
    obj_operation = pad_obj(x, pad_width=pad_width, mode=mode, constant_values=constant_values)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class repeat_obj(Operation):
    def __init__(self, x, repeats, axis=None):
        self.x = x
        self.repeats = repeats
        self.axis = axis

    def forward(self):
        return np.repeat(self.x.data, repeats=self.repeats, axis=self.axis)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = np.sum(grad, axis=self.axis)
            self.x.backward(grad_x)


def repeat(x, repeats, axis=None):
    obj_operation = repeat_obj(x, repeats=repeats, axis=axis)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class arange_obj(Operation):
    def __init__(self, start, stop=None, step=1):
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self):
        return np.arange(self.start, self.stop, self.step)

    def backward(self, grad):
        pass


def arange(start, stop=None, step=1):
    obj_operation = arange_obj(start=start, stop=stop, step=step)
    result = Tensor(obj_operation.forward(), requires_grad=False)
    result.grad_fn = obj_operation

    return result


class transpose_obj(Operation):
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


def transpose(x, axes=None):
    obj_operation = transpose_obj(x, axes=axes)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class reshape_obj(Operation):
    def __init__(self, x, shape):
        self.x = x
        self.shape = shape

    def forward(self):
        return np.reshape(self.x.data, self.shape)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = np.reshape(grad, self.x.data.shape)
            self.x.backward(grad_x)


def reshape(x, shape):
    obj_operation = reshape_obj(x, shape)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class clip_obj(Operation):
    def __init__(self, x, a_min, a_max):
        self.x = x
        self.a_min = a_min
        self.a_max = a_max

    def forward(self):
        return np.clip(self.x.data, self.a_min, self.a_max)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = grad
            self.x.backward(grad_x)


def clip(x, a_min, a_max):
    obj_operation = clip_obj(x, a_min=a_min, a_max=a_max)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class expand_dims_obj(Operation):
    def __init__(self, x, axis):
        self.x = x
        self.axis = axis

    def forward(self):
        return np.expand_dims(self.x.data, axis=self.axis)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = np.squeeze(grad, axis=self.axis)
            self.x.backward(grad_x)


def expand_dims(x, axis):
    obj_operation = expand_dims_obj(x, axis=axis)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class squeeze_obj(Operation):
    def __init__(self, x, axis):
        self.x = x
        self.axis = axis

    def forward(self):
        return np.squeeze(self.x.data, axis=self.axis)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = np.expand_dims(grad, axis=self.axis)
            self.x.backward(grad_x)


def squeeze(x, axis):
    obj_operation = squeeze_obj(x, axis=axis)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class tile_obj(Operation):
    def __init__(self, x, reps):
        self.x = x
        self.reps = reps

    def forward(self):
        return np.tile(self.x.data, reps=self.reps)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = np.sum(grad, axis=0)
            self.x.backward(grad_x)


def tile(x, reps):
    obj_operation = tile_obj(x, reps=reps)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class flatten_obj(Operation):
    def __init__(self, x):
        self.x = x
        self.orig_shape = x.data.shape

    def forward(self):
        batch_size = self.orig_shape[0]
        return self.x.data.reshape(batch_size, -1)

    def backward(self, grad):
        if self.x.requires_grad:
            # Reshape the gradient to match the original input shape
            grad_x = grad.reshape(self.orig_shape)
            self.x.backward(grad_x)


def flatten(x):
    obj_operation = flatten_obj(x)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class max_obj(Operation):
    def __init__(self, x, axis):
        self.x = x
        self.axis = axis
        self.input_shape = x.data.shape

        self.max_values = None
        self.max_indices = None

    def forward(self):
        self.max_values = np.max(self.x.data, axis=self.axis, keepdims=True)
        self.max_indices = (self.x.data == self.max_values)

        return np.squeeze(self.max_values, axis=self.axis)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_expanded = np.expand_dims(grad, axis=self.axis)
            grad_expanded = np.broadcast_to(grad_expanded, self.max_indices.shape)
            grad_x = np.zeros_like(self.x.data)
            grad_x[self.max_indices] = grad_expanded[self.max_indices]
            self.x.backward(grad_x)


def max(x, axis=None):
    obj_operation = max_obj(x, axis=axis)
    result = Tensor(obj_operation.forward(), requires_grad=x.requires_grad)
    result.grad_fn = obj_operation

    return result


class argmax_obj(Operation):
    def __init__(self, x, axis=None):
        self.x = x
        self.axis = axis

    def forward(self):
        return np.argmax(self.x.data, axis=self.axis)

    def backward(self, grad):
        return np.zeros(self.x.data.shape)


def argmax(x, axis=None):
    obj_operation = argmax_obj(x, axis=axis)
    result = Tensor(obj_operation.forward(), requires_grad=False)
    result.grad_fn = obj_operation

    return result
