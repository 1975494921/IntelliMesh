import numpy as np
from IntelliMesh.base.operation import MatMul, Add, Sub, Pow, Slice, Reshape, Transpose


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None and self.grad is None:
            self.grad = np.zeros_like(self.data)

        elif grad is not None:
            self.grad = grad

        if self._grad_fn:
            if hasattr(self._grad_fn, 'backward'):
                self._grad_fn.backward(self.grad)
            else:
                self._grad_fn(self.grad)

    @property
    def shape(self):
        return self.data.shape

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self):
        self.grad = None

    @property
    def grad_fn(self):
        return self._grad_fn

    @grad_fn.setter
    def grad_fn(self, value):
        if self.requires_grad:
            self._grad_fn = value

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            # assert self.data.shape == other.data.shape, "MatMul operation requires tensors of the same shape"

            result = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
            result.grad_fn = MatMul(self, other)
            return result
        else:
            raise ValueError("MatMul operation requires a Tensor")

    def __mul__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
            result.grad_fn = Add(self, other)
            return result

        elif isinstance(other, (int, float, complex)):
            result = Tensor(self.data * other, requires_grad=self.requires_grad)
            return result
        else:
            raise ValueError("Mul operation requires a Tensor or a scalar")

    def __add__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
            result.grad_fn = Add(self, other)
            return result

        elif isinstance(other, (int, float, complex)):
            result = Tensor(self.data + other, requires_grad=self.requires_grad)
            return result

        else:
            raise ValueError("Add operation requires a Tensor or a scalar")

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
            result.grad_fn = Sub(self, other)
            return result

        elif isinstance(other, (int, float, complex)):
            result = Tensor(self.data - other, requires_grad=self.requires_grad)
            return result
        else:
            raise ValueError("Sub operation requires a Tensor or a scalar")

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
            result.grad_fn = Add(self, other)
            return result

        elif isinstance(other, (int, float, complex)):
            result = Tensor(self.data / other, requires_grad=self.requires_grad)
            return result
        else:
            raise ValueError("Div operation requires a Tensor or a scalar")

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(other.data / self.data, requires_grad=self.requires_grad or other.requires_grad)
            result.grad_fn = Add(self, other)
            return result

        elif isinstance(other, (int, float, complex)):
            result = Tensor(other / self.data, requires_grad=self.requires_grad)
            return result

        else:
            raise ValueError("Div operation requires a Tensor or a scalar")

    def __repr__(self):
        return f"Tensor({self.data.__repr__()}, requires_grad={self.requires_grad})"

    def __pow__(self, power, modulo=None):
        if isinstance(power, (int, float, complex)):
            result = Tensor(self.data ** power, requires_grad=self.requires_grad)
            result.grad_fn = Pow(self, power)
            return result
        else:
            raise ValueError("Power operation requires a scalar")

    def __getitem__(self, slices):
        obj = Slice(self, slices)

        result = Tensor(obj.forward(), requires_grad=self.requires_grad)
        result.grad_fn = obj
        return result

    def reshape(self, shape):
        reshape = Reshape(self, shape)
        result = Tensor(reshape.forward(), requires_grad=self.requires_grad)
        result.grad_fn = reshape

        return result

    def transpose(self, axes=None):
        obj_operation = Transpose(self, axes=axes)
        result = Tensor(obj_operation.forward(), requires_grad=self.requires_grad)
        result.grad_fn = obj_operation

        return result

    def view(self, shape):
        return self.reshape(shape)
