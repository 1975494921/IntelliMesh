import numpy as np

from IntelliMesh.base.operation import Operation
from IntelliMesh.base.tensor import Tensor


# Modify the sigmoid function to use the Sigmoid class
def Sigmoid(x):
    class SIGMOID(Operation):
        def __init__(self, x):
            self.x = x
            self.out = None

        def forward(self):
            self.out = 1 / (1 + np.exp(-self.x.data))
            return self.out

        def backward(self, grad):
            # The gradient of sigmoid(x) w.r.t x is sigmoid(x) * (1 - sigmoid(x))
            sig_grad = self.out * (1 - self.out)
            grad_output = grad * sig_grad  # Chain rule
            if self.x.requires_grad:
                self.x.backward(grad_output)

    """
    Applies the sigmoid activation function on each element of the input Tensor.
    """
    if not isinstance(x, Tensor):
        raise ValueError("Sigmoid function requires a Tensor object as input.")

    # Use the Sigmoid operation class
    sigmoid_operation = SIGMOID(x)
    sigmoid_data = sigmoid_operation.forward()

    result = Tensor(sigmoid_data, requires_grad=x.requires_grad)
    if x.requires_grad:
        result.grad_fn = sigmoid_operation  # Attach the operation instance

    return result


def ReLu(x):
    class RELU(Operation):
        def __init__(self, x):
            self.x = x
            self.out = None

        def forward(self):
            self.out = np.maximum(0, self.x.data)
            return self.out

        def backward(self, grad):
            # The gradient of ReLU is 1 where the input is positive, 0 otherwise
            relu_grad = (self.x.data > 0).astype(float)
            grad_output = grad * relu_grad  # Chain rule
            if self.x.requires_grad:
                self.x.backward(grad_output)

    """
    Applies the ReLU activation function on each element of the input Tensor.
    """
    if not isinstance(x, Tensor):
        raise ValueError("ReLU function requires a Tensor object as input.")

    relu_operation = RELU(x)
    relu_data = relu_operation.forward()

    result = Tensor(relu_data, requires_grad=x.requires_grad)
    if x.requires_grad:
        result.grad_fn = relu_operation

    return result


def Tanh(x):
    class TANH(Operation):
        def __init__(self, x):
            self.x = x
            self.out = None

        def forward(self):
            self.out = np.tanh(self.x.data)
            return self.out

        def backward(self, grad):
            grad_input = grad * (1 - self.out ** 2)
            if self.x.requires_grad:
                self.x.backward(grad_input)

    if not isinstance(x, Tensor):
        raise ValueError("Tanh function requires a Tensor object as input.")

    tanh_operation = TANH(x)
    tanh_data = tanh_operation.forward()

    result = Tensor(tanh_data, requires_grad=x.requires_grad)
    if x.requires_grad:
        result.grad_fn = tanh_operation

    return result


def Softmax(x):
    class SOFTMAX(Operation):
        def __init__(self, x):
            self.x = x
            self.out = None

        def forward(self):
            e_x = np.exp(self.x.data - np.max(self.x.data, axis=-1, keepdims=True))
            self.out = e_x / np.sum(e_x, axis=-1, keepdims=True)
            return self.out

        def backward(self, grad):
            s = self.out  # Shape (batch_size, num_classes) e.g., (128, 2)
            grad_input = np.zeros_like(s)

            for i, (single_output, single_grad) in enumerate(zip(s, grad)):
                s_vector = single_output.reshape(-1, 1)
                jacobian_matrix = np.diagflat(s_vector) - np.dot(s_vector, s_vector.T)

                grad_input[i] = np.dot(jacobian_matrix, single_grad)

            if self.x.requires_grad:
                self.x.backward(grad_input)

    if not isinstance(x, Tensor):
        raise ValueError("Softmax function requires a Tensor object as input.")

    softmax_operation = SOFTMAX(x)
    softmax_data = softmax_operation.forward()

    result = Tensor(softmax_data, requires_grad=x.requires_grad)
    if x.requires_grad:
        result.grad_fn = softmax_operation

    return result
