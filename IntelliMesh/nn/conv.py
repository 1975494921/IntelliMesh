import numpy as np

from IntelliMesh.base import func as F
from IntelliMesh.base.tensor import Tensor
from .module import Module


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(
            2.0 / (in_channels * kernel_size * kernel_size)
        )
        self.weights = Tensor(weight, requires_grad=True)

        bias = np.zeros(out_channels)
        self.biases = Tensor(bias, requires_grad=True)

    def im2col(self, input_data):
        N, C, H, W = input_data.shape
        H_padded = H + 2 * self.padding
        W_padded = W + 2 * self.padding
        out_h = (H_padded - self.kernel_size) // self.stride + 1
        out_w = (W_padded - self.kernel_size) // self.stride + 1

        if self.padding > 0:
            input_padded = F.pad2d(
                input_data,
                pad_width=[(0, 0), (0, 0),
                           (self.padding, self.padding),
                           (self.padding, self.padding)],
                mode='constant',
                constant_values=0
            )

        else:
            input_padded = input_data

        # Create index arrays for im2col
        i0 = F.repeat(F.arange(self.kernel_size), self.kernel_size)
        i0 = F.tile(i0, C)
        i1 = self.stride * F.repeat(F.arange(out_h), out_w)
        j0 = F.tile(F.arange(self.kernel_size), self.kernel_size * C)
        j1 = self.stride * F.tile(F.arange(out_w), out_h)

        i = F.reshape(i0, (-1, 1)) + F.reshape(i1, (1, -1))
        j = F.reshape(j0, (-1, 1)) + F.reshape(j1, (1, -1))
        k = F.repeat(F.arange(C), self.kernel_size * self.kernel_size)
        k = F.reshape(k, (-1, 1))

        # Ensure indices do not exceed the padded input boundaries
        i = F.clip(i, 0, H_padded - 1)
        j = F.clip(j, 0, W_padded - 1)

        # Extract columns using custom index mapping
        cols = input_padded[:, k.data, i.data, j.data]
        cols = F.transpose(cols, (1, 2, 0))
        cols = F.reshape(cols, (self.kernel_size * self.kernel_size * C, -1))

        return cols

    def forward(self, x):
        N, C, H, W = x.shape
        self.x_shape = x.shape
        self.x_col = self.im2col(x)

        # Reshape weights for matrix multiplication
        W_col = self.weights.reshape((self.out_channels, -1))

        # Perform convolution using matrix multiplication and add biases
        out = W_col @ self.x_col

        # Reshape to output shape: (batch_size, out_channels, out_height, out_width)
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = F.reshape(out, (self.out_channels, out_h, out_w, N))
        out = F.transpose(out, (3, 0, 1, 2))

        return out


class MaxPool2D(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def im2col(self, input_data):
        N, C, H, W = input_data.shape
        H_padded = H + 2 * self.padding
        W_padded = W + 2 * self.padding
        out_h = (H_padded - self.kernel_size) // self.stride + 1
        out_w = (W_padded - self.kernel_size) // self.stride + 1

        # Use custom padding function
        if self.padding > 0:
            input_padded = F.pad2d(input_data,
                                   pad_width=[(0, 0), (0, 0), (self.padding, self.padding),
                                              (self.padding, self.padding)],
                                   mode='constant', constant_values=0)
        else:
            input_padded = input_data

        # Create index arrays for im2col
        i0 = F.repeat(F.arange(self.kernel_size), self.kernel_size)
        i0 = F.tile(i0, C)
        i1 = self.stride * F.repeat(F.arange(out_h), out_w)
        j0 = F.tile(F.arange(self.kernel_size), self.kernel_size * C)
        j1 = self.stride * F.tile(F.arange(out_w), out_h)

        i = F.reshape(i0, (-1, 1)) + F.reshape(i1, (1, -1))
        j = F.reshape(j0, (-1, 1)) + F.reshape(j1, (1, -1))
        k = F.repeat(F.arange(C), self.kernel_size * self.kernel_size)
        k = F.reshape(k, (-1, 1))

        # Ensure indices do not exceed the padded input boundaries
        i = F.clip(i, 0, H_padded - 1)
        j = F.clip(j, 0, W_padded - 1)

        # Extract columns using custom index mapping
        cols = input_padded[:, k.data, i.data, j.data]
        cols = F.transpose(cols, (1, 2, 0))
        cols = F.reshape(cols, (self.kernel_size * self.kernel_size * C, -1))

        return cols

    def forward(self, x):
        N, C, H, W = x.shape
        self.x_shape = x.shape
        self.x_col = self.im2col(x)

        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Reshape x_col to (C * kernel_size * kernel_size, N * out_h * out_w)
        x_col_reshaped = F.reshape(self.x_col, (C, self.kernel_size * self.kernel_size, -1))

        # Max pooling operation
        max_values = F.max(x_col_reshaped, axis=1)

        # Reshape output to (N, C, out_h, out_w)
        out = F.reshape(max_values, (N, C, out_h, out_w))

        return out


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Initialize parameters
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        # Initialize running statistics
        self.running_mean = Tensor(np.zeros((1, num_features, 1, 1)), requires_grad=False)
        self.running_var = Tensor(np.ones((1, num_features, 1, 1)), requires_grad=False)

    def forward(self, x, training=True):
        N, C, H, W = x.shape

        if training:
            batch_mean = F.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = F.var(x, axis=(0, 2, 3), keepdims=True)


            # Update running statistics
            self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * batch_var + (1 - self.momentum) * self.running_var

            # Normalize
            x_hat = (x - batch_mean) / F.sqrt(batch_var + self.eps)
        else:
            # Use running statistics for normalization during inference
            x_hat = (x - self.running_mean) / F.sqrt(self.running_var + self.eps)

        # Scale and shift
        out = self.gamma.reshape((1, C, 1, 1)) * x_hat + self.beta.reshape((1, C, 1, 1))
        return out
