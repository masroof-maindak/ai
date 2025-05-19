from typing import final

import numpy as np


@final
class Conv2D:
    def __init__(
        self, in_channels: int = 3, out_channels: int = 4, kernel_size: int = 3
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.kernels = (
            np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * 0.01
        )

        self.stride = 1

        # Padding = 0

        self.input: np.ndarray = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of a 2D convolution.

        Parameters:
            x: Input data of shape (N, H_in, W_in, C_in)
               N: batch size
               H_in: input height
               W_in: input width
               C_in: input channels

        Returns:
            Output data of shape (N, H_out, W_out, C_out)
        """
        self.input = x
        N, H_in, W_in, _ = x.shape

        out_rows = (H_in - self.kernel_size) // self.stride + 1
        out_cols = (W_in - self.kernel_size) // self.stride + 1

        output = np.zeros((N, out_rows, out_cols, self.out_channels))

        for i in range(N):
            print("[ CONV2D FORWARD ] image #", i)
            for c in range(self.out_channels):
                kernel = self.kernels[:, :, :, c]  # Shape: K, K, C_in

                for row_idx in range(out_rows):
                    start_row = row_idx * self.stride
                    end_row = start_row + self.kernel_size

                    for col_idx in range(out_cols):
                        start_col = col_idx * self.stride
                        end_col = start_col + self.kernel_size

                        patch = x[i, start_row:end_row, start_col:end_col, :]
                        output[i, row_idx, col_idx, c] = np.sum(patch * kernel)

        return output


@final
class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Gradient is passed through only where input > 0
        return grad_output * (self.input > 0)


@final
class MaxPool2D:
    def __init__(self, size: int = 2, stride: int = 2):
        self.size = size
        self.stride = stride

        self.input: np.ndarray = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of a 2D max pooling operation.

        Parameters:
            x: Input data of shape (N, H_in, W_in, C_in)
               N: batch size
               H_in: input height
               W_in: input width
               C_in: input channels (depth)

        Returns:
            Output data of shape (N, H_out, W_out, C_in)
            The number of channels remains the same.
        """
        self.input = x
        N, H_in, W_in, C_in = x.shape

        pool_h, pool_w = self.size, self.size

        out_rows = (H_in - pool_h) // self.stride + 1
        out_cols = (W_in - pool_w) // self.stride + 1

        output = np.zeros((N, out_rows, out_cols, C_in))

        for i in range(N):
            for c in range(C_in):
                for row in range(out_rows):
                    start_row = row * self.stride
                    end_row = start_row + pool_h

                    for col in range(out_cols):
                        start_col = col * self.stride
                        end_col = start_col + pool_w

                        patch = x[i, start_row:end_row, start_col:end_col, c]
                        output[i, row, col, c] = np.max(patch)

        return output


@final
class Flatten:
    def __init__(self):
        # Store the original input shape for the backward pass
        self.input_shape: tuple

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Flattens the input tensor, preserving the batch size.

        Parameters:
            x: Input data, e.g., shape (N, H, W, C)

        Returns:
            Flattened data of shape (N, H*W*C)
        """

        self.input_shape = x.shape
        output = x.reshape(x.shape[0], -1)
        return output


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


@final
class Dense:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(out_features, in_features) * 0.01
        self.biases = np.zeros(out_features)

    def forward(self, x):
        self.input = x
        z = x @ self.weights.T + self.biases
        self.output = softmax(z)
        return self.output

    def backward(self):
        pass
