from typing import final

import numpy as np
from numpy.typing import NDArray


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
        self.d_loss_kernels = np.zeros_like(self.kernels)

        self.stride = 1

        # Padding = 0

        self.input: NDArray[np.float32] = np.array([])

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
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
            # print("[ CONV2D FORWARD ] image #", i)
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

    def backward(
        self, d_loss_conv_output: NDArray[np.float32], learning_rate: float
    ) -> NDArray[np.float32]:
        """
        Performs the backward pass for the Conv2D layer.
        Updates kernel weights and computes the derivative of the loss w.r.t. the input.

        Parameters:
            d_loss_conv_output: Derivative of the loss w.r.t. the output of this Conv layer.
                                Shape: (N, H_out, W_out, C_out)
            learning_rate: Learning rate for parameter updates.

        Returns:
            d_loss_conv_input: Derivative of the loss w.r.t. the input of this Conv layer.
                               Shape: (N, H_in, W_in, C_in)
        """
        N, _, _, _ = self.input.shape
        _, out_rows, out_cols, _ = d_loss_conv_output.shape

        d_loss_conv_input = np.zeros_like(self.input)
        new_d_loss_kernels = np.zeros_like(self.kernels)

        for i in range(N):
            for c in range(self.out_channels):
                for row in range(out_rows):
                    start_row = row * self.stride
                    end_row = start_row + self.kernel_size

                    for col in range(out_cols):
                        start_col = col * self.stride
                        end_col = start_col + self.kernel_size

                        patch = self.input[i, start_row:end_row, start_col:end_col, :]

                        d_loss_conv_out_single: NDArray[np.float32] = (
                            d_loss_conv_output[i, row, col, c]
                        )

                        # What do we want to find?
                        # How much a specific kernel contributed to a specific output element (pixel)

                        # Intuition: K_ij was multiplied by a corresponding patch from the input P_ij
                        # Thus, the 'influence' of K_ij on said element is proportional to that of P_ij
                        new_d_loss_kernels[:, :, :, c] += patch * d_loss_conv_out_single

                        # What do we want to find?
                        # How much a specific input element contributed to a specific output element.

                        # Intuition: The pixel X_mn, as part of some input patch, was multiplied w/ a corresponding
                        # kernel weight K_rs. Therefore, just like above, we can claim that the sensitivity of the
                        # output element w.r.t X_mn is directly proportional to the sensitivity of the element in
                        # question w.r.t K_rs

                        # As an input pixel X_mn influences multiple elements, it's 'alteration' is the summation of
                        # 'contributions' from all the output elements it ultimately 'influenced'.
                        d_loss_conv_input[
                            i, start_row:end_row, start_col:end_col, :
                        ] += (self.kernels[:, :, :, c] * d_loss_conv_out_single)

        self.d_loss_kernels = new_d_loss_kernels
        self.kernels -= learning_rate * self.d_loss_kernels
        return d_loss_conv_input


@final
class ReLU:
    def __init__(self):
        self.input: NDArray[np.float32] = np.array([])

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_loss_relu_output: NDArray[np.float32]) -> NDArray[np.float32]:
        d_loss_relu_input = d_loss_relu_output * (self.input > 0)
        return d_loss_relu_input


@final
class MaxPool2D:
    def __init__(self, size: int = 2, stride: int = 2):
        self.size = size
        self.stride = stride
        self.input: NDArray[np.float32] = np.array([])

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
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

    def backward(
        self, d_loss_maxpool_output: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Intuition: only alter the value that was selected and propagated forward in the first place,
        i.e the maximum, because only it could affect the outcome anyway.
        """
        N, _, _, C_in = self.input.shape
        _, out_rows, out_cols, _ = d_loss_maxpool_output.shape

        pool_h, pool_w = self.size, self.size

        d_loss_maxpool_input = np.zeros_like(self.input)

        for i in range(N):
            for c in range(C_in):
                for row in range(out_rows):
                    start_row = row * self.stride
                    end_row = start_row + pool_h

                    for col in range(out_cols):
                        start_col = col * self.stride
                        end_col = start_col + pool_w

                        patch = self.input[i, start_row:end_row, start_col:end_col, c]

                        max_val = np.max(patch)

                        mask = patch == max_val

                        d_loss_maxpool_input[
                            i, start_row:end_row, start_col:end_col, c
                        ] += (mask * d_loss_maxpool_output[i, row, col, c])
        return d_loss_maxpool_input


@final
class Flatten:
    def __init__(self):
        self.input_shape: tuple[int, ...] = tuple()

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
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

    def backward(self, d_loss_flatten_out: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Basically just reshapes the output to the shape it received
        """
        d_loss_flatten_input = d_loss_flatten_out.reshape(self.input_shape)
        return d_loss_flatten_input


def softmax(X: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


@final
class Dense:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.weights: NDArray[np.float64] = (
            np.random.randn(out_features, in_features) * 0.01
        )
        self.biases: NDArray[np.float64] = np.zeros(out_features)
        self.input: NDArray[np.float32] = np.array([])
        self.output: NDArray[np.float32] = np.array([])

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        self.input = x
        z = x @ self.weights.T + self.biases
        self.output = softmax(z)
        return self.output

    def backward(
        self, d_loss_z: NDArray[np.float32], learning_rate: float
    ) -> NDArray[np.float64]:
        """
        Parameters:
            d_loss_z: Derivative of the loss w.r.t. the pre-softmax activations (z).
                      Shape: (N, out_features)
            learning_rate: Learning rate for parameter updates.

        Returns:
            d_loss_input: Derivative of the loss w.r.t. the input of this layer.
                          Shape: (N, in_features)
        """
        d_loss_w = d_loss_z.T @ self.input
        d_loss_b = np.sum(d_loss_z, axis=0)
        d_loss_input = d_loss_z @ self.weights

        self.weights -= learning_rate * d_loss_w
        self.biases -= learning_rate * d_loss_b

        return d_loss_input
