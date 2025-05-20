from typing import final

from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from losses import cross_entropy_loss

import numpy as np
from numpy.typing import NDArray


CLASS_MAP = {"Abyssinian": 0, "Bengal": 1, "Bombay": 2, "Egyptian": 3, "Russian": 4}


@final
class ShrimpleCNN:
    def __init__(self):
        # Shape = N, 120, 120, 3
        self.conv = Conv2D(in_channels=3, out_channels=8, kernel_size=3)
        # N, 118, 118, 8
        self.relu = ReLU()
        self.pool = MaxPool2D(size=2, stride=2)
        # Shape = N, 59, 59, 8
        self.flatten = Flatten()
        # Shape = N, 59 * 59 * 8
        self.fccn = Dense(in_features=8 * 59 * 59, out_features=len(CLASS_MAP))
        # Shape = N, 5

    def forward(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Forward pass of the model.

        Parameters:
            X: The input data (N, H, W, C)

        Returns:
            Pre-activations
        """

        X = self.conv.forward(X)
        X = self.relu.forward(X)
        X = self.pool.forward(X)
        X = self.flatten.forward(X)
        probs = self.fccn.forward(X)
        return probs

    def backward(self, y_pred_probs, y_one_hot, learning_rate: float) -> None:
        N = y_pred_probs.shape[0]

        # Elegant evaluation of dL/dZ when Softmax is used w/ Cross-Entropy Loss
        d_loss_dense_z = (y_pred_probs - y_one_hot) / N

        d_loss_flatten_output = self.fccn.backward(d_loss_dense_z, learning_rate)
        d_loss_pool_output = self.flatten.backward(d_loss_flatten_output)
        d_loss_relu_output = self.pool.backward(d_loss_pool_output)
        d_loss_conv_output = self.relu.backward(d_loss_relu_output)
        _ = self.conv.backward(d_loss_conv_output, learning_rate)

    def train(
        self,
        X: NDArray[np.float32],
        y: list[str],
        epochs: int = 1000,
        learning_rate: float = 0.05,
    ) -> None:
        """
        Train the model on the given data.

        Parameters:
            X: The input data (N, H, W, C)
            y: The labels (N)
            epochs: The number of epochs to train the model.
            learning_rate: The learning rate.
        """

        y_indices: NDArray[np.int8] = np.array([CLASS_MAP[label] for label in y])
        N = X.shape[0]
        num_classes = len(CLASS_MAP)

        for epoch in range(epochs + 1):
            y_pred_probs: NDArray[np.float32] = self.forward(X)

            # --- One-Hot Encoding ---

            # Basically, create an array of size x, where x is the number of classes you have,
            # and set the correct class label's corresponding index to 1, and the others to 0
            y_one_hot = np.zeros((N, num_classes))
            y_one_hot[np.arange(N), y_indices] = 1

            loss = cross_entropy_loss(y_one_hot, y_pred_probs)

            self.backward(y_pred_probs, y_one_hot, learning_rate)

            if epoch % 5 == 0:
                pred_indices = np.argmax(y_pred_probs, axis=1)
                accuracy = np.mean(pred_indices == y_indices)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X: NDArray[np.float32]):
        """
        Predict the class of the given data.

        Parameters:
            X: The input data (N, H, W, C)
        """
        y_pred_probs = self.forward(X)
        predictions_indices = np.argmax(y_pred_probs, axis=1)
        CLASS_NAMES = list(CLASS_MAP.keys())
        names: list[str] = [CLASS_NAMES[i] for i in predictions_indices]
        return np.array(names)
