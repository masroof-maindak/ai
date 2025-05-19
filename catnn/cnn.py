import numpy as np

from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from losses import cross_entropy_loss

from typing import final

CLASS_MAP = {"Abyssinian": 0, "Bengal": 1, "Bombay": 2, "Egyptian": 3, "Russian": 4}


@final
class ShrimpleCNN:
    def __init__(self):
        # Shape = N, 160, 160, 3
        self.conv = Conv2D(in_channels=3, out_channels=4, kernel_size=3)
        self.relu = ReLU()
        self.pool = MaxPool2D(size=2, stride=2)
        # Shape = N, 79, 79, 4
        self.flatten = Flatten()
        # Shape = N, 79 * 79 * 4
        self.fccn = Dense(in_features=4 * 79 * 79, out_features=len(CLASS_MAP))
        # Shape = N, 5

    def forward(self, X):
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

    def backward(self, grad_output, learning_rate: float):
        pass

    def train(
        self, X, y: list[str], epochs: int = 1000, learning_rate: float = 0.05
    ) -> None:
        """
        Train the model on the given data.

        Parameters:
            X: The input data (N, H, W, C)
            y: The labels (N)
            epochs: The number of epochs to train the model.
            learning_rate: The learning rate.
        """

        y_indices = np.array([CLASS_MAP[label] for label in y])
        N = X.shape[0]
        num_classes = len(CLASS_MAP)

        for epoch in range(epochs + 1):
            y_pred_probs = self.forward(X)

            # --- One-Hot Encoding ---
            # Basically, create an array of classes of as many classes you have
            # And set the correct class label to be 1, whereas the others are 0.
            y_one_hot = np.zeros((N, num_classes))
            y_one_hot[np.arange(N), y_indices] = 1

            loss = cross_entropy_loss(y_one_hot, y_pred_probs)

            # TODO: backprop

            return
            if epoch % 10 == 0:
                predictions_indices = np.argmax(y_pred_probs, axis=1)
                accuracy = np.mean(predictions_indices == y_indices)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X):
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
