import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def mse_deriv(y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_real)


def mse(y_pred: np.ndarray, y_real: np.ndarray) -> np.float64:
    m = y_pred.shape[0]
    return np.sum(np.power((y_real - y_pred), 2)) / m


def sigmoid(x: np.ndarray, derivative=False) -> np.ndarray:
    sig = 1 / (1 + np.exp(-x))  # f(x) = 1 / 1 + e^-x
    return sig if not derivative else sig * (1 - sig)


def relu(Z: np.ndarray, derivative=False):
    return Z > 0 if derivative else np.maximum(0, Z)


data = pd.read_csv("pima-indians-diabetes.data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

INPUT_SIZE = 8
HIDDEN_SIZE = 4
OUTPUT_SIZE = 1


class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE)
        self.b1 = np.zeros((1, HIDDEN_SIZE))
        self.W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE)
        self.b2 = np.zeros((1, OUTPUT_SIZE))

    def _forward_pass(self, X):
        self.Z1 = (X @ self.W1) + self.b1
        self.A1 = relu(self.Z1)  # shape -> (NUM_SAMPLES, 4)
        self.z2 = (self.A1 @ self.W2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def _back_propagation(self, X, y_real, alpha):
        m: int = X.shape[0]  # NUM_SAMPLES

        d_loss_a2 = mse_deriv(self.a2.flatten(), y_real).reshape(
            -1, 1  # convert the response from a list back to a column vector
        )

        d_a2_z2 = sigmoid(self.z2, derivative=True)

        d_z2_W2 = self.A1.T

        delta2 = d_loss_a2 * d_a2_z2

        d_loss_W2 = (d_z2_W2 @ delta2) / m
        d_loss_b2 = np.sum(delta2, axis=0, keepdims=True) / m

        d_loss_a1 = delta2 @ self.W2.T
        d_a1_z1 = relu(self.Z1, derivative=True)
        d_z1_W1 = X.T

        delta1 = d_loss_a1 * d_a1_z1

        d_loss_W1 = (d_z1_W1 @ delta1) / m
        d_loss_b1 = np.sum(delta1, axis=0, keepdims=True) / m

        self.W2 -= alpha * d_loss_W2
        self.b2 -= alpha * d_loss_b2
        self.W1 -= alpha * d_loss_W1
        self.b1 -= alpha * d_loss_b1

    def train(self, X, y_real, epochs, alpha):
        for epoch in range(epochs + 1):
            y_pred = self._forward_pass(X).flatten()

            loss = mse(y_pred, y_real)
            self._back_propagation(X, y_real, alpha)
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X, threshold=0.5):
        y_pred = self._forward_pass(X)
        return np.where(y_pred > threshold, 1, 0)


def main():
    nn = NeuralNetwork()
    nn.train(X_train, y_train, epochs=10000, alpha=0.01)
    y_pred = nn.predict(X_test)
    accuracy = np.mean(y_pred.flatten() == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
