import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import activation_fns as actvns
import loss_fns

data = pd.read_csv("diabetes.csv")

X = data.iloc[:, :-1].values  # Features -- Shape: (768, 8)
y = data.iloc[:, -1].values  # Labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
NN Model:

          f1 -> i
          f2 -> i
          f3 -> i h
Sample #u f4 -> i h o
          f5 -> i h
          f6 -> i h
          f7 -> i
          f8 -> i

"""

INPUT_SIZE = 8
HIDDEN_SIZE = 4
OUTPUT_SIZE = 1


class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)

        # Input layer (8 features) →  Hidden layer (4 neurons)
        self.W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE)
        self.b1 = np.ones((1, HIDDEN_SIZE))

        # Hidden layer (4 neurons) →  Output layer (1 neuron)
        self.W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE)
        self.b2 = np.ones((1, OUTPUT_SIZE))

    # param X array of training samples ; shape -> (NUM_SAMPLES, 8)
    def _forward_pass(self, X):
        # --- HIDDEN LAYER ---
        # Input to next layer i.e weighted sum + bias
        self.Z1 = (X @ self.W1) + self.b1

        # ReLU activation AKA hidden layer's output
        # Idea: we want to propagate forth what *each* nueron 'graded' all the input
        # samples, where one 'input sample' comprises 8 features
        self.A1 = actvns.relu(self.Z1)  # shape -> (NUM_SAMPLES, 4)

        # --- OUTPUT LAYER ---
        # shape -> (NUM_SAMPLES, 1) i.e the 'grades' for each of the inputs from the single output neuron
        self.z2 = (self.A1 @ self.W2) + self.b2

        # Apply a sigmoid activation function to aforementioned 'grades' to squish the output
        # between 0 and 1
        self.a2 = actvns.sigmoid(self.z2)

        return self.a2

    # https://youtu.be/YG15m2VwSjA?feature=shared
    # https://youtu.be/tIeHLnjs5U8?feature=shared
    def _back_propagation(self, X, y_real, alpha):
        m: int = X.shape[0]  # NUM_SAMPLES

        # Any layer's output is defined by three things. The prior layer's
        # activations, the weights of this layer, and the biases of this layer.
        # The idea is that we want to see *how* tweaks to either of those things
        # would affect the end goal of the cost function. This is why we start
        # from the end and 'propagate' back; the part regarding the 'activations
        # of the previous layer' lends itself quite naturally to this recursion

        # In our case, we don't need a recursive function for a mere two layers.

        # --- OUTPUT LAYER ---

        # derivative of loss function i.e cost w.r.t activations of final layer
        d_loss_a2 = loss_fns.bce_deriv(self.a2.flatten(), y_real).reshape(
            -1, 1  # convert the response from a list back to a column vector
        )

        # derivative of a2 w.r.t z2 -> NUM_SAMPLES x 1
        d_a2_z2 = actvns.sigmoid(self.z2, derivative=True)

        # activations of prior layer (transposed) -> 4 x NUM_SAMPLES
        d_z2_W2 = self.A1.T

        """
        NOTE: uncomment out the two lines below and comment the line below them
        to instead get an elegant evaluation of the BCE loss' derivative specifically
        when it is applied in tandem with 
        """
        # tmp = (self.a2.flatten() - y_real).reshape(-1, 1)
        # common1 = tmp * actvns.sigmoid(self.a2, derivative=True)
        common1 = d_loss_a2 * d_a2_z2

        # How sensitive the cost function is with respect to the weights/biases
        d_loss_W2 = (d_z2_W2 @ common1) / m

        """
        the 'third component' being multiplied with the intermediary in this case
        (that of determining the sensitivity w.r.t to the loss) is effectively 1.

        This is because of the derivation of zL with respect to bL is always 1.
        """

        d_loss_b2 = np.sum(common1, axis=0, keepdims=True) / m

        # --- HIDDEN LAYER ---

        # Chain rule
        d_loss_a1 = common1 @ self.W2.T  # CHECK: what's going on here?
        d_a1_z1 = actvns.relu(self.Z1, derivative=True)
        d_z1_W1 = X.T

        common2 = d_loss_a1 * d_a1_z1

        d_loss_W1 = (d_z1_W1 @ common2) / m
        d_loss_b1 = np.sum(common2, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W2 -= alpha * d_loss_W2
        self.b2 -= alpha * d_loss_b2
        self.W1 -= alpha * d_loss_W1
        self.b1 -= alpha * d_loss_b1

    # param X array of training samples ; shape -> (NUM_SAMPLES, 8)
    # param y array of labels
    def train(self, X, y_real, epochs, alpha):
        for epoch in range(epochs + 1):
            y_pred = self._forward_pass(X).flatten()

            """
            NOTE: flatten() is used to convert the 2D array
            i.e of outputs(column vector) into a 1D array
            """

            loss = loss_fns.bce(y_pred, y_real)
            self._back_propagation(X, y_real, alpha)
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # param X array of testing samples ; shape -> (NUM_SAMPLES, 8)
    # returns a c olumn vector of 1s or 0s denoting binary classification
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
