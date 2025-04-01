import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

#
# NN Model:
#

#
#
# h
# h o
# h
# h
#
#


class NeuralNetwork:
    def __init__(self):
        # Input layer (8 features) →  Hidden layer (4 neurons)
        self.W1 = np.random.randn(8, 4)  # 4 weights against each feature
        self.b1 = np.ones(4)  # Column vector w/ 4 rows

        # Hidden layer (4 neurons) →  Output layer (1 neuron)
        self.W2 = np.random.randn(4, 1)  # 1 weight against output of the hidden layer
        self.b2 = np.ones(1)  # Single neuron therefore the bias vector's size is 1

    # param X array of training samples ; shape -> (NUM_SAMPLES, 8)
    def forward_pass(self, X):
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
    def back_propagation(self, X, y, learning_rate):
        m = X.shape[0]  # NUM_SAMPLES

        # Output layer error
        # dZ2 =  # Hint: Derivative of binary cross-entropy loss (error at the output layer)
        # dW2 =  # Hint: Use np.dot with proper dimensions (for output layer)
        # db2 =  # Hint: Sum along axis 0 (for output layer)

        # Hidden layer error (ReLU derivative is 1 where z1 > 0)
        # dZ1 =
        # dW1 =
        # db1 =

        # Update weights and biases
        # self.W1 -=
        # self.b1 -=
        # self.W2 -=
        # self.b2 -=

        return X, y, learning_rate

    # param X array of training samples ; shape -> (NUM_SAMPLES, 8)
    # param y array of labels
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward_pass(X).flatten()
            # NOTE: flatten() is used to convert the 2D array (column vector) into a 1D array
            # Failing to do so results in an absurdly inflated loss value for some reason
            loss = loss_fns.bce(y_pred, y)
            self.back_propagation(X, y, learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # param X array of testing samples ; shape -> (NUM_SAMPLES, 8)
    def predict(self, X, threshold=0.5):
        y_pred = self.forward_pass(X)
        return np.where(y_pred > threshold, 1, 0)


def main():
    # Initialize and train
    nn = NeuralNetwork()
    nn.train(X_train, y_train, epochs=751, learning_rate=0.01)

    # Predict and calculate accuracy
    y_pred = nn.predict(X_test)
    accuracy = np.mean(y_pred.flatten() == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
