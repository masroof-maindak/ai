import numpy as np


# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Initialize weights and biases
input_size = 2
hidden_size = 2
output_size = 1

np.random.seed(42)
weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
bias1 = np.zeros((1, hidden_size))
bias2 = np.zeros((1, output_size))

# Training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Training loop
learning_rate = 0.1
num_iterations = 100000
m = X.shape[0]

for i in range(num_iterations):
    # Forward propagation
    z1 = np.dot(X, weights1) + bias1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, weights2) + bias2
    a2 = sigmoid(z2)

    # Compute loss
    loss = np.mean((a2 - y) ** 2)

    # Backward propagation
    dL_da2 = (a2 - y) * sigmoid_derivative(a2)
    dW2 = (a1.T.dot(dL_da2)) / m
    db2 = np.sum(dL_da2, axis=0, keepdims=True) / m

    dL_da1 = dL_da2.dot(weights2.T) * sigmoid_derivative(a1)
    dW1 = (X.T.dot(dL_da1)) / m
    db1 = np.sum(dL_da1, axis=0, keepdims=True) / m

    # Update weights
    weights1 -= learning_rate * dW1
    bias1 -= learning_rate * db1
    weights2 -= learning_rate * dW2
    bias2 -= learning_rate * db2

    if i % 1000 == 0:
        print(f"Loss after {i} iterations: {loss}")

print(f"Final prediction: {a2}")
