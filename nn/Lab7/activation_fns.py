import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * sigmoid(-x)
    return 1 / (1 + np.exp(-x))  # f(x) = 1 / 1 + e^-x


def relu(Z, derivative=False):
    if derivative:
        return Z > 0
    return np.maximum(0, Z)  # max(0, x)
