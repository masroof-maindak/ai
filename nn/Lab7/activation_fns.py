import numpy as np


def sigmoid(x, derivative=False) -> np.ndarray:
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))  # f(x) = 1 / 1 + e^-x


def relu(Z: np.ndarray, derivative=False):
    if derivative:
        return Z > 0
    return np.maximum(0, Z)  # max(0, x)
