import numpy as np


def sigmoid(x, derivative=False) -> np.ndarray:
    sig = 1 / (1 + np.exp(-x))  # f(x) = 1 / 1 + e^-x
    return sig if not derivative else sig * (1 - sig)


def relu(Z: np.ndarray, derivative=False):
    return Z > 0 if derivative else np.maximum(0, Z)
