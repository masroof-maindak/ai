import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # f(x) = 1 / 1 + e^-x


def relu(x):
    return np.maximum(0, x)  # max(0, x)
