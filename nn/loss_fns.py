import numpy as np


def bce(y_pred, y, derivative=False):
    m = y.shape[0]

    if derivative:
            return -(y / y_pred - (1 - y) / (1 - y_pred)) / m

    # A loss function should return high values for for bad predictions, and vice versa
    # https://youtu.be/DPSXVJF5jIs?feature=shared
    return -1 / m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
