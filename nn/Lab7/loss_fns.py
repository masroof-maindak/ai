# A loss function should return high values for for bad predictions, and vice versa

import numpy as np

EPSILON = 1e-12


def bce(y_pred: np.ndarray, y_real: np.ndarray) -> np.float64:
    m = y_real.shape[0]

    """
    # https://youtu.be/DPSXVJF5jIs?feature=shared
    # NOTE: predictions are clipped to avoid logarithm of zero
    """

    y_pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    return (
        -1
        / m
        * np.sum(
            y_real * np.log(y_pred_clipped) + (1 - y_real) * np.log(1 - y_pred_clipped)
        )
    )


def bce_deriv(y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
    y_pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    return -(
        np.divide(y_real, y_pred_clipped) - np.divide(1 - y_real, 1 - y_pred_clipped)
    )
