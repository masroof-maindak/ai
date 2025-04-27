import numpy as np

EPSILON = 1e-12


def mse_deriv(y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
    m = y_pred.shape[0]
    return (y_real - y_pred) / m


def mse(y_pred: np.ndarray, y_real: np.ndarray) -> np.float64:
    m = y_pred.shape[0]
    return np.sum(np.power((y_real - y_pred), 2)) / m


# A loss function should return high values for for bad predictions, and vice versa
# https://youtu.be/DPSXVJF5jIs?feature=shared


def bce(y_pred: np.ndarray, y_real: np.ndarray) -> np.float64:
    m = y_real.shape[0]
    # Clip predictions to prevent division by zero or near-zero
    y_pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    return (
        -1
        / m
        * np.sum(
            y_real * np.log(y_pred_clipped) + (1 - y_real) * np.log(1 - y_pred_clipped)
        )
    )


def bce_deriv(y_pred: np.ndarray, y_real: np.ndarray) -> np.ndarray:
    m = y_real.shape[0]

    y_pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)

    # Calculate the derivative: -(y_true / y_pred - (1 - y_true) / (1 - y_pred))

    derivative = -(
        np.divide(y_real, y_pred_clipped) - np.divide(1 - y_real, 1 - y_pred_clipped)
    )

    return derivative / m
