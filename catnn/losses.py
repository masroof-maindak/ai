import numpy as np
from numpy.typing import NDArray


def cross_entropy_loss(
    y_one_hot: NDArray[np.float64], y_pred_probs: NDArray[np.float32]
) -> np.float64:
    N = y_one_hot.shape[0]
    epsilon = 1e-12
    loss = -1 / N * np.sum(y_one_hot * np.log(y_pred_probs + epsilon))
    return loss
