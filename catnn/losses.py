import numpy as np
from numpy.typing import NDArray

from layers import softmax


def cross_entropy_loss(
    y_one_hot: NDArray[np.float64], logits: NDArray[np.float32]
) -> tuple[np.float64, NDArray[np.float32]]:
    N = y_one_hot.shape[0]
    probs = softmax(logits)
    epsilon = 1e-12
    loss = -1 / N * np.sum(y_one_hot * np.log(probs + epsilon))
    return loss, probs
