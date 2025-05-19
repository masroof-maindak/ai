import numpy as np


def cross_entropy_loss(y_one_hot: np.ndarray, y_pred_probs: np.ndarray) -> float:
    N = y_one_hot.shape[0]
    epsilon = 1e-12
    loss = -1 / N * np.sum(y_one_hot * np.log(y_pred_probs + epsilon))
    return loss
