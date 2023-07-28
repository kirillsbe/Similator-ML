import numpy as np
from typing import Tuple


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error loss function and gradient."""
    # YOUR CODE HERE
    loss = np.mean((y_true - y_pred)**2)
    grad = (y_pred - y_true) 
    return loss, grad


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean absolute error loss function and gradient."""
    # YOUR CODE HERE
    loss = np.mean(np.abs(y_true - y_pred))
    grad = (y_pred - y_true)
    grad[grad > 0] = 1
    grad[grad < 0] = -1
    return loss, grad