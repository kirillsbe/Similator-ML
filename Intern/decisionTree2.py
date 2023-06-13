import numpy as np

def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""
    mse = np.sum((np.mean(y) - y)**2) / len(y)
    return mse


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    mseWeighted = (mse(y_left) * len(y_left) + mse(y_right) * len(y_right)) / (len(y_left) + len(y_right))
    return mseWeighted