from __future__ import annotations
import numpy as np
import sys


def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""
    mse = np.sum((np.mean(y) - y)**2) / len(y)
    return mse


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    mseWeighted = (mse(y_left) * len(y_left) + mse(y_right) * len(y_right)) / (len(y_left) + len(y_right))
    return mseWeighted
    
    
def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Find the best split for a node (one feature)"""
    mse_min = sys.float_info.max
    for i in X[:, feature]:
        y_left, y_right = y[X[:, feature] > i], y[X[:, feature] <= i]
        mse = weighted_mse(y_left, y_right)
        if mse < mse_min:
            mse_min = mse
            best_threshold = i
    return [best_threshold, mse_min]


def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Find the best split for a node (one feature)"""
    mse_min = sys.float_info.max
    for i in range(X.shape[1]):
        threshold, mse = split(X, y, i)[0], split(X, y, i)[1]
        if mse < mse_min:
            best_threshold = threshold
            best_feature = i
            mse_min = mse
    return best_feature, best_threshold