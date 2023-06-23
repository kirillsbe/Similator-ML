from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sys

@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None

@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        mse = np.sum((np.mean(y) - y)**2) / len(y)
        return mse

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weithed mse criterion for a two given sets of target values"""
        mseWeighted = (self._mse(y_left) * len(y_left) + self._mse(y_right) * len(y_right)) / (len(y_left) + len(y_right))
        return mseWeighted
        
    def _split(self, X: np.ndarray, y: np.ndarray, feature: int) -> float:
        """Find the best split for a node (one feature)"""
        mse_min = sys.float_info.max
        for i in X[:, feature]:
            y_left, y_right = y[X[:, feature] > i], y[X[:, feature] <= i]
            mse = self._weighted_mse(y_left, y_right)
            if mse < mse_min:
                mse_min = mse
                best_threshold = i
        return [best_threshold, mse_min]

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        mse_min = sys.float_info.max
        for i in range(X.shape[1]):
            threshold, mse = self._split(X, y, i)[0], self._split(X, y, i)[1]
            if mse < mse_min:
                best_thr = threshold
                best_idx = i
                mse_min = mse
        return best_idx, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Create a leaf node if any stopping criterion is met
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1):
            value = round(np.mean(y))
            mse = self._mse(y)
            return Node(n_samples=n_samples, value=value, mse=mse)

        # Split node based on best feature and threshold
        feature, threshold = self._best_split(X, y)
        X_left, y_left = X[X[:, feature] <= threshold], y[X[:, feature] <= threshold]
        X_right, y_right = X[X[:, feature] > threshold], y[X[:, feature] > threshold]

        # Recursively split left and right child nodes
        left = self._split_node(X_left, y_left, depth=depth+1)
        right = self._split_node(X_right, y_right, depth=depth+1)

        return Node(feature=feature, threshold=threshold, n_samples=n_samples, value=round(np.mean(y)), mse=self._mse(y), left=left, right=right)


