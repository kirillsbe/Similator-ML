from typing import Tuple
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error loss function and gradient."""
    grad = y_pred - y_true
    loss = np.mean(grad**2)
    return loss, grad


class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
    ):
        # YOUR CODE HERE
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.trees_ = []
        if loss == 'mse':
            self.loss = self._mse
        else:
            self.loss = loss

    def _mse(self, y_true, y_pred):
        return mse(y_true, y_pred)

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        # YOUR CODE HERE
        self.base_pred_ = np.mean(y)
        y_pred = y.mean()
        self.loss_, grad = self.loss(y, y_pred)
        for i in range(self.n_estimators):
            grad = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X,grad)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.loss_, grad = self.loss(y, y_pred)

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.
            
        """
        # YOUR CODE HERE
        predictions = self.learning_rate*np.vstack([
                            tree.predict(X) 
                            for tree in self.trees_
                            ]).sum(axis=0) + self.base_pred_
        return predictions
