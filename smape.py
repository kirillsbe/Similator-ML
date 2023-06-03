import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    metric = np.mean(
        np.divide(
            2 * np.abs(y_true - y_pred),
            (np.abs(y_true) + np.abs(y_pred) + np.finfo(float).eps),
        )
    )
    return metric
