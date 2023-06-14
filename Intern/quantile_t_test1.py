from typing import List, Tuple
import numpy as np
from scipy import stats


def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Two-sample t-test for the means of two independent samples"""
    stat, p_value = stats.ttest_ind(control, experiment, equal_var = False)
    if p_value <= alpha:
        result = True
    else:
        result = False
    return p_value, result
