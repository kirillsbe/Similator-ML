from typing import List
from typing import Tuple

import numpy as np
import scipy.stats
from scipy.stats import t
from scipy.stats import ttest_ind


def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    bootstrapped_quantiles_control = []
    bootstrapped_quantiles_experiment = []
    for _ in range(n_bootstraps):
        bootstrapped_sample_control = np.random.choice(control, size=len(control), replace=True)
        bootstrapped_sample_experiment = np.random.choice(experiment, size=len(experiment), replace=True)
        bootstrapped_quantiles_control.append(np.quantile(bootstrapped_sample_control, quantile))
        bootstrapped_quantiles_experiment.append(np.quantile(bootstrapped_sample_experiment, quantile))
    stat, p_value = ttest_ind(bootstrapped_quantiles_control, bootstrapped_quantiles_experiment, equal_var = False)
    if p_value <= alpha:
        result = True
    else:
        result = False
    return p_value, result
