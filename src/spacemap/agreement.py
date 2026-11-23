"""Agreement metrics with explicit treatment of undefined comparisons."""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def safe_cohen_kappa(y_true, y_pred):
    """Return NaN when kappa is undefined (empty or identical one-class sets).

    Perfect agreement on one phenotype has zero expected disagreement. It is
    not evidence of discrimination and must not be reported as kappa = 1.
    """
    if len(y_true) == len(y_pred) and len(set(y_true) | set(y_pred)) < 2:
        return np.nan
    return float(cohen_kappa_score(y_true, y_pred))
