import warnings

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from spacemap.agreement import safe_cohen_kappa


def test_undefined_kappa_has_no_runtime_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        assert np.isnan(safe_cohen_kappa([], []))
        assert np.isnan(safe_cohen_kappa(['CD8'] * 5, ['CD8'] * 5))


@pytest.mark.parametrize('truth,prediction', [
    (['A', 'A'], ['B', 'B']),
    (['A', 'B', 'B', 'A'], ['A', 'A', 'B', 'A']),
    (['A', 'B'], ['A', 'B']),
])
def test_defined_kappa_matches_sklearn(truth, prediction):
    assert safe_cohen_kappa(truth, prediction) == pytest.approx(cohen_kappa_score(truth, prediction))
