"""PH diagnostics must return the actual lifelines KM-transform result."""
import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from spacemap_gui.tabs.tab8_analysis.feature_screening_core import (
    _continuous_cox_feature_summary, _cox_hr_and_ci,
)


@pytest.mark.parametrize("continuous", [False, True])
def test_ph_diagnostic_matches_lifelines_km_transform(continuous):
    rng = np.random.default_rng(416)
    values = rng.normal(size=100)
    observed = rng.exponential(12 * np.exp(-.6 * values))
    censor = rng.uniform(3, 20, size=len(values))
    data = pd.DataFrame({"time": np.minimum(observed, censor),
                         "event": (observed <= censor).astype(int),
                         "value": values, "group": (values > 0).astype(int)})
    if continuous:
        actual = _continuous_cox_feature_summary(data, "time", "event", "value")["ph_test_p"]
        fit = data[["time", "event"]].copy()
        fit["feature_z"] = (values - values.mean()) / values.std(ddof=0)
    else:
        actual = _cox_hr_and_ci(data, "time", "event", "group")[-1]
        fit = data[["time", "event", "group"]]
    model = CoxPHFitter(penalizer=.01).fit(fit, "time", "event")
    expected = proportional_hazard_test(model, fit, time_transform="km").p_value
    assert np.isfinite(actual), "A successful PH diagnostic must not silently become NaN"
    assert actual == pytest.approx(float(np.asarray(expected).reshape(-1)[0]), rel=1e-10)
