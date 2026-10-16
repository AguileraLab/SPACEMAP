"""Another project's phenotyping model: this project's markers are put on the training cells' scale first."""

import numpy as np
import pandas as pd

from spacemap_gui.studio import phenotyping_overview, reference_models


def test_quantile_matching_undoes_a_scanner_change_and_keeps_order():
    rng = np.random.default_rng(1)
    trained = rng.lognormal(3, 0.6, 20000)
    here = trained[:5000] * 2.5 + 40  # the same cells on a brighter scanner with an offset
    source = np.quantile(here, reference_models.QUANTILES)
    target = np.quantile(trained, reference_models.QUANTILES)
    mapped = reference_models._mapped(here, source, target)
    assert np.allclose(np.median(mapped), np.median(trained), rtol=0.05)
    assert np.all(np.diff(mapped[np.argsort(here)]) >= -1e-9)  # order kept, also beyond the 1st-99th percentiles
    beyond = np.array([source[-1] * 2])
    assert reference_models._mapped(beyond, source, target)[0] > target[-1]


def test_a_wrapped_model_predicts_on_the_training_scale():
    class Model:
        classes_ = np.array(["low", "high"])

        def predict_proba(self, values):
            high = (values[:, 0] > 10).astype(float)
            return np.column_stack([1 - high, high])

    maps = [(np.array([0.0, 100.0]), np.array([0.0, 20.0]))]  # here 0..100 is 0..20 there
    model = reference_models.ScaleMatched(Model(), maps)
    assert list(model.predict_proba(np.array([[30.0], [70.0]]))[:, 1]) == [0.0, 1.0]


def test_the_overview_counts_cells_per_image(tmp_path):
    table = tmp_path / "combined.csv"
    pd.DataFrame({"batch_id": ["A"] * 3 + ["B"] * 2, "phenotype_x": list("ttsss"), "CD3": range(5)}).to_csv(table, index=False)
    found = phenotyping_overview.overview(str(table))
    assert found["cells"] == 5 and found["rois"] == [
        {"roi": "A", "cells": 3, "slide": ""}, {"roi": "B", "cells": 2, "slide": ""},
    ]
    assert found["phenotypes"] == ["phenotype_x"]
