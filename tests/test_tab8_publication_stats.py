import numpy as np
import pandas as pd

from spacemap_gui.tabs.tab8_analysis.density_analysis import (
    _holm_adjust,
    compute_matched_method_statistics,
    compute_zonal_method_statistics,
)


METHODS = ["Leiden", "SOM", "Vortex", "RESOLVE"]


def test_holm_adjustment_preserves_order_and_nan():
    adjusted = _holm_adjust([0.01, 0.04, np.nan, 0.03])
    assert np.allclose(adjusted[[0, 1, 3]], [0.03, 0.06, 0.06])
    assert np.isnan(adjusted[2])


def test_matched_statistics_use_friedman_and_holm_wilcoxon():
    wide = pd.DataFrame(
        {
            "Leiden": [10, 11, 12, 13, 14, 15, 16, 17],
            "SOM": [2, 3, 4, 5, 6, 7, 8, 9],
            "Vortex": [7, 8, 9, 10, 11, 12, 13, 14],
            "RESOLVE": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    result = compute_matched_method_statistics(wide, METHODS, "test_unit")

    omnibus = result[result["test"] == "friedman"]
    pairwise = result[result["test"] == "paired_wilcoxon_signed_rank"]
    assert len(omnibus) == 1
    assert len(pairwise) == 6
    assert omnibus.iloc[0]["n_paired_units"] == 8
    assert omnibus.iloc[0]["effect_size_name"] == "Kendall_W"
    assert pairwise["multiple_testing_correction"].eq("Holm").all()
    assert (pairwise["p_value"] == pairwise["p_value_adjusted"]).all()
    assert (pairwise["p_value_adjusted"] >= pairwise["p_value_raw"]).all()


def test_identical_paired_methods_return_non_significant_wilcoxon():
    values = [1.0, 2.0, 3.0, 4.0]
    wide = pd.DataFrame({"Leiden": values, "SOM": values})
    result = compute_matched_method_statistics(wide, ["Leiden", "SOM"])
    assert len(result) == 1
    assert result.iloc[0]["p_value_raw"] == 1.0
    assert result.iloc[0]["effect_size"] == 0.0


def test_zonal_statistics_export_adjusted_zone_family():
    rows = []
    for phenotype in ["DC"]:
        for zone in [-20.0, 20.0]:
            for block in range(8):
                for method_index, method in enumerate(METHODS):
                    rows.append(
                        {
                            "phenotype": phenotype,
                            "zone_center_um": zone,
                            "block": f"r{block}",
                            "method": method,
                            "density": float(block + (4 - method_index) * 3 + zone / 100.0),
                        }
                    )
    zone_df = pd.DataFrame(rows)
    result = compute_zonal_method_statistics(
        zone_df,
        group_cols=["phenotype", "zone_center_um"],
        block_col="block",
        value_col="density",
        methods=METHODS,
    )
    assert not result.empty
    assert result["p_value_adjusted"].notna().all()
    assert result["multiple_testing_correction"].eq("Holm across distance-zone family").all()
    assert set(result["zone_center_um"]) == {-20.0, 20.0}
