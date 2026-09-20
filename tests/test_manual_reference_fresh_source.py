"""Manual Reference cell records must come from the unified CSV, never disk cache."""

import json
import os

import pandas as pd

from spacemap_gui.tabs.tab7_phenotyping.tools import (
    _load_manual_reference_csv_fresh,
    _prune_visual_manual_reference_runtime_cache,
)


def test_source_loader_reads_the_current_file_even_when_signature_is_unchanged(tmp_path):
    path = tmp_path / "unified.csv"
    pd.DataFrame({"cell_id": ["cell_1"], "phenotype_ref1": ["Treg"]}).to_csv(
        path, index=False
    )
    original_stat = path.stat()

    first = _load_manual_reference_csv_fresh(
        str(path), ["cell_id", "phenotype_ref1"]
    )
    assert first.loc[0, "phenotype_ref1"] == "Treg"

    # CD4x has the same width as Treg, and restoring the timestamp recreates the
    # exact (mtime, size) signature that a path-based cache would trust.
    pd.DataFrame({"cell_id": ["cell_1"], "phenotype_ref1": ["CD4x"]}).to_csv(
        path, index=False
    )
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert path.stat().st_size == original_stat.st_size

    second = _load_manual_reference_csv_fresh(
        str(path), ["cell_id", "phenotype_ref1"]
    )
    assert second.loc[0, "phenotype_ref1"] == "CD4x"


def test_runtime_cards_and_dropdowns_are_removed_from_tools_cache():
    cache = {
        "tools_mr_vv_output_col": "phenotype_ref1",
        "tools_mr_vv_results": [{"current_phenotype": "Treg"}],
        "tools_mr_vv_seen_keys": ["cell_1"],
        "tools_mr_vv_fetch_more": True,
        "tools_mr_vv_last_file": "/old/unified.csv",
        "tools_mr_vv_assign_pheno_cell_1": "Unclassified",
        "tools_mr_vv_assign_save_cell_1": False,
    }

    _prune_visual_manual_reference_runtime_cache(cache)

    assert cache == {"tools_mr_vv_output_col": "phenotype_ref1"}
    json.dumps(cache)  # remains a valid settings payload
