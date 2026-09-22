"""Taking a wrong manual-reference label off a cell.

Reviewing existing labels is only useful if a wrong one can be removed. Setting a cell
to Unassigned used to be refused outright — the write never happened, the CSV kept the
old label, and the UI reported success — so a mislabelled Treg stayed a Treg.
"""

import pandas as pd
import pytest

from spacemap_gui.tabs.tab7_phenotyping.tools import (
    _has_visual_manual_reference_assignment,
    _update_visual_manual_reference_assignment,
    _update_visual_manual_reference_assignments,
)


@pytest.fixture
def unified_csv(tmp_path):
    path = tmp_path / "unified.csv"
    pd.DataFrame(
        {
            "batch_id": ["reg055_A", "reg055_A", "reg048_A"],
            "cell_id": ["cell_170913", "cell_000001", "cell_152286"],
            "x_centroid": [494.092872570194, 10.5, 1012.85239852399],
            "y_centroid": [76.8077753779698, 20.5, 179.944649446494],
            "phenotype_ref1": ["Treg", "B cell", "Treg"],
        }
    ).to_csv(path, index=False)
    return path


def _target(path, **overrides):
    payload = dict(
        unified_csv_path=str(path),
        batch_column="batch_id",
        batch_value="reg055_A",
        x_column="x_centroid",
        x_value=494.092872570194,
        y_column="y_centroid",
        y_value=76.8077753779698,
        output_column="phenotype_ref1",
        cell_id_column="cell_id",
        cell_id_value="cell_170913",
    )
    payload.update(overrides)
    return payload


def _label(path, cell_id):
    """Read a label the way the application does.

    Assignments are appended to a sidecar rather than rewritten into the CSV on
    every save, because rewriting a 13 GB table took about thirteen minutes.
    spacemap.dataio.load merges them, so this is what any screen sees; reading
    the CSV directly shows only what has been committed.
    """
    from spacemap import dataio

    df = dataio.load(path)
    return df.loc[df["cell_id"] == cell_id, "phenotype_ref1"].iloc[0]


def _committed_label(path, cell_id):
    """Read a label straight from the CSV, ignoring uncommitted assignments."""
    df = pd.read_csv(path)
    return df.loc[df["cell_id"] == cell_id, "phenotype_ref1"].iloc[0]


def test_unassigned_removes_the_label_from_the_csv(unified_csv):
    _update_visual_manual_reference_assignment(
        **_target(unified_csv, phenotype_value="Unassigned"), allow_clear=True
    )

    assert pd.isna(_label(unified_csv, "cell_170913"))
    assert _label(unified_csv, "cell_000001") == "B cell"   # neighbours untouched
    assert _label(unified_csv, "cell_152286") == "Treg"


@pytest.mark.parametrize(
    "clear_choice",
    [
        "Unclassified",
        "unclassified",
        "Unknown",
        "Unlabeled",
        "Unlabelled",
        "Not Classified",
        "No Phenotype",
    ],
)
def test_all_no_label_choices_remove_the_label(unified_csv, clear_choice):
    _update_visual_manual_reference_assignment(
        **_target(unified_csv, phenotype_value=clear_choice), allow_clear=True
    )

    assert pd.isna(_label(unified_csv, "cell_170913"))


def test_clearing_is_refused_unless_the_caller_opts_in(unified_csv):
    """Other callers keep the old guard, so nothing clears a label by accident."""
    with pytest.raises(ValueError, match="concrete phenotype"):
        _update_visual_manual_reference_assignment(
            **_target(unified_csv, phenotype_value="Unassigned")
        )

    assert _label(unified_csv, "cell_170913") == "Treg"


def test_a_real_phenotype_still_writes_normally(unified_csv):
    _update_visual_manual_reference_assignment(
        **_target(unified_csv, phenotype_value="CD8 T cell"), allow_clear=True
    )

    assert _label(unified_csv, "cell_170913") == "CD8 T cell"


def test_cell_id_is_authoritative_when_display_coordinates_are_rounded(unified_csv):
    _update_visual_manual_reference_assignment(
        **_target(
            unified_csv,
            x_value=494.1,
            y_value=76.8,
            phenotype_value="CD4",
        ),
        allow_clear=True,
    )

    assert _label(unified_csv, "cell_170913") == "CD4"


def test_add_new_phenotype_is_never_written(unified_csv):
    with pytest.raises(ValueError, match="Create and select"):
        _update_visual_manual_reference_assignment(
            **_target(unified_csv, phenotype_value="Add New Phenotype"), allow_clear=True
        )

    assert _label(unified_csv, "cell_170913") == "Treg"


def _batch_payload(**overrides):
    payload = {
        "batch_column": "batch_id",
        "batch_value": "reg055_A",
        "x_column": "x_centroid",
        "x_value": 494.092872570194,
        "y_column": "y_centroid",
        "y_value": 76.8077753779698,
        "output_column": "phenotype_ref1",
        "cell_id_column": "cell_id",
        "cell_id_value": "cell_170913",
        "phenotype_value": "Unassigned",
    }
    payload.update(overrides)
    return payload


def test_batch_save_clears_when_asked(unified_csv):
    stats = _update_visual_manual_reference_assignments(
        unified_csv_path=str(unified_csv),
        assignments=[_batch_payload(clear_existing=True)],
    )

    assert stats["cleared_count"] == 1
    assert stats["verified_count"] == 1
    assert pd.isna(_label(unified_csv, "cell_170913"))


def test_batch_save_still_skips_plain_unassigned_cells(unified_csv):
    """A cell that never had a label and was left alone must not be written."""
    stats = _update_visual_manual_reference_assignments(
        unified_csv_path=str(unified_csv),
        assignments=[_batch_payload(clear_existing=False)],
    )

    assert stats["skipped_unassigned"] == 1
    assert stats["cleared_count"] == 0
    assert _label(unified_csv, "cell_170913") == "Treg"


def test_a_cleared_cell_no_longer_counts_as_assigned(unified_csv):
    """So the Review filter stops returning it."""
    _update_visual_manual_reference_assignment(
        **_target(unified_csv, phenotype_value="Unassigned"), allow_clear=True
    )

    assert not _has_visual_manual_reference_assignment(_label(unified_csv, "cell_170913"))
