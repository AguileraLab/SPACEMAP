"""A label must land on the cell it was assigned to, and on no other.

Mislabelling a cell is the worst failure this code can produce: it is silent,
it corrupts a manual reference set, and anything trained or validated against
that set inherits the error. These tests exist to make that failure loud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels


@pytest.fixture
def table(tmp_path):
    """A table big enough that an off-by-one or a shuffle would show."""
    rows = 5_000
    rng = np.random.default_rng(11)
    path = tmp_path / "unified.csv"
    pd.DataFrame(
        {
            "cell_id": [f"cell_{index}" for index in range(rows)],
            "batch_id": rng.choice(["P1", "P2", "P3"], rows),
            "x_centroid": rng.uniform(0, 10_000, rows),
            "phenotype_ref": [None] * rows,
        }
    ).to_csv(path, index=False)
    return path


def test_each_label_lands_on_its_own_cell(table):
    targets = {
        "cell_0": "Tumor",
        "cell_1": "CD8 T",
        "cell_2499": "Macrophage",
        "cell_4999": "Stroma",
    }
    base = pd.read_csv(table)
    positions = {
        cell_id: int(base.index[base["cell_id"] == cell_id][0]) for cell_id in targets
    }
    labels.append(
        table,
        [
            {"row": positions[cell_id], "column": "phenotype_ref", "value": value}
            for cell_id, value in targets.items()
        ],
    )

    frame = dataio.load(table)
    for cell_id, expected in targets.items():
        actual = frame.loc[frame["cell_id"] == cell_id, "phenotype_ref"].iloc[0]
        assert actual == expected, f"{cell_id} should be {expected}, found {actual!r}"


def test_no_other_cell_is_touched(table):
    base = pd.read_csv(table)
    position = int(base.index[base["cell_id"] == "cell_1234"][0])
    labels.append(table, [{"row": position, "column": "phenotype_ref", "value": "Tumor"}])

    frame = dataio.load(table)
    labelled = frame.loc[frame["phenotype_ref"].notna(), "cell_id"].tolist()
    assert labelled == ["cell_1234"], f"expected exactly one labelled cell, found {labelled}"


def test_positions_survive_a_narrow_column_read(table):
    """A screen that reads two columns must still see the label on the right row."""
    base = pd.read_csv(table)
    position = int(base.index[base["cell_id"] == "cell_777"][0])
    labels.append(table, [{"row": position, "column": "phenotype_ref", "value": "Treg"}])

    narrow = dataio.load(table, columns=["cell_id", "phenotype_ref"])
    assert narrow.loc[narrow["cell_id"] == "cell_777", "phenotype_ref"].iloc[0] == "Treg"
    assert int(narrow["phenotype_ref"].notna().sum()) == 1


def test_committing_writes_the_same_cells(table):
    targets = {"cell_10": "Tumor", "cell_3000": "CD4 T"}
    base = pd.read_csv(table)
    labels.append(
        table,
        [
            {
                "row": int(base.index[base["cell_id"] == cell_id][0]),
                "column": "phenotype_ref",
                "value": value,
            }
            for cell_id, value in targets.items()
        ],
    )
    labels.commit(table, rebuild_fast_copy=False)

    written = pd.read_csv(table)
    for cell_id, expected in targets.items():
        assert written.loc[written["cell_id"] == cell_id, "phenotype_ref"].iloc[0] == expected
    assert int(written["phenotype_ref"].notna().sum()) == len(targets)


def test_labels_are_refused_when_the_table_changes_underneath(table):
    """Row positions are meaningless if the table was rewritten elsewhere."""
    base = pd.read_csv(table)
    position = int(base.index[base["cell_id"] == "cell_42"][0])
    labels.append(table, [{"row": position, "column": "phenotype_ref", "value": "Tumor"}])

    # Someone re-exports the table with a different row order.
    shuffled = base.sample(frac=1.0, random_state=5).reset_index(drop=True)
    shuffled.to_csv(table, index=False)

    assert labels.pending_count(table) == 0, "stale labels must not be applied"
    frame = dataio.load(table)
    assert int(frame["phenotype_ref"].notna().sum()) == 0, (
        "a label from before the reorder must never be written onto whatever cell "
        "now occupies that position"
    )
