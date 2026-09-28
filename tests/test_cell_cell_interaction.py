"""Tab 8's cell-cell interaction statistic: the fast pair counting gives the same result as before."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spacemap_gui.tabs.tab8_analysis.cell_cell_interaction import _compute_interaction_stats


def _earlier(df, radius, permutations, seed):
    """The earlier counting (np.add.at), for one region, as reference."""
    phenotypes = sorted(df["ph"].astype(str).unique())
    index = df["ph"].map({p: i for i, p in enumerate(phenotypes)}).to_numpy()
    pairs = cKDTree(df[["x_centroid", "y_centroid"]].to_numpy()).query_pairs(r=radius, output_type="ndarray")
    n = len(phenotypes)

    def counts(labels):
        out = np.zeros((n, n))
        np.add.at(out, (labels[pairs[:, 0]], labels[pairs[:, 1]]), 1)
        np.add.at(out, (labels[pairs[:, 1]], labels[pairs[:, 0]]), 1)
        return out

    obs, rng = counts(index), np.random.default_rng(seed)
    total, squares = np.zeros((n, n)), np.zeros((n, n))
    for _ in range(permutations):
        drawn = counts(rng.permutation(index))
        total += drawn
        squares += drawn**2
    mean = total / permutations
    std = np.sqrt(np.clip(squares / permutations - mean**2, 0, None))
    std[std <= 0] = 1.0
    return (obs - mean) / std


def test_the_z_scores_are_unchanged():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x_centroid": rng.uniform(0, 400, 600), "y_centroid": rng.uniform(0, 400, 600)})
    df["ph"] = np.where(df.x_centroid < 200, "T", "B")
    df.loc[rng.random(600) < 0.3, "ph"] = "M"
    z, _, long, meta = _compute_interaction_stats(df, "ph", "", "", 25.0, 50, 5, 7)
    assert np.allclose(z.to_numpy(), _earlier(df, 25.0, 50, 7))
    assert meta["n_regions_used"] == 1 and len(long) == 9 and z.loc["T", "T"] > 2 and z.loc["T", "B"] < -2
