"""Studio's cell-cell interactions (Tab 8): which phenotypes are neighbours more, or less, often than by chance.

Tab 8's own test (``_compute_interaction_stats``): within each slide and tissue region, every pair of cells closer
than the radius is counted by phenotype pair, and compared with the same count after shuffling the phenotypes
among the cells (permutations): a z-score and a two-sided p-value per pair. The settings are Tab 8's keys
(tab8_cci_*). Nothing is written into the table; the result (z-scores, p-values, strongest pairs) stays in the
job, and the page shows it as a clickable heatmap.
"""

import numpy as np

# Studio setting -> (Tab 8 key, default, lowest, highest); None limits for text.
KEYS = {
    "phenotype_col": ("tab8_cci_pheno_col", "", None, None),
    "batch_col": ("tab8_cci_batch_col", "batch_id", None, None),
    "roi_col": ("tab8_cci_roi_col", "tissue_roi", None, None),
    "radius_um": ("tab8_cci_radius_um", 30.0, 1.0, 300.0),
    "pixel_um": ("tab8_cci_pixel_um", 0.5083, 0.01, 100.0),
    "n_permutations": ("tab8_cci_perms", 500, 50, 5000),
    "min_cells": ("tab8_cci_min_cells", 20, 5, 5000),
    "random_state": ("tab8_cci_random", 42, 0, 100000),
}


def _tab8(config):
    return (config.get("tab_configurations") or {}).get("tab8_analysis") or {}


def values(config):
    saved = _tab8(config)
    return {
        name: saved.get(key, default) if saved.get(key) not in (None, "None") else default
        for name, (key, default, *_) in KEYS.items()
    }


def check(changes):
    if not isinstance(changes, dict) or set(changes) - set(KEYS):
        raise ValueError("Unknown cell-cell interaction setting.")
    for name, value in changes.items():
        _, default, low, high = KEYS[name]
        if isinstance(default, (int, float)):
            number = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not number or not low <= value <= high or (isinstance(default, int) and int(value) != value):
                raise ValueError(f"{name.replace('_', ' ')} must be a number from {low} to {high}.")
        elif not isinstance(value, str) or len(value) > 256:
            raise ValueError(f"Choose a column for {name.replace('_', ' ')}.")
    return changes


def save(root, changes, expected):
    from spacemap_gui import persistence

    from .settings import project_lock, read_config, revision

    check(changes)
    with project_lock(root):
        config = read_config(root)
        if expected != revision(values(config)):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        for name, value in changes.items():
            persistence.update_stored_config_key(config, KEYS[name][0], value)
        persistence.save_project_config(str(root), config)
        return values(read_config(root))


def run(settings, folder, report):
    """The job: Tab 8's interaction test over the whole combined table."""
    from spacemap import dataio
    from spacemap_gui.tabs.tab8_analysis.cell_cell_interaction import _compute_interaction_stats

    chosen = settings["values"]
    available = dataio.columns(settings["table"])
    grouping = [chosen[name] for name in ("batch_col", "roi_col") if chosen[name] and chosen[name] in available]
    report("Reading the combined table…", 0.05)
    images = settings.get("images") or []
    frame = dataio.load(
        settings["table"],
        columns=list(dict.fromkeys(["x_centroid", "y_centroid", chosen["phenotype_col"], *grouping,
                                    *([settings["image_col"]] if images else [])])),
    )
    if images:  # only the chosen images' cells
        frame = frame[frame[settings["image_col"]].astype(str).isin(images)]
        if frame.empty:
            raise ValueError("None of the chosen images is in the combined table.")
    radius_px = float(chosen["radius_um"]) / float(chosen["pixel_um"])
    report(f"Counting neighbours within {chosen['radius_um']:g} µm and {int(chosen['n_permutations'])} shuffles…", 0.2)
    z, p, long, meta = _compute_interaction_stats(
        frame,
        phenotype_col=chosen["phenotype_col"],
        batch_col=chosen["batch_col"] if chosen["batch_col"] in grouping else "",
        roi_col=chosen["roi_col"] if chosen["roi_col"] in grouping else "",
        radius_px=radius_px,
        n_permutations=int(chosen["n_permutations"]),
        min_cells_per_region=int(chosen["min_cells"]),
        random_state=int(chosen["random_state"]),
    )
    if z.empty:
        raise ValueError("No region had enough cells with a phenotype and neighbours within the radius.")
    folder.mkdir(parents=True, exist_ok=True)
    long.to_csv(folder / "cci_all_pairs.csv", index=False)
    names = [str(name) for name in z.index]
    # Each unordered pair once, the strongest first (enriched and avoided alike).
    pairs = long[long["source"].astype(str) <= long["target"].astype(str)].copy()
    pairs["strength"] = pairs["z_score"].abs()
    top = pairs.sort_values(["strength", "p_value"], ascending=[False, True]).head(30)
    counts = frame[chosen["phenotype_col"]].dropna().astype(str).value_counts()
    return {
        "table": settings["table"],
        "images": images,
        "column": chosen["phenotype_col"],
        "phenotypes": [{"name": name, "cells": int(counts.get(name, 0))} for name in names],
        "profile": {
            "rows": names,
            "columns": names,
            "values": np.round(np.clip(z.to_numpy(dtype=float), -10, 10), 3).tolist(),
            "raw": np.round(z.to_numpy(dtype=float), 3).tolist(),
            "note": "z-score of how often the two phenotypes are neighbours, against shuffled phenotypes",
        },
        "p_values": np.round(p.to_numpy(dtype=float), 5).tolist(),
        "pairs": [
            {
                "a": str(row.source),
                "b": str(row.target),
                "z": round(float(row.z_score), 2),
                "p": float(row.p_value),
                "observed": int(row.observed_edges),
                "expected": round(float(row.null_mean), 1),
            }
            for row in top.itertuples()
        ],
        "meta": {**meta, "radius_um": float(chosen["radius_um"]), "cells": int(len(frame))},
    }
