"""What classic Tab 8 saved, for Studio's Spatial analysis step: triads and feature screening runs (read only).

Triads: Tab 8's ``triad_instances_<style>.csv`` tables (every triad found, with its type and place), counted by
type; a type can then be drawn on the image as a density map. Feature screening: each run's
``tables/screening_results_all.csv``, its strongest findings first, and its figures as small pictures.
"""

import base64
import io
from pathlib import Path

import pandas as pd

HITS = 15
FIGURES = 6
FIGURE_SIDE = 1100


def triads(root):
    """Each saved triad table: its name (as the image's density tables name it) and its triad types, commonest first."""
    from spacemap.viewer.catalog import find_point_tables

    found = []
    for name, path in find_point_tables(root):
        try:
            labels = pd.read_csv(path, usecols=lambda column: column in {"triad_label", "style"}, low_memory=False)
        except (OSError, ValueError):
            continue
        if "triad_label" not in labels:
            continue
        counts = labels["triad_label"].dropna().astype(str).value_counts()
        found.append(
            {
                "name": name,
                "triads": int(counts.sum()),
                "types": [{"label": label, "count": int(count)} for label, count in counts.head(40).items()],
            }
        )
    return found


def _picture(path):
    from PIL import Image

    with Image.open(path) as picture:
        picture = picture.convert("RGB")
        picture.thumbnail((FIGURE_SIDE, FIGURE_SIDE))
        out = io.BytesIO()
        picture.save(out, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def screening(root, run=None):
    """Tab 8's feature screening runs, newest first; for *run* (else the newest) its strongest findings and figures."""
    base = Path(root) / "outputs" / "spatial_feature_screening"
    runs = (
        sorted(
            (folder for folder in base.glob("*") if (folder / "tables" / "screening_results_all.csv").is_file()),
            key=lambda folder: folder.stat().st_mtime,
            reverse=True,
        )
        if base.is_dir()
        else []
    )
    names = [folder.name for folder in runs]
    if not runs:
        return {"runs": []}
    chosen = next((folder for folder in runs if folder.name == run), runs[0])
    table = pd.read_csv(chosen / "tables" / "screening_results_all.csv", low_memory=False)
    order = [column for column in ("fdr_q", "p_value") if column in table]
    if order:
        table = table.sort_values(order)
    keep = [
        column
        for column in (
            "feature_label",
            "endpoint_label",
            "stratum_label",
            "direction",
            "effect_size",
            "hazard_ratio",
            "p_value",
            "fdr_q",
        )
        if column in table
    ]
    hits = table[keep].head(HITS).astype(object).where(table[keep].head(HITS).notna(), None)
    figures = sorted((chosen / "figures").glob("*.png"))[:FIGURES] if (chosen / "figures").is_dir() else []
    return {
        "runs": names,
        "run": chosen.name,
        "tested": int(len(table)),
        "significant": int((table["fdr_q"] < 0.05).sum()) if "fdr_q" in table else None,
        "hits": hits.to_dict("records"),
        "figures": [{"name": path.stem.replace("_", " "), "image": _picture(path)} for path in figures],
    }
