"""Publication figures for any grouping of the cells: a clustered heatmap or a phenotype dot plot, as classic Tab 7 makes them.

* **Saved-run heatmap**: the run's selected Median/Mean transformed profiles, with
  same-named clusters averaged as in classic's "Generate Publication Heatmap".
* **Table heatmap**: each group's median (optional mean) of each marker
  (arcsinh(value / cofactor); corrected columns already are), compared across groups.
  Both use classic's shared figure renderer, independent X/Y clustering and a right colour bar.
* **Dot plot** (classic's ``PhenotypeDotPlot``): dot colour is the mean of each marker in the
  group (z-scored across cells), dot size the share of the group's cells above the positivity
  threshold, with a cell-count bar chart.
* **A heatmap already on screen** (a clustering's or the neighbourhoods' profile): drawn the
  same way from its numbers.

Heatmaps include every labelled cell in the combined table. Dot plots retain their per-group sampling.
Nothing is written to the project: the figure goes back to the browser to download.
"""

import io
import re

import numpy as np
import pandas as pd

PER_GROUP = 20000
FORMATS = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
CMAPS = ("RdBu_r", "coolwarm", "bwr", "viridis", "magma")
DOT_CMAPS = ("viridis", "magma", "plasma", "Reds", "Blues")


def _options(options):
    options = dict(options or {})
    fmt = str(options.get("format", "png"))
    if fmt not in FORMATS:
        raise ValueError("Choose PNG, SVG or PDF.")
    dpi = int(options.get("dpi", 300))
    if not 72 <= dpi <= 600:
        raise ValueError("Choose a resolution between 72 and 600 dpi.")
    return fmt, dpi, options


def _save(figure, fmt, dpi):
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    figure.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _clustermap(matrix, title, options):
    from spacemap.heatmap import publication_heatmap
    if options.get("cmap", "RdBu_r") not in CMAPS:
        raise ValueError("Choose one of the listed colour maps.")
    return publication_heatmap(matrix, title, options)


def matrix_figure(rows, columns, values, title, options=None):
    """A heatmap already on screen (rows x columns of numbers), as a publication figure."""
    fmt, dpi, options = _options(options)
    matrix = pd.DataFrame(
        np.asarray(values, dtype=float), index=[str(r) for r in rows], columns=[str(c) for c in columns]
    )
    if matrix.shape[0] > 500 or matrix.shape[1] > 500:
        raise ValueError("That is too large for one figure.")
    return _save(_clustermap(matrix, title, options), fmt, dpi), FORMATS[fmt]


def run_figure(root, table, job_id, names=None, options=None):
    """Classic's publication calculation, using the full saved run, never the open ROI."""
    from pathlib import Path
    from spacemap.cluster_profiles import named_profiles
    from . import cluster_profiles, jobs, cluster_names

    fmt, dpi, options = _options(options)
    job = jobs.status(root, job_id)
    result = job.get("result") or {}
    if job.get("status") != "done" or job.get("kind") not in cluster_profiles.METHODS:
        raise ValueError("Choose a finished phenotyping run.")
    if Path(result.get("table", "")).resolve() != Path(table).resolve():
        raise ValueError("This clustering belongs to another combined table.")
    profile = result.get("profile")
    if not profile or not profile.get("summaries"):
        raise ValueError(result.get("profile_notice") or "This older run lacks saved transformed profiles. Run clustering again, or choose Full combined table.")
    aggregation = options.get("aggregation", profile.get("aggregation", "median"))
    if aggregation not in {"median", "mean"}:
        raise ValueError("Choose Median or Mean for the heatmap.")
    profile = cluster_profiles.choose(result, aggregation)["profile"]
    if names is None:
        names = cluster_names.saved_names(root, table, result["column"], cluster_job_id=job_id)["names"] if result.get("clusters") else {}
    if not isinstance(names, dict) or any(not isinstance(v, str) or len(v) > 120 for v in names.values()):
        raise ValueError("Use short phenotype names for the figure.")
    matrix = pd.DataFrame(profile["values"], index=profile["rows"], columns=profile["columns"])
    matrix = named_profiles(matrix, {r: names.get(r) or r for r in matrix.index})
    cells = int(result.get("cells") or sum(c["cells"] for c in result.get("clusters", result.get("phenotypes", []))))
    title = options.get("title") or f"{job['kind'].upper()} phenotype heatmap"
    caption = f"{title}\n{profile['note']}\nFull saved run · {cells:,} cells; clusters sharing a name are averaged"
    figure = _clustermap(matrix, caption, options)
    return _save(figure, fmt, dpi), FORMATS[fmt]


def _cells(table, column, markers, names=None, *, sample=True):
    """Labelled groups and marker values; optional PER_GROUP sampling is for dot plots only."""
    from spacemap import dataio

    available = set(dataio.columns(table))
    missing = [name for name in [column, *markers] if name not in available]
    if missing:
        raise ValueError(f"The table has no {', '.join(missing[:5])}.")
    frame = dataio.load(table, columns=[column, *markers])
    groups = frame[column].astype("string")
    if names:
        groups = groups.map(lambda value: names.get(str(value), value) if pd.notna(value) else value)
    keep = groups.notna() & (groups.str.strip() != "")
    frame, groups = frame[keep.to_numpy()], groups[keep]
    rng = np.random.default_rng(0)
    rows = []
    for _, index in groups.groupby(groups.to_numpy()).indices.items():
        rows.append(index if not sample or len(index) <= PER_GROUP else rng.choice(index, PER_GROUP, replace=False))
    if not rows:
        raise ValueError(f"No cell has a value in {column}.")
    rows = np.sort(np.concatenate(rows))
    values = frame.iloc[rows][markers].apply(pd.to_numeric, errors="coerce")
    return groups.iloc[rows].to_numpy(), values


def _scaled(values, cofactor):
    """arcsinh(value / cofactor); corrected columns (_corr) already hold arcsinh values."""
    out = values.copy()
    for column in out.columns:
        if not re.search(r"_corr$", column):
            out[column] = np.arcsinh(out[column] / cofactor)
    out.columns = [re.sub(r"_corr$", "", column) for column in out.columns]
    return out


def table_figure(table, column, markers, kind, options=None, names=None):
    """A heatmap or dot plot of *column*'s groups over *markers* of the combined *table*."""
    fmt, dpi, options = _options(options)
    if not markers:
        raise ValueError("Choose the markers to show.")
    if len(markers) > 200:
        raise ValueError("Choose at most 200 markers.")
    groups, values = _cells(table, column, list(markers), names, sample=kind != "heatmap")
    cofactor = float(options.get("cofactor", 5.0) or 5.0)
    scaled = _scaled(values, cofactor)
    title = str(options.get("title") or column)
    if kind == "heatmap":
        from spacemap.cluster_profiles import standardize_groups
        aggregation = options.get("aggregation", "median")
        if aggregation not in {"median", "mean"}:
            raise ValueError("Choose Median or Mean for the heatmap.")
        summary = scaled.groupby(groups).agg(aggregation)
        caption = f"{title}\n{aggregation.title()} · arcsinh · all {len(groups):,} labelled cells in the combined table"
        figure = _clustermap(standardize_groups(summary), caption, options)
        return _save(figure, fmt, dpi), FORMATS[fmt]
    if kind != "dotplot":
        raise ValueError("Choose a heatmap or a dot plot.")
    import matplotlib

    matplotlib.use("Agg")
    from spacemap_gui.tabs.tab7_phenotyping.common import PhenotypeDotPlot

    cmap = options.get("cmap", "viridis")
    if cmap not in DOT_CMAPS:
        cmap = "viridis"
    z = (scaled - scaled.mean(axis=0)) / scaled.std(axis=0).replace(0, 1)
    cells = z.assign(_group=groups)
    plot = PhenotypeDotPlot.from_dataframe(
        cells,
        "_group",
        marker_columns=list(z.columns),
        marker_labels=list(z.columns),
        positive_threshold=float(options.get("threshold", 0.5)),
        normalization_label="arcsinh, z-scored per marker",
        title=title,
        color_map=cmap,
    )
    if plot is None:
        raise ValueError("No numeric marker values to draw.")
    figure = plot.plot(
        show_grid=bool(options.get("grid", True)),
        show_count_bars=bool(options.get("bars", True)),
        show_count_labels=bool(options.get("bars", True)),
        show_colorbar=True,
        show_size_legend=True,
    )
    return _save(figure, fmt, dpi), FORMATS[fmt]
