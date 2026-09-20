"""
High-Expression Composition panel (Sub-Phenotyping).

A visual alternative to the statistical *Functional Marker Discovery* step.
For every marker in the active panel it takes the cells that express that
marker above a Z-score cut (1.5 by default) and shows which phenotypes those
cells belong to, as one stacked bar per marker. A marker whose high-expressing
cells sit almost entirely inside one phenotype is a candidate for splitting
that phenotype; a marker whose high cells are spread across every lineage is
not.

The cells can be narrowed with any number of column filters, and split by a
categorical column (patient, batch, ...) to produce one figure per level, so a
marker can be checked for consistency before it is used. Markers picked here
are written straight into the sub-clustering marker set, for one phenotype or
for several at once.

Phenotype colours follow the house palette in ``cell_visualization``; anything
outside it is given a colour chosen to look unlike the ones already in use, and
every colour can be overridden.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import to_hex

from .cell_visualization import _DEFAULT_COLOR_MAP as HOUSE_PHENOTYPE_COLORS
from .common import get_analysis_marker_label

try:
    from spacemap.acc_utils import distinct_colors
except Exception:  # noqa: BLE001 - the fallback palette below is enough
    distinct_colors = None


DEFAULT_Z_THRESHOLD = 1.5

# A numeric column with more levels than this is far more likely to be a
# measurement than a grouping, so stratifying by it would produce hundreds of
# one-cell figures. It is flagged rather than blocked.
CONTINUOUS_UNIQUE_LIMIT = 25

# Above this many levels even a text column is suspect as a stratifier.
MANY_LEVELS_WARNING = 40

MISSING_LABEL = "Unassigned"

METRIC_ENRICHMENT = "Enrichment (log2)"
METRIC_SHARE = "Share of high cells (%)"
METRIC_POSITIVE = "% of phenotype positive"
SUMMARY_METRICS = (METRIC_ENRICHMENT, METRIC_SHARE, METRIC_POSITIVE)

MARKER_ORDERS = ("Panel order", "Alphabetical", "Grouped by dominant phenotype")

_FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5", "#393b79",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_key(value) -> str:
    """Return a Streamlit-safe suffix for a dynamic widget key."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "item"


def _sort_labels(labels) -> list[str]:
    return sorted({str(label) for label in labels}, key=lambda text: (text.lower(), text))


def _as_float(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def profile_values(values: list[str]) -> dict:
    """Describe a column from its unique values: level count and numeric-ness."""
    cleaned = [str(v) for v in values]
    numbers = [_as_float(v) for v in cleaned]
    numeric = bool(cleaned) and all(n is not None for n in numbers)
    return {
        "n_unique": len(cleaned),
        "numeric": numeric,
        "numbers": [n for n in numbers if n is not None] if numeric else [],
        "looks_continuous": numeric and len(cleaned) > CONTINUOUS_UNIQUE_LIMIT,
    }


def normalize_phenotype_labels(series: pd.Series) -> pd.Series:
    """Blank, NaN and literal 'nan'/'None' labels all become one visible group."""
    labels = series.where(series.notna(), MISSING_LABEL).astype(str).str.strip()
    return labels.replace({"": MISSING_LABEL, "nan": MISSING_LABEL, "NaN": MISSING_LABEL,
                           "None": MISSING_LABEL, "none": MISSING_LABEL})


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

def default_phenotype_colors(labels: list[str]) -> dict[str, str]:
    """House colour per phenotype, with a distinct colour for anything new.

    The house palette is matched case-insensitively so `CD4` and `cd4` do not
    end up in two different colours. Labels outside it are coloured away from
    everything already assigned, so a new phenotype never lands on a
    near-duplicate of an established one.
    """
    known = {str(name).strip().lower(): color for name, color in HOUSE_PHENOTYPE_COLORS.items()}
    mapping: dict[str, str] = {}
    pending: list[str] = []
    for label in labels:
        color = known.get(str(label).strip().lower())
        if color:
            mapping[str(label)] = color
        else:
            pending.append(str(label))

    if not pending:
        return mapping

    used = list(mapping.values())
    if distinct_colors is not None:
        try:
            extra = distinct_colors(len(pending), avoid=used)
        except Exception:  # noqa: BLE001
            extra = []
    else:
        extra = []
    if len(extra) < len(pending):
        spare = [c for c in _FALLBACK_COLORS if c not in used]
        extra = list(extra) + [spare[i % len(spare)] for i in range(len(pending) - len(extra))]
    for label, color in zip(sorted(pending, key=str.lower), extra):
        mapping[label] = color
    return mapping


def resolve_phenotype_colors(labels: list[str], key_prefix: str) -> dict[str, str]:
    """Default colours, with any colour the user picked in this session winning."""
    defaults = default_phenotype_colors(labels)
    mapping = dict(defaults)
    for label in labels:
        widget_key = f"{key_prefix}_color_{_safe_key(label)}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = defaults.get(str(label), "#000000")
        try:
            mapping[str(label)] = to_hex(st.session_state[widget_key])
        except (ValueError, TypeError):
            mapping[str(label)] = defaults.get(str(label), "#000000")
    return mapping


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def marker_reference_stats(frame: pd.DataFrame, markers: list[str], robust: bool) -> dict:
    """Centre and scale per marker, used to turn raw values into Z-scores.

    `robust` swaps mean/SD for median/MAD, which keeps a handful of very bright
    cells from inflating the scale and hiding the rest of the positive cells.
    """
    stats: dict[str, tuple[float, float]] = {}
    for marker in markers:
        values = pd.to_numeric(frame[marker], errors="coerce").to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        if values.size < 2:
            stats[marker] = (np.nan, np.nan)
            continue
        if robust:
            centre = float(np.median(values))
            scale = float(np.median(np.abs(values - centre)) * 1.4826)
            # A marker whose background is exactly zero in more than half the
            # cells has a MAD of 0, which would drop it entirely; the ordinary
            # SD still separates its positive cells.
            if not np.isfinite(scale) or scale <= 0:
                scale = float(values.std(ddof=0))
        else:
            centre = float(values.mean())
            scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            scale = np.nan
        stats[marker] = (centre, scale)
    return stats


def compute_composition(
    frame: pd.DataFrame,
    markers: list[str],
    phenotype_col: str,
    z_threshold: float,
    categories: list[str],
    stats: dict | None = None,
    robust: bool = False,
) -> dict:
    """Phenotype composition of the cells above `z_threshold`, marker by marker.

    Returns counts, the composition as a percentage of the high-expressing
    cells, the percentage of each phenotype that is high for the marker, and
    the phenotype abundances the composition should be read against.
    """
    stats = stats or marker_reference_stats(frame, markers, robust)
    labels = normalize_phenotype_labels(frame[phenotype_col])
    codes = pd.Categorical(labels, categories=categories).codes
    valid_codes = codes[codes >= 0]
    totals = np.bincount(valid_codes, minlength=len(categories)).astype("int64")

    counts = np.zeros((len(markers), len(categories)), dtype="int64")
    for row, marker in enumerate(markers):
        centre, scale = stats.get(marker, (np.nan, np.nan))
        if not np.isfinite(centre) or not np.isfinite(scale):
            continue
        values = pd.to_numeric(frame[marker], errors="coerce").to_numpy(dtype="float64")
        mask = np.isfinite(values) & (((values - centre) / scale) > z_threshold)
        selected = codes[mask]
        selected = selected[selected >= 0]
        if selected.size:
            counts[row] = np.bincount(selected, minlength=len(categories))

    counts_df = pd.DataFrame(counts, index=list(markers), columns=list(categories))
    n_high = counts_df.sum(axis=1)
    n_cells = int(totals.sum())

    share = counts_df.div(n_high.replace(0, np.nan), axis=0) * 100.0
    totals_series = pd.Series(totals, index=list(categories), dtype="int64")
    positivity = counts_df.div(totals_series.replace(0, np.nan), axis=1) * 100.0
    baseline = totals_series / n_cells * 100.0 if n_cells else totals_series.astype("float64")

    return {
        "counts": counts_df,
        "share": share.fillna(0.0),
        "positivity": positivity.fillna(0.0),
        "n_high": n_high,
        "totals": totals_series,
        "baseline": baseline,
        "n_cells": n_cells,
        "stats": stats,
    }


def order_markers(result: dict, markers: list[str], order: str) -> list[str]:
    """Order the bars: as configured, alphabetically, or by dominant phenotype."""
    if order == "Alphabetical":
        return sorted(markers, key=lambda m: get_analysis_marker_label(m).lower())
    if order == "Grouped by dominant phenotype":
        share = result["share"]
        rows = []
        for marker in markers:
            row = share.loc[marker]
            top = row.idxmax() if row.notna().any() and float(row.max()) > 0 else "~"
            rows.append((str(top).lower(), -float(row.max() if row.notna().any() else 0), marker))
        return [marker for _, _, marker in sorted(rows)]
    return list(markers)


def build_summary_table(
    result: dict,
    top_n: int,
    metric: str,
    min_cells: int,
) -> pd.DataFrame:
    """Top markers per phenotype among the high-expressing cells.

    `Share` answers "who are these bright cells?" and is what the figure draws.
    It is dominated by whichever phenotype is most abundant, so `Enrichment`
    (share against that phenotype's own abundance) is what tells you a marker
    is actually concentrated in a phenotype rather than merely common in it.
    """
    counts = result["counts"]
    share = result["share"]
    positivity = result["positivity"]
    baseline = result["baseline"]
    n_high = result["n_high"]

    rows: list[pd.DataFrame] = []
    for phenotype in counts.columns:
        base = float(baseline.get(phenotype, 0.0))
        table = pd.DataFrame({
            "Phenotype": phenotype,
            "Marker": [get_analysis_marker_label(m) for m in counts.index],
            "_marker": list(counts.index),
            "High cells (n)": counts[phenotype].to_numpy(),
            METRIC_SHARE: share[phenotype].to_numpy(),
            METRIC_POSITIVE: positivity[phenotype].to_numpy(),
            "Marker high cells (n)": n_high.to_numpy(),
            "Phenotype baseline (%)": base,
        })
        with np.errstate(divide="ignore", invalid="ignore"):
            table[METRIC_ENRICHMENT] = np.log2(
                np.where(base > 0, table[METRIC_SHARE].to_numpy() / base, np.nan)
            )
        keep = (table["High cells (n)"] >= min_cells) & (table["Marker high cells (n)"] >= min_cells)
        table = table[keep]
        if table.empty:
            continue
        table = table.sort_values(metric, ascending=False, kind="mergesort").head(max(1, top_n))
        table.insert(0, "Rank", range(1, len(table) + 1))
        rows.append(table)

    if not rows:
        return pd.DataFrame(columns=[
            "Rank", "Phenotype", "Marker", "_marker", "High cells (n)", METRIC_SHARE,
            METRIC_POSITIVE, METRIC_ENRICHMENT, "Marker high cells (n)", "Phenotype baseline (%)",
        ])
    out = pd.concat(rows, ignore_index=True)
    return out[[
        "Rank", "Phenotype", "Marker", "_marker", "High cells (n)", METRIC_SHARE,
        METRIC_POSITIVE, METRIC_ENRICHMENT, "Marker high cells (n)", "Phenotype baseline (%)",
    ]]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=48)
def composition_figure_png(
    share: pd.DataFrame,
    n_high: pd.Series,
    colors: tuple,
    title: str,
    legend_title: str,
    dpi: int,
    show_counts: bool,
) -> bytes:
    """Draw the stacked composition figure and return it as PNG bytes.

    Cached on its inputs so panning through strata or reopening the panel does
    not redraw a figure that has not changed.
    """
    color_map = dict(colors)
    markers = list(share.index)
    phenotypes = list(share.columns)
    positions = np.arange(len(markers), dtype="float64")

    width = max(9.0, min(46.0, 0.42 * len(markers) + 4.0))
    fig, ax = plt.subplots(figsize=(width, 7.0), dpi=dpi)
    fig.patch.set_facecolor("white")

    bottom = np.zeros(len(markers), dtype="float64")
    for phenotype in phenotypes:
        values = share[phenotype].to_numpy(dtype="float64")
        ax.bar(
            positions, values, bottom=bottom, width=0.8,
            color=color_map.get(str(phenotype), "#000000"),
            edgecolor="black", linewidth=0.4, label=str(phenotype),
        )
        bottom += values

    if show_counts:
        for pos, marker in zip(positions, markers):
            ax.text(
                pos, 101.0, f"n={int(n_high.get(marker, 0)):,}",
                rotation=90, ha="center", va="bottom", fontsize=6, color="#444444",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [get_analysis_marker_label(m) for m in markers],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_xlim(-0.6, len(markers) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Proportion of Cells (%)", fontsize=11)
    ax.set_xlabel("Markers", fontsize=12)
    # The count labels are drawn above the axes, so the title needs to move up
    # with them or the two collide.
    ax.set_title(title, fontsize=13, fontweight="bold", pad=34 if show_counts else 14)
    ax.tick_params(axis="y", labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(
        bbox_to_anchor=(1.01, 1.0), loc="upper left", title=legend_title,
        fontsize=8, title_fontsize=9, frameon=True, fancybox=True,
    )
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Scope: filters and stratification
# ---------------------------------------------------------------------------

def _render_filters(all_columns: list[str], column_values, key_prefix: str) -> list[dict]:
    """Column filters: a value picker for categories, a range for measurements."""
    filter_cols = st.multiselect(
        "Filter columns",
        options=all_columns,
        key=f"{key_prefix}_filter_cols",
        help="Restrict the cells before anything is measured. Leave empty to use every cell.",
    )

    filters: list[dict] = []
    for column in filter_cols:
        values = column_values(column)
        if not values:
            st.warning(f"`{column}` has no readable values - skipping this filter.")
            continue
        profile = profile_values(values)
        suffix = _safe_key(column)
        if profile["looks_continuous"]:
            numbers = profile["numbers"]
            low, high = float(min(numbers)), float(max(numbers))
            col_a, col_b = st.columns(2)
            chosen_low = col_a.number_input(
                f"`{column}` minimum", value=low, key=f"{key_prefix}_fmin_{suffix}",
                help=f"Continuous column ({profile['n_unique']:,} distinct values).",
            )
            chosen_high = col_b.number_input(
                f"`{column}` maximum", value=high, key=f"{key_prefix}_fmax_{suffix}",
            )
            filters.append({"column": column, "kind": "range",
                            "low": float(chosen_low), "high": float(chosen_high)})
        else:
            chosen = st.multiselect(
                f"Keep in `{column}` ({profile['n_unique']} levels)",
                options=values, default=values, key=f"{key_prefix}_fval_{suffix}",
            )
            if chosen and len(chosen) < len(values):
                filters.append({"column": column, "kind": "values", "values": list(chosen)})
            elif not chosen:
                st.caption(f"No value selected for `{column}` - that filter is ignored.")
    return filters


def apply_filters(frame: pd.DataFrame, filters: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Apply the filters in order, reporting what each one removed."""
    notes: list[str] = []
    for spec in filters:
        column = spec["column"]
        if column not in frame.columns:
            notes.append(f"`{column}` is not in the file - filter skipped.")
            continue
        before = len(frame)
        if spec["kind"] == "range":
            numeric = pd.to_numeric(frame[column], errors="coerce")
            frame = frame[numeric.between(spec["low"], spec["high"])]
            notes.append(f"`{column}` in [{spec['low']:g}, {spec['high']:g}]: {before:,} -> {len(frame):,}")
        else:
            wanted = {str(v) for v in spec["values"]}
            frame = frame[frame[column].astype(str).isin(wanted)]
            notes.append(f"`{column}` in {len(wanted)} level(s): {before:,} -> {len(frame):,}")
    return frame, notes


def _render_stratification(all_columns: list[str], column_values, key_prefix: str) -> dict:
    """Pick the column that splits the data into one figure per level."""
    options = ["(None - one figure over all cells)"] + list(all_columns)
    choice = st.selectbox(
        "Stratify by",
        options=options,
        key=f"{key_prefix}_strat_col",
        help="A categorical column such as patient or batch. One figure is produced per level.",
    )
    if choice == options[0]:
        return {"column": None, "levels": [], "pooled": True}

    values = column_values(choice)
    if not values:
        st.warning(f"`{choice}` has no readable values - falling back to a single figure.")
        return {"column": None, "levels": [], "pooled": True}

    profile = profile_values(values)
    if profile["looks_continuous"]:
        st.warning(
            f"`{choice}` holds {profile['n_unique']:,} distinct numbers, so it looks like a "
            "measurement rather than a grouping. Stratifying by it would produce one figure per "
            "value. Use it as a filter instead, or confirm below."
        )
        if not st.checkbox(
            f"Treat `{choice}` as categorical anyway",
            key=f"{key_prefix}_strat_force",
        ):
            return {"column": None, "levels": [], "pooled": True}
    elif profile["n_unique"] > MANY_LEVELS_WARNING:
        st.warning(
            f"`{choice}` has {profile['n_unique']} levels. Select the ones you want below - "
            "each becomes its own figure."
        )

    default_levels = values[: min(len(values), 8)]
    levels = st.multiselect(
        f"Levels of `{choice}` to plot",
        options=values, default=default_levels,
        key=f"{key_prefix}_strat_levels",
    )
    pooled = st.checkbox(
        "Also show a pooled figure over all filtered cells",
        value=True, key=f"{key_prefix}_strat_pooled",
        help="The pooled figure is what the marker ranking and the selection below are based on.",
    )
    if not levels:
        st.caption("No level selected - only the pooled figure will be produced.")
    return {"column": choice, "levels": list(levels), "pooled": True if not levels else pooled}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

TABLE_COLUMN_CONFIG = {
    "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
    "Phenotype": st.column_config.TextColumn("Phenotype"),
    "Marker": st.column_config.TextColumn("Marker"),
    "High cells (n)": st.column_config.NumberColumn(
        "High cells (n)", format="%d",
        help="Cells of this phenotype above the Z cut for this marker.",
    ),
    METRIC_SHARE: st.column_config.NumberColumn(
        METRIC_SHARE, format="%.1f",
        help="Percentage of this marker's high-expressing cells that are this phenotype - the height of the segment in the figure.",
    ),
    METRIC_POSITIVE: st.column_config.NumberColumn(
        METRIC_POSITIVE, format="%.1f",
        help="Percentage of this phenotype's own cells that are high for the marker. This is what decides whether a split is worth attempting.",
    ),
    METRIC_ENRICHMENT: st.column_config.NumberColumn(
        METRIC_ENRICHMENT, format="%.2f",
        help="Share against how abundant the phenotype is overall. 0 = exactly as expected, 1 = twice as concentrated as chance.",
    ),
    "Marker high cells (n)": st.column_config.NumberColumn("Marker high cells (n)", format="%d"),
    "Phenotype baseline (%)": st.column_config.NumberColumn(
        "Phenotype baseline (%)", format="%.1f",
        help="Percentage of all cells in scope that carry this phenotype.",
    ),
}


def _figure_title(z_threshold: float, robust: bool, group_label: str | None) -> str:
    scale = "robust Z" if robust else "Z"
    title = f"Phenotype Composition of High-Expressing Cells ({scale} > {z_threshold:g})"
    return f"{title}\n{group_label}" if group_label else title


def _stratum_support(groups: list[dict], top_n: int, metric: str, min_cells: int) -> dict:
    """How many strata rank a phenotype-marker pair in their own top list."""
    support: dict[tuple[str, str], int] = {}
    for group in groups:
        if group.get("is_pooled"):
            continue
        table = build_summary_table(group["result"], top_n, metric, min_cells)
        for _, row in table.iterrows():
            key = (str(row["Phenotype"]), str(row["_marker"]))
            support[key] = support.get(key, 0) + 1
    return support


def render_marker_composition_panel(
    *,
    project_root,
    phenotype_col: str,
    marker_cols: list[str],
    all_columns: list[str],
    selected_parents: list[str],
    chosen_markers: dict,
    load_frame,
    column_values,
    key_prefix: str = "subpheno_comp",
    editor_key_fn=None,
) -> None:
    """Render the high-expression composition panel.

    Parameters
    ----------
    load_frame : callable
        ``(tuple[str, ...]) -> pd.DataFrame | None``; reads the named columns
        from the active file. The frame is never modified in place.
    column_values : callable
        ``(str) -> list[str]``; the sorted unique values of one column.
    chosen_markers : dict
        The shared ``phenotype -> markers`` map the sub-clustering step reads.
    editor_key_fn : callable or None
        ``(phenotype) -> str`` giving the discovery table's widget key, cleared
        when markers are applied here so both views agree.
    """
    results_key = f"{key_prefix}_composition_results"
    stored = st.session_state.get(results_key)

    st.caption(
        "For every marker in the panel, the cells above a Z-score cut are taken and split by "
        f"`{phenotype_col}`. A marker whose bright cells sit in one phenotype is a candidate for "
        "splitting that phenotype; a marker spread evenly across lineages mostly separates cell "
        "types you have already resolved."
    )

    known_labels = _sort_labels(
        list(normalize_phenotype_labels(pd.Series(column_values(phenotype_col), dtype="object")))
        + list((stored or {}).get("categories", []))
    )

    with st.expander("Scope, thresholds and colours", expanded=stored is None):
        st.markdown("**Cells to include**")
        filters = _render_filters(all_columns, column_values, key_prefix)
        strat = _render_stratification(all_columns, column_values, key_prefix)

        st.markdown("**Threshold**")
        thr1, thr2, thr3 = st.columns(3)
        z_threshold = float(thr1.number_input(
            "Z-score cut", -3.0, 10.0, DEFAULT_Z_THRESHOLD, 0.1, key=f"{key_prefix}_z",
            help="A cell counts as high for a marker when its Z-score is above this value.",
        ))
        min_cells = int(thr2.number_input(
            "Minimum cells", 1, 100000, 10, key=f"{key_prefix}_min_cells",
            help="Markers with fewer high-expressing cells are left blank in the figure, and "
                 "phenotype-marker pairs below this count are kept out of the table.",
        ))
        robust = bool(thr3.checkbox(
            "Robust Z (median / MAD)", value=False, key=f"{key_prefix}_robust",
            help="Keeps a handful of very bright cells from inflating the scale and hiding the "
                 "rest of the positive cells.",
        ))
        if strat["column"]:
            z_scope = st.radio(
                "Z-scores computed", ["Within each stratum", "Across all filtered cells"],
                horizontal=True, key=f"{key_prefix}_z_scope",
                help="Within each stratum makes every figure self-contained. Across all cells "
                     "keeps one common scale, so a stratum that stained weakly shows fewer high cells.",
            )
        else:
            z_scope = "Across all filtered cells"

        st.markdown("**Figure**")
        fig1, fig2, fig3 = st.columns(3)
        marker_order = fig1.selectbox("Marker order", MARKER_ORDERS, key=f"{key_prefix}_order")
        dpi = int(fig2.number_input("DPI", 100, 600, 200, 50, key=f"{key_prefix}_dpi"))
        show_counts = bool(fig3.checkbox(
            "Annotate cell counts", value=False, key=f"{key_prefix}_show_counts",
            help="Prints the number of high-expressing cells above each bar.",
        ))

        st.markdown("**Summary table**")
        sum1, sum2 = st.columns(2)
        top_n = int(sum1.number_input(
            "Top markers per phenotype", 1, 40, 8, key=f"{key_prefix}_top_n",
        ))
        metric = sum2.selectbox(
            "Rank markers by", SUMMARY_METRICS, key=f"{key_prefix}_metric",
            help="Share answers 'who are these bright cells?' and is dominated by whichever "
                 "phenotype is most abundant. Enrichment corrects for that abundance.",
        )

        st.markdown("**Phenotype colours**")
        colors = resolve_phenotype_colors(known_labels, key_prefix)
        st.caption(
            "Defaults follow the house phenotype palette; anything outside it is given a colour "
            "chosen to look unlike the ones already in use."
        )
        color_columns = st.columns(4)
        for index, label in enumerate(known_labels):
            with color_columns[index % len(color_columns)]:
                st.color_picker(label, key=f"{key_prefix}_color_{_safe_key(label)}")
        if st.button("Reset colours to defaults", key=f"{key_prefix}_reset_colors"):
            for label in known_labels:
                st.session_state.pop(f"{key_prefix}_color_{_safe_key(label)}", None)
            st.rerun()

    colors = resolve_phenotype_colors(known_labels, key_prefix)
    run_params = {
        "phenotype_col": phenotype_col,
        "markers": list(marker_cols),
        "filters": filters,
        "stratify": strat,
        "z_threshold": z_threshold,
        "robust": robust,
        "z_scope": z_scope,
    }

    if st.button("Build Composition Figures", type="primary", key=f"{key_prefix}_run"):
        needed = list(dict.fromkeys(
            list(marker_cols) + [phenotype_col]
            + [spec["column"] for spec in filters]
            + ([strat["column"]] if strat["column"] else [])
        ))
        with st.spinner("Reading columns..."):
            frame = load_frame(tuple(needed))
        if frame is None:
            st.error("Could not read the required columns from the input file.")
            return
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            st.error("Missing from the input file: " + ", ".join(missing))
            return

        frame, notes = apply_filters(frame, filters)
        if frame.empty:
            st.error("No cells are left after filtering.")
            return

        categories = _sort_labels(normalize_phenotype_labels(frame[phenotype_col]).unique())
        with st.spinner("Scoring markers..."):
            pooled = compute_composition(
                frame, list(marker_cols), phenotype_col, z_threshold, categories, robust=robust,
            )
            groups: list[dict] = []
            if strat["pooled"] or not strat["column"] or not strat["levels"]:
                groups.append({
                    "label": "All filtered cells" if strat["column"] else "All cells",
                    "result": pooled, "is_pooled": True,
                })
            shared_stats = pooled["stats"] if z_scope == "Across all filtered cells" else None
            for level in (strat["levels"] if strat["column"] else []):
                subset = frame[frame[strat["column"]].astype(str) == str(level)]
                if len(subset) < max(min_cells, 2):
                    st.warning(f"{strat['column']} = {level}: only {len(subset):,} cells - skipped.")
                    continue
                groups.append({
                    "label": f"{strat['column']} = {level}",
                    "result": compute_composition(
                        subset, list(marker_cols), phenotype_col, z_threshold, categories,
                        stats=shared_stats, robust=robust,
                    ),
                    "is_pooled": False,
                })

        st.session_state[results_key] = {
            "pooled": pooled, "groups": groups, "categories": categories,
            "markers": list(marker_cols), "params": run_params, "notes": notes,
            "n_cells": int(len(frame)),
        }
        stored = st.session_state[results_key]

    if not stored:
        st.info("Set the scope above, then press **Build Composition Figures**.")
        return

    if stored["params"] != run_params:
        st.info("Settings have changed since these figures were built - press **Build Composition Figures** to refresh.")

    stored_params = stored["params"]
    st.caption(
        f"{stored['n_cells']:,} cells · {'robust ' if stored_params['robust'] else ''}"
        f"Z > {stored_params['z_threshold']:g} · `{stored_params['phenotype_col']}` · "
        f"{len(stored['markers'])} markers · {len(stored['groups'])} figure(s)"
    )
    for note in stored.get("notes", []):
        st.caption(f"Filter: {note}")

    categories = list(stored["categories"])
    colors = resolve_phenotype_colors(_sort_labels(known_labels + categories), key_prefix)
    color_key = tuple((label, colors.get(label, "#000000")) for label in categories)
    plot_markers = order_markers(stored["pooled"], stored["markers"], marker_order)
    output_dir = Path(project_root) / "outputs/phenotyping/marker_composition"

    saved_files: list[str] = []
    save_all = st.button(
        "Save all figures and tables", key=f"{key_prefix}_save_all",
        help=f"Writes PNG and CSV files to {output_dir}",
    )

    for index, group in enumerate(stored["groups"]):
        result = group["result"]
        with st.expander(
            f"{group['label']} — {result['n_cells']:,} cells", expanded=(index == 0)
        ):
            share = result["share"].reindex(index=plot_markers, columns=categories).fillna(0.0)
            n_high = result["n_high"].reindex(plot_markers).fillna(0).astype("int64")
            thin = n_high[n_high < min_cells].index.tolist()
            if thin:
                share.loc[thin] = 0.0

            png = composition_figure_png(
                share, n_high, color_key,
                _figure_title(stored_params["z_threshold"], stored_params["robust"],
                              None if group.get("is_pooled") and len(stored["groups"]) == 1 else group["label"]),
                "Phenotype", dpi, show_counts,
            )
            st.image(png, width="stretch")

            stem = f"composition_{_safe_key(group['label'])}"
            st.download_button(
                "Download figure (PNG)", data=png, file_name=f"{stem}.png", mime="image/png",
                key=f"{key_prefix}_dl_png_{index}",
            )
            if thin:
                st.caption(
                    f"Left blank (fewer than {min_cells} high-expressing cells): "
                    + ", ".join(get_analysis_marker_label(m) for m in thin)
                )

            table = build_summary_table(result, top_n, metric, min_cells)
            if table.empty:
                st.info("No phenotype-marker pair clears the minimum cell count here.")
            else:
                st.dataframe(
                    table.drop(columns=["_marker"]), width="stretch", hide_index=True,
                    column_config=TABLE_COLUMN_CONFIG,
                )
                st.download_button(
                    "Download table (CSV)",
                    data=table.drop(columns=["_marker"]).to_csv(index=False).encode("utf-8"),
                    file_name=f"{stem}_top_markers.csv", mime="text/csv",
                    key=f"{key_prefix}_dl_csv_{index}",
                )

            if save_all:
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / f"{stem}.png").write_bytes(png)
                    table.drop(columns=["_marker"]).to_csv(output_dir / f"{stem}_top_markers.csv", index=False)
                    result["counts"].to_csv(output_dir / f"{stem}_high_cell_counts.csv")
                    saved_files.append(stem)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not save {stem}: {exc}")

    if save_all and saved_files:
        st.success(f"Saved {len(saved_files)} figure(s) and their tables to {output_dir}")

    _render_marker_picker(
        stored=stored, selected_parents=selected_parents, chosen_markers=chosen_markers,
        top_n=top_n, metric=metric, min_cells=min_cells, key_prefix=key_prefix,
        editor_key_fn=editor_key_fn,
    )


def _render_marker_picker(
    *,
    stored: dict,
    selected_parents: list[str],
    chosen_markers: dict,
    top_n: int,
    metric: str,
    min_cells: int,
    key_prefix: str,
    editor_key_fn=None,
) -> None:
    """Turn the ranking into the marker set the sub-clustering step will use."""
    st.markdown("##### Send markers to sub-clustering")

    note = st.session_state.pop(f"{key_prefix}_applied_note", None)
    if note:
        st.success(f"Applied — {note}. Step 3 will use these markers.")

    pooled_table = build_summary_table(stored["pooled"], top_n, metric, min_cells)
    strata = [group for group in stored["groups"] if not group.get("is_pooled")]
    support = _stratum_support(stored["groups"], top_n, metric, min_cells) if strata else {}

    ranked: dict[str, list[str]] = {}
    annotations: dict[tuple[str, str], str] = {}
    for _, row in pooled_table.iterrows():
        phenotype, marker = str(row["Phenotype"]), str(row["_marker"])
        ranked.setdefault(phenotype, []).append(marker)
        label = f"{row['Marker']} — {metric.split(' (')[0].lower()} {row[metric]:.2f}, {int(row['High cells (n)']):,} cells"
        if strata:
            label += f", {support.get((phenotype, marker), 0)}/{len(strata)} strata"
        annotations[(phenotype, marker)] = label

    if strata:
        st.caption(
            f"Rankings come from the pooled figure. `k/{len(strata)} strata` counts how many "
            "individual strata also rank that marker in their own top list - a marker that only "
            "shows up in one stratum is usually a batch or patient effect, not a functional state."
        )

    # The phenotypes offered above come from the raw column values while the
    # ranking is keyed by the tidied labels, so match them case- and
    # whitespace-insensitively rather than dropping a phenotype's ranking.
    canonical = {name.strip().lower(): name for name in ranked}

    version = int(st.session_state.get(f"{key_prefix}_pick_version", 0))
    default_mode = st.session_state.pop(f"{key_prefix}_pick_mode", "keep")

    selections: dict[str, list[str]] = {}
    for phenotype in selected_parents:
        name = str(phenotype)
        ranking_key = canonical.get(name.strip().lower(), name)
        ordered = ranked.get(ranking_key, [])
        options = ordered + [m for m in stored["markers"] if m not in ordered]
        if default_mode == "clear":
            default = []
        elif default_mode == "top":
            default = ordered[:top_n]
        else:
            current = [m for m in chosen_markers.get(name, []) if m in options]
            default = current or ordered[:top_n]
        if not ordered:
            st.caption(f"**{name}** — no marker clears the minimum cell count; pick manually below.")
        selections[name] = st.multiselect(
            f"{name} — markers to subcluster on",
            options=options,
            default=default,
            key=f"{key_prefix}_pick_{_safe_key(name)}_{version}",
            format_func=lambda marker, p=ranking_key: annotations.get(
                (p, marker), get_analysis_marker_label(marker)
            ),
        )

    act1, act2, act3 = st.columns(3)
    if act1.button("Apply to Sub-Clustering", type="primary", key=f"{key_prefix}_apply"):
        applied = []
        for phenotype, picks in selections.items():
            if picks:
                chosen_markers[str(phenotype)] = list(picks)
                applied.append(f"{phenotype}: {len(picks)} markers")
            else:
                chosen_markers.pop(str(phenotype), None)
            # The discovery table holds its own tick boxes; clearing its widget
            # state makes it redraw from the markers just applied, so the two
            # views cannot disagree about what will be clustered.
            if editor_key_fn is not None:
                st.session_state.pop(editor_key_fn(phenotype), None)
        st.session_state[f"{key_prefix}_applied_note"] = "; ".join(applied) or "all selections cleared"
        st.rerun()
    if act2.button("Fill with top-ranked", key=f"{key_prefix}_fill_top"):
        st.session_state[f"{key_prefix}_pick_mode"] = "top"
        st.session_state[f"{key_prefix}_pick_version"] = version + 1
        st.rerun()
    if act3.button("Clear selections", key=f"{key_prefix}_clear_picks"):
        st.session_state[f"{key_prefix}_pick_mode"] = "clear"
        st.session_state[f"{key_prefix}_pick_version"] = version + 1
        st.rerun()

    in_use = [f"{p}: {len(chosen_markers.get(str(p), []))}" for p in selected_parents
              if chosen_markers.get(str(p))]
    st.caption(
        ("Currently set for sub-clustering — " + "; ".join(in_use))
        if in_use else
        "No markers are set for sub-clustering yet. Select above and press Apply."
    )
