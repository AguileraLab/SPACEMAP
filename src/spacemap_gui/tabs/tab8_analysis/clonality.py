"""
Tab 8 – Clonality subtab

Provides an interactive GUI for subclone discovery, diversity analysis,
clone clustering, and publication-ready figure generation.

All panels use explicit "Run" buttons so nothing auto-renders while
the user is configuring parameters.
"""

from __future__ import annotations

import io
import re
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

# ---------------------------------------------------------------------------
# Session-state key helpers
# ---------------------------------------------------------------------------
_KEY_PREFIX = "tab8_clonality_"


def _sk(suffix: str) -> str:
    """Build a namespaced session-state key."""
    return f"{_KEY_PREFIX}{suffix}"


def _default_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    return options.index(preferred) if preferred in options else fallback


def _preferred_or_blank(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else ""


def _safe_unique(series: "pd.Series | pd.DataFrame") -> list[str]:
    """Return sorted unique string values, handling both Series and DataFrame."""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    vals = series.dropna().astype(str).unique().tolist()
    try:
        return sorted(vals)
    except Exception:
        return vals


def _describe_params(**kwargs: Any) -> str:
    """Build a concise annotation string for figure titles."""
    parts = []
    for k, v in kwargs.items():
        if v is None or (isinstance(v, (list, tuple)) and not v):
            continue
        if isinstance(v, (list, tuple)):
            parts.append(f"{k}: {', '.join(str(x) for x in v[:5])}")
        else:
            parts.append(f"{k}: {v}")
    return " | ".join(parts) if parts else ""


def _safe_filename(*parts: str, ext: str = ".png") -> str:
    """Join parts into a filesystem-safe filename."""
    raw = "_".join(str(p) for p in parts if p)
    safe = re.sub(r"[^\w\-.]", "_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return (safe[:120] + ext) if safe else ("figure" + ext)


# ---------------------------------------------------------------------------
# Core diversity math (self-contained, no external dependency on code9)
# ---------------------------------------------------------------------------

def _shannon_index(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _simpson_index(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return np.nan
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def _pielou_evenness(shannon: float, richness: int) -> float:
    if pd.isna(shannon) or richness <= 1:
        return np.nan
    return shannon / np.log(richness)


def _significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _format_p(p: float) -> str:
    if pd.isna(p):
        return "p = NA"
    text = f"p = {p:.4f}" if p < 0.001 else f"p = {p:.3f}"
    stars = _significance_stars(p)
    return f"{text} {stars}".strip()


def _standard_error(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=1) / np.sqrt(len(s)))


def _safe_mode(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return pd.NA
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else pd.NA


# ---------------------------------------------------------------------------
# Diversity computation engine
# ---------------------------------------------------------------------------

def _compute_diversity_table(
    df: pd.DataFrame,
    group_cols: list[str],
    subphenotype_col: str,
    min_cells: int,
    metadata_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (metrics_df, composition_df) for the given grouping."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    clone_counts = (
        df.groupby(group_cols + [subphenotype_col], dropna=False)
        .size()
        .reset_index(name="cell_count")
    )
    totals = (
        clone_counts.groupby(group_cols, dropna=False)["cell_count"]
        .sum()
        .reset_index(name="cells_in_group")
    )
    totals["passes"] = totals["cells_in_group"] >= min_cells
    clone_counts = clone_counts.merge(totals, on=group_cols, how="left")
    clone_counts = clone_counts[clone_counts["passes"]].drop(columns=["passes"])
    if clone_counts.empty:
        return pd.DataFrame(), pd.DataFrame()

    clone_counts["clone_fraction"] = clone_counts["cell_count"] / clone_counts["cells_in_group"]

    rows = []
    for gvals, gdf in clone_counts.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(gvals, tuple):
            gvals = (gvals,)
        c = gdf["cell_count"].to_numpy(dtype=float)
        total = int(c.sum())
        richness = int((c > 0).sum())
        h = _shannon_index(c)
        row: dict[str, Any] = {}
        for col, val in zip(group_cols, gvals):
            row[col] = val
        row["cells_in_group"] = total
        row["clone_richness"] = richness
        row["shannon_index"] = h
        row["effective_clone_number"] = np.exp(h) if pd.notna(h) else np.nan
        row["pielou_evenness"] = _pielou_evenness(h, richness)
        row["simpson_gini"] = _simpson_index(c)
        max_idx = gdf["cell_count"].idxmax()
        row["dominant_subclone"] = gdf.loc[max_idx, subphenotype_col] if pd.notna(max_idx) else pd.NA
        row["dominant_fraction"] = float(gdf["cell_count"].max() / total) if total > 0 else np.nan
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    if metadata_cols:
        avail = [c for c in metadata_cols if c in df.columns and c not in group_cols]
        if avail:
            meta = df.groupby(group_cols, dropna=False)[avail].agg(_safe_mode).reset_index()
            metrics_df = metrics_df.merge(meta, on=group_cols, how="left")

    # Build group label
    metrics_df["group_label"] = metrics_df.apply(
        lambda r: " | ".join(str(r[c]) for c in group_cols), axis=1
    )

    clone_counts["group_label"] = clone_counts.apply(
        lambda r: " | ".join(str(r[c]) for c in group_cols), axis=1
    )

    return metrics_df, clone_counts


# ---------------------------------------------------------------------------
# Statistical test helpers
# ---------------------------------------------------------------------------

def _welch_t(df: pd.DataFrame, group_col: str, val_col: str, order: list | None = None) -> dict:
    valid = df[[group_col, val_col]].dropna()
    levels = order if order else list(valid[group_col].drop_duplicates())
    levels = [l for l in levels if l in set(valid[group_col])]
    if len(levels) != 2:
        return {"method": "Welch t-test", "p_value": np.nan, "n": len(valid)}
    a = valid.loc[valid[group_col] == levels[0], val_col]
    b = valid.loc[valid[group_col] == levels[1], val_col]
    if min(len(a), len(b)) < 2:
        return {"method": "Welch t-test", "p_value": np.nan, "n": len(valid)}
    result = stats.ttest_ind(a, b, equal_var=False)
    return {"method": "Welch t-test", "p_value": float(np.asarray(result[1]).item()), "n": len(valid), "statistic": float(np.asarray(result[0]).item())}


def _mann_whitney(df: pd.DataFrame, group_col: str, val_col: str, order: list | None = None) -> dict:
    valid = df[[group_col, val_col]].dropna()
    levels = order if order else list(valid[group_col].drop_duplicates())
    levels = [l for l in levels if l in set(valid[group_col])]
    if len(levels) != 2:
        return {"method": "Mann-Whitney U", "p_value": np.nan, "n": len(valid)}
    a = valid.loc[valid[group_col] == levels[0], val_col].to_numpy(dtype=float)
    b = valid.loc[valid[group_col] == levels[1], val_col].to_numpy(dtype=float)
    if min(len(a), len(b)) < 2:
        return {"method": "Mann-Whitney U", "p_value": np.nan, "n": len(valid)}
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"method": "Mann-Whitney U", "p_value": float(np.asarray(result[1]).item()), "n": len(valid), "statistic": float(np.asarray(result[0]).item())}


def _kruskal_wallis(df: pd.DataFrame, group_col: str, val_col: str, order: list | None = None) -> dict:
    valid = df[[group_col, val_col]].dropna()
    levels = order if order else list(valid[group_col].drop_duplicates())
    levels = [l for l in levels if l in set(valid[group_col])]
    groups = [valid.loc[valid[group_col] == l, val_col].to_numpy(dtype=float) for l in levels]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {"method": "Kruskal-Wallis", "p_value": np.nan, "n": len(valid)}
    result = stats.kruskal(*groups)
    return {"method": "Kruskal-Wallis", "p_value": float(np.asarray(result[1]).item()), "n": len(valid), "statistic": float(np.asarray(result[0]).item())}


def _one_way_anova(df: pd.DataFrame, group_col: str, val_col: str, order: list | None = None) -> dict:
    valid = df[[group_col, val_col]].dropna()
    levels = order if order else list(valid[group_col].drop_duplicates())
    levels = [l for l in levels if l in set(valid[group_col])]
    groups = [valid.loc[valid[group_col] == l, val_col].to_numpy(dtype=float) for l in levels]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {"method": "One-way ANOVA", "p_value": np.nan, "n": len(valid)}
    result = stats.f_oneway(*groups)
    return {"method": "One-way ANOVA", "p_value": float(np.asarray(result[1]).item()), "n": len(valid), "statistic": float(np.asarray(result[0]).item())}


_STAT_METHODS = {
    "Welch t-test": _welch_t,
    "Mann-Whitney U": _mann_whitney,
    "One-way ANOVA": _one_way_anova,
    "Kruskal-Wallis": _kruskal_wallis,
}


# ---------------------------------------------------------------------------
# Publication style helper
# ---------------------------------------------------------------------------

def _set_pub_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f3f4fb",
        "axes.edgecolor": "#4c4c4c",
        "axes.linewidth": 1.0,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.8,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#d0d0d0",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def _fig_to_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _show_fig(session_key: str) -> bool:
    """Display a cached figure from session state, handling bytes/str safely.

    Returns True if the figure was displayed, False otherwise.
    """
    raw = st.session_state.get(session_key)
    if raw is None:
        return False
    if isinstance(raw, bytes):
        st.image(io.BytesIO(raw), width="stretch")
        return True
    # Stale or invalid value — remove it
    del st.session_state[session_key]
    return False


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def _choose_k(feature_df: pd.DataFrame, fixed_k: int | None, candidates: range, random_state: int) -> tuple[int | None, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as _sil

    if feature_df.empty or len(feature_df) < 3:
        return None, np.nan
    if fixed_k is not None:
        return min(fixed_k, len(feature_df) - 1), np.nan
    valid = [k for k in candidates if 2 <= k < len(feature_df)]
    if not valid:
        return None, np.nan
    best_k, best_s = None, -np.inf
    for k in valid:
        m = KMeans(n_clusters=k, random_state=random_state, n_init=50)
        labels = m.fit_predict(feature_df)
        if len(np.unique(labels)) < 2:
            continue
        s = _sil(feature_df, labels)
        if s > best_s:
            best_s, best_k = s, k
    return (best_k, float(best_s)) if best_k else (None, np.nan)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_clonality(project_root: Path, df: pd.DataFrame, csv_path: str):
    st.subheader("Clonality & Sub-phenotype Diversity")
    st.caption(
        "Analyse subclone composition, compute ecological diversity metrics "
        "(Shannon, Simpson, Pielou), cluster regions by clone proportions, "
        "and generate publication-ready figures — all driven by columns in "
        "the loaded unified CSV."
    )

    if df.empty or len(df.columns) < 3:
        st.warning("The loaded data does not have enough columns for clonality analysis.")
        return

    all_cols = list(df.columns)

    # =================================================================
    #  Configuration  (regular widgets — changes here trigger a rerun
    #  so that downstream dynamic multiselects always have fresh options)
    # =================================================================
    with st.expander("\u2699\ufe0f Configuration", expanded=True):
        st.markdown(
            "Select the column mapping that describes your phenotype hierarchy, "
            "region identifiers, and any optional metadata columns for stratification."
        )

        c1, c2 = st.columns(2)
        with c1:
            phenotype_options = ["None"] + all_cols
            phenotype_col = st.selectbox(
                "Phenotype column",
                options=phenotype_options,
                index=_default_index(phenotype_options, "phenotype", 0),
                key=_sk("phenotype_col"),
                help="Broad phenotype label used to filter cells (e.g. Epithelial, Immune).",
            )
            subphenotype_options = [""] + all_cols
            subphenotype_col = st.selectbox(
                "Sub-phenotype / subclone column",
                options=subphenotype_options,
                index=_default_index(subphenotype_options, _preferred_or_blank(subphenotype_options, "sub_phenotype"), 0),
                key=_sk("subphenotype_col"),
                format_func=lambda x: x if x else "(not selected)",
                help="Column containing the subclone or sub-phenotype labels.",
            )
        with c2:
            batch_options = [""] + all_cols
            batch_col = st.selectbox(
                "Batch / sample column",
                options=batch_options,
                index=_default_index(batch_options, _preferred_or_blank(batch_options, "batch_id"), 0),
                key=_sk("batch_col"),
                format_func=lambda x: x if x else "(not selected)",
                help="Column that identifies each sample, slide, or batch.",
            )
            region_col = st.selectbox(
                "Region / ROI column",
                options=["None"] + all_cols,
                index=_default_index(["None"] + all_cols, "tumor_roi", 0),
                key=_sk("region_col"),
                help=(
                    "Column identifying individual regions within a batch. "
                    "Set to None for batch-level analysis only."
                ),
            )

        meta_candidates = [
            c for c in all_cols
            if c not in {phenotype_col, subphenotype_col, batch_col, region_col}
        ]
        metadata_cols: list[str] = st.multiselect(
            "Metadata columns (for stratification)",
            options=meta_candidates,
            default=[c for c in ["patient", "arm", "response", "timepoint"] if c in meta_candidates],
            key=_sk("metadata_cols"),
            help="Additional columns propagated to results tables and used for stratified figures.",
        )

        # ── Validate all selected columns actually exist in the data ──
        # (session state may persist stale values from a previously loaded CSV)
        _col_set = set(df.columns)
        if phenotype_col != "None" and phenotype_col not in _col_set:
            phenotype_col = "None"
        if not subphenotype_col:
            st.warning("Please select a valid sub-phenotype column above.")
            return
        if subphenotype_col not in _col_set:
            st.warning(
                f"The selected sub-phenotype column **{subphenotype_col}** does not exist "
                "in the loaded CSV.  Please select a valid column above."
            )
            return
        if not batch_col:
            st.warning("Please select a valid batch column above.")
            return
        if batch_col not in _col_set:
            st.warning(
                f"The selected batch column **{batch_col}** does not exist "
                "in the loaded CSV.  Please select a valid column above."
            )
            return
        if region_col != "None" and region_col not in _col_set:
            region_col = "None"
        # Drop stale metadata columns that no longer exist
        metadata_cols = [c for c in metadata_cols if c in _col_set]

        st.divider()

        # ── Dynamic target phenotype multiselect ──
        c3, c4, c5 = st.columns(3)
        with c3:
            if phenotype_col != "None" and phenotype_col in df.columns:
                _pheno_opts = _safe_unique(df[phenotype_col])
                target_phenotypes: list[str] = st.multiselect(
                    "Target phenotype value(s)",
                    options=_pheno_opts,
                    default=[],
                    key=_sk("target_phenotypes"),
                    help=(
                        "Keep only cells whose phenotype column matches one of these values. "
                        "Leave empty to include **all** cells."
                    ),
                )
            else:
                target_phenotypes = []
                st.caption("Set a phenotype column above to enable target filtering.")

        # ── Dynamic excluded sub-phenotypes multiselect ──
        with c4:
            _sub_opts = _safe_unique(df[subphenotype_col]) if subphenotype_col in df.columns else []
            excluded_subs: list[str] = st.multiselect(
                "Excluded sub-phenotypes",
                options=_sub_opts,
                default=[v for v in ["Others", "Unknown"] if v in _sub_opts],
                key=_sk("excluded_subs"),
                help="Sub-phenotype values to exclude from the analysis.",
            )

        with c5:
            min_cells: int = int(
                st.number_input(
                    "Min cells per group",
                    min_value=1,
                    value=20,
                    step=5,
                    key=_sk("min_cells"),
                    help="Groups with fewer cells than this threshold are excluded from computations.",
                )
            )

    # =================================================================
    #  Build working dataframe
    # =================================================================
    work = df.copy()
    if phenotype_col != "None" and phenotype_col in work.columns and target_phenotypes:
        work = work[work[phenotype_col].astype(str).isin(target_phenotypes)]
    if excluded_subs and subphenotype_col in work.columns:
        work = work[~work[subphenotype_col].astype(str).isin(excluded_subs)]
    if subphenotype_col not in work.columns:
        st.warning(
            f"Column **{subphenotype_col}** not found in the data. "
            "Please select a valid sub-phenotype column in Configuration."
        )
        return
    work = work[work[subphenotype_col].notna()].copy()
    work[subphenotype_col] = work[subphenotype_col].astype(str).str.strip()
    work = work[work[subphenotype_col] != ""].copy()

    st.caption(f"Working cells after phenotype / exclusion filters: **{len(work):,}**")
    if work.empty:
        st.warning("No cells remain after filtering. Adjust the Configuration above.")
        return

    unique_subclones = _safe_unique(work[subphenotype_col])
    st.caption(
        f"Unique sub-phenotypes detected: **{len(unique_subclones)}** \u2014 "
        + ", ".join(unique_subclones[:15])
        + ("\u2026" if len(unique_subclones) > 15 else "")
    )

    # Helper: columns suitable for inline panel filters
    _filterable_cols = [
        c for c in all_cols
        if c in work.columns and work[c].nunique(dropna=True) <= 200
    ]

    # =================================================================
    #  Panel 1 \u2014 Subclone Proportion Overview
    # =================================================================
    with st.expander("\U0001f4ca Panel 1 \u2014 Subclone Proportion Overview", expanded=False):
        st.info(
            "Stacked bar chart showing the relative composition of sub-phenotypes "
            "across groups you define. You can manually choose which columns to "
            "group by and add row-level filters (e.g. show only tumor_bed == 1).",
            icon="\u2139\ufe0f",
        )

        with st.form(_sk("p1_form")):
            p1g1, p1g2 = st.columns(2)
            with p1g1:
                p1_group_cols: list[str] = st.multiselect(
                    "Group-by columns",
                    options=all_cols,
                    default=[batch_col],
                    key=_sk("p1_group_cols"),
                    help="Rows are grouped by the combination of these columns to form the bar-chart x-axis.",
                )
            with p1g2:
                p1_top_n: int = int(
                    st.slider(
                        "Max groups to display",
                        min_value=5,
                        max_value=100,
                        value=30,
                        key=_sk("p1_top_n"),
                        help="Limits the number of bars shown (groups are sorted first).",
                    )
                )

            p1_sort = st.radio(
                "Sort groups by",
                ["Name", "Total cells (desc)", "Shannon index (desc)"],
                key=_sk("p1_sort"),
                horizontal=True,
                help="Determines group ordering on the x-axis.",
            )

            # ── Inline row filters (up to 3) ──
            st.markdown(
                "**Additional row filters** *(optional, applied only to this panel)*  \n"
                "After changing a filter column, click **Generate** once to refresh the available values."
            )
            _p1_filters: list[tuple[str, list[str]]] = []
            for _fi in range(3):
                _fc_c, _fv_c = st.columns(2)
                with _fc_c:
                    _fc = st.selectbox(
                        f"Filter column {_fi + 1}",
                        options=["None"] + _filterable_cols,
                        key=_sk(f"p1_fc{_fi}"),
                        help="Pick a column to restrict rows for this panel.",
                    )
                with _fv_c:
                    if _fc != "None" and _fc in work.columns:
                        _fv_opts = _safe_unique(work[_fc])
                        _fv = st.multiselect(
                            f"Keep values ({_fc})",
                            options=_fv_opts,
                            default=_fv_opts,
                            key=_sk(f"p1_fv{_fi}"),
                            help="Only rows matching these values are included.",
                        )
                        _p1_filters.append((_fc, _fv))
                    else:
                        st.write("")  # spacer

            p1c_fw, _ = st.columns(2)
            with p1c_fw:
                p1_figw: int = int(
                    st.slider("Figure width", 8, 28, 14, key=_sk("p1_figw"), help="Width of the output figure in inches.")
                )

            p1_submitted = st.form_submit_button("\U0001f504 Generate Proportion Chart", type="primary")

        # ── Compute on submit, store in session_state ──
        if p1_submitted:
            _set_pub_style()
            g_cols = list(dict.fromkeys(p1_group_cols)) if p1_group_cols else [batch_col]
            p1_work = work.copy()
            for _fc, _fv in _p1_filters:
                if _fv:
                    p1_work = p1_work[p1_work[_fc].astype(str).isin([str(v) for v in _fv])]
            if p1_work.empty:
                st.warning("No rows remain after panel filters.")
            else:
                metrics, comp = _compute_diversity_table(p1_work, g_cols, subphenotype_col, min_cells, metadata_cols)
                if comp.empty:
                    st.warning("No groups passed the minimum cell threshold.")
                else:
                    pivot = comp.pivot_table(
                        index="group_label",
                        columns=subphenotype_col,
                        values="clone_fraction",
                        aggfunc="sum",
                        fill_value=0.0,
                    )
                    if p1_sort == "Total cells (desc)" and not metrics.empty:
                        order = metrics.sort_values("cells_in_group", ascending=False)["group_label"].tolist()
                    elif p1_sort == "Shannon index (desc)" and not metrics.empty:
                        order = metrics.sort_values("shannon_index", ascending=False)["group_label"].tolist()
                    else:
                        order = sorted(pivot.index.tolist())
                    order = [o for o in order if o in pivot.index][:p1_top_n]
                    pivot = pivot.loc[order]

                    # Build descriptive title/filename
                    _p1_filter_desc = ", ".join(f"{fc}=[{','.join(fv[:3])}]" for fc, fv in _p1_filters if fv and fc in work.columns and len(fv) < len(_safe_unique(work[fc]))) if _p1_filters else ""
                    _p1_title = f"Subclone Proportions — grouped by {', '.join(g_cols)}"
                    if _p1_filter_desc:
                        _p1_title += f"\n({_p1_filter_desc})"
                    _p1_fname = _safe_filename("clonality", "proportions", *g_cols)

                    fig, ax = plt.subplots(figsize=(max(p1_figw, 0.45 * len(order)), 6))
                    pivot.plot.bar(stacked=True, ax=ax, width=0.85, edgecolor="white", linewidth=0.4)
                    ax.set_ylabel("Clone Fraction")
                    ax.set_xlabel("")
                    ax.set_title(_p1_title, fontsize=13)
                    ax.legend(
                        title="Sub-phenotype",
                        bbox_to_anchor=(1.02, 1),
                        loc="upper left",
                        fontsize=8,
                        title_fontsize=9,
                    )
                    plt.xticks(rotation=45, ha="right", fontsize=7)
                    plt.tight_layout()

                    st.session_state[_sk("p1_fig_bytes")] = _fig_to_bytes(fig)
                    st.session_state[_sk("p1_metrics")] = metrics
                    st.session_state[_sk("p1_fname")] = _p1_fname
                    plt.close(fig)

        # ── Display stored results ──
        if _show_fig(_sk("p1_fig_bytes")):
            st.download_button(
                "\u2b07 Download PNG",
                st.session_state[_sk("p1_fig_bytes")],
                file_name=st.session_state.get(_sk("p1_fname"), "clonality_proportions.png"),
                mime="image/png",
                key=_sk("p1_dl"),
            )
            if _sk("p1_metrics") in st.session_state:
                with st.expander("View metrics table"):
                    st.dataframe(st.session_state[_sk("p1_metrics")], width="stretch")

    # =================================================================
    #  Panel 2 \u2014 Diversity Metrics
    # =================================================================
    with st.expander("\U0001f4c8 Panel 2 \u2014 Diversity Metrics (Shannon / Simpson / Evenness)", expanded=False):
        st.info(
            "Compute ecological diversity indices for each group and display their "
            "distribution.  \n"
            "**Stratify** mode: box + strip plots split by column values.  \n"
            "**Compare two subsets** mode: contrast diversity between cells filtered "
            "by two different columns (e.g. tumor_bed vs epithelia_bed), with optional "
            "paired comparison.",
            icon="\u2139\ufe0f",
        )

        # -- Mode selector (outside form \u2014 rerun OK) --
        p2_mode = st.radio(
            "Mode",
            ["Stratify", "Compare two subsets"],
            key=_sk("p2_mode"),
            horizontal=True,
            help="**Stratify**: split by column values.  **Compare**: two column-based cell subsets.",
        )

        # -- Dynamic filter count (outside form \u2014 rerun on add/remove only) --
        _p2_nf_key = _sk("p2_n_filters")
        if _p2_nf_key not in st.session_state:
            st.session_state[_p2_nf_key] = 2

        # \u2500\u2500\u2500 Stratify mode \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if p2_mode == "Stratify":
            with st.form(_sk("p2_form_strat")):
                p2r1c1, p2r1c2 = st.columns(2)
                with p2r1c1:
                    p2_group_cols: list[str] = st.multiselect(
                        "Group-by columns",
                        options=all_cols,
                        default=[batch_col],
                        key=_sk("p2_group_cols"),
                        help="Defines how rows are aggregated into groups before computing diversity.",
                    )
                with p2r1c2:
                    p2_metric = st.selectbox(
                        "Metric",
                        ["shannon_index", "effective_clone_number", "pielou_evenness", "simpson_gini", "clone_richness"],
                        key=_sk("p2_metric"),
                        help=(
                            "**shannon_index**: H = \u2212\u03a3 p\u1d62 ln(p\u1d62)  \n"
                            "**effective_clone_number**: exp(H)  \n"
                            "**pielou_evenness**: H / ln(S)  \n"
                            "**simpson_gini**: 1 \u2212 \u03a3 p\u1d62\u00b2  \n"
                            "**clone_richness**: count of unique subclones"
                        ),
                    )

                p2r2c1, p2r2c2 = st.columns(2)
                with p2r2c1:
                    _strat_options = [c for c in metadata_cols + all_cols if c != subphenotype_col]
                    _strat_options = list(dict.fromkeys(_strat_options))
                    p2_strat_cols: list[str] = st.multiselect(
                        "Stratification columns",
                        options=_strat_options,
                        default=[],
                        key=_sk("p2_strat_cols"),
                        help=(
                            "Split the distribution into sub-groups for comparison. "
                            "Each selected column produces a separate subplot with its own statistics."
                        ),
                    )
                with p2r2c2:
                    p2_stat_method = st.selectbox(
                        "Statistical test",
                        ["None"] + list(_STAT_METHODS.keys()),
                        key=_sk("p2_stat"),
                        help=(
                            "**Welch t-test**: 2-group means (unequal variance).  \n"
                            "**Mann-Whitney U**: 2-group non-parametric.  \n"
                            "**One-way ANOVA**: \u22653-group parametric.  \n"
                            "**Kruskal-Wallis**: \u22653-group non-parametric."
                        ),
                    )

                p2r3c1, p2r3c2, p2r3c3 = st.columns(3)
                with p2r3c1:
                    p2_max_points: int = int(
                        st.slider(
                            "Max strip points per group",
                            50, 2000, 500,
                            key=_sk("p2_max_pts"),
                            help=(
                                "Caps the number of individual data points drawn in the "
                                "strip-plot overlay.  Large datasets can become very slow "
                                "or cluttered without this limit; points are randomly sampled."
                            ),
                        )
                    )
                with p2r3c2:
                    p2_palette_name = st.selectbox(
                        "Color palette",
                        ["Set2", "Set1", "tab10", "Paired", "pastel", "muted", "deep"],
                        key=_sk("p2_palette"),
                        help="Seaborn / matplotlib palette used to colour stratification groups.",
                    )
                with p2r3c3:
                    p2_figw: int = int(
                        st.slider("Figure width", 6, 28, 12, key=_sk("p2_figw"), help="Width in inches.")
                    )

                # \u2500\u2500 Dynamic inline row filters \u2500\u2500
                st.markdown("**Additional row filters** *(applied only to this panel)*")
                _p2_filters: list[tuple[str, list[str]]] = []
                for _fi in range(st.session_state[_p2_nf_key]):
                    _fc_c, _fv_c = st.columns(2)
                    with _fc_c:
                        _fc = st.selectbox(
                            f"Filter column {_fi + 1}",
                            options=["None"] + _filterable_cols,
                            key=_sk(f"p2_fc{_fi}"),
                            help="Pick a column to restrict rows for this panel.",
                        )
                    with _fv_c:
                        if _fc != "None" and _fc in work.columns:
                            _fv_opts = _safe_unique(work[_fc])
                            _fv = st.multiselect(
                                f"Keep values ({_fc})",
                                options=_fv_opts,
                                default=_fv_opts,
                                key=_sk(f"p2_fv{_fi}"),
                            )
                            _p2_filters.append((_fc, _fv))
                        else:
                            st.write("")

                p2_strat_submitted = st.form_submit_button("\U0001f504 Generate Diversity Plot", type="primary")

            # Add / remove filter buttons (outside form)
            _af2, _rf2, _ = st.columns([1, 1, 4])
            with _af2:
                if st.button("\u2795 Add filter", key=_sk("p2_add_f")):
                    st.session_state[_p2_nf_key] = min(st.session_state[_p2_nf_key] + 1, 10)
                    st.rerun()
            with _rf2:
                if st.button("\u2796 Remove filter", key=_sk("p2_rm_f")):
                    st.session_state[_p2_nf_key] = max(st.session_state[_p2_nf_key] - 1, 0)
                    st.rerun()

            # \u2500\u2500 Compute (Stratify) \u2500\u2500
            if p2_strat_submitted:
                _set_pub_style()
                g_cols = list(dict.fromkeys(p2_group_cols)) if p2_group_cols else [batch_col]
                p2_work = work.copy()
                for _fc, _fv in _p2_filters:
                    if _fv:
                        p2_work = p2_work[p2_work[_fc].astype(str).isin([str(v) for v in _fv])]
                if p2_work.empty:
                    st.warning("No rows remain after panel filters.")
                else:
                    extra_meta = list(dict.fromkeys(metadata_cols + p2_strat_cols))
                    metrics, _ = _compute_diversity_table(p2_work, g_cols, subphenotype_col, min_cells, extra_meta)
                    if metrics.empty:
                        st.warning("No groups passed the minimum cell threshold.")
                    else:
                        metric_col = str(p2_metric)
                        # Deduplicate columns (safety against repeated metadata)
                        metrics = metrics.loc[:, ~metrics.columns.duplicated()]
                        if metric_col not in metrics.columns:
                            st.error(f"Metric column '{metric_col}' not found in results.")
                        else:
                            strat_list = [s for s in p2_strat_cols if s in metrics.columns]
                            if not strat_list:
                                strat_list = [None]

                            n_panels = len(strat_list)
                            fig, axes = plt.subplots(
                                1, n_panels,
                                figsize=(max(p2_figw, 6 * n_panels), 7),
                                squeeze=False,
                            )
                            axes = axes.ravel()

                            for ax_idx, strat_col in enumerate(strat_list):
                                ax = axes[ax_idx]
                                plot_df = metrics.copy()

                                if strat_col and strat_col in plot_df.columns:
                                    plot_df[strat_col] = plot_df[strat_col].astype(str)
                                    strat_order = _safe_unique(plot_df[strat_col])
                                    palette = dict(zip(strat_order, sns.color_palette(p2_palette_name, len(strat_order))))
                                else:
                                    strat_col = None
                                    strat_order = None
                                    palette = None

                                if strat_col:
                                    sampled_parts = []
                                    for _, gdf in plot_df.groupby(strat_col, dropna=False):
                                        sampled_parts.append(
                                            gdf if len(gdf) <= p2_max_points
                                            else gdf.sample(p2_max_points, random_state=42)
                                        )
                                    strip_df = pd.concat(sampled_parts, ignore_index=True)
                                else:
                                    strip_df = (
                                        plot_df if len(plot_df) <= p2_max_points
                                        else plot_df.sample(p2_max_points, random_state=42)
                                    )

                                if strat_col:
                                    sns.boxplot(
                                        data=plot_df, x=strat_col, y=metric_col,
                                        order=strat_order, hue=strat_col, palette=palette,
                                        dodge=False, width=0.55, showfliers=False, ax=ax,
                                    )
                                    if ax.legend_:
                                        ax.legend_.remove()
                                    sns.stripplot(
                                        data=strip_df, x=strat_col, y=metric_col,
                                        order=strat_order, color="#333", jitter=0.18,
                                        alpha=0.4, size=4, linewidth=0.25, edgecolor="white", ax=ax,
                                    )
                                    counts = plot_df.groupby(strat_col).size().reindex(strat_order or [], fill_value=0)
                                    ax.set_xticklabels(
                                        [f"{lbl}\nn={int(counts.get(lbl, 0))}" for lbl in (strat_order or [])]
                                    )
                                    ax.set_title(f"by {strat_col}")
                                else:
                                    sns.boxplot(data=plot_df, y=metric_col, width=0.4, showfliers=False, ax=ax)
                                    sns.stripplot(data=strip_df, y=metric_col, color="#333", jitter=0.18, alpha=0.4, size=4, ax=ax)
                                    ax.set_xlabel("")
                                    ax.set_title(metric_col.replace("_", " ").title())

                                ax.set_ylabel(metric_col.replace("_", " ").title())

                                if p2_stat_method != "None" and strat_col and strat_order and len(strat_order) >= 2:
                                    test_fn = _STAT_METHODS[p2_stat_method]
                                    res = test_fn(plot_df, strat_col, metric_col, strat_order)
                                    ann = f"{res['method']}: {_format_p(res['p_value'])}"
                                    ax.text(
                                        0.5, 0.97, ann,
                                        transform=ax.transAxes, ha="center", va="top", fontsize=10,
                                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f7eadb", "edgecolor": "#777", "alpha": 0.95},
                                    )

                            _p2_strat_label = (" \u2014 by " + ", ".join(str(s) for s in strat_list if s)) if any(strat_list) else ""
                            _p2_filter_parts = [f"{fc}" for fc, fv in _p2_filters if fv and fc in work.columns and len(fv) < len(_safe_unique(work[fc]))]
                            _p2_filter_label = (" | filters: " + ", ".join(_p2_filter_parts)) if _p2_filter_parts else ""
                            _p2_title = f"Distribution of {metric_col.replace('_', ' ').title()}{_p2_strat_label}{_p2_filter_label}"
                            _p2_fname = _safe_filename("clonality", "diversity", metric_col, *(str(s) for s in strat_list if s))

                            fig.suptitle(_p2_title, y=1.02, fontsize=14)
                            plt.tight_layout()

                            st.session_state[_sk("p2_fig_bytes")] = _fig_to_bytes(fig)
                            st.session_state[_sk("p2_metrics")] = metrics
                            st.session_state[_sk("p2_fname")] = _p2_fname
                            plt.close(fig)

        # \u2500\u2500\u2500 Compare two subsets mode \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        else:
            with st.form(_sk("p2_form_compare")):
                st.markdown(
                    "Pick **two columns** and the values that define each subset.  \n"
                    "Diversity is computed independently for each subset, then compared."
                )
                p2cc1, p2cc2 = st.columns(2)
                with p2cc1:
                    p2_cmp_group_cols: list[str] = st.multiselect(
                        "Group-by columns",
                        options=all_cols,
                        default=[batch_col],
                        key=_sk("p2_cmp_group"),
                        help="Rows are aggregated by these columns before computing diversity for each subset.",
                    )
                with p2cc2:
                    p2_cmp_metric = st.selectbox(
                        "Metric",
                        ["shannon_index", "effective_clone_number", "pielou_evenness", "simpson_gini", "clone_richness"],
                        key=_sk("p2_cmp_metric"),
                    )

                st.markdown("**Subset A**")
                p2ca1, p2ca2 = st.columns(2)
                with p2ca1:
                    p2_cmp_col_a = st.selectbox(
                        "Column A",
                        options=["None"] + _filterable_cols,
                        key=_sk("p2_cmp_col_a"),
                        help="Column used to select cells for subset A.",
                    )
                with p2ca2:
                    if p2_cmp_col_a != "None" and p2_cmp_col_a in work.columns:
                        _a_opts = _safe_unique(work[p2_cmp_col_a])
                        p2_cmp_vals_a: list[str] = st.multiselect(
                            "Values for A",
                            options=_a_opts,
                            default=[_a_opts[0]] if _a_opts else [],
                            key=_sk("p2_cmp_vals_a"),
                            help="Keep cells where Column A matches any of these values.",
                        )
                    else:
                        p2_cmp_vals_a = []
                        st.caption("Select Column A first.")

                st.markdown("**Subset B**")
                p2cb1, p2cb2 = st.columns(2)
                with p2cb1:
                    p2_cmp_col_b = st.selectbox(
                        "Column B",
                        options=["None"] + _filterable_cols,
                        key=_sk("p2_cmp_col_b"),
                        help="Column used to select cells for subset B.",
                    )
                with p2cb2:
                    if p2_cmp_col_b != "None" and p2_cmp_col_b in work.columns:
                        _b_opts = _safe_unique(work[p2_cmp_col_b])
                        p2_cmp_vals_b: list[str] = st.multiselect(
                            "Values for B",
                            options=_b_opts,
                            default=[_b_opts[0]] if _b_opts else [],
                            key=_sk("p2_cmp_vals_b"),
                            help="Keep cells where Column B matches any of these values.",
                        )
                    else:
                        p2_cmp_vals_b = []
                        st.caption("Select Column B first.")

                st.divider()
                p2cx1, p2cx2, p2cx3 = st.columns(3)
                with p2cx1:
                    p2_cmp_paired = st.checkbox(
                        "Paired comparison",
                        value=False,
                        key=_sk("p2_cmp_paired"),
                        help=(
                            "Match groups from subset A and B by the group-by columns.  "
                            "Only groups present in BOTH subsets are included, and a "
                            "paired statistical test is applied."
                        ),
                    )
                with p2cx2:
                    p2_cmp_test = st.selectbox(
                        "Statistical test",
                        ["Paired t-test", "Welch t-test", "Mann-Whitney U"],
                        key=_sk("p2_cmp_test"),
                    )
                with p2cx3:
                    p2_cmp_palette = st.selectbox(
                        "Color palette",
                        ["Set2", "Set1", "tab10", "Paired", "pastel"],
                        key=_sk("p2_cmp_palette"),
                    )

                # \u2500\u2500 Dynamic filters \u2500\u2500
                st.markdown("**Additional row filters** *(applied to BOTH subsets before slicing)*")
                _p2c_filters: list[tuple[str, list[str]]] = []
                for _fi in range(st.session_state[_p2_nf_key]):
                    _fc_c, _fv_c = st.columns(2)
                    with _fc_c:
                        _fc = st.selectbox(
                            f"Filter column {_fi + 1}",
                            options=["None"] + _filterable_cols,
                            key=_sk(f"p2c_fc{_fi}"),
                            help="Pick a column to restrict rows.",
                        )
                    with _fv_c:
                        if _fc != "None" and _fc in work.columns:
                            _fv_opts = _safe_unique(work[_fc])
                            _fv = st.multiselect(
                                f"Keep values ({_fc})",
                                options=_fv_opts,
                                default=_fv_opts,
                                key=_sk(f"p2c_fv{_fi}"),
                            )
                            _p2c_filters.append((_fc, _fv))
                        else:
                            st.write("")

                p2_cmp_submitted = st.form_submit_button("\U0001f504 Generate Comparison Plot", type="primary")

            # Add / remove filter buttons (outside form)
            _af2c, _rf2c, _ = st.columns([1, 1, 4])
            with _af2c:
                if st.button("\u2795 Add filter", key=_sk("p2c_add_f")):
                    st.session_state[_p2_nf_key] = min(st.session_state[_p2_nf_key] + 1, 10)
                    st.rerun()
            with _rf2c:
                if st.button("\u2796 Remove filter", key=_sk("p2c_rm_f")):
                    st.session_state[_p2_nf_key] = max(st.session_state[_p2_nf_key] - 1, 0)
                    st.rerun()

            # \u2500\u2500 Compute (Compare) \u2500\u2500
            if p2_cmp_submitted:
                _ok = True
                if p2_cmp_col_a == "None" or not p2_cmp_vals_a:
                    st.warning("Select Column A and at least one value.")
                    _ok = False
                if p2_cmp_col_b == "None" or not p2_cmp_vals_b:
                    st.warning("Select Column B and at least one value.")
                    _ok = False
                if _ok:
                    _set_pub_style()
                    g_cols = list(dict.fromkeys(p2_cmp_group_cols)) if p2_cmp_group_cols else [batch_col]
                    base = work.copy()
                    for _fc, _fv in _p2c_filters:
                        if _fv:
                            base = base[base[_fc].astype(str).isin([str(v) for v in _fv])]

                    sub_a = base[base[p2_cmp_col_a].astype(str).isin([str(v) for v in p2_cmp_vals_a])]
                    sub_b = base[base[p2_cmp_col_b].astype(str).isin([str(v) for v in p2_cmp_vals_b])]

                    label_a = f"{p2_cmp_col_a}={','.join(p2_cmp_vals_a[:3])}"
                    label_b = f"{p2_cmp_col_b}={','.join(p2_cmp_vals_b[:3])}"

                    met_a, _ = _compute_diversity_table(sub_a, g_cols, subphenotype_col, min_cells, metadata_cols)
                    met_b, _ = _compute_diversity_table(sub_b, g_cols, subphenotype_col, min_cells, metadata_cols)

                    if met_a.empty or met_b.empty:
                        st.warning("One or both subsets have no groups passing the minimum cell threshold.")
                    else:
                        met_a = met_a.loc[:, ~met_a.columns.duplicated()]
                        met_b = met_b.loc[:, ~met_b.columns.duplicated()]
                        mc = str(p2_cmp_metric)

                        if p2_cmp_paired:
                            merged = met_a[g_cols + [mc]].merge(
                                met_b[g_cols + [mc]],
                                on=g_cols, how="inner",
                                suffixes=("_A", "_B"),
                            )
                            if len(merged) < 2:
                                st.warning("Not enough matched groups for paired comparison.")
                            else:
                                col_a_m = f"{mc}_A"
                                col_b_m = f"{mc}_B"
                                if p2_cmp_test == "Paired t-test":
                                    _r = stats.ttest_rel(merged[col_a_m], merged[col_b_m])
                                elif p2_cmp_test == "Mann-Whitney U":
                                    _r = stats.mannwhitneyu(merged[col_a_m], merged[col_b_m], alternative="two-sided")
                                else:
                                    _r = stats.ttest_ind(merged[col_a_m], merged[col_b_m], equal_var=False)
                                p_val = float(np.asarray(_r[1]).item())

                                pal = sns.color_palette(p2_cmp_palette, max(len(merged), 3))
                                fig, ax = plt.subplots(figsize=(10, 7))
                                plot_m = merged.sort_values(col_b_m).reset_index(drop=True)
                                for idx_r, row in plot_m.iterrows():
                                    c = pal[int(idx_r) % len(pal)]
                                    ax.plot([0, 1], [row[col_a_m], row[col_b_m]], color=c, lw=2, alpha=0.8)
                                    ax.scatter([0, 1], [row[col_a_m], row[col_b_m]], color=c, s=80, alpha=0.85, edgecolors="white", lw=0.5)

                                ma, mb = plot_m[col_a_m].mean(), plot_m[col_b_m].mean()
                                ax.plot([0, 1], [ma, mb], color="#ff2d2d", lw=4, alpha=0.95, zorder=5)
                                ax.scatter([0, 1], [ma, mb], color="#ff2d2d", marker="D", s=160, zorder=6, edgecolors="#d7191c", lw=1, label="Cohort mean")
                                ax.text(0.5, 0.96, f"{p2_cmp_test}: {_format_p(p_val)}", transform=ax.transAxes, ha="center", va="top", fontsize=12,
                                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f7eadb", "edgecolor": "#777", "alpha": 0.95})
                                ax.set_xticks([0, 1])
                                ax.set_xticklabels([f"{label_a}\nn={len(plot_m)}", f"{label_b}\nn={len(plot_m)}"])
                                ax.set_ylabel(mc.replace("_", " ").title())
                                _title = f"Paired {mc.replace('_',' ').title()}: {label_a} vs {label_b}"
                                ax.set_title(_title, fontsize=12)
                                ax.legend(loc="upper left")
                                plt.tight_layout()
                                st.session_state[_sk("p2_fig_bytes")] = _fig_to_bytes(fig)
                                st.session_state[_sk("p2_fname")] = _safe_filename("clonality", "compare_paired", mc, p2_cmp_col_a, p2_cmp_col_b)
                                st.session_state[_sk("p2_metrics")] = merged
                                plt.close(fig)
                        else:
                            # Unpaired: side-by-side box+strip
                            met_a["_subset_"] = label_a
                            met_b["_subset_"] = label_b
                            combined = pd.concat([met_a, met_b], ignore_index=True)
                            order = [label_a, label_b]
                            pal = dict(zip(order, sns.color_palette(p2_cmp_palette, 2)))

                            fig, ax = plt.subplots(figsize=(8, 7))
                            sns.boxplot(data=combined, x="_subset_", y=mc, order=order, hue="_subset_", palette=pal, dodge=False, width=0.5, showfliers=False, ax=ax)
                            if ax.legend_:
                                ax.legend_.remove()
                            sns.stripplot(data=combined, x="_subset_", y=mc, order=order, color="#333", jitter=0.18, alpha=0.45, size=4, ax=ax)
                            counts = combined.groupby("_subset_").size().reindex(order, fill_value=0)
                            ax.set_xticklabels([f"{lbl}\nn={int(counts.get(lbl, 0))}" for lbl in order])

                            test_fn = _STAT_METHODS.get(p2_cmp_test)
                            if test_fn and len(order) == 2:
                                res = test_fn(combined, "_subset_", mc, order)
                                ax.text(0.5, 0.97, f"{res['method']}: {_format_p(res['p_value'])}", transform=ax.transAxes, ha="center", va="top", fontsize=10,
                                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f7eadb", "edgecolor": "#777", "alpha": 0.95})

                            _title = f"{mc.replace('_',' ').title()}: {label_a} vs {label_b}"
                            ax.set_title(_title, fontsize=12)
                            ax.set_ylabel(mc.replace("_", " ").title())
                            ax.set_xlabel("")
                            plt.tight_layout()
                            st.session_state[_sk("p2_fig_bytes")] = _fig_to_bytes(fig)
                            st.session_state[_sk("p2_fname")] = _safe_filename("clonality", "compare", mc, p2_cmp_col_a, p2_cmp_col_b)
                            st.session_state[_sk("p2_metrics")] = combined
                            plt.close(fig)

        # \u2500\u2500 Display stored results (both modes) \u2500\u2500
        if _show_fig(_sk("p2_fig_bytes")):
            st.download_button(
                "\u2b07 Download PNG",
                st.session_state[_sk("p2_fig_bytes")],
                file_name=st.session_state.get(_sk("p2_fname"), "clonality_diversity.png"),
                mime="image/png",
                key=_sk("p2_dl"),
            )
            if _sk("p2_metrics") in st.session_state:
                with st.expander("View full metrics table"):
                    st.dataframe(st.session_state[_sk("p2_metrics")], width="stretch")

    # =================================================================
    #  Panel 3 \u2014 Paired Comparison
    # =================================================================
    with st.expander("\U0001f517 Panel 3 \u2014 Paired Comparison (e.g. Tumor vs Normal)", expanded=False):
        st.info(
            "Compare diversity between two region types within matched batches. "
            "For example, compare tumor vs normal epithelial diversity per sample.",
            icon="\u2139\ufe0f",
        )

        # -- Dynamic filter count (outside form) --
        _p3_nf_key = _sk("p3_n_filters")
        if _p3_nf_key not in st.session_state:
            st.session_state[_p3_nf_key] = 2

        with st.form(_sk("p3_form")):
            p3c1, p3c2 = st.columns(2)
            with p3c1:
                p3_split_col = st.selectbox(
                    "Column that splits the two groups",
                    options=["None"] + all_cols,
                    key=_sk("p3_split_col"),
                    help="Column whose values define the two groups (e.g. 'tumor' / 'normal').",
                )
            with p3c2:
                p3_metric = st.selectbox(
                    "Metric to compare",
                    ["shannon_index", "effective_clone_number", "pielou_evenness", "simpson_gini"],
                    key=_sk("p3_metric"),
                    help="Diversity metric used for the paired comparison.",
                )

            # Dynamic value selectors
            _p3_split_vals: list[str] = []
            if p3_split_col != "None" and p3_split_col in work.columns:
                _p3_split_vals = _safe_unique(work[p3_split_col])

            p3v1, p3v2 = st.columns(2)
            with p3v1:
                p3_val_a = st.selectbox(
                    "Group A value",
                    options=_p3_split_vals if _p3_split_vals else ["(select split column first)"],
                    key=_sk("p3_val_a"),
                    help="First group for the comparison.",
                )
            with p3v2:
                _remaining = [v for v in _p3_split_vals if v != p3_val_a]
                p3_val_b = st.selectbox(
                    "Group B value",
                    options=_remaining if _remaining else _p3_split_vals if _p3_split_vals else ["(select split column first)"],
                    key=_sk("p3_val_b"),
                    help="Second group for the comparison.",
                )

            p3_test = st.selectbox(
                "Statistical test",
                ["Paired t-test", "Welch t-test", "Mann-Whitney U"],
                key=_sk("p3_test"),
                help=(
                    "**Paired t-test**: requires matched batches in both groups.  \n"
                    "**Welch t-test**: unpaired, unequal variance.  \n"
                    "**Mann-Whitney U**: non-parametric."
                ),
            )

            # \u2500\u2500 Dynamic inline row filters \u2500\u2500
            st.markdown("**Additional row filters** *(applied before splitting)*")
            _p3_filters: list[tuple[str, list[str]]] = []
            for _fi in range(st.session_state[_p3_nf_key]):
                _fc_c, _fv_c = st.columns(2)
                with _fc_c:
                    _fc = st.selectbox(
                        f"Filter column {_fi + 1}",
                        options=["None"] + _filterable_cols,
                        key=_sk(f"p3_fc{_fi}"),
                        help="Pick a column to restrict rows for this panel.",
                    )
                with _fv_c:
                    if _fc != "None" and _fc in work.columns:
                        _fv_opts = _safe_unique(work[_fc])
                        _fv = st.multiselect(
                            f"Keep values ({_fc})",
                            options=_fv_opts,
                            default=_fv_opts,
                            key=_sk(f"p3_fv{_fi}"),
                        )
                        _p3_filters.append((_fc, _fv))
                    else:
                        st.write("")

            p3_submitted = st.form_submit_button("\U0001f504 Generate Paired Comparison", type="primary")

        # Add / remove filter buttons (outside form)
        _af3, _rf3, _ = st.columns([1, 1, 4])
        with _af3:
            if st.button("\u2795 Add filter", key=_sk("p3_add_f")):
                st.session_state[_p3_nf_key] = min(st.session_state[_p3_nf_key] + 1, 10)
                st.rerun()
        with _rf3:
            if st.button("\u2796 Remove filter", key=_sk("p3_rm_f")):
                st.session_state[_p3_nf_key] = max(st.session_state[_p3_nf_key] - 1, 0)
                st.rerun()

        if p3_submitted:
            if p3_split_col == "None" or not _p3_split_vals:
                st.warning("Select a valid split column first.")
            else:
                _set_pub_style()
                p3_work = work.copy()
                for _fc, _fv in _p3_filters:
                    if _fv:
                        p3_work = p3_work[p3_work[_fc].astype(str).isin([str(v) for v in _fv])]
                if p3_work.empty:
                    st.warning("No rows remain after panel filters.")
                else:
                    work_a = p3_work[p3_work[p3_split_col].astype(str) == str(p3_val_a)]
                    work_b = p3_work[p3_work[p3_split_col].astype(str) == str(p3_val_b)]

                    met_a, _ = _compute_diversity_table(work_a, [batch_col], subphenotype_col, min_cells, metadata_cols)
                    met_b, _ = _compute_diversity_table(work_b, [batch_col], subphenotype_col, min_cells, metadata_cols)
                    if met_a.empty or met_b.empty:
                        st.warning("One or both groups have no data passing thresholds.")
                    else:
                        merged = met_a[[batch_col, p3_metric]].merge(
                            met_b[[batch_col, p3_metric]],
                            on=batch_col,
                            how="inner",
                            suffixes=(f"_{p3_val_a}", f"_{p3_val_b}"),
                        )
                        col_a = f"{p3_metric}_{p3_val_a}"
                        col_b = f"{p3_metric}_{p3_val_b}"
                        if merged.empty or len(merged) < 2:
                            st.warning("Not enough matched batches for a paired comparison.")
                        else:
                            if p3_test == "Paired t-test":
                                _r = stats.ttest_rel(merged[col_a], merged[col_b])
                                res = {"method": "Paired t-test", "p_value": float(np.asarray(_r[1]).item())}
                            elif p3_test == "Mann-Whitney U":
                                _r = stats.mannwhitneyu(merged[col_a], merged[col_b], alternative="two-sided")
                                res = {"method": "Mann-Whitney U", "p_value": float(np.asarray(_r[1]).item())}
                            else:
                                _r = stats.ttest_ind(merged[col_a], merged[col_b], equal_var=False)
                                res = {"method": "Welch t-test", "p_value": float(np.asarray(_r[1]).item())}

                            _p3_filter_parts = [f"{fc}" for fc, fv in _p3_filters if fv and fc in work.columns and len(fv) < len(_safe_unique(work[fc]))]
                            _p3_filter_label = ("\nFilters: " + ", ".join(_p3_filter_parts)) if _p3_filter_parts else ""

                            plot_merged = merged.sort_values(col_b).reset_index(drop=True)
                            line_pal = sns.color_palette("Set2", max(len(plot_merged), 3))

                            fig, ax = plt.subplots(figsize=(10, 7))
                            for idx, row in plot_merged.iterrows():
                                c = line_pal[int(idx) % len(line_pal)]
                                ax.plot([0, 1], [row[col_b], row[col_a]], color=c, lw=2, alpha=0.8)
                                ax.scatter(
                                    [0, 1], [row[col_b], row[col_a]],
                                    color=c, s=80, alpha=0.85, edgecolors="white", linewidth=0.5,
                                )

                            mean_b, mean_a = plot_merged[col_b].mean(), plot_merged[col_a].mean()
                            ax.plot([0, 1], [mean_b, mean_a], color="#ff2d2d", lw=4, alpha=0.95, zorder=5)
                            ax.scatter(
                                [0, 1], [mean_b, mean_a],
                                color="#ff2d2d", marker="D", s=160, zorder=6,
                                edgecolors="#d7191c", lw=1, label="Cohort mean",
                            )

                            ann = f"{res['method']}: {_format_p(res['p_value'])}"
                            ax.text(
                                0.5, 0.96, ann, transform=ax.transAxes, ha="center", va="top",
                                fontsize=12,
                                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f7eadb", "edgecolor": "#777", "alpha": 0.95},
                            )
                            ax.set_xticks([0, 1])
                            ax.set_xticklabels([
                                f"{p3_val_b}\nn={len(plot_merged)}",
                                f"{p3_val_a}\nn={len(plot_merged)}",
                            ])
                            ax.set_ylabel(p3_metric.replace("_", " ").title())
                            _p3_title = f"Paired {p3_metric.replace('_', ' ').title()}: {p3_val_a} vs {p3_val_b} \u2014 matched by {batch_col}{_p3_filter_label}"
                            ax.set_title(_p3_title, fontsize=12)
                            _p3_fname = _safe_filename("clonality", "paired", p3_metric, str(p3_val_a), "vs", str(p3_val_b))
                            ax.legend(loc="upper left")
                            plt.tight_layout()

                            st.session_state[_sk("p3_fig_bytes")] = _fig_to_bytes(fig)
                            st.session_state[_sk("p3_merged")] = merged
                            st.session_state[_sk("p3_fname")] = _p3_fname
                            plt.close(fig)

        if _show_fig(_sk("p3_fig_bytes")):
            st.download_button(
                "\u2b07 Download PNG",
                st.session_state[_sk("p3_fig_bytes")],
                file_name=st.session_state.get(_sk("p3_fname"), "clonality_paired.png"),
                mime="image/png",
                key=_sk("p3_dl"),
            )
            if _sk("p3_merged") in st.session_state:
                with st.expander("View paired data"):
                    st.dataframe(st.session_state[_sk("p3_merged")], width="stretch")
    # =================================================================
    #  Panel 4 \u2014 Stratified Bar Charts (composite groups)
    # =================================================================
    with st.expander("\U0001f4ca Panel 4 \u2014 Stratified Mean Diversity Bar Chart", expanded=False):
        st.info(
            "Show mean diversity (\u00b1 SEM) for **composite groups** built from "
            "one or more stratification columns.\n\n"
            "For example, choosing *arm*, *response*, *timepoint* produces bars "
            "like `arm=1 | response=CR | timepoint=1`.  "
            "Statistics are computed across these composite groups.",
            icon="\u2139\ufe0f",
        )

        # -- dynamic filter count (outside form) --
        _p4_nf_key = _sk("p4_n_filters")
        if _p4_nf_key not in st.session_state:
            st.session_state[_p4_nf_key] = 0

        with st.form(_sk("p4_form")):
            p4c1, p4c2 = st.columns(2)
            with p4c1:
                p4_strat_cols: list[str] = st.multiselect(
                    "Grouping columns (combined into composite groups)",
                    options=[c for c in all_cols if c != subphenotype_col],
                    default=[c for c in ["arm", "response", "timepoint"] if c in all_cols][:2],
                    key=_sk("p4_strats"),
                    help=(
                        "Values from the selected columns are concatenated to form "
                        "composite group labels (e.g. arm=1 | response=CR).  "
                        "Each unique combination becomes one bar."
                    ),
                )
            with p4c2:
                p4_metric = st.selectbox(
                    "Metric",
                    ["shannon_index", "effective_clone_number", "pielou_evenness", "simpson_gini"],
                    key=_sk("p4_metric"),
                    help="The diversity metric to display on the y-axis.",
                )

            p4s1, p4s2 = st.columns(2)
            with p4s1:
                p4_stat = st.selectbox(
                    "Statistical test",
                    ["None"] + list(_STAT_METHODS.keys()),
                    key=_sk("p4_stat"),
                    help=(
                        "Test applied across the composite groups.\n\n"
                        "- **Welch t-test** / **Mann-Whitney U**: only when exactly 2 groups.\n"
                        "- **ANOVA** / **Kruskal-Wallis**: 2+ groups."
                    ),
                )
            with p4s2:
                p4_palette = st.selectbox(
                    "Color palette",
                    ["Set2", "Set1", "tab10", "Paired", "pastel"],
                    key=_sk("p4_palette"),
                    help="Seaborn / matplotlib palette for bars.",
                )

            p4_orientation = st.radio(
                "Bar orientation",
                ["Vertical", "Horizontal"],
                horizontal=True,
                key=_sk("p4_orient"),
                help="When many composite groups exist, horizontal bars may be easier to read.",
            )

            # -- inline row filters --
            st.markdown("**Additional row filters** *(applied before computing diversity)*")
            _p4_filters: list[tuple[str, list[str]]] = []
            for _fi in range(st.session_state[_p4_nf_key]):
                _fc_c, _fv_c = st.columns(2)
                with _fc_c:
                    _fc = st.selectbox(
                        f"Filter column {_fi + 1}",
                        options=["None"] + _filterable_cols,
                        key=_sk(f"p4_fc{_fi}"),
                        help="Pick a column to restrict rows.",
                    )
                with _fv_c:
                    if _fc != "None" and _fc in work.columns:
                        _fv_opts = _safe_unique(work[_fc])
                        _fv = st.multiselect(
                            f"Keep values ({_fc})",
                            options=_fv_opts,
                            default=_fv_opts,
                            key=_sk(f"p4_fv{_fi}"),
                        )
                        _p4_filters.append((_fc, _fv))
                    else:
                        st.write("")

            p4_submitted = st.form_submit_button("\U0001f504 Generate Stratified Bars", type="primary")

        # Add / remove filter buttons (outside form)
        _af4, _rf4, _ = st.columns([1, 1, 4])
        with _af4:
            if st.button("\u2795 Add filter", key=_sk("p4_add_f")):
                st.session_state[_p4_nf_key] = min(st.session_state[_p4_nf_key] + 1, 10)
                st.rerun()
        with _rf4:
            if st.button("\u2796 Remove filter", key=_sk("p4_rm_f")):
                st.session_state[_p4_nf_key] = max(st.session_state[_p4_nf_key] - 1, 0)
                st.rerun()

        if p4_submitted:
            if not p4_strat_cols:
                st.warning("Select at least one grouping column.")
            else:
                _set_pub_style()

                # Apply inline filters
                p4_work = work.copy()
                for _fc, _fv in _p4_filters:
                    if _fv:
                        p4_work = p4_work[p4_work[_fc].astype(str).isin([str(v) for v in _fv])]
                if p4_work.empty:
                    st.warning("No rows remain after panel filters.")
                else:
                    # Build composite group column
                    _composite_col = "__p4_group__"
                    p4_work[_composite_col] = p4_work[p4_strat_cols[0]].astype(str)
                    for _sc in p4_strat_cols[1:]:
                        p4_work[_composite_col] = (
                            p4_work[_composite_col] + " | " + _sc + "=" + p4_work[_sc].astype(str)
                        )
                    # Prepend first column name for first entry
                    p4_work[_composite_col] = (
                        p4_strat_cols[0] + "=" + p4_work[p4_strat_cols[0]].astype(str)
                        + p4_work[_composite_col].str.replace(p4_work[p4_strat_cols[0]].astype(str), "", regex=False).values
                    )
                    # Simpler: just rebuild cleanly
                    p4_work[_composite_col] = p4_work[p4_strat_cols].apply(
                        lambda r: " | ".join(f"{c}={r[c]}" for c in p4_strat_cols), axis=1
                    )

                    g_cols = [batch_col]
                    all_extra = list(dict.fromkeys(metadata_cols + p4_strat_cols + [_composite_col]))
                    metrics, _ = _compute_diversity_table(p4_work, g_cols, subphenotype_col, min_cells, all_extra)

                    if metrics.empty:
                        st.warning("No groups passed the minimum cell threshold.")
                    elif _composite_col not in metrics.columns:
                        st.warning(f"Composite grouping column was lost during aggregation. Check that all grouping columns have data.")
                    else:
                        sdf = metrics.dropna(subset=[_composite_col, p4_metric]).copy()
                        sdf = sdf.loc[:, ~sdf.columns.duplicated()]
                        sdf[_composite_col] = sdf[_composite_col].astype(str)
                        levels = sorted(sdf[_composite_col].unique().tolist())

                        if not levels:
                            st.warning("No valid composite groups found.")
                        else:
                            palette = dict(zip(levels, sns.color_palette(p4_palette, max(len(levels), 1))))
                            summary = (
                                sdf.groupby(_composite_col)[p4_metric]
                                .agg(["mean", "count", _standard_error])
                                .reset_index()
                                .rename(columns={"count": "n", "_standard_error": "sem"})
                            )
                            summary[_composite_col] = pd.Categorical(
                                summary[_composite_col], categories=levels, ordered=True
                            )
                            summary = summary.sort_values(_composite_col).reset_index(drop=True)

                            horizontal = p4_orientation == "Horizontal"
                            n_bars = len(summary)
                            if horizontal:
                                fig_w = max(8, 0.15 * max(len(l) for l in levels) + 4)
                                fig_h = max(6, 0.4 * n_bars + 2)
                            else:
                                fig_w = max(8, 1.2 * n_bars + 2)
                                fig_h = 7

                            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                            pos = np.arange(n_bars)
                            bar_colors = [palette.get(str(l), "#999") for l in summary[_composite_col]]

                            if horizontal:
                                ax.barh(pos, summary["mean"], xerr=summary["sem"],
                                        color=bar_colors, edgecolor="#333", lw=1.1, capsize=4)
                                for yp, (_, r) in zip(pos, summary.iterrows()):
                                    ax.text(r["mean"] + r["sem"] + 0.01, yp,
                                            f" {r['mean']:.3f}  n={int(r['n'])}",
                                            ha="left", va="center", fontsize=9, fontweight="bold")
                                ax.set_yticks(pos)
                                ax.set_yticklabels([str(v) for v in summary[_composite_col]], fontsize=9)
                                ax.set_xlabel(p4_metric.replace("_", " ").title())
                                ax.invert_yaxis()
                            else:
                                ax.bar(pos, summary["mean"], yerr=summary["sem"],
                                       color=bar_colors, edgecolor="#333", lw=1.1, capsize=5)
                                for xp, (_, r) in zip(pos, summary.iterrows()):
                                    ax.text(xp, r["mean"] + r["sem"] + 0.01,
                                            f"{r['mean']:.3f}\nn={int(r['n'])}",
                                            ha="center", va="bottom", fontsize=9, fontweight="bold")
                                ax.set_xticks(pos)
                                ax.set_xticklabels([str(v) for v in summary[_composite_col]],
                                                   rotation=45, ha="right", fontsize=9)
                                ax.set_ylabel(p4_metric.replace("_", " ").title())

                            # Statistical annotation
                            if p4_stat != "None" and len(levels) >= 2:
                                fn = _STAT_METHODS[p4_stat]
                                res = fn(sdf, _composite_col, p4_metric, levels)
                                ax.text(
                                    0.5, 0.97,
                                    f"{res['method']}: {_format_p(res['p_value'])}",
                                    transform=ax.transAxes, ha="center", va="top", fontsize=10,
                                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#ccc"},
                                )

                            # Filter description in title
                            _p4_filter_parts = [
                                f"{fc}" for fc, fv in _p4_filters
                                if fv and fc in work.columns and len(fv) < len(_safe_unique(work[fc]))
                            ]
                            _p4_filter_label = (" | filters: " + ", ".join(_p4_filter_parts)) if _p4_filter_parts else ""
                            _p4_strat_label = " \u00d7 ".join(p4_strat_cols)
                            _p4_title = (
                                f"Mean {p4_metric.replace('_', ' ').title()} \u2014 "
                                f"Grouped by {_p4_strat_label}{_p4_filter_label}"
                            )
                            _p4_fname = _safe_filename("clonality", "stratified", p4_metric, *p4_strat_cols)
                            fig.suptitle(_p4_title, y=1.02, fontsize=14)
                            plt.tight_layout()
                            st.session_state[_sk("p4_fig_bytes")] = _fig_to_bytes(fig)
                            st.session_state[_sk("p4_fname")] = _p4_fname
                            st.session_state[_sk("p4_summary")] = summary.rename(
                                columns={_composite_col: "group"}
                            )
                            plt.close(fig)

        if _show_fig(_sk("p4_fig_bytes")):
            st.download_button(
                "\u2b07 Download PNG",
                st.session_state[_sk("p4_fig_bytes")],
                file_name=st.session_state.get(_sk("p4_fname"), "clonality_stratified_bars.png"),
                mime="image/png",
                key=_sk("p4_dl"),
            )
            if _sk("p4_summary") in st.session_state:
                with st.expander("View summary table"):
                    st.dataframe(st.session_state[_sk("p4_summary")], width="stretch")

    # =================================================================
    #  Panel 5 \u2014 Patient Heatmap
    # =================================================================
    with st.expander("\U0001f5fa\ufe0f Panel 5 \u2014 Patient \u00d7 Region-Type Heatmap", expanded=False):
        st.info(
            "Build a heatmap summarising mean Shannon index for each patient "
            "across analysis levels (aggregate batch vs individual ROIs).",
            icon="\u2139\ufe0f",
        )

        with st.form(_sk("p5_form")):
            p5c1, p5c2 = st.columns(2)
            with p5c1:
                p5_patient_col = st.selectbox(
                    "Patient / subject column",
                    options=["None"] + all_cols,
                    index=_default_index(["None"] + all_cols, "patient", 0),
                    key=_sk("p5_patient_col"),
                    help="Column identifying biological patients. Falls back to batch column if None.",
                )
            with p5c2:
                p5_cmap = st.selectbox(
                    "Heatmap colormap",
                    ["YlOrRd", "viridis", "RdBu_r", "coolwarm", "YlGnBu"],
                    key=_sk("p5_cmap"),
                    help="Matplotlib colormap for the heatmap cells.",
                )

            p5_submitted = st.form_submit_button("\U0001f504 Generate Heatmap", type="primary")

        if p5_submitted:
            _set_pub_style()
            _p5_pat = str(p5_patient_col)
            patient_col = batch_col if _p5_pat == "None" or _p5_pat not in work.columns else _p5_pat

            agg_met, _ = _compute_diversity_table(work, [batch_col], subphenotype_col, min_cells, metadata_cols)
            if agg_met.empty:
                st.warning("Not enough data for heatmap.")
            else:
                if patient_col in agg_met.columns:
                    agg_col = agg_met.groupby(patient_col, dropna=False)["shannon_index"].mean().rename("aggregate")
                else:
                    agg_col = agg_met.set_index(batch_col)["shannon_index"].rename("aggregate")

                if region_col != "None":
                    reg_met, _ = _compute_diversity_table(work, [batch_col, region_col], subphenotype_col, min_cells, metadata_cols)
                    if not reg_met.empty and patient_col in reg_met.columns:
                        reg_col_data = reg_met.groupby(patient_col, dropna=False)["shannon_index"].mean().rename("mean_region")
                    elif not reg_met.empty:
                        reg_col_data = reg_met.groupby(batch_col, dropna=False)["shannon_index"].mean().rename("mean_region")
                    else:
                        reg_col_data = pd.Series(dtype=float, name="mean_region")
                else:
                    reg_col_data = pd.Series(dtype=float, name="mean_region")

                heatmap_df = pd.concat([agg_col, reg_col_data], axis=1).reset_index()
                heatmap_df.columns = [patient_col] + list(heatmap_df.columns[1:])
                heatmap_df = heatmap_df.sort_values("aggregate", ascending=False)

                mat = heatmap_df.set_index(patient_col).select_dtypes(include="number")
                if mat.empty:
                    st.warning("Heatmap matrix is empty.")
                else:
                    fig_h = max(6, 0.35 * len(mat))
                    fig, ax = plt.subplots(figsize=(10, fig_h))
                    sns.heatmap(
                        mat, cmap=p5_cmap, annot=True, fmt=".3f",
                        linewidths=1, linecolor="white",
                        cbar_kws={"label": "Shannon Index"}, ax=ax,
                    )
                    _p5_title = f"Shannon Diversity Heatmap — per {patient_col}"
                    _p5_fname = _safe_filename("clonality", "heatmap", patient_col)
                    ax.set_title(_p5_title, pad=14)
                    ax.set_ylabel(patient_col)
                    plt.tight_layout()
                    st.session_state[_sk("p5_fig_bytes")] = _fig_to_bytes(fig)
                    st.session_state[_sk("p5_fname")] = _p5_fname
                    plt.close(fig)

        if _show_fig(_sk("p5_fig_bytes")):
            st.download_button(
                "\u2b07 Download PNG",
                st.session_state[_sk("p5_fig_bytes")],
                file_name=st.session_state.get(_sk("p5_fname"), "clonality_heatmap.png"),
                mime="image/png",
                key=_sk("p5_dl"),
            )

    # =================================================================
    #  Panel 6 \u2014 Clone Clustering (K-Means)
    # =================================================================
    with st.expander("\U0001f9ec Panel 6 \u2014 Region Clustering by Clone Proportions", expanded=False):
        st.info(
            "Cluster regions (batch \u00d7 ROI) by their subclone proportion vectors "
            "using K-Means. Generates a PCA scatter and centroid-composition heatmap. "
            "You can rename clusters afterward and export them in Panel 7.",
            icon="\u2139\ufe0f",
        )

        if region_col == "None":
            st.warning("Region clustering requires a Region / ROI column. Set one in Configuration.")
        else:
            with st.form(_sk("p6_form")):
                p6c1, p6c2, p6c3 = st.columns(3)
                with p6c1:
                    p6_auto_k: bool = st.checkbox(
                        "Auto-select k (silhouette)",
                        value=True,
                        key=_sk("p6_auto_k"),
                        help="Automatically pick k using silhouette score across candidates.",
                    )
                with p6c2:
                    p6_fixed_k: int = int(
                        st.number_input("Fixed k", 2, 15, 4, key=_sk("p6_fixed_k"), help="Used when auto-select is off.")
                    )
                with p6c3:
                    p6_max_k: int = int(
                        st.number_input("Max k candidates", 3, 15, 7, key=_sk("p6_max_k"), help="Upper bound for auto k search.")
                    )
                p6s1, p6s2 = st.columns(2)
                with p6s1:
                    p6_seed: int = int(st.number_input("Random seed", 0, 99999, 42, key=_sk("p6_seed"), help="Seed for reproducibility."))
                with p6s2:
                    p6_min_rois: int = int(st.number_input("Min ROIs for clustering", 5, 500, 25, key=_sk("p6_min_rois"), help="Clustering is skipped if fewer ROIs are available."))

                p6_submitted = st.form_submit_button("\U0001f504 Run Clustering", type="primary")

            if p6_submitted:
                _set_pub_style()
                g_cols = [batch_col, region_col]
                metrics, comp = _compute_diversity_table(work, g_cols, subphenotype_col, min_cells, metadata_cols)
                if metrics.empty or comp.empty:
                    st.warning("Not enough data after filtering.")
                else:
                    pivot = comp.pivot_table(
                        index="group_label",
                        columns=subphenotype_col,
                        values="clone_fraction",
                        aggfunc="sum",
                        fill_value=0.0,
                    )
                    pivot = pivot.reindex(metrics["group_label"]).fillna(0.0)
                    feature_cols = pivot.columns.tolist()

                    if len(pivot) < p6_min_rois:
                        st.warning(f"Only {len(pivot)} ROIs available, need at least {p6_min_rois}.")
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fixed = None if p6_auto_k else p6_fixed_k
                            chosen_k, sil = _choose_k(pivot, fixed, range(2, p6_max_k + 1), p6_seed)

                        if chosen_k is None:
                            st.error("Could not determine a valid k. Try adjusting parameters.")
                        else:
                            from sklearn.cluster import KMeans
                            from sklearn.decomposition import PCA

                            model = KMeans(n_clusters=chosen_k, random_state=p6_seed, n_init=50)
                            raw_labels = model.fit_predict(pivot) + 1

                            lbl_df = pd.DataFrame({"group_label": pivot.index, "cluster_id": raw_labels})
                            lbl_df = lbl_df.merge(metrics[["group_label", "shannon_index"]], on="group_label", how="left")
                            order_map_src = lbl_df.groupby("cluster_id")["shannon_index"].mean().sort_values(ascending=False)
                            id_map = {old: new for new, old in enumerate(order_map_src.index, 1)}
                            lbl_df["cluster_id"] = lbl_df["cluster_id"].map(id_map).astype(int)
                            lbl_df["cluster_name"] = lbl_df["cluster_id"].map(lambda v: f"Cluster {v}")

                            metrics = metrics.merge(lbl_df[["group_label", "cluster_id", "cluster_name"]], on="group_label", how="left")
                            assign_df = metrics.merge(pivot.reset_index(), on="group_label", how="left")
                            centroids = assign_df.groupby(["cluster_id", "cluster_name"], dropna=False)[feature_cols].mean().reset_index().sort_values("cluster_id")

                            cluster_names = sorted(metrics["cluster_name"].dropna().unique().tolist())
                            pal = dict(zip(cluster_names, sns.color_palette("Set2", len(cluster_names))))

                            pca = PCA(n_components=2, random_state=p6_seed)
                            coords = pca.fit_transform(pivot)
                            scatter_df = lbl_df.copy()
                            scatter_df["PC1"] = coords[:, 0]
                            scatter_df["PC2"] = coords[:, 1]

                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.15, 1]})

                            sns.scatterplot(data=scatter_df, x="PC1", y="PC2", hue="cluster_name", palette=pal, s=60, alpha=0.8, linewidth=0.3, edgecolor="white", ax=ax1)
                            for cn, cdf in scatter_df.groupby("cluster_name", dropna=False):
                                ax1.scatter(cdf["PC1"].mean(), cdf["PC2"].mean(), s=220, marker="X", color=pal.get(cn, "#999"), edgecolor="#202020", lw=1, zorder=6)

                            title1 = f"Region Clusters (k={chosen_k}"
                            if pd.notna(sil):
                                title1 += f", silhouette={sil:.3f}"
                            title1 += f") — {subphenotype_col}"
                            _p6_fname = _safe_filename("clonality", "clusters", f"k{chosen_k}", subphenotype_col)
                            ax1.set_title(title1, pad=14)
                            ax1.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

                            sizes = metrics.groupby("cluster_name").size().to_dict()
                            cmat = centroids.set_index("cluster_name")[feature_cols]
                            cmat.index = [f"{n} (n={sizes.get(n, '?')})" for n in cmat.index]
                            sns.heatmap(cmat, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.8, linecolor="white", cbar_kws={"label": "Mean clone fraction"}, ax=ax2)
                            ax2.set_title("Cluster centroid composition", pad=14)
                            ax2.set_xlabel("Sub-phenotype")
                            ax2.set_ylabel("Cluster")

                            plt.tight_layout()
                            st.session_state[_sk("p6_fig_bytes")] = _fig_to_bytes(fig)
                            st.session_state[_sk("p6_fname")] = _p6_fname
                            plt.close(fig)

                            st.session_state[_sk("p6_metrics")] = metrics
                            st.session_state[_sk("p6_assign")] = assign_df
                            st.session_state[_sk("p6_centroids")] = centroids
                            st.session_state[_sk("p6_feature_cols")] = feature_cols
                            st.session_state[_sk("p6_cluster_pal")] = pal
                            st.session_state[_sk("p6_cluster_names")] = cluster_names

            # ── Display stored results ──
            if _show_fig(_sk("p6_fig_bytes")):
                st.download_button(
                    "\u2b07 Download PNG",
                    st.session_state[_sk("p6_fig_bytes")],
                    file_name=st.session_state.get(_sk("p6_fname"), "clonality_clusters.png"),
                    mime="image/png",
                    key=_sk("p6_dl"),
                )

                # ── Cluster naming editor (outside form) ──
                if _sk("p6_cluster_names") in st.session_state:
                    st.markdown("#### Rename Clusters")
                    st.caption("Assign custom names. These will be used in Panel 7 export.")
                    cnames = st.session_state[_sk("p6_cluster_names")]
                    editor_rows = [{"cluster_name": cn, "custom_label": ""} for cn in cnames]
                    editor_df = pd.DataFrame(editor_rows)
                    edited = st.data_editor(editor_df, num_rows="fixed", width="stretch", key=_sk("p6_editor"))
                    st.session_state[_sk("p6_editor_result")] = edited

                if _sk("p6_metrics") in st.session_state:
                    with st.expander("View cluster assignments table"):
                        _m = st.session_state[_sk("p6_metrics")]
                        display_cols = [
                            c for c in ["group_label", batch_col, region_col, "cluster_id", "cluster_name",
                                        "shannon_index", "cells_in_group", "clone_richness", "dominant_subclone"]
                            if c in _m.columns
                        ]
                        st.dataframe(_m[display_cols], width="stretch")

    # =================================================================
    #  Panel 7 \u2014 Cluster Tracking & Export
    # =================================================================
    with st.expander("\U0001f4e4 Panel 7 \u2014 Cluster Tracking & Export to CSV", expanded=False):
        st.info(
            "Track how clusters distribute across metadata groups (stacked bars) "
            "and export cluster assignments back into the unified CSV.",
            icon="\u2139\ufe0f",
        )

        has_clustering = _sk("p6_metrics") in st.session_state
        if not has_clustering:
            st.caption("Run **Panel 6** first to generate cluster assignments.")
        else:
            c_metrics: pd.DataFrame = st.session_state[_sk("p6_metrics")]
            c_pal: dict = st.session_state.get(_sk("p6_cluster_pal"), {})
            c_editor: pd.DataFrame = st.session_state.get(_sk("p6_editor_result"), pd.DataFrame())

            name_map: dict[str, str] = {}
            if not c_editor.empty:
                for _, erow in c_editor.iterrows():
                    cn = str(erow.get("cluster_name", ""))
                    cl = str(erow.get("custom_label", "")).strip()
                    if cl:
                        name_map[cn] = cl

            # ── Tracking form ──
            with st.form(_sk("p7_form")):
                p7_track_cols: list[str] = st.multiselect(
                    "Metadata columns for tracking",
                    options=[
                        c for c in all_cols
                        if c in c_metrics.columns and c not in {"cluster_name", "cluster_id"}
                    ],
                    default=[c for c in ["timepoint", "arm", "response"] if c in c_metrics.columns],
                    key=_sk("p7_track_cols"),
                    help="Each selected column gets a stacked bar subplot showing cluster fractions.",
                )
                p7_submitted = st.form_submit_button("\U0001f504 Generate Tracking Charts", type="primary")

            if p7_submitted:
                _set_pub_style()
                valid_tracks = [c for c in p7_track_cols if c in c_metrics.columns]
                if not valid_tracks:
                    st.warning("No valid tracking columns selected.")
                else:
                    cluster_order = sorted(c_metrics["cluster_name"].dropna().unique().tolist())
                    display_names = {cn: name_map.get(cn, cn) for cn in cluster_order}

                    fig, axes = plt.subplots(1, len(valid_tracks), figsize=(6 * len(valid_tracks), 5.5), squeeze=False)
                    axes = axes.ravel()

                    for ax, tcol in zip(axes, valid_tracks):
                        tracking = (
                            c_metrics.groupby([tcol, "cluster_name"], dropna=False)
                            .size()
                            .reset_index(name="n")
                        )
                        totals = tracking.groupby(tcol)["n"].sum().reset_index(name="total")
                        tracking = tracking.merge(totals, on=tcol)
                        tracking["frac"] = tracking["n"] / tracking["total"]
                        pvt = tracking.pivot(index=tcol, columns="cluster_name", values="frac").fillna(0).reindex(columns=cluster_order, fill_value=0)

                        x = np.arange(len(pvt))
                        cum = np.zeros(len(pvt))
                        for cn in cluster_order:
                            vals = pvt[cn].to_numpy(dtype=float)
                            ax.bar(x, vals, bottom=cum, color=c_pal.get(cn, "#999"), edgecolor="white", lw=0.7, width=0.7, label=display_names[cn])
                            cum += vals

                        ax.set_xticks(x)
                        ax.set_xticklabels([str(v) for v in pvt.index], rotation=30, ha="right")
                        ax.set_ylim(0, 1.02)
                        ax.set_xlabel(tcol)
                        ax.set_ylabel("Fraction of regions")
                        ax.set_title(f"Cluster distribution by {tcol}")

                    handles = [Patch(facecolor=c_pal.get(cn, "#999"), edgecolor="white", label=display_names[cn]) for cn in cluster_order]
                    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(cluster_order)), frameon=False, bbox_to_anchor=(0.5, 1.05))
                    _p7_title = f"Cluster Tracking — by {', '.join(valid_tracks)}"
                    _p7_fname = _safe_filename("clonality", "tracking", *valid_tracks)
                    fig.suptitle(_p7_title, y=1.12, fontsize=15)
                    plt.tight_layout()

                    st.session_state[_sk("p7_fig_bytes")] = _fig_to_bytes(fig)
                    st.session_state[_sk("p7_fname")] = _p7_fname
                    plt.close(fig)

            if _show_fig(_sk("p7_fig_bytes")):
                st.download_button(
                    "\u2b07 Download PNG",
                    st.session_state[_sk("p7_fig_bytes")],
                    file_name=st.session_state.get(_sk("p7_fname"), "clonality_tracking.png"),
                    mime="image/png",
                    key=_sk("p7_dl"),
                )

            # ── Export to CSV ──
            st.divider()
            st.markdown("#### Export Cluster Assignments to CSV")
            st.caption(
                "Merge cluster assignments back into the original unified CSV. "
                "New columns: `clone_cluster_id`, `clone_cluster_name`, `clone_cluster_label`."
            )
            with st.form(_sk("p7_export_form")):
                p7_out_suffix = st.text_input(
                    "Output file suffix",
                    value="_clonality",
                    key=_sk("p7_suffix"),
                    help="Appended to the original CSV filename.",
                )
                p7_export_submitted = st.form_submit_button("\U0001f504 Export Augmented CSV", type="primary")

            if p7_export_submitted:
                if c_metrics.empty or "cluster_id" not in c_metrics.columns:
                    st.warning("No cluster data available. Run Panel 6 first.")
                else:
                    # Build merge key columns dynamically — only use columns that exist
                    _merge_keys = [c for c in [batch_col, region_col] if c != "None" and c in c_metrics.columns]
                    if not _merge_keys:
                        st.warning("No valid merge-key columns (batch / region) found in cluster results.")
                    else:
                        lookup = c_metrics[_merge_keys + ["cluster_id", "cluster_name"]].copy()
                        lookup = lookup.rename(columns={"cluster_id": "clone_cluster_id", "cluster_name": "clone_cluster_name"})
                        if name_map:
                            lookup["clone_cluster_label"] = lookup["clone_cluster_name"].map(lambda v: name_map.get(v, v))
                        else:
                            lookup["clone_cluster_label"] = lookup["clone_cluster_name"]

                        out_df = df.merge(lookup, on=_merge_keys, how="left")
                        csv_p = Path(csv_path)
                        out_name = csv_p.stem + str(p7_out_suffix) + csv_p.suffix
                        out_path = csv_p.parent / out_name
                        out_df.to_csv(out_path, index=False)
                        st.success(f"Saved augmented CSV \u2192 `{out_path}`")
                        st.caption(f"Rows: {len(out_df):,} | New columns: clone_cluster_id, clone_cluster_name, clone_cluster_label")

    # =================================================================
    #  Panel 8 \u2014 Download All Tables
    # =================================================================
    with st.expander("\U0001f4be Panel 8 \u2014 Download All Tables", expanded=False):
        st.info(
            "Compute and download all diversity metric tables (batch-level "
            "and region-level) as CSV files in one click.",
            icon="\u2139\ufe0f",
        )

        with st.form(_sk("p8_form")):
            p8_submitted = st.form_submit_button("\U0001f504 Compute & Prepare Tables", type="primary")

        if p8_submitted:
            tables: dict[str, pd.DataFrame] = {}
            bmet, bcomp = _compute_diversity_table(work, [batch_col], subphenotype_col, min_cells, metadata_cols)
            if not bmet.empty:
                tables["batch_diversity_metrics"] = bmet
            if not bcomp.empty:
                tables["batch_subclone_composition"] = bcomp

            if region_col != "None":
                rmet, rcomp = _compute_diversity_table(work, [batch_col, region_col], subphenotype_col, min_cells, metadata_cols)
                if not rmet.empty:
                    tables["region_diversity_metrics"] = rmet
                if not rcomp.empty:
                    tables["region_subclone_composition"] = rcomp

            if not tables:
                st.warning("No tables to download.")
            else:
                st.session_state[_sk("p8_tables")] = tables

        if _sk("p8_tables") in st.session_state:
            tables = st.session_state[_sk("p8_tables")]
            for tname, tdf in tables.items():
                csv_bytes = tdf.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"\u2b07 Download {tname}.csv",
                    csv_bytes,
                    file_name=f"clonality_{tname}.csv",
                    mime="text/csv",
                    key=_sk(f"p8_dl_{tname}"),
                )
            st.success(f"Prepared {len(tables)} table(s).")
