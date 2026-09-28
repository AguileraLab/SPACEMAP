from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.spatial import cKDTree

REQUIRED_COORDS = ["x_centroid", "y_centroid"]


def _default_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    if preferred in options:
        return options.index(preferred)
    return fallback


def _column_options(columns: list[str]) -> list[str]:
    return [""] + list(columns)


def _preferred_or_blank(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else ""


def _fmt_opt(value: str) -> str:
    return value if value else "(not selected)"


def _safe_unique(series: pd.Series) -> list[str]:
    vals = series.dropna().astype(str).unique().tolist()
    try:
        return sorted(vals)
    except Exception:
        return vals


def _build_strat_label(strat_cols: list[str], values: Any) -> str:
    if not strat_cols:
        return "All"
    if not isinstance(values, tuple):
        values = (values,)
    return " | ".join([f"{c}={v}" for c, v in zip(strat_cols, values)])


def _compute_interaction_stats(
    df: pd.DataFrame,
    phenotype_col: str,
    batch_col: str,
    roi_col: str,
    radius_px: float,
    n_permutations: int,
    min_cells_per_region: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = df.dropna(subset=["x_centroid", "y_centroid", phenotype_col]).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    if batch_col:
        work = work[work[batch_col].notna()].copy()
    else:
        work["__cci_batch__"] = "ALL"
        batch_col = "__cci_batch__"

    if roi_col:
        work = work[work[roi_col].notna()].copy()
    else:
        work["__cci_roi__"] = "ALL"
        roi_col = "__cci_roi__"

    phenotypes = _safe_unique(work[phenotype_col])
    if len(phenotypes) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    ph_to_idx = {p: i for i, p in enumerate(phenotypes)}
    n_ph = len(phenotypes)

    region_payloads: list[tuple[np.ndarray, np.ndarray]] = []
    grouped = work.groupby([batch_col, roi_col], dropna=False)

    for _, gdf in grouped:
        if len(gdf) < int(min_cells_per_region):
            continue

        coords = gdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=float(radius_px), output_type="ndarray")
        if pairs is None or len(pairs) == 0:
            continue

        ph_idx = gdf[phenotype_col].astype(str).map(ph_to_idx).to_numpy(dtype=int)
        idx_a = pairs[:, 0].astype(int)
        idx_b = pairs[:, 1].astype(int)
        region_payloads.append((ph_idx, np.vstack([idx_a, idx_b]).T))

    if not region_payloads:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    def pair_counts(labels, pairs):
        """Neighbouring pairs of each phenotype pair, both ways round (counted with bincount: fast, exact)."""
        a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
        size = n_ph * n_ph
        return (np.bincount(a * n_ph + b, minlength=size) + np.bincount(b * n_ph + a, minlength=size)).reshape(n_ph, n_ph)

    obs = np.zeros((n_ph, n_ph), dtype=float)
    for ph_idx, pairs in region_payloads:
        obs += pair_counts(ph_idx, pairs)

    rng = np.random.default_rng(int(random_state))
    null_sum = np.zeros((n_ph, n_ph), dtype=float)
    null_sumsq = np.zeros((n_ph, n_ph), dtype=float)
    ge_count = np.zeros((n_ph, n_ph), dtype=float)
    le_count = np.zeros((n_ph, n_ph), dtype=float)

    for _ in range(int(n_permutations)):
        perm_counts = np.zeros((n_ph, n_ph), dtype=float)
        for ph_idx, pairs in region_payloads:
            perm_counts += pair_counts(rng.permutation(ph_idx), pairs)

        null_sum += perm_counts
        null_sumsq += perm_counts ** 2
        ge_count += (perm_counts >= obs).astype(float)
        le_count += (perm_counts <= obs).astype(float)

    mean_null = null_sum / float(n_permutations)
    var_null = (null_sumsq / float(n_permutations)) - (mean_null ** 2)
    var_null[var_null < 0] = 0.0
    std_null = np.sqrt(var_null)
    std_null[std_null <= 0] = 1.0

    z = (obs - mean_null) / std_null
    p_enrich = (ge_count + 1.0) / (float(n_permutations) + 1.0)
    p_deplete = (le_count + 1.0) / (float(n_permutations) + 1.0)
    p_two_sided = np.minimum(1.0, 2.0 * np.minimum(p_enrich, p_deplete))

    z_df = pd.DataFrame(z, index=phenotypes, columns=phenotypes)
    p_df = pd.DataFrame(p_two_sided, index=phenotypes, columns=phenotypes)

    rows = []
    for i, p1 in enumerate(phenotypes):
        for j, p2 in enumerate(phenotypes):
            rows.append(
                {
                    "source": p1,
                    "target": p2,
                    "observed_edges": float(obs[i, j]),
                    "null_mean": float(mean_null[i, j]),
                    "null_std": float(std_null[i, j]),
                    "z_score": float(z[i, j]),
                    "p_value": float(p_two_sided[i, j]),
                }
            )
    long_df = pd.DataFrame(rows)

    meta = {
        "n_phenotypes": int(n_ph),
        "n_regions_used": int(len(region_payloads)),
        "n_permutations": int(n_permutations),
        "radius_px": float(radius_px),
    }
    return z_df, p_df, long_df, meta


def _reorder_matrix(z_df: pd.DataFrame) -> list[str]:
    if z_df.empty or z_df.shape[0] < 3:
        return list(z_df.index)
    mat = z_df.to_numpy(dtype=float)
    dist = np.abs(mat - mat.mean(axis=1, keepdims=True))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    order = leaves_list(linkage(condensed, method="average"))
    return [str(z_df.index[i]) for i in order]


def _plot_publication_heatmap(
    z_df: pd.DataFrame,
    p_df: pd.DataFrame,
    title: str,
    cluster_rows: bool,
    annotate_sig: bool,
    z_clip: float,
    figsize_w: float,
    figsize_h: float,
):
    order = _reorder_matrix(z_df) if cluster_rows else list(z_df.index)
    z_plot = z_df.loc[order, order]
    p_plot = p_df.loc[order, order]

    fig, ax = plt.subplots(1, 1, figsize=(figsize_w, figsize_h))
    im = ax.imshow(
        z_plot.values,
        cmap="RdBu_r",
        vmin=-abs(float(z_clip)),
        vmax=abs(float(z_clip)),
        aspect="auto",
    )

    ax.set_xticks(np.arange(z_plot.shape[1]))
    ax.set_yticks(np.arange(z_plot.shape[0]))
    ax.set_xticklabels(z_plot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(z_plot.index, fontsize=8)
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Phenotype")
    ax.set_title(title)

    if annotate_sig:
        for i in range(z_plot.shape[0]):
            for j in range(z_plot.shape[1]):
                p_val = float(p_plot.iat[i, j])
                if p_val < 0.001:
                    mark = "***"
                elif p_val < 0.01:
                    mark = "**"
                elif p_val < 0.05:
                    mark = "*"
                else:
                    mark = ""
                if mark:
                    ax.text(j, i, mark, ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Z-score (enrichment/depletion)")
    plt.tight_layout()
    return fig


def _plot_top_interactions(long_df: pd.DataFrame, title: str, top_n: int = 20):
    df = long_df.copy()
    df = df[df["source"] != df["target"]].copy()
    if df.empty:
        return None

    df["abs_z"] = df["z_score"].abs()
    top = df.sort_values("abs_z", ascending=False).head(int(top_n)).copy()
    if top.empty:
        return None

    labels = [f"{r.source} ↔ {r.target}" for r in top.itertuples(index=False)]
    colors = ["#b2182b" if float(v) > 0 else "#2166ac" for v in top["z_score"].values]

    fig, ax = plt.subplots(1, 1, figsize=(10, max(4.0, 0.3 * len(top) + 1.5)))
    y = np.arange(len(top))
    ax.barh(y, top["z_score"].values, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Z-score")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")
    plt.tight_layout()
    return fig


def render_cell_cell_interaction(project_root: Path, df: pd.DataFrame):
    st.subheader("Cell-Cell Interaction")
    st.caption(
        "Permutation-based spatial neighborhood enrichment for cell-cell interactions (Schapiro/histoCAT-style)."
    )

    if len(df.columns) < 2:
        st.error("Need at least two columns to select X/Y coordinates.")
        return

    all_columns = list(df.columns)
    coord_options = _column_options(all_columns)
    default_x_col = _preferred_or_blank(coord_options, "x_centroid")
    default_y_col = _preferred_or_blank(coord_options, "y_centroid")

    with st.expander("Configuration", expanded=True):
        xy1, xy2 = st.columns(2)
        x_coord_col = xy1.selectbox(
            "X Coordinate Column",
            options=coord_options,
            index=_default_index(coord_options, default_x_col, fallback=0),
            key="tab8_cci_x_coord_col",
            format_func=_fmt_opt,
        )
        y_coord_col = xy2.selectbox(
            "Y Coordinate Column",
            options=coord_options,
            index=_default_index(coord_options, default_y_col, fallback=0),
            key="tab8_cci_y_coord_col",
            format_func=_fmt_opt,
        )

        selectable_cols = [c for c in df.columns if c not in {x_coord_col, y_coord_col, ""}]
        phenotype_candidates = [c for c in selectable_cols if "phenotype" in c.lower()]

        c1, c2, c3 = st.columns(3)
        phenotype_col = c1.selectbox(
            "Phenotyping column",
            options=["None"] + phenotype_candidates,
            index=_default_index(["None"] + phenotype_candidates, "phenotype", fallback=0),
            key="tab8_cci_pheno_col",
            help="Cell phenotype labels used for interaction counting.",
        )
        batch_col_options = [""] + selectable_cols
        batch_col = c2.selectbox(
            "Batch ID Column",
            options=batch_col_options,
            index=_default_index(batch_col_options, "batch_id", fallback=0),
            key="tab8_cci_batch_col",
            format_func=_fmt_opt,
            help="Cells are analyzed within batch and ROI to avoid cross-fragment interactions.",
        )
        roi_col_options = [""] + selectable_cols
        roi_col = c3.selectbox(
            "Tissue ROI Column",
            options=roi_col_options,
            index=_default_index(roi_col_options, "tissue_roi", fallback=0),
            key="tab8_cci_roi_col",
            format_func=_fmt_opt,
            help="Region identifier for disconnected tissue pieces.",
        )

        d1, d2, d3, d4 = st.columns(4)
        radius_um = d1.number_input(
            "Interaction radius (µm)",
            min_value=1.0,
            max_value=300.0,
            value=30.0,
            step=1.0,
            key="tab8_cci_radius_um",
            help="Cells are considered interacting when within this distance.",
        )
        pixel_size_um = d2.number_input(
            "Pixel Size (µm)",
            min_value=0.01,
            value=0.5083,
            step=0.0001,
            format="%.4f",
            key="tab8_cci_pixel_um",
            help="Converts physical radius (µm) to pixel radius for KD-tree graph construction.",
        )
        n_permutations = d3.number_input(
            "Permutations",
            min_value=50,
            max_value=5000,
            value=500,
            step=50,
            key="tab8_cci_perms",
            help="Higher values improve p-value stability but increase runtime.",
        )
        min_cells_region = d4.number_input(
            "Min cells per region",
            min_value=5,
            max_value=5000,
            value=20,
            step=1,
            key="tab8_cci_min_cells",
            help="Regions with too few cells are skipped.",
        )

        e1, e2, e3 = st.columns(3)
        tissue_col_options = ["None"] + selectable_cols
        tissue_col = e1.selectbox(
            "Tissue filter column (optional)",
            options=tissue_col_options,
            index=_default_index(tissue_col_options, "tissue", fallback=0),
            key="tab8_cci_tissue_col",
            help="Optional prefilter (for example tissue == 1).",
        )
        tissue_keep = e2.text_input("Tissue keep value", value="1", key="tab8_cci_tissue_val")
        random_state = e3.number_input("random_state", min_value=0, max_value=100000, value=42, step=1, key="tab8_cci_random")

        f1, f2 = st.columns(2)
        strat_cols = f1.multiselect(
            "Optional stratification columns",
            options=selectable_cols,
            default=[],
            key="tab8_cci_strat_cols",
            help="If selected, interaction analysis runs independently for each stratum combination.",
        )
        phenotype_subset = f2.multiselect(
            "Optional phenotype subset",
            options=_safe_unique(df[phenotype_col]) if phenotype_col != "None" and phenotype_col in df.columns else [],
            default=[],
            key="tab8_cci_pheno_subset",
            help="Leave empty to include all phenotypes.",
        )

        g1, g2, g3, g4 = st.columns(4)
        cluster_heatmap = g1.checkbox("Cluster heatmap axes", value=True, key="tab8_cci_cluster_hm")
        annotate_sig = g2.checkbox("Annotate significant cells (*)", value=True, key="tab8_cci_annot_sig")
        z_clip = g3.slider("Heatmap z-range", min_value=1.0, max_value=10.0, value=4.0, step=0.5, key="tab8_cci_z_clip")
        top_n_pairs = g4.slider("Top interactions (barplot)", min_value=5, max_value=60, value=20, step=1, key="tab8_cci_top_n")

        h1, h2, h3 = st.columns(3)
        fig_w = h1.slider("Heatmap width", min_value=6.0, max_value=18.0, value=10.0, step=0.5, key="tab8_cci_fig_w")
        fig_h = h2.slider("Heatmap height", min_value=5.0, max_value=18.0, value=9.0, step=0.5, key="tab8_cci_fig_h")
        output_dir = h3.text_input(
            "Output Directory",
            value=str(project_root / "outputs" / "spatial_analysis" / "cell_cell_interaction"),
            key="tab8_cci_outdir",
        )

        with st.expander("How to use", expanded=False):
            st.markdown(
                "- Build a radius-based interaction graph within each disconnected tissue region (batch/ROI).  \n"
                "- Compare observed adjacency counts against permutation null distributions.  \n"
                "- **Z > 2**: enriched interactions; **Z < -2**: depleted/avoidance interactions.  \n"
                "- Use optional stratification to compare interaction structure across cohorts/conditions."
            )

        with st.expander("Example interpretation", expanded=False):
            st.markdown(
                "- If `CD8_T ↔ Tumor` has strongly **negative Z**, it supports immune exclusion.  \n"
                "- If `B_cell ↔ Tfh` has strongly **positive Z**, it supports TLS-like organization.  \n"
                "- Compare strata (e.g., responder vs non-responder) to find interaction rewiring."
            )

    if not x_coord_col or not y_coord_col:
        st.info("Select X and Y coordinate columns to enable cell-cell interaction analysis.")
        return

    if x_coord_col == y_coord_col:
        st.error("X Coordinate Column and Y Coordinate Column must be different.")
        return

    work_df = df.copy()
    work_df["x_centroid"] = pd.to_numeric(work_df[x_coord_col], errors="coerce")
    work_df["y_centroid"] = pd.to_numeric(work_df[y_coord_col], errors="coerce")

    if not st.button("Run Cell-Cell Interaction", type="primary", key="tab8_cci_run"):
        return

    if phenotype_col == "None":
        st.error("Please select a phenotyping column.")
        return

    run_df = work_df.copy()
    if tissue_col != "None" and tissue_col in run_df.columns:
        run_df = run_df[run_df[tissue_col].astype(str) == str(tissue_keep)]

    run_df = run_df.dropna(subset=REQUIRED_COORDS + [phenotype_col])
    if phenotype_subset:
        run_df = run_df[run_df[phenotype_col].astype(str).isin([str(v) for v in phenotype_subset])]

    if run_df.empty:
        st.warning("No rows left after current filters.")
        return

    radius_px = float(radius_um) / float(pixel_size_um)
    if radius_px <= 0:
        st.error("Computed radius in pixels must be > 0.")
        return

    if strat_cols:
        strata = list(run_df.groupby(strat_cols, dropna=False))
    else:
        strata = [("All", run_df)]

    all_long_parts: list[pd.DataFrame] = []
    all_meta_rows: list[dict[str, Any]] = []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prog = st.progress(0)
    for i, (stratum_key, sdf) in enumerate(strata):
        if len(sdf) < max(10, int(min_cells_region)):
            prog.progress((i + 1) / max(len(strata), 1))
            continue

        z_df, p_df, long_df, meta = _compute_interaction_stats(
            df=sdf,
            phenotype_col=phenotype_col,
            batch_col=batch_col,
            roi_col=roi_col,
            radius_px=float(radius_px),
            n_permutations=int(n_permutations),
            min_cells_per_region=int(min_cells_region),
            random_state=int(random_state),
        )

        if z_df.empty:
            prog.progress((i + 1) / max(len(strata), 1))
            continue

        label = _build_strat_label(strat_cols, stratum_key)
        st.markdown(f"### Interaction results: {label}")

        fig = _plot_publication_heatmap(
            z_df=z_df,
            p_df=p_df,
            title=f"Cell-Cell Interaction Enrichment | {label}",
            cluster_rows=bool(cluster_heatmap),
            annotate_sig=bool(annotate_sig),
            z_clip=float(z_clip),
            figsize_w=float(fig_w),
            figsize_h=float(fig_h),
        )
        st.pyplot(fig)
        fig.savefig(out_dir / f"cci_heatmap_{str(label).replace(' | ', '__').replace('=', '-')}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        top_fig = _plot_top_interactions(long_df, title=f"Top interaction effects | {label}", top_n=int(top_n_pairs))
        if top_fig is not None:
            st.pyplot(top_fig)
            top_fig.savefig(out_dir / f"cci_top_pairs_{str(label).replace(' | ', '__').replace('=', '-')}.png", dpi=300, bbox_inches="tight")
            plt.close(top_fig)

        z_df.to_csv(out_dir / f"cci_zscores_{str(label).replace(' | ', '__').replace('=', '-')}.csv")
        p_df.to_csv(out_dir / f"cci_pvalues_{str(label).replace(' | ', '__').replace('=', '-')}.csv")

        long_df = long_df.copy()
        long_df["stratum"] = label
        all_long_parts.append(long_df)

        meta_row = {"stratum": label}
        meta_row.update(meta)
        all_meta_rows.append(meta_row)

        prog.progress((i + 1) / max(len(strata), 1))

    prog.empty()

    if not all_long_parts:
        st.warning("No valid stratum produced interaction outputs. Try relaxing filters or region thresholds.")
        return

    long_all = pd.concat(all_long_parts, ignore_index=True)
    meta_df = pd.DataFrame(all_meta_rows)

    long_csv = out_dir / "cci_all_strata_long.csv"
    meta_csv = out_dir / "cci_run_metadata.csv"
    long_all.to_csv(long_csv, index=False)
    meta_df.to_csv(meta_csv, index=False)

    st.success("Cell-cell interaction analysis complete.")
    st.dataframe(meta_df, width="stretch")
    st.write(
        {
            "Output_dir": str(out_dir),
            "Long_table_csv": str(long_csv),
            "Metadata_csv": str(meta_csv),
            "Strata_analyzed": int(len(meta_df)),
            "Rows_long_table": int(len(long_all)),
        }
    )
