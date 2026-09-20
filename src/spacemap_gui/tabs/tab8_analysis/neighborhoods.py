from __future__ import annotations

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree

from .common import load_unified_csv, queue_tab8_csv_reload
from .virtual_tissue import _build_region_geometry, _compute_buffer_distance

REQUIRED_COORDS = ["x_centroid", "y_centroid"]


def _default_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    if preferred in options:
        return options.index(preferred)
    return fallback


def _column_options(columns: list[str]) -> list[str]:
    return [""] + list(columns)


def _preferred_or_blank(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else ""


def _format_opt(opt: str) -> str:
    return opt if opt else "(not selected)"


def _compute_neighborhood_features(
    df: pd.DataFrame,
    phenotype_col: str,
    batch_col: str,
    roi_col: str,
    k_neighbors: int,
    min_cells_per_region: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:
        raise RuntimeError(f"scikit-learn is required for neighborhood detection: {exc}")

    work = df.copy()
    work = work.dropna(subset=REQUIRED_COORDS + [phenotype_col])

    if batch_col:
        work = work[work[batch_col].notna()].copy()
    else:
        work["__nb_batch__"] = "ALL"
        batch_col = "__nb_batch__"

    if roi_col:
        work = work[work[roi_col].notna()].copy()
    else:
        work["__nb_roi__"] = "ALL"
        roi_col = "__nb_roi__"

    if work.empty:
        return pd.DataFrame(), [], []

    phenotypes = sorted(work[phenotype_col].astype(str).dropna().unique().tolist())
    if not phenotypes:
        return pd.DataFrame(), [], []

    pheno_to_idx = {p: i for i, p in enumerate(phenotypes)}

    freq_parts: list[pd.DataFrame] = []
    meta_parts: list[pd.DataFrame] = []
    grouped = work.groupby([batch_col, roi_col], dropna=False)

    for (batch_val, roi_val), gdf in grouped:
        if len(gdf) < max(int(min_cells_per_region), 2):
            continue

        n_neighbors = min(int(k_neighbors), int(len(gdf)))
        if n_neighbors < 2:
            continue

        coords = gdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree")
        nbrs.fit(coords)
        indices = nbrs.kneighbors(coords, return_distance=False)

        ph_arr = gdf[phenotype_col].astype(str).to_numpy()
        freq = np.zeros((len(gdf), len(phenotypes)), dtype=float)

        for i in range(len(gdf)):
            neigh_idx = indices[i]
            neigh_phenos = ph_arr[neigh_idx]
            uniq_vals, counts = np.unique(neigh_phenos, return_counts=True)
            denom = float(len(neigh_idx)) if len(neigh_idx) > 0 else 1.0
            for val, cnt in zip(uniq_vals, counts):
                idx = pheno_to_idx.get(str(val))
                if idx is not None:
                    freq[i, idx] = float(cnt) / denom

        freq_cols = [f"freq_{p}" for p in phenotypes]
        freq_df = pd.DataFrame(freq, index=gdf.index, columns=freq_cols)
        meta_df = pd.DataFrame(
            {
                "__nb_batch_val__": [str(batch_val)] * len(gdf),
                "__nb_roi_val__": [str(roi_val)] * len(gdf),
            },
            index=gdf.index,
        )
        freq_parts.append(freq_df)
        meta_parts.append(meta_df)

    if not freq_parts:
        return pd.DataFrame(), [], phenotypes

    all_freq = pd.concat(freq_parts, axis=0)
    all_meta = pd.concat(meta_parts, axis=0)
    out = work.join(all_freq, how="left").join(all_meta, how="left")
    return out, [f"freq_{p}" for p in phenotypes], phenotypes


def _build_cluster_palette(cluster_ids: list[int]) -> dict[int, str]:
    cmap = plt.get_cmap("tab20", max(len(cluster_ids), 1))
    out: dict[int, str] = {}
    for i, cid in enumerate(cluster_ids):
        rgba = cmap(i)
        out[int(cid)] = "#{:02x}{:02x}{:02x}".format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
    return out


def _build_label_color_map(export_df: pd.DataFrame, merged_color_map: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    label_color_map: dict[str, str] = {}
    warnings: list[str] = []

    if "CN_name_merged" not in export_df.columns:
        return label_color_map, warnings

    work = export_df.copy()
    work["CN_name_merged"] = work["CN_name_merged"].astype(str)
    work["CN_cluster_merged"] = work["CN_cluster_merged"].astype(str)

    for merged_name, group in work.groupby("CN_name_merged", dropna=False):
        cluster_ids = [str(v) for v in group["CN_cluster_merged"].dropna().astype(str).unique().tolist()]
        candidate_colors = [merged_color_map.get(cluster_id, "#1f77b4") for cluster_id in cluster_ids]
        unique_colors = list(dict.fromkeys(candidate_colors))
        if len(unique_colors) > 1:
            warnings.append(
                f"Merged annotation '{merged_name}' is linked to multiple colors; using '{unique_colors[0]}' in the overlay."
            )
        label_color_map[str(merged_name)] = unique_colors[0] if unique_colors else "#1f77b4"

    return label_color_map, warnings


def _editor_key_suffix(phenotype_col: str, cid: int, default_name: str) -> str:
    safe_pheno = ''.join(ch if ch.isalnum() else '_' for ch in str(phenotype_col))
    safe_name = ''.join(ch if ch.isalnum() else '_' for ch in str(default_name))
    return f"{safe_pheno}_{cid}_{safe_name}"[:120]


def _make_nb_signature(
    csv_path: str,
    x_coord_col: str,
    y_coord_col: str,
    phenotype_col: str,
    batch_col: str,
    roi_col: str,
    k_neighbors: int,
    n_neighborhoods: int,
    random_state: int,
    normalize_features: bool,
    min_cells_per_region: int,
) -> str:
    return "|".join(
        [
            str(csv_path),
            str(x_coord_col),
            str(y_coord_col),
            str(phenotype_col),
            str(batch_col),
            str(roi_col),
            str(int(k_neighbors)),
            str(int(n_neighborhoods)),
            str(int(random_state)),
            str(int(bool(normalize_features))),
            str(int(min_cells_per_region)),
        ]
    )


def _build_profile_summary(cn_profiles: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
    top_n = min(3, len(cn_profiles.columns)) if len(cn_profiles.columns) > 0 else 0
    profile_rows = []
    for cid, row in cn_profiles.iterrows():
        cid_int = int(str(cid))
        order = row.sort_values(ascending=False)
        top_items = order.head(top_n)
        top_txt = ", ".join([f"{idx} ({val:.2f})" for idx, val in top_items.items()]) if top_n > 0 else ""
        profile_rows.append(
            {
                "CN_cluster": cid_int,
                "n_cells": int((cluster_df["CN_cluster"] == cid_int).sum()),
                "top_phenotypes": top_txt,
            }
        )
    return pd.DataFrame(profile_rows).sort_values("CN_cluster")


def _build_editor_table(profile_summary_df: pd.DataFrame) -> pd.DataFrame:
    cluster_ids = sorted(profile_summary_df["CN_cluster"].astype(int).tolist())
    cluster_palette = _build_cluster_palette(cluster_ids)
    rows = []
    for cid in cluster_ids:
        default_name = f"CN_{cid}"
        top_val = profile_summary_df.loc[profile_summary_df["CN_cluster"] == cid, "top_phenotypes"]
        if not top_val.empty and str(top_val.iloc[0]).strip():
            default_name = f"CN_{cid}: {str(top_val.iloc[0]).split(',')[0]}"
        n_cells = int(profile_summary_df.loc[profile_summary_df["CN_cluster"] == cid, "n_cells"].iloc[0])
        top_text = str(top_val.iloc[0]) if not top_val.empty else ""
        rows.append(
            {
                "CN_cluster": int(cid),
                "n_cells": n_cells,
                "top_phenotypes": top_text,
                "original_name": default_name,
                "merged_cluster": str(cid),
                "merged_name": default_name,
                "merged_color": cluster_palette.get(cid, "#1f77b4"),
            }
        )
    return pd.DataFrame(rows)


def _apply_editor_to_results(
    cluster_df: pd.DataFrame,
    freq_cols: list[str],
    editor_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str], list[str]]:
    work = editor_df.copy()
    work["CN_cluster"] = pd.to_numeric(work["CN_cluster"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["CN_cluster"]).copy()
    work["CN_cluster"] = work["CN_cluster"].astype(int)
    work["merged_cluster"] = work["merged_cluster"].astype(str).str.strip()
    work.loc[work["merged_cluster"] == "", "merged_cluster"] = work["CN_cluster"].astype(str)
    work["original_name"] = work["original_name"].astype(str).str.strip()
    work.loc[work["original_name"] == "", "original_name"] = work["CN_cluster"].map(lambda value: f"CN_{int(value)}")
    work["merged_name"] = work["merged_name"].astype(str).str.strip()
    work.loc[work["merged_name"] == "", "merged_name"] = work["merged_cluster"]
    work["merged_color"] = work["merged_color"].astype(str).str.strip()

    warnings: list[str] = []
    orig_name_map = dict(zip(work["CN_cluster"], work["original_name"]))
    orig_to_merged = dict(zip(work["CN_cluster"], work["merged_cluster"]))

    merged_name_map: dict[str, str] = {}
    merged_color_map: dict[str, str] = {}
    for merged_cluster, group in work.groupby("merged_cluster", dropna=False):
        names = [str(v).strip() for v in group["merged_name"].tolist() if str(v).strip()]
        colors = [str(v).strip() for v in group["merged_color"].tolist() if str(v).strip()]
        unique_names = list(dict.fromkeys(names))
        unique_colors = list(dict.fromkeys(colors))
        if len(unique_names) > 1:
            warnings.append(
                f"Merged neighborhood {merged_cluster} has multiple names; using '{unique_names[0]}'."
            )
        if len(unique_colors) > 1:
            warnings.append(
                f"Merged neighborhood {merged_cluster} has multiple colors; using '{unique_colors[0]}'."
            )
        merged_name_map[str(merged_cluster)] = unique_names[0] if unique_names else str(merged_cluster)
        merged_color_map[str(merged_cluster)] = unique_colors[0] if unique_colors else "#1f77b4"

    export_df = cluster_df.copy()
    export_df["CN_cluster_original"] = export_df["CN_cluster"].astype(int)
    export_df["CN_name_original"] = export_df["CN_cluster_original"].map(orig_name_map).astype(str)
    export_df["CN_cluster_merged"] = export_df["CN_cluster_original"].map(orig_to_merged).astype(str)
    export_df["CN_name_merged"] = export_df["CN_cluster_merged"].map(merged_name_map).astype(str)

    merged_profiles = export_df.groupby("CN_cluster_merged", dropna=False)[freq_cols].mean()
    merged_summary_rows = []
    top_n = min(3, len(merged_profiles.columns)) if len(merged_profiles.columns) > 0 else 0
    for merged_cluster, row in merged_profiles.iterrows():
        order = row.sort_values(ascending=False)
        top_items = order.head(top_n)
        top_txt = ", ".join([f"{idx} ({val:.2f})" for idx, val in top_items.items()]) if top_n > 0 else ""
        merged_summary_rows.append(
            {
                "merged_cluster": str(merged_cluster),
                "merged_name": merged_name_map.get(str(merged_cluster), str(merged_cluster)),
                "n_cells": int((export_df["CN_cluster_merged"].astype(str) == str(merged_cluster)).sum()),
                "top_phenotypes": top_txt,
            }
        )
    merged_summary_df = pd.DataFrame(merged_summary_rows).sort_values(["merged_name", "merged_cluster"])

    return export_df, merged_profiles, merged_summary_df, merged_name_map, merged_color_map, warnings


def render_neighborhoods(project_root: Path, df: pd.DataFrame, csv_path: str):
    st.subheader("Cellular Neighborhood Detection")
    st.caption(
        "Identify recurring cellular neighborhoods (CNs) by summarizing each cell's local phenotype mix "
        "and clustering those local composition vectors into recurring neighborhood types."
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
            key="tab8_nb_x_coord_col",
            format_func=_format_opt,
        )
        y_coord_col = xy2.selectbox(
            "Y Coordinate Column",
            options=coord_options,
            index=_default_index(coord_options, default_y_col, fallback=0),
            key="tab8_nb_y_coord_col",
            format_func=_format_opt,
        )

        all_cols = [c for c in df.columns if c not in {x_coord_col, y_coord_col, ""}]
        phenotype_candidates = [c for c in all_cols if "phenotype" in c.lower()]

        c1, c2, c3 = st.columns(3)
        phenotype_col = c1.selectbox(
            "Phenotyping column",
            options=["None"] + phenotype_candidates,
            index=_default_index(["None"] + phenotype_candidates, "phenotype", fallback=0),
            key="tab8_nb_pheno_col",
            help="Cell phenotype label column used to build neighborhood composition vectors.",
        )

        batch_options = [""] + all_cols
        batch_col = c2.selectbox(
            "Batch / Slide Column",
            options=batch_options,
            index=_default_index(batch_options, "batch_id", fallback=0),
            key="tab8_nb_batch_col",
            format_func=_format_opt,
            help="Groups cells by slide/sample. KNN is computed within each group to prevent cross-slide neighbors.",
        )

        roi_options = [""] + all_cols
        roi_col = c3.selectbox(
            "Tissue ROI Column",
            options=roi_options,
            index=_default_index(roi_options, "tissue_roi", fallback=0),
            key="tab8_nb_roi_col",
            format_func=_format_opt,
            help="Groups cells by disconnected tissue region (e.g. TMA cores). KNN stays within each ROI.",
        )

        st.caption(
            "**Grouping**: KNN is fitted within each *(Batch × ROI)* group. "
            "If ROI is unset → groups by Batch only. If both unset → all cells form one group."
        )

        d1, d2, d3, d4 = st.columns(4)
        k_neighbors = d1.number_input(
            "k neighbors", min_value=3, max_value=100, value=10, step=1, key="tab8_nb_k",
            help="Defines the local window around each cell. Higher values make each cell's neighborhood summary smoother and more global; lower values make it more local and sensitive.",
        )
        n_neighborhoods = d2.number_input(
            "n neighborhoods", min_value=2, max_value=50, value=10, step=1, key="tab8_nb_kmeans_k",
            help="Defines how many neighborhood types the clustering step should produce from those local windows. Higher values split patterns more finely; lower values merge them into broader classes.",
        )
        random_state = d3.number_input("random_state", min_value=0, max_value=100000, value=42, step=1, key="tab8_nb_random")
        min_cells_per_region = d4.number_input(
            "Min cells per region", min_value=5, max_value=5000, value=20, step=1, key="tab8_nb_min_cells",
            help="Regions (Batch × ROI) with fewer cells are skipped.",
        )

        with st.expander("What is the difference between k neighbors and n neighborhoods?", expanded=False):
            st.markdown(
                "- **k neighbors** controls how each cell is described. It sets the size of the local spatial window used to compute the phenotype composition vector.  \n"
                "- **n neighborhoods** controls how those cell-level windows are grouped afterward. It sets the number of clusters the algorithm will return.  \n"
                "- In short: **k neighbors = local context size**, **n neighborhoods = final number of neighborhood classes**."
            )

        normalize_features = st.checkbox(
            "Z-score features before clustering", value=True, key="tab8_nb_scale",
            help="Standardize composition vectors before KMeans. Helps when rare phenotypes would otherwise be overshadowed.",
        )

        f1, f2, f3, f4 = st.columns(4)
        output_dir = f1.text_input(
            "Output Directory",
            value=str(project_root / "outputs" / "spatial_analysis" / "neighborhoods"),
            key="tab8_nb_outdir",
        )
        output_col_name = f2.text_input("Merged neighborhood name column", value="neighborhood", key="tab8_nb_out_col")
        output_cluster_col = f3.text_input("Merged neighborhood id column", value="neighborhood_cluster", key="tab8_nb_cluster_col")
        output_original_cluster_col = f4.text_input("Original CN cluster column", value="cn_cluster_original", key="tab8_nb_orig_cluster_col")

        output_original_name_col = st.text_input(
            "Original CN name column",
            value="cn_name_original",
            key="tab8_nb_orig_name_col",
        )

        g1, g2 = st.columns(2)
        save_mode = g1.selectbox(
            "Write-back mode",
            options=["Save as new CSV", "Overwrite active CSV"],
            index=0,
            key="tab8_nb_save_mode",
        )
        save_suffix = g2.text_input("New CSV suffix", value="_with_neighborhoods", key="tab8_nb_save_suffix")

        st.markdown("#### Model-selection helper")
        st.caption("Use elbow + silhouette to pick a data-driven number of neighborhoods.")
        m1, m2, m3 = st.columns(3)
        k_min = m1.number_input(
            "K min",
            min_value=2,
            max_value=50,
            value=4,
            step=1,
            key="tab8_nb_model_k_min",
            help="Smallest cluster count tested in the helper.",
        )
        k_max = m2.number_input(
            "K max",
            min_value=3,
            max_value=80,
            value=14,
            step=1,
            key="tab8_nb_model_k_max",
            help="Largest cluster count tested in the helper.",
        )
        silhouette_sample = m3.number_input(
            "Max cells for silhouette",
            min_value=500,
            max_value=50000,
            value=8000,
            step=500,
            key="tab8_nb_model_sil_sample",
            help="Silhouette can be expensive; if data is larger, random subsampling is used.",
        )

        with st.expander("How to use model selection", expanded=False):
            st.markdown(
                "- **Elbow (Inertia)**: lower is better; look for the point where improvement starts flattening.  \n"
                "- **Silhouette**: higher is better ($-1$ to $1$), measuring cluster separation quality.  \n"
                "- Choose **K** near the elbow and with relatively high silhouette, then validate biologically from CN profiles."
            )

        st.markdown("#### k-neighbor stability helper")
        st.caption("Estimate a robust k neighbors value by testing clustering stability across k choices.")
        ks1, ks2, ks3, ks4 = st.columns(4)
        k_stab_min = ks1.number_input(
            "k min",
            min_value=3,
            max_value=100,
            value=6,
            step=1,
            key="tab8_nb_k_stab_min",
            help="Smallest k-neighbors tested.",
        )
        k_stab_max = ks2.number_input(
            "k max",
            min_value=4,
            max_value=120,
            value=20,
            step=1,
            key="tab8_nb_k_stab_max",
            help="Largest k-neighbors tested.",
        )
        k_stab_step = ks3.number_input(
            "k step",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            key="tab8_nb_k_stab_step",
            help="Step between tested k values.",
        )
        k_stab_repeats = ks4.number_input(
            "Repeats / k",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key="tab8_nb_k_stab_repeats",
            help="Independent clustering runs per k to estimate label stability.",
        )

        ks5 = st.number_input(
            "Max cells for k-stability helper",
            min_value=500,
            max_value=50000,
            value=12000,
            step=500,
            key="tab8_nb_k_stab_max_cells",
            help="If data is larger, random subsampling is used for speed.",
        )

        with st.expander("How to use k-neighbor stability", expanded=False):
            st.markdown(
                "- For each tested k-neighbors value, the helper reruns clustering multiple times.  \n"
                "- **Stability (ARI)** close to 1 means labels are reproducible across runs.  \n"
                "- **Silhouette** indicates separation quality; higher is better.  \n"
                "- Pick k with strong stability and good silhouette, then validate biological interpretability."
            )

    if not x_coord_col or not y_coord_col:
        st.info("Select X and Y coordinate columns to enable neighborhood analysis.")
        _ss = st.session_state.get("tab8_nb_result")
        _e = _ss.get("merged_export_df") if isinstance(_ss, dict) else None
        _render_cn_distance_panel(
            full_df=df,
            export_df=_e if isinstance(_e, pd.DataFrame) else None,
            merged_name_map=dict(_ss.get("merged_name_map") or {}) if isinstance(_ss, dict) else {},
            merged_color_map=dict(_ss.get("merged_color_map") or {}) if isinstance(_ss, dict) else {},
            batch_col=batch_col if batch_col else "",
            roi_col=roi_col if roi_col else "",
            csv_path=csv_path,
            save_mode=save_mode,
            save_suffix=save_suffix,
            all_columns=all_columns,
            x_coord_col="",
            y_coord_col="",
        )
        return

    if x_coord_col == y_coord_col:
        st.error("X Coordinate Column and Y Coordinate Column must be different.")
        _ss = st.session_state.get("tab8_nb_result")
        _e = _ss.get("merged_export_df") if isinstance(_ss, dict) else None
        _render_cn_distance_panel(
            full_df=df,
            export_df=_e if isinstance(_e, pd.DataFrame) else None,
            merged_name_map=dict(_ss.get("merged_name_map") or {}) if isinstance(_ss, dict) else {},
            merged_color_map=dict(_ss.get("merged_color_map") or {}) if isinstance(_ss, dict) else {},
            batch_col=batch_col if batch_col else "",
            roi_col=roi_col if roi_col else "",
            csv_path=csv_path,
            save_mode=save_mode,
            save_suffix=save_suffix,
            all_columns=all_columns,
            x_coord_col=x_coord_col,
            y_coord_col=y_coord_col,
        )
        return

    work_df = df.copy()
    work_df["x_centroid"] = pd.to_numeric(work_df[x_coord_col], errors="coerce")
    work_df["y_centroid"] = pd.to_numeric(work_df[y_coord_col], errors="coerce")

    # ==================================================================
    # Load Existing Neighborhood Results (skip re-running)
    # ==================================================================
    _nb_detect_cols = [output_col_name, output_cluster_col, output_original_cluster_col, output_original_name_col]
    _nb_existing_cols_found = [c for c in _nb_detect_cols if c in df.columns]

    if _nb_existing_cols_found:
        with st.expander("\u2705 Existing Neighborhood columns detected \u2014 Load or Skip", expanded=False):
            st.info(
                f"The unified CSV already contains neighborhood columns: "
                f"**{', '.join(_nb_existing_cols_found)}**. "
                "You can load these existing results (and edit labels) without "
                "re-running the full neighborhood detection pipeline."
            )
            if st.button("Load existing neighborhood results from CSV", key="tab8_nb_load_existing"):
                try:
                    _load_ok = True
                    _need_cols = [output_col_name, output_cluster_col]
                    _missing = [c for c in _need_cols if c not in df.columns]
                    if _missing:
                        st.error(f"Cannot load: missing columns {_missing}")
                        _load_ok = False

                    if _load_ok:
                        # Build a synthetic result matching the same shape as
                        # a freshly computed neighborhood detection run.
                        _load_df = work_df.copy()

                        # Map existing CSV columns into the internal names
                        if output_col_name in df.columns:
                            _load_df["CN_name_merged"] = df[output_col_name].astype(str)
                        if output_cluster_col in df.columns:
                            _load_df["CN_cluster_merged"] = df[output_cluster_col].astype(str)
                        if output_original_cluster_col in df.columns:
                            _load_df["CN_cluster_original"] = pd.to_numeric(
                                df[output_original_cluster_col], errors="coerce"
                            ).fillna(0).astype(int)
                            _load_df["CN_cluster"] = _load_df["CN_cluster_original"]
                        else:
                            # Synthesise original cluster from merged
                            _load_df["CN_cluster_original"] = pd.Categorical(
                                _load_df["CN_cluster_merged"]
                            ).codes.astype(int)
                            _load_df["CN_cluster"] = _load_df["CN_cluster_original"]
                        if output_original_name_col in df.columns:
                            _load_df["CN_name_original"] = df[output_original_name_col].astype(str)
                        else:
                            _load_df["CN_name_original"] = _load_df["CN_cluster_original"].astype(str)

                        _load_df["__nb_batch_val__"] = (
                            _load_df[batch_col].astype(str) if batch_col and batch_col in _load_df.columns
                            else "ALL"
                        )
                        _load_df["__nb_roi_val__"] = (
                            _load_df[roi_col].astype(str) if roi_col and roi_col in _load_df.columns
                            else "ALL"
                        )

                        # Build CN profiles from loaded data if phenotype_col is set
                        _loaded_freq_cols: list[str] = []
                        _loaded_profiles = pd.DataFrame()
                        if phenotype_col and phenotype_col != "None" and phenotype_col in _load_df.columns:
                            _phenos = sorted(_load_df[phenotype_col].dropna().astype(str).unique().tolist())
                            # Compute frequency vectors per CN cluster
                            for _p in _phenos:
                                _fc = f"freq_{_p}"
                                _loaded_freq_cols.append(_fc)
                                _load_df[_fc] = (_load_df[phenotype_col].astype(str) == _p).astype(float)
                            if _loaded_freq_cols:
                                _loaded_profiles = _load_df.groupby("CN_cluster", dropna=False)[_loaded_freq_cols].mean()
                                _loaded_profiles.columns = [c.replace("freq_", "") for c in _loaded_profiles.columns]

                        _loaded_summary = _build_profile_summary(_loaded_profiles, _load_df) if not _loaded_profiles.empty else pd.DataFrame()

                        # Compute a signature for loaded results
                        _loaded_sig = f"loaded_{len(_load_df)}_{output_col_name}"

                        st.session_state["tab8_nb_result"] = {
                            "signature": _loaded_sig,
                            "cluster_df": _load_df,
                            "cn_profiles": _loaded_profiles if not _loaded_profiles.empty else pd.DataFrame(),
                            "profile_summary_df": _loaded_summary,
                            "freq_cols": _loaded_freq_cols,
                            "phenotype_col": str(phenotype_col) if phenotype_col else "None",
                        }
                        st.session_state[f"tab8_nb_editor_{_loaded_sig}"] = (
                            _build_editor_table(_loaded_summary) if not _loaded_summary.empty
                            else pd.DataFrame()
                        )
                        st.success(
                            f"Loaded existing neighborhood results: "
                            f"{_load_df['CN_name_merged'].nunique()} merged neighborhoods, "
                            f"{len(_load_df)} cells."
                        )
                        st.rerun()
                except Exception as _exc:
                    st.error(f"Failed to load existing neighborhood results: {_exc}")


    run_model_helper = st.button("Run Model-selection Helper", key="tab8_nb_run_model_helper")
    run_k_stability_helper = st.button("Run k-neighbor Stability Helper", key="tab8_nb_run_k_stability_helper")

    if run_model_helper:
        if phenotype_col == "None":
            st.error("Please select a phenotyping column before running model selection.")
            return

        if int(k_max) <= int(k_min):
            st.error("K max must be greater than K min.")
            return

        feature_df_h, freq_cols_h, _ = _compute_neighborhood_features(
            df=work_df,
            phenotype_col=phenotype_col,
            batch_col=batch_col,
            roi_col=roi_col,
            k_neighbors=int(k_neighbors),
            min_cells_per_region=int(min_cells_per_region),
        )

        if feature_df_h.empty or not freq_cols_h:
            st.warning("No valid rows available for model selection after current filters.")
            return

        helper_df = feature_df_h.dropna(subset=freq_cols_h).copy()
        if len(helper_df) < 20:
            st.warning("Too few valid cells for reliable model selection.")
            return

        x_h = helper_df[freq_cols_h].to_numpy(dtype=float)
        if normalize_features:
            mean_h = np.nanmean(x_h, axis=0)
            std_h = np.nanstd(x_h, axis=0)
            std_h[std_h <= 0] = 1.0
            x_h = (x_h - mean_h) / std_h

        try:
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.metrics import silhouette_score
        except Exception as exc:
            st.error(f"scikit-learn model-selection dependency missing: {exc}")
            return

        n_samples = len(x_h)
        k_hi_eff = min(int(k_max), max(2, n_samples - 1))
        k_lo_eff = max(2, min(int(k_min), k_hi_eff - 1))
        if k_hi_eff <= k_lo_eff:
            st.warning("Not enough cells to evaluate the requested K range.")
            return

        k_values = list(range(int(k_lo_eff), int(k_hi_eff) + 1))
        inertia_vals: list[float] = []
        sil_vals: list[float] = []

        if n_samples > int(silhouette_sample):
            rng = np.random.default_rng(int(random_state))
            sample_idx = rng.choice(n_samples, size=int(silhouette_sample), replace=False)
            x_sil = x_h[sample_idx]
        else:
            x_sil = x_h

        progress = st.progress(0)
        for i, k in enumerate(k_values):
            model_h = MiniBatchKMeans(n_clusters=int(k), random_state=int(random_state), batch_size=1024, n_init=10)
            labels_h = model_h.fit_predict(x_h)
            inertia_vals.append(float(model_h.inertia_))

            try:
                if len(np.unique(labels_h)) > 1 and len(x_sil) > int(k):
                    model_s = MiniBatchKMeans(n_clusters=int(k), random_state=int(random_state), batch_size=1024, n_init=10)
                    labels_s = model_s.fit_predict(x_sil)
                    sil = float(silhouette_score(x_sil, labels_s)) if len(np.unique(labels_s)) > 1 else np.nan
                else:
                    sil = np.nan
            except Exception:
                sil = np.nan
            sil_vals.append(sil)
            progress.progress((i + 1) / max(len(k_values), 1))
        progress.empty()

        model_df = pd.DataFrame({"k": k_values, "inertia": inertia_vals, "silhouette": sil_vals})
        st.dataframe(model_df, width="stretch")

        fig_m, ax_m1 = plt.subplots(1, 1, figsize=(8.5, 5.0))
        ax_m1.plot(model_df["k"], model_df["inertia"], "o-", color="#1f77b4", linewidth=2, markersize=5)
        ax_m1.set_xlabel("Number of neighborhoods (K)")
        ax_m1.set_ylabel("Inertia (Elbow)", color="#1f77b4")
        ax_m1.tick_params(axis="y", labelcolor="#1f77b4")
        ax_m1.grid(True, alpha=0.25, linestyle="--")

        ax_m2 = ax_m1.twinx()
        ax_m2.plot(model_df["k"], model_df["silhouette"], "s-", color="#d62728", linewidth=1.8, markersize=4)
        ax_m2.set_ylabel("Silhouette", color="#d62728")
        ax_m2.tick_params(axis="y", labelcolor="#d62728")

        best_row = model_df.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False).head(1)
        if not best_row.empty:
            best_k = int(best_row.iloc[0]["k"])
            best_s = float(best_row.iloc[0]["silhouette"])
            st.session_state["tab8_nb_suggested_k"] = best_k
            ax_m1.axvline(best_k, color="#2ca02c", linestyle="--", linewidth=1.3)
            ax_m1.set_title(f"Model selection helper | suggested K≈{best_k} (best silhouette={best_s:.3f})")
            st.info(
                f"Suggested K from silhouette: {best_k}. You can set 'n neighborhoods' to this value, then run detection and validate by CN profile biology."
            )
            if st.button("Use suggested K", key="tab8_nb_use_suggested_k"):
                st.session_state["tab8_nb_kmeans_k"] = int(best_k)
                st.rerun()
        else:
            ax_m1.set_title("Model selection helper")

        plt.tight_layout()
        st.pyplot(fig_m)
        plt.close(fig_m)

    if run_k_stability_helper:
        if phenotype_col == "None":
            st.error("Please select a phenotyping column before running k-neighbor stability helper.")
            return

        if int(k_stab_max) < int(k_stab_min):
            st.error("k max must be greater than or equal to k min.")
            return

        k_values = list(range(int(k_stab_min), int(k_stab_max) + 1, int(k_stab_step)))
        if len(k_values) < 2:
            st.error("k-neighbor helper requires at least two tested k values.")
            return

        try:
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.metrics import adjusted_rand_score, silhouette_score
        except Exception as exc:
            st.error(f"scikit-learn k-stability dependency missing: {exc}")
            return

        rng = np.random.default_rng(int(random_state))
        rows = []
        progress = st.progress(0)

        for i, k_test in enumerate(k_values):
            feature_df_k, freq_cols_k, _ = _compute_neighborhood_features(
                df=work_df,
                phenotype_col=phenotype_col,
                batch_col=batch_col,
                roi_col=roi_col,
                k_neighbors=int(k_test),
                min_cells_per_region=int(min_cells_per_region),
            )

            if feature_df_k.empty or not freq_cols_k:
                rows.append({"k_neighbors": int(k_test), "n_cells": 0, "stability_ari": np.nan, "silhouette": np.nan, "inertia": np.nan})
                progress.progress((i + 1) / max(len(k_values), 1))
                continue

            data_k = feature_df_k.dropna(subset=freq_cols_k).copy()
            if len(data_k) < max(20, int(n_neighborhoods) + 2):
                rows.append(
                    {
                        "k_neighbors": int(k_test),
                        "n_cells": int(len(data_k)),
                        "stability_ari": np.nan,
                        "silhouette": np.nan,
                        "inertia": np.nan,
                    }
                )
                progress.progress((i + 1) / max(len(k_values), 1))
                continue

            if len(data_k) > int(ks5):
                sample_idx = rng.choice(len(data_k), size=int(ks5), replace=False)
                data_k = data_k.iloc[sample_idx].copy()

            x_k = data_k[freq_cols_k].to_numpy(dtype=float)
            if normalize_features:
                mean_k = np.nanmean(x_k, axis=0)
                std_k = np.nanstd(x_k, axis=0)
                std_k[std_k <= 0] = 1.0
                x_k = (x_k - mean_k) / std_k

            k_cluster_eff = int(min(int(n_neighborhoods), len(x_k) - 1))
            if k_cluster_eff < 2:
                rows.append(
                    {
                        "k_neighbors": int(k_test),
                        "n_cells": int(len(x_k)),
                        "stability_ari": np.nan,
                        "silhouette": np.nan,
                        "inertia": np.nan,
                    }
                )
                progress.progress((i + 1) / max(len(k_values), 1))
                continue

            labels_runs: list[np.ndarray] = []
            inertia_runs: list[float] = []
            sil_runs: list[float] = []

            for r in range(int(k_stab_repeats)):
                rs = int(random_state) + r
                model_k = MiniBatchKMeans(n_clusters=k_cluster_eff, random_state=rs, batch_size=1024, n_init=10)
                labels_k = model_k.fit_predict(x_k)
                labels_runs.append(labels_k)
                inertia_runs.append(float(model_k.inertia_))
                try:
                    if len(np.unique(labels_k)) > 1:
                        sil_runs.append(float(silhouette_score(x_k, labels_k)))
                except Exception:
                    pass

            ari_vals: list[float] = []
            for a in range(len(labels_runs)):
                for b in range(a + 1, len(labels_runs)):
                    ari_vals.append(float(adjusted_rand_score(labels_runs[a], labels_runs[b])))

            rows.append(
                {
                    "k_neighbors": int(k_test),
                    "n_cells": int(len(x_k)),
                    "stability_ari": float(np.mean(ari_vals)) if ari_vals else np.nan,
                    "silhouette": float(np.mean(sil_runs)) if sil_runs else np.nan,
                    "inertia": float(np.mean(inertia_runs)) if inertia_runs else np.nan,
                }
            )
            progress.progress((i + 1) / max(len(k_values), 1))

        progress.empty()
        kstab_df = pd.DataFrame(rows)
        st.dataframe(kstab_df, width="stretch")

        fig_k, ax_k1 = plt.subplots(1, 1, figsize=(8.5, 5.0))
        ax_k1.plot(kstab_df["k_neighbors"], kstab_df["stability_ari"], "o-", color="#2ca02c", linewidth=2, markersize=5)
        ax_k1.set_xlabel("k neighbors")
        ax_k1.set_ylabel("Stability ARI", color="#2ca02c")
        ax_k1.tick_params(axis="y", labelcolor="#2ca02c")
        ax_k1.grid(True, alpha=0.25, linestyle="--")

        ax_k2 = ax_k1.twinx()
        ax_k2.plot(kstab_df["k_neighbors"], kstab_df["silhouette"], "s-", color="#d62728", linewidth=1.6, markersize=4)
        ax_k2.set_ylabel("Silhouette", color="#d62728")
        ax_k2.tick_params(axis="y", labelcolor="#d62728")

        cand = kstab_df.dropna(subset=["stability_ari"]).copy()
        if not cand.empty:
            top_ari = float(cand["stability_ari"].max())
            cand2 = cand[cand["stability_ari"] >= (top_ari - 0.01)].copy()
            cand2 = cand2.sort_values(["silhouette", "stability_ari"], ascending=False)
            best_k_neighbor = int(cand2.iloc[0]["k_neighbors"])
            best_ari = float(cand2.iloc[0]["stability_ari"])
            best_sil = float(cand2.iloc[0]["silhouette"]) if np.isfinite(cand2.iloc[0]["silhouette"]) else np.nan

            ax_k1.axvline(best_k_neighbor, color="#1f77b4", linestyle="--", linewidth=1.2)
            ax_k1.set_title(f"k-neighbor stability helper | suggested k={best_k_neighbor} (ARI={best_ari:.3f}, silhouette={best_sil:.3f})")
            st.info(
                f"Suggested k neighbors: {best_k_neighbor} (highest stability with good silhouette)."
            )
            if st.button("Use suggested k neighbors", key="tab8_nb_use_suggested_knn"):
                st.session_state["tab8_nb_k"] = int(best_k_neighbor)
                st.rerun()
        else:
            ax_k1.set_title("k-neighbor stability helper")

        plt.tight_layout()
        st.pyplot(fig_k)
        plt.close(fig_k)

    current_signature = _make_nb_signature(
        csv_path=csv_path,
        x_coord_col=x_coord_col,
        y_coord_col=y_coord_col,
        phenotype_col=phenotype_col,
        batch_col=batch_col,
        roi_col=roi_col,
        k_neighbors=int(k_neighbors),
        n_neighborhoods=int(n_neighborhoods),
        random_state=int(random_state),
        normalize_features=bool(normalize_features),
        min_cells_per_region=int(min_cells_per_region),
    )

    if st.button("Run Neighborhood Detection", type="primary", key="tab8_nb_run"):
        if phenotype_col == "None":
            st.error("Please select a phenotyping column.")
            return

        feature_df, freq_cols, phenotypes = _compute_neighborhood_features(
            df=work_df,
            phenotype_col=phenotype_col,
            batch_col=batch_col,
            roi_col=roi_col,
            k_neighbors=int(k_neighbors),
            min_cells_per_region=int(min_cells_per_region),
        )

        if feature_df.empty or not freq_cols:
            st.warning("No valid regions/cells for neighborhood detection after current filters.")
            return

        cluster_df = feature_df.dropna(subset=freq_cols).copy()
        if cluster_df.empty:
            st.warning("No rows with valid neighborhood feature vectors.")
            return

        try:
            from sklearn.cluster import MiniBatchKMeans
        except Exception as exc:
            st.error(f"scikit-learn clustering dependency missing: {exc}")
            return

        x = cluster_df[freq_cols].to_numpy(dtype=float)
        if normalize_features:
            mean = np.nanmean(x, axis=0)
            std = np.nanstd(x, axis=0)
            std[std <= 0] = 1.0
            x = (x - mean) / std

        k_eff = int(min(int(n_neighborhoods), len(cluster_df)))
        if k_eff < 2:
            st.warning("Need at least two valid cells for clustering.")
            return

        model = MiniBatchKMeans(n_clusters=k_eff, random_state=int(random_state), batch_size=1024, n_init=10)
        cluster_df["CN_cluster"] = model.fit_predict(x)

        cn_profiles = cluster_df.groupby("CN_cluster", dropna=False)[freq_cols].mean()
        cn_profiles.columns = [c.replace("freq_", "") for c in cn_profiles.columns]

        profile_summary_df = _build_profile_summary(cn_profiles, cluster_df)
        st.session_state["tab8_nb_result"] = {
            "signature": current_signature,
            "cluster_df": cluster_df,
            "cn_profiles": cn_profiles,
            "profile_summary_df": profile_summary_df,
            "freq_cols": list(freq_cols),
            "phenotype_col": str(phenotype_col),
        }
        st.session_state[f"tab8_nb_editor_{current_signature}"] = _build_editor_table(profile_summary_df)
        st.success("Neighborhood detection complete.")

    result = st.session_state.get("tab8_nb_result")
    result_dict = result if isinstance(result, dict) else None
    _has_result = result_dict is not None
    _cluster_raw = pd.DataFrame()
    _profiles_raw = pd.DataFrame()
    _summary_raw = pd.DataFrame()

    # Guard against stale / serialised session state
    if _has_result and result_dict is not None:
        _cluster_raw = result_dict.get("cluster_df")
        _profiles_raw = result_dict.get("cn_profiles")
        _summary_raw = result_dict.get("profile_summary_df")
        if not isinstance(_cluster_raw, pd.DataFrame) or not isinstance(_profiles_raw, pd.DataFrame) or not isinstance(_summary_raw, pd.DataFrame):
            st.warning("Stale neighborhood results detected \u2014 please re-run Neighborhood Detection.")
            del st.session_state["tab8_nb_result"]
            _has_result = False

    if _has_result and result_dict is not None:
        result_signature = str(result_dict.get("signature", ""))
        if result_signature != current_signature:
            st.info("Showing the last computed neighborhood result. Re-run detection to refresh it with the current settings.")

        cluster_df = _cluster_raw.copy() if isinstance(_cluster_raw, pd.DataFrame) else pd.DataFrame()
        cn_profiles = _profiles_raw.copy() if isinstance(_profiles_raw, pd.DataFrame) else pd.DataFrame()
        profile_summary_df = _summary_raw.copy() if isinstance(_summary_raw, pd.DataFrame) else pd.DataFrame()
        freq_cols = list(result_dict.get("freq_cols") or [])
        result_phenotype_col = str(result_dict.get("phenotype_col", phenotype_col))
        current_export_df: pd.DataFrame | None = None
        current_merged_name_map: dict[str, str] = {}
        current_merged_color_map: dict[str, str] = {}

        editor_state_key = f"tab8_nb_editor_{result_signature}"
        if editor_state_key not in st.session_state:
            st.session_state[editor_state_key] = _build_editor_table(profile_summary_df)

        st.dataframe(profile_summary_df, width="stretch")

        st.markdown("#### Merge and label neighborhoods")
        with st.form(f"tab8_nb_merge_form_{result_signature}"):
            edited_editor_df = st.data_editor(
                st.session_state[editor_state_key],
                hide_index=True,
                width="stretch",
                disabled=["CN_cluster", "n_cells", "top_phenotypes", "original_name"],
                column_config={
                    "CN_cluster": st.column_config.NumberColumn("Original CN", format="%d"),
                    "n_cells": st.column_config.NumberColumn("Cells", format="%d"),
                    "top_phenotypes": st.column_config.TextColumn("Top phenotypes"),
                    "original_name": st.column_config.TextColumn("Original label"),
                    "merged_cluster": st.column_config.TextColumn("Merged group"),
                    "merged_name": st.column_config.TextColumn("Merged label"),
                    "merged_color": st.column_config.TextColumn("Merged color"),
                },
                key=f"tab8_nb_editor_widget_{result_signature}",
            )
            apply_merge_labels = st.form_submit_button("Apply merges and labels")
        if apply_merge_labels:
            st.session_state[editor_state_key] = edited_editor_df.copy()

        editor_df = st.session_state[editor_state_key].copy()
        export_df, merged_profiles, merged_summary_df, merged_name_map, merged_color_map, merge_warnings = _apply_editor_to_results(
            cluster_df=cluster_df,
            freq_cols=freq_cols,
            editor_df=editor_df,
        )
        current_export_df = export_df.copy()
        current_merged_name_map = dict(merged_name_map)
        current_merged_color_map = dict(merged_color_map)
        for warning_text in merge_warnings:
            st.warning(warning_text)

        st.markdown("#### Merged neighborhood summary")
        st.dataframe(merged_summary_df, width="stretch")

        st.markdown("#### CN Composition Heatmap")
        heat = merged_profiles.copy()
        heat.index = [merged_name_map.get(str(idx), str(idx)) for idx in heat.index]
        fig_h, ax_h = plt.subplots(1, 1, figsize=(max(8, 0.8 * heat.shape[1]), max(4, 0.65 * heat.shape[0] + 2)))
        im = ax_h.imshow(heat.values, aspect="auto", cmap="viridis")
        ax_h.set_xticks(np.arange(heat.shape[1]))
        ax_h.set_xticklabels(heat.columns, rotation=45, ha="right", fontsize=8)
        ax_h.set_yticks(np.arange(heat.shape[0]))
        ax_h.set_yticklabels([str(i) for i in heat.index], fontsize=9)
        ax_h.set_xlabel("Phenotype")
        ax_h.set_ylabel("Neighborhood")
        ax_h.set_title("Average phenotype composition per merged neighborhood")
        cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
        cbar.set_label("Frequency")
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

        st.markdown("#### Spatial neighborhood overlay")
        batch_vals = sorted(export_df["__nb_batch_val__"].dropna().astype(str).unique().tolist())
        selected_batch = st.selectbox("Overlay batch", options=batch_vals, key=f"tab8_nb_overlay_batch_{result_signature}")
        roi_vals = sorted(
            export_df[export_df["__nb_batch_val__"].astype(str) == str(selected_batch)]["__nb_roi_val__"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        selected_roi = st.selectbox("Overlay ROI", options=roi_vals, key=f"tab8_nb_overlay_roi_{result_signature}")

        o1, o2, o3 = st.columns(3)
        overlay_point_size = o1.slider("Point size", min_value=2, max_value=80, value=10, step=1, key=f"tab8_nb_overlay_ps_{result_signature}")
        overlay_alpha = o2.slider("Point alpha", min_value=0.1, max_value=1.0, value=0.85, step=0.05, key=f"tab8_nb_overlay_alpha_{result_signature}")
        show_tissue_outline = o3.checkbox("Show virtual tissue outline", value=True, key=f"tab8_nb_show_outline_{result_signature}")

        overlay_df = export_df[
            (export_df["__nb_batch_val__"].astype(str) == str(selected_batch))
            & (export_df["__nb_roi_val__"].astype(str) == str(selected_roi))
        ].copy()

        if overlay_df.empty:
            st.warning("No cells found for selected overlay batch/ROI.")
        else:
            fig_o, ax_o = plt.subplots(1, 1, figsize=(9, 7))
            overlay_color_map, overlay_color_warnings = _build_label_color_map(export_df, merged_color_map)
            for overlay_warning in overlay_color_warnings:
                st.warning(overlay_warning)

            merged_names = sorted(overlay_df["CN_name_merged"].dropna().astype(str).unique().tolist())
            for merged_name in merged_names:
                cdf = overlay_df[overlay_df["CN_name_merged"].astype(str) == str(merged_name)]
                if cdf.empty:
                    continue
                ax_o.scatter(
                    cdf["x_centroid"].to_numpy(dtype=float),
                    cdf["y_centroid"].to_numpy(dtype=float),
                    s=float(overlay_point_size),
                    alpha=float(overlay_alpha),
                    c=overlay_color_map.get(str(merged_name), "#1f77b4"),
                    edgecolors="none",
                    label=str(merged_name),
                )

            if show_tissue_outline and len(overlay_df) >= 5:
                coords_overlay = overlay_df[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
                buffer_px = _compute_buffer_distance(coords_overlay, use_median=False, percentile=99.0)
                if np.isfinite(buffer_px) and float(buffer_px) > 0:
                    tissue_geom = _build_region_geometry(
                        coords=coords_overlay,
                        method="buffer_union",
                        alpha=100.0,
                        buffer_distance=float(buffer_px),
                        min_hole_size=10000.0,
                    )
                    if tissue_geom is not None:
                        if tissue_geom.geom_type == "Polygon":
                            exterior = getattr(tissue_geom, "exterior", None)
                            if exterior is not None:
                                x, y = exterior.xy
                                ax_o.plot(x, y, "k-", linewidth=1.4)
                                for hole in list(getattr(tissue_geom, "interiors", [])):
                                    hx, hy = zip(*hole.coords)
                                    ax_o.plot(hx, hy, "k-", linewidth=0.9, alpha=0.8)
                        elif tissue_geom.geom_type == "MultiPolygon":
                            for poly in list(getattr(tissue_geom, "geoms", [])):
                                x, y = poly.exterior.xy
                                ax_o.plot(x, y, "k-", linewidth=1.2)
                                for hole in list(getattr(poly, "interiors", [])):
                                    hx, hy = zip(*hole.coords)
                                    ax_o.plot(hx, hy, "k-", linewidth=0.9, alpha=0.8)

            ax_o.set_aspect("equal")
            ax_o.invert_yaxis()
            ax_o.set_xlabel("X (pixels)")
            ax_o.set_ylabel("Y (pixels)")
            ax_o.set_title(f"Neighborhood map | batch={selected_batch} | roi={selected_roi}")
            ax_o.grid(True, alpha=0.2)
            handles, labels = ax_o.get_legend_handles_labels()
            if handles:
                uniq = dict(zip(labels, handles))
                ax_o.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best")
            plt.tight_layout()
            st.pyplot(fig_o)
            plt.close(fig_o)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cn_profile_csv = out_dir / "neighborhood_profiles.csv"
        cell_level_csv = out_dir / "neighborhood_cell_level.csv"

        export_df[output_original_cluster_col] = export_df["CN_cluster_original"].astype(int)
        export_df[output_original_name_col] = export_df["CN_name_original"].astype(str)
        export_df[output_cluster_col] = export_df["CN_cluster_merged"].astype(str)
        export_df[output_col_name] = export_df["CN_name_merged"].astype(str)
        merged_profiles_export = merged_profiles.copy()
        merged_profiles_export.index = [merged_name_map.get(str(idx), str(idx)) for idx in merged_profiles_export.index]
        merged_profiles_export.to_csv(cn_profile_csv)
        export_df.to_csv(cell_level_csv, index=False)

        st.write(
            {
                "CN_profiles_csv": str(cn_profile_csv),
                "Cell_level_csv": str(cell_level_csv),
                "Cells_clustered": int(len(export_df)),
                "Original_CN_clusters": int(profile_summary_df["CN_cluster"].nunique()),
                "Merged_neighborhoods": int(export_df["CN_cluster_merged"].astype(str).nunique()),
                "Phenotype_column_used": result_phenotype_col,
            }
        )

        if st.button("Write neighborhood columns to unified CSV", type="primary", key=f"tab8_nb_write_csv_{result_signature}"):
            try:
                full_df = load_unified_csv(str(csv_path)).copy()
            except Exception as exc:
                st.error(f"Could not read active unified CSV for write-back: {exc}")
                return

            for column_name in [output_col_name, output_cluster_col, output_original_cluster_col, output_original_name_col]:
                if column_name not in full_df.columns:
                    full_df[column_name] = np.nan

            write_idx = export_df.index.intersection(full_df.index)
            merged_name_series = export_df.loc[write_idx, [output_col_name]][output_col_name]
            merged_cluster_series = export_df.loc[write_idx, [output_cluster_col]][output_cluster_col]
            original_cluster_series = export_df.loc[write_idx, [output_original_cluster_col]][output_original_cluster_col]
            original_name_series = export_df.loc[write_idx, [output_original_name_col]][output_original_name_col]
            merged_name_values = [str(value) for value in merged_name_series.tolist()]
            merged_cluster_values = [str(value) for value in merged_cluster_series.tolist()]
            original_cluster_values = [int(value) for value in original_cluster_series.tolist()]
            original_name_values = [str(value) for value in original_name_series.tolist()]
            full_df.loc[write_idx, output_col_name] = merged_name_values
            full_df.loc[write_idx, output_cluster_col] = merged_cluster_values
            full_df.loc[write_idx, output_original_cluster_col] = original_cluster_values
            full_df.loc[write_idx, output_original_name_col] = original_name_values

            if save_mode == "Overwrite active CSV":
                out_csv_path = Path(csv_path)
            else:
                src = Path(csv_path)
                out_csv_path = src.with_name(f"{src.stem}{save_suffix}{src.suffix}")

            try:
                full_df.to_csv(out_csv_path, index=False)
                st.session_state["_tab8_nb_last_saved_csv_path"] = str(out_csv_path)
                try:
                    load_unified_csv.clear()
                except Exception:
                    pass
                st.success(f"Neighborhood columns saved to: {out_csv_path}")
                st.caption("Use the reload button in the Entity Signed Distance panel below to refresh Tab 8 from the saved CSV.")
            except Exception as exc:
                st.error(f"Failed to save unified CSV: {exc}")




    # ==================================================================
    # Panel: Entity Signed-Distance (column-agnostic, always available)
    # ==================================================================
    # Gather CN context if neighborhood results exist
    _dist_result = st.session_state.get("tab8_nb_result")
    _dist_export_df = locals().get("current_export_df")
    _dist_merged_name_map: dict = dict(locals().get("current_merged_name_map", {}))
    _dist_merged_color_map: dict = dict(locals().get("current_merged_color_map", {}))
    if _dist_result and isinstance(_dist_result, dict) and _dist_export_df is None:
        _d_export = _dist_result.get("merged_export_df")
        if isinstance(_d_export, pd.DataFrame):
            _dist_export_df = _d_export
        _d_name_map = _dist_result.get("merged_name_map")
        if isinstance(_d_name_map, dict):
            _dist_merged_name_map = dict(_d_name_map)
        _d_color_map = _dist_result.get("merged_color_map")
        if isinstance(_d_color_map, dict):
            _dist_merged_color_map = dict(_d_color_map)

    if result_dict is not None and _dist_export_df is not None:
        result_dict["merged_export_df"] = _dist_export_df
        result_dict["merged_name_map"] = _dist_merged_name_map
        result_dict["merged_color_map"] = _dist_merged_color_map

    _render_cn_distance_panel(
        full_df=df,
        export_df=_dist_export_df,
        merged_name_map=_dist_merged_name_map,
        merged_color_map=_dist_merged_color_map,
        batch_col=batch_col if batch_col else "",
        roi_col=roi_col if roi_col else "",
        csv_path=csv_path,
        save_mode=save_mode,
        save_suffix=save_suffix,
        all_columns=all_columns,
        x_coord_col=x_coord_col,
        y_coord_col=y_coord_col,
    )


# ======================================================================
# Neighborhood signed-distance helper
# ======================================================================

def _compute_cn_signed_distance(
    group_df: pd.DataFrame,
    inside_mask: np.ndarray,
) -> np.ndarray:
    """Return signed distance for every cell in *group_df*.

    * **Negative** for cells inside the selected entity: distance to the
      nearest cell that is outside that entity.
    * **Positive** for cells outside the selected entity: distance to the
      nearest cell that is inside that entity.

    Distances are computed directly between cells, not to a hull or boundary.
    """
    coords = group_df[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    result = np.full(len(group_df), np.nan, dtype=float)
    valid_mask = np.isfinite(coords).all(axis=1)
    inside_mask = np.asarray(inside_mask, dtype=bool)
    valid_inside_mask = valid_mask & inside_mask
    valid_outside_mask = valid_mask & (~inside_mask)

    if not valid_inside_mask.any() or not valid_outside_mask.any():
        return result

    inside_tree = cKDTree(coords[valid_inside_mask])
    outside_tree = cKDTree(coords[valid_outside_mask])

    inside_distances, _ = outside_tree.query(coords[valid_inside_mask], k=1)
    outside_distances, _ = inside_tree.query(coords[valid_outside_mask], k=1)
    result[valid_inside_mask] = -inside_distances
    result[valid_outside_mask] = outside_distances

    return result


def _render_cn_distance_panel(
    *,
    full_df: pd.DataFrame,
    export_df: "pd.DataFrame | None",
    merged_name_map: dict,
    merged_color_map: dict,
    batch_col: str,
    roi_col: str,
    csv_path: str,
    save_mode: str,
    save_suffix: str,
    all_columns: list[str],
    x_coord_col: str = "x_centroid",
    y_coord_col: str = "y_centroid",
):
    """Render the Entity Signed-Distance panel.

    This panel is **column-agnostic**: the user can pick *any* categorical
    column in the loaded CSV (e.g. neighborhood, phenotype, annotation,
    clone, …) and any value(s) within it to define the "inside" region.
    This means it works even if the full neighborhood detection was never
    run — allowing signed-distance computation from any entity column.
    """

    with st.expander("\U0001f4cf Panel \u2014 Entity Signed Distance", expanded=False):
        st.caption(
            "Compute a **signed distance** column from any categorical entity.  "
            "Select a **source column** and one or more **target values** that "
            "define the \u2018inside\u2019 region. Cells matching those values receive "
            "**negative** distance to their nearest non-target cell; cells outside "
            "receive **positive** distance to their nearest target cell. Distances "
            "are computed per group so that coordinates from different images never mix."
        )

        last_saved_csv = str(st.session_state.get("_tab8_nb_last_saved_csv_path", "") or "")
        if last_saved_csv:
            reload_target = Path(last_saved_csv)
            active_target = Path(csv_path)
            if reload_target.exists():
                if reload_target == active_target:
                    reload_help = "Refresh the active Tab 8 CSV from disk so newly written neighborhood columns become selectable below."
                else:
                    reload_help = "Switch Tab 8 to the newly saved CSV and refresh the available entity columns below."
                st.caption(f"Last saved neighborhood CSV: {reload_target}")
                if st.button("Reload saved CSV into Tab 8", key="tab8_nb_reload_saved_csv", help=reload_help):
                    queue_tab8_csv_reload(str(reload_target))
                    st.rerun()
        with st.expander("How signed distance works", expanded=False):
            st.markdown(
                "- Distances are computed **cell-to-cell**, not against a polygon hull.  \n"
                "- A cell in the selected entity gets a **negative** value equal to the distance to the nearest cell **outside** the entity.  \n"
                "- A cell outside the selected entity gets a **positive** value equal to the distance to the nearest cell **inside** the entity.  \n"
                "- Values near **0** lie close to the label boundary between target and non-target cells.  \n"
                "- More negative values indicate deeper penetration into the selected entity.  \n"
                "- Larger positive values indicate cells farther away from the selected entity."
            )

        # ------ Determine available entity columns ------
        # Start with all columns in the original unified CSV
        entity_columns: list[str] = list(all_columns)

        # If neighborhood results are loaded, also offer CN_name_merged
        _has_cn_merged = (
            export_df is not None
            and isinstance(export_df, pd.DataFrame)
            and "CN_name_merged" in export_df.columns
        )
        if _has_cn_merged and "CN_name_merged" not in entity_columns:
            entity_columns = ["CN_name_merged (from neighborhood results)"] + entity_columns

        if not entity_columns:
            st.info("No columns available for distance computation.")
            return

        # ------ Configuration (outside form for dynamic updates) ------
        st.markdown("**Source column & target values**")

        # Key prefix for stable widget keys
        _kp = "tab8_nb_dist2"

        entity_col_raw = st.selectbox(
            "Entity column",
            options=entity_columns,
            index=0,
            key=f"{_kp}_entity_col",
            help="Pick the column that contains the categorical labels you want to "
                 "compute distance from (e.g. 'neighborhood', 'phenotype_leiden', "
                 "'annotation', any column).",
        )

        # Resolve actual column name and source dataframe
        if entity_col_raw == "CN_name_merged (from neighborhood results)":
            _entity_col = "CN_name_merged"
            _src_df = export_df
        else:
            _entity_col = entity_col_raw
            _src_df = full_df

        if _src_df is None or not isinstance(_src_df, pd.DataFrame):
            _src_df = full_df

        # Get unique values for the selected column
        if _entity_col in _src_df.columns:
            _entity_vals = sorted(_src_df[_entity_col].dropna().astype(str).unique().tolist())
        else:
            _entity_vals = []
            st.warning(f"Column '{_entity_col}' not found in data.")

        target_values = st.multiselect(
            "Target value(s) — define inside region",
            options=_entity_vals,
            default=[],
            key=f"{_kp}_target_vals",
            help="Select one or more values. All cells matching ANY of these "
                 "values will be treated as 'inside' the region.",
        )

        # ------ Form for remaining settings (offline) ------
        with st.form(f"{_kp}_form"):
            st.markdown("**Grouping & output**")
            c1, c2 = st.columns(2)

            _batch_opts = [""] + all_columns
            _batch_default = _default_index(_batch_opts, batch_col if batch_col else "batch_id", fallback=0)
            dist_batch_col = c1.selectbox(
                "Batch column",
                options=_batch_opts,
                index=_batch_default,
                key=f"{_kp}_batch",
                format_func=_format_opt,
                help="Primary grouping column. Distance computed independently per group.",
            )

            _roi_opts = ["(none)"] + all_columns
            _roi_default = _default_index(_roi_opts, roi_col if roi_col else "(none)", fallback=0)
            dist_roi_col = c2.selectbox(
                "ROI column (optional)",
                options=_roi_opts,
                index=_roi_default,
                key=f"{_kp}_roi",
                help="Secondary grouping column. Leave as '(none)' to skip.",
            )

            # Default suffix from target values
            _default_suffix = (
                "_".join(str(v).lower().replace(" ", "_") for v in target_values[:3])
                if target_values else "entity"
            )
            col_name = st.text_input(
                "Output column name",
                value=f"neighborhood_distance_{_default_suffix}",
                key=f"{_kp}_col_name",
                help="Full name of the output column that will be added to the CSV.",
            )

            # X/Y coordinate columns (pre-filled from config, editable)
            st.markdown("**Coordinate columns**")
            xc1, xc2 = st.columns(2)
            dist_x_col = xc1.selectbox(
                "X coordinate column",
                options=[""] + all_columns,
                index=_default_index([""] + all_columns, x_coord_col, fallback=0),
                key=f"{_kp}_x_col",
                format_func=_format_opt,
            )
            dist_y_col = xc2.selectbox(
                "Y coordinate column",
                options=[""] + all_columns,
                index=_default_index([""] + all_columns, y_coord_col, fallback=0),
                key=f"{_kp}_y_col",
                format_func=_format_opt,
            )

            submitted = st.form_submit_button("\U0001f4cf Compute Signed Distance")

        # ---- execution (outside form) ----
        dist_result_key = f"{_kp}_result"

        if submitted:
            if not target_values:
                st.error("Please select at least one target value.")
            elif _entity_col not in _src_df.columns:
                st.error(f"Column '{_entity_col}' not found in data.")
            elif dist_x_col not in _src_df.columns or dist_y_col not in _src_df.columns:
                st.error(f"Coordinate columns '{dist_x_col}' / '{dist_y_col}' not found.")
            else:
                out_col = col_name.strip() or f"neighborhood_distance_{_default_suffix}"

                # Build working df with standardised x/y columns
                work = _src_df.copy()
                work["x_centroid"] = pd.to_numeric(work[dist_x_col], errors="coerce")
                work["y_centroid"] = pd.to_numeric(work[dist_y_col], errors="coerce")

                group_cols: list[str] = []
                if dist_batch_col and dist_batch_col in work.columns:
                    group_cols.append(dist_batch_col)
                if dist_roi_col and dist_roi_col != "(none)" and dist_roi_col in work.columns:
                    group_cols.append(dist_roi_col)

                work[out_col] = np.nan

                # Build inside mask from target values
                target_set = set(str(v) for v in target_values)
                target_mask_full = work[_entity_col].astype(str).isin(target_set)
                n_inside = int(target_mask_full.sum())
                n_outside = int((~target_mask_full).sum())

                if n_inside == 0:
                    st.warning(f"No cells found matching target values: {target_values}")
                else:
                    progress = st.progress(0, text="Computing signed distances\u2026")

                    if not group_cols:
                        inside = target_mask_full.to_numpy()
                        dists = _compute_cn_signed_distance(work, inside)
                        work[out_col] = dists
                    else:
                        grouped = work.groupby(group_cols, dropna=False)
                        n_groups = len(grouped)
                        for gi, (gname, gdf) in enumerate(grouped):
                            progress.progress(
                                (gi + 1) / max(n_groups, 1),
                                text=f"Group {gi + 1}/{n_groups}",
                            )
                            local_mask = target_mask_full.loc[gdf.index].to_numpy()
                            if local_mask.sum() == 0:
                                continue
                            dists = _compute_cn_signed_distance(gdf, local_mask)
                            work.loc[gdf.index, out_col] = dists

                    progress.empty()

                    # Build distribution figure
                    inside_vals = work.loc[target_mask_full, out_col].dropna()
                    outside_vals = work.loc[~target_mask_full, out_col].dropna()

                    fig_d, ax_d = plt.subplots(1, 1, figsize=(9, 4))
                    all_vals = work[out_col].dropna()
                    if len(all_vals) > 0:
                        bins = np.linspace(
                            float(all_vals.min()) - 1,
                            float(all_vals.max()) + 1,
                            80,
                        ).tolist()
                    else:
                        bins = 80

                    label_str = " | ".join(target_values[:3])
                    if len(target_values) > 3:
                        label_str += f" +{len(target_values)-3} more"

                    if len(inside_vals):
                        ax_d.hist(
                            inside_vals, bins=bins, alpha=0.6,
                            color="#1f77b4",
                            label=f"Inside ({label_str}, n={len(inside_vals)})",
                        )
                    if len(outside_vals):
                        ax_d.hist(
                            outside_vals, bins=bins, alpha=0.4,
                            color="#999999",
                            label=f"Outside (n={len(outside_vals)})",
                        )
                    ax_d.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Boundary (0)")
                    ax_d.set_xlabel("Signed distance (pixels)")
                    ax_d.set_ylabel("Cell count")
                    grp_txt = " \u00d7 ".join(group_cols) if group_cols else "all cells"
                    ax_d.set_title(
                        f"Entity distance \u2014 {label_str} | col={_entity_col} | grouped by {grp_txt}"
                    )
                    ax_d.legend(fontsize=8)
                    ax_d.grid(True, alpha=0.2)
                    plt.tight_layout()

                    buf = io.BytesIO()
                    fig_d.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig_d)
                    buf.seek(0)
                    fig_bytes = buf.getvalue()

                    st.session_state[dist_result_key] = {
                        "work": work,
                        "out_col": out_col,
                        "fig_bytes": fig_bytes,
                        "label": label_str,
                        "entity_col": _entity_col,
                        "n_inside": n_inside,
                        "n_outside": n_outside,
                        "group_cols": group_cols,
                    }
                    st.success(
                        f"Column **{out_col}** computed using nearest-cell signed distances: "
                        f"{n_inside} inside cells (negative), {n_outside} outside cells (positive)."
                    )

        # ---- display persisted result ----
        dist_res = st.session_state.get(dist_result_key)
        if dist_res and isinstance(dist_res, dict):
            fig_b = dist_res.get("fig_bytes")
            if isinstance(fig_b, bytes):
                st.image(io.BytesIO(fig_b), width="stretch")

                fname_safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", dist_res.get("out_col", "distance"))
                st.download_button(
                    "\u2b07\ufe0f Download distance histogram",
                    data=fig_b,
                    file_name=f"{fname_safe}.png",
                    mime="image/png",
                    key=f"{_kp}_dl",
                )

            # ---- Summary stats ----
            dist_work = dist_res.get("work")
            out_col_name_d = str(dist_res.get("out_col", ""))
            if isinstance(dist_work, pd.DataFrame) and out_col_name_d in dist_work.columns:
                with st.expander("Distance summary statistics", expanded=False):
                    _dvals = dist_work[out_col_name_d].dropna()
                    if len(_dvals) > 0:
                        st.write({
                            "mean": f"{_dvals.mean():.2f}",
                            "median": f"{_dvals.median():.2f}",
                            "std": f"{_dvals.std():.2f}",
                            "min": f"{_dvals.min():.2f}",
                            "max": f"{_dvals.max():.2f}",
                            "n_inside (negative)": int(dist_res.get("n_inside", 0)),
                            "n_outside (positive)": int(dist_res.get("n_outside", 0)),
                        })

            # ---- write-back button ----
            if st.button(
                "Write distance column to unified CSV",
                type="primary",
                key=f"{_kp}_write",
            ):
                if not isinstance(dist_work, pd.DataFrame):
                    st.error("Stale distance results \u2014 please re-compute.")
                else:
                    try:
                        full_csv_df = load_unified_csv(str(csv_path)).copy()
                    except Exception as exc:
                        st.error(f"Could not read unified CSV: {exc}")
                        return

                    if out_col_name_d not in full_csv_df.columns:
                        full_csv_df[out_col_name_d] = np.nan

                    common_idx = dist_work.index.intersection(full_csv_df.index)
                    if len(common_idx) == 0:
                        st.warning("No matching row indices between computed results and CSV. "
                                   "The distance was computed on a subset that may not align.")
                    else:
                        dist_series = dist_work.loc[common_idx, [out_col_name_d]][out_col_name_d]
                        full_csv_df.loc[common_idx, out_col_name_d] = dist_series.to_numpy()

                    out_csv = Path(csv_path)

                    try:
                        full_csv_df.to_csv(out_csv, index=False)
                        st.success(f"Column **{out_col_name_d}** written to unified CSV: {out_csv}")
                    except Exception as exc:
                        st.error(f"Failed to write CSV: {exc}")

