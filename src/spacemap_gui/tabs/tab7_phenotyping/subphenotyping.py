"""
Sub-Phenotyping Subtab

Splits one parent phenotype into functional states, in four steps:

0. **Marker panel** - pick one of the shared named panels, the same picker
   Leiden, SOM and Vortex use. Only those markers are surveyed.
1. **Discovery** - measure which of them actually carry structure inside that
   phenotype, instead of clustering on whatever was ticked.
2. **Subcluster** - Leiden, SOM or Vortex on the surviving markers, GPU-accelerated.
3. **Validate** - stability, batch reproducibility and marker separation per subcluster.
4. **Assign** - the shared heatmap / click-to-name / save machinery used by the
   other phenotyping tabs.

Each parent phenotype is processed independently. Clustering several parents in
one embedding would let the (already resolved) between-phenotype differences
dominate, which is the opposite of what sub-phenotyping is for.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import (
    Timer,
    build_leiden_heatmap_figure,
    build_marker_source_note,
    get_all_phenotype_options,
    get_marker_workspace_panel,
    logger,
    render_add_custom_phenotype_ui,
    render_export_split_data,
    render_phenotype_statistics,
    render_method_marker_selection,
    render_publication_heatmap,
    render_save_assignments,
    render_settings_export_import,
    render_method_intro,
    begin_phenotyping_run_audit,
    finish_phenotyping_run_audit,
)
from .marker_composition import render_marker_composition_panel

try:
    from spacemap.ph_subpheno_utils import (
        CLUSTERING_METHODS,
        build_feature_matrix,
        cluster_stability,
        gpu_status,
        marker_reference_stats,
        recommend_markers,
        score_markers,
        subcluster,
        validate_subclusters,
    )

    SUBPHENO_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - surfaced in the UI
    SUBPHENO_IMPORT_ERROR = exc
    CLUSTERING_METHODS = ("Leiden", "SOM", "Vortex")
    build_feature_matrix = cluster_stability = gpu_status = None
    marker_reference_stats = recommend_markers = score_markers = None
    subcluster = validate_subclusters = None


# Discovery reads one column set once; scoring 40k cells is instant, and the
# ranking is stable well below that, so there is nothing to gain from more.
DISCOVERY_SAMPLE = 40000

METHOD_HELP = {
    "Leiden": "Community detection on a shared-neighbour graph. The general-purpose choice.",
    "SOM": "Self-organising map, then merge nodes. Best when you want an even granularity.",
    "Vortex": "Density-gradient (X-shift). Finds tight, well-separated states; GPU-accelerated.",
}


def _read_columns(path: str, columns: list[str]) -> pd.DataFrame | None:
    """Read a subset of columns as fast as the installed libraries allow."""
    try:
        import pyarrow.csv as pv

        table = pv.read_csv(
            path,
            read_options=pv.ReadOptions(use_threads=True),
            convert_options=pv.ConvertOptions(include_columns=list(columns), strings_can_be_null=True),
        )
        return table.to_pandas(split_blocks=True, self_destruct=True)
    except Exception:
        pass
    try:
        return pd.read_csv(path, usecols=list(columns), low_memory=False)
    except Exception:
        return None


def _file_signature(path: str):
    try:
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_size
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _columns_cached(path: str, signature) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _values_cached(path: str, column: str, signature) -> list[str]:
    frame = _read_columns(path, [column])
    if frame is None or column not in frame:
        return []
    return sorted(frame[column].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False, max_entries=2)
def _load_working_frame(path: str, columns: tuple[str, ...], signature) -> pd.DataFrame | None:
    frame = _read_columns(path, list(columns))
    if frame is None:
        return None
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    frame[numeric] = frame[numeric].astype("float32")
    return frame


@st.cache_data(show_spinner=False, max_entries=2)
def _reference_cached(path: str, marker_cols: tuple[str, ...], parent_col: str, signature):
    """Dataset-wide marker statistics; the yardstick every phenotype is scored against."""
    frame = _load_working_frame(path, tuple(list(marker_cols) + [parent_col]), signature)
    if frame is None:
        return None
    return marker_reference_stats(frame, list(marker_cols), parent_col)


def _marker_candidates(columns: list[str]) -> list[str]:
    metadata = {
        "cell_id", "batch_id", "slide_id", "tissue", "x_centroid", "y_centroid",
        "area", "perimeter", "eccentricity", "solidity", "extent",
        "major_axis", "minor_axis", "orientation", "label", "index",
    }
    out = []
    for col in columns:
        low = str(col).lower()
        if low in metadata or low.endswith("_roi") or low.endswith("_roi_size"):
            continue
        if str(col).startswith(("umap_", "leiden_", "phenotype_", "sub_phenotype")):
            continue
        if str(col).endswith("_norm"):
            continue
        out.append(col)
    return out


def _cluster_key(value) -> str:
    return str(value).strip()


def _safe_label(value) -> str:
    text = str(value).strip().replace(" ", "_").replace("/", "-")
    return text or "NA"


def _state(name: str, default):
    if name not in st.session_state:
        st.session_state[name] = default
    return st.session_state[name]


SUBPHENO_GUIDE = """\
**What it does.** Splits one phenotype into functional states, for example activated and resting T cells. The markers used are chosen from the data for each phenotype rather than assumed, and each subcluster is checked for reproducibility before you name it.

**Steps**
1. **Functional Marker Discovery** finds the markers that vary most within the chosen phenotype.
2. Choose a clustering method and its settings.
3. Run.
4. **Validate and Assign**: check how stable each subcluster is, name the subclusters, and export.
"""


def render_subphenotyping(project_root, global_input_path):
    render_method_intro(
        project_root, "subpheno", "Sub-Phenotyping",
        "Split one phenotype into functional states and check each state is reproducible.",
        SUBPHENO_GUIDE, notebook="Step10_sub_phenotypes.ipynb", log_label="Sub-Phenotyping",
    )

    if score_markers is None:
        st.error("`spacemap.ph_subpheno_utils` could not be imported.")
        if SUBPHENO_IMPORT_ERROR is not None:
            st.code(f"{type(SUBPHENO_IMPORT_ERROR).__name__}: {SUBPHENO_IMPORT_ERROR}")
        return

    st.markdown(
        "Split one phenotype into functional states. Markers are chosen from the data "
        "for each phenotype rather than assumed, and every subcluster is checked for "
        "reproducibility before you name it."
    )

    has_gpu, gpu_message = gpu_status()
    (st.success if has_gpu else st.info)(gpu_message)

    cache_dir = Path(project_root) / "outputs/phenotyping/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "subphenotyping_params.json"
    cached = {}
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
        except Exception:
            cached = {}

    render_settings_export_import(
        "Sub-Phenotyping", ["subpheno_"], {"subphenotyping_params.json": cache_file}
    )

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    st.markdown("##### Input")
    use_global = st.checkbox(
        "Use Global Input File", value=bool(cached.get("use_global", True)), key="subpheno_use_global"
    )
    full_input_path = global_input_path
    input_dir = cached.get("input_dir", str(Path(project_root) / "outputs/batch_corrected"))
    selected_file = cached.get("input_file", "")

    if not use_global:
        input_dir = st.text_input("Input Directory", value=input_dir, key="subpheno_input_dir")
        files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv")) if os.path.isdir(input_dir) else []
        if files:
            idx = files.index(selected_file) if selected_file in files else 0
            selected_file = st.selectbox("Input File", files, index=idx, key="subpheno_input_file")
            full_input_path = os.path.join(input_dir, selected_file)
        else:
            st.warning(f"No CSV files found in: {input_dir}")
            full_input_path = None

    if not full_input_path or not os.path.exists(full_input_path):
        st.warning("Select a valid input CSV to continue.")
        return
    st.caption(f"Active file: `{os.path.basename(full_input_path)}`")

    signature = _file_signature(full_input_path)
    columns = _columns_cached(full_input_path, signature)
    available_marker_columns = _marker_candidates(columns)
    shared_panel, workspace_note = get_marker_workspace_panel(available_marker_columns, full_input_path)

    # ------------------------------------------------------------------
    # Marker panel - the same shared-panel picker Leiden, SOM and Vortex use.
    # Whatever is confirmed here is exactly what discovery surveys; nothing is
    # scanned by default.
    # ------------------------------------------------------------------
    st.divider()
    _, marker_cols, workspace_note = render_method_marker_selection(
        key_prefix="subpheno",
        method_name="Sub-Phenotyping",
        full_input_path=full_input_path,
        available_columns=available_marker_columns,
        shared_panel=shared_panel,
        marker_workspace_note=workspace_note,
        fallback_markers=shared_panel,
        saved_markers=cached.get("markers", []),
    )
    marker_cols = [m for m in marker_cols if m in columns]
    if not marker_cols:
        st.warning(
            "No markers are selected. Pick a shared panel above (they are created in the "
            f"common marker area) and click Save Marker Settings. {workspace_note}"
        )
        return

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    st.markdown("##### Target")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        parent_options = [c for c in columns if c.startswith("phenotype_")] or columns
        saved_parent = cached.get("parent_col", "phenotype_ml_consensus")
        parent_col = st.selectbox(
            "Parent Phenotype Column",
            parent_options,
            index=parent_options.index(saved_parent) if saved_parent in parent_options else 0,
            key="subpheno_parent_col",
        )
    with col_b:
        batch_options = [""] + columns
        saved_batch = cached.get("batch_col", "batch_id")
        batch_col = st.selectbox(
            "Batch Column",
            batch_options,
            index=batch_options.index(saved_batch) if saved_batch in batch_options else 0,
            key="subpheno_batch_col",
            help="Used to separate real biology from batch effects, and to test reproducibility.",
        )
    with col_c:
        saved_cell_id = cached.get("cell_id_col", "cell_id")
        cell_id_col = st.selectbox(
            "Cell ID Column",
            columns,
            index=columns.index(saved_cell_id) if saved_cell_id in columns else 0,
            key="subpheno_cell_id_col",
        )

    parent_values = _values_cached(full_input_path, parent_col, signature) if parent_col else []
    selected_parents = st.multiselect(
        "Phenotypes to Subcluster",
        parent_values,
        default=[v for v in cached.get("parent_values", []) if v in parent_values],
        key="subpheno_parent_values",
        help="Each phenotype is discovered, clustered and validated on its own.",
    )
    if not selected_parents:
        st.info("Pick at least one phenotype to continue.")
        return

    discovery = _state("subpheno_discovery", {})
    chosen_markers = _state("subpheno_markers", {})
    results = _state("subpheno_results", {})
    validation = _state("subpheno_validation", {})

    # Discovery scores are only meaningful for the panel they were computed on:
    # sd_ratio and the batch share are relative to the surveyed set. Changing the
    # panel therefore invalidates any table already on screen.
    marker_signature = tuple(sorted(marker_cols))
    if st.session_state.get("subpheno_marker_signature") != marker_signature:
        st.session_state["subpheno_marker_signature"] = marker_signature
        if discovery:
            discovery.clear()
            chosen_markers.clear()
            st.info("Marker panel changed - previous discovery results were cleared.")

    # ------------------------------------------------------------------
    # Step 1 - discovery
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### 1. Functional Marker Discovery")
    st.caption(
        "Measures, for each marker, how much cells of this phenotype really differ from each "
        "other once batch effects are removed. Markers that are flat here would only contribute "
        "amplified noise to the clustering."
    )

    focus = st.selectbox("Inspect phenotype", selected_parents, key="subpheno_focus")
    disc_col1, disc_col2 = st.columns([1, 3])
    with disc_col1:
        run_discovery = st.button("Run Discovery", type="primary", key="subpheno_run_discovery")
        run_all = st.button("Run for All Selected", key="subpheno_run_discovery_all")
    with disc_col2:
        st.caption(
            f"Surveying **{len(marker_cols)}** markers from panel "
            f"`{st.session_state.get('subpheno_marker_panel_name', '?')}`. "
            f"Scored on up to {DISCOVERY_SAMPLE:,} cells per phenotype. "
            "Verdicts: **enriched** varies more here than anywhere else · **graded** usable spread · "
            "**collapsed** flat here."
        )

    with st.expander("Discovery thresholds", expanded=False):
        st.caption(
            "Defaults suit a well batch-corrected panel. Loosen the batch limit if a marker you "
            "care about is being dropped for a batch share you consider acceptable."
        )
        th1, th2, th3 = st.columns(3)
        with th1:
            collapsed_below = float(st.slider(
                "Flat below", 0.1, 0.9, float(cached.get("collapsed_below", 0.40)), 0.05,
                key="subpheno_collapsed",
                help="Spread inside the phenotype, as a fraction of its spread across all cells.",
            ))
        with th2:
            max_batch = float(st.slider(
                "Max batch share", 0.05, 0.9, float(cached.get("max_batch_eta2", 0.25)), 0.05,
                key="subpheno_max_batch",
                help="Drop a marker when more than this fraction of its spread comes from batch.",
            ))
        with th3:
            max_markers = int(st.number_input(
                "Marker cap (0 = no cap)", 0, 60, int(cached.get("max_markers", 0)),
                key="subpheno_max_markers",
            ))

    if run_discovery or run_all:
        targets = selected_parents if run_all else [focus]
        needed = tuple(marker_cols + [c for c in (parent_col, batch_col) if c])
        with st.spinner("Reading markers..."):
            frame = _load_working_frame(full_input_path, needed, signature)
            reference = _reference_cached(full_input_path, tuple(marker_cols), parent_col, signature)
        if frame is None or reference is None:
            st.error("Could not read the marker columns from the input file.")
        else:
            for target in targets:
                subset = frame[frame[parent_col].astype(str) == str(target)]
                if len(subset) < 50:
                    st.warning(f"{target}: only {len(subset)} cells - too few to score.")
                    continue
                if len(subset) > DISCOVERY_SAMPLE:
                    subset = subset.sample(DISCOVERY_SAMPLE, random_state=42)
                scores = score_markers(
                    subset, marker_cols, reference=reference,
                    batch_col=batch_col or None, phenotype=str(target),
                )
                table = recommend_markers(
                    scores,
                    collapsed_below=collapsed_below,
                    max_batch_eta2=max_batch,
                    max_markers=max_markers or None,
                )
                discovery[str(target)] = table
                chosen_markers[str(target)] = table.loc[table.recommended, "marker"].tolist()
            st.rerun()

    table = discovery.get(str(focus))
    if table is None:
        st.info(f"Run discovery to see which markers carry structure inside **{focus}**.")
    else:
        keep = int(table.recommended.sum())
        st.success(f"**{focus}** - {keep} of {len(table)} markers carry usable structure.")

        editor = table[[
            "marker", "sd_ratio", "batch_eta2", "bimodality", "lineage_spec",
            "verdict", "flag", "score", "recommended",
        ]].copy()
        editor.insert(0, "use", editor.marker.isin(chosen_markers.get(str(focus), [])))
        edited = st.data_editor(
            editor,
            column_config={
                "use": st.column_config.CheckboxColumn("Use", help="Include this marker in the clustering"),
                "marker": st.column_config.TextColumn("Marker", disabled=True),
                "sd_ratio": st.column_config.NumberColumn(
                    "Spread vs. all cells", format="%.2f", disabled=True,
                    help=">= 1 varies more inside this phenotype than anywhere else; < 0.4 is flat here.",
                ),
                "batch_eta2": st.column_config.NumberColumn(
                    "Batch share", format="%.2f", disabled=True,
                    help="Fraction of the spread explained by batch rather than biology.",
                ),
                "bimodality": st.column_config.NumberColumn("Bimodality", format="%.2f", disabled=True),
                "lineage_spec": st.column_config.NumberColumn(
                    "Lineage specificity", format="%.2f", disabled=True,
                    help="High = restricted to a few cell types, so it could be spillover from a neighbour.",
                ),
                "verdict": st.column_config.TextColumn("Verdict", disabled=True),
                "flag": st.column_config.TextColumn("Why dropped", disabled=True, width="medium"),
                "score": st.column_config.NumberColumn("Usable spread", format="%.3f", disabled=True),
                "recommended": st.column_config.CheckboxColumn("Suggested", disabled=True),
            },
            hide_index=True,
            num_rows="fixed",
            key=f"subpheno_marker_editor_{_safe_label(focus)}",
            width="stretch",
        )
        chosen_markers[str(focus)] = edited.loc[edited.use, "marker"].tolist()

        pick1, pick2 = st.columns(2)
        if pick1.button("Reset to Suggested", key="subpheno_reset_markers"):
            chosen_markers[str(focus)] = table.loc[table.recommended, "marker"].tolist()
            st.rerun()
        if pick2.button("Select All Markers", key="subpheno_all_markers"):
            chosen_markers[str(focus)] = table.marker.tolist()
            st.rerun()

        current = chosen_markers.get(str(focus), [])
        st.caption(f"Using **{len(current)}** markers: {', '.join(m for m in current) or 'none'}")

    # ------------------------------------------------------------------
    # Step 1b - the same question asked visually
    # ------------------------------------------------------------------
    # Discovery ranks markers by how much structure they carry; this panel shows
    # which phenotypes the brightest cells of each marker belong to. The two
    # write to the same marker set, so either can drive the clustering.
    st.divider()
    st.markdown("### 1b. High-Expression Composition (visual alternative)")
    render_marker_composition_panel(
        project_root=project_root,
        phenotype_col=parent_col,
        marker_cols=marker_cols,
        all_columns=columns,
        selected_parents=selected_parents,
        chosen_markers=chosen_markers,
        load_frame=lambda needed_columns: _load_working_frame(full_input_path, needed_columns, signature),
        column_values=lambda column: _values_cached(full_input_path, column, signature),
        editor_key_fn=lambda phenotype: f"subpheno_marker_editor_{_safe_label(phenotype)}",
    )

    # ------------------------------------------------------------------
    # Step 2 - method
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### 2. Clustering Method")
    method = st.radio(
        "Method", list(CLUSTERING_METHODS), horizontal=True, key="subpheno_method",
        index=list(CLUSTERING_METHODS).index(cached.get("method", "Leiden"))
        if cached.get("method") in CLUSTERING_METHODS else 0,
    )
    st.caption(METHOD_HELP[method])

    params: dict = {}
    par1, par2, par3, par4 = st.columns(4)
    if method == "Leiden":
        with par1:
            params["resolution"] = float(st.slider("Resolution", 0.1, 3.0, float(cached.get("resolution", 1.0)), 0.1, key="subpheno_resolution"))
        with par2:
            params["n_neighbors"] = int(st.number_input("Neighbours", 5, 100, int(cached.get("n_neighbors", 30)), key="subpheno_n_neighbors"))
        with par3:
            params["n_pcs"] = int(st.number_input("Components", 2, 50, int(cached.get("n_pcs", 20)), key="subpheno_n_pcs"))
    elif method == "SOM":
        with par1:
            params["grid"] = int(st.number_input("Grid size", 4, 30, int(cached.get("grid", 10)), key="subpheno_grid", help="A grid x grid map; more nodes = finer detail before merging."))
        with par2:
            params["iterations"] = int(st.number_input("Iterations", 1000, 200000, int(cached.get("iterations", 20000)), 1000, key="subpheno_iterations"))
    else:
        with par1:
            params["k_neighbors"] = int(st.number_input("Neighbours (k)", 5, 200, int(cached.get("k_neighbors", 35)), key="subpheno_k_neighbors"))
        with par2:
            params["min_cluster_size"] = int(st.number_input("Min cluster size", 10, 5000, int(cached.get("min_cluster_size", 50)), 10, key="subpheno_min_cluster"))

    with par4:
        params["n_meta_clusters"] = int(st.number_input(
            "Merge to", 0, 60, int(cached.get("n_meta_clusters", 10)), key="subpheno_meta",
            help="Over-cluster, then merge look-alike clusters down to this many. 0 keeps every cluster found.",
        ))

    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        use_gpu = st.checkbox("Use GPU when available", value=bool(cached.get("use_gpu", True)), key="subpheno_use_gpu", disabled=not has_gpu)
    with adv2:
        subsample = int(st.number_input("Sketch above", 10000, 2000000, int(cached.get("subsample_size", 150000)), 10000, key="subpheno_subsample",
                                        help="Cluster a representative sketch above this many cells, then transfer labels to the rest."))
    with adv3:
        seed = int(st.number_input("Random seed", 0, 10000, int(cached.get("seed", 42)), key="subpheno_seed"))

    check_stability = st.checkbox(
        "Check stability (slower)", value=bool(cached.get("check_stability", True)), key="subpheno_stability",
        help="Re-runs the clustering on overlapping subsamples and reports how much the partition agrees with itself.",
    )

    cluster_col_name = st.text_input("Subcluster Column", value=cached.get("cluster_col", "subphenotype_cluster"), key="subpheno_cluster_col")
    output_col_name = st.text_input("Sub-Phenotype Column", value=cached.get("output_col", "sub_phenotype"), key="subpheno_output_col")

    try:
        cache_file.write_text(json.dumps({
            "use_global": use_global, "input_dir": input_dir, "input_file": selected_file,
            "parent_col": parent_col, "batch_col": batch_col, "cell_id_col": cell_id_col,
            "parent_values": selected_parents, "method": method, "use_gpu": use_gpu,
            "subsample_size": subsample, "seed": seed, "check_stability": check_stability,
            "cluster_col": cluster_col_name, "output_col": output_col_name,
            "collapsed_below": collapsed_below, "max_batch_eta2": max_batch,
            "max_markers": max_markers,
            "markers": marker_cols,
            "marker_panel_name": st.session_state.get("subpheno_marker_panel_name", ""),
            **params,
        }, indent=2))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Step 3 - run
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### 3. Run")
    ready = [p for p in selected_parents if chosen_markers.get(str(p))]
    missing = [p for p in selected_parents if not chosen_markers.get(str(p))]
    if missing:
        st.warning("Run discovery for: " + ", ".join(missing))

    if st.button("Run Sub-Phenotyping", type="primary", disabled=not ready, key="subpheno_run"):
        needed = tuple(set(marker_cols + [c for c in (parent_col, batch_col, cell_id_col) if c]))
        audit = begin_phenotyping_run_audit(Path(project_root), "subpheno", input_csv=full_input_path)
        status = st.status("Running sub-phenotyping...", expanded=True)
        temp_dir = Path(project_root) / "outputs/phenotyping/temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        produced = {}
        try:
            with Timer("Sub-Phenotyping"):
                frame = _load_working_frame(full_input_path, needed, signature)
                if frame is None:
                    raise ValueError("Could not read the input columns.")

                for parent in ready:
                    markers = chosen_markers[str(parent)]
                    subset = frame[frame[parent_col].astype(str) == str(parent)].copy()
                    status.write(f"**{parent}** - {len(subset):,} cells, {len(markers)} markers")
                    if len(subset) < 100:
                        status.write(f"  skipped: too few cells")
                        continue

                    matrix = build_feature_matrix(subset, markers, batch_col=batch_col or None)
                    labels = subcluster(
                        matrix, method, params, use_gpu=use_gpu if has_gpu else False,
                        seed=seed, subsample_size=subsample, logger=logger,
                    )
                    # Prefixing keeps subcluster names unique across parents while
                    # each parent keeps its own independent clustering.
                    subset[cluster_col_name] = [f"{_safe_label(parent)}_{c}" for c in labels]

                    norm_cols = []
                    for idx, marker in enumerate(markers):
                        col = f"{marker}_norm"
                        subset[col] = matrix[:, idx]
                        norm_cols.append(col)

                    report = validate_subclusters(
                        labels, matrix, markers,
                        batch_labels=subset[batch_col].values if batch_col else None,
                    )
                    report["cluster"] = [f"{_safe_label(parent)}_{c}" for c in report.cluster]
                    stability = float("nan")
                    if check_stability:
                        status.write("  checking stability...")
                        stability = cluster_stability(
                            matrix, method, params, n_repeats=3, seed=seed,
                            use_gpu=use_gpu if has_gpu else False, logger=logger,
                        )

                    path = temp_dir / f"subpheno_{_safe_label(parent)}_{int(time.time())}.parquet"
                    subset.to_parquet(path)
                    produced[str(parent)] = {
                        "path": str(path), "norm_cols": norm_cols, "markers": markers,
                        "n_clusters": int(len(np.unique(labels))), "n_cells": int(len(subset)),
                    }
                    validation[str(parent)] = {"report": report, "stability": stability}
                    status.write(f"  {len(np.unique(labels))} subclusters" + ("" if np.isnan(stability) else f", stability {stability:.2f}"))

            results.update(produced)
            st.session_state["subpheno_results"] = results
            st.session_state["subpheno_saved_cluster_col"] = cluster_col_name
            st.session_state["subpheno_saved_output_col"] = output_col_name
            st.session_state["subpheno_saved_cell_id"] = cell_id_col
            st.session_state.pop("subpheno_phenotype_map", None)
            status.update(label=f"Complete - {len(produced)} phenotype(s)", state="complete")
            finish_phenotyping_run_audit(
                audit, status="success", output_column=output_col_name, cluster_column=cluster_col_name,
                extra={"method": method, "parents": list(produced), "gpu": bool(has_gpu and use_gpu)},
            )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            status.update(label="Failed", state="error")
            st.error(f"Sub-phenotyping failed: {exc}")
            logger.exception("Sub-phenotyping failed")
            finish_phenotyping_run_audit(audit, status="failed", error=str(exc))

    if not results:
        return

    # ------------------------------------------------------------------
    # Step 4 - validate and assign
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### 4. Validate and Assign")
    view = st.selectbox("Phenotype", sorted(results), key="subpheno_view")
    info = results[view]
    if not Path(info["path"]).exists():
        st.warning("Result file is gone - re-run this phenotype.")
        return

    df_results = pd.read_parquet(info["path"])
    norm_cols = info["norm_cols"]
    cluster_col = st.session_state.get("subpheno_saved_cluster_col", cluster_col_name)
    output_col = st.session_state.get("subpheno_saved_output_col", output_col_name)

    checks = validation.get(view, {})
    report = checks.get("report")
    stability = checks.get("stability", float("nan"))

    tabs = st.tabs(["Validation", "Heatmap & Assignment", "Table Editor", "Export"])

    with tabs[0]:
        if not np.isnan(stability):
            if stability >= 0.7:
                st.success(f"Stability {stability:.2f} - the partition reproduces across re-runs.")
            elif stability >= 0.5:
                st.warning(f"Stability {stability:.2f} - only partly reproducible; treat fine splits with care.")
            else:
                st.error(f"Stability {stability:.2f} - the partition is not reproducible. Merge to fewer clusters or use more markers.")
        if report is not None and not report.empty:
            keep = int((report.verdict == "keep").sum())
            st.caption(
                f"{keep} of {len(report)} subclusters pass all checks. `review` means too few cells, "
                "confined to too few batches, or no marker separating it from the rest."
            )
            st.dataframe(report, width="stretch", hide_index=True)

    with tabs[1]:
        cfg = _state("subpheno_heatmap_cfg", {"cluster_x": True, "cluster_y": True, "show_counts": True, "z_min": -2.0, "z_max": 2.0})
        b1, b2, b3 = st.columns(3)
        if b1.button("Toggle X clustering", key="subpheno_toggle_x"):
            cfg["cluster_x"] = not cfg["cluster_x"]
        if b2.button("Toggle Y clustering", key="subpheno_toggle_y"):
            cfg["cluster_y"] = not cfg["cluster_y"]
        if b3.button("Toggle counts", key="subpheno_toggle_counts"):
            cfg["show_counts"] = not cfg["show_counts"]
        z_range = st.slider("Colour scale", -6.0, 6.0, (float(cfg["z_min"]), float(cfg["z_max"])), 0.5, key="subpheno_z")
        cfg["z_min"], cfg["z_max"] = z_range

        present = [c for c in norm_cols if c in df_results.columns]
        if not present:
            st.warning("Normalised columns missing - re-run this phenotype.")
        else:
            base_fig, cluster_order, _marker_order, counts, heatmap_z = build_leiden_heatmap_figure(
                f"{view}_{len(df_results)}_{cfg['z_min']}_{cfg['z_max']}",
                df_results, present, cfg["cluster_x"], cfg["cluster_y"], cfg["show_counts"],
                cluster_col=cluster_col, cluster_label="Subcluster",
                z_min=cfg["z_min"], z_max=cfg["z_max"],
                marker_source_note=build_marker_source_note(present, method_key="subpheno"),
            )
            if base_fig:
                pmap = _state("subpheno_phenotype_map", {})
                for c in cluster_order:
                    pmap.setdefault(_cluster_key(c), "Unassigned")

                fig = go.Figure(base_fig)
                fig.update_layout(dragmode=False)
                st.plotly_chart(fig, key=f"subpheno_heatmap_{_safe_label(view)}", width="stretch", config={"displayModeBar": False})

                st.caption("Need a name that is not in the list? Add it here:")
                render_add_custom_phenotype_ui(key_prefix="subpheno")
                options = get_all_phenotype_options(include_unassigned=True)

                verdicts = {}
                if report is not None and not report.empty:
                    verdicts = dict(zip(report.cluster.astype(str), report.verdict))

                rows = []
                for c in cluster_order:
                    key = _cluster_key(c)
                    top = ""
                    if not heatmap_z.empty and c in heatmap_z.columns:
                        top = ", ".join(f"{m} ({v:.2f})" for m, v in heatmap_z[c].nlargest(5).items())
                    rows.append({
                        "Cluster": key,
                        "Cells": int(counts.get(c, 0)),
                        "Check": verdicts.get(key, ""),
                        "Top Markers (Score)": top,
                        "Phenotype": pmap.get(key, "Unassigned"),
                    })

                with st.form(f"subpheno_assign_{_safe_label(view)}"):
                    edited = st.data_editor(
                        pd.DataFrame(rows),
                        column_config={
                            "Cluster": st.column_config.TextColumn(disabled=True),
                            "Cells": st.column_config.NumberColumn(disabled=True),
                            "Check": st.column_config.TextColumn(disabled=True, help="'review' = failed a validation check"),
                            "Top Markers (Score)": st.column_config.TextColumn(disabled=True, width="large"),
                            "Phenotype": st.column_config.SelectboxColumn("Phenotype", options=options, required=True),
                        },
                        disabled=["Cluster", "Cells", "Check", "Top Markers (Score)"],
                        hide_index=True, num_rows="fixed", width="stretch",
                        key=f"subpheno_editor_{_safe_label(view)}",
                    )
                    if st.form_submit_button("Sync & Update", type="primary"):
                        pmap.update({_cluster_key(r["Cluster"]): r["Phenotype"] for _, r in edited.iterrows()})
                        st.success("Assignments synced.")
                        st.rerun()

                render_phenotype_statistics(
                    df_results=df_results, cluster_col=cluster_col, phenotype_map=pmap,
                    heatmap_z_ordered=heatmap_z, cluster_label="Subcluster",
                )
                render_publication_heatmap(
                    heatmap_z_ordered=heatmap_z, phenotype_map=pmap, key_prefix="subpheno",
                    df_results=df_results, group_col=cluster_col, marker_columns=present,
                )
                render_save_assignments(
                    df_results=df_results, phenotype_map=pmap, output_col=output_col,
                    project_root=Path(project_root), key_prefix="subpheno", cluster_col=cluster_col,
                    update_cluster_cols=[cluster_col],
                    cell_id_col=st.session_state.get("subpheno_saved_cell_id", cell_id_col),
                    input_path_override=full_input_path,
                )
                render_export_split_data(
                    df_results=df_results, output_col=output_col, phenotype_map=pmap,
                    project_root=Path(project_root), key_prefix="subpheno", cluster_col=cluster_col,
                    cell_id_col=st.session_state.get("subpheno_saved_cell_id", cell_id_col),
                )

    with tabs[2]:
        pmap = _state("subpheno_phenotype_map", {})
        options = get_all_phenotype_options(include_unassigned=True)
        stats = df_results.groupby(cluster_col).size().rename("n").reset_index()
        stats["%"] = (stats.n / len(df_results) * 100).round(2)
        table_df = pd.DataFrame({
            "Cluster": stats[cluster_col].astype(str),
            "Cells": stats.n.astype(int),
            "%": stats["%"],
            "Phenotype": stats[cluster_col].astype(str).map(lambda c: pmap.get(_cluster_key(c), "Unassigned")),
        })
        edited = st.data_editor(
            table_df,
            column_config={
                "Cluster": st.column_config.TextColumn(disabled=True),
                "Cells": st.column_config.NumberColumn(disabled=True, format="%d"),
                "%": st.column_config.NumberColumn(disabled=True, format="%.1f"),
                "Phenotype": st.column_config.SelectboxColumn("Phenotype", options=options, required=True),
            },
            hide_index=True, num_rows="fixed", key=f"subpheno_table_{_safe_label(view)}",
        )
        t1, t2 = st.columns(2)
        if t1.button("Apply", type="primary", key="subpheno_apply_table"):
            pmap.update({_cluster_key(r["Cluster"]): r["Phenotype"] for _, r in edited.iterrows()})
            st.rerun()
        if t2.button("Reset to Unassigned", key="subpheno_reset_table"):
            for c in table_df.Cluster:
                pmap[_cluster_key(c)] = "Unassigned"
            st.rerun()

    with tabs[3]:
        pmap = st.session_state.get("subpheno_phenotype_map", {})
        export = df_results.copy()
        export[output_col] = export[cluster_col].map(lambda c: pmap.get(_cluster_key(c), "Unassigned"))
        drop_norm = st.checkbox("Remove _norm columns", value=True, key="subpheno_drop_norm")
        name = st.text_input("Output filename", f"subphenotyped_{_safe_label(view)}.csv", key="subpheno_export_name")
        if st.button("Export", type="primary", key="subpheno_export"):
            if drop_norm:
                export = export.drop(columns=[c for c in export.columns if c.endswith("_norm")])
            out_dir = Path(project_root) / "outputs/phenotyping"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / name
            export.to_csv(out_file, index=False)
            st.success(f"Exported: {out_file}")
