"""Standalone spatial cell tagging for the Segmentation tab.

Segmentation can already write the spatial domain columns in the same pass as
feature extraction (see :mod:`segmentation_spatial_ui`).  That is convenient but
it couples the two steps: re-tagging after a QuPath edit, after adding a new
tissue class, or after changing a distance rule would mean re-running Cellpose.

This module provides the same tagging engine as an independent step that reads
cell tables that already exist on disk.  Two layouts are handled:

``per_image``
    One CSV per image next to one annotation GeoJSON per image, matched by
    name - the layout the segmentation step produces.
``merged``
    A single CSV holding many images, with one column naming the image each row
    came from.  Every distinct value in that column is matched to a GeoJSON and
    tagged with its own annotation, then the rows are written back out in their
    original order.

Nothing about the cell table is assumed: the coordinate columns, the batch
column and the file-name suffix used for matching are all chosen in the UI, so
tables produced outside SPACEMAP work too.

The tab calls :func:`render_cell_tagging_panel`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import streamlit as st

from spacemap_gui.tabs.segmentation_spatial_ui import (
    GEOJSON_DIR_KEY as SHARED_GEOJSON_DIR_KEY,
    RULE_COLUMNS,
    geojson_classes,
    normalize_rule_records,
)
from spacemap_gui.ui_utils import directory_input_with_browse, file_signature
from spacemap_gui import terminal

# Session-state keys. ``persistence.get_tab_affected_from_key`` maps the ``seg_``
# prefix to the Segmentation page, so everything here is saved into
# <project>.json alongside the rest of this tab. Do not rename the prefix.
MODE_KEY = "seg_tag_mode"
CSV_DIR_KEY = "seg_tag_csv_dir"
CSV_SUFFIX_KEY = "seg_tag_csv_suffix"
MERGED_DIR_KEY = "seg_tag_merged_dir"
MERGED_FILE_KEY = "seg_tag_merged_file"
BATCH_COLUMN_KEY = "seg_tag_batch_column"
# The annotation folder is shared with the Segmentation subtab
# (``SHARED_GEOJSON_DIR_KEY``) so there is one setting, not two that can drift
# apart. These two keys only come into play when the user deliberately points
# tagging at a different folder.
GEOJSON_OVERRIDE_KEY = "seg_tag_geojson_override"
GEOJSON_MATCHES_KEY = "seg_tag_geojson_matches"  # {cell table or image id: annotation file name}, chosen by hand
GEOJSON_DIR_KEY = "seg_tag_geojson_dir"
OUTPUT_DIR_KEY = "seg_tag_output_dir"
OUTPUT_SUFFIX_KEY = "seg_tag_output_suffix"
X_COLUMN_KEY = "seg_tag_x_column"
Y_COLUMN_KEY = "seg_tag_y_column"
TISSUE_LABEL_KEY = "seg_tag_tissue_label"
RULES_KEY = "seg_tag_rules"
EDGE_ENABLED_KEY = "seg_tag_edge_enabled"
EDGE_REFERENCE_KEY = "seg_tag_edge_reference"
EDGE_BUFFER_KEY = "seg_tag_edge_buffer"
EDGE_COLUMN_KEY = "seg_tag_edge_column"
MIN_BED_CELLS_KEY = "seg_tag_min_cells_per_bed_target"
CLEAN_PATH_KEY = "seg_tag_require_clean_path"
BED_LABELS_KEY = "seg_tag_bed_labels"
OVERWRITE_COLUMNS_KEY = "seg_tag_overwrite_columns"
USE_GPU_KEY = "seg_tag_use_gpu_distances"

MODE_PER_IMAGE = "One CSV per image"
MODE_MERGED = "One merged CSV for several images"
MODES = (MODE_PER_IMAGE, MODE_MERGED)

DEFAULT_CSV_SUFFIX = "_cell_features"
DEFAULT_OUTPUT_SUFFIX = "_tagged"
DEFAULT_TISSUE_LABEL = "tissue"
DEFAULT_MIN_CELLS_PER_BED_TARGET = 10

# Results of the last run, kept out of the project JSON (leading underscore).
RESULT_KEY = "_seg_tag_last_run"


# ==========================================================================
# Discovery and matching
# ==========================================================================

def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab5', message, step='Cell Tagging', level=level)


def list_csv_files(directory: str) -> list[str]:
    """CSV file names in ``directory``, sorted, hidden files skipped."""
    if not directory or not os.path.isdir(directory):
        return []
    try:
        return sorted(
            name for name in os.listdir(directory)
            if name.lower().endswith(".csv") and not name.startswith(".")
        )
    except OSError:
        return []


def sample_id_from_csv(filename: str, strip_suffix: str = DEFAULT_CSV_SUFFIX) -> str:
    """Turn ``sample_1_cell_features.csv`` into ``sample_1``.

    ``strip_suffix`` is configurable because the cell table may come from
    somewhere other than this pipeline; an empty value keeps the whole stem.
    """
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    suffix = str(strip_suffix or "")
    if suffix and stem.lower().endswith(suffix.lower()):
        stem = stem[: -len(suffix)]
    return stem


def match_geojson(sample_id: str, geojson_dir: str, chosen: str | None = None) -> str | None:
    """Annotation GeoJSON for one sample id, or None.

    *chosen* is an annotation file in *geojson_dir* picked by hand for this source (names that do not
    match); otherwise the segmentation matcher finds it, so this step resolves exactly the same file the
    in-line tagging would have used.
    """
    from spacemap.segmentation_cellpose import find_annotation_geojson

    if not sample_id or not geojson_dir or not os.path.isdir(geojson_dir):
        return None
    if chosen:
        path = os.path.join(geojson_dir, os.path.basename(str(chosen)))
        return path if os.path.isfile(path) else None
    return find_annotation_geojson(str(sample_id), str(geojson_dir))


def _state(state):
    """The saved tagging settings: *state* when given (Studio), else this page's session."""
    return st.session_state if state is None else state


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_csv_header(csv_path: str, signature: tuple[int, int]) -> list[str]:
    """Column names of a CSV, read without loading the body.

    ``signature`` stays a plain argument on purpose - ``st.cache_data`` drops
    underscore-prefixed parameters from the cache key, which would pin a path to
    the first version ever read.
    """
    try:
        return list(pd.read_csv(csv_path, nrows=0).columns)
    except Exception:
        return []


def csv_columns(csv_path: str | None) -> list[str]:
    if not csv_path or not os.path.isfile(csv_path):
        return []
    return _cached_csv_header(csv_path, file_signature(csv_path))


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_batch_values(csv_path: str, column: str, signature: tuple[int, int]) -> list[str]:
    try:
        series = pd.read_csv(csv_path, usecols=[column])[column]
    except Exception:
        return []
    return [str(value) for value in pd.unique(series.dropna())]


def batch_values(csv_path: str | None, column: str) -> list[str]:
    """Distinct image ids inside a merged cell table."""
    if not csv_path or not os.path.isfile(csv_path) or not column:
        return []
    return _cached_batch_values(csv_path, column, file_signature(csv_path))


def _saved_matches(matches=None) -> dict:
    return dict(matches if matches is not None else _state(None).get(GEOJSON_MATCHES_KEY) or {})


def build_match_table(sample_ids: Sequence[str], geojson_dir: str, source_labels: Sequence[str] | None = None,
                      matches=None) -> pd.DataFrame:
    """One row per sample: which annotation matched (by name, or chosen by hand) and what classes it holds."""
    rows = []
    matches = _saved_matches(matches)
    labels = list(source_labels) if source_labels is not None else list(sample_ids)
    for label, sample_id in zip(labels, sample_ids):
        path = match_geojson(sample_id, geojson_dir, matches.get(label) or matches.get(sample_id))
        classes = geojson_classes(path)
        rows.append({
            "Source": label,
            "Matched by": sample_id,
            "Annotation file": os.path.basename(path) if path else "— none —",
            "Classes": ", ".join(classes) if classes else "",
            "Class count": len(classes),
        })
    return pd.DataFrame(rows)


def collect_classes(sample_ids: Sequence[str], geojson_dir: str, source_labels: Sequence[str] | None = None,
                    matches=None) -> list[str]:
    """Union of classification names across every matched annotation."""
    names: list[str] = []
    matches = _saved_matches(matches)
    labels = list(source_labels) if source_labels is not None else list(sample_ids)
    for label, sample_id in zip(labels, sample_ids):
        chosen = matches.get(label) or matches.get(sample_id)
        for class_name in geojson_classes(match_geojson(sample_id, geojson_dir, chosen)):
            if class_name not in names:
                names.append(class_name)
    return names


# ==========================================================================
# Rules state
# ==========================================================================

def align_rules(classes: Sequence[str], tissue_label: str, saved_records: Any) -> list[dict[str, Any]]:
    """Keep saved settings for classes that still exist and add any new ones."""
    saved = {
        record["Annotation Label"].lower(): record
        for record in normalize_rule_records(saved_records)
    }
    aligned: list[dict[str, Any]] = []
    for name in classes:
        existing = saved.get(str(name).lower())
        is_tissue = str(name).lower() == str(tissue_label).lower()
        if existing is not None:
            existing["Annotation Label"] = str(name)
            # In tab5 the selected containing tissue is authoritative: every
            # subannotation is isolated to it. The shared backend still
            # supports ungrouped rules for programmatic/legacy callers.
            existing["Group by Tissue ROI"] = not is_tissue
            aligned.append(existing)
            continue
        aligned.append({
            "Annotation Label": str(name),
            "Binary (0/1)": True,
            "ROI ID": is_tissue,
            "ROI Size": is_tissue,
            "Distance": False,
            "Bed (Use 2 Annotations)": False,
            "Bed Comp. Target": "",
            "Group by Tissue ROI": not is_tissue,
        })
    return aligned


def build_engine_inputs(state=None) -> tuple[list, Any, str, int, bool]:
    """Translate the saved UI state (or *state*) into engine objects.

    Returns ``(rules, edge_config, tissue_label, minimum_bed_cells,
    require_clean_path)`` ready for
    :func:`spacemap.cell_spatial_annotation.annotate_cells`.
    """
    from spacemap.cell_spatial_annotation import EdgeTagConfig, rules_from_records

    state = _state(state)

    tissue_label = str(state.get(TISSUE_LABEL_KEY, DEFAULT_TISSUE_LABEL))
    rules = rules_from_records(normalize_rule_records(state.get(RULES_KEY, [])))
    for rule in rules:
        rule.group_by_roi = rule.label.lower() != tissue_label.lower()
        # This panel uses one explicit bed selection; the other segmentation
        # panel and programmatic callers can continue using legacy pair rules.
        rule.bed = False
        rule.bed_target = ""
    edge_config = None
    if state.get(EDGE_ENABLED_KEY, False):
        edge_config = EdgeTagConfig(
            reference=str(state.get(EDGE_REFERENCE_KEY, DEFAULT_TISSUE_LABEL)),
            buffer_px=float(state.get(EDGE_BUFFER_KEY, 100.0)),
            column=str(state.get(EDGE_COLUMN_KEY, "edge_cell")),
        )
    minimum_bed_cells = max(
        int(state.get(MIN_BED_CELLS_KEY, DEFAULT_MIN_CELLS_PER_BED_TARGET)),
        1,
    )
    require_clean_path = bool(state.get(CLEAN_PATH_KEY, False))
    return rules, edge_config, tissue_label, minimum_bed_cells, require_clean_path


def selected_bed_labels(records=None, state=None) -> list[str]:
    """Restore the selection, migrating saved two-annotation rules once."""
    state = _state(state)
    if BED_LABELS_KEY in state:
        return list(state[BED_LABELS_KEY])
    names = []
    for record in normalize_rule_records(
        state.get(RULES_KEY, []) if records is None else records
    ):
        if record["Bed (Use 2 Annotations)"] and record["Bed Comp. Target"]:
            names.extend([record["Annotation Label"], record["Bed Comp. Target"]])
    return list(dict.fromkeys(names))


# ==========================================================================
# Execution
# ==========================================================================

def _validate_distance_engine(require_clean_path: bool) -> str:
    """Do not silently use an old module cached by a long-running GUI."""
    if not require_clean_path:
        return "Euclidean / legacy"
    from spacemap import cell_spatial_annotation

    method = getattr(cell_spatial_annotation, "CLEAN_PATH_ENGINE", None)
    if method != "Triangle visibility (CPU)":
        raise RuntimeError("Restart Streamlit to load the updated triangle visibility engine before tagging.")
    return method


def run_per_image(
    csv_dir: str,
    csv_names: Sequence[str],
    geojson_dir: str,
    output_dir: str,
    csv_suffix: str,
    output_suffix: str,
    x_col: str,
    y_col: str,
    progress=None,
    state=None,
    matches=None,
) -> list[dict[str, Any]]:
    """Tag one CSV per image. Returns one result record per input file."""
    from spacemap.cell_spatial_annotation import annotate_cells

    state = _state(state)
    matches = dict(matches if matches is not None else state.get(GEOJSON_MATCHES_KEY) or {})
    rules, edge_config, tissue_label, minimum_bed_cells, require_clean_path = build_engine_inputs(state)
    method = _validate_distance_engine(require_clean_path)
    _log_console(f"Spatial cell tagging engine: {method}")
    bed_labels = selected_bed_labels(state=state)
    overwrite_existing = bool(state.get(OVERWRITE_COLUMNS_KEY, True))
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict[str, Any]] = []

    for position, csv_name in enumerate(csv_names, start=1):
        if progress is not None:
            progress(position, len(csv_names), csv_name)
        sample_id = sample_id_from_csv(csv_name, csv_suffix)
        record: dict[str, Any] = {"Source": csv_name, "Matched by": sample_id}
        geojson_path = match_geojson(sample_id, geojson_dir, matches.get(csv_name) or matches.get(sample_id))
        if not geojson_path:
            record.update({"Status": "skipped — no annotation matched", "Cells": 0, "Columns added": 0})
            results.append(record)
            continue
        try:
            frame = pd.read_csv(os.path.join(csv_dir, csv_name))
            annotated, summary = annotate_cells(
                frame, geojson_path, rules,
                x_col=x_col or None, y_col=y_col or None,
                tissue_label=tissue_label, edge_config=edge_config,
                min_cells_per_bed_target=minimum_bed_cells,
                require_clean_path=require_clean_path,
                bed_labels=bed_labels, overwrite_existing=overwrite_existing,
                use_gpu=bool(state.get(USE_GPU_KEY, False)),
            )
            out_name = f"{sample_id_from_csv(csv_name, '')}{output_suffix}.csv"
            out_path = os.path.join(output_dir, out_name)
            annotated.to_csv(out_path, index=False)
            record.update({
                "Status": "ok",
                "Cells": int(len(annotated)),
                "Columns added": len(set(summary.get("columns", [])) - set(summary.get("columns_overwritten", []))),
                "Columns overwritten": len(summary.get("columns_overwritten", [])),
                "Columns preserved": len(summary.get("columns_preserved", [])),
                "Annotation": os.path.basename(geojson_path),
                "Output": out_name,
                "Warnings": "; ".join(summary.get("warnings", [])),
                "Distance backend": summary.get("distance_acceleration", {}).get("backend", "CPU"),
                "Distance method": summary.get("distance_method", "Euclidean / legacy"),
            })
        except Exception as error:
            record.update({"Status": f"failed — {error}", "Cells": 0, "Columns added": 0})
        results.append(record)
    return results


def run_merged(
    csv_path: str,
    batch_column: str,
    geojson_dir: str,
    output_path: str,
    x_col: str,
    y_col: str,
    progress=None,
    state=None,
    matches=None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Tag a merged cell table one image at a time.

    Each group is annotated against its own GeoJSON and the groups are then
    reassembled on the original row index, so the output keeps the input's row
    order. With overwrite enabled, selected output columns are reset across the
    whole table, including groups with missing/failed annotations.
    """
    from spacemap.cell_spatial_annotation import annotate_cells, spatial_columns_for_rules

    state = _state(state)
    matches = dict(matches if matches is not None else state.get(GEOJSON_MATCHES_KEY) or {})
    rules, edge_config, tissue_label, minimum_bed_cells, require_clean_path = build_engine_inputs(state)
    method = _validate_distance_engine(require_clean_path)
    _log_console(f"Spatial cell tagging engine: {method}")
    bed_labels = selected_bed_labels(state=state)
    overwrite_existing = bool(state.get(OVERWRITE_COLUMNS_KEY, True))
    frame = pd.read_csv(csv_path)
    if batch_column not in frame.columns:
        raise ValueError(f"Column '{batch_column}' is not in {os.path.basename(csv_path)}.")

    keys = frame[batch_column].astype(str)
    existing_output_columns = [
        column for column in spatial_columns_for_rules(rules, edge_config, bed_labels)
        if column in frame.columns
    ]
    if overwrite_existing:
        for column in existing_output_columns:
            frame[column] = float("nan")
    unique_keys = list(pd.unique(keys))
    parts: list[pd.DataFrame] = []
    results: list[dict[str, Any]] = []

    for position, key in enumerate(unique_keys, start=1):
        if progress is not None:
            progress(position, len(unique_keys), key)
        subset = frame[keys == key]
        # The batch value is handed to the shared matcher untouched: it already
        # strips real image extensions, and splitting on the last dot here would
        # mangle ids that legitimately contain one (``Patient_1.2``).
        record: dict[str, Any] = {
            "Source": key, "Matched by": key,
            "Columns overwritten": len(existing_output_columns) if overwrite_existing else 0,
            "Columns preserved": 0 if overwrite_existing else len(existing_output_columns),
        }
        geojson_path = match_geojson(key, geojson_dir, matches.get(key))
        if not geojson_path:
            parts.append(subset)
            record.update({"Status": "skipped — no annotation matched", "Cells": int(len(subset)), "Columns added": 0})
            results.append(record)
            continue
        try:
            annotated, summary = annotate_cells(
                subset, geojson_path, rules,
                x_col=x_col or None, y_col=y_col or None,
                tissue_label=tissue_label, edge_config=edge_config,
                min_cells_per_bed_target=minimum_bed_cells,
                require_clean_path=require_clean_path,
                bed_labels=bed_labels, overwrite_existing=overwrite_existing,
                use_gpu=bool(state.get(USE_GPU_KEY, False)),
            )
            parts.append(annotated)
            record.update({
                "Status": "ok",
                "Cells": int(len(annotated)),
                "Columns added": len(set(summary.get("columns", [])) - set(summary.get("columns_overwritten", []))),
                "Columns overwritten": len(summary.get("columns_overwritten", [])),
                "Columns preserved": len(summary.get("columns_preserved", [])),
                "Annotation": os.path.basename(geojson_path),
                "Warnings": "; ".join(summary.get("warnings", [])),
                "Distance backend": summary.get("distance_acceleration", {}).get("backend", "CPU"),
                "Distance method": summary.get("distance_method", "Euclidean / legacy"),
            })
        except Exception as error:
            parts.append(subset)
            record.update({"Status": f"failed — {error}", "Cells": int(len(subset)), "Columns added": 0})
        results.append(record)

    combined = pd.concat(parts).sort_index() if parts else frame
    output_directory = os.path.dirname(str(output_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined, results


# ==========================================================================
# UI
# ==========================================================================

def _render_column_meanings(tissue_label: str) -> None:
    with st.expander("What each column means", expanded=False):
        st.markdown(
            f"""
| Setting | Column added | Meaning |
|---|---|---|
| Binary (0/1) | `<label>` | 1 when the cell centroid falls inside that annotation |
| ROI ID | `<label>_roi` | Which individual polygon of that annotation contains the cell (largest = 1) |
| ROI Size | `<label>_roi_size` | Area of that containing polygon, in square pixels |
| Distance | `<label>_distance` | Signed distance to the nearest polygon of that annotation: **negative inside, positive outside** |
| Bed annotations | `<label>_bed`, `<label>_bed_id` for each selected class | Competition between two or more annotations: 1 for the closest qualifying class, 0 for the others; only the winning bed gets an ROI ID |
| Group by {tissue_label} piece | — | Automatically on for every subannotation: everything is evaluated **inside the cell's own `{tissue_label}` piece** |

**How distance and bed work together.** Say `Tumor` and `normal` both have
**Distance** ticked and **Group by {tissue_label} piece** ticked. Every cell is measured to the nearest
`Tumor` boundary and the nearest `normal` boundary *within its own piece of tissue*. If a piece
contains no `Tumor` at all, the cells in it get an empty `Tumor_distance` rather than a distance
reaching across to an unrelated piece.

Select `tumor`, `normal`, and `reactive` in **Bed annotations** to create three competing beds.
Each cell is assigned to the closest qualifying annotation within its tissue piece, using signed
distances (negative inside, positive outside). Distance columns can be enabled separately.
All beds stay blank outside the containing tissue or when no target qualifies. Exact distance
ties go to the last annotation alphabetically, ignoring case.

**Group by {tissue_label} piece is enforced in this tab** for every annotation except the containing
`{tissue_label}` row itself. This prevents a saved unchecked setting from measuring cells against a
compartment in another tissue fragment.

**Minimum cells per annotation ROI for a bed** is applied after each annotation polygon is clipped to
its containing tissue piece. A clipped Tumor or Epithelia fragment with fewer cells than this threshold
still keeps its binary/ROI annotation, but it does not seed a distance-based bed. A starting value of
10 is useful for suppressing tiny annotation artifacts; review sensitivity at 5, 10 and 20 cells.
            """
        )


def _render_rules_form(all_classes: Sequence[str], tissue_label: str) -> None:
    """Rules matrix + edge tagging, buffered in a form and saved on submit."""
    from spacemap_gui.persistence import save_active_project_state
    from spacemap.cell_spatial_annotation import EdgeTagConfig, rules_from_records, spatial_columns_for_rules

    aligned = align_rules(all_classes, tissue_label, st.session_state.get(RULES_KEY, []))
    rules_frame = pd.DataFrame(aligned, columns=[
        name for name in RULE_COLUMNS if name not in ("Bed (Use 2 Annotations)", "Bed Comp. Target")
    ])

    with st.form("seg_tag_rules_form"):
        bed_options = list(all_classes)
        saved_beds = {name.lower() for name in selected_bed_labels(aligned)}
        # Discard stale widget values after switching annotation folders.
        bed_widget_key = "seg_tag_bed_labels_widget"
        if bed_widget_key in st.session_state:
            previous = st.session_state[bed_widget_key]
            if any(name not in bed_options for name in previous):
                st.session_state[bed_widget_key] = [name for name in previous if name in bed_options]
        bed_labels = st.multiselect(
            "Bed annotations", bed_options,
            default=[name for name in bed_options if name.lower() in saved_beds],
            key=bed_widget_key,
            help="Select two or more competing classes, such as tumor, normal, and reactive. Leave empty to skip beds.",
        )
        st.caption(
            "Each selected annotation adds <label>_bed and <label>_bed_id. "
            "The closest qualifying annotation wins within each tissue piece."
        )
        edited = st.data_editor(
            rules_frame,
            column_config={
                "Annotation Label": st.column_config.TextColumn("Annotation", disabled=True, width="medium"),
                "Binary (0/1)": st.column_config.CheckboxColumn(
                    "Binary", help="Add a 0/1 membership column named after the annotation.", width="small"),
                "ROI ID": st.column_config.CheckboxColumn(
                    "ROI ID",
                    help=(
                        "Add <label>_roi with the index of the containing polygon. "
                        f"Numbered separately per {tissue_label} piece when 'Group by {tissue_label} piece' is on."
                    ),
                    width="small"),
                "ROI Size": st.column_config.CheckboxColumn(
                    "ROI size",
                    help=(
                        "Add <label>_roi_size with the containing polygon's area in px². "
                        f"Clipped to the {tissue_label} piece when 'Group by {tissue_label} piece' is on."
                    ),
                    width="small"),
                "Distance": st.column_config.CheckboxColumn(
                    "Distance",
                    help=(
                        "Add <label>_distance: signed distance to the nearest polygon of this annotation, "
                        "negative inside and positive outside. Empty when the cell's tissue piece contains "
                        "none of this annotation."
                    ),
                    width="small"),
                "Group by Tissue ROI": st.column_config.CheckboxColumn(
                    f"Group by {tissue_label} piece",
                    disabled=True,
                    help=(
                        f"Automatically evaluate every subannotation inside the '{tissue_label}' polygon "
                        "containing the cell — NOT inside this row's own polygons. This keeps a cell "
                        "from being measured against a compartment in an unrelated piece of tissue. Cells in a "
                        f"'{tissue_label}' piece holding none of this annotation get an empty distance."
                    ),
                    width="medium"),
            },
            hide_index=True,
            num_rows="fixed",
            key="seg_tag_rules_editor",
            width="stretch",
        )

        st.markdown("**Bed target quality control**")
        minimum_bed_cells = st.number_input(
            "Minimum cells per clipped annotation ROI for bed assignment",
            min_value=1,
            max_value=100000,
            value=max(
                int(st.session_state.get(
                    MIN_BED_CELLS_KEY, DEFAULT_MIN_CELLS_PER_BED_TARGET
                )),
                1,
            ),
            step=1,
            key="seg_tag_min_cells_per_bed_target_widget",
            help=(
                f"Each annotation polygon is first clipped to the containing {tissue_label} piece. "
                "A clipped fragment must contain at least this many cell centroids to seed a bed. "
                "Binary and ROI columns are retained even when a fragment is below threshold."
            ),
        )
        st.caption(
            "Recommended starting value: 10 cells. For final analysis, compare 5, 10 and 20."
        )

        st.markdown("**Fragmented tissue**")
        require_clean_path = st.checkbox(
            "Only measure distances a straight line can actually reach",
            value=bool(st.session_state.get(CLEAN_PATH_KEY, False)),
            key="seg_tag_clean_path_widget",
            help=(
                "Uses fast triangle visibility for distances grouped by tissue ROI and for tissue-scoped beds. "
                "Lines stop at selected distance/bed annotations, tissue boundaries and gaps. Cells inside a "
                "selected annotation receive a negative distance to its border; hidden classes stay blank. "
                "Cells cannot reach another tissue ROI. Small ROIs below the bed threshold still block lines."
            ),
        )
        if require_clean_path:
            st.caption("Distance method: Triangle visibility (CPU). Applies to tissue-grouped distances and beds.")

        st.markdown("**Performance**")
        use_gpu = st.checkbox(
            "Use GPU for boundary distances when available",
            value=bool(st.session_state.get(USE_GPU_KEY, False)),
            key="seg_tag_use_gpu_distances_widget",
            help=(
                "CUDA can accelerate large ordinary boundary-distance calculations. Gap-aware distances "
                "and beds use CPU workers. If no compatible GPU is available, calculations use the CPU."
            ),
        )

        st.markdown("**Edge cell tagging**")
        edge_columns = st.columns([1, 1, 1])
        edge_enabled = edge_columns[0].checkbox(
            "Flag cells near an edge",
            value=bool(st.session_state.get(EDGE_ENABLED_KEY, False)),
            key="seg_tag_edge_enabled_widget",
            help="Adds a 0/1 column marking cells within a set distance of a reference annotation's boundary.",
        )
        reference_options = list(all_classes)
        saved_reference = st.session_state.get(EDGE_REFERENCE_KEY, tissue_label)
        if saved_reference not in reference_options:
            saved_reference = tissue_label if tissue_label in reference_options else reference_options[0]
        edge_reference = edge_columns[0].selectbox(
            "Reference annotation", reference_options,
            index=reference_options.index(saved_reference),
            key="seg_tag_edge_reference_widget",
            help="Boundary the distance is measured from.",
        )
        edge_buffer = edge_columns[1].number_input(
            "Edge buffer (px)", min_value=0.0, max_value=100000.0, step=1.0,
            value=float(st.session_state.get(EDGE_BUFFER_KEY, 100.0)),
            key="seg_tag_edge_buffer_widget",
            help="Cells closer than this to the boundary are flagged.",
        )
        edge_column = edge_columns[2].text_input(
            "Column name", value=str(st.session_state.get(EDGE_COLUMN_KEY, "edge_cell")),
            key="seg_tag_edge_column_widget",
            help="Name of the 0/1 output column.",
        )
        saved = st.form_submit_button("Save tagging rules", width="stretch")

    if len(bed_labels) == 1:
        st.warning("Select at least two bed annotations, or clear the selection to skip beds.")
    if saved and len(bed_labels) != 1:
        st.session_state[RULES_KEY] = normalize_rule_records(edited.to_dict("records"))
        st.session_state[BED_LABELS_KEY] = list(bed_labels)
        st.session_state[EDGE_ENABLED_KEY] = bool(edge_enabled)
        st.session_state[EDGE_REFERENCE_KEY] = str(edge_reference)
        st.session_state[EDGE_BUFFER_KEY] = float(edge_buffer)
        st.session_state[EDGE_COLUMN_KEY] = str(edge_column)
        st.session_state[MIN_BED_CELLS_KEY] = int(minimum_bed_cells)
        st.session_state[CLEAN_PATH_KEY] = bool(require_clean_path)
        st.session_state[USE_GPU_KEY] = bool(use_gpu)
        save_active_project_state("tab5_segmentation")
        st.success("Tagging rules saved to the project configuration.")

    preview = spatial_columns_for_rules(
        rules_from_records(normalize_rule_records(edited.to_dict("records"))),
        EdgeTagConfig(column=str(edge_column)) if edge_enabled else None,
        bed_labels=bed_labels,
    )
    if preview:
        st.caption(f"Output columns ({len(preview)}): " + ", ".join(f"`{name}`" for name in preview))
    else:
        st.warning("No columns are selected yet, so running would leave the cell table unchanged.")


def _render_coordinate_pickers(columns: Sequence[str]) -> tuple[str, str]:
    """X/Y centroid pickers, defaulting to whatever auto-detection finds."""
    from spacemap.cell_spatial_annotation import resolve_coordinate_columns

    if not columns:
        return "", ""
    detected_x, detected_y = resolve_coordinate_columns(pd.DataFrame(columns=list(columns)))

    def _index(saved_key: str, detected: str | None) -> int:
        saved = st.session_state.get(saved_key)
        for candidate in (saved, detected):
            if candidate in columns:
                return list(columns).index(candidate)
        return 0

    left, right = st.columns(2)
    x_col = left.selectbox(
        "X centroid column", list(columns), index=_index(X_COLUMN_KEY, detected_x),
        key="seg_tag_x_column_widget",
        help="Cell centroid X, in full-resolution image pixels — the same space the GeoJSON polygons live in.",
    )
    y_col = right.selectbox(
        "Y centroid column", list(columns), index=_index(Y_COLUMN_KEY, detected_y),
        key="seg_tag_y_column_widget",
        help="Cell centroid Y, in full-resolution image pixels.",
    )
    st.session_state[X_COLUMN_KEY] = x_col
    st.session_state[Y_COLUMN_KEY] = y_col
    if detected_x and detected_y and (x_col != detected_x or y_col != detected_y):
        st.caption(f"Auto-detection suggested `{detected_x}` / `{detected_y}`.")
    return x_col, y_col


def _render_results() -> None:
    result = st.session_state.get(RESULT_KEY)
    if not result:
        return
    st.markdown("##### Last run")
    frame = pd.DataFrame(result.get("records", []))
    if not frame.empty:
        ok = int((frame["Status"] == "ok").sum())
        total = len(frame)
        if ok == total:
            st.success(f"Tagged {ok}/{total} source(s), {int(frame['Cells'].sum()):,} cells.")
        elif ok:
            st.warning(f"Tagged {ok}/{total} source(s). See the table for the rest.")
        else:
            st.error("Nothing was tagged. See the table below.")
        st.dataframe(frame, hide_index=True, width="stretch")
    for path in result.get("outputs", []):
        st.caption(f"Written: `{path}`")


def render_cell_tagging_panel(project_root: Path) -> None:
    """Render the standalone cell tagging step."""
    st.markdown("### Spatial cell tagging")
    st.caption(
        "Add tissue-domain columns to cell tables that already exist on disk. This is the same engine the "
        "segmentation step can run inline, exposed as its own step so you can re-tag after editing an "
        "annotation in QuPath, adding a tissue class or changing a distance rule — without re-running Cellpose."
    )

    # --- Step 1: layout of the cell tables ---
    st.markdown("#### Step 1 — How are the cell tables stored?")
    saved_mode = st.session_state.get(MODE_KEY, MODE_PER_IMAGE)
    if saved_mode not in MODES:
        saved_mode = MODE_PER_IMAGE
    mode = st.radio(
        "Cell table layout",
        MODES,
        index=MODES.index(saved_mode),
        key=MODE_KEY,
        horizontal=True,
        help=(
            "One CSV per image is what the segmentation step writes. Choose the merged option when the "
            "per-image tables have already been concatenated into a single file — you then pick the column "
            "that says which image each row came from."
        ),
    )

    # --- Step 2: inputs ---
    st.markdown("#### Step 2 — Inputs")

    # One annotation folder for the whole tab. The Segmentation subtab owns the
    # setting; this subtab follows it unless the user explicitly overrides, so
    # the same reviewed GeoJSON drives the segmentation mask, the inline spatial
    # columns and this standalone tagging step.
    shared_dir = st.session_state.get(SHARED_GEOJSON_DIR_KEY, "") or str(
        project_root / "outputs" / "processed_annotations" / "stage1b_compartments_zones"
    )
    override = st.checkbox(
        "Tag against a different annotation folder",
        value=bool(st.session_state.get(GEOJSON_OVERRIDE_KEY, False)),
        key="seg_tag_geojson_override_widget",
        help=(
            "Off (recommended): use the same Annotation GeoJSON directory as the Segmentation subtab, so "
            "there is a single reviewed annotation per ROI behind everything. Turn this on only to tag "
            "against a different set of annotation files."
        ),
    )
    st.session_state[GEOJSON_OVERRIDE_KEY] = bool(override)

    if override:
        directory_input_with_browse(
            label="Annotation GeoJSON directory",
            text_key=GEOJSON_DIR_KEY,
            default_value=st.session_state.get(GEOJSON_DIR_KEY, "") or shared_dir,
            browser_key_prefix="seg_tag_geojson_dir_browse",
            help_text=(
                "Folder holding the annotation files. Files are matched to cell tables by name, so an edit "
                "saved from QuPath under the same name is picked up on the next run."
            ),
        )
        geojson_dir = st.session_state.get(GEOJSON_DIR_KEY, "")
    else:
        st.text_input(
            "Annotation GeoJSON directory",
            value=shared_dir,
            disabled=True,
            key="seg_tag_geojson_shared_display",
            help=(
                "Shared with the Segmentation subtab — change it there and this follows. Files are matched "
                "to cell tables by name, so a QuPath edit saved under the same name is picked up on the "
                "next run."
            ),
        )
        st.caption("Shared with **Segmentation & feature extraction → Tissue mask & spatial annotation**.")
        geojson_dir = shared_dir

    sample_ids: list[str] = []
    source_labels: list[str] = []
    columns: list[str] = []
    csv_dir = ""
    csv_names: list[str] = []
    merged_path = ""
    batch_column = ""

    if mode == MODE_PER_IMAGE:
        directory_input_with_browse(
            label="Cell table (CSV) directory",
            text_key=CSV_DIR_KEY,
            default_value=st.session_state.get(CSV_DIR_KEY, "")
            or str(project_root / "outputs" / "segmentation" / "cellpose" / "features"),
            browser_key_prefix="seg_tag_csv_dir_browse",
            help_text="Folder holding one cell table per image, for example sample_1_cell_features.csv.",
        )
        csv_dir = st.session_state.get(CSV_DIR_KEY, "")
        available = list_csv_files(csv_dir)
        if not available:
            st.info(f"No CSV files found in `{csv_dir or '(not set)'}`.")
            return

        suffix_column, select_column = st.columns([1, 2])
        # Widget key deliberately differs from the persisted key: a value restored
        # from the project JSON plus a widget default on the same key is what
        # Streamlit warns about, so the widget writes back instead.
        csv_suffix = suffix_column.text_input(
            "File-name suffix to strip when matching",
            value=str(st.session_state.get(CSV_SUFFIX_KEY, DEFAULT_CSV_SUFFIX)),
            key="seg_tag_csv_suffix_widget",
            help=(
                "Removed from the CSV stem before looking for the annotation file. With the default, "
                "sample_1_cell_features.csv is matched as sample_1. Clear it to match on the whole stem."
            ),
        )
        st.session_state[CSV_SUFFIX_KEY] = csv_suffix
        stored = st.session_state.get("seg_tag_csv_select")
        if isinstance(stored, list) and any(name not in available for name in stored):
            st.session_state["seg_tag_csv_select"] = [name for name in stored if name in available]
        csv_names = select_column.multiselect(
            "Cell tables to tag",
            available,
            default=available,
            key="seg_tag_csv_select",
            help="Every selected table is tagged with its own matching annotation file.",
        )
        if not csv_names:
            st.warning("Select at least one cell table.")
            return
        sample_ids = [sample_id_from_csv(name, csv_suffix) for name in csv_names]
        source_labels = list(csv_names)
        columns = csv_columns(os.path.join(csv_dir, csv_names[0]))

    else:
        directory_input_with_browse(
            label="Merged CSV directory",
            text_key=MERGED_DIR_KEY,
            default_value=st.session_state.get(MERGED_DIR_KEY, "")
            or st.session_state.get(CSV_DIR_KEY, "")
            or str(project_root / "outputs" / "segmentation" / "cellpose" / "features"),
            browser_key_prefix="seg_tag_merged_dir_browse",
            help_text="Folder holding the single merged cell table.",
        )
        merged_dir = st.session_state.get(MERGED_DIR_KEY, "")
        available = list_csv_files(merged_dir)
        if not available:
            st.info(f"No CSV files found in `{merged_dir or '(not set)'}`.")
            return
        stored = st.session_state.get(MERGED_FILE_KEY)
        index = available.index(stored) if stored in available else 0
        merged_name = st.selectbox(
            "Merged cell table", available, index=index, key=MERGED_FILE_KEY,
            help="The single CSV holding cells from several images.",
        )
        merged_path = os.path.join(merged_dir, merged_name)
        columns = csv_columns(merged_path)
        if not columns:
            st.error(f"Could not read the header of `{merged_name}`.")
            return

        saved_batch = st.session_state.get(BATCH_COLUMN_KEY)
        guesses = [name for name in columns if str(name).lower() in
                   ("batch", "batch_id", "image", "image_id", "sample", "sample_id", "roi", "roi_id", "source")]
        default_batch = saved_batch if saved_batch in columns else (guesses[0] if guesses else columns[0])
        batch_column = st.selectbox(
            "Image ID column", columns, index=columns.index(default_batch), key=BATCH_COLUMN_KEY,
            help=(
                "The column naming the image each row came from. Every distinct value is matched to an "
                "annotation file and tagged with its own annotation."
            ),
        )
        values = batch_values(merged_path, batch_column)
        if not values:
            st.error(f"Column `{batch_column}` holds no values.")
            return
        st.caption(f"`{batch_column}` holds {len(values)} distinct image id(s).")
        sample_ids = list(values)
        source_labels = list(values)

    # --- matching overview ---
    match_table = build_match_table(sample_ids, geojson_dir, source_labels)
    matched = int((match_table["Annotation file"] != "— none —").sum()) if not match_table.empty else 0
    if matched == len(sample_ids):
        st.success(f"Matched an annotation file for all {len(sample_ids)} source(s).")
    elif matched:
        st.warning(f"Matched {matched} of {len(sample_ids)} source(s). Unmatched ones are skipped and left untagged.")
    else:
        st.error(
            "No annotation files matched. Check the GeoJSON directory, and the suffix setting if the "
            "cell-table names carry one."
        )
    with st.expander("Cell table ↔ annotation matching", expanded=matched < len(sample_ids)):
        st.dataframe(match_table, hide_index=True, width="stretch")

    all_classes = collect_classes(sample_ids, geojson_dir, source_labels)
    if not all_classes:
        return

    # --- Step 3: coordinates ---
    st.markdown("#### Step 3 — Which columns hold the cell coordinates?")
    st.caption(
        "Coordinates must be in full-resolution image pixels, the same space the GeoJSON polygons use. "
        "Nothing is hard-coded: pick whichever columns this table uses."
    )
    x_col, y_col = _render_coordinate_pickers(columns)

    # --- Step 4: tissue label ---
    st.markdown("#### Step 4 — Which annotation marks a piece of tissue?")
    saved_tissue = st.session_state.get(TISSUE_LABEL_KEY)
    if saved_tissue not in all_classes:
        saved_tissue = next(
            (name for name in all_classes if name.lower() == DEFAULT_TISSUE_LABEL), all_classes[0]
        )
    tissue_label = st.selectbox(
        "Containing tissue annotation",
        all_classes,
        index=all_classes.index(saved_tissue),
        key=TISSUE_LABEL_KEY,
        help=(
            "The annotation that defines one piece of tissue. Every separate polygon of it is one piece, "
            "numbered by size. Rules with 'Group by ... piece' are evaluated inside the piece a cell sits in."
        ),
    )
    _render_column_meanings(tissue_label)

    # --- Step 5: rules ---
    st.markdown("#### Step 5 — Columns to add for each annotation")
    _render_rules_form(all_classes, tissue_label)

    # --- Step 6: output ---
    st.markdown("#### Step 6 — Output")
    overwrite_columns = st.checkbox(
        "Overwrite existing annotation columns (replace entire columns)",
        value=bool(st.session_state.get(OVERWRITE_COLUMNS_KEY, True)),
        key="seg_tag_overwrite_columns_widget",
        help=(
            "On: replace every value in each selected output column, including blanks. "
            "In merged tables, rows with missing or failed annotations are cleared too. "
            "Off: keep existing columns unchanged and add only missing columns. "
            "Columns not selected in the saved rules are unchanged."
        ),
    )
    st.session_state[OVERWRITE_COLUMNS_KEY] = bool(overwrite_columns)
    directory_input_with_browse(
        label="Output directory",
        text_key=OUTPUT_DIR_KEY,
        default_value=st.session_state.get(OUTPUT_DIR_KEY, "")
        or str(project_root / "outputs" / "segmentation" / "tagged_features"),
        browser_key_prefix="seg_tag_output_dir_browse",
        help_text="Tagged cell tables are written here. The originals are never modified.",
    )
    output_dir = st.session_state.get(OUTPUT_DIR_KEY, "")
    output_suffix = st.text_input(
        "Output file suffix",
        value=str(st.session_state.get(OUTPUT_SUFFIX_KEY, DEFAULT_OUTPUT_SUFFIX)),
        key="seg_tag_output_suffix_widget",
        help=(
            "Appended to the input file name to name the output, so "
            "sample_1_cell_features.csv becomes sample_1_cell_features_tagged.csv. Keep it non-empty so a "
            "tagged table can never overwrite the segmentation output it came from."
        ),
    )
    st.session_state[OUTPUT_SUFFIX_KEY] = output_suffix
    if not str(output_suffix).strip():
        st.error("The output suffix cannot be empty — that would overwrite the input cell tables.")
        return
    if not output_dir:
        st.warning("Set an output directory before running.")
        return

    rules, edge_config, _, _minimum_bed_cells, _clean_path = build_engine_inputs()
    beds = selected_bed_labels()
    if len(beds) == 1:
        st.info("Select at least two bed annotations and save the tagging rules before running.")
        return
    if not any(rule.is_active() for rule in rules) and edge_config is None and not beds:
        st.info("Configure and save at least one rule in step 5 before running.")
        return

    # --- run ---
    if st.button("Run cell tagging", type="primary", width="stretch", key="seg_tag_run"):
        from spacemap_gui.persistence import save_active_project_state

        save_active_project_state("tab5_segmentation")
        status = st.empty()
        progress_bar = st.progress(0.0)

        def progress(done: int, total: int, name: str) -> None:
            progress_bar.progress(min(done / max(total, 1), 1.0))
            method = "Triangle visibility (CPU)" if st.session_state.get(CLEAN_PATH_KEY, False) else "Euclidean"
            status.write(f"Tagging {done}/{total}: {name} · {method}")

        try:
            with st.spinner("Tagging cells..."):
                if mode == MODE_PER_IMAGE:
                    records = run_per_image(
                        csv_dir, csv_names, geojson_dir, output_dir,
                        str(st.session_state.get(CSV_SUFFIX_KEY, DEFAULT_CSV_SUFFIX)),
                        str(output_suffix), x_col, y_col, progress=progress,
                    )
                    outputs = [
                        os.path.join(output_dir, record["Output"])
                        for record in records if record.get("Output")
                    ]
                else:
                    stem = os.path.splitext(os.path.basename(merged_path))[0]
                    out_path = os.path.join(output_dir, f"{stem}{output_suffix}.csv")
                    _, records = run_merged(
                        merged_path, batch_column, geojson_dir, out_path,
                        x_col, y_col, progress=progress,
                    )
                    outputs = [out_path]
            st.session_state[RESULT_KEY] = {"records": records, "outputs": outputs}
            progress_bar.empty()
            status.empty()
        except Exception as error:
            progress_bar.empty()
            status.empty()
            st.error(f"Cell tagging failed: {error}")

    _render_results()
