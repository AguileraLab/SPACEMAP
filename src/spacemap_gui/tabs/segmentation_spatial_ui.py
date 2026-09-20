"""Tissue-mask and spatial-annotation controls for the Segmentation tab.

The Annotation tab now runs *before* Segmentation, so each ROI image usually has
a reviewed annotation GeoJSON next to it (``P11_roi_1.ome.tiff`` ->
``P11_roi_1_zones.geojson``). This module turns that file into two things:

1. **A segmentation mask.** One class of the annotation (``tissue`` by default)
   restricts Cellpose to the reviewed, artifact-free tissue, so no GPU time is
   spent on background and no cells are called in blurred or lint regions.
2. **Spatial columns on the cell table.** The remaining classes (Epithelia,
   Stroma, Tumor, TLS, zones, ...) are written onto the extracted cells in the
   same pass as feature extraction, so the CSV that lands on disk is already
   ready for batch correction.

Kept out of ``tab5_segmentation.py`` because that module is already large; the
tab calls :func:`render_spatial_panel` and :func:`build_spatial_config`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import streamlit as st

from spacemap_gui.ui_utils import directory_input_with_browse, file_signature

# Session-state keys. The ``tab4_`` prefix is what persistence maps to the
# Segmentation page, so these settings are saved into <project>.json.
GEOJSON_DIR_KEY = "tab4_annot_geojson_dir"
MASK_SOURCE_KEY = "tab4_mask_source"
MASK_CLASSES_KEY = "tab4_mask_classes"
SPATIAL_ENABLED_KEY = "tab4_spatial_enabled"
SPATIAL_RULES_KEY = "tab4_spatial_rules"
TISSUE_LABEL_KEY = "tab4_spatial_tissue_label"
EDGE_ENABLED_KEY = "tab4_spatial_edge_enabled"
EDGE_REFERENCE_KEY = "tab4_spatial_edge_reference"
EDGE_BUFFER_KEY = "tab4_spatial_edge_buffer"
EDGE_COLUMN_KEY = "tab4_spatial_edge_column"
MIN_BED_CELLS_KEY = "tab4_spatial_min_cells_per_bed_target"
CLEAN_PATH_KEY = "tab4_spatial_require_clean_path"

MASK_SOURCE_NONE = "No mask (whole image)"
MASK_SOURCE_ANNOTATION = "Annotation GeoJSON (Annotation tab)"
MASK_SOURCE_DRAWN = "Drawn tissue mask (this tab)"
MASK_SOURCES = (MASK_SOURCE_NONE, MASK_SOURCE_ANNOTATION, MASK_SOURCE_DRAWN)

DEFAULT_TISSUE_LABEL = "tissue"
DEFAULT_MIN_CELLS_PER_BED_TARGET = 10

RULE_COLUMNS = [
    "Annotation Label", "Binary (0/1)", "ROI ID", "ROI Size", "Distance",
    "Bed (Use 2 Annotations)", "Bed Comp. Target", "Group by Tissue ROI",
]


# ==========================================================================
# Matching and class discovery
# ==========================================================================

def matched_geojson(image_name: str, geojson_dir: str) -> str | None:
    """Annotation GeoJSON matching one image by name, or None."""
    from spacemap.segmentation_cellpose import find_annotation_geojson

    if not geojson_dir:
        return None
    return find_annotation_geojson(str(image_name), str(geojson_dir))


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_classes(geojson_path: str, signature: tuple[int, int]) -> list[str]:
    from spacemap.cell_spatial_annotation import list_annotation_classes

    return list_annotation_classes(geojson_path)


def geojson_classes(geojson_path: str | None) -> list[str]:
    """Classification names in an annotation GeoJSON, re-read whenever it changes.

    ``signature`` must stay a plain argument: ``st.cache_data`` drops
    underscore-prefixed parameters from the cache key, so passing the timestamp
    as ``_mtime`` pinned every path to the first version read - editing a
    GeoJSON in place had no effect until it was moved to a different folder.
    """
    if not geojson_path or not os.path.isfile(geojson_path):
        return []
    return _cached_classes(geojson_path, file_signature(geojson_path))


def build_match_table(image_names: Sequence[str], geojson_dir: str) -> pd.DataFrame:
    """One row per image: which annotation file matched and what it contains."""
    rows = []
    for name in image_names:
        path = matched_geojson(name, geojson_dir)
        classes = geojson_classes(path)
        rows.append({
            "Image": name,
            "Matched annotation": os.path.basename(path) if path else "— none —",
            "Classes": ", ".join(classes) if classes else "",
            "Class count": len(classes),
        })
    return pd.DataFrame(rows)


def default_mask_classes(classes: Sequence[str]) -> list[str]:
    """Prefer a class literally named ``tissue``; otherwise leave it to the user."""
    for name in classes:
        if str(name).lower() == DEFAULT_TISSUE_LABEL:
            return [name]
    return []


# ==========================================================================
# Rules state
# ==========================================================================

def normalize_rule_records(records: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        label = record.get("Annotation Label", "")
        if not label:
            continue
        normalized.append({
            "Annotation Label": str(label),
            "Binary (0/1)": bool(record.get("Binary (0/1)", True)),
            "ROI ID": bool(record.get("ROI ID", False)),
            "ROI Size": bool(record.get("ROI Size", False)),
            "Distance": bool(record.get("Distance", False)),
            "Bed (Use 2 Annotations)": bool(record.get("Bed (Use 2 Annotations)", False)),
            "Bed Comp. Target": str(record.get("Bed Comp. Target", "") or ""),
            "Group by Tissue ROI": bool(record.get("Group by Tissue ROI", False)),
        })
    return normalized


def _align_rules(classes: Sequence[str], tissue_label: str) -> list[dict[str, Any]]:
    """Keep saved settings for classes that still exist and add any new ones."""
    saved = {
        record["Annotation Label"].lower(): record
        for record in normalize_rule_records(st.session_state.get(SPATIAL_RULES_KEY, []))
    }
    aligned: list[dict[str, Any]] = []
    for name in classes:
        existing = saved.get(str(name).lower())
        is_tissue = str(name).lower() == str(tissue_label).lower()
        if existing is not None:
            existing["Annotation Label"] = str(name)
            existing["Group by Tissue ROI"] = not is_tissue
            aligned.append(existing)
            continue
        aligned.append({
            "Annotation Label": str(name),
            "Binary (0/1)": True,
            "ROI ID": is_tissue,
            "ROI Size": is_tissue,
            "Distance": is_tissue,
            "Bed (Use 2 Annotations)": False,
            "Bed Comp. Target": "",
            "Group by Tissue ROI": not is_tissue,
        })
    return aligned


# ==========================================================================
# UI
# ==========================================================================

def render_spatial_panel(project_root: Path, image_names: Sequence[str], input_dir: str) -> None:
    """Render the mask source, class picker and spatial-column configuration."""
    from spacemap_gui.persistence import save_active_project_state

    st.caption(
        "The Annotation tab writes one reviewed GeoJSON per ROI. Use it here to keep segmentation "
        "inside real tissue and to tag every cell with the tissue domain it sits in."
    )

    directory_input_with_browse(
        label="Annotation GeoJSON directory",
        text_key=GEOJSON_DIR_KEY,
        default_value=st.session_state.get(GEOJSON_DIR_KEY, "")
        or str(project_root / "outputs" / "processed_annotations" / "stage1b_compartments_zones"),
        browser_key_prefix="tab4_annot_geojson_dir_browse",
        help_text=(
            "Folder holding the annotation files produced by step 4 of the Annotation tab "
            "(for example P11_roi_1_zones.geojson). Files are matched to images by name."
        ),
    )
    geojson_dir = st.session_state.get(GEOJSON_DIR_KEY, "")

    saved_source = st.session_state.get(MASK_SOURCE_KEY, MASK_SOURCE_NONE)
    if saved_source not in MASK_SOURCES:
        saved_source = MASK_SOURCE_NONE
    st.selectbox(
        "Segmentation region",
        MASK_SOURCES,
        index=MASK_SOURCES.index(saved_source),
        key=MASK_SOURCE_KEY,
        help=(
            "No mask segments the whole image. Annotation GeoJSON restricts Cellpose to a chosen class "
            "of the reviewed annotation, which is the recommended route now that the Annotation tab runs "
            "first. Drawn tissue mask uses the masks drawn in this tab's editor."
        ),
    )
    mask_source = st.session_state.get(MASK_SOURCE_KEY, MASK_SOURCE_NONE)

    # --- matching overview ---
    match_table = build_match_table(image_names, geojson_dir)
    matched_count = int((match_table["Matched annotation"] != "— none —").sum()) if not match_table.empty else 0
    if image_names:
        if matched_count == len(image_names):
            st.success(f"Matched an annotation file for all {len(image_names)} image(s).")
        elif matched_count:
            st.warning(f"Matched {matched_count} of {len(image_names)} image(s). Unmatched images run without a mask or spatial columns.")
        else:
            st.info("No annotation files matched. Check the directory above, or run the Annotation tab first.")
    with st.expander("Image ↔ annotation matching", expanded=matched_count < len(image_names)):
        st.dataframe(match_table, hide_index=True, width="stretch")

    # Union of classes across matched files drives the pickers.
    all_classes: list[str] = []
    for name in image_names:
        for class_name in geojson_classes(matched_geojson(name, geojson_dir)):
            if class_name not in all_classes:
                all_classes.append(class_name)

    if not all_classes:
        st.session_state.setdefault(MASK_CLASSES_KEY, [])
        return

    # --- mask class ---
    if mask_source == MASK_SOURCE_ANNOTATION:
        saved_classes = [name for name in st.session_state.get(MASK_CLASSES_KEY, []) if name in all_classes]
        if not saved_classes:
            saved_classes = default_mask_classes(all_classes)
        stored = st.session_state.get("tab4_mask_classes_widget")
        if isinstance(stored, list) and any(name not in all_classes for name in stored):
            st.session_state["tab4_mask_classes_widget"] = [n for n in stored if n in all_classes]
        chosen = st.multiselect(
            "Polygon class used as the segmentation mask",
            all_classes,
            default=saved_classes,
            key="tab4_mask_classes_widget",
            help=(
                "Only cells inside these polygons are segmented. `tissue` is the reviewed, artifact-free "
                "region from the Annotation tab and is the usual choice. Selecting several classes unions them. "
                "Classes named artefact/exclude/necrosis are always cut out."
            ),
        )
        st.session_state[MASK_CLASSES_KEY] = chosen
        if not chosen:
            st.warning("Select at least one class, or set the segmentation region to 'No mask'.")

    # --- spatial columns ---
    st.markdown("**Spatial columns on the cell table**")
    st.checkbox(
        "Tag cells with their tissue domain during feature extraction",
        value=bool(st.session_state.get(SPATIAL_ENABLED_KEY, True)),
        key=SPATIAL_ENABLED_KEY,
        help=(
            "Adds the domain columns to the features CSV in the same pass as segmentation, so no separate "
            "annotation step is needed before batch correction."
        ),
    )
    if not st.session_state.get(SPATIAL_ENABLED_KEY, True):
        return

    tissue_options = list(all_classes)
    saved_tissue = st.session_state.get(TISSUE_LABEL_KEY)
    if saved_tissue not in tissue_options:
        saved_tissue = next(
            (name for name in tissue_options if name.lower() == DEFAULT_TISSUE_LABEL),
            tissue_options[0],
        )
    st.markdown("**Step 1 — which annotation marks a piece of tissue?**")
    st.selectbox(
        "Containing tissue annotation",
        tissue_options,
        index=tissue_options.index(saved_tissue),
        key=TISSUE_LABEL_KEY,
        help=(
            "The annotation that defines a tissue piece. Every separate polygon of it is one piece, and rules "
            "with 'Group by tissue piece' are evaluated inside the piece a cell belongs to, so a cell is never "
            "measured against a compartment in an unrelated tissue region."
        ),
    )
    tissue_label = st.session_state.get(TISSUE_LABEL_KEY, DEFAULT_TISSUE_LABEL)
    st.caption(
        f"Each separate `{tissue_label}` polygon counts as one tissue piece, numbered by size in "
        f"`{tissue_label}_roi`. Every subannotation row below is then measured inside a cell's own piece "
        "only — grouping always uses this annotation, never the row's own polygons."
    )

    with st.expander("Column meanings", expanded=False):
        st.markdown(
            f"""
| Setting | Column added | Meaning |
|---|---|---|
| Binary (0/1) | `<label>` | 1 when the cell centroid is inside that annotation |
| ROI ID | `<label>_roi` | Which individual polygon of that annotation contains the cell (largest = 1) |
| ROI Size | `<label>_roi_size` | Area of that containing polygon, in square pixels |
| Distance | `<label>_distance` | Signed distance to the nearest polygon: **negative inside, positive outside** |
| Bed | `<label>_bed`, `<target>_bed` (+ `_bed_id`) | Proximity competition: each cell is assigned to whichever of the two annotations it is closer to |
| Group by {tissue_label} piece | — | Automatically on for every subannotation: everything is evaluated inside the cell's own `{tissue_label}` piece |

**Group by {tissue_label} piece** is enforced for every subannotation and always groups by the
*containing tissue annotation* selected above, never by the row's own polygons. Therefore:

* `<label>_roi` is numbered **per {tissue_label} piece**, so one polygon spanning two pieces becomes a
  separate ROI on each side and cells in different pieces never share an id.
* `<label>_roi_size` is the area inside that piece rather than the whole polygon.
* Distance and bed only see polygons in the cell's own piece, so a cell is never measured against a
  compartment in an unrelated piece of tissue.
* A cell whose `{tissue_label}` piece contains none of that annotation gets `NaN` for its distance.
* A sub-annotation crossing two tissue pieces is clipped into independent fragments with independent
  ROI and bed IDs on each side.
            """
        )

    aligned = _align_rules(all_classes, tissue_label)
    rules_frame = pd.DataFrame(aligned, columns=RULE_COLUMNS)

    st.markdown("**Step 2 — columns to add for each annotation**")
    with st.form("tab4_spatial_rules_form"):
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
                    "Distance", help="Add <label>_distance: negative inside the annotation, positive outside.", width="small"),
                "Bed (Use 2 Annotations)": st.column_config.CheckboxColumn(
                    "Bed", help="Split cells by which of two annotations they are closer to.", width="small"),
                "Bed Comp. Target": st.column_config.SelectboxColumn(
                    "Bed target", options=[""] + all_classes,
                    help="The competing annotation, for example Malignant against Normal.", width="medium"),
                "Group by Tissue ROI": st.column_config.CheckboxColumn(
                    f"Group by {tissue_label} piece",
                    disabled=True,
                    help=(
                        f"Automatically evaluate every subannotation inside the '{tissue_label}' polygon "
                        "containing the cell — NOT inside this row's own polygons. Splits <label>_roi per piece, "
                        "clips <label>_roi_size to it, and "
                        "keeps distance and bed from reaching across to an unrelated piece of tissue. Cells in a "
                        f"'{tissue_label}' piece holding none of this annotation get NaN distance."
                    ),
                    width="medium"),
            },
            hide_index=True,
            num_rows="fixed",
            key="tab4_spatial_rules_editor",
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
            key="tab4_spatial_min_cells_per_bed_target_widget",
            help=(
                f"Each annotation polygon is first clipped to its containing {tissue_label} piece. "
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
            key="tab4_spatial_clean_path_widget",
            help=(
                "On a section made of several separate scraps of biopsy, a straight line to the nearest "
                "annotation can cross bare glass into another scrap, or cut across the mouth of a folded "
                "piece. With this on, a distance is written only when the line from the cell to the "
                "annotation stays inside the cell's own tissue piece; cells with no such line are left "
                "blank instead of being given a longer substitute. Affects distance columns only."
            ),
        )
        st.caption(
            "Leave off to reproduce earlier runs exactly. Turning it on adds no columns - it only blanks "
            "distances that could not be measured honestly, and re-points others at a reachable target."
        )

        st.markdown("**Edge cell tagging**")
        edge_columns = st.columns([1, 1, 1])
        edge_enabled = edge_columns[0].checkbox(
            "Flag cells near an edge",
            value=bool(st.session_state.get(EDGE_ENABLED_KEY, False)),
            key="tab4_spatial_edge_enabled_widget",
            help="Adds a 0/1 column marking cells within a set distance of a reference annotation's boundary.",
        )
        edge_reference_options = list(all_classes)
        saved_reference = st.session_state.get(EDGE_REFERENCE_KEY, tissue_label)
        if saved_reference not in edge_reference_options:
            saved_reference = tissue_label
        edge_reference = edge_columns[0].selectbox(
            "Reference annotation", edge_reference_options,
            index=edge_reference_options.index(saved_reference),
            key="tab4_spatial_edge_reference_widget",
            help="Boundary the distance is measured from.",
        )
        edge_buffer = edge_columns[1].number_input(
            "Edge buffer (px)", min_value=0.0, max_value=100000.0, step=1.0,
            value=float(st.session_state.get(EDGE_BUFFER_KEY, 100.0)),
            key="tab4_spatial_edge_buffer_widget",
            help="Cells closer than this to the boundary are flagged.",
        )
        edge_column = edge_columns[2].text_input(
            "Column name", value=str(st.session_state.get(EDGE_COLUMN_KEY, "edge_cell")),
            key="tab4_spatial_edge_column_widget",
            help="Name of the 0/1 output column.",
        )
        saved = st.form_submit_button("Save spatial column settings", width="stretch")

    if saved:
        st.session_state[SPATIAL_RULES_KEY] = normalize_rule_records(edited.to_dict("records"))
        st.session_state[EDGE_ENABLED_KEY] = bool(edge_enabled)
        st.session_state[EDGE_REFERENCE_KEY] = str(edge_reference)
        st.session_state[EDGE_BUFFER_KEY] = float(edge_buffer)
        st.session_state[EDGE_COLUMN_KEY] = str(edge_column)
        st.session_state[MIN_BED_CELLS_KEY] = int(minimum_bed_cells)
        st.session_state[CLEAN_PATH_KEY] = bool(require_clean_path)
        save_active_project_state("tab5_segmentation")
        st.success("Spatial column settings saved to the project configuration.")

    preview = preview_columns(
        normalize_rule_records(edited.to_dict("records")),
        bool(edge_enabled), str(edge_column),
    )
    if preview:
        st.caption(f"Columns that will be added ({len(preview)}): " + ", ".join(f"`{name}`" for name in preview))


def preview_columns(records: Sequence[dict[str, Any]], edge_enabled: bool, edge_column: str) -> list[str]:
    from spacemap.cell_spatial_annotation import EdgeTagConfig, rules_from_records, spatial_columns_for_rules

    rules = rules_from_records(records)
    edge = EdgeTagConfig(column=edge_column) if edge_enabled else None
    return spatial_columns_for_rules(rules, edge)


def render_mask_overlay(image_path: str, image_name: str) -> None:
    """Show the chosen mask polygon on the image thumbnail."""
    geojson_dir = st.session_state.get(GEOJSON_DIR_KEY, "")
    path = matched_geojson(image_name, geojson_dir)
    if not path:
        st.info("No annotation GeoJSON matched this image, so no mask preview is available.")
        return

    classes = geojson_classes(path)
    mask_classes = [name for name in st.session_state.get(MASK_CLASSES_KEY, []) if name in classes]
    if not mask_classes:
        mask_classes = default_mask_classes(classes)
    if not mask_classes:
        st.info("Select a mask class to preview it on the image.")
        return

    try:
        from spacemap_gui.tabs.tab4_annotation.common import overlay_polygons_on_thumbnail, render_class_legend

        image, colors = overlay_polygons_on_thumbnail(
            image_path, [path],
            class_colors={name: "#00e676" for name in mask_classes},
            class_filter=mask_classes,
            fill_alpha=60,
        )
        st.image(image, width="stretch", caption=f"{os.path.basename(path)} — {', '.join(mask_classes)}")
        render_class_legend(colors)
        st.caption("Segmentation runs only inside the highlighted region.")
    except Exception as error:
        st.warning(f"Could not render the mask preview: {error}")


# ==========================================================================
# Configuration handed to the batch runner
# ==========================================================================

def build_spatial_config(image_name: str) -> dict[str, Any]:
    """Per-image mask and annotation settings for :func:`segment_image_cellpose`.

    Returns a dict with ``geojson_path``, ``mask_classes``, ``rules``,
    ``tissue_label``, ``min_cells_per_bed_target`` and ``edge_config``; empty values mean "not configured",
    which the runner treats as no mask / no spatial columns.
    """
    geojson_dir = st.session_state.get(GEOJSON_DIR_KEY, "")
    path = matched_geojson(image_name, geojson_dir)
    classes = geojson_classes(path)

    mask_classes: list[str] = []
    if st.session_state.get(MASK_SOURCE_KEY) == MASK_SOURCE_ANNOTATION:
        mask_classes = [name for name in st.session_state.get(MASK_CLASSES_KEY, []) if name in classes]
        if not mask_classes:
            mask_classes = default_mask_classes(classes)

    rules: list[dict[str, Any]] = []
    edge_config: dict[str, Any] | None = None
    if st.session_state.get(SPATIAL_ENABLED_KEY, True):
        available = {str(name).lower() for name in classes}
        tissue_label = str(st.session_state.get(TISSUE_LABEL_KEY, DEFAULT_TISSUE_LABEL))
        for record in normalize_rule_records(st.session_state.get(SPATIAL_RULES_KEY, [])):
            if record["Annotation Label"].lower() not in available:
                continue
            scoped_record = dict(record)
            scoped_record["Group by Tissue ROI"] = (
                scoped_record["Annotation Label"].lower() != tissue_label.lower()
            )
            rules.append(scoped_record)
        if st.session_state.get(EDGE_ENABLED_KEY, False):
            edge_config = {
                "reference": st.session_state.get(EDGE_REFERENCE_KEY, DEFAULT_TISSUE_LABEL),
                "buffer": float(st.session_state.get(EDGE_BUFFER_KEY, 100.0)),
                "column": st.session_state.get(EDGE_COLUMN_KEY, "edge_cell"),
            }

    return {
        "geojson_path": path,
        "mask_classes": mask_classes,
        "rules": rules,
        "tissue_label": st.session_state.get(TISSUE_LABEL_KEY, DEFAULT_TISSUE_LABEL),
        "min_cells_per_bed_target": max(
            int(st.session_state.get(
                MIN_BED_CELLS_KEY, DEFAULT_MIN_CELLS_PER_BED_TARGET
            )),
            1,
        ),
        "edge_config": edge_config,
        "require_clean_path": bool(st.session_state.get(CLEAN_PATH_KEY, False)),
    }
