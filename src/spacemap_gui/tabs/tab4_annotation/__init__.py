"""Tab 4: Annotation - unsupervised tissue macro-structure discovery.

The tab walks one ROI image through four sequential steps, each in its own
subtab and its own module:

1. ``s1_tissue_detection``     - virtual H&E optical-density tissue masking.
2. ``s2_artifact_removal``     - blur, flat-bright glare and filamentous lint.
2b. ``s2b_tissue_pieces``      - split the valid tissue into separate pieces,
   report narrow bridges, and drop pieces too small for the analysis radius.
3. ``s3_pixel_classification`` - marker normalization, pixel clustering and
   interactive merging of clusters into biological domains.
4. ``s4_pool_and_zones``       - pooling domains into higher-level polygons and
   building inner/outer buffer zones.

Between any two steps the intermediate GeoJSON can be opened and refined in
QuPath: every step re-reads its input from disk rather than from cached state.

The fifth subtab keeps the previous cell-level annotation workflow available
until it moves to the Segmentation tab.
"""

__package__ = "spacemap_gui.tabs.tab4_annotation"

import streamlit as st

from .common import ensure_state, render_directory_block, render_sample_selector


SUBTAB_LABELS = (
    "1. Tissue Detection",
    "2. Artifact Removal",
    "2b. Tissue Pieces",
    "3. Pixel Classification",
    "4. Pool & Zones",
    "5. Cell Annotation (legacy)",
)

_LOADED_KEY = "_annot_ta_loaded_subtabs"


def _is_loaded(name: str) -> bool:
    return bool(st.session_state.get(_LOADED_KEY, {}).get(name, False))


def _mark_loaded(name: str) -> None:
    st.session_state.setdefault(_LOADED_KEY, {})[name] = True


def render() -> None:
    from pathlib import Path

    from spacemap_gui.ui_utils import require_active_project

    project_root = require_active_project()
    if project_root is None:
        return

    st.session_state["_currently_rendering_tab"] = "tab4"
    ensure_state(project_root)

    st.markdown("## Step 4: Tissue Annotation")
    st.markdown(
        "Discover tissue macro-structures directly from the imaging data: detect valid tissue, remove "
        "acquisition artifacts, classify the remaining pixels, and export reviewed tissue annotations."
    )

    with st.expander("How this tab works", expanded=False):
        st.markdown(
            """
- **Step 1** builds the tissue mask from the virtual H&E channels and writes `*_tissue_boundary.geojson`.
- **Step 2** subtracts out-of-focus regions, flat bright glare and filamentous lint, writing
  `*_valid_tissue.geojson` and `*_detected_lint.geojson`.
- **Step 2b** splits that valid tissue into its physically separate pieces, measures each one, reports
  narrow bridges (necks) inside a piece, and drops pieces too small to support the analysis radius. It
  writes `*_tissue_pieces.geojson`, which is a drop-in replacement for the valid-tissue file downstream.
- **Step 3** normalizes the marker stack over the reviewed valid tissue, clusters the pixels, and lets you
  click the size-aware balanced heatmap to merge clusters into domains such as Epithelia, Stroma or
  Endothelia. It writes `*_raw_macro_clusters.geojson`, `*_valid_tissue_from_clusters_highres.geojson` and
  `*_macro_domains.geojson`.
- **Step 4** pools domains into higher-level polygons (for example a single `tissue` polygon) and adds
  inner/outer buffer zones, writing `*_zones.geojson`.

Every step reads its input GeoJSON from disk, so you can refine any intermediate file in QuPath, save it
under the same name, and continue. Steps 1, 2 and 4 also run in batch across several images; step 3 is
per-image because cluster identities and merge decisions are image-specific.

Parameters are edited in forms and only take effect when you press their Save button, so browsing settings
never triggers a recomputation. Saved settings go into the project JSON and reload automatically.
            """
        )

    # Directories and the working image are chosen once for the whole tab.
    # Streamlit executes every subtab in the same script run, so these widgets
    # cannot live inside the subtabs without colliding on their keys.
    with st.expander("Directories and image selection", expanded=True):
        render_directory_block(include_pool_dirs=True)
        st.markdown("---")
        sample_id, batch_samples = render_sample_selector()

    if not sample_id:
        st.info("Select an input directory containing ROI images to begin.")
        return

    tabs = st.tabs(list(SUBTAB_LABELS))

    with tabs[0]:
        from . import s1_tissue_detection

        s1_tissue_detection.render(project_root, sample_id, batch_samples)

    with tabs[1]:
        from . import s2_artifact_removal

        s2_artifact_removal.render(project_root, sample_id, batch_samples)

    with tabs[2]:
        from . import s2b_tissue_pieces

        s2b_tissue_pieces.render(project_root, sample_id, batch_samples)

    with tabs[3]:
        if not _is_loaded("s3"):
            st.info(
                "Pixel classification loads clustering libraries on demand. Click below to open this step."
            )
            if st.button("Load Pixel Classification", key="annot_ta_load_s3"):
                _mark_loaded("s3")
                st.rerun()
        if _is_loaded("s3"):
            from . import s3_pixel_classification

            s3_pixel_classification.render(project_root, sample_id)

    with tabs[4]:
        from . import s4_pool_and_zones

        s4_pool_and_zones.render(project_root, sample_id, batch_samples)

    with tabs[5]:
        st.caption(
            "Cell-level annotation from existing GeoJSON files. This workflow is unchanged and stays here "
            "until it moves into the Segmentation tab."
        )
        if not _is_loaded("legacy"):
            st.info("Click below to load the cell annotation workflow.")
            if st.button("Load Cell Annotation", key="annot_ta_load_legacy"):
                _mark_loaded("legacy")
                st.rerun()
        if _is_loaded("legacy"):
            from spacemap_gui.tabs import legacy_cell_annotation

            legacy_cell_annotation.render()


if __name__ == "__main__":
    from spacemap_gui.ui_utils import require_active_project

    require_active_project()
    render()
