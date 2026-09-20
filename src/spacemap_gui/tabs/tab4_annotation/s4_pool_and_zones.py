"""Subtab 4 - pool merged domains into higher-level polygons and build zones.

This is the step that turns the discovered tissue domains into the annotation
layers the downstream cell annotation consumes: optional pooled polygons (for
example ``Epithelia + Stroma + Endothelia -> tissue``) plus inner/outer buffer
zones around chosen compartments.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from spacemap_gui.process_log import record_process_event

from .common import (
    TAB_CONTEXT,
    describe_output,
    log_annotation,
    overlay_polygons_on_thumbnail,
    qupath_hint,
    prune_widget_selection,
    render_class_legend,
    render_image_preview,
    resolve_image_path,
    save_project_settings,
    sticky_file_selectbox,
)

POOL_RULES_KEY = "annot_ta_pool_rules"
ZONE_RULES_KEY = "annot_ta_zone_rules"
CONFIG_KEY = "annot_ta_pool_config"
KEEP_SOURCES_KEY = "annot_ta_pool_keep_sources"
CLIP_CLASS_KEY = "annot_ta_zone_clip_class"

OUTPUT_SUFFIX = "zones"


def _source_directory() -> str:
    return st.session_state.get("annot_ta_pool_input_dir", "") or st.session_state.get("annot_ta_output_dir", "")


def _output_directory() -> str:
    return st.session_state.get("annot_ta_pool_output_dir", "")


def _candidate_files(sample_id: str) -> list[str]:
    directory = _source_directory()
    if not directory or not os.path.isdir(directory):
        return []
    names = sorted(name for name in os.listdir(directory) if name.lower().endswith(".geojson"))
    preferred = [f"{sample_id}_macro_domains.geojson"]
    ordered = [name for name in preferred if name in names]
    ordered += [name for name in names if name not in ordered and name.startswith(sample_id)]
    ordered += [name for name in names if name not in ordered]
    return ordered


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_class_names(source_path: str, mtime: float) -> list[str]:
    from spacemap.tissue_annotation_utils import read_geojson_class_names

    return read_geojson_class_names(source_path)


def _class_names(source_path: str) -> list[str]:
    """Classification names in a GeoJSON, keyed on its modification time.

    Still effectively "always from disk" - a QuPath save changes the timestamp
    and invalidates the entry - but a large file is not re-parsed on every
    rerun of the page.
    """
    if not source_path or not os.path.isfile(source_path):
        return []
    return _cached_class_names(source_path, os.path.getmtime(source_path))


def _pool_rules() -> list[dict]:
    rules = st.session_state.get(POOL_RULES_KEY, []) or []
    normalized = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        normalized.append({
            "name": str(rule.get("name", "")),
            "sources": [str(value) for value in rule.get("sources", [])],
            "subtract": [str(value) for value in rule.get("subtract", [])],
        })
    return normalized


def _zone_rules() -> list[dict]:
    rules = st.session_state.get(ZONE_RULES_KEY, []) or []
    normalized = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        normalized.append({
            "label": str(rule.get("label", "")),
            "width_px": float(rule.get("width_px", 100)),
            "inner": int(rule.get("inner", 0)),
            "outer": int(rule.get("outer", 0)),
        })
    return normalized


def _run_single(sample_id: str, source_filename: str) -> dict:
    """Pool classes and append zones for one sample."""
    from spacemap.tissue_annotation_utils import (
        _iter_geojson_features,
        generate_zone_features,
        pool_geojson_classes,
        write_feature_collection,
    )

    source_path = os.path.join(_source_directory(), source_filename)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source GeoJSON not found: {source_path}")

    output_dir = _output_directory()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sample_id}_{OUTPUT_SUFFIX}.geojson")

    pool_result = pool_geojson_classes(
        source_path,
        _pool_rules(),
        output_path,
        keep_source_classes=bool(st.session_state.get(KEEP_SOURCES_KEY, True)),
        config=st.session_state.get(CONFIG_KEY, {}),
        class_colors=st.session_state.get("annot_ta_domain_colors", {}),
    )

    zone_rules = [rule for rule in _zone_rules() if rule["inner"] > 0 or rule["outer"] > 0]
    zone_features: list[dict] = []
    if zone_rules:
        clip_class = st.session_state.get(CLIP_CLASS_KEY) or None
        if clip_class in ("", "(none)"):
            clip_class = None
        # Zones are built from the pooled file so a freshly pooled class can be a zone source.
        zone_features = generate_zone_features(
            output_path, zone_rules, clip_class=clip_class, config=st.session_state.get(CONFIG_KEY, {})
        )
        if zone_features:
            features = _iter_geojson_features(output_path)
            features.extend(zone_features)
            write_feature_collection(features, output_path)

    return {
        "output_path": output_path,
        "pooled": pool_result["created"],
        "zone_count": len(zone_features),
        "source_path": source_path,
    }


def render(project_root: Path, sample_id: str, batch_samples: list[str]) -> None:
    st.markdown("### Step 4 - Pool domains and create zones")
    st.caption(
        "Loads the merged-domain GeoJSON from disk, optionally pools several domains into new named polygons "
        "(keeping the originals), and adds inner/outer buffer zones. The result is the annotation file the "
        "cell annotation step consumes."
    )


    candidates = _candidate_files(sample_id)
    if not candidates:
        st.warning(
            f"No GeoJSON files found in `{_source_directory() or '(not set)'}`. Run step 3 first, or point the "
            "domain directory at your QuPath-refined files."
        )
        return

    left, right = st.columns([1.35, 1.0])

    with left:
        st.markdown("#### Source annotation")
        source_filename = sticky_file_selectbox(
            label="Domain GeoJSON",
            candidates=candidates,
            preferred=f"{sample_id}_macro_domains.geojson",
            key=f"annot_ta_s4_source_file_{sample_id}",
            help_text=(
                "Defaults to the step 3 merged-domain export as soon as it exists. Pick another file if you "
                "refined it in QuPath. The file is re-read from disk on every render and on every run."
            ),
        )
        source_path = os.path.join(_source_directory(), source_filename)
        class_names = _class_names(source_path)
        if not class_names:
            st.warning("No classified annotations were found in the selected file.")
            return
        st.caption(f"Classes in this file: {', '.join(class_names)}")

        # ---------------- pooling ----------------
        st.markdown("#### Pooling rules")
        st.caption(
            "Each rule unions the selected source classes into one new polygon under a new name. "
            "For example pooling Epithelia, Stroma and Endothelia into `tissue` while keeping the originals."
        )
        rules = _pool_rules()
        add_column, keep_column = st.columns([1, 2])
        if add_column.button("+ Add pooling rule", key="annot_ta_s4_add_pool", width="stretch"):
            rules.append({"name": "", "sources": [], "subtract": []})
            st.session_state[POOL_RULES_KEY] = rules
            st.rerun()
        keep_sources = keep_column.checkbox(
            "Keep the original domain polygons in the output",
            value=bool(st.session_state.get(KEEP_SOURCES_KEY, True)),
            key="annot_ta_s4_keep_sources",
            help="Leave on to write both the pooled polygon and every source domain into the output file.",
        )
        st.session_state[KEEP_SOURCES_KEY] = keep_sources

        if rules:
            with st.form("annot_ta_s4_pool_form"):
                edited_rules = []
                removed = []
                for position, rule in enumerate(rules):
                    columns = st.columns([2, 3, 3, 1])
                    name = columns[0].text_input(
                        "Pooled name", value=rule["name"], key=f"annot_ta_s4_pool_name_{position}",
                        help="Classification name given to the pooled polygon, for example `tissue`.",
                    )
                    sources = columns[1].multiselect(
                        "Source classes",
                        class_names,
                        default=[value for value in rule["sources"] if value in class_names],
                        key=f"annot_ta_s4_pool_sources_{position}",
                        help="Classes unioned into the pooled polygon.",
                    )
                    subtract = columns[2].multiselect(
                        "Subtract classes",
                        class_names,
                        default=[value for value in rule["subtract"] if value in class_names],
                        key=f"annot_ta_s4_pool_subtract_{position}",
                        help="Optional classes cut out of the pooled polygon, for example necrosis or artifacts.",
                    )
                    remove = columns[3].checkbox("Del", key=f"annot_ta_s4_pool_remove_{position}", help="Delete this rule on save.")
                    edited_rules.append({"name": name, "sources": sources, "subtract": subtract})
                    removed.append(remove)
                saved_pool = st.form_submit_button("Save pooling rules", width="stretch")
            if saved_pool:
                st.session_state[POOL_RULES_KEY] = [
                    rule for rule, drop in zip(edited_rules, removed) if not drop and rule["name"].strip()
                ]
                save_project_settings()
                st.success("Pooling rules saved.")
                st.rerun()
        else:
            st.info("No pooling rules defined. The output will contain the source domains and any zones you configure.")

        # ---------------- zones ----------------
        st.markdown("#### Zone rules")
        st.caption(
            "Buffer zones around a compartment boundary. `outer_1..n` expand outward, `inner_1..n` erode inward. "
            "Zones at the same level are merged across all source labels, so `outer_1` is a single annotation."
        )
        poolable_names = sorted({rule["name"] for rule in _pool_rules() if rule["name"].strip()})
        zone_targets = list(dict.fromkeys(list(class_names) + poolable_names))
        existing_zone_rules = {rule["label"]: rule for rule in _zone_rules()}
        zone_table = pd.DataFrame([
            {
                "Annotation": label,
                "Zone width (px)": float(existing_zone_rules.get(label, {}).get("width_px", 100)),
                "Inner zones": int(existing_zone_rules.get(label, {}).get("inner", 0)),
                "Outer zones": int(existing_zone_rules.get(label, {}).get("outer", 0)),
            }
            for label in zone_targets
        ])

        clip_options = ["(none)"] + zone_targets
        saved_clip = st.session_state.get(CLIP_CLASS_KEY, "(none)")
        clip_index = clip_options.index(saved_clip) if saved_clip in clip_options else 0

        with st.form("annot_ta_s4_zone_form"):
            edited_zones = st.data_editor(
                zone_table,
                column_config={
                    "Annotation": st.column_config.TextColumn("Annotation", disabled=True, width="medium"),
                    "Zone width (px)": st.column_config.NumberColumn(
                        "Zone width (px)", min_value=1.0, max_value=100000.0, step=1.0,
                        help="Width of each buffer ring, in full-resolution pixels.",
                    ),
                    "Inner zones": st.column_config.NumberColumn(
                        "Inner zones", min_value=0, max_value=20, step=1,
                        help="Number of rings eroded inward from the boundary. 0 disables inner zones.",
                    ),
                    "Outer zones": st.column_config.NumberColumn(
                        "Outer zones", min_value=0, max_value=20, step=1,
                        help="Number of rings expanded outward from the boundary. 0 disables outer zones.",
                    ),
                },
                hide_index=True,
                num_rows="fixed",
                key="annot_ta_s4_zone_editor",
                width="stretch",
            )
            clip_class = st.selectbox(
                "Clip outer zones to",
                clip_options,
                index=clip_index,
                key="annot_ta_s4_clip_class",
                help=(
                    "Outer zones are intersected with this annotation so they never extend past the tissue. "
                    "Pick the pooled tissue polygon when you have one."
                ),
            )
            saved_zones = st.form_submit_button("Save zone rules", width="stretch")

        if saved_zones:
            st.session_state[ZONE_RULES_KEY] = [
                {
                    "label": str(row["Annotation"]),
                    "width_px": float(row["Zone width (px)"]),
                    "inner": int(row["Inner zones"]),
                    "outer": int(row["Outer zones"]),
                }
                for _, row in edited_zones.iterrows()
                if int(row["Inner zones"]) > 0 or int(row["Outer zones"]) > 0
            ]
            st.session_state[CLIP_CLASS_KEY] = clip_class
            save_project_settings()
            st.success("Zone rules saved.")
            st.rerun()

        # ---------------- run ----------------
        st.markdown("#### Run")
        run_column, batch_column = st.columns(2)
        run_single = run_column.button("Pool and zone (this image)", type="primary", width="stretch", key="annot_ta_s4_run")
        run_batch = batch_column.button(
            f"Run batch ({len(batch_samples)} images)", width="stretch", key="annot_ta_s4_run_batch",
            disabled=not batch_samples,
            help=(
                "Applies the same pooling and zone rules to every selected image, using each image's own "
                "domain GeoJSON. Rules referencing a class an image does not have are skipped for that image."
            ),
        )

        if run_single or run_batch:
            targets = batch_samples if run_batch else [sample_id]
            progress_bar = st.progress(0.0)
            status = st.empty()
            started = time.time()
            record_process_event(
                TAB_CONTEXT, "Pooling and zones started",
                details={"samples": len(targets), "pool_rules": len(_pool_rules()), "zone_rules": len(_zone_rules())},
            )
            completed = 0
            for index, target in enumerate(targets):
                try:
                    target_candidates = _candidate_files(target)
                    if run_batch:
                        preferred_name = f"{target}_macro_domains.geojson"
                        chosen = preferred_name if preferred_name in target_candidates else (
                            target_candidates[0] if target_candidates else ""
                        )
                    else:
                        chosen = source_filename
                    if not chosen:
                        raise FileNotFoundError(f"No domain GeoJSON found for '{target}'.")
                    status.write(f"Processing {target} ({index + 1}/{len(targets)}) from {chosen}...")
                    result = _run_single(target, chosen)
                    created = [rule for rule in result["pooled"] if rule.get("status") == "created"]
                    skipped = [rule for rule in result["pooled"] if rule.get("status") != "created"]
                    log_annotation(
                        f"{target}: pooled {len(created)}/{len(result['pooled'])} rule(s), "
                        f"{result['zone_count']} zone layer(s) -> {result['output_path']}",
                        subtab="s4",
                    )
                    for rule in skipped:
                        st.warning(f"{target}: pooling rule '{rule['name']}' skipped ({rule['status']}).")
                    if target == sample_id:
                        st.session_state["_annot_ta_s4_last_result"] = result
                    completed += 1
                except Exception as error:
                    st.error(f"{target}: {error}")
                    log_annotation(f"{target}: pooling/zones failed - {error}", subtab="s4")
                progress_bar.progress((index + 1) / len(targets))
            status.empty()
            record_process_event(
                TAB_CONTEXT, "Pooling and zones complete", status="complete",
                duration_s=time.time() - started,
                details={"samples": len(targets), "completed": completed},
            )
            st.success(f"Pooling and zoning finished for {completed}/{len(targets)} image(s).")

        result = st.session_state.get("_annot_ta_s4_last_result")
        if isinstance(result, dict) and result.get("pooled"):
            st.dataframe(pd.DataFrame(result["pooled"]), hide_index=True, width="stretch")

        output_path = os.path.join(_output_directory(), f"{sample_id}_{OUTPUT_SUFFIX}.geojson")
        describe_output(output_path, "Pooled and zoned GeoJSON")
        if os.path.isfile(output_path):
            st.caption(f"Classes written: {', '.join(_class_names(output_path))}")
            qupath_hint(output_path, "The cell annotation step")

    with right:
        st.markdown("#### Preview")
        image_path = resolve_image_path(sample_id)
        if not image_path:
            st.info("Select an image to preview.")
            return

        layer_options = {
            "Pooled + zones (output)": os.path.join(_output_directory(), f"{sample_id}_{OUTPUT_SUFFIX}.geojson"),
            "Source domains": os.path.join(_source_directory(), source_filename),
        }
        available = [name for name, path in layer_options.items() if path and os.path.isfile(path)]
        if available:
            prune_widget_selection("annot_ta_s4_layers", available)
            chosen = st.multiselect(
                "Overlay layers", available, default=available[:1],
                key="annot_ta_s4_layers",
                help="Polygons are read from disk each render, so QuPath edits show up immediately.",
            )
            if chosen:
                image, colors = overlay_polygons_on_thumbnail(
                    image_path,
                    [layer_options[name] for name in chosen],
                    class_colors=st.session_state.get("annot_ta_domain_colors", {}),
                    fill_alpha=80,
                )
                st.image(image, width="stretch", caption=f"{sample_id} - {', '.join(chosen)}")
                render_class_legend(colors)
            else:
                render_image_preview(image_path, "annot_ta_s4")
        else:
            render_image_preview(image_path, "annot_ta_s4")
