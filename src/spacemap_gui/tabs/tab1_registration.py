"""Tab 1: Registration UI (Batch & Multi-Stack)."""

import csv
import colorsys
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TextIO

import numpy as np
import pandas as pd
import streamlit as st
import tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError for large images
import cv2

from spacemap.registration import (
    extract_dapi_channel,
    preprocess_he_image,
    register_images,
    register_codex_to_codex,
    merge_multi_round_ome_tiff,
    resolve_reference_channel,
)
from spacemap.utils import get_channel_info_qptiff, hex_to_rgb
from spacemap.background_subtraction_utils import (
    get_channel_info,
    get_image_mpp,
    load_preview_channels_compatible,
    normalize_image_stem,
    is_supported_image_file,
)
from spacemap_gui.ui_utils import (
    directory_input_with_browse,
    show_preflight_errors,
    validate_run_directories,
    validate_selected_files,
)
from spacemap_gui.process_log import record_process_event
from spacemap_gui import terminal

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab1', message, step=None, level=level)


@dataclass
class ChannelRecord:
    """Container for a single imaging channel."""
    original_name: str
    display_name: str
    export: bool = True
    show: bool = False
    color_hex: str = "#FFFFFF"
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Export": self.export,
            "Show": self.show,
            "Channel": self.display_name,  # Editable name
            "Original": self.original_name, # Reference
            "Hex": self.color_hex,
            "Color": self.color_hex, # For color picker
            "Dye": self.metadata.get("dye", "-"),
            "Exp": self.metadata.get("exposure", "-"),
        }

@dataclass
class ImageContainer:
    """Universal reader for QPTIFF / OME-TIFF with channel + metadata access."""
    path: Path
    image_type: str
    channels: List[ChannelRecord]
    mpp: Optional[float] = None

    @classmethod
    def from_path(cls, path: str | Path, image_type: str = "codex") -> "ImageContainer":
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        
        if suffix == ".qptiff":
            return cls._from_qptiff(file_path, image_type)
        return cls._from_ome(file_path, image_type)

    @classmethod
    def _from_qptiff(cls, file_path: Path, image_type: str) -> "ImageContainer":
        channels: List[ChannelRecord] = []
        try:
            names, colors, metadata = get_channel_info_qptiff(str(file_path))
        except Exception as exc:
            _log_console(f"Channel parse failed for {file_path.name}: {exc}", level="error")
            names, colors, metadata = [], [], []

        if not names:
            names = ["Channel 0"]
            colors = ["#FFFFFF"]
            metadata = [{}]

        # Force RGB channels for H&E if detection failed or returned any single channel
        # (covers "Channel 0", "Color", or any other 1-channel result from the parser)
        if image_type == "he" and len(names) <= 1:
             names = ["HE_Red", "HE_Green", "HE_Blue"]
             colors = ["#FF0000", "#00FF00", "#0000FF"]
             metadata = [{}, {}, {}]

        for name, col, meta in zip(names, colors, metadata):
            channels.append(
                ChannelRecord(
                    original_name=name,
                    display_name=name,
                    export=True,
                    show=False,
                    color_hex=col,
                    metadata=meta or {},
                )
            )

        mpp = get_image_mpp(str(file_path)).get('codex')

        return cls(
            path=file_path,
            image_type=image_type,
            channels=channels,
            mpp=mpp,
        )

    @classmethod
    def _from_ome(cls, file_path: Path, image_type: str) -> "ImageContainer":
        try:
            names, colors, metadata, mpp_dict = get_channel_info(str(file_path))
        except Exception as exc:
            raise ValueError(f"Failed to read OME-TIFF metadata for {file_path}: {exc}")

        if not names and image_type == "he":
            names = ["HE_Red", "HE_Green", "HE_Blue"]
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            metadata = [{}, {}, {}]

        if not names:
            names = ["Channel 0"]
            colors = [(255, 255, 255)]
            metadata = [{}]

        channels: List[ChannelRecord] = []
        for idx, name in enumerate(names):
            channel_name = str(name)
            col = colors[idx] if idx < len(colors) else (255, 255, 255)
            meta = metadata[idx] if idx < len(metadata) else {}
            meta_items = meta.items() if isinstance(meta, dict) else []
            meta_dict = {str(k): str(v) for k, v in meta_items}
            color_hex = '#{:02X}{:02X}{:02X}'.format(*col) if isinstance(col, tuple) else str(col)
            channels.append(
                ChannelRecord(
                    original_name=channel_name,
                    display_name=channel_name,
                    export=True,
                    show=False,
                    color_hex=color_hex,
                    metadata=meta_dict,
                )
            )

        return cls(
            path=file_path,
            image_type=image_type,
            channels=channels,
            mpp=mpp_dict.get('codex'),
        )

    def get_export_config(self) -> Dict[str, Any]:
        """Generate the config dictionary for registration functions.
        
        Uses index-based keys to correctly handle duplicate channel names
        (e.g. two channels both named 'FoxP3' on different fluorophores).
        """
        keep_indices = []
        rename_by_index = {}
        colors_by_index = {}
        
        for idx, c in enumerate(self.channels):
            if c.export:
                keep_indices.append(idx)
                if c.display_name != c.original_name:
                    rename_by_index[idx] = c.display_name
                colors_by_index[idx] = hex_to_rgb(c.color_hex)
                
        return {
            "keep_indices": keep_indices,
            "rename_by_index": rename_by_index,
            "colors_by_index": colors_by_index,
        }

    def export_schema_rows(self) -> List[Dict[str, Any]]:
        """Return a portable schema representation for CSV import/export."""
        rows: List[Dict[str, Any]] = []
        for idx, channel in enumerate(self.channels):
            rows.append(
                {
                    "channel_index": idx,
                    "original_name": channel.original_name,
                    "dye": channel.metadata.get("dye", "-"),
                    "exposure": channel.metadata.get("exposure", "-"),
                    "new_name": channel.display_name,
                    "color_hex": channel.color_hex,
                    "export": channel.export,
                    "show": channel.show,
                    "image_type": self.image_type,
                }
            )
        return rows

    def apply_schema_rows(self, schema_rows: List[Dict[str, Any]]):
        """Apply schema rows using smart matching (by exact index+name, name+dye, or name sequence)."""
        if not schema_rows:
            return

        consumed_rows = set()

        for idx, channel in enumerate(self.channels):
            chan_name_norm = _normalize_channel_identifier(channel.original_name)
            chan_dye = str(channel.metadata.get("dye", "-")).strip().lower()
            
            matched_row = None
            
            # Preference 1: Index + Name Match (e.g. index is the same AND biomarker name is the same)
            for r_idx, row in enumerate(schema_rows):
                if r_idx in consumed_rows:
                    continue
                row_idx = row.get("channel_index")
                if row_idx is not None:
                    try:
                        if int(row_idx) == idx:
                            row_name_norm = _normalize_channel_identifier(row.get("original_name"))
                            if row_name_norm == chan_name_norm:
                                matched_row = row
                                consumed_rows.add(r_idx)
                                break
                    except (TypeError, ValueError):
                        pass

            # Preference 2: Name + Dye Match (for multiple fluorophores of same biomarker, e.g. FoxP3 AF750 vs Cy5)
            if matched_row is None and chan_dye not in ("", "-", "unknown"):
                for r_idx, row in enumerate(schema_rows):
                    if r_idx in consumed_rows:
                        continue
                    row_name_norm = _normalize_channel_identifier(row.get("original_name"))
                    row_dye = str(row.get("dye", "-")).strip().lower()
                    if row_name_norm == chan_name_norm and row_dye == chan_dye:
                        matched_row = row
                        consumed_rows.add(r_idx)
                        break

            # Preference 3: Name Match only (matches sequential occurrences of the same biomarker name)
            if matched_row is None:
                for r_idx, row in enumerate(schema_rows):
                    if r_idx in consumed_rows:
                        continue
                    row_name_norm = _normalize_channel_identifier(row.get("original_name"))
                    if row_name_norm == chan_name_norm:
                        matched_row = row
                        consumed_rows.add(r_idx)
                        break

            if matched_row is not None:
                new_name = str(matched_row.get("new_name", "")).strip()
                color_hex = str(matched_row.get("color_hex", "")).strip()
                export_value = matched_row.get("export")
                show_value = matched_row.get("show")

                if new_name:
                    channel.display_name = new_name
                if color_hex:
                    if not color_hex.startswith("#") and len(color_hex) in (3, 6):
                        color_hex = f"#{color_hex}"
                    channel.color_hex = color_hex
                parsed_export = _coerce_bool(export_value)
                if parsed_export is not None:
                    channel.export = parsed_export
                parsed_show = _coerce_bool(show_value)
                if parsed_show is not None:
                    channel.show = parsed_show


def _normalize_channel_identifier(name: Optional[str]) -> str:
    return "".join(ch for ch in str(name or "").strip().lower() if ch.isalnum())


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _is_default_or_missing_color(color_hex: Optional[str]) -> bool:
    normalized = str(color_hex or "").strip().upper()
    return normalized in {"", "#", "#FFFFFF", "FFFFFF", "#FFF", "FFF"}


def _generate_distinct_hex_colors(count: int) -> List[str]:
    if count <= 0:
        return []

    colors: List[str] = []
    hue_offset = 0.61803398875
    value_cycle = [0.95, 0.82, 0.68]
    saturation_cycle = [0.78, 0.9, 0.66]

    for idx in range(count):
        hue = (idx * hue_offset) % 1.0
        saturation = saturation_cycle[idx % len(saturation_cycle)]
        value = value_cycle[(idx // len(saturation_cycle)) % len(value_cycle)]
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append("#{:02X}{:02X}{:02X}".format(int(red * 255), int(green * 255), int(blue * 255)))
    return colors


def _assign_distinct_colors(container: ImageContainer, missing_only: bool = True):
    target_indices = [
        idx
        for idx, channel in enumerate(container.channels)
        if (not missing_only) or _is_default_or_missing_color(channel.color_hex)
    ]
    if not target_indices:
        return

    existing_colors = {
        str(channel.color_hex).strip().upper()
        for idx, channel in enumerate(container.channels)
        if idx not in target_indices and str(channel.color_hex).strip()
    }

    palette_size = max(len(container.channels) * 3, len(target_indices) + len(existing_colors) + 8)
    palette = _generate_distinct_hex_colors(palette_size)
    available_colors = [color for color in palette if color.upper() not in existing_colors]
    if len(available_colors) < len(target_indices):
        available_colors.extend(_generate_distinct_hex_colors(len(target_indices) - len(available_colors) + len(container.channels)))

    for channel_index, color_hex in zip(target_indices, available_colors):
        container.channels[channel_index].color_hex = color_hex


def _normalize_affix(text: Optional[str]) -> str:
    return str(text or "").strip()


def _matches_affix_rule(stem: str, prefix: str, suffix: str) -> bool:
    prefix = _normalize_affix(prefix)
    suffix = _normalize_affix(suffix)
    lowered = stem.lower()
    if prefix and not lowered.startswith(prefix.lower()):
        return False
    if suffix and not lowered.endswith(suffix.lower()):
        return False
    return True


def _strip_affixes(stem: str, prefix: str, suffix: str) -> str:
    prefix = _normalize_affix(prefix)
    suffix = _normalize_affix(suffix)
    result = stem
    if prefix and result.lower().startswith(prefix.lower()):
        result = result[len(prefix):]
    if suffix and result.lower().endswith(suffix.lower()):
        result = result[:-len(suffix)]
    return result


def _derive_pair_identity(filename: str, prefix: str, suffix: str) -> str:
    stem = normalize_image_stem(filename)
    return _strip_affixes(stem, prefix, suffix)


def _build_role_file_map(
    filenames: List[str],
    role_prefix: str,
    role_suffix: str,
    other_prefix: str,
    other_suffix: str,
) -> Dict[str, str]:
    role_prefix = _normalize_affix(role_prefix)
    role_suffix = _normalize_affix(role_suffix)
    other_prefix = _normalize_affix(other_prefix)
    other_suffix = _normalize_affix(other_suffix)

    explicit_role = bool(role_prefix or role_suffix)
    explicit_other = bool(other_prefix or other_suffix)
    mapping: Dict[str, str] = {}

    for filename in filenames:
        stem = normalize_image_stem(filename)
        matches_role = _matches_affix_rule(stem, role_prefix, role_suffix)
        matches_other = _matches_affix_rule(stem, other_prefix, other_suffix)

        if explicit_role:
            if not matches_role:
                continue
        elif explicit_other and matches_other:
            continue

        identity = _strip_affixes(stem, role_prefix, role_suffix)
        if identity:
            mapping[identity] = filename

    return mapping


def _parse_channel_schema_csv(uploaded_file) -> List[Dict[str, Any]]:
    if uploaded_file is None:
        return []

    text = uploaded_file.getvalue().decode("utf-8-sig")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    required_columns = {"new_name"}
    available_columns = {str(name or "").strip() for name in (reader.fieldnames or [])}
    if not ({"channel_index"} <= available_columns or {"original_name"} <= available_columns):
        raise ValueError("CSV must include either 'channel_index' or 'original_name'.")
    if not required_columns <= available_columns:
        raise ValueError("CSV must include 'new_name'. Optional columns: color_hex, export, show, image_type.")

    rows: List[Dict[str, Any]] = []
    for row in reader:
        normalized = {str(k).strip(): v for k, v in row.items()}
        rows.append(normalized)
    return rows


def _schema_template_signature(container: ImageContainer) -> str:
    payload = {
        "image_type": str(container.image_type or "").strip().lower(),
        "original_names": [
            _normalize_channel_identifier(channel.original_name)
            for channel in container.channels
        ],
    }
    return json.dumps(payload, sort_keys=True)


def _store_schema_template(session_key: str, container: ImageContainer, schema_rows: List[Dict[str, Any]]):
    st.session_state[session_key] = {
        "rows": schema_rows,
        "image_type": container.image_type,
        "signature": _schema_template_signature(container),
    }


def _resolve_schema_template_rows(template_payload: Any, container: ImageContainer) -> List[Dict[str, Any]]:
    if not isinstance(template_payload, dict):
        return []

    rows = template_payload.get("rows")
    if not isinstance(rows, list):
        return []

    template_image_type = str(template_payload.get("image_type") or "").strip().lower()
    if template_image_type and template_image_type != str(container.image_type or "").strip().lower():
        return []

    # Removed strict signature check to allow smart name+dye matching across different panels/orders
    return rows


def _apply_stored_schema_template(
    container: ImageContainer,
    template_payload: Any,
    key_prefix: Optional[str] = None,
):
    schema_rows = _resolve_schema_template_rows(template_payload, container)
    if schema_rows:
        _apply_schema_to_container(container, schema_rows, key_prefix)


def _migrate_tab1_registration_state():
    version_key = "tab1_registration_state_version"
    current_version = 1
    if st.session_state.get(version_key) == current_version:
        return

    st.session_state[version_key] = current_version
    for session_key in list(st.session_state.keys()):
        if not isinstance(session_key, str):
            continue
        if session_key in {"pair_configs", "multi_r1_container", "multi_r2_container", "multi_he_container"}:
            st.session_state.pop(session_key, None)
            continue
        if session_key.startswith("multi_codex_extra_") and session_key.endswith("_container"):
            st.session_state.pop(session_key, None)
            continue
        if session_key.endswith("_schema_template") and not isinstance(st.session_state.get(session_key), dict):
            st.session_state.pop(session_key, None)


def _sync_container_editor_state(container: ImageContainer, key_prefix: str):
    for idx, channel in enumerate(container.channels):
        st.session_state[f"{key_prefix}_exp_{idx}"] = channel.export
        st.session_state[f"{key_prefix}_show_{idx}"] = channel.show
        st.session_state[f"{key_prefix}_name_{idx}"] = channel.display_name
        st.session_state[f"{key_prefix}_color_{idx}"] = channel.color_hex
        st.session_state[f"{key_prefix}_picker_{idx}"] = channel.color_hex


def _container_editor_state_token(container: ImageContainer) -> str:
    payload = {
        "path": str(container.path),
        "image_type": str(container.image_type or "").strip().lower(),
        "original_names": [str(channel.original_name) for channel in container.channels],
    }
    return json.dumps(payload, sort_keys=True)


def _reset_container_editor_state(container: ImageContainer, key_prefix: str):
    tracked_prefixes = tuple(
        f"{key_prefix}_{field}"
        for field in ("exp_", "show_", "name_", "color_", "picker_", "anchor_")
    )
    for session_key in list(st.session_state.keys()):
        if not isinstance(session_key, str):
            continue
        if session_key.startswith(tracked_prefixes):
            del st.session_state[session_key]
    _sync_container_editor_state(container, key_prefix)


def _ensure_container_editor_state(container: ImageContainer, key_prefix: str):
    token_key = f"{key_prefix}_editor_state_token"
    current_token = _container_editor_state_token(container)
    if st.session_state.get(token_key) != current_token:
        _reset_container_editor_state(container, key_prefix)
        st.session_state[token_key] = current_token


def _sync_anchor_checkbox_state(anchor_state_key: str, key_prefix: str, container: ImageContainer):
    selected_anchor = _ensure_anchor_value(anchor_state_key, container)
    for idx, channel in enumerate(container.channels):
        st.session_state[f"{key_prefix}_anchor_{idx}"] = channel.original_name == selected_anchor


def _apply_schema_to_container(container: ImageContainer, schema_rows: List[Dict[str, Any]], key_prefix: Optional[str] = None):
    container.apply_schema_rows(schema_rows)
    if key_prefix:
        _sync_container_editor_state(container, key_prefix)


def _apply_distinct_colors_to_container(
    container: ImageContainer,
    key_prefix: str,
    missing_only: bool,
    session_template_key: Optional[str] = None,
):
    _assign_distinct_colors(container, missing_only=missing_only)
    _sync_container_editor_state(container, key_prefix)
    if session_template_key:
        _store_schema_template(session_template_key, container, _schema_rows_from_container(container))


def _render_distinct_color_tools(
    container: ImageContainer,
    key_prefix: str,
    label: str,
    session_template_key: Optional[str] = None,
):
    fill_label = f"Fill {label} missing colors"
    recolor_label = f"Recolor all {label} channels"
    color_col1, color_col2 = st.columns(2)
    if color_col1.button(fill_label, key=f"{key_prefix}_fill_missing_colors"):
        _apply_distinct_colors_to_container(
            container,
            key_prefix,
            missing_only=True,
            session_template_key=session_template_key,
        )
        st.success(f"Assigned distinctive colors to {label} channels with missing/default colors.")
    if color_col2.button(recolor_label, key=f"{key_prefix}_recolor_all"):
        _apply_distinct_colors_to_container(
            container,
            key_prefix,
            missing_only=False,
            session_template_key=session_template_key,
        )
        st.success(f"Recolored all {label} channels with a distinctive palette.")


def _load_batch_pair_config(pair_data: Dict[str, Any]) -> Dict[str, ImageContainer]:
    codex_container = ImageContainer.from_path(pair_data["CODEX Path"], "codex")
    he_container = ImageContainer.from_path(pair_data["H&E Path"], "he")

    batch_codex_schema = st.session_state.get("batch_codex_schema_template")
    batch_he_schema = st.session_state.get("batch_he_schema_template")
    if batch_codex_schema:
        _apply_stored_schema_template(codex_container, batch_codex_schema)
    if batch_he_schema:
        _apply_stored_schema_template(he_container, batch_he_schema)

    for channel in codex_container.channels:
        if "DAPI" in channel.original_name.upper() or "HOECHST" in channel.original_name.upper():
            channel.show = True

    return {
        "codex": codex_container,
        "he": he_container,
    }


def _ensure_batch_pair_config(pair_key: str, pair_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, ImageContainer]]:
    if pair_data is None:
        return None

    if "pair_configs" not in st.session_state:
        st.session_state.pair_configs = {}

    cached_config = st.session_state.pair_configs.get(pair_key)
    needs_reload = (
        cached_config is None
        or str(cached_config["codex"].path) != pair_data["CODEX Path"]
        or str(cached_config["he"].path) != pair_data["H&E Path"]
    )
    if needs_reload:
        st.session_state.pair_configs[pair_key] = _load_batch_pair_config(pair_data)
    return st.session_state.pair_configs[pair_key]


def _pair_settings_signature(pair_key: str, codex_container: ImageContainer, he_container: ImageContainer) -> str:
    payload = {
        "pair_key": pair_key,
        "codex_schema": _schema_rows_from_container(codex_container),
        "he_schema": _schema_rows_from_container(he_container),
        "codex_mpp": float(st.session_state.get(f"pair_codex_mpp_{pair_key}", codex_container.mpp or 0.0)),
        "he_mpp": float(st.session_state.get(f"pair_he_mpp_{pair_key}", he_container.mpp or 0.0)),
        "anchor": st.session_state.get(f"pair_anchor_{pair_key}"),
    }
    return json.dumps(payload, sort_keys=True)


def _apply_active_pair_settings_to_all_pairs(
    active_pair_key: str,
    pair_lookup: Dict[str, Dict[str, Any]],
    active_codex_container: ImageContainer,
    active_he_container: ImageContainer,
):
    codex_schema_rows = _schema_rows_from_container(active_codex_container)
    he_schema_rows = _schema_rows_from_container(active_he_container)
    codex_mpp = float(st.session_state.get(f"pair_codex_mpp_{active_pair_key}", active_codex_container.mpp or 0.0))
    he_mpp = float(st.session_state.get(f"pair_he_mpp_{active_pair_key}", active_he_container.mpp or 0.0))
    active_anchor = st.session_state.get(f"pair_anchor_{active_pair_key}")

    _store_schema_template("batch_codex_schema_template", active_codex_container, codex_schema_rows)
    _store_schema_template("batch_he_schema_template", active_he_container, he_schema_rows)

    for pair_key, pair_data in pair_lookup.items():
        if pair_key == active_pair_key:
            continue
        pair_config = _ensure_batch_pair_config(pair_key, pair_data)
        if pair_config is None:
            continue

        _apply_schema_to_container(pair_config["codex"], codex_schema_rows, f"batch_{pair_key}_codex")
        _apply_schema_to_container(pair_config["he"], he_schema_rows, f"batch_{pair_key}_he")

        st.session_state[f"pair_codex_mpp_{pair_key}"] = codex_mpp
        st.session_state[f"pair_he_mpp_{pair_key}"] = he_mpp

        anchor_state_key = f"pair_anchor_{pair_key}"
        st.session_state[anchor_state_key] = _default_anchor_name(pair_config["codex"], preferred_name=active_anchor)
        _sync_anchor_checkbox_state(anchor_state_key, f"batch_{pair_key}_codex", pair_config["codex"])


def _build_registered_he_export_config(he_container: ImageContainer, registered_he_path: str) -> Optional[Dict[str, Any]]:
    """Build an index-based export config for a registered H&E file.
    
    Channel indices in the container map 1:1 to the registered file channels.
    """
    export_config = {"keep_indices": [], "rename_by_index": {}, "colors_by_index": {}}
    for idx, channel in enumerate(he_container.channels):
        if not channel.export:
            continue
        export_config["keep_indices"].append(idx)
        export_config["rename_by_index"][idx] = channel.display_name
        export_config["colors_by_index"][idx] = hex_to_rgb(channel.color_hex)

    return export_config


def _schema_rows_from_container(container: ImageContainer) -> List[Dict[str, Any]]:
    return container.export_schema_rows()

def _extract_mpp(tif: tifffile.TiffFile) -> Optional[float]:
    """Best-effort extraction of micron per pixel from TIFF metadata."""
    try:
        res = tif.pages[0].resolution
        if res and res[0] and res[1]:
            # resolution is (num, den) in pixels per unit; assume inch if unit missing
            px_per_unit = res[0] / res[1]
            mpp = 25400.0 / px_per_unit  # µm per pixel when unit is inch
            return mpp
    except Exception:
        return None
    return None


@st.cache_data
def _auto_mpp_value(image_path: str | Path, fallback: float) -> float:
    try:
        detected = get_image_mpp(str(image_path)).get('codex')
        return float(detected) if detected else fallback
    except Exception:
        return fallback


@st.cache_data
def _raw_detected_mpp_value(image_path: str | Path) -> Optional[float]:
    try:
        detected = get_image_mpp(str(image_path)).get('codex')
        return float(detected) if detected else None
    except Exception:
        return None


@st.cache_data
def _suggest_directory_mpp(directory: str, filenames: Tuple[str, ...], fallback: float) -> float:
    detected_values: List[float] = []
    for filename in filenames:
        detected = _raw_detected_mpp_value(os.path.join(directory, filename))
        if detected is not None:
            detected_values.append(float(detected))
    if detected_values:
        return float(np.median(np.asarray(detected_values, dtype=np.float32)))
    return float(fallback)


def _sync_detected_input_value(
    input_key: str,
    detected_key: str,
    detected_value: float,
    source_key: Optional[str] = None,
    source_value: Optional[Any] = None,
):
    current_value = st.session_state.get(input_key)
    previous_detected = st.session_state.get(detected_key)
    source_changed = False
    if source_key is not None:
        source_changed = st.session_state.get(source_key) != source_value
    if source_changed or current_value is None or previous_detected is None or float(current_value) == float(previous_detected):
        st.session_state[input_key] = float(detected_value)
    st.session_state[detected_key] = float(detected_value)
    if source_key is not None:
        st.session_state[source_key] = source_value


def _ensure_float_session_value(key: str, default_value: float):
    if key not in st.session_state or st.session_state.get(key) is None:
        st.session_state[key] = float(default_value)


def _ensure_int_session_value(key: str, default_value: int):
    if key not in st.session_state or st.session_state.get(key) is None:
        st.session_state[key] = int(default_value)


DEFAULT_INTERMEDIATE_SUBDIR = "intermediate_images"

# Coarse-alignment keypoints. Whole-slide H&E-to-reference pairs need several thousand;
# at the widget minimum of 500 the affine palom fits is under-determined and the block
# warp can ask OpenCV for a source region past its 32767 px limit.
DEFAULT_ALIGNMENT_KEYPOINTS = 10000


def _intermediate_subdir_name(session_key: str) -> str:
    """Subfolder under the output directory that holds intermediate images.

    Falls back to the default when the field is blank, including when a blank value was
    persisted by an older project bundle — intermediates should never be written loose
    into the output directory alongside the merged results.
    """
    value = str(st.session_state.get(session_key, "") or "").strip().strip("/\\")
    return value or DEFAULT_INTERMEDIATE_SUBDIR


def _uploaded_file_token(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    payload = uploaded_file.getvalue()
    return f"{uploaded_file.name}:{len(payload)}:{hashlib.sha1(payload).hexdigest()}"


def _default_anchor_name(container: ImageContainer, preferred_name: Optional[str] = None) -> str:
    channel_names = [channel.original_name for channel in container.channels]
    if preferred_name:
        preferred_lower = str(preferred_name).strip().lower()
        for channel_name in channel_names:
            if str(channel_name).strip().lower() == preferred_lower:
                return channel_name
    _ref_idx, ref_name = resolve_reference_channel(channel_names)
    if ref_name is not None:
        return str(ref_name)
    return channel_names[0] if channel_names else "Channel 0"


def _ensure_anchor_value(session_key: str, container: ImageContainer, preferred_name: Optional[str] = None) -> str:
    valid_names = {channel.original_name for channel in container.channels}
    current_value = st.session_state.get(session_key)
    if current_value not in valid_names:
        st.session_state[session_key] = _default_anchor_name(container, preferred_name=preferred_name)
    return str(st.session_state[session_key])


def _handle_anchor_checkbox_change(
    anchor_state_key: str,
    checkbox_key: str,
    selected_channel_name: str,
    fallback_channel_name: str,
    key_prefix: str,
    channel_names: List[str],
):
    if st.session_state.get(checkbox_key):
        chosen_anchor = selected_channel_name
    elif st.session_state.get(anchor_state_key) == selected_channel_name:
        chosen_anchor = fallback_channel_name
    else:
        chosen_anchor = st.session_state.get(anchor_state_key, fallback_channel_name)

    st.session_state[anchor_state_key] = chosen_anchor
    for idx, channel_name in enumerate(channel_names):
        st.session_state[f"{key_prefix}_anchor_{idx}"] = channel_name == chosen_anchor


@st.cache_data
def _channel_names_for_image(image_path: str) -> List[str]:
    try:
        channel_names, _channel_colors, _channel_meta, _mpp = get_channel_info(image_path)
        return [str(name) for name in channel_names]
    except Exception:
        return []


@st.cache_data
def _auto_reference_channel_name(image_path: str) -> Optional[str]:
    channel_names = _channel_names_for_image(image_path)
    if not channel_names:
        return None
    _ref_index, ref_name = resolve_reference_channel(channel_names)
    return str(ref_name) if ref_name is not None else None


def _selected_reference_channel_name(selection: Optional[str]) -> Optional[str]:
    if not selection or selection == "<Auto>":
        return None
    return str(selection)


def _schema_rows_to_csv(schema_rows: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    fieldnames = ["channel_index", "original_name", "new_name", "color_hex", "export", "show", "image_type"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in schema_rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output.getvalue()


def _build_pair_entry(
    prefix: str,
    codex_dir: str,
    he_dir: str,
    codex_file: str,
    he_file: str,
    source: str,
    fallback_codex_mpp: float,
    fallback_he_mpp: float,
) -> Dict[str, Any]:
    codex_path = os.path.join(codex_dir, codex_file)
    he_path = os.path.join(he_dir, he_file)
    pair_key = f"{prefix}::{codex_file}::{he_file}"
    return {
        "Pair Key": pair_key,
        "Prefix": prefix,
        "CODEX File": codex_file,
        "H&E File": he_file,
        "CODEX Path": codex_path,
        "H&E Path": he_path,
        "Pair Source": source,
        "Detected CODEX MPP": _auto_mpp_value(codex_path, fallback_codex_mpp),
        "Detected H&E MPP": _auto_mpp_value(he_path, fallback_he_mpp),
        "Auto Reference Channel": _auto_reference_channel_name(codex_path),
    }


def get_image_metadata_log(file_path: str) -> Dict[str, str]:
    """Safely extract metadata fields for logging.

    Parameters
    ----------
    file_path
        Path to an image file (TIFF/OME-TIFF/QPTIFF).

    Returns
    -------
    Dict[str, str]
        A flat dictionary of string fields suitable for CSV logging.

    Notes
    -----
    This function is best-effort and must never raise; on failure it returns
    consistent keys filled with "Extraction Failed".
    """

    def _failed() -> Dict[str, str]:
        failed_val = "Extraction Failed"
        return {
            "Size_GB": failed_val,
            "DType": failed_val,
            "Shape": failed_val,
            "Height": failed_val,
            "Width": failed_val,
            "Channels": failed_val,
            "MPP": failed_val,
            "Pyramid_Levels": failed_val,
            "Markers": failed_val,
            "Channel_Count": failed_val,
            "Metadata_Status": failed_val,
        }

    try:
        abs_path = os.path.abspath(file_path)
        size_gb = os.path.getsize(abs_path) / (1024 ** 3)

        dtype_str = "Unknown"
        shape_str = "Unknown"
        height: Optional[int] = None
        width: Optional[int] = None
        channels: Optional[int] = None
        mpp_val: Optional[float] = None
        pyramid_levels_str = "Single Level"
        markers_str = "Unknown"
        channel_count: Optional[int] = None

        # Markers / channel names
        suffix = Path(abs_path).suffix.lower()
        try:
            if suffix == ".qptiff":
                names, _, _ = get_channel_info_qptiff(abs_path)
                if names:
                    markers_str = "; ".join([str(n) for n in names])
                    channel_count = len(names)
            else:
                names, _, _, _ = get_channel_info(abs_path)
                if names:
                    markers_str = "; ".join([str(n) for n in names])
                    channel_count = len(names)
        except Exception:
            # Keep as Unknown; do not fail metadata extraction solely on marker parsing.
            pass

        with tifffile.TiffFile(abs_path) as tif:
            # Resolution / MPP
            mpp_val = _extract_mpp(tif)

            series = tif.series[0] if tif.series else None
            if series is not None:
                shape = series.shape
                axes = getattr(series, "axes", None)
            else:
                shape = tif.pages[0].shape
                axes = getattr(tif.pages[0], "axes", None)

            shape_str = str(tuple(int(s) for s in shape))

            # dtype
            try:
                dtype_str = str(tif.pages[0].dtype)
            except Exception:
                dtype_str = "Unknown"

            # Height / Width / Channels (best-effort)
            try:
                if axes and "Y" in axes and "X" in axes:
                    height = int(shape[axes.index("Y")])
                    width = int(shape[axes.index("X")])
                    if "C" in axes:
                        channels = int(shape[axes.index("C")])
                elif len(shape) == 2:
                    height, width = int(shape[0]), int(shape[1])
                elif len(shape) == 3:
                    # Heuristic: if a dim is 3, treat as channels
                    if 3 in shape:
                        c_dim = int(list(shape).index(3))
                        channels = 3
                        if c_dim == 0:
                            height, width = int(shape[1]), int(shape[2])
                        elif c_dim == 2:
                            height, width = int(shape[0]), int(shape[1])
                    else:
                        # Assume (C, Y, X) if first dim is small-ish
                        if int(shape[0]) <= 64:
                            channels = int(shape[0])
                            height, width = int(shape[1]), int(shape[2])
                        else:
                            # Assume (Y, X, C)
                            height, width = int(shape[0]), int(shape[1])
                            channels = int(shape[2])
            except Exception:
                height = None
                width = None
                channels = None

            # Pyramid levels
            try:
                if series is not None and hasattr(series, "levels") and len(series.levels) > 1:
                    level_dims: List[str] = []
                    for lvl in series.levels:
                        lvl_shape = lvl.shape
                        lvl_axes = getattr(lvl, "axes", axes)
                        if lvl_axes and "Y" in lvl_axes and "X" in lvl_axes:
                            h = int(lvl_shape[lvl_axes.index("Y")])
                            w = int(lvl_shape[lvl_axes.index("X")])
                            level_dims.append(f"({h}, {w})")
                        elif len(lvl_shape) >= 2:
                            level_dims.append(f"({int(lvl_shape[-2])}, {int(lvl_shape[-1])})")
                    pyramid_levels_str = ", ".join(level_dims) if level_dims else "Single Level"
                else:
                    pyramid_levels_str = "Single Level"
            except Exception:
                pyramid_levels_str = "Unknown"

        return {
            "Size_GB": f"{size_gb:.4f}",
            "DType": dtype_str,
            "Shape": shape_str,
            "Height": str(height) if height is not None else "Unknown",
            "Width": str(width) if width is not None else "Unknown",
            "Channels": str(channels) if channels is not None else "Unknown",
            "MPP": f"{mpp_val:.6f}" if isinstance(mpp_val, (int, float)) and mpp_val else "Unknown",
            "Pyramid_Levels": pyramid_levels_str,
            "Markers": markers_str,
            "Channel_Count": str(channel_count) if channel_count is not None else "Unknown",
            "Metadata_Status": "OK",
        }
    except Exception:
        return _failed()


def _prefix_keys(prefix: str, d: Dict[str, str]) -> Dict[str, str]:
    """Prefix all keys in a dictionary for flattened logging."""
    return {f"{prefix}{k}": v for k, v in d.items()}


PREVIEW_LOW_PERCENTILE = 0.01
PREVIEW_HIGH_PERCENTILE = 99.99


def _normalize_plane_for_display(
    image: np.ndarray,
    low_percentile: float = PREVIEW_LOW_PERCENTILE,
    high_percentile: float = PREVIEW_HIGH_PERCENTILE,
) -> np.ndarray:
    """Normalize a 2D plane for display while preserving high bit-depth until render."""
    plane = np.asarray(image, dtype=np.float32)
    if plane.size == 0:
        return plane

    finite_mask = np.isfinite(plane)
    if not finite_mask.any():
        return np.zeros_like(plane, dtype=np.float32)

    valid = plane[finite_mask]
    p_low = float(np.percentile(valid, low_percentile))
    p_high = float(np.percentile(valid, high_percentile))
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        p_low = float(valid.min())
        p_high = float(valid.max())
        if p_high <= p_low:
            return np.zeros_like(plane, dtype=np.float32)

    normalized = np.clip((plane - p_low) / (p_high - p_low), 0.0, 1.0)
    normalized[~finite_mask] = 0.0
    return normalized.astype(np.float32, copy=False)


def _normalize_rgb_for_display(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA image, got shape {rgb.shape}")

    rgb = rgb[..., :3]
    normalized = np.zeros(rgb.shape, dtype=np.float32)
    for channel_index in range(3):
        normalized[..., channel_index] = _normalize_plane_for_display(rgb[..., channel_index])
    return normalized


def _build_he_preview_image(file_path: str) -> Optional[np.ndarray]:
    preview_data, _full_shape, is_rgb = load_preview_channels_compatible(file_path, [0, 1, 2], max_dim=1200)
    if is_rgb and 'rgb' in preview_data:
        return _normalize_rgb_for_display(np.asarray(preview_data['rgb']))

    if len(preview_data) >= 3:
        ordered_planes: List[np.ndarray] = []
        for idx in sorted(preview_data.keys())[:3]:
            plane = np.asarray(preview_data[idx])
            if plane.ndim != 2:
                return None
            ordered_planes.append(_normalize_plane_for_display(plane))
        return np.stack(ordered_planes, axis=-1)

    return None

# -----------------------------------------------------------------------------
# Thumbnail utilities
# -----------------------------------------------------------------------------

@st.cache_data
def load_thumbnail(file_path: str, is_codex: bool = True) -> Optional[np.ndarray]:
    """Load a thumbnail using source preview data with percentile normalization."""
    try:
        if is_codex:
            preview_data, _full_shape, is_rgb = load_preview_channels_compatible(file_path, [0], max_dim=1200)
            if is_rgb and 'rgb' in preview_data:
                img = np.asarray(preview_data['rgb'])
            else:
                img = np.asarray(next(iter(preview_data.values()))) if preview_data else None

            if img is None:
                return None

            if img.ndim == 3:
                c_dim = int(np.argmin(img.shape))
                if c_dim == 0:
                    img = img[0]
                elif c_dim == 2:
                    img = img[..., 0]
                else:
                    img = img[:, 0, :]

            normalized = _normalize_plane_for_display(img)
            colored = np.zeros((normalized.shape[0], normalized.shape[1], 3), dtype=np.float32)
            colored[:, :, 2] = normalized
            return colored

        return _build_he_preview_image(file_path)

    except Exception as e:
        _log_console(f"Thumbnail load error: {e}", level="error")
        return None


@st.cache_data
def load_reference_thumbnail(
    file_path: str,
    reference_channel: Optional[str] = None,
    reference_channel_index: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Load a preview of the selected registration reference channel."""
    try:
        channel_names = _channel_names_for_image(file_path)
        if len(channel_names) <= 1:
            preview_data, _full_shape, is_rgb = load_preview_channels_compatible(file_path, [0], max_dim=1200)
            if is_rgb and 'rgb' in preview_data:
                img = np.asarray(preview_data['rgb'])
                if img.ndim == 3:
                    img = img[..., 0]
            else:
                img = np.asarray(next(iter(preview_data.values()))) if preview_data else None
        else:
            ref_index, _ref_name = resolve_reference_channel(
                channel_names,
                reference_channel=reference_channel,
                reference_channel_index=reference_channel_index,
            )
            preview_data, _full_shape, _is_rgb = load_preview_channels_compatible(file_path, [ref_index], max_dim=1200)
            img = np.asarray(preview_data.get(ref_index)) if ref_index in preview_data else None

        if img is None:
            return None

        if img.ndim == 3:
            if img.shape[-1] in (3, 4):
                img = img[..., 0]
            elif img.shape[0] in (3, 4):
                img = img[0]
            else:
                img = img[..., 0]

        normalized = _normalize_plane_for_display(img)
        colored = np.zeros((normalized.shape[0], normalized.shape[1], 3), dtype=np.float32)
        colored[:, :, 2] = normalized
        return colored
    except Exception as e:
        _log_console(f"Reference thumbnail load error: {e}", level="error")
        return None

@st.cache_data
def load_preview_channels(file_path: str, indices_to_load: List[int]) -> Tuple[Dict[int, np.ndarray], Tuple[int, ...], bool]:
    """Load specific channels for preview (similar to Tab 3)."""
    try:
        preview_data, full_shape, is_rgb = load_preview_channels_compatible(file_path, indices_to_load, max_dim=1200)
        if is_rgb and 'rgb' in preview_data:
            return {0: np.asarray(preview_data['rgb'])}, full_shape, True
        return {int(idx): np.asarray(img) for idx, img in preview_data.items()}, full_shape, False
            
    except Exception as e:
        _log_console(f"Preview load error: {e}", level="error")
        return {}, (0,0), False

def create_composite_thumbnail(img_data_dict: Dict[int, np.ndarray], channel_colors: Dict[int, str]) -> np.ndarray:
    """Create a composite RGB image from selected channels."""
    if not img_data_dict:
        return np.zeros((500, 500, 3), dtype=np.uint8)

    # Get shape from first channel
    first_img = next(iter(img_data_dict.values()))
    first_img = np.asarray(first_img)
    if first_img.ndim == 3:
        if first_img.shape[-1] in (3, 4):
            return _normalize_rgb_for_display(first_img)
        if first_img.shape[0] in (3, 4):
            return _normalize_rgb_for_display(np.moveaxis(first_img, 0, -1))
        first_img = first_img[..., 0]
    h, w = first_img.shape

    composite = np.zeros((h, w, 3), dtype=np.float32)

    for idx, img in img_data_dict.items():
        if idx not in channel_colors:
            continue

        img = np.asarray(img)
        if img.ndim == 3:
            if img.shape[-1] in (3, 4):
                img = img[..., 0]
            elif img.shape[0] in (3, 4):
                img = img[0]
            else:
                img = img[..., 0]

        img_norm = _normalize_plane_for_display(img)

        # Apply color
        hex_color = channel_colors[idx]
        rgb = hex_to_rgb(hex_color)

        for c in range(3):
            composite[:, :, c] += img_norm * (rgb[c] / 255.0)

    composite = np.clip(composite, 0, 1)
    return composite.astype(np.float32, copy=False)


def _apply_preview_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a preview image for display only (expands canvas to fit full content)."""
    if abs(angle_deg) < 0.01:
        return image
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(round(h * sin_a + w * cos_a))
    new_h = int(round(h * cos_a + w * sin_a))
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

def render_custom_channel_editor(
    container: ImageContainer,
    key_prefix: str,
    compact: bool = False,
    anchor_state_key: Optional[str] = None,
    preferred_anchor_name: Optional[str] = None,
):
    """Renders the channel configuration using custom columns (Tab 3 style)."""
    _ensure_container_editor_state(container, key_prefix)
    
    # Separate channels into CODEX and HE
    codex_channels = []
    he_channels = []
    
    for i, ch in enumerate(container.channels):
        if ch.original_name.startswith("HE_") or ch.original_name.startswith("Channel_"): # Heuristic for HE
             # If it's explicitly HE type container, all are HE
             if container.image_type == "he":
                 he_channels.append((i, ch))
             else:
                 # If mixed or unknown, check name
                 if "HE_" in ch.original_name:
                     he_channels.append((i, ch))
                 else:
                     codex_channels.append((i, ch))
        else:
            if container.image_type == "he":
                he_channels.append((i, ch))
            else:
                codex_channels.append((i, ch))

    # Define columns based on compact mode
    show_anchor_column = anchor_state_key is not None and container.image_type == "codex"
    codex_channel_names = [channel.original_name for _, channel in codex_channels]
    selected_anchor_name = None
    fallback_anchor_name = None
    if show_anchor_column and codex_channels and anchor_state_key is not None:
        selected_anchor_name = _ensure_anchor_value(anchor_state_key, container, preferred_name=preferred_anchor_name)
        fallback_anchor_name = _default_anchor_name(container, preferred_name=preferred_anchor_name)

    if compact:
        # Compact: Export, Show, Channel, Hex, Color
        cols_spec = [0.5, 0.5, 0.7, 2.2, 1.5, 0.5] if show_anchor_column else [0.5, 0.5, 2.5, 1.5, 0.5]
        header_cols = ["**Export**", "**Show**", "**Anchor**", "**Channel**", "**Hex**", "**Color**"] if show_anchor_column else ["**Export**", "**Show**", "**Channel**", "**Hex**", "**Color**"]
    else:
        # Full: Export, Show, Channel, Hex, Color, Dye, Exp
        cols_spec = [0.7, 0.7, 0.8, 2.6, 2, 1, 1.5, 1.5] if show_anchor_column else [0.7, 0.7, 3, 2, 1, 1.5, 1.5]
        header_cols = ["**Export**", "**Show**", "**Anchor**", "**Channel**", "**Hex**", "**Color**", "**Dye**", "**Exp**"] if show_anchor_column else ["**Export**", "**Show**", "**Channel**", "**Hex**", "**Color**", "**Dye**", "**Exp**"]

    # --- CODEX Section ---
    if codex_channels:
        st.markdown("#### CODEX Channels")
        # Header
        cols = st.columns(cols_spec)
        for col, text in zip(cols, header_cols):
            col.markdown(text)
        
        for i, ch in codex_channels:
            render_channel_row(
                i,
                ch,
                key_prefix,
                compact,
                allow_anchor=show_anchor_column,
                anchor_state_key=anchor_state_key,
                selected_anchor_name=selected_anchor_name,
                fallback_anchor_name=fallback_anchor_name,
                anchor_channel_names=codex_channel_names,
            )

    # --- HE Section ---
    if he_channels:
        st.markdown("#### H&E Channels")
        # Header
        cols = st.columns(cols_spec)
        for col, text in zip(cols, header_cols):
            col.markdown(text)
        
        for i, ch in he_channels:
            render_channel_row(i, ch, key_prefix, compact)

def render_channel_row(
    i: int,
    ch: ChannelRecord,
    key_prefix: str,
    compact: bool = False,
    allow_anchor: bool = False,
    anchor_state_key: Optional[str] = None,
    selected_anchor_name: Optional[str] = None,
    fallback_anchor_name: Optional[str] = None,
    anchor_channel_names: Optional[List[str]] = None,
):
    """Render a single channel row."""
    c_anchor = None
    c_dye = None
    c_expos = None
    if compact:
        if allow_anchor:
            c_exp, c_show, c_anchor, c_name, c_hex, c_color = st.columns([0.5, 0.5, 0.7, 2.2, 1.5, 0.5])
        else:
            c_exp, c_show, c_name, c_hex, c_color = st.columns([0.5, 0.5, 2.5, 1.5, 0.5])
    else:
        if allow_anchor:
            c_exp, c_show, c_anchor, c_name, c_hex, c_color, c_dye, c_expos = st.columns([0.7, 0.7, 0.8, 2.6, 2, 1, 1.5, 1.5])
        else:
            c_exp, c_show, c_name, c_hex, c_color, c_dye, c_expos = st.columns([0.7, 0.7, 3, 2, 1, 1.5, 1.5])
    
    # Keys
    export_key = f"{key_prefix}_exp_{i}"
    show_key = f"{key_prefix}_show_{i}"
    name_key = f"{key_prefix}_name_{i}"
    color_key = f"{key_prefix}_color_{i}"
    picker_key = f"{key_prefix}_picker_{i}"
    anchor_checkbox_key = f"{key_prefix}_anchor_{i}"
    
    # Initialize session state if needed
    if export_key not in st.session_state: st.session_state[export_key] = ch.export
    if show_key not in st.session_state: st.session_state[show_key] = ch.show
    if name_key not in st.session_state: st.session_state[name_key] = ch.display_name
    base_color = str(st.session_state.get(color_key, ch.color_hex) or ch.color_hex or "#ffffff")
    st.session_state[color_key] = base_color
    if picker_key not in st.session_state:
        st.session_state[picker_key] = base_color
    if allow_anchor and anchor_state_key is not None and anchor_checkbox_key not in st.session_state:
        st.session_state[anchor_checkbox_key] = ch.original_name == selected_anchor_name

    # Export
    new_exp = c_exp.checkbox("##", key=export_key, label_visibility="collapsed")
    ch.export = new_exp
    
    # Show
    # Note: Using a callback or checking state change might be safer for "Show" if it triggers re-runs
    new_show = c_show.checkbox("##", key=show_key, label_visibility="collapsed")
    ch.show = new_show

    if allow_anchor and anchor_state_key is not None and anchor_channel_names is not None and c_anchor is not None:
        c_anchor.checkbox(
            "##",
            key=anchor_checkbox_key,
            label_visibility="collapsed",
            on_change=_handle_anchor_checkbox_change,
            args=(
                anchor_state_key,
                anchor_checkbox_key,
                ch.original_name,
                fallback_anchor_name or selected_anchor_name or ch.original_name,
                key_prefix,
                anchor_channel_names,
            ),
        )
    
    # Name
    new_name = c_name.text_input("##", key=name_key, label_visibility="collapsed")
    ch.display_name = new_name
    
    # Color Sync
    def update_color():
        current_color = st.session_state.get(color_key, ch.color_hex)
        picked_color = st.session_state.get(picker_key, current_color)
        st.session_state[color_key] = picked_color
        st.session_state[picker_key] = picked_color
        
    # Color Picker
    c_color.color_picker("##", key=picker_key, on_change=update_color, label_visibility="collapsed")
    
    # Hex Text
    synced_color = str(st.session_state.get(color_key, base_color) or base_color)
    c_hex.text(synced_color)
    ch.color_hex = synced_color
    
    if not compact and c_dye is not None and c_expos is not None:
        # Dye
        c_dye.text(ch.metadata.get("dye", "-"))
        
        # Exposure
        c_expos.text(ch.metadata.get("exposure", "-"))


def render_schema_tools(
    container: ImageContainer,
    key_prefix: str,
    label: str,
    help_text: str,
    session_template_key: Optional[str] = None,
):
    """Render CSV schema import/export controls for a channel container."""
    upload = st.file_uploader(
        f"Load {label} schema CSV",
        type=["csv"],
        key=f"{key_prefix}_schema_upload",
        help=help_text,
    )
    upload_token_key = f"{key_prefix}_schema_upload_token"
    upload_token = _uploaded_file_token(upload)
    if upload is None:
        st.session_state.pop(upload_token_key, None)
    elif upload_token != st.session_state.get(upload_token_key):
        try:
            schema_rows = _parse_channel_schema_csv(upload)
            _apply_schema_to_container(container, schema_rows, key_prefix)
            if session_template_key:
                _store_schema_template(session_template_key, container, schema_rows)
            st.session_state[upload_token_key] = upload_token
            st.success(f"Applied {label} schema CSV.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not apply {label} schema CSV: {exc}")

    st.download_button(
        f"Download {label} schema CSV",
        data=_schema_rows_to_csv(_schema_rows_from_container(container)),
        file_name=f"{label.lower().replace(' ', '_')}_schema.csv",
        mime="text/csv",
        key=f"{key_prefix}_schema_download",
    )
    _render_distinct_color_tools(container, key_prefix, label, session_template_key=session_template_key)


def render_batch_registration(project_root: Path):
    """Sub-tab 1: CODEX & HE batch registration."""
    # Initialize session state for pair configs EARLY to avoid crashes in right column
    if "pair_configs" not in st.session_state:
        st.session_state.pair_configs = {}
    
    if "active_pair_key" not in st.session_state:
        st.session_state.active_pair_key = None

    st.markdown("### CODEX & HE Batch Registration")    
    # Layout: Left (Controls) | Right (Visualization)
    left_col, right_col = st.columns([1.2, 0.8])
    
    with left_col:
        st.subheader("1. Input & Output Directories")
        codex_dir = directory_input_with_browse(
            label="CODEX Directory",
            text_key="batch_codex_dir",
            default_value=str(project_root / "data/codex"),
            browser_key_prefix="batch_codex_dir",
            start_dir_fallback=str(project_root),
        )
        he_dir = directory_input_with_browse(
            label="H&E Directory",
            text_key="batch_he_dir",
            default_value=str(project_root / "data/he"),
            browser_key_prefix="batch_he_dir",
            start_dir_fallback=str(project_root),
        )

        # Avoid global key collisions with other tabs (e.g., Tab 4 Segmentation)
        if "tab1_batch_output_dir" not in st.session_state and "batch_output_dir" in st.session_state:
            st.session_state["tab1_batch_output_dir"] = st.session_state["batch_output_dir"]
        
        output_dir = directory_input_with_browse(
            label="Output Directory",
            text_key="tab1_batch_output_dir",
            default_value=str(project_root / "outputs/registered"),
            browser_key_prefix="tab1_batch_output_dir",
            start_dir_fallback=str(project_root),
        )

        codex_file_names_for_defaults = sorted([f for f in os.listdir(codex_dir) if is_supported_image_file(f)]) if os.path.isdir(codex_dir) else []
        he_file_names_for_defaults = sorted([f for f in os.listdir(he_dir) if is_supported_image_file(f)]) if os.path.isdir(he_dir) else []

        detected_codex_default = _suggest_directory_mpp(codex_dir, tuple(codex_file_names_for_defaults), 0.5083) if codex_file_names_for_defaults else 0.5083
        detected_he_default = _suggest_directory_mpp(he_dir, tuple(he_file_names_for_defaults), 0.2495) if he_file_names_for_defaults else 0.2495
        _sync_detected_input_value(
            "batch_mpp_codex",
            "batch_mpp_codex_detected",
            detected_codex_default,
            source_key="batch_mpp_codex_source",
            source_value=(codex_dir, tuple(codex_file_names_for_defaults)),
        )
        _sync_detected_input_value(
            "batch_mpp_he",
            "batch_mpp_he_detected",
            detected_he_default,
            source_key="batch_mpp_he_source",
            source_value=(he_dir, tuple(he_file_names_for_defaults)),
        )
        _ensure_float_session_value("batch_mpp_codex", detected_codex_default)
        _ensure_float_session_value("batch_mpp_he", detected_he_default)
        _ensure_int_session_value("batch_alignment_n_keypoints", DEFAULT_ALIGNMENT_KEYPOINTS)
        
        st.subheader("2. Parameters")
        c1, c2, c3 = st.columns(3)
        mpp_codex = c1.number_input(
            "Default CODEX MPP (µm/px)", 
            format="%.4f", 
            key="batch_mpp_codex",
            help="Auto-filled from detected CODEX metadata in the selected directory when available. Used only when a pair lacks usable metadata, and you can still edit each pair's MPP below."
        )
        mpp_he = c2.number_input(
            "Default H&E MPP (µm/px)", 
            format="%.4f", 
            key="batch_mpp_he",
            help="Auto-filled from detected H&E metadata in the selected directory when available. Used only when a pair lacks usable metadata, and you can still edit each pair's MPP below."
        )
        alignment_n_keypoints = c3.number_input(
            "Alignment Keypoints",
            min_value=500,
            max_value=50000,
            step=500,
            value=DEFAULT_ALIGNMENT_KEYPOINTS,
            key="batch_alignment_n_keypoints",
            help="Number of keypoints Palom uses for coarse affine alignment. Higher values can improve robustness but increase runtime.",
        )
        st.caption(
            f"Directory-detected defaults: CODEX {detected_codex_default:.4f} µm/px, "
            f"H&E {detected_he_default:.4f} µm/px"
        )
        
        # Keyed so the pairing rule is saved with the project.
        for affix_key, affix_default in (
            ("tab1_batch_codex_prefix", ""), ("tab1_batch_codex_suffix", "_codex"),
            ("tab1_batch_he_prefix", ""), ("tab1_batch_he_suffix", "_he"),
        ):
            if st.session_state.get(affix_key) is None:
                st.session_state[affix_key] = affix_default
        c4, c5, c6, c7 = st.columns(4)
        codex_prefix = c4.text_input(
            "CODEX Prefix",
            key="tab1_batch_codex_prefix",
            help="Optional exact prefix to strip before matching, applied to the normalized stem without extension. Example: 'CODEX_' turns 'CODEX_CRC02.ome.tif' into 'CRC02'."
        )
        suffix_codex = c5.text_input(
            "CODEX Suffix",
            key="tab1_batch_codex_suffix",
            help="Optional exact suffix to strip before matching. If CODEX prefix and suffix are both empty, files matching the H&E rule are excluded from the CODEX candidate set automatically."
        )
        he_prefix = c6.text_input(
            "H&E Prefix",
            key="tab1_batch_he_prefix",
            help="Optional exact prefix to strip before matching, applied to the normalized stem without extension."
        )
        suffix_he = c7.text_input(
            "H&E Suffix",
            key="tab1_batch_he_suffix",
            help="Optional exact suffix to strip before matching. Example: with empty CODEX affixes and H&E suffix '-HE', 'CRC02.ome.tif' pairs with 'CRC02-HE.ome.tif'."
        )

        if "tab1_batch_intermediate_subdir" not in st.session_state:
            st.session_state["tab1_batch_intermediate_subdir"] = "intermediate_images"
        intermediate_subdir = st.text_input(
            "Intermediate Images Subfolder",
            key="tab1_batch_intermediate_subdir",
            help="Subfolder name under the Output Directory where intermediate/temporary files (resized H&E, anchor channel, registered H&E) will be saved."
        )

        if "tab1_create_log" not in st.session_state:
            st.session_state["tab1_create_log"] = True
        create_log = st.checkbox(
            "Create Processing Log (CSV)",
            help="If enabled, writes detailed CSV logs under Output Directory/log/session_YYMMDD_HHMMSS",
            key="tab1_create_log",
        )

        # Scan for candidates (cached in session state)
        scan_cache_key = (codex_dir, he_dir, codex_prefix, suffix_codex, he_prefix, suffix_he)
        if ("tab1_scan_cache" in st.session_state and 
            st.session_state.get("tab1_scan_cache_key") == scan_cache_key):
            codex_candidate_files, he_candidate_files, auto_pairs_by_codex = st.session_state["tab1_scan_cache"]
        else:
            codex_candidate_files, he_candidate_files, auto_pairs_by_codex = [], [], {}
            if os.path.exists(codex_dir) and os.path.exists(he_dir):
                try:
                    codex_file_names = sorted([f for f in os.listdir(codex_dir) if is_supported_image_file(f)])
                    he_file_names = sorted([f for f in os.listdir(he_dir) if is_supported_image_file(f)])

                    codex_files = _build_role_file_map(
                        codex_file_names,
                        codex_prefix,
                        suffix_codex,
                        he_prefix,
                        suffix_he,
                    )
                    he_files = _build_role_file_map(
                        he_file_names,
                        he_prefix,
                        suffix_he,
                        codex_prefix,
                        suffix_codex,
                    )

                    codex_candidate_files = sorted(codex_files.values())
                    he_candidate_files = sorted(he_files.values())

                    for prefix_val, codex_file in codex_files.items():
                        auto_pairs_by_codex[codex_file] = he_files.get(prefix_val)
                except Exception as e:
                    st.error(f"Error scanning directories: {e}")
            st.session_state["tab1_scan_cache"] = (codex_candidate_files, he_candidate_files, auto_pairs_by_codex)
            st.session_state["tab1_scan_cache_key"] = scan_cache_key

        if codex_candidate_files and he_candidate_files:
            with st.expander("Manual Pair Overrides", expanded=False):
                st.caption("Auto matching removes the configured prefix and suffix from each filename stem, then matches what remains. Use overrides below to change any CODEX -> H&E pair manually or choose Skip to exclude a CODEX image.")
                he_choices = ["<Auto>", "<Skip>"] + he_candidate_files
                for codex_file in codex_candidate_files:
                    select_key = f"tab1_pair_override_{codex_file}"
                    auto_he = auto_pairs_by_codex.get(codex_file)
                    current_selection = st.session_state.get(select_key, "<Auto>")
                    if current_selection not in he_choices:
                        current_selection = "<Auto>"
                    help_text = f"Auto match: {auto_he}" if auto_he else "Auto match: none"
                    st.selectbox(
                        codex_file,
                        options=he_choices,
                        index=he_choices.index(current_selection),
                        key=select_key,
                        help=help_text,
                    )

        # Cache the built pairs list based on directory inputs, fallbacks, and override selections
        overrides_key = tuple(st.session_state.get(f"tab1_pair_override_{f}", "<Auto>") for f in codex_candidate_files)
        pairs_cache_key = (codex_dir, he_dir, codex_prefix, suffix_codex, he_prefix, suffix_he, mpp_codex, mpp_he, overrides_key)
        
        if ("tab1_pairs_cache" in st.session_state and 
            st.session_state.get("tab1_pairs_cache_key") == pairs_cache_key):
            pairs, unmatched_codex_files, unmatched_he_files = st.session_state["tab1_pairs_cache"]
        else:
            pairs = []
            used_he_files = []
            if codex_candidate_files and he_candidate_files:
                for codex_file in codex_candidate_files:
                    prefix = _derive_pair_identity(codex_file, codex_prefix, suffix_codex)
                    auto_he = auto_pairs_by_codex.get(codex_file)
                    selection = st.session_state.get(f"tab1_pair_override_{codex_file}", "<Auto>")

                    if selection == "<Skip>":
                        continue

                    chosen_he = auto_he if selection == "<Auto>" else selection
                    if not chosen_he or chosen_he not in he_candidate_files:
                        continue

                    used_he_files.append(chosen_he)
                    pairs.append(
                        _build_pair_entry(
                            prefix=prefix,
                            codex_dir=codex_dir,
                            he_dir=he_dir,
                            codex_file=codex_file,
                            he_file=chosen_he,
                            source="auto" if selection == "<Auto>" else "manual",
                            fallback_codex_mpp=mpp_codex,
                            fallback_he_mpp=mpp_he,
                        )
                    )

            matched_codex_files = {p["CODEX File"] for p in pairs}
            matched_he_files = set(used_he_files)
            unmatched_codex_files = [f for f in codex_candidate_files if f not in matched_codex_files]
            unmatched_he_files = [f for f in he_candidate_files if f not in matched_he_files]
            
            st.session_state["tab1_pairs_cache"] = (pairs, unmatched_codex_files, unmatched_he_files)
            st.session_state["tab1_pairs_cache_key"] = pairs_cache_key

        # --- Feedback & Table ---
        st.subheader("3. Detected Pairs")
        
        if not pairs:
             if os.path.isdir(codex_dir) and os.path.isdir(he_dir):
                 st.info("No matching pairs found. Check suffixes and file extensions.")
             else:
                 st.warning("Please select valid directories.")
        else:
            # Status Metrics
            st.markdown(f"**Found {len(pairs)} matching pairs.**")

        if unmatched_he_files:
            st.caption(f"Unmatched H&E files: {', '.join(unmatched_he_files[:6])}{' ...' if len(unmatched_he_files) > 6 else ''}")

        if pairs:
            pairs = sorted(pairs, key=lambda pair: (pair["Prefix"], pair["CODEX File"], pair["H&E File"]))
            
            # Prepare Data for Editor
            # We use a helper dict to map back to full paths later, but show simple table
            df_data = []
            for p in pairs:
                pair_key = p["Pair Key"]
                prefix = p["Prefix"]
                # Default "Process" to True if not set
                process_val = st.session_state.get(f"proc_{pair_key}", True)
                overwrite_val = st.session_state.get(f"over_{pair_key}", False)
                
                df_data.append({
                    "Pair Key": pair_key,
                    "Process": process_val,
                    "Prefix": prefix,
                    "Source": p["Pair Source"],
                    "CODEX File": p["CODEX File"],
                    "H&E File": p["H&E File"],
                    "Anchor": st.session_state.get(f"pair_anchor_{pair_key}", p.get("Auto Reference Channel") or "Channel 0"),
                    "CODEX MPP": float(st.session_state.get(f"pair_codex_mpp_{pair_key}", p["Detected CODEX MPP"])),
                    "H&E MPP": float(st.session_state.get(f"pair_he_mpp_{pair_key}", p["Detected H&E MPP"])),
                    "Overwrite": overwrite_val
                })
            
            df = pd.DataFrame(df_data)

            # Wrap table in a form to prevent reloads on individual check clicks
            with st.form("detected_pairs_form"):
                st.caption("**Tip:** Highlight multiple checkboxes (click first, hold `Shift`, click last) and press `Spacebar` to toggle them instantly offline.")
                edited_df = st.data_editor(
                    df,
                    hide_index=True,
                    column_config={
                        "Process": st.column_config.CheckboxColumn(
                            "Process",
                            help="Select to process this pair",
                            default=True,
                        ),
                        "Overwrite": st.column_config.CheckboxColumn(
                            "Overwrite",
                            help="Overwrite existing output",
                            default=False,
                        ),
                        "Pair Key": st.column_config.TextColumn("Pair Key", disabled=True),
                        "Prefix": st.column_config.TextColumn("Prefix", disabled=True),
                        "Source": st.column_config.TextColumn("Source", disabled=True),
                        "CODEX File": st.column_config.TextColumn("CODEX File", disabled=True),
                        "H&E File": st.column_config.TextColumn("H&E File", disabled=True),
                        "Anchor": st.column_config.TextColumn("Anchor", disabled=True, help="Select the editable anchor channel in the Edit Channels panel for the active pair."),
                        "CODEX MPP": st.column_config.NumberColumn("CODEX MPP", format="%.4f", help="Per-pair CODEX MPP used for registration. Auto-filled from metadata but editable."),
                        "H&E MPP": st.column_config.NumberColumn("H&E MPP", format="%.4f", help="Per-pair H&E MPP used for preprocessing. Auto-filled from metadata but editable."),
                    },
                    disabled=["Pair Key", "Prefix", "Source", "CODEX File", "H&E File", "Anchor"],
                    key="batch_pairs_editor",
                )
                
                col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 1, 2])
                with col_btn_1:
                    select_all = st.form_submit_button("Select All", width='stretch')
                with col_btn_2:
                    deselect_all = st.form_submit_button("Deselect All", width='stretch')
                with col_btn_3:
                    save_table_changes = st.form_submit_button("Save Table Changes", type="primary", width='stretch')
            
            if select_all:
                for index, row in edited_df.iterrows():
                    pair_key = row["Pair Key"]
                    st.session_state[f"proc_{pair_key}"] = True
                    st.session_state[f"over_{pair_key}"] = row["Overwrite"]
                    st.session_state[f"pair_codex_mpp_{pair_key}"] = float(row["CODEX MPP"])
                    st.session_state[f"pair_he_mpp_{pair_key}"] = float(row["H&E MPP"])
                st.rerun()
            elif deselect_all:
                for index, row in edited_df.iterrows():
                    pair_key = row["Pair Key"]
                    st.session_state[f"proc_{pair_key}"] = False
                    st.session_state[f"over_{pair_key}"] = row["Overwrite"]
                    st.session_state[f"pair_codex_mpp_{pair_key}"] = float(row["CODEX MPP"])
                    st.session_state[f"pair_he_mpp_{pair_key}"] = float(row["H&E MPP"])
                st.rerun()
            elif save_table_changes:
                # Sync edits back to session state vars used by run loop
                for index, row in edited_df.iterrows():
                    pair_key = row["Pair Key"]
                    st.session_state[f"proc_{pair_key}"] = row["Process"]
                    st.session_state[f"over_{pair_key}"] = row["Overwrite"]
                    st.session_state[f"pair_codex_mpp_{pair_key}"] = float(row["CODEX MPP"])
                    st.session_state[f"pair_he_mpp_{pair_key}"] = float(row["H&E MPP"])
                st.success("Table changes saved successfully!")
                st.rerun()
            
            st.markdown("---")

            # --- Visualization Selection ---
            pair_keys = [p["Pair Key"] for p in pairs]
            pair_lookup = {p["Pair Key"]: p for p in pairs}
            
            # Use session state to track selection
            if "active_pair_key" not in st.session_state:
                st.session_state.active_pair_key = pair_keys[0] if pair_keys else None

            # Helper to handle selection change
            def on_pair_select():
                # Trigger load logic if needed
                pass

            selected_pair_key = st.selectbox(
                "Select Pair to Visualize", 
                options=pair_keys,
                index=pair_keys.index(st.session_state.active_pair_key) if st.session_state.active_pair_key in pair_keys else 0,
                key="batch_viz_select",
                on_change=on_pair_select,
                format_func=lambda pair_key: f"{pair_lookup[pair_key]['Prefix']} | HE: {pair_lookup[pair_key]['H&E File']}",
                help="Select a pair to configure channels and preview."
            )
            st.session_state.active_pair_key = selected_pair_key
            
            # Load Logic (Auto-load on select if not already loaded)
            if selected_pair_key:
                p_data = pair_lookup.get(selected_pair_key)
                if p_data:
                    try:
                        _ensure_batch_pair_config(selected_pair_key, p_data)
                    except Exception as e:
                        st.error(f"Error loading {p_data['Prefix']}: {e}")

            # Channel Editor for Active Pair
            if selected_pair_key and selected_pair_key in st.session_state.pair_configs:
                active_pair = pair_lookup[selected_pair_key]
                active_config = st.session_state.pair_configs[selected_pair_key]
                anchor_key = f"pair_anchor_{selected_pair_key}"
                resolved_anchor = _ensure_anchor_value(
                    anchor_key,
                    active_config["codex"],
                    preferred_name=active_pair.get("Auto Reference Channel"),
                )

                with st.expander(f"Edit Channels: {active_pair['Prefix']}", expanded=True):
                    if "batch_auto_apply_pair_settings" not in st.session_state:
                        st.session_state["batch_auto_apply_pair_settings"] = True
                    auto_apply_all = st.checkbox(
                        "Auto apply active pair settings to all pairs",
                        key="batch_auto_apply_pair_settings",
                        help="When enabled, changes to the active pair's channel names, colors, export/show flags, anchor, and per-pair MPP values propagate to all detected pairs.",
                    )
                    st.caption(
                        f"Current anchor: {resolved_anchor} | "
                        f"Auto detection: {active_pair.get('Auto Reference Channel') or 'Channel 0'} | "
                        "Choose the anchor in the CODEX channel table below."
                    )

                    # Tucked away schema utilities collapsible section
                    with st.expander("Schema Templates & Color Utilities", expanded=False):
                        schema_col1, schema_col2 = st.columns(2)

                        with schema_col1:
                            st.markdown("**CODEX Schema Tools**")
                            codex_schema_upload = st.file_uploader(
                                "Load CODEX schema CSV",
                                type=["csv"],
                                key=f"batch_codex_schema_upload_{selected_pair_key}",
                                help="CSV columns: channel_index or original_name, new_name, optional color_hex, export, show. This is useful when source OME files have uninformative channel labels.",
                            )
                            codex_upload_token_key = f"batch_codex_schema_upload_token_{selected_pair_key}"
                            codex_upload_token = _uploaded_file_token(codex_schema_upload)
                            if codex_schema_upload is None:
                                st.session_state.pop(codex_upload_token_key, None)
                            elif codex_upload_token != st.session_state.get(codex_upload_token_key):
                                try:
                                    codex_schema_rows = _parse_channel_schema_csv(codex_schema_upload)
                                    _apply_schema_to_container(active_config["codex"], codex_schema_rows, f"batch_{selected_pair_key}_codex")
                                    _store_schema_template("batch_codex_schema_template", active_config["codex"], codex_schema_rows)
                                    st.session_state[codex_upload_token_key] = codex_upload_token
                                    st.success("Applied CODEX schema CSV to the active pair.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Could not apply CODEX schema CSV: {exc}")

                            st.download_button(
                                "Download CODEX schema CSV",
                                data=_schema_rows_to_csv(_schema_rows_from_container(active_config["codex"])),
                                file_name=f"{active_pair['Prefix']}_codex_schema.csv",
                                mime="text/csv",
                                key=f"download_batch_codex_schema_{selected_pair_key}",
                                width='stretch',
                            )

                            if st.button("Apply CODEX schema to all pairs", key=f"apply_batch_codex_schema_{selected_pair_key}", width='stretch'):
                                schema_rows = _schema_rows_from_container(active_config["codex"])
                                _store_schema_template("batch_codex_schema_template", active_config["codex"], schema_rows)
                                for pair_data_key, pair_data in st.session_state.pair_configs.items():
                                    _apply_schema_to_container(pair_data["codex"], schema_rows, f"batch_{pair_data_key}_codex")
                                st.success("Stored the active CODEX schema as the batch default and applied it to loaded pairs.")

                            _render_distinct_color_tools(
                                active_config["codex"],
                                f"batch_{selected_pair_key}_codex",
                                "CODEX",
                                session_template_key="batch_codex_schema_template",
                            )

                        with schema_col2:
                            st.markdown("**H&E Schema Tools**")
                            he_schema_upload = st.file_uploader(
                                "Load H&E schema CSV",
                                type=["csv"],
                                key=f"batch_he_schema_upload_{selected_pair_key}",
                                help="Use this if H&E channels need fixed output names or colors. For RGB H&E, channel_index 0, 1, 2 usually correspond to red, green, blue.",
                            )
                            he_upload_token_key = f"batch_he_schema_upload_token_{selected_pair_key}"
                            he_upload_token = _uploaded_file_token(he_schema_upload)
                            if he_schema_upload is None:
                                st.session_state.pop(he_upload_token_key, None)
                            elif he_upload_token != st.session_state.get(he_upload_token_key):
                                try:
                                    he_schema_rows = _parse_channel_schema_csv(he_schema_upload)
                                    _apply_schema_to_container(active_config["he"], he_schema_rows, f"batch_{selected_pair_key}_he")
                                    _store_schema_template("batch_he_schema_template", active_config["he"], he_schema_rows)
                                    st.session_state[he_upload_token_key] = he_upload_token
                                    st.success("Applied H&E schema CSV to the active pair.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Could not apply H&E schema CSV: {exc}")

                            st.download_button(
                                "Download H&E schema CSV",
                                data=_schema_rows_to_csv(_schema_rows_from_container(active_config["he"])),
                                file_name=f"{active_pair['Prefix']}_he_schema.csv",
                                mime="text/csv",
                                key=f"download_batch_he_schema_{selected_pair_key}",
                                width='stretch',
                            )

                            if st.button("Apply H&E schema to all pairs", key=f"apply_batch_he_schema_{selected_pair_key}", width='stretch'):
                                schema_rows = _schema_rows_from_container(active_config["he"])
                                _store_schema_template("batch_he_schema_template", active_config["he"], schema_rows)
                                for pair_data_key, pair_data in st.session_state.pair_configs.items():
                                    _apply_schema_to_container(pair_data["he"], schema_rows, f"batch_{pair_data_key}_he")
                                st.success("Stored the active H&E schema as the batch default and applied it to loaded pairs.")

                            _render_distinct_color_tools(
                                active_config["he"],
                                f"batch_{selected_pair_key}_he",
                                "H&E",
                                session_template_key="batch_he_schema_template",
                            )

                    render_custom_channel_editor(
                        active_config["codex"],
                        f"batch_{selected_pair_key}_codex",
                        anchor_state_key=anchor_key,
                        preferred_anchor_name=active_pair.get("Auto Reference Channel"),
                    )
                    render_custom_channel_editor(active_config["he"], f"batch_{selected_pair_key}_he")

                if auto_apply_all:
                    current_signature = _pair_settings_signature(
                        selected_pair_key,
                        active_config["codex"],
                        active_config["he"],
                    )
                    signature_state_key = f"batch_pair_settings_signature_{selected_pair_key}"
                    auto_apply_was_enabled = st.session_state.get("batch_auto_apply_pair_settings_prev", True)
                    if (not auto_apply_was_enabled) or st.session_state.get(signature_state_key) != current_signature:
                        _apply_active_pair_settings_to_all_pairs(
                            selected_pair_key,
                            pair_lookup,
                            active_config["codex"],
                            active_config["he"],
                        )
                        st.session_state[signature_state_key] = current_signature
                else:
                    st.session_state[f"batch_pair_settings_signature_{selected_pair_key}"] = _pair_settings_signature(
                        selected_pair_key,
                        active_config["codex"],
                        active_config["he"],
                    )
                st.session_state["batch_auto_apply_pair_settings_prev"] = auto_apply_all
        
        st.markdown("---")
        
        # Output Container
        output_container = st.container()

        # RUN Logic
        if st.button("Run Registration", type="primary", disabled=not pairs):
            # Sync any unsaved edits from the data editor first!
            if "batch_pairs_editor" in st.session_state and st.session_state["batch_pairs_editor"]:
                editor_state = st.session_state["batch_pairs_editor"]
                edited_rows = editor_state.get("edited_rows", {})
                for row_idx_str, col_edits in edited_rows.items():
                    row_idx = int(row_idx_str)
                    if row_idx < len(pairs):
                        pair_key = pairs[row_idx]["Pair Key"]
                        if "Process" in col_edits:
                            st.session_state[f"proc_{pair_key}"] = col_edits["Process"]
                        if "Overwrite" in col_edits:
                            st.session_state[f"over_{pair_key}"] = col_edits["Overwrite"]
                        if "CODEX MPP" in col_edits:
                            st.session_state[f"pair_codex_mpp_{pair_key}"] = float(col_edits["CODEX MPP"])
                        if "H&E MPP" in col_edits:
                            st.session_state[f"pair_he_mpp_{pair_key}"] = float(col_edits["H&E MPP"])

            selected_pairs = [
                pair for pair in pairs
                if st.session_state.get(f"proc_{pair['Pair Key']}", False)
            ]
            preflight_errors = validate_run_directories(
                {
                    "CODEX input directory": codex_dir,
                    "H&E input directory": he_dir,
                },
                output_dir,
            )
            if not selected_pairs:
                preflight_errors.append("Select at least one matched CODEX and H&E pair to process.")
            else:
                selected_paths = [
                    file_path
                    for pair in selected_pairs
                    for file_path in (pair["CODEX Path"], pair["H&E Path"])
                ]
                preflight_errors.extend(
                    validate_selected_files(selected_paths, item_name="registration input")
                )

                invalid_mpp_pairs = [
                    pair["Prefix"]
                    for pair in selected_pairs
                    if float(st.session_state.get(f"pair_codex_mpp_{pair['Pair Key']}", mpp_codex)) <= 0
                    or float(st.session_state.get(f"pair_he_mpp_{pair['Pair Key']}", mpp_he)) <= 0
                ]
                if invalid_mpp_pairs:
                    preflight_errors.append(
                        "CODEX and H&E MPP values must be greater than zero for: "
                        + ", ".join(invalid_mpp_pairs[:3])
                    )

            intermediate_path = Path(intermediate_subdir.strip()) if intermediate_subdir.strip() else None
            if intermediate_path and (intermediate_path.is_absolute() or ".." in intermediate_path.parts):
                preflight_errors.append(
                    "Intermediate Images Subfolder must be a folder name inside the output directory."
                )

            if preflight_errors:
                record_process_event(
                    "tab1",
                    "Registration preflight failed",
                    status="warning",
                    details={"errors": preflight_errors},
                )
                show_preflight_errors(preflight_errors)
                return

            record_process_event(
                "tab1",
                "Registration started",
                details={
                    "selected_pairs": len(selected_pairs),
                    "codex_directory": codex_dir,
                    "he_directory": he_dir,
                    "output_directory": output_dir,
                },
            )

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a dedicated status container for the "Outline" of steps
            step_status = st.status("Initializing...", expanded=True)

            os.makedirs(output_dir, exist_ok=True)
            
            def log_step(msg: str):
                ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                full_msg = f"{ts} {msg}"
                _log_console(msg)
                step_status.write(full_msg) # UI Log
                status_text.text(msg) # Simple text status
                record_process_event("tab1", "Registration step", message=msg)

            # --- Logging setup (optional) ---
            session_dir: Optional[str] = None
            pairs_details_path: Optional[str] = None
            session_stats_path: Optional[str] = None
            pairs_details_fp: Optional[TextIO] = None
            pairs_writer: Optional[csv.DictWriter] = None
            pairs_header: Optional[List[str]] = None
            logging_failed = False
            session_start_dt = datetime.now()
            session_start_ts = session_start_dt.strftime("%Y-%m-%d")
            session_start_time = session_start_dt.strftime("%H:%M:%S")
            session_start_epoch = time.time()

            if create_log:
                log_root = os.path.join(output_dir, "log")
                os.makedirs(log_root, exist_ok=True)
                session_dir = os.path.join(log_root, f"session_{session_start_dt.strftime('%y%m%d_%H%M%S')}")
                os.makedirs(session_dir, exist_ok=True)

                pairs_details_path = os.path.join(session_dir, "pairs_details.csv")
                session_stats_path = os.path.join(session_dir, "session_stats.csv")

                # Delay header creation until first row so we can guarantee consistent fields.
                try:
                    pairs_details_fp = open(pairs_details_path, "w", newline="", encoding="utf-8")
                except OSError as exc:
                    logging_failed = True
                    create_log = False
                    pairs_details_fp = None
                    log_step(f"Logging disabled: could not create session log files ({exc}).")
            
            processed_count = 0
            for i, p in enumerate(pairs):
                prefix = p["Prefix"]
                pair_key = p["Pair Key"]
                if not st.session_state.get(f"proc_{pair_key}", False):
                    continue
                
                start_time = time.time()
                step_status.update(label=f"Processing {prefix}...", state="running")
                log_step(f"Starting processing for {prefix}")

                pair_row: Dict[str, str] = {}
                now_dt = datetime.now()
                pair_row.update(
                    {
                        "Prefix": str(prefix),
                        "Date": now_dt.strftime("%Y-%m-%d"),
                        "Time": now_dt.strftime("%H:%M:%S"),
                        "CODEX_Input": os.path.abspath(p["CODEX Path"]),
                        "HE_Input": os.path.abspath(p["H&E Path"]),
                    }
                )

                # Input metadata (best-effort)
                if create_log:
                    codex_meta = get_image_metadata_log(p["CODEX Path"])
                    he_meta = get_image_metadata_log(p["H&E Path"])
                    pair_row.update(_prefix_keys("CODEX_", codex_meta))
                    pair_row.update(_prefix_keys("HE_", he_meta))

                # Predict final output path for logging/verification
                final_out = os.path.join(output_dir, f"{prefix}_merged.ome.tiff")
                pair_row["Output_Merged"] = os.path.abspath(final_out)
                
                try:
                    pair_config = _ensure_batch_pair_config(pair_key, p)
                    codex_container = pair_config["codex"] if pair_config else None
                    he_container = pair_config["he"] if pair_config else None
                    c_conf = codex_container.get_export_config() if codex_container else None
                    h_conf = he_container.get_export_config() if he_container else None
                    
                    overwrite = st.session_state.get(f"over_{pair_key}", False)
                    selected_anchor = _selected_reference_channel_name(st.session_state.get(f"pair_anchor_{pair_key}"))
                    anchor_label = selected_anchor or p.get("Auto Reference Channel") or "Channel 0"

                    # 1. Preprocess HE
                    log_step(f"[{prefix}] Preprocessing H&E...")
                    pair_codex_mpp = float(st.session_state.get(f"pair_codex_mpp_{pair_key}", p.get("Detected CODEX MPP", _auto_mpp_value(p["CODEX Path"], mpp_codex))))
                    pair_he_mpp = float(st.session_state.get(f"pair_he_mpp_{pair_key}", p.get("Detected H&E MPP", _auto_mpp_value(p["H&E Path"], mpp_he))))
                    he_rotation = float(st.session_state.get(f"batch_he_rotation_{pair_key}", 0.0))
                    if abs(he_rotation) > 0.01:
                        log_step(f"[{prefix}] H&E pre-rotation: {he_rotation:.0f}°")
                    
                    # Compute intermediate dir
                    intermediate_subdir_name = _intermediate_subdir_name("tab1_batch_intermediate_subdir")
                    intermediate_dir = os.path.join(output_dir, intermediate_subdir_name)
                    os.makedirs(intermediate_dir, exist_ok=True)

                    he_resized_path = preprocess_he_image(p["H&E Path"], intermediate_dir, {'hne': pair_he_mpp, 'codex': pair_codex_mpp}, overwrite=overwrite, pre_rotation_deg=he_rotation)
                    
                    # 2. Extract reference
                    log_step(f"[{prefix}] Extracting registration reference...")
                    log_step(f"[{prefix}] Using anchor marker: {anchor_label}")
                    dapi_path = extract_dapi_channel(
                        p["CODEX Path"],
                        intermediate_dir,
                        overwrite=overwrite,
                        reference_channel=selected_anchor,
                    )
                    
                    # 3. Register HE
                    log_step(f"[{prefix}] Registering H&E to {anchor_label}...")
                    he_reg_path = register_images(
                        he_resized_path,
                        dapi_path,
                        intermediate_dir,
                        {'hne': pair_codex_mpp, 'codex': pair_codex_mpp},
                        overwrite=overwrite,
                        reference_channel=selected_anchor,
                        n_keypoints=int(alignment_n_keypoints),
                    )
                    
                    # 4. Merge
                    log_step(f"[{prefix}] Merging stacks...")
                    images_to_merge = [p["CODEX Path"], he_reg_path]
                    
                    # Fix H&E config for the registered file (which loses original channel names)
                    final_h_conf = None
                    if he_container is not None:
                        final_h_conf = _build_registered_he_export_config(he_container, he_reg_path)
                    
                    configs = [c_conf, final_h_conf] if c_conf else None
                    
                    merge_multi_round_ome_tiff(images_to_merge, final_out, mpp_codex, configs, overwrite=overwrite)
                    
                    processed_count += 1
                    elapsed = time.time() - start_time
                    log_step(f"[{prefix}] Done in {elapsed:.2f}s")

                    output_metadata = get_image_metadata_log(final_out) if os.path.exists(final_out) else {}
                    record_process_event(
                        "tab1",
                        "Registration pair complete",
                        status="complete",
                        duration_s=elapsed,
                        details={
                            "pair": prefix,
                            "codex_input": p["CODEX Path"],
                            "he_input": p["H&E Path"],
                            "output": final_out,
                            "output_metadata": output_metadata,
                        },
                    )

                    pair_row["Processing_Time_Sec"] = f"{elapsed:.3f}"
                    pair_row["Status"] = "Success"
                    pair_row["Output_Registered_HE"] = os.path.abspath(he_reg_path)

                    # Output verification metadata
                    if create_log and os.path.exists(final_out):
                        out_meta = get_image_metadata_log(final_out)
                        pair_row.update(_prefix_keys("Output_", out_meta))
                    elif create_log:
                        pair_row.update(_prefix_keys("Output_", get_image_metadata_log(final_out)))
                    
                    # Show results in output container
                    with output_container:
                        st.success(f"{prefix} processed in {elapsed/60:.2f} min")
                        
                        # Display QC Thumbnails
                        qc_col1, qc_col2 = st.columns(2)
                        
                        # Reference thumbnail
                        dapi_thumb = load_reference_thumbnail(dapi_path, reference_channel=selected_anchor)
                        if dapi_thumb is not None:
                            qc_col1.image(dapi_thumb, caption=f"{prefix} Reference ({anchor_label})", width="stretch") 

                        # Registered HE Thumbnail
                        he_reg_thumb = load_thumbnail(he_reg_path, is_codex=False)
                        if he_reg_thumb is not None:
                            qc_col2.image(he_reg_thumb, caption=f"{prefix} H&E (Registered)", width="stretch")
                        
                        st.markdown("---")

                except Exception as e:
                    # st.error(f"Error processing {prefix}: {e}") # Let log_step handle visual error if needed
                    log_step(f"ERROR processing {prefix}: {e}")

                    elapsed = time.time() - start_time
                    pair_row["Processing_Time_Sec"] = f"{elapsed:.3f}"
                    pair_row["Status"] = f"{type(e).__name__}: {e}"
                    record_process_event(
                        "tab1",
                        "Registration pair failed",
                        message=str(e),
                        status="error",
                        duration_s=elapsed,
                        details={"pair": prefix, "codex_input": p["CODEX Path"], "he_input": p["H&E Path"]},
                    )

                    if create_log:
                        # Attempt output metadata even on failure (may help diagnose partial outputs)
                        if os.path.exists(final_out):
                            out_meta = get_image_metadata_log(final_out)
                            pair_row.update(_prefix_keys("Output_", out_meta))
                        else:
                            pair_row.update(_prefix_keys("Output_", {k: "Not Created" for k in get_image_metadata_log(p["CODEX Path"]).keys()}))
                finally:
                    # Write pair logs immediately for real-time monitoring
                    if create_log and session_dir and not logging_failed:
                        try:
                            # Ensure header exists for master log
                            if pairs_writer is None and pairs_details_fp is not None:
                                pairs_header = list(pair_row.keys())
                                pairs_writer = csv.DictWriter(pairs_details_fp, fieldnames=pairs_header)
                                pairs_writer.writeheader()

                            # Keep column set consistent; fill missing keys if needed
                            if pairs_writer is not None and pairs_header is not None and pairs_details_fp is not None:
                                safe_row = {k: pair_row.get(k, "") for k in pairs_header}
                                pairs_writer.writerow(safe_row)
                                pairs_details_fp.flush()

                            # Per-pair log file
                            pair_log_path = os.path.join(session_dir, f"pair_{prefix}.csv")
                            with open(pair_log_path, "w", newline="", encoding="utf-8") as fp:
                                w = csv.DictWriter(fp, fieldnames=list(pair_row.keys()))
                                w.writeheader()
                                w.writerow(pair_row)
                        except OSError as exc:
                            logging_failed = True
                            create_log = False
                            if pairs_details_fp is not None:
                                try:
                                    pairs_details_fp.close()
                                except Exception:
                                    pass
                                pairs_details_fp = None
                            log_step(f"Logging disabled after filesystem error: {exc}")
                
                if len(pairs) > 0:
                    progress_bar.progress(min((i + 1) / len(pairs), 1.0))
            
            step_status.update(label="Registration Completed!", state="complete", expanded=False)
            status_text.text("Processing complete!")
            st.success(f"Processed {processed_count} pairs.")
            st.toast("Registration Run Finished!")
            total_elapsed_sec = time.time() - session_start_epoch
            record_process_event(
                "tab1",
                "Registration complete",
                status="complete",
                duration_s=total_elapsed_sec,
                details={"pairs_processed": processed_count, "pairs_selected": len(selected_pairs), "output_directory": output_dir},
            )

            # Close master log file if open
            if pairs_details_fp is not None:
                try:
                    pairs_details_fp.close()
                except Exception:
                    pass

            # Session summary stats
            if create_log and session_dir and session_stats_path and not logging_failed:
                total_pairs_detected = len(pairs)
                stats_row = {
                    "Session_Date": session_start_ts,
                    "Session_Start_Time": session_start_time,
                    "Session_Folder": os.path.abspath(session_dir),
                    "Pairs_Detected": str(total_pairs_detected),
                    "Pairs_Processed_Success": str(processed_count),
                    "Total_Time_Min": f"{(total_elapsed_sec / 60.0):.3f}",
                    "Total_Time_Hr": f"{(total_elapsed_sec / 3600.0):.4f}",
                    "Input_CODEX_Dir": os.path.abspath(codex_dir),
                    "Input_HE_Dir": os.path.abspath(he_dir),
                    "Output_Dir": os.path.abspath(output_dir),
                    "Pairs_Details_CSV": os.path.abspath(pairs_details_path) if pairs_details_path else "",
                }

                try:
                    with open(session_stats_path, "w", newline="", encoding="utf-8") as fp:
                        w = csv.DictWriter(fp, fieldnames=list(stats_row.keys()))
                        w.writeheader()
                        w.writerow(stats_row)
                except OSError as exc:
                    log_step(f"Could not write session summary log: {exc}")

    # Right Column: Visualization
    with right_col:
        st.subheader("Visualization")
        if st.session_state.active_pair_key and st.session_state.active_pair_key in st.session_state.pair_configs:
            pair_key = st.session_state.active_pair_key
            config = st.session_state.pair_configs[pair_key]
            
            if st.session_state.active_pair_key:
                 st.markdown(f"**Previewing:** `{pair_key}`")
            
            # Find active pair path
            active_pair_data = next((p for p in pairs if p["Pair Key"] == pair_key), None)
            
            if active_pair_data:
                prefix = active_pair_data["Prefix"]
                current_codex_mpp = float(st.session_state.get(f"pair_codex_mpp_{pair_key}", active_pair_data["Detected CODEX MPP"]))
                current_he_mpp = float(st.session_state.get(f"pair_he_mpp_{pair_key}", active_pair_data["Detected H&E MPP"]))
                current_anchor = st.session_state.get(f"pair_anchor_{pair_key}") or active_pair_data.get("Auto Reference Channel") or "Channel 0"
                st.caption(
                    f"Pair source: {active_pair_data['Pair Source']} | "
                    f"Anchor: {current_anchor} | "
                    f"Current MPPs: CODEX {current_codex_mpp:.4f}, "
                    f"H&E {current_he_mpp:.4f}"
                )
                # CODEX Visualization
                st.markdown("**CODEX Preview**")
                codex_cont = config["codex"]
                show_indices = [i for i, ch in enumerate(codex_cont.channels) if ch.show]
                
                if show_indices:
                    # Load specific channels
                    img_data, shape, is_rgb = load_preview_channels(str(codex_cont.path), show_indices)
                    if img_data:
                        # Map index to hex color
                        idx_to_color = {i: codex_cont.channels[i].color_hex for i in show_indices}
                        comp = create_composite_thumbnail(img_data, idx_to_color)
                        st.image(comp, width="stretch", caption=f"{prefix} CODEX Composite")
                    else:
                        st.warning("Could not load channel data.")
                else:
                    st.info("Select 'Show' on channels to visualize.")

                # HE Visualization
                st.markdown("**H&E Preview**")
                he_cont = config["he"]
                he_rot_key = f"batch_he_rotation_{pair_key}"
                if he_rot_key not in st.session_state:
                    st.session_state[he_rot_key] = 0.0
                st.number_input(
                    "H&E Pre-Rotation (°)", min_value=-180.0, max_value=180.0, step=1.0,
                    value=0.0,
                    key=he_rot_key,
                    help="Rotate H&E before registration (counter-clockwise). Pre-processing only — registration method is unchanged.",
                )
                thumb = load_thumbnail(str(he_cont.path), is_codex=False)
                if thumb is not None:
                    he_rot_preview = float(st.session_state.get(he_rot_key, 0.0))
                    if abs(he_rot_preview) > 0.01:
                        thumb = _apply_preview_rotation(thumb, he_rot_preview)
                    rot_label = f" (rotation: {he_rot_preview:.0f}°)" if abs(he_rot_preview) > 0.01 else ""
                    st.image(thumb, width="stretch", caption=f"{prefix} H&E{rot_label}")
                else:
                    st.warning("Could not load H&E thumbnail.")
        else:
             if pairs:
                 st.info("Select a pair to visualize.")
             else:
                 st.info("Load images to visualize.")


@st.fragment
def render_multi_stack_registration(project_root: Path):
    """Sub-tab 2: Multi CODEX & HE registration (N rounds + optional H&E)."""
    st.markdown("### Multi CODEX & HE Registration")
    st.markdown("""
    Register multiple CODEX rounds and an optional H&E image to a single anchor CODEX (R1),
    then merge everything into one multi-channel OME-TIFF.
    Use **Pre-Rotation** on any moving image to correct for camera/stage rotation before palom alignment.
    """)
    
    # Layout: Left (Inputs) | Right (Visualization)
    left_col, right_col = st.columns([1.2, 0.8])

    # Initialize session state
    if "multi_r1_container" not in st.session_state:
        st.session_state.multi_r1_container = None
    if "multi_he_container" not in st.session_state:
        st.session_state.multi_he_container = None
    if "multi_n_extra_codex_rounds" not in st.session_state:
        st.session_state["multi_n_extra_codex_rounds"] = 1

    n_extra = int(st.session_state.get("multi_n_extra_codex_rounds", 1))

    def get_image_files(directory):
        if not directory or not os.path.isdir(directory):
            return []
        return sorted([f for f in os.listdir(directory) if is_supported_image_file(f)])

    def _sync_selected_mpp(dir_key: str, file_key: str, mpp_key: str, fallback: float):
        directory = st.session_state.get(dir_key)
        filename = st.session_state.get(file_key)
        if directory and filename:
            st.session_state[mpp_key] = _auto_mpp_value(os.path.join(directory, filename), fallback)

    # Apply any pending MPP updates BEFORE widgets are instantiated.
    # Load handlers write to _pending_<key> then call st.rerun(), so on the
    # next (fragment) run this block applies them before any number_input is created.
    for _pk in [k for k in st.session_state if k.startswith("_pending_")]:
        _rk = _pk[len("_pending_"):]
        st.session_state[_rk] = st.session_state.pop(_pk)

    with left_col:
        # ---------------------------------------------------------------- R1 Anchor
        st.subheader("1. CODEX R1 (Anchor)")
        r1_dir = directory_input_with_browse(
            label="R1 Directory",
            text_key="multi_r1_dir",
            default_value=str(project_root / "data/codex"),
            browser_key_prefix="multi_r1_dir",
            start_dir_fallback=str(project_root),
        )
        r1_files = get_image_files(r1_dir)

        def on_r1_change():
            st.session_state.multi_r1_container = None
            _sync_selected_mpp("multi_r1_dir", "multi_r1_file", "multi_r1_mpp", 0.5083)

        r1_file = (
            st.selectbox("R1 Image", options=r1_files, key="multi_r1_file", on_change=on_r1_change)
            if r1_files else None
        )
        if not r1_files:
            st.warning("No images found in R1 directory.")

        if r1_file:
            _sync_selected_mpp("multi_r1_dir", "multi_r1_file", "multi_r1_mpp", 0.5083)
        _ensure_float_session_value("multi_r1_mpp", 0.5083)

        r1_mpp = st.number_input(  # noqa: F841 – value read via session state in run loop
            "R1 MPP", format="%.4f", key="multi_r1_mpp",
            help="Auto-filled from image metadata. Edit for a manual override.",
        )

        if st.button("Load R1", key="btn_load_r1", disabled=not r1_file):
            assert r1_file is not None
            r1_path_load = os.path.join(r1_dir, r1_file)
            st.session_state.multi_r1_container = ImageContainer.from_path(r1_path_load, "codex")
            _apply_stored_schema_template(
                st.session_state.multi_r1_container,
                st.session_state.get("multi_r1_schema_template"),
                "multi_r1",
            )
            if st.session_state.multi_r1_container.mpp:
                st.session_state["_pending_multi_r1_mpp"] = float(st.session_state.multi_r1_container.mpp)
            _ensure_anchor_value("multi_r1_anchor", st.session_state.multi_r1_container)
            st.rerun()

        if st.session_state.multi_r1_container:
            st.success("Loaded")
            cur_r1_anchor = _ensure_anchor_value("multi_r1_anchor", st.session_state.multi_r1_container)
            st.caption(f"R1 anchor: {cur_r1_anchor}. Change it in the R1 channel table below.")
            render_schema_tools(
                st.session_state.multi_r1_container, "multi_r1", "R1",
                "CSV: channel_index or original_name, new_name, color_hex, export, show.",
                session_template_key="multi_r1_schema_template",
            )
            with st.expander("R1 Channels", expanded=False):
                render_custom_channel_editor(
                    st.session_state.multi_r1_container, "multi_r1",
                    compact=True, anchor_state_key="multi_r1_anchor",
                )

        st.markdown("---")

        # ---------------------------------------------------------------- Extra CODEX Rounds
        st.subheader("Extra CODEX Rounds (Moving)")

        for i in range(n_extra):
            round_label = f"R{i + 2}"
            sk = f"multi_codex_extra_{i}"

            if f"{sk}_container" not in st.session_state:
                st.session_state[f"{sk}_container"] = None
            # Migrate old R2 state into slot 0 on first upgrade
            if i == 0 and f"{sk}_dir" not in st.session_state and "multi_r2_dir" in st.session_state:
                st.session_state[f"{sk}_dir"] = st.session_state["multi_r2_dir"]
                st.session_state[f"{sk}_file"] = st.session_state.get("multi_r2_file")
                st.session_state[f"{sk}_mpp"] = st.session_state.get("multi_r2_mpp", 0.5083)
                if st.session_state.get("multi_r2_container") and st.session_state[f"{sk}_container"] is None:
                    st.session_state[f"{sk}_container"] = st.session_state["multi_r2_container"]

            st.markdown(f"**{round_label} (Moving)**")

            ri_dir = directory_input_with_browse(
                label=f"{round_label} Directory",
                text_key=f"{sk}_dir",
                default_value=str(project_root / "data/codex"),
                browser_key_prefix=f"{sk}_dir",
                start_dir_fallback=str(project_root),
            )
            ri_files = get_image_files(ri_dir)

            def _make_round_change_cb(slot):
                def _cb():
                    st.session_state[f"{slot}_container"] = None
                    _sync_selected_mpp(f"{slot}_dir", f"{slot}_file", f"{slot}_mpp", 0.5083)
                return _cb

            ri_file = (
                st.selectbox(
                    f"{round_label} Image", options=ri_files,
                    key=f"{sk}_file", on_change=_make_round_change_cb(sk),
                )
                if ri_files else None
            )
            if not ri_files:
                st.warning(f"No images found in {round_label} directory.")

            if ri_file:
                _sync_selected_mpp(f"{sk}_dir", f"{sk}_file", f"{sk}_mpp", 0.5083)
            _ensure_float_session_value(f"{sk}_mpp", 0.5083)

            st.number_input(
                f"{round_label} MPP", format="%.4f", key=f"{sk}_mpp",
                help="Auto-filled from image metadata. Edit for a manual override.",
            )

            _ensure_float_session_value(f"{sk}_rotation", 0.0)
            st.number_input(
                f"{round_label} Pre-Rotation (°)",
                min_value=-180.0, max_value=180.0, step=1.0,
                value=0.0,
                key=f"{sk}_rotation",
                help=(
                    f"Rotate {round_label} before palom alignment (counter-clockwise). "
                    "Registration method is unchanged — rotation is pre-processing only."
                ),
            )

            if st.button(f"Load {round_label}", key=f"btn_load_{sk}", disabled=not ri_file):
                assert ri_file is not None
                ri_path_load = os.path.join(ri_dir, ri_file)
                st.session_state[f"{sk}_container"] = ImageContainer.from_path(ri_path_load, "codex")
                _apply_stored_schema_template(
                    st.session_state[f"{sk}_container"],
                    st.session_state.get(f"{sk}_schema_template"),
                    sk,
                )
                if st.session_state[f"{sk}_container"].mpp:
                    st.session_state[f"_pending_{sk}_mpp"] = float(st.session_state[f"{sk}_container"].mpp)
                _ensure_anchor_value(f"{sk}_anchor", st.session_state[f"{sk}_container"])
                st.rerun()

            if st.session_state.get(f"{sk}_container"):
                st.success("Loaded")
                cur_anchor = _ensure_anchor_value(f"{sk}_anchor", st.session_state[f"{sk}_container"])
                st.caption(f"{round_label} anchor: {cur_anchor}.")
                render_schema_tools(
                    st.session_state[f"{sk}_container"], sk, round_label,
                    "CSV: channel_index or original_name, new_name, color_hex, export, show.",
                    session_template_key=f"{sk}_schema_template",
                )
                with st.expander(f"{round_label} Channels", expanded=False):
                    render_custom_channel_editor(
                        st.session_state[f"{sk}_container"], sk,
                        compact=True, anchor_state_key=f"{sk}_anchor",
                    )

            if i < n_extra - 1:
                st.markdown("---")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("+ Add CODEX Round", key="btn_add_codex_round"):
                st.session_state["multi_n_extra_codex_rounds"] = n_extra + 1
                st.rerun()
        with btn_col2:
            if st.button("- Remove Last Round", key="btn_remove_codex_round", disabled=n_extra < 1):
                rm_sk = f"multi_codex_extra_{n_extra - 1}"
                for sfx in ("_container", "_dir", "_file", "_mpp", "_anchor", "_rotation", "_schema_template"):
                    st.session_state.pop(f"{rm_sk}{sfx}", None)
                st.session_state["multi_n_extra_codex_rounds"] = max(0, n_extra - 1)
                st.rerun()

        if "multi_include_he" not in st.session_state:
            st.session_state["multi_include_he"] = True
        include_he = st.checkbox(
            "Include H&E Image", key="multi_include_he",
        )

        if include_he:
            st.subheader("H&E Image")
            he_dir = directory_input_with_browse(
                label="H&E Directory",
                text_key="multi_he_dir",
                default_value=str(project_root / "data/he"),
                browser_key_prefix="multi_he_dir",
                start_dir_fallback=str(project_root),
            )
            he_files = get_image_files(he_dir)

            def on_he_change():
                st.session_state.multi_he_container = None
                _sync_selected_mpp("multi_he_dir", "multi_he_file", "multi_he_mpp", 0.2495)

            he_file = (
                st.selectbox("H&E Image", options=he_files, key="multi_he_file", on_change=on_he_change)
                if he_files else None
            )
            if not he_files:
                st.warning("No images found in H&E directory.")

            if he_file:
                _sync_selected_mpp("multi_he_dir", "multi_he_file", "multi_he_mpp", 0.2495)
            _ensure_float_session_value("multi_he_mpp", 0.2495)

            st.number_input(
                "H&E MPP", format="%.4f", key="multi_he_mpp",
                help="Auto-filled from image metadata. Edit for a manual override.",
            )

            _ensure_float_session_value("multi_he_rotation", 0.0)
            st.number_input(
                "H&E Pre-Rotation (°)",
                min_value=-180.0, max_value=180.0, step=1.0,
                value=0.0,
                key="multi_he_rotation",
                help=(
                    "Rotate H&E before palom alignment (counter-clockwise). "
                    "Registration method is unchanged — rotation is pre-processing only."
                ),
            )

            if st.button("Load H&E", key="btn_load_he", disabled=not he_file):
                assert he_file is not None
                he_path_load = os.path.join(he_dir, he_file)
                st.session_state.multi_he_container = ImageContainer.from_path(he_path_load, "he")
                _apply_stored_schema_template(
                    st.session_state.multi_he_container,
                    st.session_state.get("multi_he_schema_template"),
                    "multi_he",
                )
                if st.session_state.multi_he_container.mpp:
                    st.session_state["_pending_multi_he_mpp"] = float(st.session_state.multi_he_container.mpp)
                st.rerun()

            if st.session_state.multi_he_container:
                st.success("Loaded")
                render_schema_tools(
                    st.session_state.multi_he_container, "multi_he", "H&E",
                    "For RGB H&E, index 0/1/2 = R/G/B. Optional: color_hex, export, show.",
                    session_template_key="multi_he_schema_template",
                )
                with st.expander("H&E Channels", expanded=False):
                    render_custom_channel_editor(st.session_state.multi_he_container, "multi_he", compact=True)

        st.markdown("---")

        # ---------------------------------------------------------------- Output
        st.subheader("Output Configuration")
        output_dir = directory_input_with_browse(
            label="Output Directory",
            text_key="multi_out_dir",
            default_value=str(project_root / "outputs/registered"),
            browser_key_prefix="multi_out_dir",
            start_dir_fallback=str(project_root),
        )

        if "multi_intermediate_subdir" not in st.session_state:
            st.session_state["multi_intermediate_subdir"] = "intermediate_images"
        intermediate_subdir = st.text_input(
            "Intermediate Images Subfolder",
            key="multi_intermediate_subdir",
            help="Subfolder name under the Output Directory where intermediate/temporary files (resized H&E, anchor channels, registered rounds) will be saved."
        )

        r1_file_cur = st.session_state.get("multi_r1_file", "")
        suggested_name = "merged_output.ome.tiff"
        if r1_file_cur:
            base = Path(r1_file_cur).stem
            for rem in ("_codex", "_R1", ".qptiff", ".ome"):
                base = base.replace(rem, "")
            suggested_name = f"{base}_multiround_merged.ome.tiff"

        if "multi_last_suggested_name" not in st.session_state:
            st.session_state["multi_last_suggested_name"] = suggested_name
            st.session_state["multi_out_name"] = suggested_name
        elif st.session_state["multi_last_suggested_name"] != suggested_name:
            st.session_state["multi_last_suggested_name"] = suggested_name
            st.session_state["multi_out_name"] = suggested_name

        output_name = st.text_input("Output Filename", key="multi_out_name")

        _ensure_int_session_value("multi_alignment_n_keypoints", DEFAULT_ALIGNMENT_KEYPOINTS)
        alignment_n_keypoints = st.number_input(
            "Alignment Keypoints", min_value=500, max_value=50000, step=500,
            value=DEFAULT_ALIGNMENT_KEYPOINTS,
            key="multi_alignment_n_keypoints",
            help="Number of SIFT/ORB keypoints for coarse palom registration.",
        )

        if "multi_overwrite" not in st.session_state:
            st.session_state["multi_overwrite"] = False
        overwrite = st.checkbox("Overwrite existing files", key="multi_overwrite")

        # ---------------------------------------------------------------- Readiness
        r1_loaded = st.session_state.multi_r1_container is not None
        extras_loaded = all(
            st.session_state.get(f"multi_codex_extra_{i}_container") is not None
            for i in range(n_extra)
        )
        he_loaded = (not include_he) or (st.session_state.multi_he_container is not None)
        has_something = n_extra > 0 or include_he
        has_output_name = bool(output_name.strip())

        if not has_something:
            st.warning("Add at least one extra CODEX round or enable H&E.")
        if r1_loaded and not extras_loaded and n_extra > 0:
            st.info(f"Load all {n_extra} extra CODEX round(s) to enable registration.")
        if r1_loaded and include_he and not he_loaded:
            st.info("Load H&E (or uncheck 'Include H&E') to enable registration.")
        if r1_loaded and extras_loaded and he_loaded and has_something and not has_output_name:
            st.info("Please enter a valid Output Filename to enable registration.")

        run_disabled = not (r1_loaded and extras_loaded and he_loaded and has_something and has_output_name)

        if st.button("Run Multi-Stack Registration", type="primary", disabled=run_disabled):
            full_output_path = os.path.join(output_dir, output_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Compute intermediate directory path
            intermediate_subdir_name = _intermediate_subdir_name("multi_intermediate_subdir")
            intermediate_dir = os.path.join(output_dir, intermediate_subdir_name)
            os.makedirs(intermediate_dir, exist_ok=True)

            status = st.empty()

            try:
                r1_container = st.session_state.multi_r1_container
                assert r1_container is not None
                r1_path = str(r1_container.path)
                resolved_r1_mpp = float(st.session_state.get("multi_r1_mpp", 0.5083))
                r1_anchor = st.session_state.get("multi_r1_anchor")
                r1_anchor_label = r1_anchor or _auto_reference_channel_name(r1_path) or "Channel 0"

                # Step 1: Extract R1 reference
                status.text(f"Extracting R1 reference ({r1_anchor_label})...")
                r1_dapi_path = extract_dapi_channel(
                    r1_path, intermediate_dir, overwrite=overwrite, reference_channel=r1_anchor,
                )

                # Step 2: Register each extra CODEX round to R1
                reg_paths = []
                extra_configs = []

                for i in range(n_extra):
                    sk = f"multi_codex_extra_{i}"
                    round_label = f"R{i + 2}"
                    ri_cont = st.session_state[f"{sk}_container"]
                    ri_path = str(ri_cont.path)
                    resolved_ri_mpp = float(st.session_state.get(f"{sk}_mpp", 0.5083))
                    ri_anchor = st.session_state.get(f"{sk}_anchor")
                    ri_rotation = float(st.session_state.get(f"{sk}_rotation", 0.0))
                    ri_anchor_label = ri_anchor or _auto_reference_channel_name(ri_path) or "Channel 0"

                    status.text(f"Extracting {round_label} reference ({ri_anchor_label})...")
                    ri_dapi_path = extract_dapi_channel(
                        ri_path, intermediate_dir, overwrite=overwrite, reference_channel=ri_anchor,
                    )

                    base_ri = Path(ri_path).stem
                    for ext in (".ome.tiff", ".ome.tif", ".qptiff", ".tiff", ".tif", "_codex"):
                        base_ri = base_ri.replace(ext, "")
                    ri_reg_path = os.path.join(intermediate_dir, f"{base_ri}_registered_to_R1.ome.tiff")

                    rot_note = f" (pre-rotation {ri_rotation:.0f}°)" if abs(ri_rotation) > 0.01 else ""
                    status.text(f"Registering {round_label} -> R1{rot_note}...")
                    register_codex_to_codex(
                        r1_dapi_path, ri_dapi_path, ri_path, ri_reg_path,
                        {"codex_ref": resolved_r1_mpp, "codex_mov": resolved_ri_mpp},
                        overwrite=overwrite,
                        fixed_reference_channel=r1_anchor,
                        moving_reference_channel=ri_anchor,
                        n_keypoints=int(alignment_n_keypoints),
                        pre_rotation_deg_moving=ri_rotation,
                    )
                    reg_paths.append(ri_reg_path)
                    extra_configs.append(ri_cont.get_export_config())

                # Step 3: Preprocess and register H&E (optional)
                he_reg_path = None
                final_h_conf = None

                if include_he:
                    he_container = st.session_state.multi_he_container
                    assert he_container is not None
                    he_path = str(he_container.path)
                    resolved_he_mpp = float(st.session_state.get("multi_he_mpp", 0.2495))
                    he_rotation = float(st.session_state.get("multi_he_rotation", 0.0))

                    rot_note = f" (pre-rotation {he_rotation:.0f}°)" if abs(he_rotation) > 0.01 else ""
                    status.text(f"Preprocessing H&E{rot_note}...")
                    he_resized_path = preprocess_he_image(
                        he_path, intermediate_dir,
                        {"hne": resolved_he_mpp, "codex": resolved_r1_mpp},
                        overwrite=overwrite,
                        pre_rotation_deg=he_rotation,
                    )

                    status.text(f"Registering H&E -> R1 ({r1_anchor_label})...")
                    he_reg_path = register_images(
                        he_resized_path, r1_dapi_path, intermediate_dir,
                        {"hne": resolved_r1_mpp, "codex": resolved_r1_mpp},
                        overwrite=overwrite,
                        reference_channel=r1_anchor,
                        n_keypoints=int(alignment_n_keypoints),
                    )

                    h_conf = he_container.get_export_config()
                    if h_conf:
                        # H&E container indices map 1:1 to registered file channels.
                        final_h_conf = dict(h_conf)

                # Step 4: Merge all images
                status.text("Merging all stacks...")
                images_to_merge = [r1_path] + reg_paths
                merge_configs = [r1_container.get_export_config()] + extra_configs

                if include_he and he_reg_path:
                    images_to_merge.append(he_reg_path)
                    merge_configs.append(final_h_conf)

                merge_multi_round_ome_tiff(
                    images_to_merge, full_output_path, resolved_r1_mpp,
                    merge_configs, overwrite=overwrite,
                )

                status.text("Done!")
                st.success(f"Registration complete! Output: {full_output_path}")

            except Exception as e:
                st.error(f"Error during registration: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ---------------------------------------------------------------- Right Column: Visualization
    with right_col:
        st.subheader("Visualization")

        if st.session_state.multi_r1_container:
            st.markdown("**R1 (Anchor) Preview**")
            cont = st.session_state.multi_r1_container
            show_indices = [j for j, ch in enumerate(cont.channels) if ch.show]
            if show_indices:
                img_data, _shape, _is_rgb = load_preview_channels(str(cont.path), show_indices)
                if img_data:
                    idx_to_color = {j: cont.channels[j].color_hex for j in show_indices}
                    st.image(create_composite_thumbnail(img_data, idx_to_color), width="stretch", caption="R1 (Anchor)")
            else:
                st.info("Select 'Show' on R1 channels to preview.")

        for i in range(n_extra):
            sk = f"multi_codex_extra_{i}"
            round_label = f"R{i + 2}"
            cont = st.session_state.get(f"{sk}_container")
            if cont:
                ri_rot = float(st.session_state.get(f"{sk}_rotation", 0.0))
                rot_label = f" (rotation: {ri_rot:.0f}°)" if abs(ri_rot) > 0.01 else ""
                st.markdown(f"**{round_label} Preview{rot_label}**")
                show_indices = [j for j, ch in enumerate(cont.channels) if ch.show]
                if show_indices:
                    img_data, _shape, _is_rgb = load_preview_channels(str(cont.path), show_indices)
                    if img_data:
                        idx_to_color = {j: cont.channels[j].color_hex for j in show_indices}
                        comp = create_composite_thumbnail(img_data, idx_to_color)
                        if abs(ri_rot) > 0.01:
                            comp = _apply_preview_rotation(comp, ri_rot)
                        st.image(comp, width="stretch", caption=f"{round_label}{rot_label}")
                else:
                    st.info(f"Select 'Show' on {round_label} channels to preview.")

        if include_he and st.session_state.multi_he_container:
            he_rot = float(st.session_state.get("multi_he_rotation", 0.0))
            rot_label = f" (rotation: {he_rot:.0f}°)" if abs(he_rot) > 0.01 else ""
            st.markdown(f"**H&E Preview{rot_label}**")
            cont = st.session_state.multi_he_container
            thumb = load_thumbnail(str(cont.path), is_codex=False)
            if thumb is not None:
                if abs(he_rot) > 0.01:
                    thumb = _apply_preview_rotation(thumb, he_rot)
                st.image(thumb, width="stretch", caption=f"H&E{rot_label}")
            else:
                st.warning("Could not load H&E thumbnail.")


def render():
    """Main render function for Tab 1."""
    import streamlit as st
    from spacemap_gui.ui_utils import require_active_project
    project_root = require_active_project()
    if project_root is None:
        return
    
    _migrate_tab1_registration_state()
    st.markdown("## Registration Pipeline")
    st.markdown("""
    Perform image registration and merging for spatial analysis.
    *   **CODEX & HE batch registration**: Process multiple CODEX and H&E pairs automatically matching by filename prefix.
    *   **Multi CODEX & HE registration**: Manually register multiple rounds (e.g., separate CODEX runs) and H&E into a single stack.
    """)
    
    tab1, tab2 = st.tabs(["CODEX & HE batch registration", "Multi CODEX & HE registration"])
    
    with tab1:
        render_batch_registration(project_root)
        
    with tab2:
        render_multi_stack_registration(project_root)

    st.markdown("---")
    if st.button("Save Project", key="tab1_save_project", width='stretch'):
        from spacemap_gui.persistence import save_active_project_state
        save_active_project_state("tab1_registration")
        st.success("Project settings saved to configuration file.")

if __name__ == "__main__":
    from spacemap_gui.ui_utils import require_active_project
    require_active_project()
    render()

