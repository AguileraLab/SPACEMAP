"""Direct, lightweight image previews requested through Assistant chat.

This is deliberately a small registered action rather than generated Python.
It reuses the Visualization tab's OME-TIFF reader, reads only a suitable
pyramid level, and writes ordinary PNG previews beside project outputs.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import require_assistant_read_path, require_assistant_write_path


HE_CHANNELS = ("HE_Red", "HE_Green", "HE_Blue")
IMAGE_SUFFIXES = (".ome.tiff", ".ome.tif", ".qptiff", ".tiff", ".tif")
GEOJSON_SUFFIXES = (".geojson", ".json")
SCALE_BAR_UM_CHOICES = (5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0)


class PreviewActionError(RuntimeError):
    """A safe, user-readable preview error."""


def _is_tiff(path: Path) -> bool:
    return path.name.lower().endswith(IMAGE_SUFFIXES)


def is_multi_step_scientific_request(prompt: str) -> bool:
    """Keep workflow prompts out of the small, deterministic preview shortcut.

    A scientific workflow may mention thumbnails, channels, overlays, or a scale
    bar as one of many requested steps.  Those words do not make the whole
    request a thumbnail request.  The shortcut must only handle a direct image
    display/edit request; otherwise it prevents the selected model and action
    planner from seeing the prompt at all.
    """
    lowered = str(prompt).casefold()
    signals = (
        r"\b(?:optimi[sz](?:e|ation)|parameter[\s-]*(?:search|sweep|trial|comparison))\b",
        r"\b(?:test|compare|try)\b.{0,80}\b(?:settings?|parameters?|configurations?)\b",
        r"\b(?:true|false)[\s-]*(?:positive|negative)s?\b",
        r"\b(?:classify|review|assess)\b.{0,80}\b(?:every|each|detected|candidate)?\s*polygons?\b",
        r"\b(?:detect|detection|artifact[\s-]*removal|out[\s-]*of[\s-]*focus)\b",
        r"\b(?:replace|overwrite|correct|fix|edit)\b.{0,100}\bgeojson\b",
        r"\b(?:run|execute|perform|start)\b.{0,100}\b(?:workflow|analysis|detection|optimization|search)\b",
        r"\b(?:best|recommended)\b.{0,80}\b(?:settings?|parameters?|configurations?|result)\b",
    )
    # Require more than one independent workflow clue. This keeps ordinary
    # requests such as "show the detected blur overlay" in the preview route.
    return sum(bool(re.search(pattern, lowered, flags=re.DOTALL)) for pattern in signals) >= 2


def parse_image_preview_request(prompt: str) -> dict[str, Any] | None:
    """Recognize image views and edits such as channels, overlays, and scale bars."""
    cleaned = str(prompt).replace("\\_", "_")
    lowered = cleaned.lower()
    if is_multi_step_scientific_request(cleaned):
        return None
    raw_paths = re.findall(r"(?<![A-Za-z0-9])(/[^\s`\"']+)", cleaned)
    paths = [value.rstrip(".,;:?!)]}") for value in raw_paths]
    view_verb = bool(
        re.search(
            r"(?<![a-z0-9])(?:display|show|view|see|render|make|create|generate|add|put|place|edit|change)(?![a-z0-9])",
            lowered,
        )
    )
    overlay_verb = bool(
        re.search(r"(?<![a-z0-9])(?:overlay|draw|drop)(?![a-z0-9])", lowered)
    )
    thumbnail_noun = bool(
        re.search(r"(?<![a-z0-9])(?:thumbnail|preview)s?(?![a-z0-9])", lowered)
    )
    visual_noun = bool(
        re.search(
            r"(?<![a-z0-9])(?:images?|rois?|tiffs?|layers?|channels?)(?![a-z0-9])",
            lowered,
        )
    )
    overlay_noun = bool(
        re.search(
            r"(?<![a-z0-9])(?:geojson|segmentations?|annotations?)(?![a-z0-9])",
            lowered,
        )
    )
    wants_scale_bar = bool(re.search(r"\bscale[\s-]*bar\b", lowered))
    wants_view = (
        (thumbnail_noun and (view_verb or overlay_verb or bool(paths)))
        or (view_verb and visual_noun)
        or (overlay_verb and (visual_noun or overlay_noun))
    )
    wants_non_he = any(
        term in lowered
        for term in (
            "non-he",
            "non he",
            "non-h&e",
            "non h&e",
            "non h and e",
            "codex layer",
            "codex channel",
            "codex thumbnail",
            "codex preview",
        )
    )
    he_only_text = re.sub(r"\bnon[-\s]?(?:h&e|he|h\s+and\s+e)\b", "", lowered)
    wants_he = bool(
        re.search(
            r"(?<![a-z0-9])(?:h\s*&\s*e|h\s+and\s+e|he[\s_-]+(?:layers?|images?|thumbnails?|previews?|red))(?![a-z0-9])",
            he_only_text,
            flags=re.IGNORECASE,
        )
    )
    if not wants_view and not wants_scale_bar:
        return None
    modes = []
    if wants_he:
        modes.append("he")
    if wants_non_he:
        modes.append("non_he")
    geojson_paths = [
        value for value in paths if value.lower().endswith(GEOJSON_SUFFIXES)
    ]
    source_paths = [
        value for value in paths if not value.lower().endswith(GEOJSON_SUFFIXES)
    ]
    return {
        "source_path": source_paths[0] if source_paths else "",
        "geojson_paths": geojson_paths,
        "modes": modes,
        "request_text": cleaned,
        "max_size": 768,
        "scale_bar": _parse_scale_bar_request(cleaned) if wants_scale_bar else None,
    }


def _parse_scale_bar_request(prompt: str) -> dict[str, Any]:
    """Extract scale-bar settings without mistaking MPP for the bar length."""
    text = str(prompt).replace("\\_", "_")
    lowered = text.casefold()
    number = r"([0-9]+(?:\.[0-9]+)?)"
    unit = r"(?:µm|μm|um|microns?|micromet(?:er|re)s?)"

    mpp: float | None = None
    for pattern in (
        rf"(?:each\s+)?pixel\s*(?:is|=|:|measures?)?\s*{number}\s*{unit}",
        rf"(?:mpp|microns?\s+per\s+pixel|micromet(?:er|re)s?\s+per\s+pixel)\s*(?:is|=|:)?\s*{number}",
        rf"{number}\s*{unit}\s*(?:per|/)\s*pixel",
    ):
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            mpp = float(match.group(1))
            break

    length_um: float | None = None
    for pattern in (
        rf"{number}\s*{unit}\s*(?:long\s+)?scale[\s-]*bar",
        rf"scale[\s-]*bar\s*(?:of|length|measuring|=|:)?\s*{number}\s*{unit}",
    ):
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            length_um = float(match.group(1))
            break

    if any(term in lowered for term in ("bottom left", "bottom-left", "lower left", "left bottom", "left-bottom", "button left", "left button")):
        position = "bottom-left"
    elif any(term in lowered for term in ("bottom right", "bottom-right", "lower right", "right bottom", "right-bottom", "button right")):
        position = "bottom-right"
    elif any(term in lowered for term in ("top left", "top-left", "upper left", "left top", "left-top")):
        position = "top-left"
    elif any(term in lowered for term in ("top right", "top-right", "upper right", "right top", "right-top")):
        position = "top-right"
    else:
        position = "bottom-left"

    return {
        "mpp": mpp,
        "length_um": length_um,
        "position": position,
        "color": "#FFFFFF",
    }


def parse_he_preview_request(prompt: str) -> dict[str, Any] | None:
    """Backward-compatible name for the general image-channel request parser."""
    return parse_image_preview_request(prompt)


def parse_visual_assessment_request(prompt: str) -> dict[str, Any] | None:
    """Recognize a request to visually QC a TIFF against one or more GeoJSONs."""
    cleaned = str(prompt).replace("\\_", "_")
    lowered = cleaned.casefold()
    assessment_verb = bool(
        re.search(
            r"(?<![a-z0-9])(?:check|review|assess|inspect|validate|evaluate|qc)(?![a-z0-9])",
            lowered,
        )
        or re.search(r"(?<![a-z0-9])look\s+at(?![a-z0-9])", lowered)
    )
    if not assessment_verb:
        return None
    raw_paths = re.findall(r"(?<![A-Za-z0-9])(/[^\s`\"']+)", cleaned)
    paths = [value.rstrip(".,;:?!)]}") for value in raw_paths]
    images = [value for value in paths if value.lower().endswith(IMAGE_SUFFIXES)]
    geojsons = [value for value in paths if value.lower().endswith(GEOJSON_SUFFIXES)]
    if not images or not geojsons:
        return None
    he_evidence = bool(
        re.search(
            r"(?<![a-z0-9])(?:h\s*&\s*e|h\s+and\s+e|blood[\s_-]*clot|ha?emorrhage)(?![a-z0-9])",
            lowered,
        )
    )
    return {
        "source_path": images[0],
        "geojson_paths": geojsons,
        "preferred_mode": "he" if he_evidence else "automatic",
        "request_text": cleaned,
        "max_size": 1024,
    }


def match_requested_channels(prompt: str, channel_names: list[str]) -> list[str]:
    """Match any image channel named in natural language, without a fixed marker list."""
    text = str(prompt).replace("\\_", "_")
    matched: list[str] = []
    excluded = {*HE_CHANNELS, "R", "G", "B"}
    for name in channel_names:
        if name in excluded:
            continue
        pieces = [re.escape(piece) for piece in re.findall(r"[A-Za-z0-9]+", str(name))]
        if not pieces:
            continue
        pattern = r"(?<![A-Za-z0-9])" + r"[\W_]*".join(pieces) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            matched.append(str(name))
    return matched


def preview_channel_names(source_path: str | Path) -> list[str]:
    """Read channel names from the first TIFF named by a preview request."""
    _source_dir, images = _source_images(source_path)
    source = images[0]
    signature = source.stat()
    return list(
        _cached_channel_names(str(source), int(signature.st_size), int(signature.st_mtime_ns))
    )


@lru_cache(maxsize=64)
def _cached_channel_names(
    source_path: str,
    _source_size: int,
    _source_mtime_ns: int,
) -> tuple[str, ...]:
    from spacemap_gui.tabs.tab9_visualization.overlay_core import read_ome_channel_names

    return tuple(read_ome_channel_names(source_path))


def _source_images(source_path: str | Path) -> tuple[Path, list[Path]]:
    source = require_assistant_read_path(source_path, must_exist=True)
    if source.is_file():
        if not _is_tiff(source):
            raise PreviewActionError(
                "The preview source must be a TIFF image or a folder of TIFF images."
            )
        return source.parent, [source]
    if not source.is_dir():
        raise PreviewActionError("The preview source is not a readable file or folder.")
    images = sorted(
        path for path in source.iterdir() if path.is_file() and _is_tiff(path)
    )
    if not images:
        raise PreviewActionError("I did not find any TIFF images directly inside that folder.")
    return source, images


def _preview_stem(path: Path) -> str:
    name = path.name
    for suffix in IMAGE_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)] or "image"
    return path.stem or "image"


def _cached_result(
    source: Path,
    destination: Path,
    *,
    preview_mode: str,
    channels: list[str],
) -> dict[str, Any] | None:
    """Reuse a preview only when its embedded source signature still matches."""
    if not destination.is_file():
        return None
    from PIL import Image

    signature = source.stat()
    try:
        with Image.open(destination) as cached:
            metadata = cached.info
            if (
                metadata.get("spacemap_preview_schema") != "2"
                or metadata.get("preview_mode") != preview_mode
                or json.loads(metadata.get("channels_json", "[]")) != channels
                or int(metadata.get("source_size", -1)) != int(signature.st_size)
                or int(metadata.get("source_mtime_ns", -1)) != int(signature.st_mtime_ns)
            ):
                return None
            width, height = cached.size
            return {
                "source_path": str(source),
                "preview_path": str(destination),
                "preview_mode": preview_mode,
                "channels": channels,
                "pyramid_level": int(metadata["pyramid_level"]),
                "full_height": int(metadata["full_height"]),
                "full_width": int(metadata["full_width"]),
                "preview_width": int(width),
                "preview_height": int(height),
                "cached": True,
            }
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _generate_previews(
    source_path: str | Path,
    *,
    preview_mode: str,
    marker_channels: list[str] | None = None,
    channel_names: list[str] | None = None,
    max_size: int = 768,
) -> list[dict[str, Any]]:
    """Create one H&E or representative CODEX PNG per image."""
    if int(max_size) < 128 or int(max_size) > 2_048:
        raise PreviewActionError("Thumbnail size must be between 128 and 2,048 pixels.")

    source_dir, images = _source_images(source_path)
    output_dir = require_assistant_write_path(source_dir / "_spacemap_ai_previews")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imported lazily so ordinary Assistant chat startup does not load the
    # Visualization stack until an image preview is actually requested.
    from spacemap_gui.tabs.tab9_visualization.overlay_core import (
        default_marker_selection,
        load_base_image,
        read_ome_channel_names,
    )

    results: list[dict[str, Any]] = []
    for source in images:
        if preview_mode == "he":
            channels = list(HE_CHANNELS)
            requested_channels = None
            require_he = True
            label = "HE"
        elif preview_mode == "non_he":
            all_channels = list(channel_names or read_ome_channel_names(source))
            excluded = {*HE_CHANNELS, "R", "G", "B"}
            non_he_channels = [name for name in all_channels if name not in excluded]
            channels = default_marker_selection(non_he_channels)
            if not channels:
                channels = non_he_channels[:3]
            if not channels:
                raise PreviewActionError(
                    f"{source.name} does not contain any non-H&E/CODEX channels."
                )
            requested_channels = channels
            require_he = False
            label = "CODEX"
        elif preview_mode == "markers":
            all_channels = list(channel_names or read_ome_channel_names(source))
            lookup = {name.casefold(): name for name in all_channels}
            missing = [name for name in (marker_channels or []) if name.casefold() not in lookup]
            if missing:
                raise PreviewActionError(
                    f"{source.name} does not contain requested channel(s): {', '.join(missing)}."
                )
            channels = [lookup[name.casefold()] for name in (marker_channels or [])]
            if not channels:
                raise PreviewActionError("No marker channel was requested.")
            requested_channels = channels
            require_he = False
            label = re.sub(r"[^A-Za-z0-9]+", "_", "_".join(channels)).strip("_") or "markers"
        else:
            raise PreviewActionError(f"Unsupported preview mode: {preview_mode}")
        destination = require_assistant_write_path(
            output_dir / f"{_preview_stem(source)}_{label}_thumbnail.png"
        )
        cached = _cached_result(
            source,
            destination,
            preview_mode=preview_mode,
            channels=channels,
        )
        if cached is not None:
            results.append(cached)
            continue

        try:
            load_options: dict[str, Any] = {
                "channels": requested_channels,
                "max_size": int(max_size),
                "require_he": require_he,
            }
            image, full_shape, pyramid_level = load_base_image(source, **load_options)
        except ValueError as error:
            raise PreviewActionError(str(error)) from error
        from PIL.PngImagePlugin import PngInfo

        source_signature = source.stat()
        png_info = PngInfo()
        png_info.add_text("spacemap_preview_schema", "2")
        png_info.add_text("preview_mode", preview_mode)
        png_info.add_text("channels_json", json.dumps(channels))
        png_info.add_text("source_size", str(source_signature.st_size))
        png_info.add_text("source_mtime_ns", str(source_signature.st_mtime_ns))
        png_info.add_text("pyramid_level", str(pyramid_level))
        png_info.add_text("full_height", str(full_shape[0]))
        png_info.add_text("full_width", str(full_shape[1]))
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            image.save(temporary, format="PNG", optimize=True, pnginfo=png_info)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        results.append(
            {
                "source_path": str(source),
                "preview_path": str(destination),
                "preview_mode": preview_mode,
                "channels": channels,
                "pyramid_level": int(pyramid_level),
                "full_height": int(full_shape[0]),
                "full_width": int(full_shape[1]),
                "preview_width": int(image.width),
                "preview_height": int(image.height),
                "cached": False,
            }
        )
    return results


def generate_he_previews(
    source_path: str | Path,
    *,
    max_size: int = 768,
) -> list[dict[str, Any]]:
    """Create one H&E RGB PNG per image using low-resolution pyramid planes."""
    return _generate_previews(source_path, preview_mode="he", max_size=max_size)


def generate_non_he_previews(
    source_path: str | Path,
    *,
    channel_names: list[str] | None = None,
    max_size: int = 768,
) -> list[dict[str, Any]]:
    """Create one representative CODEX composite PNG per image."""
    return _generate_previews(
        source_path,
        preview_mode="non_he",
        channel_names=channel_names,
        max_size=max_size,
    )


def generate_marker_previews(
    source_path: str | Path,
    channels: list[str],
    *,
    channel_names: list[str] | None = None,
    max_size: int = 768,
) -> list[dict[str, Any]]:
    """Create one thumbnail from explicitly requested marker channels."""
    return _generate_previews(
        source_path,
        preview_mode="markers",
        marker_channels=list(channels),
        channel_names=channel_names,
        max_size=max_size,
    )


def generate_scale_bar_previews(
    base_previews: list[dict[str, Any]],
    *,
    mpp: float | None = None,
    length_um: float | None = None,
    position: str = "bottom-left",
) -> list[dict[str, Any]]:
    """Add a white scale bar while preserving the exact prior visual state."""
    if not base_previews:
        raise PreviewActionError("A previous or newly generated image preview is required.")
    allowed_positions = {"top-left", "top-right", "bottom-left", "bottom-right", "auto"}
    if position not in allowed_positions:
        raise PreviewActionError(
            "Scale-bar position must be top-left, top-right, bottom-left, or bottom-right."
        )

    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo

    results: list[dict[str, Any]] = []
    for preview in base_previews:
        previous_scale = preview.get("scale_bar") or {}
        undecorated_path = str(previous_scale.get("base_preview") or preview["preview_path"])
        base_path = require_assistant_read_path(undecorated_path, must_exist=True)
        source_path = require_assistant_read_path(preview["source_path"], must_exist=True)
        resolved_mpp = float(mpp or 0)
        if resolved_mpp <= 0:
            from spacemap.background_subtraction_utils import get_image_mpp

            detected = get_image_mpp(str(source_path)).get("codex")
            resolved_mpp = float(detected or 0)
        if resolved_mpp <= 0:
            raise PreviewActionError(
                "I could not read the source pixel size. Tell me the micrometers per pixel (MPP)."
            )

        with Image.open(base_path) as opened:
            base = opened.convert("RGB")
        full_width = int(preview.get("full_width") or 0)
        if full_width <= 0 or base.width <= 0:
            raise PreviewActionError("The previous preview is missing its original image size.")
        # A candidate tile represents only its level-0 crop, not the entire
        # slide.  Using ``full_width`` for a tile made a 280 µm crop look like
        # an 8.5 mm overview and could draw a scientifically impossible bar.
        represented_width = float(full_width)
        crop_box = preview.get("crop_box")
        if isinstance(crop_box, (list, tuple)) and len(crop_box) >= 4:
            crop_width = float(crop_box[2]) - float(crop_box[0])
            if crop_width > 0:
                represented_width = crop_width
        preview_mpp = resolved_mpp * (represented_width / float(base.width))
        resolved_length = float(length_um or 0)
        if resolved_length <= 0:
            maximum = base.width * 0.22 * preview_mpp
            choices = [value for value in SCALE_BAR_UM_CHOICES if value <= maximum]
            resolved_length = float(choices[-1] if choices else SCALE_BAR_UM_CHOICES[0])
        bar_px = max(1, int(round(resolved_length / preview_mpp)))
        if bar_px > int(base.width * 0.8):
            raise PreviewActionError(
                f"A {resolved_length:g} µm scale bar is too wide for this thumbnail. "
                "Choose a shorter length."
            )

        rendered = base.convert("RGBA")
        draw = ImageDraw.Draw(rendered, "RGBA")
        font_size = max(11, int(round(base.width * 0.016)))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        label = f"{resolved_length:g} µm"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        padding = max(10, int(round(base.width * 0.012)))
        margin = padding
        bar_height = max(4, int(round(base.height * 0.006)))
        block_width = padding * 2 + max(bar_px, text_width)
        block_height = padding * 3 + bar_height + text_height
        if position == "top-left":
            x0, y0 = margin, margin
        elif position == "top-right":
            x0, y0 = max(base.width - block_width - margin, 0), margin
        elif position == "bottom-right":
            x0 = max(base.width - block_width - margin, 0)
            y0 = max(base.height - block_height - margin, 0)
        else:
            x0, y0 = margin, max(base.height - block_height - margin, 0)
        draw.rounded_rectangle(
            [x0, y0, x0 + block_width, y0 + block_height],
            radius=max(6, padding // 2),
            fill=(0, 0, 0, 145),
        )
        bar_left = x0 + padding
        bar_top = y0 + padding
        draw.rectangle(
            [bar_left, bar_top, bar_left + bar_px, bar_top + bar_height],
            fill=(255, 255, 255, 255),
        )
        draw.text(
            (x0 + padding, bar_top + bar_height + padding),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )
        rendered = rendered.convert("RGB")
        settings = {
            "base_preview": str(base_path),
            "source_mpp": resolved_mpp,
            "preview_mpp": preview_mpp,
            "represented_source_width_px": represented_width,
            "length_um": resolved_length,
            "position": position,
            "color": "#FFFFFF",
        }
        settings_id = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        destination = require_assistant_write_path(
            base_path.with_name(f"{base_path.stem}_scale_bar_{settings_id}.png")
        )
        png_info = PngInfo()
        png_info.add_text("spacemap_scale_bar_schema", "1")
        png_info.add_text("scale_bar_json", json.dumps(settings, sort_keys=True))
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            rendered.save(temporary, format="PNG", optimize=True, pnginfo=png_info)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        results.append(
            {
                **preview,
                "preview_path": str(destination),
                "preview_width": int(rendered.width),
                "preview_height": int(rendered.height),
                "scale_bar": settings,
                "cached": False,
            }
        )
    return results


def _polygon_area_px2(coordinates: list[Any]) -> float:
    """Return polygon area from GeoJSON rings without loading a geometry library."""
    def ring_area(ring: list[Any]) -> float:
        points = [point for point in ring if isinstance(point, (list, tuple)) and len(point) >= 2]
        if len(points) < 3:
            return 0.0
        return abs(
            sum(
                float(points[index][0]) * float(points[(index + 1) % len(points)][1])
                - float(points[(index + 1) % len(points)][0]) * float(points[index][1])
                for index in range(len(points))
            )
        ) / 2.0

    if not coordinates:
        return 0.0
    return max(0.0, ring_area(coordinates[0]) - sum(ring_area(ring) for ring in coordinates[1:]))


def _polygon_components(feature: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a Polygon or MultiPolygon feature into individually reviewable candidates."""
    geometry = feature.get("geometry") or {}
    geometry_type = str(geometry.get("type") or "")
    coordinates = geometry.get("coordinates") or []
    polygons = [coordinates] if geometry_type == "Polygon" else coordinates if geometry_type == "MultiPolygon" else []
    return [
        {
            "type": "Feature",
            "properties": dict(feature.get("properties") or {}),
            "geometry": {"type": "Polygon", "coordinates": polygon},
        }
        for polygon in polygons
        if polygon
    ]


def generate_annotation_qc_previews(
    source_path: str | Path,
    geojson_paths: list[str],
    *,
    preferred_mode: str = "automatic",
    overview_size: int = 1024,
    tile_size: int = 768,
    max_tiles: int = 3,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a local overview and representative polygon tiles for visual QC."""
    source_dir, images = _source_images(source_path)
    if len(images) != 1:
        raise PreviewActionError("Visual QC currently needs one TIFF image at a time.")
    source = images[0]
    if not 256 <= int(overview_size) <= 2048 or not 256 <= int(tile_size) <= 2048:
        raise PreviewActionError("QC preview sizes must be between 256 and 2,048 pixels.")
    max_tiles = min(6, max(1, int(max_tiles)))

    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo
    from spacemap.background_subtraction_utils import get_image_mpp
    from spacemap_gui.tabs.tab9_visualization.overlay_core import (
        crop_base_to_fullres_box,
        feature_color,
        feature_name,
        geometry_bounds,
        load_base_image,
        load_geojson,
        default_marker_selection,
        read_ome_channel_names,
        render_overlay,
    )

    features: list[dict[str, Any]] = []
    overlay_sources: list[dict[str, Any]] = []
    for raw_path in geojson_paths:
        path = require_assistant_read_path(raw_path, must_exist=True)
        if not path.is_file() or path.suffix.casefold() not in GEOJSON_SUFFIXES:
            raise PreviewActionError(f"QC annotation must be a GeoJSON file: {path}")
        data = load_geojson(path)
        current = [item for item in (data.get("features") or []) if isinstance(item, dict)]
        features.extend(current)
        stat = path.stat()
        overlay_sources.append(
            {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        )
    if not features:
        raise PreviewActionError("The requested GeoJSON has no polygons to review.")

    components = [component for feature in features for component in _polygon_components(feature)]
    ranked = sorted(
        components,
        key=lambda item: _polygon_area_px2(item["geometry"]["coordinates"]),
        reverse=True,
    )
    if not ranked:
        raise PreviewActionError("The requested GeoJSON has no Polygon or MultiPolygon geometry.")

    source_stat = source.stat()
    signature = {
        "source": str(source),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "overlays": overlay_sources,
        "preferred_mode": preferred_mode,
        "overview_size": int(overview_size),
        "tile_size": int(tile_size),
        "max_tiles": int(max_tiles),
        "output_dir": str(output_dir or ""),
    }
    signature_id = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    preview_dir = require_assistant_write_path(
        Path(output_dir) if output_dir else source_dir / "_spacemap_ai_previews"
    )
    preview_dir.mkdir(parents=True, exist_ok=True)

    require_he = preferred_mode == "he"
    selected_channels: list[str] | None = None
    if preferred_mode == "codex":
        available = list(read_ome_channel_names(source))
        evidence_names: list[str] = []
        for feature in features:
            block = (feature.get("properties") or {}).get("spacemap") or {}
            for evidence in block.get("candidate_markers") or []:
                marker = str((evidence or {}).get("marker") or "") if isinstance(evidence, dict) else ""
                if marker in available and marker not in evidence_names:
                    evidence_names.append(marker)
        dapi = next((name for name in available if "dapi" in name.casefold()), None)
        selected_channels = ([dapi] if dapi else []) + [
            name for name in evidence_names if name != dapi
        ][:3]
        if not selected_channels:
            selected_channels = default_marker_selection(available)
        if not selected_channels:
            selected_channels = [
                name for name in available if name not in {*HE_CHANNELS, "R", "G", "B"}
            ][:3]
        if not selected_channels:
            raise PreviewActionError("No CODEX channels were found for the QC view.")
    try:
        high_base, full_shape, pyramid_level = load_base_image(
            source,
            channels=selected_channels,
            max_size=2048,
            require_he=require_he,
        )
    except ValueError as error:
        raise PreviewActionError(str(error)) from error
    channels = (
        list(HE_CHANNELS)
        if require_he
        else list(selected_channels or ["automatic image view"])
    )
    color_lookup = {
        feature_name(feature.get("properties") or {}): feature_color(
            feature.get("properties") or {}, "#E6194B"
        )
        for feature in features
    }

    try:
        detected_mpp = float(get_image_mpp(str(source)).get("codex") or 0)
    except (OSError, TypeError, ValueError):
        detected_mpp = 0.0
    total_area_px2 = sum(
        _polygon_area_px2(component["geometry"]["coordinates"])
        for component in components
    )
    summary = {
        "feature_count": len(features),
        "polygon_count": len(components),
        "total_area_px2": round(float(total_area_px2), 2),
        "mpp": detected_mpp or None,
        "total_area_mm2": (
            round(total_area_px2 * detected_mpp * detected_mpp / 1_000_000.0, 4)
            if detected_mpp > 0
            else None
        ),
        "reviewed_tile_count": min(max_tiles, len(ranked)),
    }

    def save_qc_image(image: Image.Image, destination: Path, extra: dict[str, Any]) -> None:
        info = PngInfo()
        info.add_text("spacemap_qc_preview_schema", "1")
        info.add_text("qc_signature", json.dumps(signature, sort_keys=True))
        info.add_text("qc_summary", json.dumps(summary, sort_keys=True))
        info.add_text("qc_details", json.dumps(extra, sort_keys=True))
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            image.convert("RGB").save(temporary, format="PNG", optimize=True, pnginfo=info)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    overview = high_base.copy()
    overview.thumbnail((int(overview_size), int(overview_size)), Image.Resampling.LANCZOS)
    overview, overview_stats = render_overlay(
        base=overview,
        full_shape=full_shape,
        tissue_features=features,
        tissue_color_lookup=color_lookup,
        tissue_alpha=40,
        tissue_outline_alpha=255,
        tissue_outline_width=3,
    )
    overview_path = require_assistant_write_path(
        preview_dir / f"{_preview_stem(source)}_QC_overview_{signature_id}.png"
    )
    save_qc_image(overview, overview_path, {"kind": "overview", "stats": overview_stats})
    previews: list[dict[str, Any]] = [
        {
            "source_path": str(source),
            "preview_path": str(overview_path),
            "preview_mode": "qc_he" if require_he else "qc_image",
            "qc_kind": "overview",
            "channels": channels,
            "pyramid_level": int(pyramid_level),
            "full_height": int(full_shape[0]),
            "full_width": int(full_shape[1]),
            "preview_width": int(overview.width),
            "preview_height": int(overview.height),
            "overlay_sources": overlay_sources,
            "overlay_stats": overview_stats,
            "qc_summary": summary,
            "cached": False,
        }
    ]

    try:
        label_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        label_font = ImageFont.load_default()
    full_h, full_w = int(full_shape[0]), int(full_shape[1])
    for index, component in enumerate(ranked[:max_tiles], start=1):
        bounds = geometry_bounds(component.get("geometry"))
        if bounds is None:
            continue
        min_x, min_y, max_x, max_y = bounds
        span = max(max_x - min_x, max_y - min_y, 400.0) * 1.35
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        crop_box = (
            max(0.0, center_x - span / 2.0),
            max(0.0, center_y - span / 2.0),
            min(float(full_w), center_x + span / 2.0),
            min(float(full_h), center_y + span / 2.0),
        )
        cropped, effective_box = crop_base_to_fullres_box(high_base, full_shape, crop_box)
        rendered, tile_stats = render_overlay(
            base=cropped,
            full_shape=full_shape,
            crop_box=effective_box,
            tissue_features=[component],
            tissue_color_lookup=color_lookup,
            tissue_alpha=28,
            tissue_outline_alpha=255,
            tissue_outline_width=4,
        )
        ratio = int(tile_size) / max(rendered.size)
        if ratio != 1.0:
            rendered = rendered.resize(
                (max(1, round(rendered.width * ratio)), max(1, round(rendered.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        draw = ImageDraw.Draw(rendered, "RGBA")
        label = f"Candidate {index} · red outline = proposed annotation"
        text_box = draw.textbbox((0, 0), label, font=label_font)
        draw.rectangle(
            [8, 8, text_box[2] - text_box[0] + 22, text_box[3] - text_box[1] + 22],
            fill=(0, 0, 0, 175),
        )
        draw.text((15, 13), label, fill=(255, 255, 255, 255), font=label_font)
        tile_path = require_assistant_write_path(
            preview_dir / f"{_preview_stem(source)}_QC_tile_{index:02d}_{signature_id}.png"
        )
        component_area = _polygon_area_px2(component["geometry"]["coordinates"])
        save_qc_image(
            rendered,
            tile_path,
            {
                "kind": "candidate_tile",
                "candidate": index,
                "crop_box": effective_box,
                "area_px2": component_area,
                "stats": tile_stats,
            },
        )
        previews.append(
            {
                **previews[0],
                "preview_path": str(tile_path),
                "qc_kind": "candidate_tile",
                "candidate": index,
                "crop_box": list(effective_box),
                "candidate_area_px2": round(float(component_area), 2),
                "preview_width": int(rendered.width),
                "preview_height": int(rendered.height),
                "overlay_stats": tile_stats,
            }
        )
    return {"previews": previews, "summary": summary}


def _geojson_kind(path: Path, features: list[dict[str, Any]], prompt: str) -> str:
    """Infer whether a GeoJSON contains cell polygons or tissue annotations."""
    name = path.name.casefold()
    if any(term in name for term in ("cell", "segment")):
        return "cells"
    if any(term in name for term in ("annotation", "tissue", "roi", "bound")):
        return "tissue"
    lowered = str(prompt).casefold()
    if len(features) > 500 or any(term in lowered for term in ("segmentation", "cell boundaries")):
        return "cells"
    sample = features[:20]
    if any(
        isinstance((feature.get("properties") or {}).get("classification"), dict)
        for feature in sample
    ):
        return "tissue"
    return "cells" if len(features) > 100 else "tissue"


def generate_overlay_previews(
    base_previews: list[dict[str, Any]],
    geojson_paths: list[str],
    *,
    prompt: str = "",
) -> list[dict[str, Any]]:
    """Draw SPACEMAP segmentation/annotation GeoJSONs on generated base previews."""
    if not base_previews:
        raise PreviewActionError("A TIFF base image is required for this overlay preview.")
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from spacemap_gui.tabs.tab9_visualization.overlay_core import (
        feature_color,
        feature_name,
        load_geojson,
        render_overlay,
    )

    tissue_features: list[dict[str, Any]] = []
    cell_features: list[dict[str, Any]] = []
    source_signatures: list[dict[str, Any]] = []
    for raw_path in geojson_paths:
        path = require_assistant_read_path(raw_path, must_exist=True)
        if not path.is_file() or not path.name.lower().endswith(GEOJSON_SUFFIXES):
            raise PreviewActionError(f"Overlay input must be a GeoJSON file: {path}")
        data = load_geojson(path)
        features = list(data.get("features") or [])
        kind = _geojson_kind(path, features, prompt)
        if kind == "cells":
            cell_features.extend(features)
        else:
            tissue_features.extend(features)
        signature = path.stat()
        source_signatures.append(
            {
                "path": str(path),
                "kind": kind,
                "size": int(signature.st_size),
                "mtime_ns": int(signature.st_mtime_ns),
                "features": len(features),
            }
        )
    if not source_signatures:
        return base_previews

    signature_json = json.dumps(source_signatures, sort_keys=True)
    signature_id = hashlib.sha256(signature_json.encode("utf-8")).hexdigest()[:10]
    tissue_colors = {
        feature_name(feature.get("properties", {}) or {}): feature_color(
            feature.get("properties", {}) or {}, "#00D9FF"
        )
        for feature in tissue_features
    }
    results: list[dict[str, Any]] = []
    for preview in base_previews:
        base_path = require_assistant_read_path(preview["preview_path"], must_exist=True)
        destination = require_assistant_write_path(
            base_path.with_name(f"{base_path.stem}_overlay_{signature_id}.png")
        )
        with Image.open(base_path) as opened:
            base = opened.convert("RGB")
        rendered, stats = render_overlay(
            base=base,
            full_shape=(int(preview["full_height"]), int(preview["full_width"])),
            tissue_features=tissue_features,
            tissue_color_lookup=tissue_colors or None,
            tissue_alpha=45,
            tissue_outline_alpha=230,
            tissue_outline_width=2,
            cell_features=cell_features,
            cell_alpha=70,
            cell_outline_alpha=235,
            cell_color_mode="fixed",
            cell_fixed_color="#FF5B40",
        )
        png_info = PngInfo()
        png_info.add_text("spacemap_overlay_schema", "1")
        png_info.add_text("overlay_signature", signature_json)
        png_info.add_text("base_preview", str(base_path))
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            rendered.save(temporary, format="PNG", optimize=True, pnginfo=png_info)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        results.append(
            {
                **preview,
                "preview_path": str(destination),
                "preview_width": int(rendered.width),
                "preview_height": int(rendered.height),
                "overlay_sources": source_signatures,
                "overlay_stats": stats,
                "cached": False,
            }
        )
    return results
