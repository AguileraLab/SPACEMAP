"""Per-marker background detection from the images themselves.

Tab 2 offers two ways to remove background. When the panel includes blank
cycles, subtracting them is the most direct route. When it does not, the
background still has to come from somewhere, and this module estimates it from
each marker's own pixel histogram.

Working on pixels rather than per-cell means matters: a cell mean has already
averaged noise over a few hundred pixels, which narrows and shifts the
background mode, so a threshold derived from cell means does not mean the same
thing. The detected values are written in the CSV layout the RESOLVE
phenotyping step already reads, so the two steps stay connected.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from .thresholding_engine import THRESHOLD_METHODS

DEFAULT_MIN_COMPONENT_AREA_PX = 30_000.0
THRESHOLD_SCHEMA_VERSION = "spacemap-threshold-v2"


def _polygon_components(geometry) -> Iterator:
    """Yield disconnected polygon components without losing feature semantics.

    ``masks_from_geojson`` remains feature-based by default.  This helper is
    used only when the caller explicitly asks to treat disconnected components
    as independent background regions.
    """
    geometry_type = getattr(geometry, "geom_type", "")
    if geometry_type == "Polygon":
        yield geometry
    elif geometry_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from _polygon_components(part)


# Backwards-compatible public entry point, with one shared mathematical engine.
from .thresholding_engine import estimate_background as find_background


def _kept_polygon_components(geometry, minimum_area: float) -> tuple[list, int]:
    components = list(_polygon_components(geometry))
    kept = [component for component in components if float(component.area) >= minimum_area]
    return kept, len(components) - len(kept)


def geojson_component_summary(
    path: str | os.PathLike,
    classes: Sequence[str] | None = None,
    min_component_area_px: float = 0.0,
) -> dict[str, float | int]:
    """Count selected full-resolution polygon pieces before rasterization."""
    from shapely.geometry import shape as to_shape

    with open(path) as handle:
        collection = json.load(handle)
    wanted = {str(c).strip().lower() for c in (classes or []) if str(c).strip()}
    minimum = max(float(min_component_area_px or 0.0), 0.0)
    total = kept = ignored = features = 0
    total_area = kept_area = 0.0
    for feature in collection.get("features", []):
        classification = (feature.get("properties") or {}).get("classification") or {}
        name = str(classification.get("name", "")).strip().lower()
        if wanted and name not in wanted:
            continue
        try:
            geometry = to_shape(feature["geometry"])
        except Exception:
            continue
        if geometry.is_empty:
            continue
        components = list(_polygon_components(geometry))
        if not components:
            continue
        features += 1
        total += len(components)
        for component in components:
            area = float(component.area)
            total_area += area
            if area >= minimum:
                kept += 1
                kept_area += area
            else:
                ignored += 1
    return {
        "features": features,
        "components": total,
        "kept_components": kept,
        "ignored_components": ignored,
        "total_area_px": total_area,
        "kept_area_px": kept_area,
        "min_component_area_px": minimum,
    }


def mask_from_geojson(
    path: str | os.PathLike,
    shape: tuple[int, int],
    downsample: float,
    classes: Sequence[str] | None = None,
    min_component_area_px: float = 0.0,
) -> np.ndarray:
    """Rasterize tissue polygons onto the downsampled grid.

    Restricting the histogram to real tissue keeps empty slide out of the noise
    mode, which otherwise drags the estimate down.
    """
    from rasterio.features import rasterize
    from shapely.affinity import scale as scale_geom
    from shapely.geometry import shape as to_shape

    with open(path) as handle:
        collection = json.load(handle)
    wanted = {str(c).strip().lower() for c in (classes or []) if str(c).strip()}
    geometries = []
    for feature in collection.get("features", []):
        classification = (feature.get("properties") or {}).get("classification") or {}
        name = str(classification.get("name", "")).strip().lower()
        if wanted and name not in wanted:
            continue
        try:
            geometry = to_shape(feature["geometry"])
        except Exception:
            continue
        if geometry.is_empty:
            continue
        kept, _ = _kept_polygon_components(
            geometry, max(float(min_component_area_px or 0.0), 0.0)
        )
        factor = 1.0 / float(downsample or 1.0)
        geometries.extend(
            scale_geom(component, xfact=factor, yfact=factor, origin=(0, 0))
            for component in kept
        )
    if not geometries:
        return np.zeros(shape, dtype=bool)
    return rasterize(((g, 1) for g in geometries), out_shape=shape, dtype="uint8").astype(bool)


def otsu_tissue_mask(image: np.ndarray) -> np.ndarray:
    """Fallback mask when no annotation GeoJSON is available."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk, remove_small_holes

    data = np.asarray(image, dtype=np.float32)
    if not np.isfinite(data).any() or data.max() <= data.min():
        return np.ones(data.shape, dtype=bool)
    mask = data > threshold_otsu(data)
    mask = binary_closing(mask, disk(5))
    return remove_small_holes(mask, area_threshold=3000)


def image_levels(image_path: str | os.PathLike) -> pd.DataFrame:
    """Pyramid levels available in the file, with the downsample each implies."""
    from spacemap.tissue_annotation_utils import PyramidImageReader
    import tifffile

    reader = PyramidImageReader(image_path)
    full_height, full_width = reader.level_shapes[0]
    with tifffile.TiffFile(str(image_path)) as tif:
        dtype = str(np.dtype(tif.pages[0].dtype))
    rows = []
    for index, (height, width) in enumerate(reader.level_shapes):
        factor = full_height / height if height else 1.0
        rows.append({
            "level": index,
            "height": int(height),
            "width": int(width),
            "downsample": round(float(factor), 2),
            "megapixels": round(height * width / 1e6, 2),
            "dtype": dtype,
        })
    return pd.DataFrame(rows)


def masks_from_geojson(
    path: str | os.PathLike,
    shape: tuple[int, int],
    downsample: float,
    classes: Sequence[str] | None = None,
    region_unit: str = "feature",
    min_component_area_px: float = 0.0,
) -> dict[str, np.ndarray]:
    """One mask per annotation feature or disconnected polygon component.

    Kept separate from the pooled mask because the two answer different
    questions: pooling gives the most stable single estimate, while per-polygon
    values expose whether background actually varies across the slide.

    ``region_unit='feature'`` is the backward-compatible default and preserves
    the annotator's decision to store a MultiPolygon as one sampling region.
    ``region_unit='component'`` splits MultiPolygons (and polygonal geometry
    collections) into disconnected components. Components that cover no pixel
    centre on the working pyramid level are discarded.
    """
    from rasterio.features import rasterize
    from shapely.affinity import scale as scale_geom
    from shapely.geometry import shape as to_shape

    with open(path) as handle:
        collection = json.load(handle)
    wanted = {str(c).strip().lower() for c in (classes or []) if str(c).strip()}
    unit = str(region_unit or "feature").strip().lower()
    if unit not in {"feature", "component"}:
        raise ValueError("region_unit must be 'feature' or 'component'.")

    factor = 1.0 / float(downsample or 1.0)
    masks: dict[str, np.ndarray] = {}
    feature_counter = 0
    region_counter = 0
    for feature in collection.get("features", []):
        classification = (feature.get("properties") or {}).get("classification") or {}
        name = str(classification.get("name", "")).strip()
        if wanted and name.lower() not in wanted:
            continue
        try:
            geometry = to_shape(feature["geometry"])
        except Exception:
            continue
        if geometry.is_empty:
            continue
        feature_counter += 1
        components, _ = _kept_polygon_components(
            geometry, max(float(min_component_area_px or 0.0), 0.0)
        )
        if not components:
            continue
        regions = [components] if unit == "feature" else [[component] for component in components]
        for component_position, region_parts in enumerate(regions, start=1):
            try:
                scaled = [
                    scale_geom(region, xfact=factor, yfact=factor, origin=(0, 0))
                    for region in region_parts
                ]
                raster = rasterize(
                    ((region, 1) for region in scaled), out_shape=shape, dtype="uint8"
                ).astype(bool)
            except Exception:
                continue
            # Sub-pixel/off-frame components should not masquerade as valid
            # background samples.
            if not raster.any():
                continue
            region_counter += 1
            if unit == "component":
                key = f"{name or 'region'}_{feature_counter}_component_{component_position}"
            else:
                key = f"{name or 'region'}_{region_counter}"
            masks[key] = raster
    return masks


_REGION_AGGREGATION_ALIASES = {
    "individual": "individual",
    "preserve": "individual",
    "median": "median",
    "collapse_median": "median",
    "mean": "mean",
    "collapse_mean": "mean",
    "p05": "p05",
    "q05": "p05",
    "collapse_p05": "p05",
    "p25": "p25",
    "q25": "p25",
    "collapse_p25": "p25",
    "p75": "p75",
    "q75": "p75",
    "collapse_p75": "p75",
}
_REGION_QUANTILES = {"p05": 0.05, "p25": 0.25, "p75": 0.75}
_REGION_AGGREGATION_PROVENANCE = {
    "median": "median_across_region_thresholds",
    "mean": "mean_across_region_thresholds",
    "p05": "5th_percentile_across_region_thresholds",
    "p25": "25th_percentile_across_region_thresholds",
    "p75": "75th_percentile_across_region_thresholds",
}


def _normalize_region_aggregation(value: str | None) -> str:
    mode = _REGION_AGGREGATION_ALIASES.get(str(value or "individual").strip().lower())
    if mode is None:
        raise ValueError(
            "export_mode must be 'individual', 'median', 'mean', 'p05', 'p25', or 'p75'."
        )
    return mode


def _aggregate_region_frame(values: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "mean":
        return values.mean(axis=1, skipna=True)
    if mode == "median":
        return values.median(axis=1, skipna=True)
    return values.quantile(_REGION_QUANTILES[mode], axis=1, interpolation="linear")


def _aggregate_region_array(values: np.ndarray, mode: str) -> float:
    if mode == "mean":
        return float(np.mean(values))
    if mode == "median":
        return float(np.median(values))
    return float(np.quantile(values, _REGION_QUANTILES[mode], method="linear"))


def _roi_request_value_range(
    plane: np.ndarray, raster: np.ndarray | None
) -> tuple[float, float]:
    """Return the min/max in the rectangular QuPath annotation request.

    ImageJ derives its 8-bit conversion range from this rectangle and only then
    applies the irregular annotation mask to the histogram.
    """
    if raster is None or raster.shape != plane.shape or not raster.any():
        values = plane
    else:
        rows = np.flatnonzero(raster.any(axis=1))
        columns = np.flatnonzero(raster.any(axis=0))
        values = plane[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 0.0
    return float(finite.min()), float(finite.max())


def detect_backgrounds(
    image_path: str | os.PathLike,
    markers: Sequence[str] | None = None,
    method: str = "Triangle",
    downsample: int = 4,
    mask: np.ndarray | None = None,
    region_masks: dict[str, np.ndarray] | None = None,
    aggregate: str = "mean",
    progress=None,
    params=None,
) -> pd.DataFrame:
    """Background level for every requested marker in one image.

    With ``region_masks`` a threshold is measured inside each region separately
    and combined with ``aggregate``; the per-region values are returned too.
    Mean is the default because the Groovy reference averages the thresholds
    from its QuPath annotation objects.
    """
    from spacemap.tissue_annotation_utils import PyramidImageReader
    import tifffile

    reader = PyramidImageReader(image_path)
    names = list(reader.channel_names)
    targets = [n for n in (markers or names) if n in names]
    selected_aggregate = _normalize_region_aggregation(aggregate)
    if selected_aggregate == "individual":
        selected_aggregate = "mean"
    rows = []
    with tifffile.TiffFile(str(image_path)) as tif:
        source_dtype = np.dtype(tif.pages[0].dtype)
        for position, name in enumerate(targets):
            if progress is not None:
                progress(position + 1, len(targets), name)
            plane = reader.read_channel(tif, names.index(name), int(downsample))
            if source_dtype == np.dtype(np.uint16):
                # QuPath materializes the arbitrary-downsample request as an
                # unsigned-short ImagePlus before ImageJ thresholds it.
                plane = np.rint(plane)
            record = {"Marker": name, "method": method, "downsample": int(downsample)}

            if region_masks:
                per_region = {}
                region_fallbacks = {}
                region_errors = {}
                for region, raster in region_masks.items():
                    if raster.shape != plane.shape or not raster.any():
                        continue
                    result = find_background(
                        plane[raster], method, params=params,
                        value_range=_roi_request_value_range(plane, raster),
                        imagej_dtype=source_dtype,
                    )
                    per_region[region] = result["threshold"]
                    if result["fallback_status"] != "none":
                        region_fallbacks[region] = result["fallback_status"]
                        region_errors[region] = result["threshold_error"]
                if per_region:
                    values = np.array(list(per_region.values()), dtype=float)
                    record["threshold"] = _aggregate_region_array(values, selected_aggregate)
                    record["n_regions"] = len(per_region)
                    record["region_spread"] = float(values.max() / max(values.min(), 1e-9))
                    record["n_pixels"] = int(sum(int(m.sum()) for m in region_masks.values()))
                    record["peak"] = np.nan
                    record["detected_aggregation"] = _REGION_AGGREGATION_PROVENANCE[
                        selected_aggregate
                    ]
                    record["threshold_fallback_status"] = (
                        "none" if not region_fallbacks else "; ".join(
                            f"{region}={status}" for region, status in region_fallbacks.items()
                        )
                    )
                    record["threshold_error"] = "; ".join(
                        f"{region}: {error}" for region, error in region_errors.items()
                    )
                    record.update({f"region::{k}": v for k, v in per_region.items()})
                    rows.append(record)
                    continue

            selected = plane[mask] if mask is not None and mask.shape == plane.shape else plane
            result = find_background(
                selected, method, params=params,
                value_range=_roi_request_value_range(plane, mask),
                imagej_dtype=source_dtype,
            )
            record.update({
                "threshold": result["threshold"],
                "peak": result["peak"],
                "n_pixels": result["n_pixels"],
                "n_regions": 1,
                "region_spread": 1.0,
                "detected_aggregation": "threshold_from_pooled_pixels",
                "threshold_fallback_status": result["fallback_status"],
                "threshold_error": result["threshold_error"],
            })
            rows.append(record)
    return pd.DataFrame(rows)


def align_region_aggregate(table: pd.DataFrame, export_mode: str = "individual") -> pd.DataFrame:
    """Synchronize the displayed scalar with the selected region aggregation.

    Streamlit reruns when a user changes the export selector, but detection does
    not need to be repeated because the independent region thresholds are
    already present. Individual export preserves the aggregate selected when
    detection ran; a combined export uses its selected statistic. Rows without
    region measurements (pooled/Otsu fallbacks in a mixed batch) are unchanged.
    """
    requested_mode = _normalize_region_aggregation(export_mode)
    frame = table.copy()
    if requested_mode == "individual":
        return frame
    selected_mode = requested_mode
    region_columns = [
        column for column in frame.columns if str(column).startswith("region::")
    ]
    if not region_columns:
        return frame
    numeric_regions = frame[region_columns].apply(pd.to_numeric, errors="coerce")
    selected = _aggregate_region_frame(numeric_regions, selected_mode)
    aggregation = _REGION_AGGREGATION_PROVENANCE[selected_mode]
    has_region_measurement = selected.notna()
    frame.loc[has_region_measurement, "threshold"] = selected.loc[has_region_measurement]
    frame.loc[has_region_measurement, "detected_aggregation"] = aggregation
    return frame


def prepare_threshold_export(table: pd.DataFrame, export_mode: str = "individual", *, precision: int = 2) -> pd.DataFrame:
    """Prepare a RESOLVE-compatible threshold table with explicit provenance.

    Parameters
    ----------
    table:
        Detection output. Per-region measurements are identified by
        ``region::`` columns.
    export_mode:
        ``individual`` preserves one ``Polygon_*`` value per region.
        ``median``, ``mean``, ``p05``, ``p25``, and ``p75`` combine the
        independently detected polygon thresholds into one ``Polygon_1``.
        A table without region columns naturally produces one ``Polygon_1``.

    Manual overrides and diagnostics remain explicit in this prepared in-memory
    frame. ``write_threshold_csv`` then emits the compact legacy Marker +
    Polygon layout and applies any manual override to its saved polygon values.
    """
    if not isinstance(precision, int) or not 0 <= precision <= 15:
        raise ValueError("Choose threshold export precision from 0 to 15 decimals.")
    frame = table.copy()
    if "threshold" not in frame.columns:
        raise ValueError("The threshold table needs a 'threshold' column.")

    requested_mode = _normalize_region_aggregation(export_mode)

    # Concatenated batch results can carry all other images' region columns as
    # all-NaN placeholders. Only export regions that exist in this table.
    region_columns = [
        c for c in frame.columns
        if str(c).startswith("region::") and pd.to_numeric(frame[c], errors="coerce").notna().any()
    ]
    numeric_regions = (
        frame[region_columns].apply(pd.to_numeric, errors="coerce")
        if region_columns else pd.DataFrame(index=frame.index)
    )
    detected = pd.to_numeric(frame["threshold"], errors="coerce")
    frame["threshold_detected"] = detected
    if "manual_override" in frame.columns:
        manual = pd.to_numeric(frame["manual_override"], errors="coerce")
    else:
        manual = pd.Series(np.nan, index=frame.index, dtype=float)
    frame["manual_override"] = manual
    if "threshold_fallback_status" not in frame.columns:
        frame["threshold_fallback_status"] = "none"
    threshold_fallback = frame["threshold_fallback_status"].fillna("none").astype(str)
    unresolved_fallback = threshold_fallback.ne("none") & manual.isna()
    if unresolved_fallback.any():
        markers = frame.loc[unresolved_fallback, "Marker"].astype(str).tolist()
        raise ValueError(
            "Threshold detection used the 150 fallback for "
            + ", ".join(markers[:8])
            + (" ..." if len(markers) > 8 else "")
            + ". Enter a manual override for each affected marker before export."
        )

    polygon_columns: list[str]
    if region_columns and requested_mode == "individual":
        for position, column in enumerate(region_columns, start=1):
            frame[f"Polygon_{position}"] = numeric_regions[column].round(precision)
        polygon_columns = [f"Polygon_{i}" for i in range(1, len(region_columns) + 1)]
        selected = detected
        effective_mode = "individual"
        aggregation = "individual_region_thresholds"
    elif region_columns:
        selected = _aggregate_region_frame(numeric_regions, requested_mode)
        aggregation = _REGION_AGGREGATION_PROVENANCE[requested_mode]
        effective_mode = f"collapse_{requested_mode}"
        polygon_columns = ["Polygon_1"]
    else:
        selected = detected
        effective_mode = "pooled"
        aggregation = "threshold_from_pooled_pixels"
        polygon_columns = ["Polygon_1"]

    frame["threshold_final"] = manual.combine_first(selected)
    if effective_mode.startswith("collapse_"):
        # Keep the selected region statistic visible even when a manual override
        # exists. ``threshold_final`` records the override independently.
        frame["Polygon_1"] = selected.round(precision)
    elif effective_mode == "pooled":
        # Preserve historical pooled-mode behaviour for Polygon-only readers.
        frame["Polygon_1"] = frame["threshold_final"].round(precision)
    # Version 2 makes the scalar column authoritative for readers that prefer a
    # named threshold over Polygon_* values. The untouched detector result is
    # retained separately for auditability.
    frame["threshold"] = frame["threshold_final"]

    if "detected_aggregation" not in frame.columns:
        frame["detected_aggregation"] = (
            "median_across_region_thresholds" if region_columns
            else "threshold_from_pooled_pixels"
        )
    if "source_mode" not in frame.columns:
        frame["source_mode"] = "geojson_region_thresholds" if region_columns else "pooled_pixels"
    if "region_unit" not in frame.columns:
        frame["region_unit"] = "feature" if region_columns else "pooled"
    if "fallback_status" not in frame.columns:
        frame["fallback_status"] = "none"
    frame["schema_version"] = THRESHOLD_SCHEMA_VERSION
    frame["export_mode"] = effective_mode
    frame["aggregation"] = aggregation
    frame.attrs["polygon_columns"] = polygon_columns
    frame.attrs["region_columns"] = region_columns
    return frame


def write_threshold_csv(
    table: pd.DataFrame,
    output_path: str | os.PathLike,
    image_stem: str | None = None,
    export_mode: str = "individual",
    *, precision: int = 2,
) -> Path:
    """Write the compact legacy layout RESOLVE reads.

    The file intentionally contains only ``Marker`` and contiguous
    ``Polygon_1..Polygon_N`` columns. Diagnostic/provenance fields remain in
    the in-memory review table instead of making the user-facing CSV harder to
    inspect. A manual override replaces every saved polygon value for that
    marker, so it remains authoritative under any downstream aggregation.
    """
    del image_stem  # Kept for API compatibility with the original writer.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = prepare_threshold_export(table, export_mode=export_mode, precision=precision)
    polygon_columns = frame.attrs["polygon_columns"]
    output = frame[["Marker"] + polygon_columns].copy()
    manual = pd.to_numeric(frame.get("manual_override"), errors="coerce")
    overridden = manual.notna()
    if overridden.any():
        for column in polygon_columns:
            output.loc[overridden, column] = manual.loc[overridden].round(precision)
    output.to_csv(output_path, index=False, float_format=f"%.{precision}f")
    return output_path


def threshold_csv_name(image_stem: str) -> str:
    """Filename the RESOLVE loader searches for first."""
    return f"{image_stem}.ome.tiff_-_Merged_Image.csv"


def subtract_background_image(
    image_path: str | os.PathLike,
    thresholds: dict[str, float],
    output_path: str | os.PathLike,
    clip_negative: bool = True,
    progress=None,
) -> Path:
    """Write a tiled, pyramidal OME-TIFF with each channel's background removed.

    The layout intentionally matches the ROI writer: one full-resolution IFD
    per channel, three 2x SubIFDs, 256x256 tiles, and DEFLATE compression. OME
    channel names, display colors, fluorophore metadata, and physical pixel
    size are reconstructed from the source image.
    """
    import cv2
    import tifffile
    from spacemap.background_subtraction_utils import get_channel_info
    from spacemap.tissue_annotation_utils import PyramidImageReader
    from spacemap.utils import create_ome_metadata

    reader = PyramidImageReader(image_path)
    source_names = list(reader.channel_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the same normalized metadata reader as Tab 3's ROI writer. Fall back
    # to the pyramid reader's names only when the source metadata is incomplete.
    metadata_names, channel_colors, channel_metadata, mpp = get_channel_info(
        str(image_path)
    )
    names = (
        list(metadata_names)
        if len(metadata_names) == len(source_names)
        else source_names
    )
    if len(channel_colors) != len(names):
        channel_colors = [(255, 255, 255)] * len(names)
    if len(channel_metadata) != len(names):
        channel_metadata = [{} for _ in names]

    # read_channel returns float32 for arithmetic, so obtain the storage type
    # from the source header and cast each corrected plane back before writing.
    with tifffile.TiffFile(str(image_path)) as probe:
        page = probe.pages[0]
        dtype = np.dtype(page.dtype)
        height, width = reader.read_channel(probe, 0, 1).shape

    mpp_value = mpp.get("codex") if isinstance(mpp, dict) else None
    try:
        mpp_value = float(mpp_value)
        if not np.isfinite(mpp_value) or mpp_value <= 0:
            raise ValueError
    except (TypeError, ValueError):
        # Tab 3 requires an MPP value. For generic TIFFs without one, retain a
        # valid OME physical scale rather than emitting malformed XML.
        mpp_value = 1.0

    ome_pixel_types = {
        "bool": "bit",
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "double-complex",
    }
    pixel_type = ome_pixel_types.get(dtype.name, dtype.name)
    ome_xml = create_ome_metadata(
        channel_names=names,
        channel_colors=channel_colors,
        SizeX=width,
        SizeY=height,
        pixel_type=pixel_type,
        n_channels=len(names),
        physical_size_x=mpp_value,
        physical_size_y=mpp_value,
        channel_metadata=channel_metadata,
    )

    tile_size = (256, 256)
    n_pyramid_levels = 3
    with tifffile.TiffFile(str(image_path)) as tif:
        with tifffile.TiffWriter(str(output_path), bigtiff=True) as writer:
            # Stream one channel and its pyramid at a time. A full multi-channel
            # stack can be tens of gigabytes and must never be materialized.
            for index, name in enumerate(names):
                if progress is not None:
                    progress(index + 1, len(names), name)
                plane = reader.read_channel(tif, index, 1)
                source_name = source_names[index]
                offset = float(
                    thresholds.get(source_name, thresholds.get(name, 0.0))
                )
                corrected = plane.astype(np.float32, copy=False)
                if offset:
                    corrected = corrected - offset
                if clip_negative:
                    corrected = np.clip(corrected, 0, None)
                if np.issubdtype(dtype, np.integer):
                    limits = np.iinfo(dtype)
                    corrected = np.clip(corrected, limits.min, limits.max)
                corrected = np.ascontiguousarray(corrected.astype(dtype, copy=False))

                writer.write(
                    corrected,
                    subifds=n_pyramid_levels,
                    photometric="minisblack",
                    compression="deflate",
                    tile=tile_size,
                    description=ome_xml if index == 0 else None,
                    planarconfig="CONTIG",
                    metadata=None,  # the OME description alone: no second, tifffile 'shaped' one
                )

                previous = corrected
                for level in range(n_pyramid_levels):
                    scale = 2 ** (level + 1)
                    level_width = max(1, width // scale)
                    level_height = max(1, height // scale)
                    downsampled = cv2.resize(
                        previous,
                        (level_width, level_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    downsampled = np.ascontiguousarray(
                        downsampled.astype(dtype, copy=False)
                    )
                    writer.write(
                        downsampled,
                        photometric="minisblack",
                        compression="deflate",
                        tile=tile_size,
                        planarconfig="CONTIG",
                        metadata=None,  # the OME description alone: no second, tifffile 'shaped' one
                    )
                    previous = downsampled
    # The new image keeps its parent's metadata (exposure, wavelengths, acquisition notes).
    try:
        from spacemap.image_metadata import inherit_ome

        inherit_ome(image_path, output_path, list(range(len(names))), notes={
            "Background removed from": Path(image_path).name,
            "Levels subtracted": "; ".join(f"{name}: {value:g}" for name, value in thresholds.items()) or "none",
        })
    except Exception as exc:  # the pixels are written; missing notes must not fail the step
        print(f"Could not copy the parent image's metadata to {Path(output_path).name}: {exc}")
    return output_path
