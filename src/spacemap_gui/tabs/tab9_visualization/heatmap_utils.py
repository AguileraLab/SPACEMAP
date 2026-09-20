"""Pure data helpers for publication heatmaps and phenotype dot plots.

The functions in this module intentionally do not import Streamlit.  Keeping
the numerical work separate from the GUI makes the transformation sequence
testable and gives notebook code a small, stable API to reuse.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


INPUT_AS_IS = "As-is"
INPUT_ARCSINH = "ArcSinh"
INPUT_INVERSE_ARCSINH = "Inverse ArcSinh (corrected markers only)"
INPUT_TRANSFORM_OPTIONS = (INPUT_AS_IS, INPUT_ARCSINH, INPUT_INVERSE_ARCSINH)
NORMALIZATION_STEPS = ("Z1", "Z2", "CDF", "NLog")


def _normalise_block(
    values: np.ndarray,
    *,
    apply_z1: bool,
    apply_z2: bool,
    apply_cdf: bool,
    apply_nlog: bool,
    epsilon: float,
) -> np.ndarray:
    """Apply the same ordered normalization sequence used by phenotyping."""
    processed = np.asarray(values, dtype=float).copy()

    if apply_z1:
        means = np.nanmean(processed, axis=0)
        stds = np.nanstd(processed, axis=0)
        stds = np.where(np.isfinite(stds) & (stds > 0), stds, epsilon)
        processed = (processed - means) / stds
        processed = np.clip(processed, -10.0, 10.0)

    if apply_z2:
        means = np.nanmean(processed, axis=1, keepdims=True)
        stds = np.nanstd(processed, axis=1, keepdims=True)
        stds = np.where(np.isfinite(stds) & (stds > 0), stds, epsilon)
        processed = (processed - means) / stds

    if apply_cdf:
        processed = norm.cdf(processed)

    if apply_nlog:
        processed = -np.log(np.maximum(1.0 - processed, epsilon))

    return np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)


def apply_marker_transformations(
    df: pd.DataFrame,
    marker_columns: Sequence[str],
    *,
    group_columns: Sequence[str] | None = None,
    input_transform: str = INPUT_AS_IS,
    arcsinh_cofactor: float = 5.0,
    apply_z1: bool = True,
    apply_z2: bool = False,
    apply_cdf: bool = False,
    apply_nlog: bool = False,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    """Transform marker values while retaining row indices and marker names.

    The operation order is input-scale transformation first, followed by
    ``Z1 -> Z2 -> CDF -> NLog``.  Z1 is marker-wise and Z2 is cell-wise.  If
    grouping columns are supplied, the full sequence is applied separately in
    each group, matching the behavior of SPACEMAP's phenotyping utilities.
    """
    markers = [str(column) for column in marker_columns]
    missing = [column for column in markers if column not in df.columns]
    if missing:
        raise ValueError(f"Marker columns not found: {missing}")
    if input_transform not in INPUT_TRANSFORM_OPTIONS:
        raise ValueError(f"Unknown input transformation: {input_transform}")
    if not np.isfinite(arcsinh_cofactor) or arcsinh_cofactor <= 0:
        raise ValueError("ArcSinh cofactor must be a positive finite number.")

    numeric = df[markers].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)

    if input_transform == INPUT_ARCSINH:
        values = np.arcsinh(values / float(arcsinh_cofactor))
    elif input_transform == INPUT_INVERSE_ARCSINH:
        corrected_positions = [index for index, marker in enumerate(markers) if marker.lower().endswith("_corr")]
        if corrected_positions:
            with np.errstate(over="ignore", invalid="ignore"):
                values[:, corrected_positions] = (
                    np.sinh(values[:, corrected_positions]) * float(arcsinh_cofactor)
                )

    normalization_enabled = any((apply_z1, apply_z2, apply_cdf, apply_nlog))
    if not normalization_enabled:
        transformed = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(transformed, index=df.index, columns=markers)

    valid_groups = [str(column) for column in (group_columns or []) if str(column) in df.columns]
    transformed = np.zeros_like(values, dtype=float)
    if valid_groups:
        # ``indices`` are positional indices, so this remains safe even when a
        # filtered frame retains non-consecutive index labels.
        grouped_positions = df.groupby(valid_groups, sort=False, dropna=False).indices.values()
        for positions in grouped_positions:
            positions_array = np.asarray(positions, dtype=int)
            transformed[positions_array, :] = _normalise_block(
                values[positions_array, :],
                apply_z1=apply_z1,
                apply_z2=apply_z2,
                apply_cdf=apply_cdf,
                apply_nlog=apply_nlog,
                epsilon=epsilon,
            )
    else:
        transformed = _normalise_block(
            values,
            apply_z1=apply_z1,
            apply_z2=apply_z2,
            apply_cdf=apply_cdf,
            apply_nlog=apply_nlog,
            epsilon=epsilon,
        )

    return pd.DataFrame(transformed, index=df.index, columns=markers)


def build_group_statistics(
    marker_values: pd.DataFrame,
    group_values: pd.Series,
    *,
    positive_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return mean expression, percent positive, and cell count per group."""
    if len(marker_values) != len(group_values):
        raise ValueError("Marker values and grouping values must contain the same number of rows.")

    group_name = "__spacemap_heatmap_group"
    groups = group_values.astype(str).to_numpy()
    grouped_values = marker_values.copy()
    grouped_values[group_name] = groups

    means = grouped_values.groupby(group_name, sort=False)[list(marker_values.columns)].mean()
    fractions = (
        marker_values.gt(float(positive_threshold))
        .assign(**{group_name: groups})
        .groupby(group_name, sort=False)[list(marker_values.columns)]
        .mean()
        .mul(100.0)
    )
    counts = pd.Series(groups, dtype="object").value_counts(sort=False)
    counts = counts.reindex(means.index).fillna(0).astype(int)
    means.index.name = "phenotype"
    fractions.index.name = "phenotype"
    counts.index.name = "phenotype"
    counts.name = "cell_count"
    return means, fractions, counts


def unique_display_labels(columns: Sequence[str], suffixes: Sequence[str]) -> list[str]:
    """Strip display suffixes without creating ambiguous duplicate labels."""
    original = [str(column) for column in columns]
    stripped: list[str] = []
    for column in original:
        label = column
        for suffix in suffixes:
            suffix_text = str(suffix)
            if suffix_text and label.endswith(suffix_text):
                label = label[: -len(suffix_text)]
                break
        stripped.append(label)

    duplicate_labels = {label for label in stripped if stripped.count(label) > 1}
    return [source if label in duplicate_labels else label for source, label in zip(original, stripped)]


def normalization_label(
    *,
    input_transform: str,
    apply_z1: bool,
    apply_z2: bool,
    apply_cdf: bool,
    apply_nlog: bool,
) -> str:
    """Build a compact, human-readable transformation description."""
    parts = [] if input_transform == INPUT_AS_IS else [input_transform]
    parts.extend(
        step
        for step, enabled in zip(
            NORMALIZATION_STEPS,
            (apply_z1, apply_z2, apply_cdf, apply_nlog),
        )
        if enabled
    )
    return " -> ".join(parts) if parts else "Input values (as-is)"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def publication_file_stem(
    figure_kind: str,
    phenotype_column: str,
    settings: Mapping[str, Any],
) -> str:
    """Create a readable, deterministic filename that encodes its settings."""
    canonical = json.dumps(_jsonable(settings), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]

    transform = str(settings.get("normalization", "as-is"))
    marker_order = str(settings.get("marker_order", "selected"))
    phenotype_order = str(settings.get("phenotype_order", "observed"))
    marker_count = int(settings.get("marker_count", 0) or 0)

    def slug(value: str, limit: int = 30) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
        return (cleaned or "none")[:limit].rstrip("-")

    return "__".join(
        (
            slug(figure_kind, 20),
            slug(phenotype_column, 40),
            f"m{marker_count}",
            slug(transform, 35),
            f"markers-{slug(marker_order, 20)}",
            f"groups-{slug(phenotype_order, 20)}",
            digest,
        )
    )


def settings_as_json(settings: Mapping[str, Any]) -> str:
    """Serialize figure settings for a sidecar reproducibility file."""
    return json.dumps(_jsonable(settings), indent=2, sort_keys=True)
