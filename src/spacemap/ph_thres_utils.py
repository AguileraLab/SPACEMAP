"""ph_thres_utils.py
Unified threshold-based phenotyping utilities (reference-aligned)

This module consolidates all utilities required for the threshold-based
phenotyping workflow, implementing the reference script logic while remaining
import-friendly for the Step6C notebook. The goals:
    * Batch / ROI specific threshold subtraction
    * Flexible, ordered normalization steps (Z1 -> Z2 -> CDF -> -Log)
    * Binary phenotype flag construction with AND / OR logic
    * Hierarchy pruning (child phenotypes override parent)
    * Tie-breaking using highest unique normalized marker expression
    * Heatmap generation on (normalized) marker values

Functions are intentionally stateless (no hidden globals) except that some
helper routines accept phenotype definitions / hierarchy as parameters.
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from tqdm.auto import tqdm
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

NORMALIZATION_EPSILON = 1e-9


def _canonical_tie_score_transform(value: object) -> str:
    """Map UI and legacy names to a supported threshold-distance transform."""
    key = "" if value is None else str(value).strip().lower()
    aliases = {
        "": "identity_residual",
        "none": "identity_residual",
        "identity": "identity_residual",
        "identity_residual": "identity_residual",
        "raw": "identity_residual",
        "raw_residual": "identity_residual",
        "arcsinh": "asinh_difference",
        "asinh": "asinh_difference",
        "asinh_difference": "asinh_difference",
        "arcsinh_difference": "asinh_difference",
        "asinh_distance": "asinh_difference",
        "log": "log1p_difference",
        "log1p": "log1p_difference",
        "log_difference": "log1p_difference",
        "log1p_difference": "log1p_difference",
    }
    if key not in aliases:
        raise ValueError(f"Unknown RESOLVE transformation: {value!r}")
    return aliases[key]


def _transform_threshold_distance(
    raw_values: pd.DataFrame,
    threshold_values: pd.DataFrame | pd.Series | float,
    transformation: object,
    arcsinh_cofactor: object = 5.0,
) -> pd.DataFrame:
    """Return a non-negative transformed distance above the marker threshold."""
    runtime_transform = _canonical_tie_score_transform(transformation)
    if runtime_transform == "asinh_difference":
        try:
            cofactor = float(arcsinh_cofactor)
        except (TypeError, ValueError) as exc:
            raise ValueError("ArcSinh cofactor must be a positive number") from exc
        if not np.isfinite(cofactor) or cofactor <= 0:
            raise ValueError("ArcSinh cofactor must be a positive finite number")
        distance = (
            np.arcsinh(raw_values / cofactor)
            - np.arcsinh(threshold_values / cofactor)
        )
    elif runtime_transform == "log1p_difference":
        transformed_raw = np.log1p(raw_values.clip(lower=0))
        if hasattr(threshold_values, "clip"):
            transformed_threshold = np.log1p(threshold_values.clip(lower=0))
        else:
            transformed_threshold = np.log1p(max(float(threshold_values), 0.0))
        distance = transformed_raw - transformed_threshold
    else:
        distance = raw_values - threshold_values
    return distance.clip(lower=0)

__all__ = [
        # Core functions used by notebook
        "apply_thresholds_per_batch",
        "apply_z1_positivity",
        "assign_phenotypes_binary_tiebreak", 
        "apply_threshold_phenotypes",
        "generate_threshold_heatmap",
        # Internal helper functions
        "apply_flexible_normalization",
        "set_classification_flags",
        "prune_parent_categories",
]

def apply_flexible_normalization(
    df: pd.DataFrame,
    marker_cols: Sequence[str],
    group_cols: Optional[Sequence[str]],
    suffix: str,
    apply_z1: bool = True,
    apply_z2: bool = True,
    apply_cdf: bool = True,
    apply_nlog: bool = True,
    order: Optional[Sequence[str]] = None,
    epsilon: float = NORMALIZATION_EPSILON,
) -> pd.DataFrame:
    """Reference-aligned flexible normalization with configurable order.

    Order of operations (if enabled):
      Z1 (marker-wise within group) -> Z2 (cell-wise) -> CDF -> -Log.

    Returns ONLY the new normalized columns (indexed like input df).
    """
    marker_cols = [m for m in marker_cols if m in df.columns]
    if not marker_cols:
        return pd.DataFrame(index=df.index)

    valid_group_cols: List[str] = []
    if group_cols:
        valid_group_cols = [c for c in group_cols if c in df.columns]

    new_col_names = [m + suffix for m in marker_cols]

    step_order = ["Z1", "Z2", "CDF", "NLog"]
    if order:
        step_order = [str(s).strip() for s in order if str(s).strip()]

    def _normalize_block(block: pd.DataFrame) -> pd.DataFrame:
        data = block[marker_cols].astype(float).values
        for step in step_order:
            if step == "Z1" and apply_z1:
                means = np.nanmean(data, axis=0)
                stds = np.nanstd(data, axis=0)
                stds[stds == 0] = epsilon
                data = (data - means) / stds
            elif step == "Z2" and apply_z2:
                means2 = np.nanmean(data, axis=1, keepdims=True)
                stds2 = np.nanstd(data, axis=1, keepdims=True)
                stds2[stds2 == 0] = epsilon
                data = (data - means2) / stds2
            elif step == "CDF" and apply_cdf:
                data = norm.cdf(data)
            elif step == "NLog" and apply_nlog:
                data = -np.log(np.maximum(1 - data, epsilon))
        # Replace invalid
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(data, index=block.index, columns=new_col_names)

    if valid_group_cols:
        parts = []
        for _, g in df.groupby(valid_group_cols, group_keys=False):
            parts.append(_normalize_block(g))
        norm_df = pd.concat(parts).sort_index()
    else:
        norm_df = _normalize_block(df)

    return norm_df


# ---------------------------------------------------------------------------
# Reference phenotyping logic utilities
# ---------------------------------------------------------------------------
def set_classification_flags(df_with_binary_markers: pd.DataFrame, bin_suffix: str, phenotype_definitions: Dict[str, List[List[str]]]) -> pd.DataFrame:
    """Construct initial phenotype flag matrix using OR-of-AND logic."""
    result_df = pd.DataFrame(index=df_with_binary_markers.index)
    marker_cols_present = [col for col in df_with_binary_markers.columns if col.endswith(bin_suffix)]
    raw_marker_to_bin_col = {re.sub(f'{bin_suffix}$', '', col): col for col in marker_cols_present}

    for phenotype in phenotype_definitions.keys():
        result_df[phenotype] = 0

    for phenotype, or_conditions in phenotype_definitions.items():
        or_result = pd.Series(0, index=df_with_binary_markers.index)
        for and_condition in or_conditions:
            bin_cols = [raw_marker_to_bin_col.get(m) for m in and_condition]
            if not all(c is not None and c in df_with_binary_markers.columns for c in bin_cols):
                continue
            and_result = pd.Series(1, index=df_with_binary_markers.index)
            for bin_col in bin_cols:
                and_result = and_result & df_with_binary_markers[bin_col].astype(bool)
            or_result = or_result | and_result
        result_df[phenotype] = or_result.astype(int)
    return result_df


def prune_parent_categories(categories_df: pd.DataFrame, hierarchy: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Unset parent flags where child (more specific) phenotype is positive."""
    pruned = categories_df.copy()
    for child, parent in hierarchy.items():
        if parent is None:
            continue
        if child in pruned.columns and parent in pruned.columns:
            mask = pruned[child] == 1
            pruned.loc[mask, parent] = 0
    return pruned


def _assign_single_row(row: pd.Series, phenotype_flag_cols: List[str], norm_marker_cols_map: Dict[str, str],
                       phenotype_definitions: Dict[str, List[List[str]]], hierarchy: Dict[str, Optional[str]]) -> str:
    positive = [p for p in phenotype_flag_cols if row.get(p, 0) == 1]
    if not positive:
        return "unclassified"
    if len(positive) == 1:
        return positive[0]
    # hierarchy pruning
    pruned = positive.copy()
    for child, parent in hierarchy.items():
        if parent and child in pruned and parent in pruned:
            pruned.remove(parent)
    if len(pruned) == 1:
        return pruned[0]
    # tie-break using unique marker max
    phenotype_to_markers: Dict[str, set] = {}
    for pheno in pruned:
        markers = set()
        for and_clause in phenotype_definitions.get(pheno, []):
            markers.update(and_clause)
        phenotype_to_markers[pheno] = markers
    all_markers_union = set.union(*phenotype_to_markers.values())
    phenotype_max_expr: Dict[str, float] = {}
    for pheno, markers in phenotype_to_markers.items():
        other_markers = all_markers_union - markers
        non_shared = markers - other_markers  # unique markers (fast but simplistic)
        use_markers = non_shared if non_shared else markers
        max_expr = -np.inf
        for m in use_markers:
            col = norm_marker_cols_map.get(m)
            if col and pd.notna(row.get(col, np.nan)):
                val = row[col]
                if val > max_expr:
                    max_expr = val
        phenotype_max_expr[pheno] = max_expr if max_expr > -np.inf else 0.0
    return max(phenotype_max_expr.items(), key=lambda kv: kv[1])[0]


def apply_supportive_markers(
    assignments: "pd.Series",
    df_processed: "pd.DataFrame",
    bin_marker_cols: Sequence[str],
    norm_marker_cols: Sequence[str],
    phenotype_definitions: Dict[str, List[List[str]]],
    supportive: Dict[str, List[str]],
) -> "pd.Series":
    """Let non-essential evidence settle contests between qualifying phenotypes.

    A supportive marker never makes a phenotype fire. It only matters when a
    cell satisfies the required markers of more than one phenotype and something
    has to break the tie: the phenotype whose supportive markers are strongest
    wins. Any one supportive marker suffices, so they combine with max.

    Qualification is read from the *binary* (thresholded) columns. The
    normalised columns cannot be used for this - the CDF step maps every value
    into (0, 1), so a "> 0" test there is true for every cell and every marker,
    which would make every cell look contested and hand the decision entirely to
    the supportive markers.
    """
    import numpy as np
    import pandas as pd

    if not supportive:
        return assignments

    bin_map = {}
    for col in bin_marker_cols:
        if "_bin_" in col:
            bin_map[col.split("_bin_")[0]] = col
    norm_map = {col.split("_norm_")[0]: col for col in norm_marker_cols if "_norm_" in col}
    if not bin_map:
        return assignments

    def support_strength(markers: Sequence[str]) -> "pd.Series":
        columns = [norm_map[m] for m in markers if m in norm_map and norm_map[m] in df_processed.columns]
        if not columns:
            return pd.Series(0.0, index=df_processed.index)
        return df_processed[columns].apply(pd.to_numeric, errors="coerce").max(axis=1).fillna(0.0)

    qualifies = {}
    for name, groups in phenotype_definitions.items():
        satisfied = pd.Series(False, index=df_processed.index)
        for group in groups:
            columns = [bin_map[m] for m in group if m in bin_map and bin_map[m] in df_processed.columns]
            if len(columns) != len(group):
                continue
            satisfied = satisfied | df_processed[columns].astype(bool).all(axis=1)
        qualifies[name] = satisfied
    if not qualifies:
        return assignments

    contested = pd.concat(qualifies.values(), axis=1).sum(axis=1).gt(1)
    if not contested.any():
        return assignments

    names = list(qualifies)
    scores = pd.DataFrame(
        {n: support_strength(supportive.get(n, [])).where(qualifies[n], -np.inf) for n in names},
        index=df_processed.index,
    )
    best = scores.max(axis=1)
    winner = scores.idxmax(axis=1)
    # Only move a cell that is genuinely contested, whose winner has real
    # supportive evidence, and where the current call also qualified - so a
    # phenotype without supportive markers is never displaced by default.
    current_qualifies = pd.Series(
        [bool(qualifies.get(a, pd.Series(False, index=df_processed.index)).iloc[i])
         for i, a in enumerate(assignments)],
        index=df_processed.index,
    )
    movable = contested & current_qualifies & np.isfinite(best) & best.gt(0) & winner.ne(assignments)
    result = assignments.copy()
    result.loc[movable] = winner.loc[movable]
    return result


def apply_exclusion_markers(
    assignments: "pd.Series",
    df_processed: "pd.DataFrame",
    norm_marker_cols: Sequence[str],
    phenotype_definitions: Dict[str, List[List[str]]],
    exclusions: Dict[str, List[str]],
    reassign: bool = True,
) -> "pd.Series":
    """Overturn a call when a forbidden marker outweighs the ones that made it.

    The comparison is deliberately relative, not absolute. Spillover from a
    touching neighbour puts a little of almost every marker into almost every
    cell - on our data roughly one in ten true CD8 cells carries some CD4 - so
    vetoing on "any signal at all" discards far more correct calls than wrong
    ones. A veto only fires when a forbidden marker is *stronger* than the
    markers that justified the phenotype, which is the situation that actually
    indicates a misassignment.

    With ``reassign`` the cell moves to the phenotype whose own required marker
    is the forbidden one that won, rather than being discarded.
    """
    import numpy as np
    import pandas as pd

    if not exclusions:
        return assignments

    norm_map = {col.split('_norm_')[0]: col for col in norm_marker_cols if '_norm_' in col}

    def strength(marker: str) -> "pd.Series":
        column = norm_map.get(marker)
        if column is None or column not in df_processed.columns:
            return pd.Series(np.nan, index=df_processed.index)
        return pd.to_numeric(df_processed[column], errors="coerce")

    result = assignments.copy()
    for phenotype, forbidden in exclusions.items():
        forbidden = [m for m in (forbidden or []) if m in norm_map]
        if not forbidden:
            continue
        target = assignments.eq(phenotype)
        if not target.any():
            continue
        # The best evidence this phenotype had: the strongest marker among the
        # ones its own rules require.
        required = sorted({m for group in phenotype_definitions.get(phenotype, []) for m in group})
        required = [m for m in required if m in norm_map]
        if not required:
            continue
        own = pd.concat([strength(m) for m in required], axis=1).max(axis=1)
        # Column labels must be the bare marker names: idxmax below returns a
        # label, and it has to be something the phenotype lookup recognises.
        rival = pd.concat([strength(m).rename(m) for m in forbidden], axis=1)
        rival.columns = list(forbidden)
        rival_best = rival.max(axis=1)
        vetoed = target & rival_best.gt(own).fillna(False)
        if not vetoed.any():
            continue
        if not reassign:
            result.loc[vetoed] = "Others"
            continue
        winner_marker = rival.idxmax(axis=1)
        # Move the cell to whichever phenotype requires the marker that won.
        owner = {}
        for name, groups in phenotype_definitions.items():
            for group in groups:
                for marker in group:
                    owner.setdefault(marker, name)
        result.loc[vetoed] = [
            owner.get(winner_marker.get(index), "Others") for index in result.index[vetoed]
        ]
    return result


def assign_phenotypes_binary_tiebreak(
    df_processed: pd.DataFrame,
    bin_marker_cols: Sequence[str],
    norm_marker_cols: Sequence[str],
    phenotype_col_name: str,
    phenotype_definitions: Dict[str, List[List[str]]],
    phenotype_hierarchy: Dict[str, Optional[str]],
) -> pd.DataFrame:
    """High-level phenotype assignment wrapper (reference aligned)."""
    if not bin_marker_cols:
        raise ValueError("No binary marker columns provided.")
    bin_suffix = '_' + bin_marker_cols[0].split('_')[-2] + '_' + bin_marker_cols[0].split('_')[-1]
    flags = set_classification_flags(df_processed, bin_suffix, phenotype_definitions)
    pruned = prune_parent_categories(flags, phenotype_hierarchy)
    norm_map = {col.split('_norm_')[0]: col for col in norm_marker_cols if '_norm_' in col}
    apply_df = pd.concat([
        *(df_processed[c] for c in norm_marker_cols if c in df_processed.columns),
        pruned
    ], axis=1)
    phenotype_flag_cols = [c for c in pruned.columns if c in apply_df.columns]
    assigned = apply_df.apply(
        _assign_single_row,
        axis=1,
        args=(phenotype_flag_cols, norm_map, phenotype_definitions, phenotype_hierarchy),
    )
    out = pd.DataFrame(index=df_processed.index)
    out[phenotype_col_name] = assigned
    return out


def _numbered_threshold_columns(columns, polygon_prefix: str = "Polygon") -> List[str]:
    """Return only naturally ordered ``<prefix>_<integer>`` columns."""
    prefix = str(polygon_prefix or "Polygon").strip().rstrip("_")
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$", flags=re.IGNORECASE)
    numbered = []
    for position, column in enumerate(columns):
        match = pattern.fullmatch(str(column))
        if match:
            numbered.append((int(match.group(1)), position, column))
    return [column for _, _, column in sorted(numbered)]


def _normalise_threshold_marker(marker: str) -> str:
    marker = str(marker).strip()
    marker = re.sub(r"\s*\(.*\)$", "", marker)
    marker = re.sub(r"\s*\[.*\]$", "", marker)
    for suffix in ("_corr_norm", "_norm", "_corr"):
        if marker.endswith(suffix):
            marker = marker[:-len(suffix)]
            break
    return marker.strip()


def _extract_thresholds_v2(
    table: pd.DataFrame,
    markers: Sequence[str],
    threshold_statistic: str,
    threshold_source: str = "Auto",
    marker_col_name: str | None = None,
    value_col_name: str | None = None,
    polygon_prefix: str = "Polygon",
) -> Dict[str, float]:
    """Extract a complete v2 threshold mapping or raise ``ValueError``.

    Auto precedence matches the RESOLVE review loader: explicit value column,
    manual override, numbered polygon values, then final scalar.
    """
    if table.empty:
        raise ValueError("threshold CSV is empty")

    columns_lower = {str(column).lower(): column for column in table.columns}
    marker_col = (
        columns_lower.get(str(marker_col_name).lower()) if marker_col_name else None
    ) or columns_lower.get("marker") or columns_lower.get("markers")
    if marker_col is None:
        raise ValueError("threshold CSV has no marker column")

    explicit_value_col = (
        columns_lower.get(str(value_col_name).lower()) if value_col_name else None
    )
    final_scalar_cols = list(dict.fromkeys(column for column in (
        columns_lower.get("threshold_final"),
        columns_lower.get("final_threshold"),
        columns_lower.get("threshold"),
        columns_lower.get("median"),
        columns_lower.get("value"),
        columns_lower.get("cutoff"),
        columns_lower.get("gate"),
    ) if column))
    manual_value_cols = [column for column in (
        columns_lower.get("manual_override"),
        columns_lower.get("threshold_override"),
        columns_lower.get("override_value"),
    ) if column]
    override_flag_cols = [column for column in (
        columns_lower.get("override"),
        columns_lower.get("override_applied"),
        columns_lower.get("manual_override_applied"),
        columns_lower.get("is_override"),
    ) if column]
    polygon_cols = _numbered_threshold_columns(table.columns, polygon_prefix)

    source_key = str(threshold_source or "Auto").strip().lower().replace("_", " ")
    if source_key in {"polygon", "polygons", "polygon value", "polygon values"}:
        source_key = "polygon"
    elif source_key in {"final", "scalar", "final scalar", "precomputed", "direct"}:
        source_key = "scalar"
    else:
        source_key = "auto"

    statistic_key = str(threshold_statistic or "median").strip().lower()
    statistic_key = {
        "p05": "percentile_05", "p5": "percentile_05", "q05": "percentile_05",
        "percentile_5": "percentile_05",
        "p25": "percentile_25", "q25": "percentile_25",
        "p75": "percentile_75", "q75": "percentile_75",
    }.get(statistic_key, statistic_key)

    def _numeric(value) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) else None

    def _first_numeric(row: pd.Series, columns: Sequence[str]):
        for column in columns:
            value = _numeric(row.get(column))
            if value is not None:
                return value
        return None

    def _flag_true(value) -> bool:
        if value is None or pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"", "0", "false", "no", "n", "none", "nan"}:
            return False
        if text in {"1", "true", "yes", "y", "manual", "override", "overridden"}:
            return True
        try:
            return float(text) != 0.0
        except ValueError:
            return False

    def _manual_value(row: pd.Series) -> float | None:
        for column in manual_value_cols:
            raw = row.get(column)
            if raw is None or pd.isna(raw):
                continue
            if isinstance(raw, (bool, np.bool_)) or (
                isinstance(raw, str)
                and raw.strip().lower() in {
                    "true", "false", "yes", "no", "y", "n", "manual", "override", "overridden"
                }
            ):
                if _flag_true(raw):
                    return _first_numeric(row, final_scalar_cols)
                continue
            value = _numeric(raw)
            if value is not None:
                return value
        if any(_flag_true(row.get(column)) for column in override_flag_cols):
            return _first_numeric(row, final_scalar_cols)
        return None

    def _polygon_stat(values: pd.Series) -> float | None:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return None
        if statistic_key == "mean":
            return float(numeric.mean())
        if statistic_key == "min":
            return float(numeric.min())
        if statistic_key == "percentile_05":
            return float(numeric.quantile(0.05))
        if statistic_key == "percentile_25":
            return float(numeric.quantile(0.25))
        if statistic_key == "percentile_75":
            return float(numeric.quantile(0.75))
        return float(numeric.median())

    rows_by_marker = {}
    for _, row in table.dropna(subset=[marker_col]).iterrows():
        key = _normalise_threshold_marker(row[marker_col]).casefold()
        rows_by_marker.setdefault(key, row)

    thresholds: Dict[str, float] = {}
    missing = []
    for marker in markers:
        row = rows_by_marker.get(_normalise_threshold_marker(marker).casefold())
        if row is None:
            missing.append(str(marker))
            continue
        value = None
        if source_key in {"auto", "scalar"} and explicit_value_col:
            value = _numeric(row.get(explicit_value_col))
        if value is None and source_key == "auto":
            value = _manual_value(row)
        if value is None and source_key in {"auto", "polygon"} and polygon_cols:
            value = _polygon_stat(row[polygon_cols])
        if value is None and source_key in {"auto", "scalar"}:
            value = _first_numeric(row, final_scalar_cols)
        if value is None:
            missing.append(str(marker))
        else:
            thresholds[str(marker)] = float(value)

    if missing:
        raise ValueError("missing threshold(s) for marker(s): " + ", ".join(missing))
    return thresholds


def apply_thresholds_per_batch(
    df_main: pd.DataFrame,
    method_tag: str,
    threshold_roi_dir: str | Path,
    markers_for_thresholding_raw: Sequence[str],
    threshold_statistic: str,
    norm_group_cols: Optional[Sequence[str]],
    norm_flags: Tuple[bool, bool, bool, bool],
    batch_col: str,
    norm_order: Optional[Sequence[str]] = None,
    apply_thresholds: bool = True,
    tie_score_transform: str = "identity_residual",
    tie_score_cofactor: float = 5.0,
    threshold_source: str = "Auto",
    threshold_marker_col: str | None = None,
    threshold_value_col: str | None = None,
    polygon_prefix: str = "Polygon",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply batch-specific polygon thresholds + normalization (reference logic).
    
    Args:
        df_main: Main dataframe with marker columns
        method_tag: Tag for column suffixes
        threshold_roi_dir: Directory containing threshold CSV files
        markers_for_thresholding_raw: List of marker column names
        threshold_statistic: Statistic to use from threshold files. Supported
            values are 'median', 'min', 'mean', 'percentile_05' (p05),
            'percentile_25' (p25), and 'percentile_75' (p75).
        norm_group_cols: Columns to group by for normalization
        norm_flags: Tuple of normalization flags (z1, z2, cdf, nlog)
        batch_col: Column name containing batch identifiers
        norm_order: Optional order of normalization steps
        apply_thresholds: If False, skip thresholding and use raw markers as-is
        tie_score_transform: Continuous score sent to normalization after binary
            gating. ``identity_residual`` uses ``max(raw-threshold, 0)``.
            ``asinh_difference`` uses
            ``max(asinh(raw/cofactor)-asinh(threshold/cofactor), 0)``.
            ``log1p_difference`` uses
            ``max(log1p(raw)-log1p(threshold), 0)``.
            This setting never changes the binary positive/negative gate.
        tie_score_cofactor: Positive ArcSinh cofactor used only by
            ``asinh_difference``.
        threshold_source: ``Auto``, ``Polygon values``, or ``Final scalar``.
            Auto uses explicit value, manual override, numbered polygons, then
            final scalar precedence.
        threshold_marker_col: Optional marker-column override.
        threshold_value_col: Optional explicit value-column override.
        polygon_prefix: Prefix for strictly numbered polygon columns.
        
    Returns:
        Tuple of (binary_df, normalized_df)
    """
    bin_suffix = f"_bin_{method_tag}"
    norm_suffix = f"_norm_{method_tag}"
    
    # If not applying thresholds, use raw markers directly
    if not apply_thresholds:
        print("Skipping threshold subtraction - using raw markers as-is")
        # Create binary columns from raw markers (>0 = positive)
        df_binary = pd.DataFrame(index=df_main.index)
        for marker in markers_for_thresholding_raw:
            if marker in df_main.columns:
                df_binary[f"{marker}{bin_suffix}"] = (df_main[marker] > 0).fillna(False).astype(int)
        
        # Apply normalization to raw markers
        apply_z1, apply_z2, apply_cdf, apply_nlog = norm_flags
        group_cols = [c for c in (norm_group_cols or []) if c in df_main.columns]
        markers_present = [m for m in markers_for_thresholding_raw if m in df_main.columns]
        
        raw_block = df_main[markers_present].apply(pd.to_numeric, errors="coerce")
        transformed = _transform_threshold_distance(
            raw_block,
            0.0,
            tie_score_transform,
            tie_score_cofactor,
        )
        norm_input = (
            pd.concat([transformed, df_main[group_cols]], axis=1)
            if group_cols else transformed
        )
        df_norm = apply_flexible_normalization(
            norm_input,
            markers_present,
            group_cols,
            norm_suffix,
            apply_z1=apply_z1,
            apply_z2=apply_z2,
            apply_cdf=apply_cdf,
            apply_nlog=apply_nlog,
            order=norm_order,
        )
        
        return df_binary, df_norm
    
    # Apply threshold subtraction logic.  Validate every batch x marker first,
    # so a missing/malformed file cannot yield a partially thresholded result.
    threshold_roi_dir = Path(threshold_roi_dir)
    if not threshold_roi_dir.is_dir():
        raise ValueError(f"Threshold directory does not exist: {threshold_roi_dir}")
    if not batch_col or batch_col not in df_main.columns:
        raise ValueError(f"Batch column is missing: {batch_col}")
    if df_main[batch_col].isna().any():
        raise ValueError(f"Batch column '{batch_col}' contains missing values")

    markers_present = [m for m in markers_for_thresholding_raw if m in df_main.columns]
    missing_data_markers = [m for m in markers_for_thresholding_raw if m not in df_main.columns]
    if missing_data_markers:
        raise ValueError(
            "Data is missing requested marker column(s): " + ", ".join(missing_data_markers)
        )
    if not markers_present:
        raise ValueError("No requested marker columns are present in the data")

    thresholds_by_batch: Dict[object, Dict[str, float]] = {}
    batches = df_main[batch_col].unique()
    for batch_id in batches:
        safe_batch_id = str(batch_id).strip()
        prefix_match = re.match(r'(P\d+(?:_P\d+)?_roi_\d+)', safe_batch_id)
        if prefix_match:
            safe_batch_id = prefix_match.group(1)
        else:
            for pattern in [r'\.ome', r'_cell_features']:
                if re.search(pattern, safe_batch_id):
                    safe_batch_id = re.split(pattern, safe_batch_id)[0]
                    break
        candidate_files = [
            threshold_roi_dir / f"{safe_batch_id}.ome.tiff_-_Merged_Image.csv",
            threshold_roi_dir / f"{safe_batch_id}_-_Merged_Image.csv",
            threshold_roi_dir / f"{safe_batch_id}.csv",
        ]
        thresh_file = next((path for path in candidate_files if path.is_file()), None)
        if thresh_file is None:
            raise ValueError(
                f"No threshold CSV matched batch '{batch_id}' in {threshold_roi_dir}"
            )
        try:
            threshold_table = pd.read_csv(thresh_file)
            thresholds_by_batch[batch_id] = _extract_thresholds_v2(
                threshold_table,
                markers_present,
                threshold_statistic,
                threshold_source=threshold_source,
                marker_col_name=threshold_marker_col,
                value_col_name=threshold_value_col,
                polygon_prefix=polygon_prefix,
            )
        except Exception as exc:
            raise ValueError(
                f"Invalid threshold coverage for batch '{batch_id}' in {thresh_file.name}: {exc}"
            ) from exc

    df_binary = pd.DataFrame(index=df_main.index)
    df_norm = pd.DataFrame(index=df_main.index)
    for batch_id in tqdm(batches, desc="Processing batches"):
        mask = df_main[batch_col] == batch_id
        df_batch = df_main.loc[mask]
        thresholds = thresholds_by_batch[batch_id]
        raw_block = df_batch[markers_present].apply(pd.to_numeric, errors='coerce')
        thr_series = pd.Series(thresholds, dtype=float).reindex(markers_present)
        if thr_series.isna().any() or not np.isfinite(thr_series).all():
            raise ValueError(f"Non-finite threshold mapping for batch '{batch_id}'")
        residual = raw_block.subtract(thr_series, axis=1)
        for marker in markers_present:
            df_binary.loc[mask, f"{marker}{bin_suffix}"] = (
                residual[marker] > 0
            ).fillna(False).astype(int)

        temp = _transform_threshold_distance(
            raw_block,
            thr_series,
            tie_score_transform,
            tie_score_cofactor,
        )
        apply_z1, apply_z2, apply_cdf, apply_nlog = norm_flags
        group_cols = [c for c in (norm_group_cols or []) if c in df_batch.columns]
        norm_input = pd.concat([temp, df_batch[group_cols]], axis=1) if group_cols else temp
        norm_block = apply_flexible_normalization(
            norm_input,
            markers_present,
            group_cols,
            norm_suffix,
            apply_z1=apply_z1,
            apply_z2=apply_z2,
            apply_cdf=apply_cdf,
            apply_nlog=apply_nlog,
            order=norm_order,
        )
        for column in norm_block.columns:
            df_norm.loc[mask, column] = norm_block[column]

    expected_binary = [f"{marker}{bin_suffix}" for marker in markers_present]
    if any(column not in df_binary.columns for column in expected_binary):
        raise ValueError("Threshold application did not produce every requested binary marker")
    if df_binary[expected_binary].isna().any().any():
        raise ValueError("Threshold application left cells without binary marker coverage")
    return df_binary, df_norm


DEFAULT_Z1_CUTOFF = 1.5


def apply_z1_positivity(
    df_main: pd.DataFrame,
    markers: Sequence[str],
    batch_col: Optional[str] = None,
    method_tag: str = "resolve",
    cutoff: float = DEFAULT_Z1_CUTOFF,
    cofactor: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Call a marker positive where its Z1 score is above ``cutoff``.

    For projects without threshold tables. Each raw marker is transformed with
    ``arcsinh(value / cofactor)`` and turned into a Z1 score (marker-wise z-score)
    within each batch; values above ``cutoff`` count as positive. Z2, CDF and
    NLog are not applied because each rescales the score and would change what
    the cutoff means. The Z1 scores are returned as the tie-break scores.

    This is a relative rule: markers carried by a large share of cells, or with a
    few extremely bright cells, tend to be under-called. Prefer image-based
    thresholds when they exist.

    Returns ``(binary, scores)`` with columns ``<marker>_bin_<tag>`` and
    ``<marker>_norm_<tag>``, the same layout as ``apply_thresholds_per_batch``.
    """
    markers_present = [marker for marker in markers if marker in df_main.columns]
    raw = df_main[markers_present].apply(pd.to_numeric, errors="coerce")
    transformed = np.arcsinh(raw / float(cofactor))
    group_cols = [batch_col] if batch_col and batch_col in df_main.columns else []
    norm_input = pd.concat([transformed, df_main[group_cols]], axis=1) if group_cols else transformed
    scores = apply_flexible_normalization(
        norm_input,
        markers_present,
        group_cols or None,
        f"_norm_{method_tag}",
        apply_z1=True,
        apply_z2=False,
        apply_cdf=False,
        apply_nlog=False,
        order=["Z1"],
    )
    # Cells with no batch value are left out of every batch group; they are
    # scored 0 (not positive) rather than dropped, so the output covers every cell.
    scores = scores.reindex(df_main.index, fill_value=0.0)
    binary = pd.DataFrame(index=df_main.index)
    for marker in markers_present:
        binary[f"{marker}_bin_{method_tag}"] = (scores[f"{marker}_norm_{method_tag}"] > float(cutoff)).astype(int)
    return binary, scores


def generate_threshold_heatmap(df, assignment_col, markers, marker_suffix, figsize, save_path):
    """
    Generates a heatmap of mean marker expression for each phenotype.

    Args:
        df (pd.DataFrame): The input data with phenotype assignments.
        assignment_col (str): The column with final phenotype labels.
        markers (list): The list of base marker names.
        marker_suffix (str): The suffix of the marker columns to use for the heatmap.
        figsize (tuple): The figure size.
        save_path (str or Path): The path to save the heatmap image.

    Returns:
        matplotlib.axes.Axes: The Axes object of the heatmap.
    """
    marker_cols = [f"{m}{marker_suffix}" for m in markers if f"{m}{marker_suffix}" in df.columns]
    
    if not marker_cols:
        print("Error: No marker columns found for heatmap.")
        return None

    mean_expression = df.groupby(assignment_col)[marker_cols].mean()
    
    # Clean up column names for display
    mean_expression.columns = [col.replace(marker_suffix, '') for col in marker_cols]
    
    # Z-score normalize the data for better color scaling
    mean_expression_zscore = mean_expression.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=0)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        mean_expression_zscore, 
        cmap='bwr',  # Red-Blue reversed (red=high, blue=low)
        center=0,  # Center colormap at 0
        vmin=-2,   # Set symmetric color scale
        vmax=2,
        annot=False, 
        cbar_kws={'label': 'Z-score Expression', 'shrink': 0.8},
        linewidths=0.1,
        linecolor='white'
    )
    
    plt.xlabel('Markers', fontsize=12)
    plt.ylabel('Phenotypes', fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return ax


def apply_threshold_phenotypes(
    data,
    assignment_df,
    phenotype_col,
    output_base_dir,
    method_prefix,
    cell_id_col
):
    """
    Apply threshold phenotype assignments to unified dataset (compatible with Leiden/SOM pattern).
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with cell data (should exclude normalized columns)
    assignment_df : pd.DataFrame  
        Dataframe containing phenotype assignments with same index as data
    phenotype_col : str
        Name of phenotype column to add
    output_base_dir : Path
        Base output directory for saving files
    method_prefix : str
        Method prefix for output files
    cell_id_col : str
        Name of cell ID column for proper anchoring
        
    Returns
    -------
    dict
        Dictionary with results including unified_file, summary info, etc.
    """
    from pathlib import Path
    import pandas as pd
    
    if output_base_dir is None:
        raise ValueError("output_base_dir must be provided")
    
    output_base_dir = Path(output_base_dir)
    phenotyping_dir = output_base_dir
    phenotyping_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate cell ID column
    if cell_id_col not in data.columns:
        raise ValueError(f"Cell ID column '{cell_id_col}' not found in data")
    
    print(f"Applying {method_prefix} phenotypes to unified dataset...")
    
    # Create working copy and add phenotype assignments
    result_data = data.copy()
    result_data[phenotype_col] = assignment_df[phenotype_col].reindex(data.index)
    
    # Filter out any normalized columns that might have slipped in
    norm_cols = [c for c in result_data.columns if c.endswith('_norm')]
    if norm_cols:
        result_data = result_data.drop(columns=norm_cols)
        print(f"Filtered out {len(norm_cols)} normalized columns")
    
    # Define unified output file
    unified_file = phenotyping_dir / 'all_samples_combined_corrected_phenotyped.csv'
    
    # Handle existing unified file using cell_id-based merging (like Leiden/SOM)
    if unified_file.exists():
        print(f"Updating existing unified file: {unified_file}")
        existing_df = pd.read_csv(unified_file)
        
        if cell_id_col not in existing_df.columns:
            raise ValueError(f"Existing unified file missing '{cell_id_col}' column")
        
        # Remove existing threshold column if present
        if phenotype_col in existing_df.columns:
            existing_df = existing_df.drop(columns=[phenotype_col])
            print(f"Removed existing {phenotype_col} column")
        
        # Prepare update data with only cell_id and new phenotype column
        update_data = result_data[[cell_id_col, phenotype_col]].copy()
        
        # Merge based on cell_id
        existing_df = existing_df.set_index(cell_id_col)
        update_data = update_data.set_index(cell_id_col)
        
        # Update with new phenotype column
        existing_df[phenotype_col] = update_data[phenotype_col]
        
        # Reset index to get cell_id back as column (avoid duplicates)
        if cell_id_col in existing_df.columns:
            existing_df = existing_df.drop(columns=[cell_id_col])
        unified_df = existing_df.reset_index()
        
        print(f"Updated {len(unified_df):,} cells with {phenotype_col} assignments")
    else:
        # Create new unified file
        print(f"Creating new unified file: {unified_file}")
        unified_df = result_data.copy()
    
    # Save unified file
    unified_df.to_csv(unified_file, index=False)
    print(f"Saved unified file: {unified_file}")
    
    # Generate summary statistics
    phenotype_counts = result_data[phenotype_col].value_counts()
    summary_df = pd.DataFrame({
        'cell_count': phenotype_counts,
        'percentage': (phenotype_counts / len(result_data) * 100).round(1)
    })
    
    # Save summary file
    summary_file = phenotyping_dir / f'{method_prefix}_phenotype_summary.csv'
    summary_df.to_csv(summary_file)
    
    # Create mapping file (for compatibility)
    mapping_df = pd.DataFrame({
        'phenotype': summary_df.index,
        'cell_count': summary_df['cell_count'].values
    })
    mapping_file = phenotyping_dir / f'{method_prefix}_phenotype_mapping.csv'
    mapping_df.to_csv(mapping_file, index=False)
    
    print(f"Generated summary files:")
    print(f"  Summary: {summary_file}")
    print(f"  Mapping: {mapping_file}")
    
    return {
        'data': unified_df,
        'unified_file': unified_file,
        'summary_file': summary_file,
        'summary_df': summary_df,
        'mapping_file': mapping_file,
        'phenotype_col': phenotype_col
    }
