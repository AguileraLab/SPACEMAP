"""Leakage-resistant grouped validation for SPACEMAP phenotyping models.

This module contains no Streamlit or file-writing code. It retrains MR-ML and
Consensus-ML inside each leave-one-group-out fold and returns compact tables for
interactive review, plotting, and explicit export by the caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import ml_consensus
from . import ml_mr_utils


ProgressCallback = Callable[[str], None]
ROI_SUFFIX_PATTERN = re.compile(r"(?i)(?:[_\-\s]+roi[_\-\s]*\d+)$")
EXCLUDED_GROUP_TOKEN = "<exclude>"
INVALID_LABELS = {"", "unclassified", "unknown", "nan", "none"}


@dataclass
class GroupedValidationConfig:
    """Configuration for strict grouped phenotyping validation."""

    primary_reference_column: str
    evaluation_reference_columns: list[str]
    component_method_columns: list[str]
    marker_columns: list[str]
    validation_group_column: str
    group_mode: str = "As stored"
    group_mapping: dict[str, str] = field(default_factory=dict)

    development_reference_column: str | None = None
    internal_test_reference_column: str | None = None
    morphology_columns: list[str] = field(default_factory=list)
    compartment_columns: list[str] = field(default_factory=list)
    feature_weights: dict[str, float] = field(
        default_factory=lambda: {
            "markers": 1.0,
            "morphology": 1.0,
            "compartments": 0.5,
        }
    )
    use_z_normalization: bool = False
    batch_normalization_column: str = "batch_id"
    drop_invalid_marker_rows: bool = True

    mr_ml_method: str = "random_forest"
    consensus_ml_method: str = "random_forest"
    mr_n_estimators: int = 5000
    consensus_n_estimators: int = 1500
    quick_mode: bool = False
    quick_n_estimators: int = 100
    random_state: int = 42

    consensus_min_methods: int = 3
    consensus_min_training_cells: int = 50
    refined_confidence_threshold: float = 0.75
    refined_low_confidence_label: str = "Others"
    excluded_phenotypes: list[str] = field(default_factory=lambda: ["Others"])

    run_grouped_mr_ml: bool = True
    run_grouped_consensus_ml: bool = True
    run_internal_split: bool = False

    filter_column: str | None = None
    filter_value: Any = None


@dataclass
class GroupedValidationResult:
    """Tables returned by run_grouped_validation."""

    oof_predictions: pd.DataFrame
    internal_predictions: pd.DataFrame
    fold_audit: pd.DataFrame
    f1_weights: pd.DataFrame
    consensus_training_counts: pd.DataFrame
    feature_audit: dict[str, Any]
    configuration: dict[str, Any]


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(str(message))


def _unique(values: Sequence[str | None]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _excluded_set(values: Sequence[str]) -> set[str]:
    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    }


def _clean_labels(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _valid_reference_mask(
    series: pd.Series,
    excluded_phenotypes: Sequence[str],
) -> pd.Series:
    cleaned = _clean_labels(series)
    invalid = INVALID_LABELS | _excluded_set(excluded_phenotypes)
    return series.notna() & cleaned.ne("") & ~cleaned.str.lower().isin(invalid)


def _labeled_mask(series: pd.Series) -> pd.Series:
    cleaned = _clean_labels(series)
    return series.notna() & cleaned.ne("") & ~cleaned.str.lower().isin(INVALID_LABELS)


def build_validation_groups(
    series: pd.Series,
    mode: str,
    mapping: Mapping[str, str] | None = None,
) -> pd.Series:
    """Return normalized group labels using the same rules as the GUI scorer."""

    cleaned = series.astype("string").str.strip()
    if mode == "Remove ROI suffix":
        return cleaned.map(
            lambda value: ROI_SUFFIX_PATTERN.sub("", str(value)).strip()
            if pd.notna(value)
            else pd.NA
        ).astype("string")
    if mode == "Manual mapping":
        lookup = {
            str(source).strip(): str(target).strip()
            for source, target in (mapping or {}).items()
            if str(source).strip() and str(target).strip()
        }

        def _resolve(value):
            if pd.isna(value):
                return pd.NA
            target = lookup.get(str(value), str(value))
            # EXCLUDED_GROUP_TOKEN drops the rows entirely: an unmapped group is
            # excluded from training, weighting and prediction alike. A blank
            # target cannot express this because the lookup falls back to the
            # raw value, which would silently keep the group.
            if str(target).strip().lower() == EXCLUDED_GROUP_TOKEN:
                return pd.NA
            return target

        return cleaned.map(_resolve).astype("string")
    return cleaned


def _ordered_groups(series: pd.Series) -> list[str]:
    return list(dict.fromkeys(series.dropna().astype(str).tolist()))


def _filter_mask(series: pd.Series, value: Any) -> pd.Series:
    if value is None or (isinstance(value, str) and not value.strip()):
        return pd.Series(True, index=series.index)
    if pd.api.types.is_numeric_dtype(series):
        try:
            numeric_value = float(value)
            return pd.to_numeric(series, errors="coerce").eq(numeric_value)
        except (TypeError, ValueError):
            pass
    return series.astype("string").str.strip().eq(str(value).strip())


def _model_estimators(config: GroupedValidationConfig) -> tuple[int, int]:
    if config.quick_mode:
        quick = max(1, int(config.quick_n_estimators))
        return min(max(1, int(config.mr_n_estimators)), quick), min(
            max(1, int(config.consensus_n_estimators)),
            quick,
        )
    return max(1, int(config.mr_n_estimators)), max(
        1,
        int(config.consensus_n_estimators),
    )


def _required_columns(config: GroupedValidationConfig) -> list[str]:
    references = _unique(
        [
            config.primary_reference_column,
            *config.evaluation_reference_columns,
            config.development_reference_column,
            config.internal_test_reference_column,
        ]
    )
    return _unique(
        [
            *references,
            *config.component_method_columns,
            *config.marker_columns,
            *config.morphology_columns,
            *config.compartment_columns,
            config.validation_group_column,
            config.batch_normalization_column if config.use_z_normalization else None,
            config.filter_column,
            "cell_id",
            "batch_id",
            "tissue_roi",
            "x_centroid",
            "y_centroid",
            "patient_id",
            "slide_id",
            "timepoint",
        ]
    )


def _validate_configuration(
    frame: pd.DataFrame,
    config: GroupedValidationConfig,
) -> None:
    if not (
        config.run_grouped_mr_ml
        or config.run_grouped_consensus_ml
        or config.run_internal_split
    ):
        raise ValueError("Select at least one strict validation analysis.")
    if not config.component_method_columns:
        raise ValueError("Select at least two component phenotype methods.")
    if len(config.component_method_columns) < 2:
        raise ValueError("Strict consensus validation needs at least two component methods.")
    if not config.marker_columns:
        raise ValueError("Select at least one marker feature.")
    if config.consensus_min_methods < 2:
        raise ValueError("Minimum consensus must be at least two methods.")
    if config.consensus_min_methods > len(config.component_method_columns):
        raise ValueError(
            "Minimum consensus cannot exceed the number of component methods."
        )
    if not 0 <= float(config.refined_confidence_threshold) <= 1:
        raise ValueError("The refined confidence threshold must be between 0 and 1.")
    if config.run_internal_split and (
        not config.development_reference_column
        or not config.internal_test_reference_column
    ):
        raise ValueError(
            "Internal validation requires development and held-out reference columns."
        )

    required = _unique(
        [
            config.primary_reference_column,
            *config.evaluation_reference_columns,
            config.development_reference_column if config.run_internal_split else None,
            config.internal_test_reference_column if config.run_internal_split else None,
            *config.component_method_columns,
            *config.marker_columns,
            *config.morphology_columns,
            *config.compartment_columns,
            config.validation_group_column,
            config.batch_normalization_column if config.use_z_normalization else None,
            config.filter_column,
        ]
    )
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Required columns were not found: {missing}")


def _prepare_working_frame(
    frame: pd.DataFrame,
    config: GroupedValidationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    available = [column for column in _required_columns(config) if column in frame.columns]
    work = frame.loc[:, available].copy()
    work.insert(0, "source_csv_row", np.arange(len(frame), dtype=np.int64))

    audit: dict[str, Any] = {
        "n_input_rows": int(len(work)),
        "filter_column": config.filter_column,
        "filter_value": config.filter_value,
        "drop_invalid_marker_rows": bool(config.drop_invalid_marker_rows),
    }
    if config.filter_column:
        keep = _filter_mask(work[config.filter_column], config.filter_value)
        audit["n_filter_excluded"] = int((~keep).sum())
        work = work.loc[keep].copy()
    else:
        audit["n_filter_excluded"] = 0

    marker_numeric = work.loc[:, config.marker_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite_markers = pd.Series(
        np.isfinite(marker_numeric.to_numpy(dtype=float, copy=False)).all(axis=1),
        index=work.index,
    )
    audit["n_invalid_marker_rows"] = int((~finite_markers).sum())
    if config.drop_invalid_marker_rows:
        work = work.loc[finite_markers].copy()
    work.reset_index(drop=True, inplace=True)

    work["validation_group"] = build_validation_groups(
        work[config.validation_group_column],
        config.group_mode,
        config.group_mapping,
    )
    missing_group = (
        work["validation_group"].isna()
        | work["validation_group"].astype("string").str.strip().eq("")
    )
    audit["n_model_rows"] = int(len(work))
    audit["n_missing_validation_group"] = int(missing_group.sum())
    return work, audit


def _prepare_features(
    work: pd.DataFrame,
    config: GroupedValidationConfig,
) -> tuple[pd.DataFrame, list[str], list[str], int]:
    features, marker_columns, feature_columns = ml_mr_utils.prepare_feature_matrix(
        work,
        marker_suffix=None,
        clustering_markers=config.marker_columns,
        morphology_cols=config.morphology_columns,
        compartment_cols=config.compartment_columns,
        feature_weights=config.feature_weights,
        use_z_normalization=config.use_z_normalization,
        batch_normalization_col=config.batch_normalization_column,
    )
    features = features.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    values = features.to_numpy(dtype=float, copy=False)
    n_nonfinite = int((~np.isfinite(values)).sum())
    if n_nonfinite:
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features, list(marker_columns), list(feature_columns), n_nonfinite


def _uniform_weights(
    classes: Sequence[str],
    methods: Sequence[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        1.0,
        index=sorted(set(str(value) for value in classes)),
        columns=list(methods),
        dtype=float,
    )


def _fit_predict_mr_fold(
    *,
    work: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
    training_reference_column: str,
    methods: list[str],
    n_estimators: int,
    config: GroupedValidationConfig,
    fold: str,
    analysis: str,
    enforce_group_separation: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    if not len(train_indices) or not len(prediction_indices):
        raise ValueError(f"{analysis}/{fold}: empty training or prediction partition.")
    overlap = np.intersect1d(train_indices, prediction_indices)
    if len(overlap):
        raise AssertionError(
            f"{analysis}/{fold}: {len(overlap)} rows occur in training and prediction."
        )

    training_groups = sorted(
        work.loc[train_indices, "validation_group"].dropna().astype(str).unique()
    )
    prediction_groups = sorted(
        work.loc[prediction_indices, "validation_group"].dropna().astype(str).unique()
    )
    group_overlap = sorted(set(training_groups) & set(prediction_groups))
    if enforce_group_separation and group_overlap:
        raise AssertionError(
            f"{analysis}/{fold}: group leakage detected: {group_overlap}"
        )

    y_train = _clean_labels(
        work.loc[train_indices, training_reference_column]
    ).astype(str)
    classes = sorted(y_train.unique())
    if len(classes) < 2:
        raise ValueError(f"{analysis}/{fold}: fewer than two training phenotypes.")

    weight_input = work.loc[
        train_indices,
        [training_reference_column, *methods],
    ].copy()
    weights = ml_mr_utils.calculate_f1_scores(
        weight_input,
        training_reference_column,
        methods,
        excluded_phenotypes=config.excluded_phenotypes,
    )
    if weights is None or weights.empty:
        weights = _uniform_weights(classes, methods)
    weights = (
        weights.reindex(index=classes, columns=methods, fill_value=0.0)
        .fillna(0.0)
        .astype(float)
    )

    encoder = LabelEncoder().fit(classes)
    training_frame = features.loc[train_indices, feature_columns].copy()
    label_column = "__strict_manual_reference"
    training_frame[label_column] = y_train.to_numpy()
    model = ml_mr_utils._fit_ml_mr_model(
        training_frame,
        feature_columns,
        label_column,
        encoder,
        n_estimators,
        config.random_state,
        config.mr_ml_method,
    )
    predictions, confidence = ml_mr_utils.predict_weighted_ensemble(
        features.loc[prediction_indices, feature_columns],
        model,
        work.loc[prediction_indices, methods],
        weights,
        encoder,
        excluded_phenotypes=config.excluded_phenotypes,
    )

    weights_long = (
        weights.rename_axis("phenotype")
        .reset_index()
        .melt(
            id_vars="phenotype",
            var_name="component_method",
            value_name="f1_weight",
        )
    )
    weights_long.insert(0, "analysis", analysis)
    weights_long.insert(1, "fold", str(fold))

    audit = {
        "analysis": analysis,
        "fold": str(fold),
        "model": "MR-ML",
        "n_train": int(len(train_indices)),
        "n_predicted": int(len(prediction_indices)),
        "training_reference_column": training_reference_column,
        "training_groups": training_groups,
        "prediction_groups": prediction_groups,
        "train_prediction_group_overlap": group_overlap,
        "training_classes": classes,
        "n_training_classes": int(len(classes)),
        "n_estimators": int(n_estimators),
        "ml_method": config.mr_ml_method,
    }
    return (
        np.asarray(predictions, dtype=object),
        np.asarray(confidence, dtype=float),
        weights_long,
        audit,
    )


def _fit_predict_consensus_fold(
    *,
    work: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    prediction_indices: np.ndarray,
    held_out_group: str,
    methods: list[str],
    n_estimators: int,
    config: GroupedValidationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    heldout = work["validation_group"].eq(held_out_group)
    mapped_group = (
        work["validation_group"].notna()
        & work["validation_group"].astype("string").str.strip().ne("")
    )
    train_mask = (
        work["is_consensus_recomputed"].astype(bool) & mapped_group & ~heldout
    )
    train_indices = work.index[train_mask].to_numpy()
    if len(train_indices) < int(config.consensus_min_training_cells):
        raise ValueError(
            f"Consensus-ML/{held_out_group}: only {len(train_indices):,} "
            "direct-consensus training cells remain outside the held-out group."
        )

    training_groups = sorted(
        work.loc[train_indices, "validation_group"].dropna().astype(str).unique()
    )
    if held_out_group in training_groups:
        raise AssertionError(
            f"Consensus-ML/{held_out_group}: held-out group entered model fitting."
        )

    y_train = _clean_labels(
        work.loc[train_indices, "consensus_phenotype_recomputed"]
    ).astype(str)
    classes = sorted(y_train.unique())
    if len(classes) < 2:
        raise ValueError(
            f"Consensus-ML/{held_out_group}: fewer than two training phenotypes."
        )

    encoder = LabelEncoder().fit(classes)
    scaler = StandardScaler()
    x_train = features.loc[train_indices, feature_columns]
    x_train_scaled = scaler.fit_transform(x_train)
    model = ml_consensus._create_classifier(
        config.consensus_ml_method,
        n_estimators,
        config.random_state,
    )
    model.fit(x_train_scaled, encoder.transform(y_train))

    direct = (
        work.loc[prediction_indices, "is_consensus_recomputed"]
        .astype(bool)
        .to_numpy()
    )
    raw = np.empty(len(prediction_indices), dtype=object)
    confidence = np.full(len(prediction_indices), np.nan, dtype=float)

    if direct.any():
        direct_indices = prediction_indices[direct]
        raw[direct] = _clean_labels(
            work.loc[direct_indices, "consensus_phenotype_recomputed"]
        ).astype(str)
        confidence[direct] = (
            work.loc[direct_indices, "consensus_count_recomputed"]
            .to_numpy(dtype=float)
            / float(len(methods))
        )

    non_direct = ~direct
    if non_direct.any():
        non_direct_indices = prediction_indices[non_direct]
        x_test = features.loc[non_direct_indices, feature_columns]
        x_test_scaled = scaler.transform(x_test)
        encoded = model.predict(x_test_scaled)
        raw[non_direct] = encoder.inverse_transform(encoded)
        confidence[non_direct] = model.predict_proba(x_test_scaled).max(axis=1)

    counts = y_train.value_counts().sort_index().rename("n_training").reset_index()
    counts.columns = ["phenotype", "n_training"]
    counts.insert(0, "held_out_group", str(held_out_group))

    audit = {
        "analysis": "strict_grouped_consensus_ml",
        "fold": str(held_out_group),
        "model": "Consensus-ML",
        "n_train": int(len(train_indices)),
        "n_predicted": int(len(prediction_indices)),
        "n_direct_assignments": int(direct.sum()),
        "n_ml_assignments": int(non_direct.sum()),
        "training_groups": training_groups,
        "prediction_groups": [str(held_out_group)],
        "train_prediction_group_overlap": [],
        "training_classes": classes,
        "n_training_classes": int(len(classes)),
        "n_estimators": int(n_estimators),
        "ml_method": config.consensus_ml_method,
    }
    return raw, confidence, audit, counts


def _refined_predictions(
    raw: np.ndarray,
    confidence: np.ndarray,
    config: GroupedValidationConfig,
) -> np.ndarray:
    refined = np.asarray(raw, dtype=object).copy()
    refined[
        np.asarray(confidence, dtype=float)
        < float(config.refined_confidence_threshold)
    ] = str(config.refined_low_confidence_label)
    return refined


def _evaluation_references(config: GroupedValidationConfig) -> list[str]:
    return _unique(
        [
            config.primary_reference_column,
            *config.evaluation_reference_columns,
        ]
    )


def _manual_reference_union(
    work: pd.DataFrame,
    references: Sequence[str],
) -> pd.Series:
    mask = pd.Series(False, index=work.index)
    for column in references:
        mask |= _labeled_mask(work[column])
    return mask


def _output_columns(
    work: pd.DataFrame,
    config: GroupedValidationConfig,
    references: Sequence[str],
) -> list[str]:
    preferred = [
        "source_csv_row",
        "cell_id",
        "batch_id",
        "tissue_roi",
        "x_centroid",
        "y_centroid",
        "patient_id",
        "slide_id",
        "timepoint",
        config.validation_group_column,
        "validation_group",
        *references,
        *config.component_method_columns,
        "consensus_phenotype_recomputed",
        "is_consensus_recomputed",
        "consensus_count_recomputed",
        "consensus_route",
    ]
    return [column for column in _unique(preferred) if column in work.columns]


def _run_internal_split(
    *,
    work: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    config: GroupedValidationConfig,
    n_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    development = str(config.development_reference_column)
    heldout = str(config.internal_test_reference_column)
    train_mask = _valid_reference_mask(
        work[development],
        config.excluded_phenotypes,
    )
    test_mask = _valid_reference_mask(
        work[heldout],
        config.excluded_phenotypes,
    )
    overlap = int((train_mask & test_mask).sum())
    if overlap:
        raise ValueError(
            f"The internal reference columns overlap on {overlap:,} rows; "
            "development and held-out cells must be row-disjoint."
        )

    train_indices = work.index[train_mask].to_numpy()
    test_indices = work.index[test_mask].to_numpy()
    raw, confidence, weights, audit = _fit_predict_mr_fold(
        work=work,
        features=features,
        feature_columns=feature_columns,
        train_indices=train_indices,
        prediction_indices=test_indices,
        training_reference_column=development,
        methods=list(config.component_method_columns),
        n_estimators=n_estimators,
        config=config,
        fold=heldout,
        analysis="internal_split_reference",
        enforce_group_separation=False,
    )
    columns = _output_columns(
        work,
        config,
        _unique([development, heldout]),
    )
    result = work.loc[test_indices, columns].copy()
    result.insert(0, "analysis", "internal_split_reference")
    result.insert(1, "fold", heldout)
    result["prediction_mr_ml_internal"] = raw
    result["mr_ml_confidence_internal"] = confidence
    result["prediction_mr_ml_internal_refined"] = _refined_predictions(
        raw,
        confidence,
        config,
    )
    audit["test_reference_column"] = heldout
    audit["row_overlap"] = overlap
    return result.reset_index(drop=True), weights, audit


def run_grouped_validation(
    frame: pd.DataFrame,
    config: GroupedValidationConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> GroupedValidationResult:
    """Retrain and evaluate MR-ML and Consensus-ML without group leakage.

    The returned out-of-fold table contains only rows carrying at least one
    selected manual reference. No input columns are changed and no files are
    written.
    """

    _validate_configuration(frame, config)
    _emit(progress_callback, "Preparing eligible rows and validation groups")
    work, feature_audit = _prepare_working_frame(frame, config)

    references = _evaluation_references(config)
    reference_union = _manual_reference_union(work, references)
    mapped_group = (
        work["validation_group"].notna()
        & work["validation_group"].astype("string").str.strip().ne("")
    )
    mapped_reference_union = reference_union & mapped_group
    groups = _ordered_groups(work.loc[mapped_reference_union, "validation_group"])
    if (config.run_grouped_mr_ml or config.run_grouped_consensus_ml) and len(groups) < 2:
        raise ValueError(
            "Strict grouped validation needs at least two mapped validation groups."
        )

    primary_mask = _valid_reference_mask(
        work[config.primary_reference_column],
        config.excluded_phenotypes,
    ) & mapped_group
    primary_groups = _ordered_groups(
        work.loc[
            primary_mask & work["validation_group"].notna(),
            "validation_group",
        ]
    )
    if config.run_grouped_mr_ml and len(primary_groups) < 2:
        raise ValueError(
            "Grouped MR-ML needs evaluable primary-reference cells in at least "
            "two validation groups."
        )

    feature_audit.update(
        {
            "primary_reference_column": config.primary_reference_column,
            "evaluation_reference_columns": references,
            "n_evaluable_primary_reference": int(primary_mask.sum()),
            "n_selected_reference_rows": int(reference_union.sum()),
            "n_mapped_reference_rows": int(mapped_reference_union.sum()),
            "n_reference_rows_without_group": int(
                (reference_union & ~mapped_reference_union).sum()
            ),
            "validation_groups": groups,
            "n_validation_groups": int(len(groups)),
        }
    )

    _emit(progress_callback, "Building the shared marker-feature matrix")
    features, marker_columns, feature_columns, n_nonfinite = _prepare_features(
        work,
        config,
    )
    feature_audit.update(
        {
            "marker_columns": marker_columns,
            "feature_columns": feature_columns,
            "n_markers": int(len(marker_columns)),
            "n_features": int(len(feature_columns)),
            "n_nonfinite_feature_values_replaced": int(n_nonfinite),
        }
    )

    _emit(progress_callback, "Recomputing direct consensus from component methods")
    votes = ml_consensus.identify_consensus(
        work.loc[:, config.component_method_columns],
        list(config.component_method_columns),
        int(config.consensus_min_methods),
        list(config.excluded_phenotypes),
    )
    work["consensus_phenotype_recomputed"] = votes[
        "consensus_phenotype"
    ].to_numpy()
    work["is_consensus_recomputed"] = votes["is_consensus"].astype(bool).to_numpy()
    work["consensus_count_recomputed"] = (
        pd.to_numeric(votes["consensus_count"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    work["consensus_route"] = np.where(
        work["is_consensus_recomputed"],
        "Direct consensus",
        "Non-consensus",
    )
    feature_audit["n_direct_consensus"] = int(
        work["is_consensus_recomputed"].sum()
    )
    feature_audit["direct_consensus_fraction"] = float(
        work["is_consensus_recomputed"].mean()
    )

    mr_estimators, consensus_estimators = _model_estimators(config)
    output_columns = _output_columns(work, config, references)
    fold_frames: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    weight_tables: list[pd.DataFrame] = []
    consensus_count_tables: list[pd.DataFrame] = []

    if config.run_grouped_mr_ml or config.run_grouped_consensus_ml:
        for fold_number, group in enumerate(groups, start=1):
            _emit(
                progress_callback,
                f"Fold {fold_number}/{len(groups)}: holding out {group}",
            )
            heldout = work["validation_group"].eq(group)
            prediction_mask = mapped_reference_union & heldout
            prediction_indices = work.index[prediction_mask].to_numpy()
            if not len(prediction_indices):
                continue

            fold_frame = work.loc[prediction_indices, output_columns].copy()
            fold_frame.insert(0, "analysis", "strict_leave_one_group_out")
            fold_frame.insert(1, "fold", str(group))

            if config.run_grouped_mr_ml:
                train_mask = primary_mask & ~heldout
                train_indices = work.index[train_mask].to_numpy()
                raw, confidence, weights, audit = _fit_predict_mr_fold(
                    work=work,
                    features=features,
                    feature_columns=feature_columns,
                    train_indices=train_indices,
                    prediction_indices=prediction_indices,
                    training_reference_column=config.primary_reference_column,
                    methods=list(config.component_method_columns),
                    n_estimators=mr_estimators,
                    config=config,
                    fold=str(group),
                    analysis="strict_grouped_mr_ml",
                    enforce_group_separation=True,
                )
                fold_frame["prediction_mr_ml_strict"] = raw
                fold_frame["mr_ml_confidence_strict"] = confidence
                fold_frame["prediction_mr_ml_strict_refined"] = (
                    _refined_predictions(raw, confidence, config)
                )
                weight_tables.append(weights)
                fold_audits.append(audit)

            if config.run_grouped_consensus_ml:
                raw, confidence, audit, counts = _fit_predict_consensus_fold(
                    work=work,
                    features=features,
                    feature_columns=feature_columns,
                    prediction_indices=prediction_indices,
                    held_out_group=str(group),
                    methods=list(config.component_method_columns),
                    n_estimators=consensus_estimators,
                    config=config,
                )
                fold_frame["prediction_consensus_ml_strict"] = raw
                fold_frame["consensus_ml_confidence_strict"] = confidence
                fold_frame["prediction_consensus_ml_strict_refined"] = (
                    _refined_predictions(raw, confidence, config)
                )
                fold_audits.append(audit)
                consensus_count_tables.append(counts)

            fold_frames.append(fold_frame.reset_index(drop=True))

    oof_predictions = (
        pd.concat(fold_frames, ignore_index=True)
        if fold_frames
        else pd.DataFrame()
    )
    if not oof_predictions.empty:
        if oof_predictions["source_csv_row"].duplicated().any():
            raise AssertionError(
                "A manual-reference row received predictions in more than one fold."
            )
        expected = int(mapped_reference_union.sum())
        if len(oof_predictions) != expected:
            raise AssertionError(
                f"Out-of-fold output has {len(oof_predictions):,} rows; "
                f"{expected:,} mapped manual-reference rows were expected."
            )

    internal_predictions = pd.DataFrame()
    if config.run_internal_split:
        _emit(
            progress_callback,
            "Running the row-disjoint development/reference internal check",
        )
        internal_predictions, weights, audit = _run_internal_split(
            work=work,
            features=features,
            feature_columns=feature_columns,
            config=config,
            n_estimators=mr_estimators,
        )
        weight_tables.append(weights)
        fold_audits.append(audit)

    feature_audit.update(
        {
            "mr_n_estimators_used": int(mr_estimators),
            "consensus_n_estimators_used": int(consensus_estimators),
            "quick_mode": bool(config.quick_mode),
            "verified_group_separation": bool(
                config.run_grouped_mr_ml or config.run_grouped_consensus_ml
            ),
        }
    )
    _emit(progress_callback, "Strict grouped validation complete")

    return GroupedValidationResult(
        oof_predictions=oof_predictions,
        internal_predictions=internal_predictions,
        fold_audit=pd.DataFrame(fold_audits),
        f1_weights=(
            pd.concat(weight_tables, ignore_index=True)
            if weight_tables
            else pd.DataFrame()
        ),
        consensus_training_counts=(
            pd.concat(consensus_count_tables, ignore_index=True)
            if consensus_count_tables
            else pd.DataFrame()
        ),
        feature_audit=feature_audit,
        configuration=asdict(config),
    )


def run_grouped_learning_curve(
    frame: pd.DataFrame,
    config: GroupedValidationConfig,
    *,
    target_cells_per_class: Sequence[int],
    repeats: int,
    n_estimators: int,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fit balanced ML-MR learning curves with complete groups held out.

    Every point is built from new models. For each repeat and training-size
    target, the manual-reference training cells are sampled only from groups
    outside the current fold; the complete hidden group is then predicted.
    """
    _validate_configuration(frame, config)
    work, _audit = _prepare_working_frame(frame, config)
    features, _markers, feature_columns, _nonfinite = _prepare_features(work, config)
    mapped = (
        work["validation_group"].notna()
        & work["validation_group"].astype("string").str.strip().ne("")
    )
    primary = _valid_reference_mask(
        work[config.primary_reference_column],
        config.excluded_phenotypes,
    ) & mapped
    groups = _ordered_groups(work.loc[primary, "validation_group"])
    if len(groups) < 2:
        raise ValueError(
            "The learning curve needs evaluable primary-reference cells in at "
            "least two mapped validation groups."
        )
    labels = sorted(
        work.loc[primary, config.primary_reference_column].astype(str).unique()
    )
    targets = sorted({max(1, int(value)) for value in target_cells_per_class})
    if not targets:
        raise ValueError("Select at least one training-cell target for the learning curve.")
    rows: list[dict[str, Any]] = []
    for repeat_number in range(max(1, int(repeats))):
        seed = int(config.random_state) + repeat_number
        rng = np.random.default_rng(seed)
        repeat_config = replace(config, random_state=seed)
        for target in targets:
            pooled_truth: list[str] = []
            pooled_prediction: list[str] = []
            fold_training_sizes: list[int] = []
            for group in groups:
                heldout = work["validation_group"].eq(group)
                candidate_indices = work.index[primary & ~heldout].to_numpy()
                prediction_indices = work.index[primary & heldout].to_numpy()
                candidate_labels = work.loc[
                    candidate_indices, config.primary_reference_column
                ].astype(str)
                selected: list[int] = []
                for phenotype in labels:
                    pool = candidate_indices[
                        candidate_labels.to_numpy() == phenotype
                    ]
                    count = min(int(target), len(pool))
                    if count:
                        selected.extend(
                            rng.choice(pool, size=count, replace=False).tolist()
                        )
                training_indices = np.asarray(sorted(selected), dtype=np.int64)
                if not len(training_indices) or not len(prediction_indices):
                    continue
                prediction, _confidence, _weights, _fold_audit = _fit_predict_mr_fold(
                    work=work,
                    features=features,
                    feature_columns=feature_columns,
                    train_indices=training_indices,
                    prediction_indices=prediction_indices,
                    training_reference_column=config.primary_reference_column,
                    methods=list(config.component_method_columns),
                    n_estimators=max(1, int(n_estimators)),
                    config=repeat_config,
                    fold=str(group),
                    analysis="grouped_annotation_learning_curve",
                    enforce_group_separation=True,
                )
                truth = work.loc[
                    prediction_indices, config.primary_reference_column
                ].astype(str)
                score = f1_score(
                    truth,
                    prediction,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
                rows.append(
                    {
                        "repeat": repeat_number + 1,
                        "seed": seed,
                        "target_cells_per_class": target,
                        "fold": str(group),
                        "n_training_cells": int(len(training_indices)),
                        "n_test_cells": int(len(prediction_indices)),
                        "macro_f1": float(score),
                    }
                )
                fold_training_sizes.append(int(len(training_indices)))
                pooled_truth.extend(truth.tolist())
                pooled_prediction.extend(np.asarray(prediction).astype(str).tolist())
            if pooled_truth:
                rows.append(
                    {
                        "repeat": repeat_number + 1,
                        "seed": seed,
                        "target_cells_per_class": target,
                        "fold": "Pooled",
                        "n_training_cells": int(np.mean(fold_training_sizes)),
                        "n_test_cells": int(len(pooled_truth)),
                        "macro_f1": float(
                            f1_score(
                                pooled_truth,
                                pooled_prediction,
                                labels=labels,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                    }
                )
        _emit(
            progress_callback,
            f"Learning-curve repeat {repeat_number + 1}/{max(1, int(repeats))} complete",
        )
    return pd.DataFrame(rows)
