"""ml_mr_utils.py

Utility functions for Machine Learning based on Manual Reference (ML-MR) phenotyping.

This module implements a sophisticated machine learning approach to cell phenotyping that:
1. Uses manually annotated cells as the training reference
2. Applies weighted ensemble based on method performance (F1-scores) from comparison analysis
3. Trains a Random Forest classifier on marker expression data
4. Combines ML predictions with weighted method votes to generate final phenotype assignments
5. Creates a separate refined output that relabels low-support assignments without altering the primary result

The main entry point is the `run_ml_mr_analysis` function, which orchestrates the
entire workflow based on configuration provided from the notebook.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

__all__ = [
    "calculate_f1_scores",
    "load_performance_weights",
    "apply_z_normalization",
    "prepare_morphology_features",
    "prepare_compartment_features",
    "prepare_feature_matrix",
    "prepare_training_data",
    "train_ml_mr_model",
    "predict_weighted_ensemble",
    "generate_performance_heatmaps",
    "generate_expression_heatmap",
    "generate_confidence_stratified_heatmaps",
    "evaluate_ml_mr_performance",
    "run_ml_mr_analysis",
]


_INVALID_REFERENCE_LABELS = {"", "unclassified", "unknown", "nan", "none"}

# Sentinel written in place of a declined call so the cell still counts against
# recall. Matches the Tab 9 validation view, which keeps abstentions in scope.
UNASSIGNED_PREDICTION_LABEL = "__unassigned__"


def _normalized_excluded_phenotypes(
    excluded_phenotypes: list[str] | None,
) -> set[str]:
    """Return non-empty phenotype exclusions normalized for matching."""
    return {
        str(label).strip().lower()
        for label in (excluded_phenotypes or [])
        if str(label).strip()
    }


def _clean_excluded_phenotypes(
    excluded_phenotypes: list[str] | None,
) -> list[str]:
    """Deduplicate exclusions case-insensitively while preserving display text."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for label in excluded_phenotypes or []:
        display = str(label).strip()
        normalized = display.lower()
        if not display or normalized in seen:
            continue
        cleaned.append(display)
        seen.add(normalized)
    return cleaned


def _valid_reference_mask(
    series: pd.Series,
    excluded_phenotypes: list[str] | None = None,
) -> pd.Series:
    """Return rows containing usable, non-excluded manual-reference labels."""
    cleaned = series.astype("string").str.strip()
    invalid_labels = _INVALID_REFERENCE_LABELS | _normalized_excluded_phenotypes(
        excluded_phenotypes
    )
    return series.notna() & ~cleaned.str.lower().isin(invalid_labels)


def _looks_like_reference_column(column: str) -> bool:
    """Identify columns that must never be used as ensemble method votes."""
    name = str(column).strip().lower()
    reviewer_reference = name in {
        "phenotype_bd",
        "phenotype_bd_x",
        "phenotype_bd_y",
        "phenotype_ap",
        "phenotype_ap_x",
        "phenotype_ap_y",
    }
    named_manual_reference = name in {
        "phenotype_manual",
        "manual_phenotype",
        "ground_truth",
        "phenotype_ground_truth",
    } or name.startswith("phenotype_manual_ref")
    return (
        name.startswith("phenotype_ref")
        or reviewer_reference
        or named_manual_reference
    )


def _looks_like_annotation_metadata(column: str) -> bool:
    name = str(column).strip().lower()
    return name == "phenotype_manual_status" or name.endswith(
        ("_match_distance", "_match_status")
    )


def _validation_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_train: int,
    test_indices: np.ndarray,
    validation_mode: str,
    validation_column: str,
    ml_method: str,
) -> dict:
    """Build the common validation payload from final phenotype labels."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "n_train": int(n_train),
        "n_test": int(len(y_true)),
        "test_indices": np.asarray(test_indices),
        "validation_mode": validation_mode,
        "validation_column": validation_column,
        "ml_method": ml_method,
    }


def apply_z_normalization(
    df: pd.DataFrame,
    marker_cols: list[str],
    batch_col: str = 'batch_id'
) -> pd.DataFrame:
    """
    Apply Z-score normalization to marker columns grouped by batch.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    marker_cols : list[str]
        List of marker column names to normalize
    batch_col : str
        Column name to group by for normalization
        
    Returns
    -------
    pd.DataFrame
        Dataframe with normalized marker columns
    """
    print(f"Applying Z-normalization to {len(marker_cols)} markers grouped by '{batch_col}'...")
    
    if batch_col not in df.columns:
        print(f"WARNING: Batch column '{batch_col}' not found. Skipping normalization.")
        return df
    
    result_df = df.copy()
    
    # Check which markers exist in the dataframe
    existing_markers = [col for col in marker_cols if col in df.columns]
    missing_markers = [col for col in marker_cols if col not in df.columns]
    
    if missing_markers:
        print(f"WARNING: {len(missing_markers)} markers not found in dataframe: {missing_markers[:5]}...")
    
    if not existing_markers:
        print("No markers found for normalization.")
        return result_df
    
    print(f"Normalizing {len(existing_markers)} existing markers...")
    
    # Apply Z-normalization grouped by batch
    for marker in existing_markers:
        # Group by batch and apply z-score normalization
        result_df[marker] = result_df.groupby(batch_col)[marker].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    
    print("Z-normalization completed.")
    return result_df


def calculate_f1_scores(
    df: pd.DataFrame,
    ground_truth_col: str,
    phenotyping_methods: list[str],
    excluded_phenotypes: list[str] | None = None,
    score_on: str = "assigned",
) -> pd.DataFrame:
    """
    Calculate F1 scores for each phenotyping method against ground truth.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing ground truth and method predictions
    ground_truth_col : str
        Name of the ground truth column
    phenotyping_methods : list[str]
        List of phenotyping method column names
    excluded_phenotypes : list[str], optional
        Labels ignored as manual targets and automated-method predictions.

    score_on : {"assigned", "all_reference"}
        ``"assigned"`` (default) scores each method only on the cells it
        actually called. ``"all_reference"`` scores every method on all
        reference cells, counting a declined call as a recall miss.

    Notes
    -----
    The default is deliberate. These numbers are *vote weights*, and a weight
    is applied only on the cells where that method casts a vote, so the
    quantity that matters is "when this method calls class c, how often is it
    right?" - which is conditional on it having voted. Scoring a method on
    cells it declined would mix in coverage, which the ensemble already
    handles by receiving no vote from that method on those cells.

    This is therefore *not* the same quantity as the Tab 9 method-comparison
    F1, which scores every method over identical rows. Both are correct for
    their own purpose; they should not be read as interchangeable. Coverage is
    logged here so a low-coverage, high-weight method is visible.

    Returns
    -------
    pd.DataFrame
        Pivot table with cell types as index, methods as columns, F1 scores as values
    """
    print("Calculating F1 scores for method weighting...")

    # Filter valid ground truth data
    if ground_truth_col not in df.columns:
        print(f"Ground-truth column '{ground_truth_col}' was not found.")
        return pd.DataFrame()

    valid_gt_df = df.loc[
        _valid_reference_mask(df[ground_truth_col], excluded_phenotypes)
    ].copy()

    if len(valid_gt_df) == 0:
        print("No valid ground truth data found!")
        return pd.DataFrame()

    f1_data = []
    y_true = valid_gt_df[ground_truth_col].astype("string").str.strip()

    for method in phenotyping_methods:
        if method not in df.columns:
            print(f"Method '{method}' not found in dataframe, skipping...")
            continue

        assigned = _valid_reference_mask(valid_gt_df[method], excluded_phenotypes)
        n_assigned = int(assigned.sum())
        if n_assigned == 0:
            print(f"No valid predictions found for {method}")
            continue

        if score_on == "all_reference":
            method_true = y_true
            y_pred = (
                valid_gt_df[method]
                .astype("string")
                .str.strip()
                .where(assigned, UNASSIGNED_PREDICTION_LABEL)
            )
        else:
            method_true = y_true.loc[assigned]
            y_pred = valid_gt_df.loc[assigned, method].astype("string").str.strip()

        # Real phenotypes only. The abstention sentinel must never become a
        # class of its own, or every method that ever declines a call is
        # penalized by a flat 1/(K+1) dilution unrelated to how often it did so.
        labels = sorted(
            label
            for label in (set(method_true) | set(y_pred))
            if label != UNASSIGNED_PREDICTION_LABEL
        )

        # Calculate per-class F1 scores
        _, _, f1_scores, _ = precision_recall_fscore_support(
            method_true, y_pred, labels=labels, average=None, zero_division=0
        )

        # Ensure f1_scores is an array
        f1_scores = np.asarray(f1_scores)

        # Store results
        for label, f1_value in zip(labels, f1_scores):
            f1_data.append({
                'Method': method,
                'Cell_Type': label,
                'F1_Score': f1_value
            })

        coverage = n_assigned / len(valid_gt_df)
        print(
            f"  {method}: {len(valid_gt_df)} reference cells scored "
            f"({n_assigned} assigned, coverage {coverage:.1%})"
        )

    if not f1_data:
        print("No F1 scores calculated!")
        return pd.DataFrame()
    
    # Create pivot table
    f1_df = pd.DataFrame(f1_data)
    f1_pivot = f1_df.pivot(index='Cell_Type', columns='Method', values='F1_Score').fillna(0)
    
    print(f"F1 scores calculated for {len(f1_pivot)} cell types and {len(f1_pivot.columns)} methods")
    return f1_pivot


def prepare_morphology_features(
    df: pd.DataFrame,
    morphology_cols: list[str],
    use_z_normalization: bool = False,
    batch_col: str = 'batch_id'
) -> pd.DataFrame:
    """
    Transform and optionally Z-normalize morphology features.
    
    Transformations:
    - log1p for 'area', 'major_axis_length', 'minor_axis_length' (size/lengths)
    - sin(theta) and cos(theta) for 'orientation' (replaces original)
    - No change for others (e.g., eccentricity, solidity, extent)
    
    Then apply per-batch Z-normalization if enabled.
    
    Returns: DataFrame with transformed/normalized morphology columns (orientation replaced by sin/cos).
    """
    if not morphology_cols:
        return pd.DataFrame()
    
    # Ensure batch_col is available for normalization if needed
    cols_to_extract = list(set(morphology_cols + ([batch_col] if use_z_normalization else [])))
    missing_cols = [c for c in cols_to_extract if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing morphology columns: {missing_cols}")
        morphology_cols = [c for c in morphology_cols if c in df.columns]
        cols_to_extract = [c for c in cols_to_extract if c in df.columns]
        if not morphology_cols:
            return pd.DataFrame()

    morph_df = df[cols_to_extract].copy()
    
    # Apply transformations
    size_length_cols = [col for col in morphology_cols if col in ['area', 'major_axis_length', 'minor_axis_length'] and col in morph_df.columns]
    for col in size_length_cols:
        morph_df[col] = np.log1p(morph_df[col])
    
    if 'orientation' in morph_df.columns:
        morph_df['orientation_sin'] = np.sin(morph_df['orientation'])
        morph_df['orientation_cos'] = np.cos(morph_df['orientation'])
        morph_df.drop('orientation', axis=1, inplace=True)
    
    # Z-normalization if enabled
    if use_z_normalization and batch_col in morph_df.columns:
        feature_cols = [col for col in morph_df.columns if col != batch_col]
        morph_df = apply_z_normalization(morph_df, feature_cols, batch_col)

    # If normalization was skipped, drop batch_col if it was added
    if batch_col in morph_df.columns:
        morph_df.drop(batch_col, axis=1, inplace=True)

    return morph_df


def prepare_compartment_features(
    df: pd.DataFrame,
    compartment_cols: list[str]
) -> pd.DataFrame:
    """
    Extract binary compartment features (0/1). No normalization applied.
    
    Returns: DataFrame with compartment columns.
    """
    if not compartment_cols:
        return pd.DataFrame()
    
    missing_cols = [c for c in compartment_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing compartment columns: {missing_cols}")
        compartment_cols = [c for c in compartment_cols if c in df.columns]
        if not compartment_cols:
            return pd.DataFrame()
    
    return df[compartment_cols].astype(float).copy()  # Ensure float for weighting later


def prepare_feature_matrix(
    unified_df: pd.DataFrame,
    marker_suffix: str | None = None,
    clustering_markers: list[str] | None = None,
    morphology_cols: list[str] | None = None,
    compartment_cols: list[str] | None = None,
    feature_weights: dict[str, float] | None = None,
    use_z_normalization: bool = False,
    batch_normalization_col: str = "batch_id",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build one feature matrix shared by training, validation, and prediction.

    Creating the matrix once prevents the former scale mismatch where manually
    labelled cells were normalized separately from the full prediction table.
    Batch-wise Z scores remain an optional, label-free preprocessing operation.
    """
    if feature_weights is None:
        feature_weights = {"markers": 1.0, "morphology": 1.0, "compartments": 0.5}

    if clustering_markers is not None:
        marker_cols = [column for column in clustering_markers if column in unified_df.columns]
        missing_markers = [column for column in clustering_markers if column not in unified_df.columns]
        if missing_markers:
            print(f"WARNING: {len(missing_markers)} specified markers not found: {missing_markers[:5]}...")
    else:
        if marker_suffix is None:
            raise ValueError("Either clustering_markers or marker_suffix must be provided")
        marker_cols = sorted(column for column in unified_df.columns if column.endswith(marker_suffix))

    if not marker_cols:
        raise ValueError("No marker columns found")

    marker_input_cols = list(marker_cols)
    if use_z_normalization and batch_normalization_col in unified_df.columns:
        marker_input_cols.append(batch_normalization_col)
    elif use_z_normalization:
        print(
            f"WARNING: Batch column '{batch_normalization_col}' not found; "
            "marker Z-normalization was skipped."
        )

    marker_df = unified_df.loc[:, marker_input_cols].copy()
    for column in marker_cols:
        marker_df[column] = pd.to_numeric(marker_df[column], errors="coerce")
    if use_z_normalization and batch_normalization_col in marker_df.columns:
        marker_df = apply_z_normalization(marker_df, marker_cols, batch_normalization_col)
    if batch_normalization_col in marker_df.columns:
        marker_df = marker_df.drop(columns=[batch_normalization_col])
    marker_df *= float(feature_weights.get("markers", 1.0))

    morph_df = prepare_morphology_features(
        unified_df,
        morphology_cols or [],
        use_z_normalization,
        batch_normalization_col,
    )
    morph_df *= float(feature_weights.get("morphology", 1.0))

    comp_df = prepare_compartment_features(unified_df, compartment_cols or [])
    comp_df *= float(feature_weights.get("compartments", 1.0))

    feature_parts = [marker_df]
    if not morph_df.empty:
        feature_parts.append(morph_df)
    if not comp_df.empty:
        feature_parts.append(comp_df)
    feature_df = pd.concat(feature_parts, axis=1)
    if feature_df.columns.duplicated().any():
        duplicates = feature_df.columns[feature_df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate model feature columns are not allowed: {duplicates}")

    all_feature_cols = list(feature_df.columns)
    print(
        f"Prepared shared feature matrix: {len(marker_cols)} markers, "
        f"{len(morph_df.columns)} morphology, {len(comp_df.columns)} compartments"
    )
    return feature_df, marker_cols, all_feature_cols


def load_performance_weights(performance_csv_path: Path, phenotyping_methods: list[str]) -> pd.DataFrame | None:
    """Load F1-score based weights from the comparison analysis results.

    The expected structure (from spacemap.comparison_utils) is a *per-class* performance CSV
    containing rows with columns: Method, Cell_Type, Metric, F1_Score (among others).
    We extract the rows where Metric == 'F1_Score' and pivot to produce a matrix
    (index=Cell_Type, columns=Method, values=F1_Score).

    This function is made robust to several alternative legacy formats:
    1. No 'Metric' column (already filtered so every row corresponds to F1) -> pivot directly.
    2. Lower‑case 'metric' column.
    3. Column naming variations (e.g. 'F1' instead of 'F1_Score'). In that case we
       try to alias to 'F1_Score'.
    4. Missing methods -> we add them with weight 0 so downstream logic still works.
    """
    print(f"Loading performance weights from: {performance_csv_path}")

    if not performance_csv_path.exists():
        print(f"Performance file not found: {performance_csv_path}")
        return None

    try:
        df = pd.read_csv(performance_csv_path)
        if df.empty:
            print("Performance file is empty.")
            return None

        # Normalize column names (strip spaces)
        df.columns = [c.strip() for c in df.columns]

        # Harmonize F1 column naming
        if 'F1_Score' not in df.columns:
            if 'F1' in df.columns:
                df.rename(columns={'F1': 'F1_Score'}, inplace=True)
            elif 'f1' in df.columns:
                df.rename(columns={'f1': 'F1_Score'}, inplace=True)

        required_base_cols = {'Method', 'Cell_Type', 'F1_Score'}
        if not required_base_cols.issubset(df.columns):
            print(f"Performance file missing required columns: {required_base_cols - set(df.columns)}")
            return None

        # Filter rows for F1 metric only if a metric column exists
        metric_col = None
        if 'Metric' in df.columns:
            metric_col = 'Metric'
        elif 'metric' in df.columns:
            metric_col = 'metric'

        if metric_col:
            f1_df = df[df[metric_col] == 'F1_Score'].copy()
            if f1_df.empty:
                print("No rows with Metric == 'F1_Score'; using all rows as fallback.")
                f1_df = df.copy()
        else:
            # No metric column -> assume already per-class F1 summary
            f1_df = df.copy()

        # Validate again that F1_Score column exists after filtering
        if 'F1_Score' not in f1_df.columns:
            print("Could not locate 'F1_Score' column after processing.")
            return None

        # Pivot to weights matrix
        try:
            weights_df = f1_df.pivot(index='Cell_Type', columns='Method', values='F1_Score')
        except KeyError as e:
            print(f"Pivot failed, missing column: {e}")
            return None

        weights_df = weights_df.fillna(0.0)

        # Ensure all requested methods included
        for method in phenotyping_methods:
            if method not in weights_df.columns:
                print(f"Adding missing method column with zero weights: {method}")
                weights_df[method] = 0.0

        # Sort columns consistently
        weights_df = weights_df[sorted(weights_df.columns)]

        print("Successfully created weight lookup table.")
        print(f"  Cell types (n={len(weights_df)}): {list(weights_df.index)[:10]}{'...' if len(weights_df)>10 else ''}")
        print(f"  Methods: {list(weights_df.columns)}")
        return weights_df

    except Exception as e:
        print(f"ERROR loading performance weights: {e}")
        return None


def prepare_training_data(
    unified_df: pd.DataFrame,
    ground_truth_column: str,
    marker_suffix: str | None = None,
    phenotyping_methods: list[str] | None = None,
    clustering_markers: list[str] | None = None,
    morphology_cols: list[str] | None = None,
    compartment_cols: list[str] | None = None,
    feature_weights: dict[str, float] | None = None,
    use_z_normalization: bool = False,
    batch_normalization_col: str = 'batch_id',
) -> tuple[pd.DataFrame, list[str], list[str], LabelEncoder] | tuple[None, None, None, None]:
    """
    Prepare full feature set training data: markers + transformed morphology + compartments.
    
    Parameters
    ----------
    unified_df : pd.DataFrame
        The unified phenotyping dataframe
    ground_truth_column : str
        Name of the ground truth column (e.g., 'phenotype_ref1')
    marker_suffix : str, optional
        Suffix for marker columns (e.g., '_corr'). Used only if clustering_markers is None.
    phenotyping_methods : list[str], optional
        List of phenotyping method column names
    clustering_markers : list[str], optional
        Specific list of marker columns to use. If provided, overrides marker_suffix.
    morphology_cols : list[str], optional
        List of morphology feature column names
    compartment_cols : list[str], optional
        List of compartment binary column names
    feature_weights : dict[str, float], optional
        Weights for feature groups: 'markers', 'morphology', 'compartments'
    use_z_normalization : bool
        Whether to apply Z-normalization to markers and morphology
    batch_normalization_col : str
        Column to group by for Z-normalization
        
    Returns
    -------
    tuple
        (training_df, marker_columns, all_feature_columns, label_encoder) or (None, None, None, None) if failed
    """
    print("Preparing comprehensive training data from manually annotated cells...")
    
    if ground_truth_column not in unified_df.columns:
        print(f"ERROR: Ground-truth column '{ground_truth_column}' was not found")
        return None, None, None, None

    training_mask = _valid_reference_mask(unified_df[ground_truth_column])
    if not training_mask.any():
        print(f"ERROR: No training data found in column '{ground_truth_column}'")
        return None, None, None, None

    try:
        feature_df, marker_cols, all_feature_cols = prepare_feature_matrix(
            unified_df,
            marker_suffix=marker_suffix,
            clustering_markers=clustering_markers,
            morphology_cols=morphology_cols,
            compartment_cols=compartment_cols,
            feature_weights=feature_weights,
            use_z_normalization=use_z_normalization,
            batch_normalization_col=batch_normalization_col,
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: Could not prepare model features: {exc}")
        return None, None, None, None

    training_df = feature_df.loc[training_mask].copy()
    training_df[ground_truth_column] = (
        unified_df.loc[training_mask, ground_truth_column].astype("string").str.strip()
    )
    morph_count = len([column for column in all_feature_cols if column not in marker_cols and column not in (compartment_cols or [])])
    comp_count = len([column for column in (compartment_cols or []) if column in all_feature_cols])
    
    print(f"Found {len(training_df)} training cells with:")
    print(f"  - {len(marker_cols)} markers")
    print(f"  - {morph_count} morphology features")
    print(f"  - {comp_count} compartment features")
    print(f"  - Total features: {len(all_feature_cols)}")
    print(f"Ground truth classes: {training_df[ground_truth_column].value_counts().to_dict()}")
    
    # Only the manual training reference defines the classes the supervised
    # model is allowed to learn. Method outputs must not introduce classes.
    all_possible_labels = set(training_df[ground_truth_column].unique())

    label_encoder = LabelEncoder()
    label_encoder.fit(list(all_possible_labels))
    print(f"Label encoder fitted with {len(all_possible_labels)} unique labels")
    
    return training_df, marker_cols, all_feature_cols, label_encoder

# =============================================================================
# CLASSIFIER FACTORY
# =============================================================================

# Available ML methods for display in the UI
ML_METHOD_OPTIONS = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm": "SVM (RBF Kernel)",
}


def _balanced_fit_kwargs(model, y_encoded):
    """Give GradientBoosting the class balancing RF and SVM get for free.

    RandomForest and SVC accept class_weight="balanced"; GradientBoosting has
    no such parameter and would otherwise train unweighted, quietly biasing it
    toward abundant phenotypes only when that classifier is selected.
    """
    from sklearn.utils.class_weight import compute_sample_weight

    if isinstance(model, GradientBoostingClassifier):
        return {"sample_weight": compute_sample_weight("balanced", y_encoded)}
    return {}


def _create_classifier(ml_method: str, n_estimators: int, random_state: int):
    """Create a classifier instance based on the selected ML method."""
    if ml_method == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=min(n_estimators, 500),
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            subsample=0.8,
        )
    elif ml_method == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
                C=1.0,
                gamma="scale",
            ),
        )
    else:  # "random_forest" (default)
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )


def train_ml_mr_model(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    ground_truth_column: str,
    label_encoder: LabelEncoder,
    test_size: float = 0.3,
    n_estimators: int = 150,
    random_state: int = 42,
    ml_method: str = "random_forest",
) -> tuple:
    """
    Train the ML-MR model using the selected classifier.
    
    Parameters
    ----------
    training_df : pd.DataFrame
        Training data with ground truth labels
    feature_cols : list[str]
        List of all feature column names (markers + morphology + compartments)
    ground_truth_column : str
        Name of the ground truth column
    label_encoder : LabelEncoder
        Fitted label encoder
    test_size : float
        Fraction of data to use for testing
    n_estimators : int
        Number of trees in random forest / gradient boosting
    random_state : int
        Random state for reproducibility
    ml_method : str
        Classifier type: 'random_forest', 'gradient_boosting', or 'svm'.
        
    Returns
    -------
    tuple
        (trained_model, validation_results)
    """
    method_display = ML_METHOD_OPTIONS.get(ml_method, ml_method)
    print(f"Training ML-MR model ({method_display})...")
    
    # Prepare features and labels
    X = training_df[feature_cols].fillna(0)
    y = training_df[ground_truth_column]
    y_encoded = label_encoder.transform(y)

    # Split using explicit indices so we can reference the same rows later
    idx_all = training_df.index.to_numpy()
    # Use a Series to index encoded labels by original index for safe selection
    # Build a Series with explicit list to satisfy type checkers
    y_encoded_s = pd.Series(list(np.asarray(y_encoded)), index=training_df.index)

    idx_train, idx_test = train_test_split(
        idx_all, test_size=test_size, random_state=random_state, stratify=y_encoded_s.loc[idx_all]
    )

    # Build train/test partitions using the saved indices
    X_train = X.loc[idx_train]
    X_test = X.loc[idx_test]
    y_train = y_encoded_s.loc[idx_train].to_numpy()
    y_test = y_encoded_s.loc[idx_test].to_numpy()
    
    # Train model
    model = _create_classifier(ml_method, n_estimators, random_state)
    
    model.fit(X_train, y_train, **_balanced_fit_kwargs(model, y_train))
    
    # Validate on test set
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    validation_results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy_score(y_test_labels, y_pred_labels),
        'macro_f1': f1_score(y_test_labels, y_pred_labels, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0),
        'y_true': y_test_labels,
        'y_pred': y_pred_labels,
        'classification_report': classification_report(y_test_labels, y_pred_labels, zero_division=0),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_indices': idx_train,
        # Keep test indices to align other method predictions for visualization
        'test_indices': idx_test,
        'validation_mode': 'internal_stratified_cell_split',
        'validation_column': ground_truth_column,
        'ml_method': method_display,
    }
    
    print(f"Model training complete. Validation accuracy: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return model, validation_results


def _fit_ml_mr_model(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    ground_truth_column: str,
    label_encoder: LabelEncoder,
    n_estimators: int,
    random_state: int,
    ml_method: str,
):
    """Fit the production classifier on every permitted training-reference row."""
    X_train = training_df.loc[:, feature_cols].fillna(0)
    y_train = label_encoder.transform(training_df[ground_truth_column])
    model = _create_classifier(ml_method, n_estimators, random_state)
    model.fit(X_train, y_train, **_balanced_fit_kwargs(model, y_train))
    return model


def predict_weighted_ensemble(
    X_data: pd.DataFrame,
    base_model: RandomForestClassifier,
    other_methods_predictions: pd.DataFrame,
    weights_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    excluded_phenotypes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict phenotypes using weighted ensemble of ML model and method votes.
    
    Parameters
    ----------
    X_data : pd.DataFrame
        Feature data for prediction
    base_model : RandomForestClassifier
        Trained ML model
    other_methods_predictions : pd.DataFrame
        Predictions from other phenotyping methods
    weights_df : pd.DataFrame
        Weight lookup table (cell_type x method)
    label_encoder : LabelEncoder
        Fitted label encoder
    excluded_phenotypes : list[str], optional
        Automated-method labels that cannot contribute an ensemble vote.
        
    Returns
    -------
    tuple
        (final_predictions, confidence_scores)
    """
    print("Generating weighted ensemble predictions...")
    
    # Get base model probabilities
    base_probabilities = base_model.predict_proba(X_data)
    
    # Initialize final scores matrix
    n_samples = X_data.shape[0]
    n_total_classes = len(label_encoder.classes_)
    final_scores = np.zeros((n_samples, n_total_classes))
    
    # Add base model probabilities
    model_class_indices = base_model.classes_
    final_scores[:, model_class_indices] = base_probabilities
    
    # Create class to index mapping
    class_to_idx = {cls: i for i, cls in enumerate(label_encoder.classes_)}
    excluded_normalized = _normalized_excluded_phenotypes(excluded_phenotypes)
    
    # Add weighted votes from other methods.
    # Vectorized per method: the previous per-cell Python loop performed one
    # .loc lookup per cell per method, which dominated runtime on large slides.
    # np.add.at accumulates identically, including repeated (row, class) pairs.
    for method in tqdm(weights_df.columns, desc="Applying weighted votes"):
        if method not in other_methods_predictions.columns:
            continue

        predictions = other_methods_predictions[method].astype("string").str.strip()
        eligible = predictions.notna() & ~predictions.str.lower().isin(excluded_normalized)
        if not eligible.any():
            continue

        # Map each prediction to its column in final_scores and its weight.
        class_idx = predictions.map(class_to_idx)
        weight_lookup = weights_df[method]
        weight = predictions.map(weight_lookup)
        usable = (eligible & class_idx.notna() & weight.notna()).to_numpy()
        if not usable.any():
            continue

        rows = np.flatnonzero(usable)
        cols = class_idx.to_numpy()[usable].astype(int)
        np.add.at(final_scores, (rows, cols), weight.to_numpy(dtype=float)[usable])
    
    # Normalize the accumulated support. This preserves the ensemble decision
    # while giving the score a transparent interpretation: the fraction of
    # available model-plus-method support assigned to the winning phenotype.
    support_totals = final_scores.sum(axis=1, keepdims=True)
    normalized_support = np.divide(
        final_scores,
        support_totals,
        out=np.zeros_like(final_scores),
        where=support_totals > 0,
    )
    final_prediction_indices = np.argmax(normalized_support, axis=1)
    final_predictions = label_encoder.inverse_transform(final_prediction_indices)
    prediction_confidence = np.max(normalized_support, axis=1)
    
    print(f"Generated predictions for {len(final_predictions)} cells")
    
    return final_predictions, prediction_confidence


def generate_performance_heatmaps(
    y_true: np.ndarray,
    predictions_dict: dict,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Generate performance comparison heatmaps."""
    print(f"Generating performance heatmaps{title_suffix}...")
    
    # Align lengths and filter out mismatched arrays
    y_true_arr = np.asarray(y_true)
    usable_preds: dict[str, np.ndarray] = {}
    for name, arr in predictions_dict.items():
        pred_arr = np.asarray(arr)
        if len(pred_arr) != len(y_true_arr):
            print(f"  Skipping '{name}' due to length mismatch: y_true={len(y_true_arr)} vs pred={len(pred_arr)}")
            continue
        usable_preds[name] = pred_arr
    if not usable_preds:
        print("No prediction arrays with matching length. Skipping heatmaps.")
        return

    # Get unique labels
    all_labels = sorted(list(set(y_true_arr) | set().union(*[set(pred) for pred in usable_preds.values()])))
    
    metrics_data = []
    for method_name, y_pred in usable_preds.items():
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_arr, y_pred, labels=all_labels, average=None, zero_division=0
        )
        precision = np.asarray(precision)
        recall = np.asarray(recall)
        f1 = np.asarray(f1)
        for label_name, p, r, f in zip(all_labels, precision, recall, f1):
            metrics_data.append({
                'Method': method_name,
                'Cell_Type': label_name,
                'Precision': float(p),
                'Recall': float(r),
                'F1-Score': float(f)
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create heatmaps
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'Performance Comparison{title_suffix}', fontsize=16)
    
    try:
        precision_pivot = metrics_df.pivot(index='Cell_Type', columns='Method', values='Precision')
        sns.heatmap(precision_pivot, annot=True, fmt='.3f', cmap='Reds', ax=ax1, vmin=0, vmax=1)
        ax1.set_title('Precision by Cell Type')
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Cell Type')
        
        recall_pivot = metrics_df.pivot(index='Cell_Type', columns='Method', values='Recall')
        sns.heatmap(recall_pivot, annot=True, fmt='.3f', cmap='Greens', ax=ax2, vmin=0, vmax=1)
        ax2.set_title('Recall by Cell Type')
        ax2.set_xlabel('Method')
        ax2.set_ylabel('')
        
        f1_pivot = metrics_df.pivot(index='Cell_Type', columns='Method', values='F1-Score')
        sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='Blues', ax=ax3, vmin=0, vmax=1)
        ax3.set_title('F1-Score by Cell Type')
        ax3.set_xlabel('Method')
        ax3.set_ylabel('')
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance heatmaps saved to: {output_path}")
        
    except Exception as e:
        print(f"Could not generate performance heatmaps: {e}")


def generate_expression_heatmap(
    df: pd.DataFrame,
    marker_cols: list[str],
    phenotype_col: str,
    title: str,
    output_path: Path,
) -> None:
    """Generate marker expression heatmap by phenotype."""
    print(f"Generating expression heatmap: {title}")
    
    try:
        mean_expression = df.groupby(phenotype_col)[marker_cols].mean()
        
        # Z-score normalization
        scaler = StandardScaler()
        z_scores = scaler.fit_transform(mean_expression)
        z_scores_df = pd.DataFrame(z_scores, index=mean_expression.index, columns=mean_expression.columns)
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(z_scores_df.T, cmap='bwr', center=0, annot=True, fmt='.2f', linewidths=0.5)
        plt.title(title, fontsize=14)
        plt.xlabel('Phenotype')
        plt.ylabel('Markers')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Expression heatmap saved to: {output_path}")
        
    except Exception as e:
        print(f"Could not generate expression heatmap: {e}")


def generate_confidence_stratified_heatmaps(
    df: pd.DataFrame,
    marker_cols: list[str],
    phenotype_col: str,
    confidence_col: str,
    output_dir: Path,
) -> None:
    """Generate confidence-stratified expression heatmaps."""
    print("Generating confidence-stratified heatmaps...")
    
    if confidence_col not in df.columns:
        print(f"Warning: '{confidence_col}' column not found. Skipping.")
        return
    
    # Define confidence bins
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    df_copy = df.copy()
    df_copy['confidence_bin'] = pd.cut(df_copy[confidence_col], bins=bins, labels=labels, include_lowest=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for bin_label in labels:
        print(f"  Generating heatmap for confidence bin: {bin_label}")
        subset_df = df_copy[df_copy['confidence_bin'] == bin_label]
        
        if len(subset_df) < 2 or subset_df[phenotype_col].nunique() < 2:
            print(f"    Skipping bin '{bin_label}', not enough data or classes.")
            continue
        
        try:
            mean_expression = subset_df.groupby(phenotype_col)[marker_cols].mean()
            
            scaler = StandardScaler()
            z_scores = scaler.fit_transform(mean_expression)
            z_scores_df = pd.DataFrame(z_scores, index=mean_expression.index, columns=mean_expression.columns)
            
            plt.figure(figsize=(16, 10))
            sns.heatmap(z_scores_df.T, cmap='bwr', center=0, annot=True, fmt='.2f', linewidths=0.5)
            plt.title(f'ML-MR Mean Marker Expression (Z-Score) for Confidence Range: {bin_label}', fontsize=14)
            plt.xlabel('Phenotype')
            plt.ylabel('Markers')
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            output_path = output_dir / f"expression_heatmap_confidence_{bin_label.replace('%', '')}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Heatmap for bin '{bin_label}' saved to: {output_path}")
            
        except Exception as e:
            print(f"  Could not generate heatmap for bin '{bin_label}': {e}")


def evaluate_ml_mr_performance(
    validation_results: dict,
    output_dir: Path,
) -> None:
    """Evaluate and save ML-MR model performance."""
    print("Evaluating ML-MR model performance...")
    
    # Save detailed validation report
    report_path = output_dir / "ml_mr_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write("ML-MR Model Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Validation mode: {validation_results.get('validation_mode', 'unknown')}\n")
        f.write(f"Validation reference: {validation_results.get('validation_column', 'unknown')}\n")
        f.write(f"Training samples: {validation_results['n_train']}\n")
        f.write(f"Test samples: {validation_results['n_test']}\n")
        f.write(f"Validation accuracy: {validation_results['accuracy']:.3f}\n")
        f.write(f"Balanced accuracy: {validation_results.get('balanced_accuracy', float('nan')):.3f}\n")
        f.write(f"Macro F1: {validation_results.get('macro_f1', float('nan')):.3f}\n")
        f.write(f"Weighted F1: {validation_results.get('weighted_f1', float('nan')):.3f}\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 30 + "\n")
        f.write(validation_results['classification_report'])
    
    print(f"Validation report saved to: {report_path}")
    
    # Generate confusion matrix
    try:
        cm = confusion_matrix(validation_results['y_true'], validation_results['y_pred'])
        labels = sorted(list(set(validation_results['y_true']) | set(validation_results['y_pred'])))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Final ML-MR Ensemble Confusion Matrix (Validation Set)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = output_dir / "ml_mr_confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
        
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")


def run_ml_mr_analysis(
    unified_csv_path: Path,
    performance_csv_path: Path | None,
    output_dir: Path,
    ground_truth_column: str,
    phenotyping_methods: list[str],
    marker_suffix: str = '_corr',
    clustering_markers: list[str] | None = None,
    morphology_cols: list[str] | None = None,
    compartment_cols: list[str] | None = None,
    feature_weights: dict[str, float] | None = None,
    use_z_normalization: bool = False,
    batch_normalization_col: str = 'batch_id',
    test_size: float = 0.3,
    n_estimators: int = 150,
    random_state: int = 42,
    # New optional parameters for auto-generation & robustness
    dataset_id_column: str = 'dataset_name',
    min_consensus_methods: int = 2,
    auto_generate_performance: bool = True,
    figure_dpi: int = 150,
    ml_method: str = "random_forest",
    output_phenotype_col: str = "phenotype_ml_mr",
    confidence_col_name: str = "ml_mr_confidence",
    refined_output_phenotype_col: str = "phenotype_ml_mr_refined",
    refined_confidence_threshold: float = 0.75,
    refined_low_confidence_label: str = "Others",
    progress_callback=None,
    preloaded_df: pd.DataFrame = None,
    validation_ground_truth_column: str | None = None,
    analysis_filter_column: str | None = None,
    analysis_filter_value: object = None,
    drop_invalid_marker_rows: bool = True,
    excluded_phenotypes: list[str] | None = None,
    write_backup_csv: bool = False,
    update_source_csv: bool = False,
    results_callback=None,
) -> bool:
    """
    Run the complete ML-MR analysis workflow.
    
    Parameters
    ----------
    unified_csv_path : Path
        Path to the unified phenotyping CSV file
    performance_csv_path : Path | None
        Optional path to a per-class performance CSV from comparison analysis
    output_dir : Path
        Directory to save outputs
    ground_truth_column : str
        Name of the ground truth column (e.g., 'phenotype_ref1')
    phenotyping_methods : list[str]
        List of phenotyping method column names
    marker_suffix : str
        Suffix for marker columns
    test_size : float
        Fraction of training data to use for validation
    n_estimators : int
        Number of trees in random forest
    random_state : int
        Random state for reproducibility
    output_phenotype_col : str
        Column name for ML-MR phenotype predictions (default: 'phenotype_ml_mr')
    confidence_col_name : str
        Column name for ML-MR confidence scores (default: 'ml_mr_confidence')
    refined_output_phenotype_col : str
        Column containing a conservative copy of the MR-ML result in which
        scores below ``refined_confidence_threshold`` are replaced by
        ``refined_low_confidence_label``.
    refined_confidence_threshold : float
        Support-score threshold in the inclusive range 0 to 1. Scores strictly
        below this value are relabeled in the refined output only.
    refined_low_confidence_label : str
        Label assigned to low-confidence cells in the refined output.
    preloaded_df : pd.DataFrame, optional
        If provided, skip CSV loading and use this DataFrame directly.
    validation_ground_truth_column : str, optional
        Row-disjoint held-out manual-reference column used only for final
        ensemble validation. When omitted, an internal stratified cell split
        is used. A separate column does not by itself imply patient/ROI-level
        independence.
    analysis_filter_column : str, optional
        Restrict training, validation, and prediction to rows whose value in
        this column equals ``analysis_filter_value``. Output values outside the
        selected population are preserved (or left missing for new columns).
    analysis_filter_value : object, optional
        Value matched by string representation when ``analysis_filter_column``
        is supplied (for example, ``1`` for ``tissue == 1``).
    drop_invalid_marker_rows : bool
        Exclude rows containing missing or infinite values in the selected
        marker panel before fitting or prediction. Excluded rows retain prior
        outputs (or remain missing for newly created output columns).
    excluded_phenotypes : list[str], optional
        Phenotype labels excluded case-insensitively from manual-reference
        training/validation targets and automated-method votes. Every eligible
        cell is still predicted as one of the remaining training classes. This
        does not prevent the separate refined output from using the configured
        low-confidence label.
    write_backup_csv : bool
        Write a separate ``*_ml_mr.csv`` result file. Disabled by default.
    update_source_csv : bool
        Update ``unified_csv_path`` in place. Disabled by default so callers
        can review assignments before explicitly saving them.
    results_callback : callable, optional
        Receives the result DataFrame for in-memory review regardless
        of whether either CSV-writing option is enabled.
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    from datetime import datetime as _dt

    def _ts() -> str:
        return _dt.now().strftime("%H:%M:%S")

    def _notify(msg: str) -> None:
        """Print to terminal AND optionally to UI callback."""
        stamped = f"[{_ts()}] {msg}"
        print(stamped, flush=True)
        if progress_callback:
            try:
                progress_callback(stamped)
            except Exception:
                pass

    _notify("=" * 70)
    _notify("STARTING ML-MR (Machine Learning based on Manual Reference) ANALYSIS")
    _notify("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data (skip if caller already provided a DataFrame)
    if preloaded_df is not None:
        source_df = preloaded_df.copy()
        _notify(f"[OK] Using pre-loaded dataset: {len(source_df):,} cells, {len(source_df.columns)} columns")
    else:
        _notify("Step 1/7: Loading data...")
        try:
            source_df = pd.read_csv(unified_csv_path)
            _notify(f"[OK] Loaded dataset: {len(source_df):,} cells, {len(source_df.columns)} columns")
        except Exception as e:
            _notify(f"[ERROR] Loading unified CSV: {e}")
            return False

    unified_df = source_df
    analysis_filter_applied = analysis_filter_column is not None
    if analysis_filter_applied:
        if analysis_filter_column not in source_df.columns:
            _notify(f"[ERROR] Analysis filter column '{analysis_filter_column}' was not found.")
            return False
        target = str(analysis_filter_value).strip()
        analysis_mask = source_df[analysis_filter_column].astype("string").str.strip().eq(target)
        if not analysis_mask.any():
            _notify(
                f"[ERROR] Analysis filter {analysis_filter_column} == {analysis_filter_value!r} "
                "selected no rows."
            )
            return False
        unified_df = source_df.loc[analysis_mask].copy()
        _notify(
            f"[OK] Analysis filter {analysis_filter_column} == {analysis_filter_value!r}: "
            f"{len(unified_df):,}/{len(source_df):,} rows selected"
        )

    invalid_marker_rows_dropped = 0
    if drop_invalid_marker_rows:
        marker_candidates = (
            list(clustering_markers)
            if clustering_markers is not None
            else [column for column in unified_df.columns if column.endswith(marker_suffix)]
        )
        existing_marker_candidates = [
            column for column in marker_candidates if column in unified_df.columns
        ]
        if existing_marker_candidates:
            marker_values = unified_df[existing_marker_candidates].apply(
                pd.to_numeric, errors="coerce"
            )
            valid_marker_mask = pd.Series(
                np.isfinite(marker_values.to_numpy()).all(axis=1),
                index=unified_df.index,
            )
            invalid_marker_rows_dropped = int((~valid_marker_mask).sum())
            if invalid_marker_rows_dropped:
                unified_df = unified_df.loc[valid_marker_mask].copy()
                _notify(
                    f"[OK] Excluded {invalid_marker_rows_dropped:,} rows with missing/"
                    "infinite selected-marker values"
                )
                if unified_df.empty:
                    _notify("[ERROR] No rows remain after selected-marker completeness filtering.")
                    return False

    row_subset_applied = not unified_df.index.equals(source_df.index)

    if ground_truth_column not in unified_df.columns:
        _notify(f"[ERROR] Training reference column '{ground_truth_column}' was not found.")
        return False
    if not phenotyping_methods:
        _notify("[ERROR] Select at least one non-reference phenotyping method to integrate.")
        return False
    if ml_method not in ML_METHOD_OPTIONS:
        _notify(f"[ERROR] Unsupported ML classifier: '{ml_method}'.")
        return False
    missing_methods = [method for method in phenotyping_methods if method not in unified_df.columns]
    if missing_methods:
        _notify(f"[ERROR] Missing phenotyping method columns: {missing_methods}")
        return False
    protected_inputs = {
        ground_truth_column,
        validation_ground_truth_column,
        *phenotyping_methods,
    }
    protected_inputs.discard(None)
    output_columns = {
        output_phenotype_col,
        confidence_col_name,
        refined_output_phenotype_col,
    }
    if any(not isinstance(column, str) or not column.strip() for column in output_columns):
        _notify("[ERROR] Output phenotype, confidence, and refined column names cannot be blank.")
        return False
    if len(output_columns) != 3:
        _notify("[ERROR] Output phenotype, confidence, and refined columns must be unique.")
        return False
    collisions = sorted(output_columns & protected_inputs)
    if collisions:
        _notify(
            "[ERROR] Output columns cannot overwrite training, validation, or automated-method "
            f"input columns: {collisions}"
        )
        return False
    try:
        refined_confidence_threshold = float(refined_confidence_threshold)
    except (TypeError, ValueError):
        _notify("[ERROR] Refined confidence threshold must be numeric and between 0 and 1.")
        return False
    if not np.isfinite(refined_confidence_threshold) or not 0 <= refined_confidence_threshold <= 1:
        _notify("[ERROR] Refined confidence threshold must be between 0 and 1.")
        return False
    refined_low_confidence_label = str(refined_low_confidence_label).strip()
    if not refined_low_confidence_label:
        _notify("[ERROR] Low-confidence phenotype label cannot be blank.")
        return False
    excluded_phenotypes = _clean_excluded_phenotypes(excluded_phenotypes)
    excluded_normalized = _normalized_excluded_phenotypes(excluded_phenotypes)
    if excluded_phenotypes:
        _notify(
            "[OK] Phenotypes excluded from training, validation, and method votes: "
            f"{excluded_phenotypes}"
        )
    forbidden_methods = [
        method for method in phenotyping_methods
        if method in {
            ground_truth_column,
            validation_ground_truth_column,
            output_phenotype_col,
            refined_output_phenotype_col,
        }
        or _looks_like_reference_column(method)
        or _looks_like_annotation_metadata(method)
    ]
    if forbidden_methods:
        _notify(
            "[ERROR] Manual-reference and output columns cannot be ensemble methods: "
            f"{forbidden_methods}"
        )
        return False

    training_mask = _valid_reference_mask(
        unified_df[ground_truth_column], excluded_phenotypes
    )
    if not training_mask.any():
        _notify(f"[ERROR] No usable labels found in '{ground_truth_column}'.")
        return False
    training_reference_counts = (
        unified_df.loc[training_mask, ground_truth_column]
        .astype("string")
        .str.strip()
        .value_counts()
        .sort_index()
    )
    training_class_count = int(len(training_reference_counts))
    if training_class_count < 2:
        _notify(
            f"[ERROR] Training reference '{ground_truth_column}' contains fewer than two phenotype classes."
        )
        return False
    sparse_training_classes = training_reference_counts[training_reference_counts < 5]
    if not sparse_training_classes.empty:
        _notify(
            "[WARN] Fewer than five training cells are available for: "
            f"{sparse_training_classes.to_dict()}. Treat those class estimates as unstable."
        )

    validation_mask = None
    if validation_ground_truth_column:
        if validation_ground_truth_column not in unified_df.columns:
            _notify(
                f"[ERROR] Held-out validation column '{validation_ground_truth_column}' was not found."
            )
            return False
        if validation_ground_truth_column == ground_truth_column:
            _notify("[ERROR] Training and validation reference columns must be different.")
            return False
        validation_mask = _valid_reference_mask(
            unified_df[validation_ground_truth_column], excluded_phenotypes
        )
        if not validation_mask.any():
            _notify(f"[ERROR] No usable labels found in '{validation_ground_truth_column}'.")
            return False
        overlap_count = int((training_mask & validation_mask).sum())
        if overlap_count:
            _notify(
                f"[ERROR] {overlap_count:,} cells have labels in both training and validation "
                "columns. Held-out validation requires disjoint rows."
            )
            return False

    # Calculate final method weights strictly from the training reference.
    _notify("\nStep 1.5/7: Calculating training-only F1 method weights...")
    training_reference_df = unified_df.loc[training_mask].copy()
    weights_df = calculate_f1_scores(
        training_reference_df,
        ground_truth_column,
        phenotyping_methods,
        excluded_phenotypes=excluded_phenotypes,
    )
    if (
        weights_df.empty
        and validation_mask is None
        and performance_csv_path
        and performance_csv_path.is_file()
    ):
        _notify("Training-reference F1 calculation failed; trying the supplied performance table...")
        weights_df = load_performance_weights(performance_csv_path, phenotyping_methods)
    if weights_df is not None and not weights_df.empty and excluded_normalized:
        keep_weight_rows = ~pd.Index(weights_df.index).astype(str).str.strip().str.lower().isin(
            excluded_normalized
        )
        weights_df = weights_df.loc[keep_weight_rows]
    if weights_df is None or weights_df.empty:
        _notify("[WARN] F1 weights unavailable; assigning uniform method weights of 1.0.")
        candidate_labels = set(training_reference_df[ground_truth_column].astype(str).str.strip())
        if not candidate_labels:
            _notify("[ERROR] Could not infer phenotype classes for method weighting.")
            return False
        weights_df = pd.DataFrame(
            1.0,
            index=sorted(candidate_labels),
            columns=phenotyping_methods,
            dtype=float,
        )
    _notify(f"[OK] Method-weight matrix: {weights_df.shape[0]} phenotypes x {weights_df.shape[1]} methods")

    # Build one shared matrix so training, validation, and prediction use the
    # exact same transformed values.
    _notify("\nStep 2/7: Preparing shared feature matrix...")
    try:
        all_X, marker_cols, all_feature_cols = prepare_feature_matrix(
            unified_df,
            marker_suffix=marker_suffix,
            clustering_markers=clustering_markers,
            morphology_cols=morphology_cols,
            compartment_cols=compartment_cols,
            feature_weights=feature_weights,
            use_z_normalization=use_z_normalization,
            batch_normalization_col=batch_normalization_col,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _notify(f"[ERROR] Could not prepare model features: {exc}")
        return False
    nan_count = int(all_X.isna().sum().sum())
    if nan_count:
        _notify(f"[INFO] Filling {nan_count:,} missing feature values with 0")
        all_X = all_X.fillna(0)

    training_df = all_X.loc[training_mask].copy()
    training_df[ground_truth_column] = (
        unified_df.loc[training_mask, ground_truth_column].astype("string").str.strip()
    )
    all_possible_labels = set(training_df[ground_truth_column].astype(str))
    label_encoder = LabelEncoder().fit(sorted(all_possible_labels))
    marker_cols_list = list(marker_cols)
    method_display = ML_METHOD_OPTIONS.get(ml_method, ml_method)
    _notify(
        f"[OK] Training reference: {len(training_df):,} cells; "
        f"features: {len(all_feature_cols)}; classes: {len(label_encoder.classes_)}"
    )

    _notify(f"\nStep 3/7: Training ML-MR model ({method_display})...")
    if validation_mask is not None:
        model = _fit_ml_mr_model(
            training_df,
            all_feature_cols,
            ground_truth_column,
            label_encoder,
            n_estimators,
            random_state,
            ml_method,
        )
        validation_results = None
        _notify(
            f"[OK] Model fitted on all '{ground_truth_column}' cells; "
            f"'{validation_ground_truth_column}' remains untouched for validation."
        )
    else:
        try:
            validation_model, split_results = train_ml_mr_model(
                training_df,
                all_feature_cols,
                ground_truth_column,
                label_encoder,
                test_size,
                n_estimators,
                random_state,
                ml_method=ml_method,
            )
        except ValueError as exc:
            _notify(f"[ERROR] Internal stratified split failed: {exc}")
            return False
        split_train_indices = split_results["train_indices"]
        split_test_indices = split_results["test_indices"]
        validation_weights = calculate_f1_scores(
            unified_df.loc[split_train_indices],
            ground_truth_column,
            phenotyping_methods,
            excluded_phenotypes=excluded_phenotypes,
        )
        if validation_weights.empty:
            validation_weights = pd.DataFrame(
                1.0,
                index=label_encoder.classes_,
                columns=phenotyping_methods,
                dtype=float,
            )
        split_predictions, _ = predict_weighted_ensemble(
            all_X.loc[split_test_indices, all_feature_cols],
            validation_model,
            unified_df.loc[split_test_indices, phenotyping_methods],
            validation_weights,
            label_encoder,
            excluded_phenotypes=excluded_phenotypes,
        )
        validation_results = _validation_results(
            unified_df.loc[split_test_indices, ground_truth_column].astype(str).str.strip().to_numpy(),
            split_predictions,
            n_train=len(split_train_indices),
            test_indices=split_test_indices,
            validation_mode="internal_stratified_cell_split_exploratory",
            validation_column=ground_truth_column,
            ml_method=method_display,
        )
        # Production predictions use every training-reference cell; held-out
        # rows are used only for the reported internal diagnostic above.
        model = _fit_ml_mr_model(
            training_df,
            all_feature_cols,
            ground_truth_column,
            label_encoder,
            n_estimators,
            random_state,
            ml_method,
        )
        _notify(
            f"[OK] Exploratory internal final-ensemble accuracy: "
            f"{validation_results['accuracy']:.4f}; production model refitted on all references."
        )

    # Apply the fitted classifier plus training-derived method weights to all cells.
    _notify(f"\nStep 4/7: Applying final MR-ML ensemble to all {len(unified_df):,} cells...")
    final_predictions, final_confidence = predict_weighted_ensemble(
        all_X.loc[:, all_feature_cols],
        model,
        unified_df.loc[:, phenotyping_methods],
        weights_df,
        label_encoder,
        excluded_phenotypes=excluded_phenotypes,
    )
    unified_df[output_phenotype_col] = final_predictions
    unified_df[confidence_col_name] = final_confidence
    unified_df[refined_output_phenotype_col] = unified_df[output_phenotype_col].copy()
    low_confidence_mask = (
        unified_df[confidence_col_name] < refined_confidence_threshold
    )
    n_low_confidence = int(low_confidence_mask.sum())
    unified_df.loc[low_confidence_mask, refined_output_phenotype_col] = (
        refined_low_confidence_label
    )
    _notify(
        f"[OK] Refined output '{refined_output_phenotype_col}': {n_low_confidence:,} "
        f"cells relabeled as '{refined_low_confidence_label}' "
        f"(support < {refined_confidence_threshold:g})"
    )

    if validation_mask is not None:
        validation_indices = unified_df.index[validation_mask].to_numpy()
        validation_results = _validation_results(
            unified_df.loc[validation_indices, validation_ground_truth_column]
            .astype(str)
            .str.strip()
            .to_numpy(),
            unified_df.loc[validation_indices, output_phenotype_col].astype(str).to_numpy(),
            n_train=len(training_df),
            test_indices=validation_indices,
            validation_mode="held_out_reference_column",
            validation_column=str(validation_ground_truth_column),
            ml_method=method_display,
        )
        train_classes = set(training_df[ground_truth_column].astype(str))
        validation_classes = set(validation_results["y_true"])
        unseen = sorted(validation_classes - train_classes)
        if unseen:
            _notify(f"[WARN] Validation contains classes absent from training: {unseen}")
        _notify(
            f"[OK] Held-out final-ensemble validation: accuracy={validation_results['accuracy']:.4f}, "
            f"balanced_accuracy={validation_results['balanced_accuracy']:.4f}, "
            f"macro_F1={validation_results['macro_f1']:.4f}"
        )

    assert validation_results is not None
    
    _notify(f"[OK] ML-MR predictions generated for all {len(unified_df):,} cells")
    _notify(f"  Phenotype distribution: {unified_df[output_phenotype_col].value_counts().to_dict()}")
    
    # Prepare results for review. CSV persistence is explicitly opt-in.
    _notify("\nStep 5/7: Preparing results for review...")
    updated_csv_path = unified_csv_path.parent / f"{unified_csv_path.stem}_ml_mr.csv"
    if row_subset_applied:
        saved_df = source_df.copy()
        if output_phenotype_col not in saved_df.columns:
            saved_df[output_phenotype_col] = pd.NA
        if confidence_col_name not in saved_df.columns:
            saved_df[confidence_col_name] = np.nan
        if refined_output_phenotype_col not in saved_df.columns:
            saved_df[refined_output_phenotype_col] = pd.NA
        saved_df.loc[unified_df.index, output_phenotype_col] = unified_df[output_phenotype_col]
        saved_df.loc[unified_df.index, confidence_col_name] = unified_df[confidence_col_name]
        saved_df.loc[unified_df.index, refined_output_phenotype_col] = unified_df[
            refined_output_phenotype_col
        ]
    else:
        saved_df = unified_df

    if results_callback is not None:
        try:
            results_callback(saved_df)
        except Exception as exc:
            _notify(f"[ERROR] Could not return ML-MR results for review: {exc}")
            return False

    csv_result_df = None
    if write_backup_csv or update_source_csv:
        csv_result_df = saved_df.drop(
            columns=["__tab7_internal_cell_id"], errors="ignore"
        )
    if write_backup_csv and csv_result_df is not None:
        csv_result_df.to_csv(updated_csv_path, index=False)
        _notify(f"[OK] Optional result CSV saved to: {updated_csv_path}")
    if update_source_csv and csv_result_df is not None:
        csv_result_df.to_csv(unified_csv_path, index=False)
        _notify(f"[OK] Unified CSV updated: {unified_csv_path}")
    if not write_backup_csv and not update_source_csv:
        _notify(
            "[OK] Automatic CSV writing is off; assignments are available for review only."
        )
    
    # Evaluate performance
    _notify("\nStep 6/7: Evaluating performance...")
    evaluate_ml_mr_performance(validation_results, output_dir)
    _notify("[OK] Performance evaluation complete")
    
    # Generate visualizations
    _notify("\nStep 7/7: Generating visualizations...")
    
    # Performance heatmaps on validation set using the exact test indices
    test_indices = validation_results.get('test_indices', None)
    validation_predictions = {
        'ML-MR': validation_results['y_pred'],
        'Ground Truth': validation_results['y_true'],
    }
    if test_indices is not None:
        for m in phenotyping_methods:
            if m in unified_df.columns:
                raw_list = pd.Series(unified_df.loc[test_indices, m]).tolist()
                preds = np.array([
                    'unclassified' if pd.isna(x) else str(x) for x in raw_list
                ], dtype=object)
                validation_predictions[m] = preds
    else:
        _notify("[WARN] test indices not available; skipping method overlays in performance heatmaps.")

    generate_performance_heatmaps(
        validation_results['y_true'],
        {k: v for k, v in validation_predictions.items() if k != 'Ground Truth'},
        output_dir / "ml_mr_performance_heatmaps.png",
        " (Validation Set)"
    )
    _notify("[OK] Performance heatmaps saved")

    # Expression heatmap
    generate_expression_heatmap(
        unified_df, marker_cols_list, output_phenotype_col,
        'ML-MR: Mean Marker Expression (Z-Score) by Phenotype',
        output_dir / "ml_mr_expression_heatmap.png"
    )

    # Confidence-stratified heatmaps
    confidence_dir = output_dir / "confidence_stratified"
    generate_confidence_stratified_heatmaps(
        unified_df, marker_cols_list, output_phenotype_col, confidence_col_name, confidence_dir
    )
    _notify("[OK] All visualizations saved")

    # Save summary statistics
    summary_path = output_dir / "ml_mr_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ML-MR Analysis Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Input rows: {len(source_df)}\n")
        f.write(f"Analysis rows processed: {len(unified_df)}\n")
        if analysis_filter_applied:
            f.write(
                f"Analysis filter: {analysis_filter_column} == {analysis_filter_value!r}\n"
            )
        f.write(f"Rows excluded for invalid selected markers: {invalid_marker_rows_dropped}\n")
        f.write(f"Training cells: {len(training_df)}\n")
        f.write(f"Training reference: {ground_truth_column}\n")
        f.write(f"Excluded phenotypes: {excluded_phenotypes}\n")
        f.write(f"Training-reference class counts: {training_reference_counts.to_dict()}\n")
        f.write(f"Validation mode: {validation_results.get('validation_mode', 'unknown')}\n")
        f.write(f"Validation reference: {validation_results.get('validation_column', 'unknown')}\n")
        f.write(f"Validation accuracy: {validation_results['accuracy']:.3f}\n")
        f.write(f"Balanced accuracy: {validation_results.get('balanced_accuracy', float('nan')):.3f}\n")
        f.write(f"Macro F1: {validation_results.get('macro_f1', float('nan')):.3f}\n")
        f.write(f"Weighted F1: {validation_results.get('weighted_f1', float('nan')):.3f}\n")
        f.write(f"Features used: {len(marker_cols_list)} markers\n")
        f.write(f"Methods integrated: {len(phenotyping_methods)}\n\n")
        f.write(f"Refined output column: {refined_output_phenotype_col}\n")
        f.write(f"Refined confidence threshold: {refined_confidence_threshold:g}\n")
        f.write(f"Refined low-confidence label: {refined_low_confidence_label}\n")
        f.write(f"Cells relabeled in refined output: {n_low_confidence}\n\n")
        f.write("ML-MR Phenotype Distribution:\n")
        f.write("-" * 25 + "\n")
        for phenotype, count in unified_df[output_phenotype_col].value_counts().items():
            percentage = (count / len(unified_df)) * 100
            f.write(f"{phenotype}: {count} ({percentage:.1f}%)\n")
        f.write(f"\nMean confidence score: {unified_df[confidence_col_name].mean():.3f}\n")
        f.write(f"Median confidence score: {unified_df[confidence_col_name].median():.3f}\n")
        f.write("\nRefined ML-MR Phenotype Distribution:\n")
        f.write("-" * 33 + "\n")
        for phenotype, count in unified_df[refined_output_phenotype_col].value_counts().items():
            percentage = (count / len(unified_df)) * 100
            f.write(f"{phenotype}: {count} ({percentage:.1f}%)\n")

    _notify("\n" + "=" * 80)
    _notify("ML-MR ANALYSIS COMPLETE")
    _notify("=" * 80)
    _notify(f"Reports and figures saved to: {output_dir}")
    _notify(
        f"Review columns prepared: '{output_phenotype_col}', '{confidence_col_name}', "
        f"'{refined_output_phenotype_col}'"
    )
    if update_source_csv:
        _notify(f"Updated unified file: {unified_csv_path}")
    else:
        _notify("Unified CSV unchanged; use the explicit assignment-save action after review.")

    return True
