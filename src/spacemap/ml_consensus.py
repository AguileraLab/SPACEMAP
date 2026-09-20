"""ml_consensus.py

Machine Learning (Unsupervised) Phenotyping

This module implements both supervised consensus-based and unsupervised ML workflows:

SUPERVISED MODE (original):
1. Identifies consensus cells across multiple phenotyping methods (>= min_consensus_methods agreeing).
2. Uses consensus cells as high-confidence training data.
3. Trains a RandomForest classifier on marker expression features.

UNSUPERVISED MODE (new):
1. Uses corrected markers, morphology features, and compartment binaries.
2. Applies dimensionality reduction (PCA) and clustering (HDBSCAN + GMM).
3. Auto-labels clusters based on marker signatures without supervision.
4. Generates confidence scores and handles unclassified assignments.

Both modes produce per-cell outputs:
- phenotype_ml_consensus: final assigned phenotype
- ml_consensus_score: confidence score (0-1 scale)

Outputs are written under: outputs/phenotyping/ML_consensus

Design goals:
- Deterministic / reproducible via random_state
- Graceful fallbacks and phenotype-agnostic approach
- Separation of logic into composable functions
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from dataclasses import dataclass
from dataclasses import field
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

_INVALID_REFERENCE_LABELS = {"", "unclassified", "unknown", "nan", "none"}


def _valid_reference_mask(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return series.notna() & ~cleaned.str.lower().isin(_INVALID_REFERENCE_LABELS)


def _looks_like_reference_column(column: str) -> bool:
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


__all__ = [
    "ConsensusConfig",
    "apply_z_normalization",
    "identify_consensus",
    "prepare_training_data",
    "prepare_morphology_features",
    "prepare_compartment_features",
    "train_and_validate",
    "train_final_model",
    "predict_non_consensus",
    "build_final_outputs",
    "generate_validation_artifacts",
    "generate_expression_heatmaps",
    "run_ml_consensus_analysis",
]


def apply_z_normalization(
    df: pd.DataFrame,
    feature_cols: List[str],
    batch_col: str = 'batch_id'
) -> pd.DataFrame:
    """
    Apply Z-score normalization to feature columns grouped by batch.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : List[str]
        List of feature column names to normalize
    batch_col : str
        Column name to group by for normalization

    Returns
    -------
    pd.DataFrame
        Dataframe with normalized marker columns
    """
    print(f"Applying Z-normalization to {len(feature_cols)} features grouped by '{batch_col}'...")

    if batch_col not in df.columns:
        print(f"WARNING: Batch column '{batch_col}' not found. Skipping normalization.")
        return df

    result_df = df.copy()

    # Check which markers exist in the dataframe
    existing_markers = [col for col in feature_cols if col in df.columns]
    missing_markers = [col for col in feature_cols if col not in df.columns]

    if missing_markers:
        print(f"WARNING: {len(missing_markers)} features not found in dataframe: {missing_markers[:5]}...")

    if not existing_markers:
        print("No features found for normalization.")
        return result_df

    print(f"Normalizing {len(existing_markers)} existing features...")

    # Apply Z-normalization grouped by batch
    for marker in existing_markers:
        # Group by batch and apply z-score normalization
        result_df[marker] = result_df.groupby(batch_col)[marker].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    print("Z-normalization completed.")
    return result_df


@dataclass
class ConsensusConfig:
    unified_csv: Path
    output_dir: Path
    phenotyping_methods: List[str]
    marker_suffix: str = "_corr"
    clustering_markers: Optional[List[str]] = None  # Specific markers; overrides suffix if provided
    morphology_cols: Optional[List[str]] = None  # Morphology features, e.g., ['area', 'major_axis_length', 'minor_axis_length', 'eccentricity', 'solidity', 'extent', 'orientation']
    compartment_cols: Optional[List[str]] = None  # Binary compartment indicators, e.g., ['is_tumor', 'is_epithelia', 'is_stroma', 'is_endothelia']
    feature_weights: Dict[str, float] = field(default_factory=lambda: {'markers': 1.0, 'morphology': 1.0, 'compartments': 0.5})  # Weights to scale feature groups (e.g., down-weight compartments for soft context)
    use_z_normalization: bool = False  # Apply Z-norm to markers and transformed morphology (per batch)
    batch_normalization_col: str = "batch_id"
    min_consensus_methods: int = 3
    test_size: float = 0.3
    n_estimators: int = 150
    random_state: int = 42
    dataset_id_column: str = "batch_id"
    figure_dpi: int = 300
    min_training_cells: int = 50  # minimal number of consensus cells required
    write_backup_csv: bool = False  # optional parallel result CSV; review-first default is off
    update_source_csv: bool = False  # explicit opt-in; GUI saves only after user review
    excluded_phenotypes: List[str] = field(default_factory=list)  # Phenotypes that cannot form a consensus (e.g. "Unclassified")
    output_phenotype_col: str = "phenotype_ml_consensus"  # Column name for output phenotypes
    confidence_col_name: str = "ml_consensus_score"  # Column name for confidence scores
    consensus_col_name: str = "consensus"  # Binary consensus column (1=consensus, 0=non-consensus)
    refined_confidence_threshold: float = 0.75  # Threshold below which cells get relabeled in refined column
    refined_low_confidence_label: str = "Others"  # Label for low-confidence cells in refined column
    ml_method: str = "random_forest"  # ML classifier: "random_forest", "gradient_boosting", or "svm"
    validation_ground_truth_column: Optional[str] = None  # Optional row-disjoint manual reference for final-output validation
    analysis_filter_column: Optional[str] = None  # Optional row eligibility column (for example, "tissue")
    analysis_filter_value: object = None  # Exact value selected by string representation
    drop_invalid_marker_rows: bool = True  # Exclude rows with missing/infinite selected markers
    rescue_unreachable_classes: bool = True  # Admit (min_consensus-1) plurality cells for phenotypes that never reach consensus, which would otherwise be unpredictable


# =============================================================================
# CONSENSUS IDENTIFICATION
# =============================================================================

def identify_consensus(df: pd.DataFrame, methods: List[str], min_consensus: int, excluded_phenotypes: List[str] = None) -> pd.DataFrame:
    """Identify consensus phenotypes across specified method columns.

    Consensus definition: at least `min_consensus` methods agree on the *same*
    non-null, non-'unclassified' label.

    When ``excluded_phenotypes`` is provided (e.g. ["Others"]), those labels are
    replaced with NaN **before** the mode computation so they never participate
    in the consensus vote.

    Returns copy of df with columns:
      - consensus_phenotype
      - is_consensus (bool)
      - consensus_count (int number of agreeing methods)
    """
    missing = [m for m in methods if m not in df.columns]
    if missing:
        raise ValueError(f"Missing phenotyping method columns: {missing}")

    work = df.copy()

    # Build a clean phenotype matrix (replace invalid / excluded with None)
    invalid_phenos = {"unclassified", "unknown", "", "nan", "none"}
    excluded_normalized = {
        str(label).strip().lower() for label in (excluded_phenotypes or [])
    }

    n_rows = len(work)
    n_methods = len(methods)

    # Extract values as a 2D object array — fast
    vals = np.empty((n_rows, n_methods), dtype=object)
    for i, col in enumerate(methods):
        raw = work[col]
        # Strip accidental whitespace while preserving phenotype capitalization.
        s = raw.astype("string").str.strip()
        normalized = s.str.lower()
        mask = raw.isna() | normalized.isin(invalid_phenos | excluded_normalized)
        clean = s.astype(object)
        clean[mask] = None
        vals[:, i] = clean.to_numpy(dtype=object)

    # ── Vectorized mode computation ──
    # Collect all unique non-None labels (use set, not np.unique which sorts and fails on str/None mix)
    unique_labels = set()
    for i in range(n_methods):
        col_vals = vals[:, i]
        unique_labels.update(v for v in set(col_vals) if v is not None)

    if not unique_labels:
        work["consensus_phenotype"] = np.nan
        work["is_consensus"] = False
        work["consensus_count"] = 0
        return work

    # For each unique label, count how many methods agree → pick the max
    best_label = np.full(n_rows, None, dtype=object)
    best_count = np.zeros(n_rows, dtype=np.int32)

    tied_best = np.zeros(n_rows, dtype=bool)
    for label in sorted(unique_labels, key=str):
        count = np.zeros(n_rows, dtype=np.int32)
        for i in range(n_methods):
            count += (vals[:, i] == label).astype(np.int32)
        better = count > best_count
        tied = (count == best_count) & (count > 0) & ~better
        best_count[better] = count[better]
        best_label[better] = label
        tied_best[better] = False
        tied_best[tied] = True

    # A 2-vs-2 tie is not a valid consensus even when min_consensus == 2.
    is_consensus = (best_count >= min_consensus) & (best_label != None) & ~tied_best  # noqa: E711

    work["consensus_phenotype"] = pd.array(best_label, dtype=object)
    work.loc[~is_consensus, "consensus_phenotype"] = np.nan
    work["is_consensus"] = is_consensus
    work["consensus_count"] = best_count

    # Plurality label regardless of the threshold, blank on a tie. Needed to
    # rescue phenotypes that never reach min_consensus anywhere - without this
    # they are absent from the training labels and become unpredictable.
    plurality = pd.array(best_label, dtype=object)
    work["majority_phenotype"] = plurality
    work.loc[tied_best | (best_count == 0), "majority_phenotype"] = np.nan

    return work


def augment_unreachable_classes(
    df_cons: pd.DataFrame,
    training_mask: np.ndarray,
    min_consensus: int,
    notify=None,
) -> Tuple[np.ndarray, pd.Series, Dict[str, int]]:
    """Admit plurality-labelled cells for phenotypes that never reach consensus.

    A classifier can only ever emit a label it saw in training. Any phenotype
    that never reaches ``min_consensus`` anywhere is therefore unpredictable,
    and every non-consensus cell truly belonging to it is guaranteed wrong -
    a structural zero in per-class recall that no amount of tuning removes.
    Rare and ambiguous phenotypes are exactly the ones this hits.

    For those phenotypes only, cells whose plurality vote reached
    ``min_consensus - 1`` untied methods are added to the training set. Classes
    that already have consensus support are untouched.

    Returns the widened training mask, its labels, and the counts added.
    """
    mask = np.asarray(training_mask, dtype=bool).copy()
    labels = df_cons["consensus_phenotype"].where(mask)

    if "majority_phenotype" not in df_cons.columns or min_consensus <= 1:
        return mask, labels.loc[mask].astype(str), {}

    represented = set(labels.loc[mask].dropna().astype(str).unique())
    plurality = df_cons["majority_phenotype"]
    counts = pd.to_numeric(df_cons["consensus_count"], errors="coerce").fillna(0)

    # Only near-misses qualify, and only for otherwise-absent phenotypes.
    near_miss = (counts >= (min_consensus - 1)) & plurality.notna() & ~mask
    candidates = plurality.where(near_miss)
    missing = [
        str(label)
        for label in candidates.dropna().astype(str).unique()
        if str(label) not in represented
    ]

    added: Dict[str, int] = {}
    for label in missing:
        rows = near_miss & plurality.astype("string").eq(label)
        n_rows = int(rows.sum())
        if n_rows == 0:
            continue
        mask |= rows.to_numpy(dtype=bool)
        labels = labels.mask(rows, label)
        added[label] = n_rows

    if added and notify is not None:
        notify(
            f"[INFO] Rescued {len(added)} phenotype(s) that never reached "
            f"{min_consensus}-method consensus, using {min_consensus - 1}-method "
            f"plurality cells: {added}. Without this they could not be predicted at all."
        )

    return mask, labels.loc[mask].astype(str), added


def prepare_morphology_features(
    df: pd.DataFrame,
    morphology_cols: List[str],
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
        return pd.DataFrame(index=df.index)  # Empty if no cols

    # Ensure batch_col is available for normalization if needed
    cols_to_extract = list(set(morphology_cols + ([batch_col] if use_z_normalization else [])))
    missing_cols = [c for c in cols_to_extract if c not in df.columns]
    if missing_cols:
        # Don't fail, but warn if columns are missing.
        # The function can still proceed with the columns that do exist.
        print(f"Warning: The following requested columns are missing from the DataFrame: {missing_cols}")
        cols_to_extract = [c for c in cols_to_extract if c in df.columns]
        morphology_cols = [c for c in morphology_cols if c in df.columns]

    morph_df = df[cols_to_extract].copy()

    # Apply transformations
    size_length_cols = [col for col in morphology_cols if col in ['area', 'major_axis_length', 'minor_axis_length'] and col in morph_df.columns]
    for col in size_length_cols:
        morph_df[col] = np.log1p(morph_df[col])

    if 'orientation' in morph_df.columns:
        morph_df['orientation_sin'] = np.sin(morph_df['orientation'])
        morph_df['orientation_cos'] = np.cos(morph_df['orientation'])
        morph_df = morph_df.drop(columns=['orientation'])  # Replace with sin/cos

    # Z-normalization if enabled
    if use_z_normalization and batch_col in morph_df.columns:
        # Identify columns to normalize (all except the batch column itself)
        cols_to_normalize = [col for col in morph_df.columns if col != batch_col]
        normalized_data = apply_z_normalization(morph_df, cols_to_normalize, batch_col)
        # Return only the feature columns, not the batch_col
        return normalized_data.drop(columns=[batch_col])

    # If normalization was skipped, drop batch_col if it was added
    if batch_col in morph_df.columns:
        morph_df = morph_df.drop(columns=[batch_col])

    return morph_df

def prepare_compartment_features(
    df: pd.DataFrame,
    compartment_cols: List[str]
) -> pd.DataFrame:
    """
    Extract binary compartment features (0/1). No normalization applied.

    Returns: DataFrame with compartment columns.
    """
    if not compartment_cols:
        return pd.DataFrame(index=df.index)  # Empty if no cols

    return df[compartment_cols].astype(float).copy()  # Ensure float for weighting later


# =============================================================================
# TRAINING DATA PREPARATION
# =============================================================================

def prepare_training_data(
    df: pd.DataFrame,
    marker_suffix: Optional[str] = None,
    clustering_markers: Optional[List[str]] = None,
    morphology_cols: Optional[List[str]] = None,
    compartment_cols: Optional[List[str]] = None,
    feature_weights: Optional[Dict[str, float]] = None,
    use_z_normalization: bool = False,
    batch_normalization_col: str = 'batch_id',
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepare full feature set: markers + transformed morphology + compartments.

    Applies Z-norm to markers and morphology (if enabled), then scales by group weights.

    Returns: (feature_df, marker_cols, all_feature_cols)
    """
    if feature_weights is None:
        feature_weights = {'markers': 1.0, 'morphology': 1.0, 'compartments': 0.5}

    # Markers
    if clustering_markers is not None:
        marker_cols = [col for col in clustering_markers if col in df.columns]
    else:
        if marker_suffix is None:
            raise ValueError("Either clustering_markers or marker_suffix must be provided")
        marker_cols = sorted([c for c in df.columns if c.endswith(marker_suffix)])

    if not marker_cols:
        raise ValueError("No marker columns found")

    # Include batch column for normalization if needed
    cols_for_markers = marker_cols
    if use_z_normalization:
        if batch_normalization_col not in df.columns:
            print(f"WARNING: Batch column '{batch_normalization_col}' not found for marker normalization. Skipping.")
        else:
            cols_for_markers = list(set(marker_cols + [batch_normalization_col]))

    marker_df = df[cols_for_markers].copy()
    for marker in marker_cols:
        marker_df[marker] = pd.to_numeric(marker_df[marker], errors="coerce")
    if use_z_normalization and batch_normalization_col in marker_df.columns:
        print(f"Applying Z-normalization to {len(marker_cols)} marker features grouped by '{batch_normalization_col}'...")
        marker_df = apply_z_normalization(marker_df, marker_cols, batch_normalization_col)

    # Drop batch col if it exists before weighting
    if batch_normalization_col in marker_df.columns:
        marker_df = marker_df.drop(columns=[batch_normalization_col])

    marker_df *= feature_weights.get('markers', 1.0)  # Apply weight

    # Morphology
    morph_df = prepare_morphology_features(df, morphology_cols or [], use_z_normalization, batch_normalization_col)
    morph_df *= feature_weights.get('morphology', 1.0)
    morph_cols = morph_df.columns.tolist()

    # Compartments
    comp_df = prepare_compartment_features(df, compartment_cols or [])
    comp_df *= feature_weights.get('compartments', 1.0)
    comp_cols = comp_df.columns.tolist()

    # Warn if expected keys are missing in feature_weights
    expected_keys = {'markers', 'morphology', 'compartments'}
    missing_keys = expected_keys - set(feature_weights.keys())
    if missing_keys:
        print(f"WARNING: Missing keys in feature_weights: {missing_keys}. Defaulting to 1.0.")

    # Concatenate
    feature_df = pd.concat([marker_df, morph_df, comp_df], axis=1)
    if feature_df.columns.duplicated().any():
        duplicates = feature_df.columns[feature_df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate model feature columns are not allowed: {duplicates}")
    all_feature_cols = marker_cols + morph_cols + comp_cols

    print(f"Prepared features: {len(marker_cols)} markers, {len(morph_cols)} morphology, {len(comp_cols)} compartments")
    return feature_df, marker_cols, all_feature_cols

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
    """Create a classifier instance based on the selected ML method.

    Parameters
    ----------
    ml_method : str
        One of 'random_forest', 'gradient_boosting', 'svm'.
    n_estimators : int
        Number of estimators (trees for RF/GB, ignored for SVM).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn estimator
        A fitted-ready classifier instance.
    """
    if ml_method == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=min(n_estimators, 500),  # GB is slower; cap at 500
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            subsample=0.8,
        )
    elif ml_method == "svm":
        return SVC(
            kernel="rbf",
            probability=True,  # Required for predict_proba
            class_weight="balanced",
            random_state=random_state,
            C=1.0,
            gamma="scale",
        )
    else:  # "random_forest" (default)
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )


# =============================================================================
# MODEL TRAINING & VALIDATION
# =============================================================================

def train_and_validate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    n_estimators: int,
    random_state: int,
    ml_method: str = "random_forest",
) -> Tuple:
    """Train a classifier with train/validation split and return model & validation metrics.

    Parameters
    ----------
    ml_method : str
        Classifier type: 'random_forest', 'gradient_boosting', or 'svm'.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    method_display = ML_METHOD_OPTIONS.get(ml_method, ml_method)
    print(f"  Training classifier: {method_display}")
    model = _create_classifier(ml_method, n_estimators, random_state)
    model.fit(X_train_scaled, y_train, **_balanced_fit_kwargs(model, y_train))

    y_val_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    report = classification_report(
        label_encoder.inverse_transform(y_val),
        label_encoder.inverse_transform(y_val_pred),
        zero_division=0,
    )

    metrics = {
        "validation_accuracy": val_accuracy,
        "balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
        "macro_f1": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_val, y_val_pred, average="weighted", zero_division=0),
        "classification_report": report,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "classes": list(label_encoder.classes_),
        "ml_method": method_display,
        "validation_mode": "internal_consensus_cell_split_exploratory",
        "validation_column": "consensus_phenotype",
    }
    return model, metrics, y_val, y_val_pred, label_encoder, scaler


def train_final_model(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    label_encoder: LabelEncoder,
    scaler: StandardScaler,
    n_estimators: int,
    random_state: int,
    ml_method: str = "random_forest",
):
    """Train final model on all consensus samples (fit scaler already used)."""
    y_encoded = label_encoder.transform(y_all)
    X_scaled = scaler.transform(X_all)  # Use the already-fitted scaler, do not refit
    model = _create_classifier(ml_method, n_estimators, random_state)
    model.fit(X_scaled, y_encoded, **_balanced_fit_kwargs(model, y_encoded))
    return model


# =============================================================================
# PREDICTION FOR NON-CONSENSUS
# =============================================================================

def predict_non_consensus(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    X_non_consensus: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> Tuple[np.ndarray, np.ndarray]:
    if X_non_consensus.empty:
        return np.array([], dtype=object), np.array([], dtype=float)
    X_scaled = scaler.transform(X_non_consensus)
    pred_encoded = model.predict(X_scaled)
    pred_labels = label_encoder.inverse_transform(pred_encoded)
    pred_proba = model.predict_proba(X_scaled)
    max_conf = pred_proba.max(axis=1)
    return pred_labels, max_conf


# =============================================================================
# BUILD FINAL OUTPUTS
# =============================================================================

def build_final_outputs(
    df: pd.DataFrame,
    consensus_col: str,
    is_consensus_col: str,
    pred_labels: np.ndarray,
    pred_conf: np.ndarray,
    non_consensus_mask: np.ndarray,
    output_phenotype_col: str = "phenotype_ml_consensus",
    confidence_col_name: str = "ml_consensus_score",
    consensus_col_name: str = "consensus",
    refined_confidence_threshold: float = 0.75,
    refined_low_confidence_label: str = "Others",
    n_consensus_methods: int | None = None,
) -> pd.DataFrame:
    result = df.copy()
    # Phenotype column: consensus phenotype if available, else predicted
    result[output_phenotype_col] = result[consensus_col]
    if pred_labels.size > 0:
        result.loc[non_consensus_mask, output_phenotype_col] = pred_labels

    # Score semantics differ by assignment route. Direct consensus assignments
    # use the agreeing-method fraction; model-imputed assignments use maximum
    # class probability. Neither is a calibrated probability.
    result[confidence_col_name] = np.nan
    consensus_mask = result[is_consensus_col].astype(bool)
    if n_consensus_methods and "consensus_count" in result.columns:
        result.loc[consensus_mask, confidence_col_name] = (
            result.loc[consensus_mask, "consensus_count"] / float(n_consensus_methods)
        )
    else:
        result.loc[consensus_mask, confidence_col_name] = 1.0
    if pred_conf.size > 0:
        result.loc[non_consensus_mask, confidence_col_name] = pred_conf

    # Binary consensus column (1 = consensus, 0 = non-consensus)
    result[consensus_col_name] = result[is_consensus_col].astype(int)

    # Refined phenotype column: cells below confidence threshold get relabeled
    refined_col = f"{output_phenotype_col}_refined"
    result[refined_col] = result[output_phenotype_col].copy()
    low_conf_mask = result[confidence_col_name] < refined_confidence_threshold
    n_relabeled = int(low_conf_mask.sum())
    result.loc[low_conf_mask, refined_col] = refined_low_confidence_label
    print(f"  Refined column '{refined_col}': {n_relabeled} cells relabeled as '{refined_low_confidence_label}' (threshold < {refined_confidence_threshold})")

    return result


# =============================================================================
# VALIDATION ARTIFACTS
# =============================================================================

def generate_validation_artifacts(
    metrics: dict,
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: Path,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    y_val_arr = np.asarray(y_val)
    y_pred_arr = np.asarray(y_val_pred)
    if np.issubdtype(y_val_arr.dtype, np.number):
        y_val_labels = label_encoder.inverse_transform(y_val_arr.astype(int))
    else:
        y_val_labels = y_val_arr.astype(str)
    if np.issubdtype(y_pred_arr.dtype, np.number):
        y_pred_labels = label_encoder.inverse_transform(y_pred_arr.astype(int))
    else:
        y_pred_labels = y_pred_arr.astype(str)

    # Classification report
    report_path = output_dir / "ml_consensus_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ML Consensus Validation Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Validation mode: {metrics.get('validation_mode', 'internal_consensus_split')}\n")
        f.write(f"Validation reference: {metrics.get('validation_column', 'consensus_phenotype')}\n")
        f.write(f"Validation accuracy: {metrics['validation_accuracy']:.4f}\n\n")
        f.write(f"Balanced accuracy: {metrics.get('balanced_accuracy', float('nan')):.4f}\n")
        f.write(f"Macro F1: {metrics.get('macro_f1', float('nan')):.4f}\n")
        f.write(f"Weighted F1: {metrics.get('weighted_f1', float('nan')):.4f}\n\n")
        if "direct_consensus_n" in metrics:
            f.write("Assignment-route validation:\n")
            f.write(
                f"  Direct consensus: n={metrics['direct_consensus_n']}, "
                f"accuracy={metrics.get('direct_consensus_accuracy', float('nan')):.4f}, "
                f"macro F1={metrics.get('direct_consensus_macro_f1', float('nan')):.4f}\n"
            )
            f.write(
                f"  Model imputed: n={metrics['model_imputed_n']}, "
                f"accuracy={metrics.get('model_imputed_accuracy', float('nan')):.4f}, "
                f"macro F1={metrics.get('model_imputed_macro_f1', float('nan')):.4f}\n\n"
            )
        f.write(metrics["classification_report"])

    # Confusion matrix
    try:
        labels = sorted(set(y_val_labels) | set(y_pred_labels))
        cm = confusion_matrix(y_val_labels, y_pred_labels, labels=labels)
        plt.figure(figsize=(max(8, 0.6 * len(labels)), max(6, 0.6 * len(labels))))
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(cmap="Blues", xticks_rotation=90, colorbar=False)
        plt.title("Validation Confusion Matrix (Consensus ML)")
        plt.tight_layout()
        plt.savefig(output_dir / "ml_consensus_confusion_matrix.png", dpi=dpi)
        plt.close()
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")

    # Per-class metrics heatmap
    try:
        inv_y_val = y_val_labels
        inv_y_pred = y_pred_labels
        classes = sorted(list(set(inv_y_val) | set(inv_y_pred)))
        prec, rec, f1, support = precision_recall_fscore_support(
            inv_y_val, inv_y_pred, labels=classes, average=None, zero_division=0
        )
        perf_df = pd.DataFrame(
            {
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "Support": support,
            },
            index=classes,
        )
        plt.figure(figsize=(10, max(6, 0.4 * len(classes))))
        sns.heatmap(perf_df[["Precision", "Recall", "F1"]], annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
        plt.title("Per-Class Validation Metrics (Consensus ML)")
        plt.tight_layout()
        plt.savefig(output_dir / "ml_consensus_per_class_metrics.png", dpi=dpi)
        plt.close()
    except Exception as e:
        print(f"Could not generate per-class metrics heatmap: {e}")


# =============================================================================
# EXPRESSION HEATMAPS
# =============================================================================

def generate_expression_heatmaps(
    df: pd.DataFrame,
    marker_cols: List[str],
    phenotype_col: str,
    confidence_col: str,
    output_dir: Path,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overall mean expression
    try:
        mean_expr = df.groupby(phenotype_col)[marker_cols].mean()
        if mean_expr.empty:
            return
        scaler = StandardScaler()
        z = scaler.fit_transform(mean_expr)
        z_df = pd.DataFrame(z, index=mean_expr.index, columns=mean_expr.columns)
        plt.figure(figsize=(min(24, 0.5 * len(mean_expr)), 10))
        sns.heatmap(z_df.T, cmap="bwr", center=0, linewidths=0.3)
        plt.title("Mean Marker Z-Score by ML Consensus Phenotype")
        plt.xlabel("Phenotype")
        plt.ylabel("Markers")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "ml_consensus_expression_heatmap.png", dpi=dpi)
        plt.close()
    except Exception as e:
        print(f"Expression heatmap failed: {e}")

    # Confidence-binned heatmaps
    if confidence_col in df.columns:
        bins = [0, 0.25, 0.5, 0.75, 1.0]
        labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
        temp = df.copy()
        temp["conf_bin"] = pd.cut(temp[confidence_col], bins=bins, labels=labels, include_lowest=True)
        for label in labels:
            subset = temp[temp["conf_bin"] == label]
            if subset[phenotype_col].nunique() < 2:
                continue
            try:
                mean_expr = subset.groupby(phenotype_col)[marker_cols].mean()
                if mean_expr.empty:
                    continue
                scaler = StandardScaler()
                z = scaler.fit_transform(mean_expr)
                z_df = pd.DataFrame(z, index=mean_expr.index, columns=mean_expr.columns)
                plt.figure(figsize=(min(24, 0.5 * len(mean_expr)), 10))
                sns.heatmap(z_df.T, cmap="bwr", center=0, linewidths=0.3)
                plt.title(f"Mean Marker Z-Score ({label} confidence)")
                plt.xlabel("Phenotype")
                plt.ylabel("Markers")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(output_dir / f"ml_consensus_expression_{label.replace('%','')}.png", dpi=dpi)
                plt.close()
            except Exception as e:
                print(f"Confidence heatmap failed for {label}: {e}")


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run_ml_consensus_analysis(
    config: ConsensusConfig,
    progress_callback=None,
    preloaded_df: pd.DataFrame = None,
    results_callback=None,
) -> bool:
    """Run the full ML consensus pipeline.

    Parameters
    ----------
    config : ConsensusConfig
        Full pipeline configuration.
    progress_callback : callable, optional
        A function that accepts a string message, called at each major step
        so the UI can display live progress (e.g. ``status.write``).
    preloaded_df : pd.DataFrame, optional
        If provided, skip CSV loading and use this DataFrame directly.
        This avoids a redundant read when the UI has already loaded / filtered
        the data.
    results_callback : callable, optional
        Receives the result DataFrame for interactive review. This is
        independent of the two opt-in CSV-writing settings in ``config``.
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
    _notify("STARTING ML CONSENSUS ANALYSIS")
    _notify("=" * 70)

    os.makedirs(config.output_dir, exist_ok=True)

    # Load data (skip if caller already provided a DataFrame)
    if preloaded_df is not None:
        source_df = preloaded_df.copy()
        _notify(f"[OK] Using pre-loaded dataset: {len(source_df):,} cells, {len(source_df.columns)} columns")
    else:
        _notify("Loading unified CSV...")
        try:
            from spacemap.ph_accel_utils import read_csv_fast
            source_df = read_csv_fast(config.unified_csv)
            _notify(f"[OK] Loaded dataset: {len(source_df):,} cells, {len(source_df.columns)} columns")
        except Exception as e:
            _notify(f"[ERROR] Loading unified CSV: {e}")
            return False

    df = source_df
    analysis_filter_applied = config.analysis_filter_column is not None
    if analysis_filter_applied:
        if config.analysis_filter_column not in source_df.columns:
            _notify(
                f"[ERROR] Analysis filter column '{config.analysis_filter_column}' was not found."
            )
            return False
        target = str(config.analysis_filter_value).strip()
        analysis_mask = (
            source_df[config.analysis_filter_column].astype("string").str.strip().eq(target)
        )
        if not analysis_mask.any():
            _notify(
                f"[ERROR] Analysis filter {config.analysis_filter_column} == "
                f"{config.analysis_filter_value!r} selected no rows."
            )
            return False
        df = source_df.loc[analysis_mask].copy()
        _notify(
            f"[OK] Analysis filter {config.analysis_filter_column} == "
            f"{config.analysis_filter_value!r}: {len(df):,}/{len(source_df):,} rows selected"
        )

    invalid_marker_rows_dropped = 0
    if config.drop_invalid_marker_rows:
        marker_candidates = (
            list(config.clustering_markers)
            if config.clustering_markers is not None
            else [column for column in df.columns if column.endswith(config.marker_suffix)]
        )
        existing_marker_candidates = [column for column in marker_candidates if column in df.columns]
        if existing_marker_candidates:
            marker_values = df[existing_marker_candidates].apply(pd.to_numeric, errors="coerce")
            valid_marker_mask = pd.Series(
                np.isfinite(marker_values.to_numpy()).all(axis=1), index=df.index
            )
            invalid_marker_rows_dropped = int((~valid_marker_mask).sum())
            if invalid_marker_rows_dropped:
                df = df.loc[valid_marker_mask].copy()
                _notify(
                    f"[OK] Excluded {invalid_marker_rows_dropped:,} rows with missing/"
                    "infinite selected-marker values"
                )
                if df.empty:
                    _notify("[ERROR] No rows remain after selected-marker completeness filtering.")
                    return False

    row_subset_applied = not df.index.equals(source_df.index)

    # Validate the vote panel before identifying consensus. Manual references
    # must never participate in pseudo-label construction.
    if len(config.phenotyping_methods) < 2:
        _notify("[ERROR] Consensus ML requires at least two automated phenotyping methods.")
        return False
    if config.ml_method not in ML_METHOD_OPTIONS:
        _notify(f"[ERROR] Unsupported ML classifier: '{config.ml_method}'.")
        return False
    if len(set(config.phenotyping_methods)) != len(config.phenotyping_methods):
        _notify("[ERROR] Consensus method columns must be unique.")
        return False
    missing_methods = [
        method for method in config.phenotyping_methods if method not in df.columns
    ]
    if missing_methods:
        _notify(f"[ERROR] Missing consensus method columns: {missing_methods}")
        return False
    refined_output_column = f"{config.output_phenotype_col}_refined"
    output_columns = {
        config.output_phenotype_col,
        config.confidence_col_name,
        config.consensus_col_name,
        refined_output_column,
    }
    if len(output_columns) != 4:
        _notify("[ERROR] Consensus output, score, binary, and refined columns must be unique.")
        return False
    protected_inputs = set(config.phenotyping_methods)
    if config.validation_ground_truth_column:
        protected_inputs.add(config.validation_ground_truth_column)
    collisions = sorted(output_columns & protected_inputs)
    if collisions:
        _notify(f"[ERROR] Consensus output columns cannot overwrite input columns: {collisions}")
        return False
    if config.min_consensus_methods < 2 or config.min_consensus_methods > len(config.phenotyping_methods):
        _notify(
            f"[ERROR] Minimum agreement must be between 2 and {len(config.phenotyping_methods)} "
            f"for the selected method panel (received {config.min_consensus_methods})."
        )
        return False
    forbidden_methods = [
        method for method in config.phenotyping_methods
        if _looks_like_reference_column(method)
        or _looks_like_annotation_metadata(method)
        or method in {
            config.validation_ground_truth_column,
            config.output_phenotype_col,
            f"{config.output_phenotype_col}_refined",
        }
    ]
    if forbidden_methods:
        _notify(
            "[ERROR] Manual-reference and ML-output columns cannot be consensus votes: "
            f"{forbidden_methods}"
        )
        return False

    external_validation_mask = np.zeros(len(df), dtype=bool)
    if config.validation_ground_truth_column:
        if config.validation_ground_truth_column not in df.columns:
            _notify(
                f"[ERROR] Validation reference column '{config.validation_ground_truth_column}' was not found."
            )
            return False
        external_validation_mask = _valid_reference_mask(
            df[config.validation_ground_truth_column]
        ).to_numpy(dtype=bool)
        if not external_validation_mask.any():
            _notify(
                f"[ERROR] No usable labels found in '{config.validation_ground_truth_column}'."
            )
            return False

    # Identify consensus
    n_methods = len(config.phenotyping_methods)
    display_excluded = config.excluded_phenotypes[:3] + ["..."] if len(config.excluded_phenotypes) > 3 else config.excluded_phenotypes
    _notify("\nStep 1/8: Identifying consensus cells...")
    _notify(f"  Methods ({n_methods}): {config.phenotyping_methods}")
    _notify(f"  Min agreement required: {config.min_consensus_methods} of {n_methods}")
    _notify(f"  Excluded phenotypes: {display_excluded}")
    df_cons = identify_consensus(df, config.phenotyping_methods, config.min_consensus_methods, config.excluded_phenotypes)
    n_consensus = int(df_cons["is_consensus"].sum())
    n_non = len(df_cons) - n_consensus
    _notify(f"[OK] Consensus: {n_consensus:,} cells ({n_consensus / max(1,len(df_cons)) * 100:.1f}%) -- will train ML")
    _notify(f"     Non-consensus: {n_non:,} cells ({n_non / max(1,len(df_cons)) * 100:.1f}%) -- ML will predict")

    training_consensus_mask = df_cons["is_consensus"].to_numpy(dtype=bool) & ~external_validation_mask
    n_training_consensus = int(training_consensus_mask.sum())
    if config.validation_ground_truth_column:
        _notify(
            f"  Held-out manual validation cells: {int(external_validation_mask.sum()):,}; "
            f"consensus cells permitted for model training: {n_training_consensus:,}"
        )

    if n_training_consensus < config.min_training_cells:
        _notify(
            f"[ERROR] Not enough permitted consensus cells ({n_training_consensus}) to train "
            f"(min {config.min_training_cells}). Aborting."
        )
        return False

    # Prepare feature matrix
    _notify("\nStep 2/8: Preparing feature matrix...")
    try:
        feature_df, marker_cols, feature_cols = prepare_training_data(
            df_cons,
            config.marker_suffix,
            config.clustering_markers,
            config.morphology_cols,
            config.compartment_cols,
            config.feature_weights,
            config.use_z_normalization,
            config.batch_normalization_col,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _notify(f"[ERROR] Could not prepare model features: {exc}")
        return False
    _notify(f"[OK] Feature matrix: {len(feature_cols)} features ({len(marker_cols)} markers)")
    # Fill NaN values — required for classifiers like SVM that reject missing data
    nan_count = int(feature_df.isna().sum().sum())
    if nan_count > 0:
        _notify(f"[INFO] Filling {nan_count:,} NaN values in feature matrix with 0")
        feature_df = feature_df.fillna(0)

    consensus_mask = df_cons["is_consensus"].values.astype(bool)
    non_consensus_mask = ~consensus_mask

    if config.rescue_unreachable_classes:
        training_consensus_mask, y_consensus, _rescued = augment_unreachable_classes(
            df_cons,
            training_consensus_mask,
            config.min_consensus_methods,
            notify=_notify,
        )
    else:
        y_consensus = df_cons.loc[training_consensus_mask, "consensus_phenotype"].astype(str)

    X_consensus = feature_df.loc[training_consensus_mask]
    X_non_consensus = feature_df.loc[non_consensus_mask]
    if y_consensus.nunique() < 2:
        _notify("[ERROR] Permitted consensus training cells contain fewer than two phenotype classes.")
        return False
    consensus_training_counts = y_consensus.value_counts().sort_index()
    sparse_consensus_classes = consensus_training_counts[consensus_training_counts < 20]
    if not sparse_consensus_classes.empty:
        _notify(
            "[WARN] Fewer than 20 consensus training cells are available for: "
            f"{sparse_consensus_classes.to_dict()}. Treat those class predictions cautiously."
        )

    # Train & validate
    _notify("\nStep 3/8: Training & validation...")
    ml_method = config.ml_method
    method_display = ML_METHOD_OPTIONS.get(ml_method, ml_method)
    _notify(f"  ML method: {method_display}")
    if config.validation_ground_truth_column:
        label_encoder = LabelEncoder()
        label_encoder.fit(y_consensus)
        metrics = None
        y_val = np.array([], dtype=object)
        y_val_pred = np.array([], dtype=object)
        _notify(
            "  Internal pseudo-label split skipped; final outputs will be scored against "
            f"'{config.validation_ground_truth_column}'."
        )
    else:
        try:
            _, metrics, y_val, y_val_pred, label_encoder, _ = train_and_validate(
                X_consensus,
                y_consensus,
                config.test_size,
                config.n_estimators,
                config.random_state,
                ml_method=ml_method,
            )
        except ValueError as exc:
            _notify(f"[ERROR] Exploratory consensus split failed: {exc}")
            return False
        _notify(f"[OK] Exploratory pseudo-label validation accuracy: {metrics['validation_accuracy']:.4f}")

    # Train final on all consensus
    _notify("\nStep 4/8: Training final model on all consensus cells...")
    scaler = StandardScaler()
    scaler.fit(X_consensus)
    final_model = train_final_model(
        X_consensus,
        y_consensus,
        label_encoder,
        scaler,
        config.n_estimators,
        config.random_state,
        ml_method=ml_method,
    )
    _notify("[OK] Final model trained")

    # Predict non-consensus
    _notify(f"\nStep 5/8: Predicting {X_non_consensus.shape[0]:,} non-consensus cells...")
    pred_labels, pred_conf = predict_non_consensus(final_model, scaler, X_non_consensus, label_encoder)
    _notify(f"[OK] Predicted {pred_labels.size:,} non-consensus cells")

    # Build outputs
    _notify("\nStep 6/8: Building final output dataframe...")
    output_phenotype_col = config.output_phenotype_col
    confidence_col_name = config.confidence_col_name
    consensus_col_name = config.consensus_col_name
    refined_col = f"{output_phenotype_col}_refined"
    result_df = build_final_outputs(
        df_cons,
        consensus_col="consensus_phenotype",
        is_consensus_col="is_consensus",
        pred_labels=pred_labels,
        pred_conf=pred_conf,
        non_consensus_mask=non_consensus_mask,
        output_phenotype_col=output_phenotype_col,
        confidence_col_name=confidence_col_name,
        consensus_col_name=consensus_col_name,
        refined_confidence_threshold=config.refined_confidence_threshold,
        refined_low_confidence_label=config.refined_low_confidence_label,
        n_consensus_methods=n_methods,
    )

    _notify("[OK] Output dataframe ready")

    if config.validation_ground_truth_column:
        validation_indices = df.index[external_validation_mask]
        y_val = (
            df.loc[validation_indices, config.validation_ground_truth_column]
            .astype("string")
            .str.strip()
            .to_numpy(dtype=object)
        )
        y_val_pred = result_df.loc[validation_indices, output_phenotype_col].astype(str).to_numpy()
        metrics = {
            "validation_accuracy": accuracy_score(y_val, y_val_pred),
            "balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
            "macro_f1": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_val, y_val_pred, average="weighted", zero_division=0),
            "classification_report": classification_report(y_val, y_val_pred, zero_division=0),
            "n_train": n_training_consensus,
            "n_val": int(len(validation_indices)),
            "classes": sorted(set(y_consensus)),
            "ml_method": method_display,
            "validation_mode": "held_out_manual_reference_final_output",
            "validation_column": config.validation_ground_truth_column,
        }
        validation_direct_mask = (
            result_df.loc[validation_indices, "is_consensus"].astype(bool).to_numpy()
        )
        for prefix, route_mask in (
            ("direct_consensus", validation_direct_mask),
            ("model_imputed", ~validation_direct_mask),
        ):
            route_count = int(route_mask.sum())
            metrics[f"{prefix}_n"] = route_count
            if route_count:
                route_true = y_val[route_mask]
                route_pred = y_val_pred[route_mask]
                metrics[f"{prefix}_accuracy"] = accuracy_score(route_true, route_pred)
                metrics[f"{prefix}_macro_f1"] = f1_score(
                    route_true, route_pred, average="macro", zero_division=0
                )
            else:
                metrics[f"{prefix}_accuracy"] = float("nan")
                metrics[f"{prefix}_macro_f1"] = float("nan")
        unseen_validation_classes = sorted(set(y_val) - set(label_encoder.classes_))
        if unseen_validation_classes:
            _notify(
                "[WARN] Held-out validation contains classes absent from consensus training: "
                f"{unseen_validation_classes}"
            )
        _notify(
            f"[OK] Held-out final-output validation: accuracy={metrics['validation_accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"macro_F1={metrics['macro_f1']:.4f}"
        )
        _notify(
            "  Validation routes: "
            f"direct consensus n={metrics['direct_consensus_n']} "
            f"(accuracy={metrics['direct_consensus_accuracy']:.4f}); "
            f"model imputed n={metrics['model_imputed_n']} "
            f"(accuracy={metrics['model_imputed_accuracy']:.4f})"
        )

    assert metrics is not None

    # Prepare review results. CSV persistence is explicitly opt-in.
    try:
        # Merge by source index. Rows outside an analysis filter retain prior
        # values, or remain missing when these are newly created columns.
        saved_df = source_df.copy() if row_subset_applied else df.copy()
        output_payload = {
            output_phenotype_col: result_df[output_phenotype_col],
            confidence_col_name: result_df[confidence_col_name],
            consensus_col_name: result_df[consensus_col_name],
            refined_col: result_df[refined_col],
            "is_consensus": result_df["is_consensus"].astype(int),
        }
        for column, values in output_payload.items():
            if column not in saved_df.columns:
                saved_df[column] = np.nan if column == confidence_col_name else pd.NA
            saved_df.loc[result_df.index, column] = values
        if results_callback is not None:
            results_callback(saved_df)
        csv_result_df = None
        if config.write_backup_csv or config.update_source_csv:
            csv_result_df = saved_df.drop(
                columns=["__tab7_internal_cell_id"], errors="ignore"
            )
        if config.write_backup_csv and csv_result_df is not None:
            updated_path = config.unified_csv.parent / f"{config.unified_csv.stem}_ml_consensus.csv"
            csv_result_df.to_csv(updated_path, index=False)
            _notify(f"[OK] Optional result CSV saved to: {updated_path}")
        if config.update_source_csv and csv_result_df is not None:
            csv_result_df.to_csv(config.unified_csv, index=False)
            _notify(f"[OK] Unified CSV updated: {config.unified_csv}")
        if not config.write_backup_csv and not config.update_source_csv:
            _notify(
                "[OK] Automatic CSV writing is off; assignments are available for review only."
            )
    except Exception as e:
        _notify(f"[ERROR] Preparing or saving ML-Consensus results: {e}")
        return False

    # Generate artifacts
    _notify("\nStep 7/8: Generating validation artifacts...")
    generate_validation_artifacts(
        metrics, y_val, y_val_pred, label_encoder, config.output_dir, config.figure_dpi
    )
    _notify("[OK] Validation artifacts saved")

    _notify("\nStep 8/8: Generating expression heatmaps...")
    generate_expression_heatmaps(
        result_df, marker_cols, output_phenotype_col, confidence_col_name, config.output_dir, config.figure_dpi
    )
    _notify("[OK] Expression heatmaps saved")

    # Per-method contribution to the consensus. Written unconditionally because
    # it is the only output that shows whether every selected method is pulling
    # its weight, or whether one is merely echoing the others.
    try:
        contributions = analyze_method_contributions(
            df,
            config.phenotyping_methods,
            config.min_consensus_methods,
            excluded_phenotypes=config.excluded_phenotypes,
            reference_column=config.validation_ground_truth_column,
        )
        for name, table in contributions.items():
            table.to_csv(config.output_dir / f"consensus_method_{name}.csv")
        figure_path = plot_method_contributions(
            contributions,
            config.output_dir,
            dpi=config.figure_dpi,
        )
        loo = contributions["leave_one_out"]
        metric = (
            "macro_f1_vs_reference"
            if "macro_f1_vs_reference" in loo.columns
            else "coverage"
        )
        drops = loo.drop(index=[i for i in loo.index if str(i).startswith("(nothing")])
        if not drops.empty:
            weakest = drops[metric].idxmax()  # removing it hurts least
            _notify(
                f"[OK] Method contributions saved -> {figure_path.name}. "
                f"Removing '{weakest}' costs the least {metric.replace('_', ' ')}."
            )
        else:
            _notify(f"[OK] Method contributions saved -> {figure_path.name}")
    except Exception as exc:
        _notify(f"[WARN] Method-contribution analysis skipped: {exc}")

    # Save summary
    summary_path = config.output_dir / "ml_consensus_summary.txt"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("ML Consensus Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Input rows: {len(source_df)}\n")
            f.write(f"Analysis rows processed: {len(df)}\n")
            if analysis_filter_applied:
                f.write(
                    f"Analysis filter: {config.analysis_filter_column} == "
                    f"{config.analysis_filter_value!r}\n"
                )
            f.write(
                f"Rows excluded for invalid selected markers: {invalid_marker_rows_dropped}\n"
            )
            f.write(f"Consensus cells: {n_consensus}\n")
            f.write(f"Consensus training cells: {n_training_consensus}\n")
            f.write(f"Consensus training class counts: {consensus_training_counts.to_dict()}\n")
            f.write(f"Validation mode: {metrics.get('validation_mode', 'unknown')}\n")
            f.write(f"Validation reference: {metrics.get('validation_column', 'consensus_phenotype')}\n")
            f.write(f"Validation accuracy: {metrics['validation_accuracy']:.4f}\n")
            f.write(f"Balanced accuracy: {metrics.get('balanced_accuracy', float('nan')):.4f}\n")
            f.write(f"Macro F1: {metrics.get('macro_f1', float('nan')):.4f}\n")
            f.write(f"Weighted F1: {metrics.get('weighted_f1', float('nan')):.4f}\n")
            if "direct_consensus_n" in metrics:
                f.write(
                    f"Held-out direct-consensus cells: {metrics['direct_consensus_n']} "
                    f"(accuracy={metrics.get('direct_consensus_accuracy', float('nan')):.4f})\n"
                )
                f.write(
                    f"Held-out model-imputed cells: {metrics['model_imputed_n']} "
                    f"(accuracy={metrics.get('model_imputed_accuracy', float('nan')):.4f})\n"
                )
            f.write(f"Phenotyping methods: {', '.join(config.phenotyping_methods)}\n")
            f.write(f"Features used: {len(feature_cols)} (markers: {len(marker_cols)})\n")
            f.write(f"\nConsensus binary column: {consensus_col_name}\n")
            f.write(f"Refined column: {refined_col} (threshold: {config.refined_confidence_threshold}, label: '{config.refined_low_confidence_label}')\n")
            f.write("\nPhenotype distribution (final):\n")
            dist = result_df[output_phenotype_col].value_counts()
            for label, count in dist.items():
                f.write(f"  {label}: {count} ({count/len(result_df)*100:.1f}%)\n")
            f.write("\nConfidence statistics (non-consensus predictions only):\n")
            nc_scores = result_df.loc[~result_df["is_consensus"], confidence_col_name].dropna()
            if not nc_scores.empty:
                f.write(f"  Mean: {nc_scores.mean():.3f}\n")
                f.write(f"  Median: {nc_scores.median():.3f}\n")
                f.write(f"  Min: {nc_scores.min():.3f}\n")
                f.write(f"  Max: {nc_scores.max():.3f}\n")
            else:
                f.write("  (No non-consensus predictions)\n")
    except Exception as e:
        _notify(f"Could not write summary: {e}")

    _notify("\n" + "=" * 80)
    _notify("ML CONSENSUS ANALYSIS COMPLETE")
    _notify("=" * 80)
    _notify(f"Reports and figures directory: {config.output_dir}")
    _notify(
        f"Review columns prepared: {output_phenotype_col}, {confidence_col_name}, "
        f"{consensus_col_name}, {refined_col}"
    )
    if config.update_source_csv:
        _notify(f"Updated unified file: {config.unified_csv}")
    else:
        _notify("Unified CSV unchanged; use the explicit assignment-save action after review.")
    return True


if __name__ == "__main__":  # Simple manual test hook
    pass


# =============================================================================
# METHOD CONTRIBUTION TO CONSENSUS
# =============================================================================

def analyze_method_contributions(
    df: pd.DataFrame,
    methods: List[str],
    min_consensus: int,
    excluded_phenotypes: Optional[List[str]] = None,
    reference_column: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Quantify what each method contributes to the consensus.

    Three complementary views, because "which method is best" has three
    different answers depending on what you mean:

    ``agreement``   at each agreement depth, how often each method sided with
                    the consensus label. A method that almost always sides with
                    it is behaving like the others.
    ``dissent``     on cells that reached consensus *without* unanimity, which
                    method was the hold-out. Frequent dissent is not automatically
                    bad - a method seeing something the rest miss looks identical
                    here to one that is simply noisy, which is why the third view
                    matters.
    ``leave_one_out`` what consensus coverage and (given a reference) accuracy
                    become if that method is removed entirely. This is the only
                    one that measures whether the method is *needed*.
    """
    available = [m for m in methods if m in df.columns]
    if len(available) < 2:
        raise ValueError("At least two method columns are required.")

    cons = identify_consensus(df, available, min_consensus, excluded_phenotypes)
    is_cons = cons["is_consensus"].to_numpy(dtype=bool)
    depth = pd.to_numeric(cons["consensus_count"], errors="coerce").fillna(0).astype(int).to_numpy()
    label = cons["consensus_phenotype"].astype("string")

    invalid = {"unclassified", "unknown", "", "nan", "none"}
    invalid |= {str(x).strip().lower() for x in (excluded_phenotypes or [])}
    votes = {}
    for m in available:
        v = df[m].astype("string").str.strip()
        votes[m] = v.where(~v.str.lower().isin(invalid) & v.notna())

    depths = sorted({int(d) for d in depth[is_cons]})

    # --- agreement with the consensus label, per depth ---
    rows = []
    for m in available:
        # Comparing nullable strings yields pd.NA wherever either side is
        # missing; NA means 'did not match', so fold it to False.
        agree_all = (votes[m] == label).fillna(False).to_numpy(dtype=bool)
        cast_all = votes[m].notna().to_numpy(dtype=bool)
        rec = {"method": m}
        for d in depths:
            sel = is_cons & (depth == d)
            n = int(sel.sum())
            rec[f"agreed_at_{d}"] = float(agree_all[sel].mean()) if n else float("nan")
            rec[f"n_at_{d}"] = n
        rec["voted_anywhere"] = float(cast_all.mean())
        rec["agreed_overall"] = float(agree_all[is_cons].mean()) if is_cons.any() else float("nan")
        rows.append(rec)
    agreement = pd.DataFrame(rows).set_index("method")

    # --- who dissented on non-unanimous consensus cells ---
    partial = is_cons & (depth < len(available))
    rows = []
    for m in available:
        disagreed = partial & ~(votes[m] == label).fillna(False).to_numpy(dtype=bool)
        silent = partial & votes[m].isna().to_numpy(dtype=bool)
        rows.append({
            "method": m,
            "dissented": int(disagreed.sum() - silent.sum()),
            "abstained": int(silent.sum()),
            "n_partial_consensus": int(partial.sum()),
            "dissent_rate": float((disagreed.sum() - silent.sum()) / partial.sum()) if partial.any() else float("nan"),
        })
    dissent = pd.DataFrame(rows).set_index("method").sort_values("dissent_rate", ascending=False)

    # --- what happens if the method is removed ---
    ref_ok = reference_column in df.columns if reference_column else False
    if ref_ok:
        rv = df[reference_column].astype("string").str.strip()
        ref_mask = df[reference_column].notna() & ~rv.str.lower().isin(invalid)
    rows = []
    full_k = min(min_consensus, len(available))
    # Removing a method also makes the rule stricter (3-of-4 becomes 3-of-3, i.e.
    # unanimity), and stricter rules always score better on the cells they keep.
    # Comparing a 3-method panel against the full 4-method panel at its original
    # k would therefore credit the removal for a gain that is purely the rule
    # change, so the full panel is also evaluated at the reduced k as a matched
    # baseline.
    for drop in [None, "__matched__"] + available:
        matched = drop == "__matched__"
        subset = list(available) if (drop is None or matched) else [m for m in available if m != drop]
        k = min(full_k, len(available) - 1) if matched else min(full_k, len(subset))
        sub = identify_consensus(df, subset, k, excluded_phenotypes)
        sc = sub["is_consensus"].to_numpy(dtype=bool)
        rec = {
            "removed": (
                "(nothing removed)" if drop is None
                else f"(nothing removed, {k} of {len(available)} agreeing)" if matched
                else drop
            ),
            "methods_left": len(subset),
            "min_agree_used": k,
            "consensus_cells": int(sc.sum()),
            "coverage": float(sc.mean()),
        }
        if ref_ok:
            both = ref_mask.to_numpy(dtype=bool) & sc
            if both.any():
                t = df.loc[both, reference_column].astype(str)
                p = sub.loc[both, "consensus_phenotype"].astype(str)
                rec["accuracy_vs_reference"] = float((t.values == p.values).mean())
                rec["macro_f1_vs_reference"] = float(
                    f1_score(t, p, average="macro", zero_division=0)
                )
                rec["reference_cells_covered"] = int(both.sum())
        rows.append(rec)
    leave_one_out = pd.DataFrame(rows).set_index("removed")

    return {"agreement": agreement, "dissent": dissent, "leave_one_out": leave_one_out}


def plot_method_contributions(
    tables: Dict[str, pd.DataFrame],
    plots_dir,
    filename: str = "consensus_method_contributions.png",
    dpi: int = 300,
    display_names: Optional[Dict[str, str]] = None,
):
    """Three-panel summary of what each method brings to the consensus."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    nice = lambda m: (display_names or {}).get(m, str(m).replace("phenotype_", ""))

    agreement, dissent, loo = tables["agreement"], tables["dissent"], tables["leave_one_out"]
    depth_cols = sorted(
        [c for c in agreement.columns if c.startswith("agreed_at_")],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # 1. agreement by depth
    ax = axes[0]
    idx = np.arange(len(agreement))
    width = 0.8 / max(len(depth_cols), 1)
    for i, c in enumerate(depth_cols):
        d = c.rsplit("_", 1)[1]
        n = int(agreement[f"n_at_{d}"].iloc[0]) if f"n_at_{d}" in agreement else 0
        ax.bar(idx + i * width, agreement[c] * 100, width,
               label=f"{d} of {len(agreement)} agreed (n={n:,})")
    ax.set_xticks(idx + width * (len(depth_cols) - 1) / 2)
    ax.set_xticklabels([nice(m) for m in agreement.index], rotation=20, ha="right")
    ax.set_ylabel("% of those cells where the method\nmatched the consensus label")
    ax.set_title("How often each method sided with the consensus")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8)

    # 2. who dissented
    ax = axes[1]
    n_partial = int(dissent["n_partial_consensus"].iloc[0]) if len(dissent) else 0
    rates = dissent["dissent_rate"].astype(float)
    if n_partial == 0 or not np.isfinite(rates).any():
        # Unanimity was required, so no consensus cell had a hold-out to identify.
        ax.text(0.5, 0.5,
                "Every consensus cell was unanimous,\nso no method was ever the hold-out.\n\n"
                "Lower the agreement cutoff to see this panel.",
                ha="center", va="center", fontsize=10, color="#555", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title("Who disagreed when the others agreed")
    else:
        order = dissent.sort_values("dissent_rate")
        ax.barh([nice(m) for m in order.index], order["dissent_rate"] * 100, color="#c0392b")
        for i, (v, ab) in enumerate(zip(order["dissent_rate"] * 100, order["abstained"])):
            ax.text(v + 0.5, i, f"{v:.1f}%" + (f"  (+{ab:,} silent)" if ab else ""),
                    va="center", fontsize=8)
        ax.set_xlabel("% of non-unanimous consensus cells where this method was the hold-out")
        ax.set_title("Who disagreed when the others agreed")
        ax.set_xlim(0, max(float(rates.max()) * 130, 10))

    # 3. leave-one-out
    ax = axes[2]
    metric = "macro_f1_vs_reference" if "macro_f1_vs_reference" in loo.columns else "coverage"
    lbl = "macro-F1 vs manual reference" if metric.startswith("macro") else "coverage"
    baseline_rows = [i for i in loo.index if str(i).startswith("(nothing")]
    # The baseline is the configuration actually in use - all methods at the
    # chosen cutoff. Dropping a method unavoidably changes the rule as well
    # (3-of-4 becomes 3-of-3), so coverage is printed on every bar: a bar above
    # the line that also lost a third of its coverage has not won anything.
    base = float(loo.loc["(nothing removed)", metric])
    drops = loo.drop(index=baseline_rows)
    colors = ["#c0392b" if v < base else "#27ae60" for v in drops[metric]]
    ax.bar([nice(m) for m in drops.index], drops[metric], color=colors, zorder=2)
    ax.axhline(base, ls="--", c="#333", lw=1.5, zorder=3)
    for i, (v, cov) in enumerate(zip(drops[metric], drops["coverage"])):
        ax.text(i, v, f"{v:.3f}\n{cov:.0%} cov", ha="center", va="bottom", fontsize=8)
    lo = min(float(drops[metric].min()), base)
    hi = max(float(drops[metric].max()), base)
    pad = max((hi - lo) * 0.45, 0.02)
    ax.set_ylim(max(0.0, lo - pad), hi + pad * 1.5)
    ax.set_ylabel(lbl)
    ax.set_title(
        f"If this method were removed\n"
        f"(dashed = all methods kept: {base:.3f} at {float(loo.loc['(nothing removed)', 'coverage']):.0%} coverage)"
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    path = plots_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
