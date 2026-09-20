"""comparison_utils.py

This module contains the core logic for the comprehensive phenotyping
validation analysis performed in the Step8_Phenotype_Comparison.ipynb notebook.

It includes functions for:
- Data loading and preparation.
- Validation of individual phenotyping methods against a ground truth.
- Identification and performance analysis of a consensus phenotype.
- Generation of comparative visualizations (performance plots, confusion matrices).
- Generation of CSV reports and a final summary text file.

The main entry point is the `run_analysis` function, which orchestrates the
entire workflow based on the configuration provided from the notebook.
"""
from __future__ import annotations

import os
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

__all__ = [
    "prepare_data_for_analysis",
    "validate_individual_methods",
    "identify_consensus_cells",
    "calculate_consensus_performance",
    "create_comprehensive_performance_comparison",
    "create_comprehensive_confusion_matrices",
    "create_comprehensive_per_class_analysis",
    "create_method_agreement_analysis",
    "generate_csv_outputs",
    "generate_report",
    "run_analysis",
]


# =============================================================================
# DATA PREPARATION
# =============================================================================
def prepare_data_for_analysis(input_file: Path, ground_truth_column: str, dataset_id_column: str) -> pd.DataFrame | None:
    """
    Loads and prepares the main CSV for validation analysis.
    """
    print(f"Preparing data from: {input_file}")
    try:
        df = pd.read_csv(input_file)

        if ground_truth_column not in df.columns:
            print(f"ERROR: Ground truth column '{ground_truth_column}' not found.")
            return None
        df.rename(columns={ground_truth_column: "ground_truth"}, inplace=True)

        if dataset_id_column not in df.columns:
            print(f"ERROR: Dataset ID column '{dataset_id_column}' not found.")
            return None
        df.rename(columns={dataset_id_column: "dataset_name"}, inplace=True)

        # Filter out rows with invalid ground truth labels
        bad_labels = ["unclassified", "unknown", ""]
        df = df[~df["ground_truth"].isin(bad_labels)].copy()
        df.dropna(subset=["ground_truth"], inplace=True)

        print("Data preparation successful.")
        print(f"  Total cells for analysis: {len(df)}")
        print(f"  Ground truth cell types: {df['ground_truth'].nunique()}")
        print(f"  Datasets found: {df['dataset_name'].nunique()}")
        return df

    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_file}")
        return None
    except Exception as e:
        print(f"ERROR during data preparation: {e}")
        return None


# =============================================================================
# INDIVIDUAL METHOD VALIDATION
# =============================================================================
def validate_individual_methods(df: pd.DataFrame, methods: list[str]) -> list[dict]:
    """Validate each individual phenotyping method against ground truth."""
    print("Validating individual phenotyping methods...")
    individual_results = []
    available_methods = [m for m in methods if m in df.columns]

    for method in available_methods:
        print(f"  Analyzing {method}...")

        method_df = df[df[method].notna() & (df[method] != "unknown") & df["ground_truth"].notna()].copy()

        if len(method_df) == 0:
            print(f"    No valid predictions found for {method}")
            continue

        y_true = method_df["ground_truth"]
        y_pred = method_df[method]

        labels = sorted(list(set(y_true) | set(y_pred)))
        precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        results = {
            "method": method,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_w,
            "recall_weighted": recall_w,
            "f1_weighted": f1_w,
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "matthews_corr": matthews_corrcoef(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
            "labels": labels,
            "per_class_precision": precision_per,
            "per_class_recall": recall_per,
            "per_class_f1": f1_per,
            "per_class_support": support_per,
            "n_cells": len(method_df),
            "coverage": (len(method_df) / len(df)) * 100,
        }
        individual_results.append(results)
        print(f"    {method}: {results['accuracy']:.3f} accuracy ({len(method_df)} cells)")

    return individual_results


# =============================================================================
# CONSENSUS ANALYSIS (PERFORMANCE OPTIMIZED)
# =============================================================================
def identify_consensus_cells(df: pd.DataFrame, methods: list[str], min_agreement: int) -> pd.DataFrame:
    """
    Identify consensus cells where at least min_agreement methods agree.
    (Vectorized for high performance)
    """
    print(f"Identifying consensus cells (>{min_agreement} methods agreeing)...")

    df_consensus = df.copy()
    available_methods = [m for m in methods if m in df.columns]
    print(f"  Available methods: {available_methods}")

    preds_df = df_consensus[available_methods]

    valid_predictions_count = preds_df.replace("unknown", np.nan).notna().sum(axis=1)
    consensus_phenotype = preds_df.mode(axis=1)[0]
    consensus_count = preds_df.eq(consensus_phenotype, axis=0).sum(axis=1)
    is_consensus = (consensus_count >= min_agreement) & (valid_predictions_count >= min_agreement)

    df_consensus["consensus_phenotype"] = consensus_phenotype
    df_consensus["consensus_count"] = consensus_count
    df_consensus["is_consensus"] = is_consensus

    df_consensus.loc[~is_consensus, ["consensus_phenotype", "consensus_count"]] = (None, 0)

    def get_agreeing_methods(row: pd.Series) -> str:
        if not row["is_consensus"]:
            return ""
        return ", ".join([m for m in available_methods if row[m] == row["consensus_phenotype"]])

    df_consensus["consensus_methods"] = df_consensus.apply(get_agreeing_methods, axis=1)

    consensus_rate = (df_consensus["is_consensus"].sum() / len(df_consensus)) * 100
    print(f"  Found {df_consensus['is_consensus'].sum()} consensus cells ({consensus_rate:.1f}%)")

    return df_consensus


def calculate_consensus_performance(df_consensus: pd.DataFrame) -> dict | None:
    """Calculate performance metrics for consensus predictions."""
    print("Calculating consensus performance metrics...")

    consensus_df = df_consensus[
        df_consensus["is_consensus"] & df_consensus["ground_truth"].notna() & df_consensus["consensus_phenotype"].notna()
    ].copy()

    if len(consensus_df) == 0:
        print("  No valid consensus cells found for performance calculation!")
        return None

    y_true = consensus_df["ground_truth"]
    y_pred = consensus_df["consensus_phenotype"]

    labels = sorted(list(set(y_true) | set(y_pred)))
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    results = {
        "method": "Consensus",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "matthews_corr": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
        "per_class_precision": precision_per,
        "per_class_recall": recall_per,
        "per_class_f1": f1_per,
        "per_class_support": support_per,
        "n_cells": len(consensus_df),
        "coverage": (len(consensus_df) / len(df_consensus)) * 100,
    }

    print(f"  Consensus accuracy: {results['accuracy']:.3f}")
    print(f"  Consensus F1-score: {results['f1_weighted']:.3f}")
    print(f"  Cohen's Kappa: {results['cohen_kappa']:.3f}")

    return results


# =============================================================================
# VISUALIZATIONS
# =============================================================================
def create_comprehensive_performance_comparison(all_results: list[dict], output_dir: Path, figure_dpi: int) -> None:
    print("Creating comprehensive performance comparison...")
    if not all_results:
        print("  No results to visualize!")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(
        "Comprehensive Phenotyping Validation: Individual Methods + Consensus",
        fontsize=16,
        fontweight="bold",
    )

    methods = [r["method"] for r in all_results]
    accuracies = [r["accuracy"] for r in all_results]

    colors = ["red" if m == "Consensus" else "skyblue" for m in methods]
    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_title("Overall Accuracy comparison")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(methods))
    width = 0.2

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [r[metric] for r in all_results]
        ax2.bar(x + i * width, values, width, label=name, alpha=0.8)

    ax2.set_title("Multiple Metrics comparison")
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Methods")
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.legend()
    ax2.set_ylim(0, 1)

    kappas = [r["cohen_kappa"] for r in all_results]
    colors = ["red" if m == "Consensus" else "lightgreen" for m in methods]
    bars = ax3.bar(methods, kappas, color=colors, alpha=0.8, edgecolor="black")
    ax3.set_title("Cohen's Kappa (Agreement) comparison")
    ax3.set_ylabel("Cohen's Kappa")
    ax3.set_ylim(-0.1, 1)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    n_cells = [r["n_cells"] for r in all_results]
    colors = ["red" if m == "Consensus" else "orange" for m in methods]
    bars = ax4.bar(methods, n_cells, color=colors, alpha=0.8, edgecolor="black")
    ax4.set_title("Number of Cells Analyzed")
    ax4.set_ylabel("Number of Cells")
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 50,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        output_dir / "comprehensive_performance_comparison.png",
        dpi=figure_dpi,
        bbox_inches="tight",
    )
    plt.close()


def create_comprehensive_confusion_matrices(all_results: list[dict], output_dir: Path, figure_dpi: int) -> None:
    print("Creating comprehensive confusion matrices...")
    if not all_results:
        return

    n_methods = len(all_results)
    n_cols = 2
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows), squeeze=False)
    axes = axes.flatten()

    fig.suptitle("Confusion Matrices: All Methods + Consensus", fontsize=16, fontweight="bold")

    for i, result in enumerate(all_results):
        ax = axes[i]
        cm = result["confusion_matrix"]
        labels = result["labels"]

        im = ax.imshow(cm, cmap="Blues")

        for i_cm in range(len(labels)):
            for j_cm in range(len(labels)):
                ax.text(j_cm, i_cm, cm[i_cm, j_cm], ha="center", va="center", color="black", fontweight="bold")

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        title_color = "red" if result["method"] == "Consensus" else "black"
        ax.set_title(f'{result["method"]}\n(Acc: {result["accuracy"]:.3f})', color=title_color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        output_dir / "comprehensive_confusion_matrices.png",
        dpi=figure_dpi,
        bbox_inches="tight",
    )
    plt.close()


def create_comprehensive_per_class_analysis(all_results: list[dict], output_dir: Path, figure_dpi: int) -> None:
    print("Creating comprehensive per-class analysis...")
    if not all_results:
        return

    all_cell_types = sorted(list(set(l for r in all_results for l in r["labels"])))
    methods = [r["method"] for r in all_results]

    f1_matrix = pd.DataFrame(index=all_cell_types, columns=methods, dtype=float)
    precision_matrix = pd.DataFrame(index=all_cell_types, columns=methods, dtype=float)
    recall_matrix = pd.DataFrame(index=all_cell_types, columns=methods, dtype=float)

    for result in all_results:
        method = result["method"]
        for i, label in enumerate(result["labels"]):
            f1_matrix.loc[label, method] = result["per_class_f1"][i]
            precision_matrix.loc[label, method] = result["per_class_precision"][i]
            recall_matrix.loc[label, method] = result["per_class_recall"][i]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, max(8, len(all_cell_types) * 0.5)))
    fig.suptitle("Per-Class Performance: All Methods + Consensus", fontsize=16, fontweight="bold")

    sns.heatmap(
        precision_matrix,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        ax=ax1,
        cbar_kws={"label": "Precision"},
        vmin=0,
        vmax=1,
    )
    ax1.set_title("Precision by Cell Type")

    sns.heatmap(
        recall_matrix, annot=True, fmt=".2f", cmap="Greens", ax=ax2, cbar_kws={"label": "Recall"}, vmin=0, vmax=1
    )
    ax2.set_title("Recall by Cell Type")

    sns.heatmap(
        f1_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax3, cbar_kws={"label": "F1-Score"}, vmin=0, vmax=1
    )
    ax3.set_title("F1-Score by Cell Type")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Method")
        ax.set_ylabel("Cell Type" if ax == ax1 else "")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        output_dir / "comprehensive_per_class_analysis.png",
        dpi=figure_dpi,
        bbox_inches="tight",
    )
    plt.close()


def create_method_agreement_analysis(df_consensus: pd.DataFrame, phenotyping_methods: list[str], min_consensus_methods: int, output_dir: Path, figure_dpi: int) -> None:
    print("Creating method agreement analysis...")
    if "is_consensus" not in df_consensus.columns:
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Method Agreement Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Consensus Rate by Dataset
    if "dataset_name" in df_consensus.columns:
        consensus_by_dataset = (
            df_consensus.groupby("dataset_name")["is_consensus"].value_counts(normalize=True).unstack().fillna(0)
        )
        consensus_rate = consensus_by_dataset.get(True, 0) * 100

        bars = ax1.bar(consensus_rate.index, consensus_rate.values, color=sns.color_palette("Set2", len(consensus_rate)))
        ax1.set_title("Consensus Rate by Dataset")
        ax1.set_ylabel("Consensus Rate (%)")
        ax1.set_ylim(0, 100)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

    # Plot 2: Distribution of Consensus Strength
    consensus_cells_df = df_consensus[df_consensus["is_consensus"]]
    if not consensus_cells_df.empty:
        consensus_counts = consensus_cells_df["consensus_count"].value_counts().sort_index()
        ax2.bar(
            consensus_counts.index,
            consensus_counts.values,
            color=sns.color_palette("viridis", len(consensus_counts)),
        )
        ax2.set_title("Distribution of Consensus Strength")
        ax2.set_xlabel("Number of Methods Agreeing")
        ax2.set_ylabel("Number of Cells")
        for count, cells in consensus_counts.items():
            ax2.text(count, cells, str(cells), ha="center", va="bottom")

    # Plot 3: Cell Type Distribution
    if not consensus_cells_df.empty:
        dist_df = pd.DataFrame(
            {
                "Ground Truth": consensus_cells_df["ground_truth"].value_counts(),
                "Consensus": consensus_cells_df["consensus_phenotype"].value_counts(),
            }
        ).fillna(0).astype(int)

        dist_df.plot(kind="bar", ax=ax3, width=0.8)
        ax3.set_title("Cell Type Distribution: GT vs Consensus")
        ax3.set_ylabel("Number of Cells")
        ax3.tick_params(axis="x", rotation=45)
        ax3.legend()

    # Plot 4: Summary Text
    total_cells = len(df_consensus)
    num_consensus_cells = df_consensus["is_consensus"].sum()
    consensus_rate = (num_consensus_cells / total_cells) * 100 if total_cells > 0 else 0

    summary_text = (
        f"VALIDATION SUMMARY\n\n"
        f"Total Cells: {total_cells:,}\n"
        f"Consensus Cells: {num_consensus_cells:,}\n"
        f"Consensus Rate: {consensus_rate:.1f}%\n\n"
        f"Methods Analyzed: {len(phenotyping_methods)}\n"
        f"Min Methods for Consensus: {min_consensus_methods}"
    )

    ax4.text(
        0.05,
        0.5,
        summary_text,
        fontsize=12,
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.7),
    )
    ax4.axis("off")
    ax4.set_title("Analysis Summary")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "method_agreement_analysis.png", dpi=figure_dpi, bbox_inches="tight")
    plt.close()


# =============================================================================
# CSV AND REPORT GENERATION
# =============================================================================
def generate_csv_outputs(
    df_consensus: pd.DataFrame, all_results: list[dict], phenotyping_methods: list[str], output_dir: Path
) -> None:
    print("Generating CSV outputs...")

    # Overall performance comparison
    comparison_df = pd.DataFrame(all_results).drop(
        columns=[
            "confusion_matrix",
            "labels",
            "per_class_precision",
            "per_class_recall",
            "per_class_f1",
            "per_class_support",
        ]
    )
    comparison_df.sort_values("accuracy", ascending=False, inplace=True)
    comparison_df.to_csv(output_dir / "performance_comparison.csv", index=False)
    print("  Performance comparison saved.")

    # Per-class performance
    per_class_data = []
    for result in all_results:
        for i, cell_type in enumerate(result["labels"]):
            per_class_data.append(
                {
                    "Method": result["method"],
                    "Cell_Type": cell_type,
                    "Precision": result["per_class_precision"][i],
                    "Recall": result["per_class_recall"][i],
                    "F1_Score": result["per_class_f1"][i],
                    "Support": result["per_class_support"][i],
                }
            )
    per_class_df = pd.DataFrame(per_class_data)
    per_class_df.to_csv(output_dir / "per_class_performance.csv", index=False)
    print("  Per-class performance saved.")

    # Detailed consensus cell data
    if "is_consensus" in df_consensus.columns:
        consensus_cells = df_consensus[df_consensus["is_consensus"]].copy()
        if not consensus_cells.empty:
            output_cols = [
                "dataset_name",
                "cell_id",
                "x_centroid",
                "y_centroid",
                "ground_truth",
                "consensus_phenotype",
                "consensus_count",
                "consensus_methods",
            ] + phenotyping_methods
            available_cols = [col for col in output_cols if col in consensus_cells.columns]

            consensus_detailed = consensus_cells[available_cols].copy()
            consensus_detailed["consensus_correct"] = (
                consensus_detailed["ground_truth"] == consensus_detailed["consensus_phenotype"]
            )
            consensus_detailed.to_csv(output_dir / "detailed_consensus_cells.csv", index=False)
            print(f"  Detailed consensus cells saved: {len(consensus_detailed)} cells")


def generate_report(
    df_consensus: pd.DataFrame,
    all_results: list[dict],
    phenotyping_methods: list[str],
    min_consensus_methods: int,
    output_dir: Path,
) -> None:
    print("Generating analysis report...")
    report_path = output_dir / "analysis_report.txt"

    with open(report_path, "w") as f:
        f.write("PHENOTYPING VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("ANALYSIS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total cells analyzed: {len(df_consensus)}\n")
        f.write(f"Methods: {', '.join(phenotyping_methods)}\n")

        if "is_consensus" in df_consensus.columns:
            consensus_cells = df_consensus["is_consensus"].sum()
            consensus_rate = (consensus_cells / len(df_consensus)) * 100 if len(df_consensus) > 0 else 0
            f.write(f"Consensus cells found: {consensus_cells} ({consensus_rate:.1f}%)\n")
            f.write(f"Minimum methods for consensus: {min_consensus_methods}\n\n")

        if not all_results:
            f.write("No results to report.\n")
            return

        sorted_results = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)

        f.write("PERFORMANCE RANKING (BY ACCURACY)\n")
        f.write("-" * 40 + "\n")
        for i, result in enumerate(sorted_results, 1):
            f.write(f"{i}. {result['method']:<20} Accuracy: {result['accuracy']:.3f}\n")
        f.write("\n")

        f.write("DETAILED PERFORMANCE METRICS\n")
        f.write("-" * 50 + "\n")
        for result in sorted_results:
            f.write(f"\n--- {result['method'].upper()} ---\n")
            f.write(f"  Accuracy:             {result['accuracy']:.3f}\n")
            f.write(f"  F1-Score (weighted):  {result['f1_weighted']:.3f}\n")
            f.write(f"  Precision (weighted): {result['precision_weighted']:.3f}\n")
            f.write(f"  Recall (weighted):    {result['recall_weighted']:.3f}\n")
            f.write(f"  Cohen's Kappa:        {result['cohen_kappa']:.3f}\n")
            f.write(f"  Matthews Corr Coef:   {result['matthews_corr']:.3f}\n")
            f.write(f"  Cells analyzed:       {result['n_cells']}\n")
            f.write(f"  Coverage:             {result['coverage']:.1f}%\n")

    print(f"  Analysis report saved: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_analysis(
    input_file: Path,
    output_dir: Path,
    ground_truth_column: str,
    dataset_id_column: str,
    phenotyping_methods: list[str],
    min_consensus_methods: int,
    figure_dpi: int,
    perform_consensus_analysis: bool = True,
) -> None:
    """Main function to run the entire validation analysis workflow."""
    print("=" * 80)
    print("STARTING COMPREHENSIVE PHENOTYPING VALIDATION ANALYSIS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")
    sns.set_palette("husl")

    print("\nStep 1: Loading and preparing data...")
    df = prepare_data_for_analysis(input_file, ground_truth_column, dataset_id_column)
    if df is None:
        print("Analysis halted: Could not load data.")
        return

    print("\nStep 2: Validating individual phenotyping methods...")
    individual_results = validate_individual_methods(df, phenotyping_methods)
    if not individual_results:
        print("Analysis halted: No individual method results were generated.")
        return

    all_results = individual_results.copy()
    df_consensus = df.copy() # Initialize with base df

    if perform_consensus_analysis:
        print("\nStep 3: Identifying consensus cells...")
        df_consensus = identify_consensus_cells(df, phenotyping_methods, min_consensus_methods)

        print("\nStep 4: Calculating consensus performance...")
        consensus_results = calculate_consensus_performance(df_consensus)

        if consensus_results:
            all_results.append(consensus_results)
    else:
        print("\nSkipping consensus analysis as per configuration.")


    print(f"  Total result sets to process: {len(all_results)}")

    print("\nStep 5: Generating visualizations...")
    create_comprehensive_performance_comparison(all_results, output_dir, figure_dpi)
    create_comprehensive_confusion_matrices(all_results, output_dir, figure_dpi)
    create_comprehensive_per_class_analysis(all_results, output_dir, figure_dpi)
    if perform_consensus_analysis:
        create_method_agreement_analysis(df_consensus, phenotyping_methods, min_consensus_methods, output_dir, figure_dpi)

    print("\nStep 6: Generating CSV outputs...")
    generate_csv_outputs(df_consensus, all_results, phenotyping_methods, output_dir)

    print("\nStep 7: Generating final report...")
    generate_report(df_consensus, all_results, phenotyping_methods, min_consensus_methods, output_dir)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"Results saved in: {output_dir}")
    print("=" * 80)
