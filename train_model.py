"""
================================================================================
Mobile Phone Customer Churn Classification Using Machine Learning
================================================================================
File: train_model.py
Purpose: Command-line training pipeline for the two supported telecom churn
         datasets (Telco Customer Churn, Cell2Cell Telecom Churn).

         1. Reads one CSV file (the uploaded file is never modified).
         2. Identifies which supported dataset it is and validates the target
            (exactly two classes).
         3. Removes identifier / high-cardinality columns automatically and runs
            a heuristic leakage scan.
         4. Splits 80/20 (random_state=42) and trains:
              - Logistic Regression
              - Decision Tree
              - K-Nearest Neighbors (KNN)
              - Gaussian Naive Bayes
         5. Evaluates Accuracy, Precision, Recall and F1 Score, then selects the
            best model by F1 Score.
         6. Writes DATASET-SPECIFIC artifacts so one dataset never overwrites
            another dataset's results:
              outputs/<key>_model_results.csv
              outputs/<key>_accuracy_comparison.png
              outputs/<key>_confusion_matrix.png
              models/<key>_churn_model.pkl
            where <key> is telco or cell2cell.

Examples:
    python train_model.py --data Telco-Customer-Churn.csv
    python train_model.py --data cell2celltrain.csv --out-prefix cell2cell_

Run `python train_model.py --help` for every option.
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataset_manager import (
    SUPPORTED_DATASETS,
    DatasetError,
    detect_target,
    ensure_supported_dataset,
    prepare_dataset,
    scan_for_leakage,
    suggest_excluded_columns,
    train_models,
    validate_target,
)

# ------------------------------------------------------------------------------
# 1. WORKSPACE DIRECTORIES
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def ensure_directories() -> None:
    """Creates the data/ models/ outputs/ folders when missing."""
    for folder in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        os.makedirs(folder, exist_ok=True)


# ------------------------------------------------------------------------------
# 2. STATIC PLOT EXPORTS
# ------------------------------------------------------------------------------
def save_comparison_plot(results_df: pd.DataFrame, path: str) -> None:
    """Bar chart comparing Accuracy and F1 Score for every benchmarked model."""
    plt.figure(figsize=(9, 5))
    x_indices = np.arange(len(results_df))
    width = 0.35

    plt.bar(x_indices - width / 2, results_df["Accuracy"] * 100, width, label="Accuracy (%)", color="#1E3A8A")
    plt.bar(x_indices + width / 2, results_df["F1 Score"] * 100, width, label="F1 Score (%)", color="#0D9488")

    plt.xlabel("Classification Algorithm", fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Model Performance Comparison (Accuracy vs. F1 Score)", fontsize=13, fontweight="bold")
    plt.xticks(x_indices, results_df["Model"], fontsize=10)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(frameon=True)

    for i in range(len(results_df)):
        acc_val = results_df.loc[i, "Accuracy"] * 100
        f1_val = results_df.loc[i, "F1 Score"] * 100
        plt.text(i - width / 2, acc_val + 1.2, f"{acc_val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        plt.text(i + width / 2, f1_val + 1.2, f"{f1_val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_confusion_plot(
    cm: np.ndarray,
    best_model_name: str,
    path: str,
    labels: tuple[str, str] = ("Retained (0)", "Churned (1)"),
) -> None:
    """Confusion-matrix heatmap for the selected production model."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(labels),
        yticklabels=list(labels),
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix: {best_model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("Actual True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ------------------------------------------------------------------------------
# 3. TRAIN ON ONE SUPPORTED DATASET CSV
# ------------------------------------------------------------------------------
def train_from_csv(
    csv_path: str,
    target_col: str | None = None,
    drop_cols: list[str] | None = None,
    output_prefix: str | None = None,
):
    """
    Trains, benchmarks and exports the four classifiers for one CSV file.

    Parameters
    ----------
    csv_path:
        Path to the Telco or Cell2Cell CSV. The file is only read.
    target_col:
        Optional override for the label column (auto-detected by default).
    drop_cols:
        Extra feature columns to exclude on top of the automatic suggestions.
    output_prefix:
        Artifact prefix. Defaults to the detected dataset key, e.g. ``telco_``.

    Returns
    -------
    dict with results_df, best_model_name and every artifact path.
    """
    ensure_directories()

    if not csv_path:
        raise DatasetError("No CSV path was provided. Use --data path/to/dataset.csv")
    if not os.path.exists(csv_path):
        raise DatasetError(
            f"File not found: {csv_path}\n"
            "Download one of the supported datasets (see README.md) and pass its path with --data."
        )

    print("\n" + "=" * 76)
    print(" MOBILE PHONE CUSTOMER CHURN — MODEL TRAINING")
    print("=" * 76)

    # ---------------------------------------------------------------- load ----
    print(f"[Step 1] Reading CSV: {csv_path}")
    try:
        frame = pd.read_csv(csv_path, low_memory=False)
    except Exception as error:
        raise DatasetError(f"Could not read '{csv_path}' as CSV: {error}") from error

    frame.columns = [str(column).strip() for column in frame.columns]
    print(f"  -> Raw shape: {frame.shape[0]:,} rows x {frame.shape[1]} columns")

    # ----------------------------------------------------------- identify ----
    dataset = ensure_supported_dataset(frame)
    print(f"  -> Dataset identified: {dataset.name} ({dataset.key})")

    target = target_col if target_col in frame.columns else None
    if target is None:
        target = detect_target(frame, dataset.target_candidates)
    if not target:
        raise DatasetError(
            "Could not detect a binary target column. Pass one explicitly with "
            f"--target COLUMN (expected: {', '.join(dataset.target_candidates)})."
        )

    report = validate_target(frame, target)
    if not report.is_valid:
        raise DatasetError(f"Invalid target column '{target}': {report.error}")

    print(f"  -> Target column: '{target}'")
    print(
        f"  -> Classes: {report.negative_label} ({report.n_negative:,}) vs "
        f"{report.positive_label} ({report.n_positive:,}) — "
        f"{report.positive_rate:.2f}% positive"
    )
    if report.n_missing:
        print(f"  -> Dropped {report.n_missing:,} row(s) with an empty target.")

    findings = scan_for_leakage(frame, target)
    if findings:
        for finding in findings:
            print(f"  -> [Leakage scan] {', '.join(finding.columns)}: {finding.detail}")
    else:
        print("  -> [Leakage scan] No duplicate-label / post-outcome / perfect-separator columns.")

    # ------------------------------------------------------- preprocessing ----
    suggested = suggest_excluded_columns(frame, target)
    exclusions = sorted(set(drop_cols or []) | set(suggested))
    if suggested:
        print(f"  -> Auto-excluded (IDs / free text / high cardinality): {', '.join(suggested)}")
    if drop_cols:
        print(f"  -> Excluded from --drop: {', '.join(sorted(set(drop_cols)))}")

    X, y, schema = prepare_dataset(frame, target, exclusions)

    print(
        f"  -> Features kept: {len(schema.numeric_cols)} numeric + "
        f"{len(schema.categorical_cols)} categorical = {schema.n_features}"
    )
    if schema.dropped_cols:
        print("  -> Columns not used as features:")
        for column, reason in schema.dropped_cols.items():
            print(f"       - {column}: {reason}")

    # -------------------------------------------------------------- train ----
    print("\n[Step 2] Splitting 80/20 (random_state=42) and training 4 algorithms...")
    bundle = train_models(X, y, schema)
    results_df = bundle["results_df"]
    best_model_name = bundle["best_model_name"]

    display_df = results_df.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[column] = display_df[column].apply(lambda value: f"{value * 100:.2f}%")
    print()
    print("=" * 76)
    print("                     MODEL PERFORMANCE COMPARISON")
    print("=" * 76)
    print(display_df.to_string(index=False))
    print("=" * 76)
    print(
        f"  -> Training rows: {bundle['train_size']:,} | "
        f"Test rows: {bundle['test_size']:,}"
    )
    print(f"  -> Best model by F1 Score: {best_model_name}")

    # -------------------------------------------------------------- export ----
    prefix = output_prefix or f"{dataset.key}_"
    results_csv_path = os.path.join(OUTPUTS_DIR, f"{prefix}model_results.csv")
    comparison_plot_path = os.path.join(OUTPUTS_DIR, f"{prefix}accuracy_comparison.png")
    cm_plot_path = os.path.join(OUTPUTS_DIR, f"{prefix}confusion_matrix.png")
    model_export_path = os.path.join(MODELS_DIR, f"{prefix}churn_model.pkl")

    print("\n[Step 3] Saving dataset-specific artifacts...")
    results_df.to_csv(results_csv_path, index=False)
    save_comparison_plot(results_df, comparison_plot_path)
    save_confusion_plot(
        bundle["confusion_matrix"],
        best_model_name,
        cm_plot_path,
        labels=(schema.negative_label, schema.positive_label),
    )

    joblib.dump(
        {
            "pipeline": bundle["best_pipeline"],
            "best_model_name": best_model_name,
            "dataset_key": dataset.key,
            "dataset_name": dataset.name,
            "source_file": os.path.abspath(csv_path),
            "feature_names": schema.feature_cols,
            "num_cols": schema.numeric_cols,
            "cat_cols": schema.categorical_cols,
            "metrics": bundle["best_metrics"],
            "confusion_matrix": bundle["confusion_matrix"].tolist(),
            "train_size": bundle["train_size"],
            "test_size": bundle["test_size"],
            "target_col": schema.target,
            "positive_label": schema.positive_label,
            "negative_label": schema.negative_label,
        },
        model_export_path,
    )

    print(f"  -> Metrics          : {results_csv_path}")
    print(f"  -> Comparison chart : {comparison_plot_path}")
    print(f"  -> Confusion matrix : {cm_plot_path}")
    print(f"  -> Model artifact   : {model_export_path}")
    print("\nTraining completed successfully.")
    print("Next: streamlit run app.py  (then upload this same CSV)\n")

    return {
        "dataset": dataset,
        "results_df": results_df,
        "best_model_name": best_model_name,
        "model_export_path": model_export_path,
        "results_csv_path": results_csv_path,
        "comparison_plot_path": comparison_plot_path,
        "cm_plot_path": cm_plot_path,
        "confusion_matrix": bundle["confusion_matrix"],
        "fitted_pipelines": bundle["pipelines"],
        "schema": schema,
        "frame": frame,
        "target": target,
    }


def print_supported_datasets() -> None:
    """Prints the two supported datasets and their download links."""
    print("\nSupported datasets (upload the CSV in the app or pass it with --data):")
    for dataset in SUPPORTED_DATASETS:
        print(f"\n  * {dataset.name}  [{dataset.expected_rows}]")
        print(f"    target: {' / '.join(dataset.target_candidates)}")
        for label, url in dataset.sources:
            print(f"    - {label}: {url}")
    print()


# ------------------------------------------------------------------------------
# 4. COMMAND LINE INTERFACE
# ------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train, benchmark and export churn classifiers for one supported telecom dataset.",
        epilog=(
            "Examples:\n"
            "  python train_model.py --data Telco-Customer-Churn.csv\n"
            "  python train_model.py --data cell2celltrain.csv --drop CustomerID\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to one supported churn dataset CSV (Telco or Cell2Cell).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Binary target column (auto-detected when omitted).",
    )
    parser.add_argument(
        "--drop",
        nargs="*",
        default=[],
        metavar="COLUMN",
        help="Extra feature columns to exclude (IDs, names, free text, ...).",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Artifact prefix (default: the detected dataset key, e.g. telco_).",
    )
    args = parser.parse_args(argv)

    try:
        train_from_csv(
            args.data,
            target_col=args.target,
            drop_cols=args.drop,
            output_prefix=args.out_prefix,
        )
    except DatasetError as error:
        print(f"\n[!] {error}\n", file=sys.stderr)
        print_supported_datasets()
        return 1
    except Exception as error:  # pragma: no cover - defensive CLI guard
        print(f"\n[!] Training failed: {error}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
