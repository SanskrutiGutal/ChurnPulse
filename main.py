"""
================================================================================
Mobile Phone Customer Churn Classification Using Machine Learning
================================================================================
File: main.py
Description: Master CLI entry point. Trains, benchmarks and exports the four
             classifiers for one of the two supported telecom churn datasets,
             then runs a sample prediction using the first row of that same file.

             The web application is the primary interface:
                 streamlit run app.py

Usage:
    python main.py --data data/Telco-Customer-Churn.csv
    python main.py --data cell2celltrain.csv
    python main.py --data cell2celltrain.csv --drop CustomerID
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from train_model import train_from_csv
from dataset_manager import DatasetError


def run_sample_inference(summary: dict) -> None:
    """
    Scores the first row of the uploaded dataset with the exported pipeline.
    Nothing is hard-coded: feature names, labels and classes come from the
    schema inferred for that file.
    """
    frame = summary["frame"]
    schema = summary["schema"]
    pipeline = summary["fitted_pipelines"][summary["best_model_name"]]

    row = frame.loc[[frame.index[0]], schema.feature_cols]
    pred = int(pipeline.predict(row)[0])

    probability = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probability = float(pipeline.predict_proba(row)[0][1])
        except Exception:
            probability = None

    print("\n" + "=" * 76)
    print("SAMPLE PREDICTION (first row of the uploaded CSV)")
    print(f"Model: {summary['best_model_name']} — {summary['dataset'].name}")
    print("=" * 76)

    actual_raw = frame.iloc[0][schema.target]
    print(f"  Actual target value in the CSV : {actual_raw}")
    print(f"  Predicted class                : {schema.positive_label if pred == 1 else schema.negative_label}")
    if probability is not None:
        print(f"  Probability of {schema.positive_label:<14}: {probability * 100:.2f}%")
    print("  (A single row is only a smoke test — use the app's Prediction page for real scenarios.)")
    print("=" * 76)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train, benchmark and export churn classifiers for one supported telecom dataset.",
        epilog=(
            "Examples:\n"
            "  python main.py --data data/Telco-Customer-Churn.csv\n"
            "  python main.py --data cell2celltrain.csv\n"
            "  python main.py --data cell2celltrain.csv --drop CustomerID\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to one supported churn dataset CSV (Telco or Cell2Cell).",
    )
    parser.add_argument("--target", default=None, help="Binary target column (auto-detected when omitted).")
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
    args = parser.parse_args()

    print("=" * 76)
    print(" MOBILE PHONE CUSTOMER CHURN CLASSIFICATION USING MACHINE LEARNING")
    print("=" * 76)
    print("Pipeline:")
    print("  1. Data validation   : dataset identification + binary target check")
    print("  2. Preprocessing     : median/mode imputation, scaling, one-hot encoding")
    print("                         (fitted on the training split only — no leakage)")
    print("  3. Models            : Logistic Regression, Decision Tree, KNN, Gaussian NB")
    print("  4. Evaluation        : Accuracy, Precision, Recall, F1 Score (80/20, seed 42)")
    print("  5. Deployment        : dataset-specific .pkl pipeline + Streamlit app")
    print("=" * 76)

    if not os.path.exists(args.data):
        print(f"\n[!] File not found: {args.data}", file=sys.stderr)
        print("Download a supported dataset first (links in README.md).\n", file=sys.stderr)
        return 1

    try:
        summary = train_from_csv(
            args.data,
            target_col=args.target,
            drop_cols=args.drop,
            output_prefix=args.out_prefix,
        )
    except DatasetError as error:
        print(f"\n[!] {error}\n", file=sys.stderr)
        return 1
    except Exception as error:  # pragma: no cover - defensive CLI guard
        print(f"\n[!] Training failed: {error}\n", file=sys.stderr)
        return 1

    run_sample_inference(summary)

    print("\n" + "=" * 76)
    print(" NEXT STEPS")
    print("=" * 76)
    print("1. Launch the interactive Streamlit app:")
    print("   streamlit run app.py")
    print("2. Upload the same CSV with the single upload box on the welcome screen:")
    print("   Upload Dataset (CSV)")
    print("\nArtifacts generated for this dataset:")
    print(f"   - Metrics table   : {summary['results_csv_path']}")
    print(f"   - Accuracy chart  : {summary['comparison_plot_path']}")
    print(f"   - Confusion matrix: {summary['cm_plot_path']}")
    print(f"   - Serialized model: {summary['model_export_path']}")
    print("=" * 76 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
