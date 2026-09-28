"""Compare the conventional and quantum-inspired anomaly detectors.

This stage is intentionally simple and beginner-friendly. It runs both detectors on
exactly the same transaction data, aligns their results by transaction_id, and then
compares their predictions against the known ground truth labels for evaluation.

The goal is to show how the two detectors behave differently on the same dataset,
without changing either algorithm.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from data_loader import load_transactions_csv, get_features_for_anomaly_detection
from detect.conventional_model import detect_conventional_anomalies
from detect.quantum_model import detect_quantum_anomalies


REQUIRED_COMPARISON_COLUMNS = [
    "transaction_id",
    "amount",
    "location",
    "device_id",
    "conventional_anomaly_score",
    "conventional_prediction",
    "conventional_label",
    "quantum_anomaly_score",
    "quantum_prediction",
    "quantum_label",
]


def _safe_label_from_prediction(prediction_value: int) -> str:
    """Convert numeric predictions to labels used by the UI and reports."""
    if prediction_value == 1:
        return "NORMAL"
    if prediction_value == -1:
        return "ANOMALY"
    return "UNKNOWN"


def _compute_confusion_values(actual_labels: pd.Series, predicted_labels: pd.Series) -> Dict[str, int]:
    """Compute basic confusion-matrix counts for a binary-like problem.

    We treat the anomaly class as positive and the normal class as negative.
    """
    tp = int(((actual_labels == "ANOMALY") & (predicted_labels == "ANOMALY")).sum())
    tn = int(((actual_labels == "NORMAL") & (predicted_labels == "NORMAL")).sum())
    fp = int(((actual_labels == "NORMAL") & (predicted_labels == "ANOMALY")).sum())
    fn = int(((actual_labels == "ANOMALY") & (predicted_labels == "NORMAL")).sum())

    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _calculate_metrics(actual_labels: pd.Series, predicted_labels: pd.Series) -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        actual_labels: True labels from ground_truth.csv.
        predicted_labels: Labels produced by a detector.

    Returns:
        A dictionary with accuracy, precision, recall, and F1.
    """
    confusion = _compute_confusion_values(actual_labels, predicted_labels)

    tp = confusion["true_positives"]
    tn = confusion["true_negatives"]
    fp = confusion["false_positives"]
    fn = confusion["false_negatives"]

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positives": fp,
        "false_negatives": fn,
    }


def _warn_if_problematic_detector(label_series: pd.Series, detector_name: str) -> None:
    """Report a clear warning if a detector is classifying everything the same way."""
    unique_labels = set(label_series.unique())
    if len(unique_labels) == 1:
        print(
            f"Warning: {detector_name} classified every transaction as {next(iter(unique_labels))}. "
            "This may indicate a threshold or score-sensitivity issue that should be reviewed."
        )


def run_detector_and_measure(detector_func, raw_df: pd.DataFrame, detector_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run one detector and measure the execution time.

    Args:
        detector_func: The detector function to call.
        raw_df: Transaction data.
        detector_name: Human-readable name for the detector.

    Returns:
        A DataFrame with predictions and a metadata dictionary with timing and summary info.
    """
    start_time = time.perf_counter()
    results = detector_func(raw_df)
    elapsed_seconds = time.perf_counter() - start_time

    # Keep the result format easy to inspect in the console and in tests.
    summary = {
        "detector_name": detector_name,
        "execution_time_seconds": float(elapsed_seconds),
        "total_predictions": int(len(results)),
        "normal_count": int((results["prediction_label"] if "prediction_label" in results.columns else results["quantum_prediction_label"]).eq("NORMAL").sum()),
        "anomaly_count": int((results["prediction_label"] if "prediction_label" in results.columns else results["quantum_prediction_label"]).eq("ANOMALY").sum()),
        "min_score": float(results[[col for col in results.columns if "score" in col][0]].min()),
        "max_score": float(results[[col for col in results.columns if "score" in col][0]].max()),
        "mean_score": float(results[[col for col in results.columns if "score" in col][0]].mean()),
    }

    return results, summary


def _normalize_ground_truth_labels(ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the label column name to a single internal format.

    Qypher uses "label" for the evaluation CSV, while older code and generated data
    sometimes used "anomaly_label". We accept both names and normalize to "label"
    so the project works without user-side CSV edits.
    """
    normalized = ground_truth_df.copy()

    if "label" not in normalized.columns and "anomaly_label" in normalized.columns:
        normalized = normalized.rename(columns={"anomaly_label": "label"})
    elif "label" in normalized.columns and "anomaly_label" in normalized.columns:
        normalized = normalized.drop(columns=["anomaly_label"])

    if "label" not in normalized.columns:
        raise ValueError(
            "Ground-truth CSV must contain a 'label' or 'anomaly_label' column. "
            f"Columns found: {list(normalized.columns)}"
        )

    return normalized


def compare_detectors(
    transactions_path: str = "data/transactions.csv",
    ground_truth_path: str = "data/ground_truth.csv",
) -> Dict[str, Any]:
    """Run both detectors and compare their results against the known labels.

    This is the public function app.py can call later.

    Returns:
        A dictionary containing:
        - comparison_df
        - conventional_summary
        - quantum_summary
        - conventional_metrics
        - quantum_metrics
        - ground_truth_size
    """
    raw_df = load_transactions_csv(transactions_path)
    ground_truth_df = _normalize_ground_truth_labels(pd.read_csv(ground_truth_path))

    conventional_results, conventional_summary = run_detector_and_measure(
        detect_conventional_anomalies,
        raw_df,
        "Conventional Isolation Forest",
    )
    quantum_results, quantum_summary = run_detector_and_measure(
        detect_quantum_anomalies,
        raw_df,
        "Quantum-inspired detector",
    )

    conventional_results = conventional_results.rename(columns={
        "anomaly_score": "conventional_anomaly_score",
        "prediction": "conventional_prediction",
        "prediction_label": "conventional_label",
    })
    quantum_results = quantum_results.rename(columns={
        "quantum_anomaly_score": "quantum_anomaly_score",
        "quantum_prediction": "quantum_prediction",
        "quantum_prediction_label": "quantum_label",
    })

    conventional_results["transaction_id"] = conventional_results["transaction_id"].astype(str)
    quantum_results["transaction_id"] = quantum_results["transaction_id"].astype(str)
    ground_truth_df["transaction_id"] = ground_truth_df["transaction_id"].astype(str)

    merged = conventional_results.merge(
        quantum_results,
        on=["transaction_id", "amount", "location", "device_id"],
        how="inner",
    )

    # Keep only the expected columns for the combined comparison output.
    merged = merged[REQUIRED_COMPARISON_COLUMNS]

    # Merge with the ground truth only for evaluation.
    evaluation_df = merged.merge(
        ground_truth_df[["transaction_id", "label"]],
        on="transaction_id",
        how="left",
    )

    actual_labels = evaluation_df["label"].str.upper().fillna("NORMAL")

    conventional_pred_labels = evaluation_df["conventional_label"].astype(str).str.upper()
    quantum_pred_labels = evaluation_df["quantum_label"].astype(str).str.upper()

    conventional_metrics = _calculate_metrics(actual_labels, conventional_pred_labels)
    quantum_metrics = _calculate_metrics(actual_labels, quantum_pred_labels)

    _warn_if_problematic_detector(conventional_pred_labels, "Conventional Isolation Forest")
    _warn_if_problematic_detector(quantum_pred_labels, "Quantum-inspired detector")

    return {
        "comparison_df": merged,
        "conventional_summary": conventional_summary,
        "quantum_summary": quantum_summary,
        "conventional_metrics": conventional_metrics,
        "quantum_metrics": quantum_metrics,
        "ground_truth_size": int(len(ground_truth_df)),
    }


def print_comparison_summary(results: Dict[str, Any]) -> None:
    """Print a simple report for the terminal demo."""
    comparison_df = results["comparison_df"]
    conventional_summary = results["conventional_summary"]
    quantum_summary = results["quantum_summary"]
    conventional_metrics = results["conventional_metrics"]
    quantum_metrics = results["quantum_metrics"]

    print("\nDetect Comparison Summary")
    print("-" * 120)
    print(f"Transactions compared: {len(comparison_df)}")
    print(f"Ground truth rows loaded: {results['ground_truth_size']}")

    print("\nConventional detector")
    print(f"  - Predictions: {conventional_summary['total_predictions']}")
    print(f"  - NORMAL: {conventional_summary['normal_count']}")
    print(f"  - ANOMALY: {conventional_summary['anomaly_count']}")
    print(f"  - Execution time: {conventional_summary['execution_time_seconds']:.6f}s")
    print(f"  - Accuracy: {conventional_metrics['accuracy']:.4f}")
    print(f"  - Precision: {conventional_metrics['precision']:.4f}")
    print(f"  - Recall: {conventional_metrics['recall']:.4f}")
    print(f"  - F1: {conventional_metrics['f1_score']:.4f}")
    print(f"  - False positives: {conventional_metrics['false_positives']}")
    print(f"  - False negatives: {conventional_metrics['false_negatives']}")

    print("\nQuantum-inspired detector")
    print(f"  - Predictions: {quantum_summary['total_predictions']}")
    print(f"  - NORMAL: {quantum_summary['normal_count']}")
    print(f"  - ANOMALY: {quantum_summary['anomaly_count']}")
    print(f"  - Execution time: {quantum_summary['execution_time_seconds']:.6f}s")
    print(f"  - Accuracy: {quantum_metrics['accuracy']:.4f}")
    print(f"  - Precision: {quantum_metrics['precision']:.4f}")
    print(f"  - Recall: {quantum_metrics['recall']:.4f}")
    print(f"  - F1: {quantum_metrics['f1_score']:.4f}")
    print(f"  - False positives: {quantum_metrics['false_positives']}")
    print(f"  - False negatives: {quantum_metrics['false_negatives']}")

    print("\nTop 10 highest-risk conventional transactions")
    conventional_top = comparison_df.sort_values(
        by="conventional_anomaly_score",
        ascending=False,
    ).head(10)
    print(conventional_top[[
        "transaction_id",
        "amount",
        "location",
        "device_id",
        "conventional_anomaly_score",
        "conventional_label",
    ]].to_string(index=False))

    print("\nTop 10 highest-risk quantum transactions")
    quantum_top = comparison_df.sort_values(
        by="quantum_anomaly_score",
        ascending=False,
    ).head(10)
    print(quantum_top[[
        "transaction_id",
        "amount",
        "location",
        "device_id",
        "quantum_anomaly_score",
        "quantum_label",
    ]].to_string(index=False))


if __name__ == "__main__":
    print_comparison_summary(compare_detectors())
