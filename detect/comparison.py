"""Fair comparison of Qypher detectors on a shared labelled holdout."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import get_features_for_anomaly_detection, load_transactions_csv
from detect.conventional_model import ConventionalAnomalyDetector, detect_conventional_anomalies
from detect.quantum_model import QuantumInspiredDetector, detect_quantum_anomalies


HOLDOUT_FRACTION = 0.25
RANDOM_STATE = 42
VALID_LABELS = {"NORMAL", "ANOMALY"}
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
    if prediction_value == 1:
        return "NORMAL"
    if prediction_value == -1:
        return "ANOMALY"
    return "UNKNOWN"


def _compute_confusion_values(actual_labels: pd.Series, predicted_labels: pd.Series) -> dict[str, int]:
    """Count confusion-matrix outcomes with ANOMALY as the positive class."""
    return {
        "true_positives": int(((actual_labels == "ANOMALY") & (predicted_labels == "ANOMALY")).sum()),
        "true_negatives": int(((actual_labels == "NORMAL") & (predicted_labels == "NORMAL")).sum()),
        "false_positives": int(((actual_labels == "NORMAL") & (predicted_labels == "ANOMALY")).sum()),
        "false_negatives": int(((actual_labels == "ANOMALY") & (predicted_labels == "NORMAL")).sum()),
    }


def _calculate_metrics(actual_labels: pd.Series, predicted_labels: pd.Series) -> dict[str, float]:
    confusion = _compute_confusion_values(actual_labels, predicted_labels)
    tp = confusion["true_positives"]
    tn = confusion["true_negatives"]
    fp = confusion["false_positives"]
    fn = confusion["false_negatives"]
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        **confusion,
    }


def _warn_if_problematic_detector(label_series: pd.Series, detector_name: str) -> None:
    unique_labels = set(label_series.dropna().unique())
    if len(unique_labels) == 1:
        print(
            f"Warning: {detector_name} classified every evaluation transaction as "
            f"{next(iter(unique_labels))}. Review the detector threshold and evaluation data."
        )


def run_detector_and_measure(
    detector_func,
    raw_df: pd.DataFrame,
    detector_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Retain the original full-data runner for standalone detector reports."""
    started = time.perf_counter()
    results = detector_func(raw_df)
    elapsed = time.perf_counter() - started
    label_column = "prediction_label" if "prediction_label" in results else "quantum_prediction_label"
    score_column = next(column for column in results if "score" in column)
    summary = {
        "detector_name": detector_name,
        "execution_time_seconds": float(elapsed),
        "total_predictions": int(len(results)),
        "normal_count": int(results[label_column].eq("NORMAL").sum()),
        "anomaly_count": int(results[label_column].eq("ANOMALY").sum()),
        "min_score": float(results[score_column].min()),
        "max_score": float(results[score_column].max()),
        "mean_score": float(results[score_column].mean()),
    }
    return results, summary


def _normalize_ground_truth_labels(ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """Accept the current and legacy label column names and normalize values."""
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
    if "transaction_id" not in normalized.columns:
        raise ValueError("Ground-truth CSV must contain a 'transaction_id' column.")

    labels = normalized["label"].astype("string").str.strip().str.upper()
    normalized["label"] = labels.replace({
        "ANOMALY": "ANOMALY",
        "FRAUD": "ANOMALY",
        "1": "ANOMALY",
        "NORMAL": "NORMAL",
        "LEGITIMATE": "NORMAL",
        "0": "NORMAL",
    })
    normalized["transaction_id"] = normalized["transaction_id"].astype(str)
    if normalized["transaction_id"].duplicated().any():
        raise ValueError("Ground-truth transaction_id values must be unique.")
    return normalized


def _load_ground_truth(raw_df: pd.DataFrame, ground_truth_path: str) -> pd.DataFrame:
    if "label" in raw_df.columns or "anomaly_label" in raw_df.columns:
        label_column = "label" if "label" in raw_df.columns else "anomaly_label"
        return _normalize_ground_truth_labels(
            raw_df[["transaction_id", label_column]].rename(columns={label_column: "label"})
        )
    if not Path(ground_truth_path).is_file():
        return pd.DataFrame(columns=["transaction_id", "label"])
    try:
        return _normalize_ground_truth_labels(pd.read_csv(ground_truth_path))
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["transaction_id", "label"])


def _shared_holdout_positions(
    raw_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Select one reproducible evaluation sample shared by both detectors."""
    raw_ids = raw_df["transaction_id"].astype(str)
    labelled = ground_truth_df[ground_truth_df["label"].isin(VALID_LABELS)]
    matching_ids = set(labelled["transaction_id"].astype(str))
    labelled_positions = np.flatnonzero(raw_ids.isin(matching_ids).to_numpy())
    candidate_positions = labelled_positions if len(labelled_positions) else np.arange(len(raw_df))

    if len(candidate_positions) >= 2:
        test_count = min(
            len(candidate_positions) - 1,
            max(1, int(np.ceil(len(candidate_positions) * HOLDOUT_FRACTION))),
        )
        label_by_id = labelled.set_index("transaction_id")["label"]
        stratify = None
        if len(labelled_positions):
            candidate_labels = raw_ids.iloc[candidate_positions].map(label_by_id)
            class_counts = candidate_labels.value_counts()
            class_count = len(class_counts)
            if (
                class_count > 1
                and class_counts.min() >= 2
                and test_count >= class_count
                and len(candidate_positions) - test_count >= class_count
            ):
                stratify = candidate_labels.to_numpy()
        _, test_positions = train_test_split(
            candidate_positions,
            test_size=test_count,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=stratify,
        )
        test_positions = np.sort(test_positions)
        train_positions = np.setdiff1d(np.arange(len(raw_df)), test_positions)
        return train_positions, test_positions, True

    test_positions = candidate_positions
    train_positions = np.setdiff1d(np.arange(len(raw_df)), test_positions)
    if len(train_positions) == 0:
        train_positions = test_positions
        return train_positions, test_positions, False
    return train_positions, test_positions, True


def _prediction_summary(
    results: pd.DataFrame,
    detector_name: str,
    label_column: str,
    score_column: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    scores = pd.to_numeric(results[score_column], errors="coerce")
    return {
        "detector_name": detector_name,
        "execution_time_seconds": float(elapsed_seconds),
        "total_predictions": int(len(results)),
        "normal_count": int(results[label_column].eq("NORMAL").sum()),
        "anomaly_count": int(results[label_column].eq("ANOMALY").sum()),
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "mean_score": float(scores.mean()),
    }


def compare_detectors(
    transactions_path: str = "data/transactions.csv",
    ground_truth_path: str = "data/ground_truth.csv",
) -> dict[str, Any]:
    """Fit both unsupervised models on one training split and evaluate a shared holdout."""
    raw_df = load_transactions_csv(transactions_path)
    if raw_df.empty:
        raise ValueError("At least one transaction is required for detector comparison.")
    if raw_df["transaction_id"].astype(str).duplicated().any():
        raise ValueError("Transaction IDs must be unique for model comparison.")

    ground_truth_df = _load_ground_truth(raw_df, ground_truth_path)
    train_positions, test_positions, holdout_available = _shared_holdout_positions(
        raw_df,
        ground_truth_df,
    )
    prepared = get_features_for_anomaly_detection(raw_df)
    feature_matrix = prepared["features"].select_dtypes(include=["number"])
    transaction_data = prepared["transaction_data"]
    train_features = feature_matrix.iloc[train_positions]
    test_features = feature_matrix.iloc[test_positions]
    test_transactions = transaction_data.iloc[test_positions]

    started = time.perf_counter()
    conventional_model = ConventionalAnomalyDetector(contamination=0.08, random_state=RANDOM_STATE)
    conventional_model.fit(train_features)
    conventional_results = conventional_model.predict_transactions(test_features, test_transactions)
    conventional_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    quantum_model = QuantumInspiredDetector(similarity_threshold=0.4, random_state=RANDOM_STATE)
    quantum_model.fit(train_features)
    quantum_results = quantum_model.predict_transactions(test_features, test_transactions)
    quantum_elapsed = time.perf_counter() - started

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
    for prediction_df in (conventional_results, quantum_results):
        prediction_df["transaction_id"] = prediction_df["transaction_id"].astype(str)

    comparison_df = conventional_results.merge(
        quantum_results,
        on=["transaction_id", "amount", "location", "device_id"],
        how="inner",
        validate="one_to_one",
    )[REQUIRED_COMPARISON_COLUMNS]
    if len(comparison_df) != len(test_positions):
        raise ValueError("The two detectors did not return aligned holdout predictions.")

    evaluation_df = comparison_df.merge(
        ground_truth_df[["transaction_id", "label"]],
        on="transaction_id",
        how="inner",
        validate="one_to_one",
    )
    evaluation_df = evaluation_df[evaluation_df["label"].isin(VALID_LABELS)].copy()
    explanation_columns = [
        column for column in (
            "transaction_id",
            "hour",
            "is_new_device",
            "distance_from_usual",
            "failed_attempts",
            "merchant_category",
            "payment_method",
        ) if column in raw_df.columns
    ]
    if len(explanation_columns) > 1:
        evaluation_df = evaluation_df.merge(
            raw_df[explanation_columns],
            on="transaction_id",
            how="left",
            validate="one_to_one",
        )
    actual_labels = evaluation_df["label"]
    conventional_predictions = evaluation_df["conventional_label"].astype(str).str.upper()
    quantum_predictions = evaluation_df["quantum_label"].astype(str).str.upper()
    conventional_metrics = _calculate_metrics(actual_labels, conventional_predictions)
    quantum_metrics = _calculate_metrics(actual_labels, quantum_predictions)

    conventional_summary = _prediction_summary(
        conventional_results,
        "Conventional Isolation Forest",
        "conventional_label",
        "conventional_anomaly_score",
        conventional_elapsed,
    )
    quantum_summary = _prediction_summary(
        quantum_results,
        "Quantum-inspired detector",
        "quantum_label",
        "quantum_anomaly_score",
        quantum_elapsed,
    )
    _warn_if_problematic_detector(conventional_results["conventional_label"], "Isolation Forest")
    _warn_if_problematic_detector(quantum_results["quantum_label"], "Quantum-inspired detector")

    return {
        "comparison_df": comparison_df,
        "conventional_summary": conventional_summary,
        "quantum_summary": quantum_summary,
        "conventional_metrics": conventional_metrics,
        "quantum_metrics": quantum_metrics,
        "evaluation_df": evaluation_df,
        "ground_truth_size": int(len(ground_truth_df)),
        "ground_truth_matched_count": int(len(evaluation_df)),
        "training_transaction_count": int(len(train_positions)),
        "evaluation_transaction_count": int(len(test_positions)),
        "holdout_available": holdout_available,
    }


def print_comparison_summary(results: dict[str, Any]) -> None:
    """Print a readable report for the terminal demo."""
    print("\nDetect Comparison Summary")
    print("-" * 100)
    print(f"Training transactions: {results['training_transaction_count']}")
    print(f"Evaluation transactions: {results['evaluation_transaction_count']}")
    print(f"Labelled evaluation transactions: {results['ground_truth_matched_count']}")

    for title, summary_key, metrics_key in (
        ("Conventional Isolation Forest", "conventional_summary", "conventional_metrics"),
        ("Quantum-inspired detector", "quantum_summary", "quantum_metrics"),
    ):
        summary = results[summary_key]
        metrics = results[metrics_key]
        print(f"\n{title}")
        print(f"  - Anomalies detected: {summary['anomaly_count']}")
        print(f"  - Execution time: {summary['execution_time_seconds']:.6f}s")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall: {metrics['recall']:.4f}")
        print(f"  - F1: {metrics['f1_score']:.4f}")
        print(f"  - False positives: {metrics['false_positives']}")
        print(f"  - False negatives: {metrics['false_negatives']}")

    print("\nShared holdout predictions")
    print(results["comparison_df"].head(10).to_string(index=False))


if __name__ == "__main__":
    print_comparison_summary(compare_detectors())
