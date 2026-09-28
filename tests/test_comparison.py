import numpy as np
import pandas as pd

from generate_data import main as generate_dataset
from detect.comparison import compare_detectors


def test_compare_detectors_produces_expected_results():
    """Verify the comparison stage runs and creates the expected structure."""
    # Reset the canonical dataset so the comparison test is independent of earlier
    # live-transaction submissions in other tests.
    generate_dataset()

    results = compare_detectors()

    comparison_df = results["comparison_df"]

    assert not comparison_df.empty
    assert len(comparison_df) == 1000

    expected_columns = [
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

    for column in expected_columns:
        assert column in comparison_df.columns

    assert np.isfinite(comparison_df["conventional_anomaly_score"]).all()
    assert np.isfinite(comparison_df["quantum_anomaly_score"]).all()

    metrics = [
        results["conventional_metrics"],
        results["quantum_metrics"],
    ]

    for metric in metrics:
        assert set(metric.keys()) >= {"accuracy", "precision", "recall", "f1_score", "false_positives", "false_negatives"}
        assert np.isfinite(list(metric.values())[:4]).all()


def test_compare_detectors_accepts_label_ground_truth_column(tmp_path):
    """The evaluation layer should support the Qypher CSV naming convention using label."""
    transactions = pd.read_csv("data/transactions.csv")
    ground_truth = transactions[["transaction_id"]].copy()
    ground_truth["label"] = np.where(np.arange(len(ground_truth)) % 10 == 0, "anomaly", "normal")
    ground_truth_path = tmp_path / "ground_truth_with_label.csv"
    ground_truth.to_csv(ground_truth_path, index=False)

    results = compare_detectors(
        transactions_path="data/transactions.csv",
        ground_truth_path=str(ground_truth_path),
    )

    assert "comparison_df" in results
    assert "conventional_metrics" in results
    assert "quantum_metrics" in results


def test_analyze_transactions_returns_transaction_level_scores_for_all_rows(tmp_path):
    """The app analysis path should keep one anomaly score and action per transaction."""
    transactions = pd.read_csv("data/transactions.csv").head(30).copy()
    dataset_path = tmp_path / "qypher_transaction_dataset.csv"
    transactions.to_csv(dataset_path, index=False)

    analysis = __import__("app", fromlist=["analyze_transactions"]).analyze_transactions(str(dataset_path))

    assert len(analysis["monitor_df"]) == 30
    assert len(analysis["transaction_level_results"]) == 30
    assert {"transaction_id", "anomaly_score", "anomaly_status", "action"}.issubset(
        analysis["transaction_level_results"].columns
    )
