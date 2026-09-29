import numpy as np
import pandas as pd
import pytest

from generate_data import main as generate_dataset
from detect.comparison import compare_detectors


def test_compare_detectors_produces_expected_results():
    """Verify the comparison stage runs and creates the expected structure."""
    # Reset the benchmark files so evaluation is independent of earlier tests.
    generate_dataset()

    results = compare_detectors()

    comparison_df = results["comparison_df"]

    assert not comparison_df.empty
    assert len(comparison_df) == results["evaluation_transaction_count"]
    assert results["training_transaction_count"] + results["evaluation_transaction_count"] == 1000
    assert results["holdout_available"] is True
    assert len(results["evaluation_df"]) == results["ground_truth_matched_count"]
    assert set(results["evaluation_df"]["transaction_id"]).issubset(
        set(comparison_df["transaction_id"])
    )
    assert {
        "hour",
        "is_new_device",
        "distance_from_usual",
        "failed_attempts",
    }.issubset(results["evaluation_df"].columns)

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
    assert results["training_transaction_count"] + results["evaluation_transaction_count"] == len(transactions)


def test_metrics_ignore_transactions_without_matching_ground_truth(tmp_path):
    transactions = pd.read_csv("data/transactions.csv").head(5)
    transactions_path = tmp_path / "transactions.csv"
    transactions.to_csv(transactions_path, index=False)
    ground_truth = pd.DataFrame({
        "transaction_id": [transactions.iloc[0]["transaction_id"]],
        "label": ["anomaly"],
    })
    ground_truth_path = tmp_path / "ground_truth.csv"
    ground_truth.to_csv(ground_truth_path, index=False)

    results = compare_detectors(str(transactions_path), str(ground_truth_path))

    assert results["ground_truth_matched_count"] == 1
    assert results["evaluation_transaction_count"] == 1
    assert results["evaluation_df"]["transaction_id"].tolist() == [transactions.iloc[0]["transaction_id"]]
    for key in ("conventional_metrics", "quantum_metrics"):
        metrics = results[key]
        assert metrics["false_positives"] + metrics["false_negatives"] <= 1


@pytest.mark.parametrize("row_count", [10, 30, 100, 1000])
@pytest.mark.parametrize("model", ["isolation_forest", "qsvm"])
def test_analyze_transactions_returns_individual_results_for_every_row(tmp_path, row_count, model):
    """The CSV path must preserve one full result for every input transaction."""
    transactions = pd.read_csv("data/transactions.csv").head(row_count).copy()
    dataset_path = tmp_path / "qypher_transaction_dataset.csv"
    transactions.to_csv(dataset_path, index=False)

    analysis = __import__("app", fromlist=["analyze_transactions"]).analyze_transactions(
        str(dataset_path),
        model=model,
        cache_token=f"size-{row_count}",
    )

    assert analysis["input_row_count"] == row_count
    assert analysis["result_row_count"] == row_count
    assert len(analysis["monitor_df"]) == row_count
    assert len(analysis["transaction_level_results"]) == row_count
    assert analysis["monitor_df"]["transaction_id"].is_unique
    assert analysis["monitor_df"]["trust_score"].between(0, 100).all()
    assert analysis["monitor_df"]["risk_factors"].map(lambda factors: isinstance(factors, list)).all()
    assert analysis["monitor_df"]["model_key"].eq(model).all()
    assert {"transaction_id", "model_key", "detection", "trust_score", "decision", "risk_factors"}.issubset(
        analysis["transaction_level_results"].columns
    )
    selected_factor = "isolation_forest" if model == "isolation_forest" else "quantum_inspired"
    unselected_factor = "quantum_inspired" if model == "isolation_forest" else "isolation_forest"
    for _, row in analysis["monitor_df"].iterrows():
        factor_keys = {factor["key"] for factor in row["risk_factors"]}
        assert (selected_factor in factor_keys) == (row["detection"] == "ANOMALY")
        assert unselected_factor not in factor_keys
    assert {"transaction_id", "detection_method", "model_score", "risk_level", "risk_factors"}.issubset(
        analysis["monitor_df"].columns
    )


def test_new_transaction_entry_ui_is_removed():
    import inspect
    import app

    assert not hasattr(app, "render_new_transaction_form")
    assert "New Transaction" not in inspect.getsource(app.main)
