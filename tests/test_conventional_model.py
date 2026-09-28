import pandas as pd

from detect.conventional_model import detect_conventional_anomalies


def test_conventional_detector_runs_and_outputs_prediction_for_every_transaction():
    """Verify the conventional model runs and produces a result for every record."""
    transactions = pd.read_csv("data/transactions.csv")

    results = detect_conventional_anomalies(transactions)

    assert not results.empty
    assert len(results) == len(transactions)
    assert "transaction_id" in results.columns
    assert "anomaly_score" in results.columns
    assert "prediction" in results.columns
    assert "prediction_label" in results.columns

    assert set(results["prediction"].unique()).issubset({-1, 1})
    assert set(results["prediction_label"].unique()).issubset({"NORMAL", "ANOMALY"})

    # Confirm both possible labels can be present in the dataset.
    assert {"NORMAL", "ANOMALY"}.issubset(set(results["prediction_label"].unique()))
