import numpy as np
import pandas as pd

from detect.quantum_model import detect_quantum_anomalies


def test_quantum_detector_runs_and_outputs_valid_results():
    """Verify the quantum-inspired detector produces a prediction and score for every transaction."""
    transactions = pd.read_csv("data/transactions.csv")

    results = detect_quantum_anomalies(transactions)

    assert not results.empty
    assert len(results) == len(transactions)

    required_columns = [
        "transaction_id",
        "amount",
        "location",
        "device_id",
        "quantum_anomaly_score",
        "quantum_prediction",
        "quantum_prediction_label",
    ]

    for column in required_columns:
        assert column in results.columns, f"Missing expected column: {column}"

    assert all(pd.notna(results["quantum_anomaly_score"]))
    assert all(pd.notna(results["quantum_prediction"]))
    assert all(pd.notna(results["quantum_prediction_label"]))
    assert np.isfinite(results["quantum_anomaly_score"]).all()

    valid_predictions = {-1, 1}
    assert set(results["quantum_prediction"].unique()).issubset(valid_predictions)

    valid_labels = {"NORMAL", "ANOMALY"}
    assert set(results["quantum_prediction_label"].unique()).issubset(valid_labels)
