"""Temporary command-line demo for the quantum-inspired detector.

This script is meant for inspection only. It loads the synthetic transaction CSV,
uses the existing quantum-inspired detector, prints a summary, and highlights the
transactions with the highest anomaly scores.

The goal is to check whether the detector is identifying unusually different
transactions without building the dashboard or changing the model logic.
"""

from __future__ import annotations

import pandas as pd

from data_loader import load_transactions_csv
from detect.quantum_model import detect_quantum_anomalies


def main() -> None:
    """Run the quantum-inspired detector and inspect the most suspicious transactions."""
    csv_path = "data/transactions.csv"

    print("Loading transaction data...")
    raw_df = load_transactions_csv(csv_path)

    print("Running quantum-inspired anomaly detector...")
    results = detect_quantum_anomalies(raw_df)

    total_transactions = len(results)
    normal_count = (results["quantum_prediction_label"] == "NORMAL").sum()
    anomaly_count = (results["quantum_prediction_label"] == "ANOMALY").sum()
    min_score = results["quantum_anomaly_score"].min()
    max_score = results["quantum_anomaly_score"].max()
    mean_score = results["quantum_anomaly_score"].mean()

    print("\nSummary")
    print("-" * 120)
    print(f"Total transactions: {total_transactions}")
    print(f"Number of NORMAL predictions: {normal_count}")
    print(f"Number of ANOMALY predictions: {anomaly_count}")
    print(f"Minimum quantum anomaly score: {min_score:.12f}")
    print(f"Maximum quantum anomaly score: {max_score:.12f}")
    print(f"Mean quantum anomaly score: {mean_score:.12f}")

    print("\nTop 20 transactions by highest quantum anomaly score")
    print("-" * 120)
    top_results = results.sort_values(by="quantum_anomaly_score", ascending=False).head(20)
    display_columns = [
        "transaction_id",
        "amount",
        "location",
        "device_id",
        "quantum_anomaly_score",
        "quantum_prediction_label",
    ]
    print(top_results[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
