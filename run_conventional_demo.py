"""Simple command-line demo for the conventional transaction anomaly detector.

This script loads the generated transaction CSV, preprocesses the data with the
existing loader functions, and runs the current Isolation Forest implementation.
It then prints:
- the first 20 transaction results in a readable table,
- the total number of transactions,
- how many were classified as NORMAL and ANOMALY,
- the 10 transactions with the highest anomaly scores.

This is intentionally a simple demo only; it does not build the UI or quantum detector.
"""

from __future__ import annotations

import pandas as pd

from data_loader import load_transactions_csv, get_features_for_anomaly_detection
from detect.conventional_model import detect_conventional_anomalies


def main() -> None:
    """Run the conventional anomaly detection demo."""
    csv_path = "data/transactions.csv"

    print("Loading transaction data...")
    raw_df = load_transactions_csv(csv_path)

    print("Preprocessing transaction data...")
    prepared = get_features_for_anomaly_detection(raw_df)

    print("Running conventional Isolation Forest detector...")
    results = detect_conventional_anomalies(raw_df)

    print("\nFirst 20 transactions")
    print("-" * 120)
    display_columns = [
        "transaction_id",
        "amount",
        "location",
        "device_id",
        "anomaly_score",
        "prediction_label",
    ]
    print(results[display_columns].head(20).to_string(index=False))

    total_transactions = len(results)
    normal_count = (results["prediction_label"] == "NORMAL").sum()
    anomaly_count = (results["prediction_label"] == "ANOMALY").sum()

    print("\nSummary")
    print("-" * 120)
    print(f"Total transactions: {total_transactions}")
    print(f"Number classified as NORMAL: {normal_count}")
    print(f"Number classified as ANOMALY: {anomaly_count}")

    print("\nTop 10 highest-risk transactions")
    print("-" * 120)
    top_risk = results.sort_values(by="anomaly_score", ascending=False).head(10)
    print(top_risk[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
