"""Generate a synthetic transaction dataset for the Qypher prototype.

This script creates a realistic dataset of 1,000 transactions for a financial
security demo. Most records are normal, while a smaller set are intentionally
anomalous. The dataset is saved to the data/ folder so it can be used later by
ML and detection components.
"""

import os
import random
from datetime import datetime, timedelta

import pandas as pd


SEED = 42
TOTAL_TRANSACTIONS = 1000
ANOMALY_COUNT = 80
BASE_DATE = datetime(2026, 1, 15)


def build_normal_locations():
    """Locations commonly used by normal transactions."""
    return [
        "New York",
        "Chicago",
        "Boston",
        "Seattle",
        "Austin",
        "Denver",
        "Los Angeles",
        "Miami",
        "San Francisco",
        "Dallas",
        "Atlanta",
        "Phoenix",
    ]


def build_anomalous_locations():
    """Locations that are less common for a typical user profile."""
    return [
        "Tokyo",
        "Dubai",
        "Paris",
        "Singapore",
        "Mexico City",
        "Rio de Janeiro",
        "Berlin",
        "Lagos",
        "Toronto",
        "London",
    ]


def build_merchant_categories():
    """Common merchant categories for everyday spending."""
    return [
        "Grocery",
        "Coffee Shop",
        "Dining",
        "Fuel",
        "Retail",
        "Pharmacy",
        "Entertainment",
        "Streaming",
        "Travel",
        "Utilities",
    ]


def build_anomalous_categories():
    """Categories that can signal abnormal or risky spending."""
    return [
        "Luxury Retail",
        "Cash Advance",
        "International Travel",
        "Electronics",
        "Digital Goods",
        "Gambling",
        "Crypto Exchange",
        "High Risk Merchant",
    ]


def build_payment_methods():
    """Common payment choices in a normal customer profile."""
    return ["Credit Card", "Debit Card", "Digital Wallet", "Bank Transfer"]


def build_transaction_types():
    """Typical transaction kinds seen in daily activity."""
    return ["purchase", "transfer", "refund", "bill_payment", "withdrawal"]


def generate_normal_transaction(index, device_ids, device_usual_locations):
    """Create one realistic normal transaction."""
    random_hour = random.randint(0, 23)
    transaction_time = BASE_DATE + timedelta(
        hours=random_hour,
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    device_id = random.choice(device_ids)
    location = device_usual_locations[device_id]

    merchant_category = random.choice(build_merchant_categories())
    payment_method = random.choice(build_payment_methods())
    transaction_type = random.choice(build_transaction_types())

    # Most normal transactions happen in common ranges and at standard times.
    base_amount = random.uniform(10, 180)
    if merchant_category in ["Travel", "Utilities", "Entertainment"]:
        base_amount *= 1.6

    amount = round(base_amount, 2)
    is_new_device = random.random() < 0.03
    distance_from_usual = round(random.uniform(0, 30), 2)

    # Most failed attempts are zero; occasional retries happen normally.
    failed_attempts = 0
    if random.random() < 0.08:
        failed_attempts = random.randint(1, 2)

    return {
        "transaction_id": f"TXN{index:06d}",
        "timestamp": transaction_time.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": amount,
        "location": location,
        "merchant_category": merchant_category,
        "payment_method": payment_method,
        "device_id": device_id,
        "transaction_type": transaction_type,
        "hour": random_hour,
        "is_new_device": is_new_device,
        "distance_from_usual": distance_from_usual,
        "failed_attempts": failed_attempts,
    }


def generate_anomalous_transaction(index, device_ids, device_usual_locations):
    """Create a realistic abnormal transaction with several risky signals together."""
    random_hour = random.choice([0, 1, 2, 3, 4, 5, 6, 23])
    transaction_time = BASE_DATE + timedelta(
        hours=random_hour,
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    device_id = random.choice(device_ids)
    usual_location = device_usual_locations[device_id]
    unusual_location = random.choice(build_anomalous_locations())

    # Use a risky merchant category and uncommon location together.
    merchant_category = random.choice(build_anomalous_categories())
    payment_method = random.choice(["Credit Card", "Digital Wallet"])
    transaction_type = random.choice(["purchase", "withdrawal", "transfer", "cash_advance"])

    # Larger amounts are a common indicator of suspicious behavior.
    amount = round(random.uniform(700, 4500), 2)
    is_new_device = True
    distance_from_usual = round(random.uniform(350, 2200), 2)
    failed_attempts = random.randint(2, 7)

    return {
        "transaction_id": f"TXN{index:06d}",
        "timestamp": transaction_time.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": amount,
        "location": unusual_location,
        "merchant_category": merchant_category,
        "payment_method": payment_method,
        "device_id": device_id,
        "transaction_type": transaction_type,
        "hour": random_hour,
        "is_new_device": is_new_device,
        "distance_from_usual": distance_from_usual,
        "failed_attempts": failed_attempts,
    }


def main():
    """Generate the transaction CSV and the evaluation ground truth CSV."""
    random.seed(SEED)

    # Create the data folder automatically if it does not already exist.
    os.makedirs("data", exist_ok=True)

    device_ids = [f"DEV-{i:04d}" for i in range(1, 101)]
    device_usual_locations = {
        device_id: random.choice(build_normal_locations()) for device_id in device_ids
    }

    # Use a fixed sample of anomaly rows so the dataset is repeatable.
    anomaly_indices = random.sample(range(TOTAL_TRANSACTIONS), ANOMALY_COUNT)

    records = []
    ground_truth = []

    for index in range(TOTAL_TRANSACTIONS):
        if index in anomaly_indices:
            record = generate_anomalous_transaction(index + 1, device_ids, device_usual_locations)
            label = "anomaly"
        else:
            record = generate_normal_transaction(index + 1, device_ids, device_usual_locations)
            label = "normal"

        records.append(record)
        ground_truth.append({
            "transaction_id": record["transaction_id"],
            "label": label,
        })

    # Create the main dataset without the anomaly label, as required.
    transactions_df = pd.DataFrame(records)
    columns = [
        "transaction_id",
        "timestamp",
        "amount",
        "location",
        "merchant_category",
        "payment_method",
        "device_id",
        "transaction_type",
        "hour",
        "is_new_device",
        "distance_from_usual",
        "failed_attempts",
    ]
    transactions_df = transactions_df[columns]

    # Save the data files.
    transactions_df.to_csv("data/transactions.csv", index=False)

    ground_truth_df = pd.DataFrame(ground_truth)
    ground_truth_df = ground_truth_df[["transaction_id", "label"]]
    ground_truth_df.to_csv("data/ground_truth.csv", index=False)

    print(f"Created {len(records)} transactions in data/transactions.csv")
    print(f"Created ground truth labels in data/ground_truth.csv")
    print(f"Anomalies injected: {ANOMALY_COUNT}")


if __name__ == "__main__":
    main()
