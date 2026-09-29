import pathlib

import pandas as pd
import pytest

from data_loader import (
    get_features_for_anomaly_detection,
    load_transactions_csv,
    normalize_boolean_value,
    preprocess_transactions,
    validate_required_columns,
)


def test_transactions_csv_loads_and_has_required_columns():
    """Confirm that the generated transaction CSV can be loaded and validated."""
    project_root = pathlib.Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "transactions.csv"

    df = load_transactions_csv(str(csv_path))

    assert not df.empty
    assert list(df.columns) is not None
    assert validate_required_columns(df) is True


def test_boolean_normalization_does_not_treat_false_strings_as_true():
    assert normalize_boolean_value("False") is False
    assert normalize_boolean_value(" no ") is False
    assert normalize_boolean_value("TRUE") is True
    assert normalize_boolean_value(True) is True


def test_missing_numeric_features_are_filled_and_invalid_values_are_rejected():
    transactions = pd.read_csv("data/transactions.csv").head(3)
    numeric_columns = ["amount", "hour", "distance_from_usual", "failed_attempts"]
    transactions[numeric_columns] = None

    prepared = get_features_for_anomaly_detection(transactions)
    assert prepared["features"].notna().all().all()

    transactions = pd.read_csv("data/transactions.csv").head(3)
    transactions["amount"] = transactions["amount"].astype(str)
    transactions.loc[0, "amount"] = "not-a-number"
    with pytest.raises(ValueError, match="amount.*non-numeric"):
        preprocess_transactions(transactions)
