"""Persistent storage for individual transaction analysis results."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from data_loader import normalize_boolean_value


HISTORY_PATH = "data/history.csv"
HISTORY_COLUMNS = [
    "transaction_id",
    "account",
    "timestamp",
    "amount",
    "detection_method",
    "location",
    "device_id",
    "signature_valid",
    "conventional_label",
    "quantum_label",
    "anomaly_status",
    "conventional_score",
    "quantum_score",
    "trust_score",
    "action",
    "reason",
    "alert_message",
    "conventional_processing_time",
    "quantum_processing_time",
    "risk_level",
    "risk_factors",
]


def ensure_history_file() -> None:
    """Create the history directory and file if they do not already exist."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    if not os.path.exists(HISTORY_PATH) or os.path.getsize(HISTORY_PATH) == 0:
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(HISTORY_PATH, index=False)


def load_history() -> pd.DataFrame:
    """Load the saved transaction records."""
    ensure_history_file()
    try:
        return pd.read_csv(HISTORY_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def append_history_record(record: dict[str, Any]) -> pd.DataFrame:
    """Append one processed transaction result."""
    return append_history_records([record])


def append_history_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Append a batch of transaction-level results with one history write."""
    if not records:
        return load_history()
    ensure_history_file()

    history_df = load_history()
    new_records = pd.DataFrame(records)
    if history_df.empty:
        updated = new_records.reindex(
            columns=list(dict.fromkeys(HISTORY_COLUMNS + list(new_records.columns)))
        )
    else:
        updated = pd.concat([history_df, new_records], ignore_index=True)
    updated.to_csv(HISTORY_PATH, index=False)
    return updated


def create_history_record(
    transaction_id: str,
    timestamp: str,
    amount: float,
    signature_valid: bool,
    conventional_label: str,
    quantum_label: str,
    conventional_score: float,
    quantum_score: float,
    trust_score: float,
    action: str,
    alert_message: str,
    location: str = "",
    device_id: str = "",
    anomaly_status: str = "",
    reason: str = "",
    conventional_processing_time: float = 0.0,
    quantum_processing_time: float = 0.0,
    risk_level: str = "",
    risk_factors: list[dict[str, Any]] | None = None,
    detection_method: str = "",
    account: str = "QYP-ACCT-1042",
) -> dict[str, Any]:
    """Build one persisted record, retaining the existing history schema."""
    return {
        "transaction_id": transaction_id,
        "account": account,
        "timestamp": timestamp,
        "amount": amount,
        "detection_method": detection_method,
        "location": location,
        "device_id": device_id,
        "signature_valid": normalize_boolean_value(signature_valid),
        "conventional_label": conventional_label,
        "quantum_label": quantum_label,
        "anomaly_status": anomaly_status or (
            "ANOMALY"
            if "ANOMALY" in {conventional_label.upper(), quantum_label.upper()}
            else "NORMAL"
        ),
        "conventional_score": float(conventional_score),
        "quantum_score": float(quantum_score),
        "trust_score": float(trust_score),
        "action": action,
        "reason": reason,
        "alert_message": alert_message,
        "conventional_processing_time": float(conventional_processing_time),
        "quantum_processing_time": float(quantum_processing_time),
        "risk_level": risk_level,
        "risk_factors": json.dumps(risk_factors or [], ensure_ascii=True),
    }
