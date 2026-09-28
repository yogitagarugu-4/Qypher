"""Persistent transaction history for Qypher.

This module persists a history of processed transactions in data/history.csv.
It appends new records without overwriting earlier history, so the system can keep
an audit trail for later dashboard display or analysis.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any

import pandas as pd

from detect.conventional_model import detect_conventional_anomalies
from detect.quantum_model import detect_quantum_anomalies
from protect.signature_simulator import simulate_signature_for_transaction
from protect.verifier import verify_transaction_signature
from respond.alerting import create_alert_message, handle_alert
from respond.policy_engine import decide_action
from respond.trust_score import calculate_trust_score


HISTORY_PATH = "data/history.csv"
TRANSACTIONS_PATH = "data/transactions.csv"


def ensure_history_file() -> None:
    """Create the history directory and CSV file if they do not already exist."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    if not os.path.exists(HISTORY_PATH):
        empty_df = pd.DataFrame(columns=[
            "transaction_id",
            "timestamp",
            "amount",
            "signature_valid",
            "conventional_label",
            "quantum_label",
            "conventional_score",
            "quantum_score",
            "trust_score",
            "action",
            "alert_message",
        ])
        empty_df.to_csv(HISTORY_PATH, index=False)


def append_history_record(record: Dict[str, Any]) -> pd.DataFrame:
    """Append one processed transaction record to the history CSV."""
    ensure_history_file()

    history_df = pd.read_csv(HISTORY_PATH)
    new_record = pd.DataFrame([record])
    updated = pd.concat([history_df, new_record], ignore_index=True)
    updated.to_csv(HISTORY_PATH, index=False)

    return updated


def analyze_transaction(transaction: Dict[str, Any], persist_history: bool = False) -> Dict[str, Any]:
    """Analyze one transaction event and return its individual result.

    This is the single transaction-level entry point used by both the live form and
    CSV-driven demo uploads. It keeps the same PROTECT -> DETECT -> RESPOND flow
    and returns one score/status/action for one transaction only.
    """
    tx = dict(transaction)

    if "transaction_id" not in tx or not str(tx["transaction_id"]).strip():
        tx["transaction_id"] = "TXN_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    if "timestamp" not in tx or not str(tx["timestamp"]).strip():
        tx["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tx_df = pd.DataFrame([tx])
    signed = simulate_signature_for_transaction(tx)
    signature_valid = verify_transaction_signature(tx, signed["signature"])

    conventional_results = detect_conventional_anomalies(tx_df)
    quantum_results = detect_quantum_anomalies(tx_df)

    current_conventional = conventional_results.iloc[0]
    current_quantum = quantum_results.iloc[0]

    conventional_label = str(current_conventional["prediction_label"]).upper()
    quantum_label = str(current_quantum["quantum_prediction_label"]).upper()
    conventional_score = float(current_conventional["anomaly_score"])
    quantum_score = float(current_quantum["quantum_anomaly_score"])

    trust_score = calculate_trust_score(
        signature_valid=signature_valid,
        conventional_label=conventional_label,
        quantum_label=quantum_label,
        amount=float(tx.get("amount", 0.0)),
        distance_from_usual=float(tx.get("distance_from_usual", 0.0)),
        failed_attempts=int(tx.get("failed_attempts", 0)),
        is_new_device=bool(tx.get("is_new_device", False)),
    )

    action = decide_action(trust_score)
    alert_message = create_alert_message(action, "Transaction evaluated by Qypher response policy.")
    if action in {"DENY", "QUARANTINE"}:
        handle_alert(action, "Transaction evaluated by Qypher response policy.")

    result = {
        "transaction_id": str(tx["transaction_id"]),
        "timestamp": str(tx["timestamp"]),
        "amount": float(tx.get("amount", 0.0)),
        "signature_valid": signature_valid,
        "conventional_label": conventional_label,
        "quantum_label": quantum_label,
        "conventional_score": conventional_score,
        "quantum_score": quantum_score,
        "trust_score": round(trust_score, 2),
        "action": action,
        "alert_message": alert_message,
        "anomaly_score": conventional_score,
        "anomaly_status": conventional_label,
        "risk_score": round(trust_score, 2),
        "alert": action in {"DENY", "QUARANTINE"},
    }

    if persist_history:
        history_record = create_history_record(
            transaction_id=str(tx["transaction_id"]),
            timestamp=str(tx["timestamp"]),
            amount=float(tx.get("amount", 0.0)),
            signature_valid=signature_valid,
            conventional_label=conventional_label,
            quantum_label=quantum_label,
            conventional_score=conventional_score,
            quantum_score=quantum_score,
            trust_score=trust_score,
            action=action,
            alert_message=alert_message,
        )
        append_history_record(history_record)

    return {
        "transaction_id": result["transaction_id"],
        "signature": signed["signature"],
        "signature_valid": signature_valid,
        "conventional_label": conventional_label,
        "quantum_label": quantum_label,
        "trust_score": round(trust_score, 2),
        "action": action,
        "alert_message": alert_message,
        "anomaly_score": conventional_score,
        "anomaly_status": conventional_label,
        "risk_score": round(trust_score, 2),
        "alert": action in {"DENY", "QUARANTINE"},
        "timestamp": str(tx["timestamp"]),
        "amount": float(tx.get("amount", 0.0)),
    }


def submit_new_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Process a newly submitted transaction through PROTECT, DETECT, and RESPOND."""
    # Update the canonical dataset with the live transaction before evaluating it.
    if os.path.exists(TRANSACTIONS_PATH):
        transactions_df = pd.read_csv(TRANSACTIONS_PATH)
    else:
        transactions_df = pd.DataFrame()

    tx = dict(transaction)
    if "transaction_id" not in tx or not str(tx["transaction_id"]).strip():
        tx["transaction_id"] = "TXN_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    if "timestamp" not in tx or not str(tx["timestamp"]).strip():
        tx["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame([tx])
    updated_transactions = pd.concat([transactions_df, new_row], ignore_index=True)
    updated_transactions.to_csv(TRANSACTIONS_PATH, index=False)

    return analyze_transaction(tx, persist_history=True)


def load_history() -> pd.DataFrame:
    """Load the saved history CSV."""
    ensure_history_file()
    return pd.read_csv(HISTORY_PATH)


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
) -> Dict[str, Any]:
    """Create one history row in the exact format required by the project."""
    return {
        "transaction_id": transaction_id,
        "timestamp": timestamp,
        "amount": amount,
        "signature_valid": bool(signature_valid),
        "conventional_label": conventional_label,
        "quantum_label": quantum_label,
        "conventional_score": float(conventional_score),
        "quantum_score": float(quantum_score),
        "trust_score": float(trust_score),
        "action": action,
        "alert_message": alert_message,
    }
