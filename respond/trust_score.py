"""Deterministic, explainable transaction-level trust scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd


RISK_LEVELS = (
    (30, "HIGH RISK"),
    (60, "SUSPICIOUS"),
    (80, "MODERATE"),
    (100, "TRUSTED"),
)


def assess_transaction_risk(
    transaction: dict[str, Any],
    conventional_label: str,
    quantum_label: str,
    signature_valid: bool,
    *,
    reference_amount_median: float | None = None,
    reference_amount_mad: float | None = None,
    transactions_last_hour: int | None = None,
) -> dict[str, Any]:
    """Return a transaction's score and only the risk factors supported by its data.

    The score starts at 100. The documented deductions are: amount 22; new device
    18; unusual hour 12; location distance 15 or 22; five-or-more hourly
    transactions 15; failed attempts 10 or 18; Isolation Forest anomaly 24;
    quantum-inspired anomaly 28; invalid signature 70. Multiple supported signals
    accumulate and the result is clamped to 0–100. Detector-specific raw scores
    are shown to users but are not compared directly because their scales are not
    calibrated to one another.
    """
    factors: list[dict[str, Any]] = []

    def add_factor(key: str, label: str, points: int, evidence: str, severity: str) -> None:
        factors.append({
            "key": key,
            "label": label,
            "points": points,
            "evidence": evidence,
            "severity": severity,
        })

    amount = _finite_number(transaction.get("amount"))
    hour = _transaction_hour(transaction)
    distance = _finite_number(transaction.get("distance_from_usual"))
    attempts = int(_finite_number(transaction.get("failed_attempts")))

    if reference_amount_median is not None and amount is not None:
        mad = max(float(reference_amount_mad or 0.0), 0.0)
        amount_limit = max(float(reference_amount_median) * 3.0, mad * 6.0, 1000.0)
        if amount > amount_limit:
            add_factor(
                "amount",
                "Unusually high transaction amount",
                22,
                f"{amount:,.2f} exceeds the account-pattern threshold of {amount_limit:,.2f}",
                "HIGH",
            )

    if bool(transaction.get("is_new_device", False)):
        add_factor("device", "New device detected", 18, "is_new_device is true", "HIGH")

    if hour is not None and 0 <= hour <= 4:
        add_factor("time", "Unusual transaction time", 12, f"Transaction hour is {hour:02d}:00", "MEDIUM")

    if distance is not None and distance > 100:
        points = 22 if distance > 500 else 15
        severity = "HIGH" if distance > 500 else "MEDIUM"
        add_factor(
            "location",
            "Location differs from usual behaviour",
            points,
            f"Distance from usual location is {distance:,.1f}",
            severity,
        )

    frequency = transactions_last_hour
    if frequency is None:
        frequency = int(_finite_number(transaction.get("transactions_last_hour")) or 0)
    if frequency >= 5:
        add_factor(
            "frequency",
            "Rapid transaction frequency",
            15,
            f"{frequency} transactions occurred within one hour",
            "HIGH",
        )

    if attempts > 0:
        points = 18 if attempts >= 3 else 10
        severity = "HIGH" if attempts >= 3 else "MEDIUM"
        add_factor("failed_attempts", "Failed attempts observed", points, f"{attempts} failed attempt(s)", severity)

    if str(conventional_label).upper() == "ANOMALY":
        add_factor(
            "isolation_forest",
            "Isolation Forest anomaly",
            24,
            "The Isolation Forest model classified this transaction as ANOMALY",
            "MEDIUM",
        )
    if str(quantum_label).upper() == "ANOMALY":
        add_factor(
            "quantum_inspired",
            "Quantum-Inspired QSVM anomaly",
            28,
            "The quantum-inspired model classified this transaction as ANOMALY",
            "HIGH",
        )
    if not signature_valid:
        add_factor("signature", "Invalid transaction signature", 70, "Signature verification failed", "HIGH")

    score = max(0.0, min(100.0, 100.0 - sum(factor["points"] for factor in factors)))
    risk_level = next(level for maximum, level in RISK_LEVELS if score <= maximum)
    return {
        "trust_score": round(score, 2),
        "risk_level": risk_level,
        "risk_factors": factors,
        "explanation": [factor["evidence"] for factor in factors],
    }


def calculate_trust_score(
    signature_valid: bool,
    conventional_label: str,
    quantum_label: str,
    amount: float,
    distance_from_usual: float,
    failed_attempts: int,
    is_new_device: bool,
) -> float:
    """Compatibility wrapper for callers using the original scoring interface."""
    return assess_transaction_risk(
        {
            "amount": amount,
            "distance_from_usual": distance_from_usual,
            "failed_attempts": failed_attempts,
            "is_new_device": is_new_device,
        },
        conventional_label,
        quantum_label,
        signature_valid,
        reference_amount_median=0.0,
    )["trust_score"]


def _transaction_hour(transaction: dict[str, Any]) -> int | None:
    hour = _finite_number(transaction.get("hour"))
    if hour is not None:
        return int(hour)
    timestamp = pd.to_datetime(transaction.get("timestamp"), errors="coerce")
    return None if pd.isna(timestamp) else int(timestamp.hour)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and number not in (float("inf"), float("-inf")) else None
