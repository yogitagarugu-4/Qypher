"""Trust scoring for Qypher.

This layer calculates a simple trust score between 0 and 100 based on:
- signature validity
- conventional anomaly result
- quantum anomaly result
- transaction risk features such as amount, distance, failed attempts, and new device

The goal is to keep the logic easy to explain and easy to replace later.
"""

from __future__ import annotations

from typing import Dict, Any


def calculate_trust_score(
    signature_valid: bool,
    conventional_label: str,
    quantum_label: str,
    amount: float,
    distance_from_usual: float,
    failed_attempts: int,
    is_new_device: bool,
) -> float:
    """Compute a simple trust score between 0 and 100.

    The score is intentionally explainable:
    - valid signatures help trust
    - anomaly labels reduce trust
    - high amount, long distance, retries, and new device also reduce trust
    """
    score = 100.0

    # Signature validity: a valid signature keeps the transaction trusted.
    if not signature_valid:
        score -= 45

    # Conventional anomaly result.
    if conventional_label == "ANOMALY":
        score -= 30
    elif conventional_label == "NORMAL":
        score += 5

    # Quantum anomaly result.
    if quantum_label == "ANOMALY":
        score -= 25
    elif quantum_label == "NORMAL":
        score += 5

    # Risk-based features.
    if amount > 1000:
        score -= 15
    elif amount > 500:
        score -= 8

    if distance_from_usual > 500:
        score -= 20
    elif distance_from_usual > 100:
        score -= 10

    if failed_attempts > 3:
        score -= 20
    elif failed_attempts > 1:
        score -= 8

    if is_new_device:
        score -= 10

    # Keep the score inside the required 0-100 range.
    if score < 0:
        score = 0
    if score > 100:
        score = 100

    return round(score, 2)
