"""Decision policy engine for Qypher.

This layer takes a trust score and decides what action should be taken.
"""

from __future__ import annotations

from typing import Dict, Any


# Inclusive score boundaries; change these values to configure policy behaviour.
DECISION_THRESHOLDS = {
    "allow": 81,
    "monitor": 61,
    "quarantine_below": 15,
}


def decide_action(trust_score: float) -> str:
    """Convert a trust score into a human-readable action."""
    if trust_score >= DECISION_THRESHOLDS["allow"]:
        return "ALLOW"
    if trust_score >= DECISION_THRESHOLDS["monitor"]:
        return "MONITOR"
    if trust_score < DECISION_THRESHOLDS["quarantine_below"]:
        return "QUARANTINE"
    return "DENY"


def evaluate_action(trust_score: float) -> Dict[str, Any]:
    """Return action and explanation in a simple structure."""
    action = decide_action(trust_score)
    if action == "ALLOW":
        reason = "Trust is high and the transaction appears consistent with normal behavior."
    elif action == "MONITOR":
        reason = "The transaction requires review but is not yet considered unsafe."
    elif action == "DENY":
        reason = "A significant risk signal was detected and the transaction should be blocked."
    else:
        reason = "This transaction has a very high-risk profile and should be isolated."

    return {
        "trust_score": trust_score,
        "action": action,
        "reason": reason,
    }


def evaluate_transaction(
    trust_score: float,
    signature_valid: bool,
    conventional_label: str,
    quantum_label: str,
    amount: float,
    distance_from_usual: float,
    failed_attempts: int,
    is_new_device: bool,
    risk_factors: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Add transaction-specific evidence to the response-policy explanation."""
    decision = evaluate_action(trust_score)
    if risk_factors is not None:
        evidence = [factor["evidence"] for factor in risk_factors]
    else:
        evidence = []
        if not signature_valid:
            evidence.append("signature INVALID")
        if conventional_label.upper() == "ANOMALY":
            evidence.append("Isolation Forest flagged an anomaly (classified this transaction as ANOMALY)")
        if quantum_label.upper() == "ANOMALY":
            evidence.append("Quantum-Inspired QSVM classified this transaction as ANOMALY")
        if distance_from_usual > 100:
            evidence.append("unusual distance from the usual location")
        if failed_attempts > 0:
            evidence.append(f"{failed_attempts} failed attempt(s)")
        if is_new_device:
            evidence.append("new device")

    explanation = "; ".join(evidence) if evidence else (
        "valid signature, both detectors NORMAL, and no configured behavioural risk threshold exceeded"
    )
    decision["reason"] = f"{decision['reason']} Evidence: {explanation}."
    return decision
