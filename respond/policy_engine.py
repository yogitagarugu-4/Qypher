"""Decision policy engine for Qypher.

This layer takes a trust score and decides what action should be taken.
"""

from __future__ import annotations

from typing import Dict, Any


def decide_action(trust_score: float) -> str:
    """Convert a trust score into a human-readable action."""
    if trust_score >= 80:
        return "ALLOW"
    if 50 <= trust_score <= 79:
        return "MONITOR"
    if 20 <= trust_score <= 49:
        return "DENY"
    return "QUARANTINE"


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
