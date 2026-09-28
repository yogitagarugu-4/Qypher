"""Alerting utilities for Qypher.

When a transaction is denied or quarantined, we create a clear alert message.
If the system supports audio playback, we also attempt to play a beep.
If not, we print a simple alert message to the console.
"""

from __future__ import annotations

from typing import Dict, Any


try:
    import winsound  # Only available on Windows
except ImportError:  # pragma: no cover
    winsound = None


def create_alert_message(action: str, reason: str) -> str:
    """Create a clear message for suspicious actions."""
    if action in {"DENY", "QUARANTINE"}:
        return f"ALERT: {action} - {reason}"
    return "No alert required."


def trigger_sound_alert() -> None:
    """Try to play a beep on Windows if available.

    If sound is not available, print a simple fallback alert.
    """
    if winsound is not None:
        # A simple warning tone.
        winsound.Beep(900, 250)
        winsound.Beep(700, 250)
        return

    print("ALERT: Suspicious transaction detected")


def handle_alert(action: str, reason: str) -> str:
    """Create the alert message and trigger a sound if possible."""
    message = create_alert_message(action, reason)
    if action in {"DENY", "QUARANTINE"}:
        trigger_sound_alert()
    return message
