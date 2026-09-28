"""Detection modules for the Qypher prototype."""

from .conventional_model import ConventionalAnomalyDetector, detect_conventional_anomalies

__all__ = ["ConventionalAnomalyDetector", "detect_conventional_anomalies"]
