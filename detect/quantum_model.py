"""Quantum-inspired anomaly detection for the Qypher prototype.

IMPORTANT:
This is NOT a real quantum computer implementation. It is a simplified,
student-friendly simulation inspired by quantum state concepts such as unit
vectors, amplitudes, and similarity/distance between states.

The goal is to demonstrate how a quantum-inspired approach can be used to score
transactions in a way that is conceptually similar to amplitude-based pattern
comparison.

This implementation is deliberately simple and deterministic so it can be:
- explained easily to students,
- compared directly with the conventional Isolation Forest detector,
- replaced later by a more advanced quantum ML approach such as QSVM.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from data_loader import get_features_for_anomaly_detection


class QuantumInspiredDetector:
    """A lightweight quantum-inspired anomaly detector.

    The idea is intentionally simple:
    1. Each transaction is converted into a vector of numeric features.
    2. Each vector is normalized to unit length, similar to a quantum state vector.
    3. We compare it to a reference normal state, calculated from the dataset.
    4. A transaction is considered more anomalous when its state is less similar to
       the usual transaction state.

    This is a prototype only. It is not executed on a real quantum computer.
    """

    def __init__(self, similarity_threshold: float = 0.4, random_state: int = 42):
        self.similarity_threshold = similarity_threshold
        self.random_state = random_state
        self.reference_vector_: np.ndarray | None = None
        self.feature_columns_: list[str] | None = None

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Make sure the feature matrix is valid and numeric.

        Args:
            features: Model-ready feature matrix from data_loader.

        Returns:
            A numeric DataFrame with no missing values.
        """
        cleaned = features.copy()
        cleaned = cleaned.fillna(0.0)

        # Keep only numeric columns so the vector math is stable.
        numeric_columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        return cleaned[numeric_columns]

    def fit(self, features: pd.DataFrame) -> "QuantumInspiredDetector":
        """Build a reference normal state from the training data.

        Args:
            features: Prepared feature matrix from data_loader.

        Returns:
            self for chaining.
        """
        prepared = self._prepare_features(features)
        self.feature_columns_ = prepared.columns.tolist()

        # Convert the feature matrix into a 2D numpy array for vector math.
        matrix = prepared.to_numpy(dtype=float)

        # Build a reference normal vector as the average of all transaction vectors.
        # This acts like a typical transaction pattern in a quantum-inspired state space.
        # We also weight a few high-risk signals more strongly so unusual values stand out.
        weights = np.ones(matrix.shape[1], dtype=float)

        # A simple, explainable weighting scheme: unusual transaction size,
        # long distance away from usual location, and failed attempts matter more.
        for idx, column_name in enumerate(self.feature_columns_):
            if "amount" in column_name:
                weights[idx] = 2.5
            elif "distance_from_usual" in column_name:
                weights[idx] = 2.0
            elif "failed_attempts" in column_name:
                weights[idx] = 2.5
            elif "is_new_device" in column_name:
                weights[idx] = 1.5

        weighted_matrix = matrix * weights
        mean_vector = weighted_matrix.mean(axis=0)
        self.reference_vector_ = self._normalize_vector(mean_vector)

        return self

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Convert a vector to a unit-length vector.

        This is a quantum-inspired idea: a state vector is often normalized so its
        amplitudes represent a valid probability-like distribution.
        """
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.zeros_like(vector)
        return vector / norm

    def _quantum_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Compute similarity between two normalized vectors.

        In quantum-inspired thinking, similarity is often based on the overlap
        between two state vectors. The higher the overlap, the more similar the
        states are.

        Here we use the cosine similarity, which is a simple and intuitive measure:
            similarity = dot(a, b) / (|a| * |b|)

        Since both vectors are normalized, this becomes essentially the dot product.
        """
        vector_a = self._normalize_vector(vector_a)
        vector_b = self._normalize_vector(vector_b)
        similarity = float(np.dot(vector_a, vector_b))

        # Keep the result bounded in a stable range.
        return max(0.0, min(1.0, similarity))

    def predict_transactions(self, features: pd.DataFrame, transaction_data: pd.DataFrame) -> pd.DataFrame:
        """Generate quantum-inspired anomaly scores and labels for each transaction.

        Args:
            features: Prepared feature matrix.
            transaction_data: Original transaction information for display.

        Returns:
            A DataFrame containing:
            - transaction_id
            - quantum_anomaly_score
            - quantum_prediction
            - quantum_prediction_label
        """
        if self.reference_vector_ is None or self.feature_columns_ is None:
            raise ValueError("The detector must be fitted before making predictions.")

        prepared = self._prepare_features(features)
        prepared = prepared.reindex(columns=self.feature_columns_, fill_value=0.0)

        matrix = prepared.to_numpy(dtype=float)
        results = []

        for index, row in enumerate(matrix):
            normalized_row = self._normalize_vector(row)
            similarity = self._quantum_similarity(normalized_row, self.reference_vector_)

            # Distance from the normal reference state.
            # If the transaction is similar to the reference, it should be near normal.
            # The farther away the transaction state is, the more anomalous it is.
            distance = 1.0 - similarity

            # Use a squared distance to make unusual transactions stand out more clearly.
            anomaly_score = float(distance ** 2)

            # A transaction is anomalous when it differs enough from the normal state.
            # The threshold is intentionally simple and explainable for a student prototype.
            if anomaly_score >= self.similarity_threshold:
                prediction = -1
                prediction_label = "ANOMALY"
            else:
                prediction = 1
                prediction_label = "NORMAL"

            results.append({
                "transaction_id": transaction_data.iloc[index]["transaction_id"],
                "quantum_anomaly_score": anomaly_score,
                "quantum_prediction": prediction,
                "quantum_prediction_label": prediction_label,
            })

        result_df = pd.DataFrame(results)

        # Keep the original transaction metadata in a reusable way for the dashboard.
        summary = transaction_data[["transaction_id", "amount", "location", "device_id"]].copy()
        final_df = summary.merge(result_df, on="transaction_id", how="left")

        return final_df[[
            "transaction_id",
            "amount",
            "location",
            "device_id",
            "quantum_anomaly_score",
            "quantum_prediction",
            "quantum_prediction_label",
        ]].copy()


def detect_quantum_anomalies(raw_transaction_df: pd.DataFrame) -> pd.DataFrame:
    """Run the quantum-inspired detector end-to-end.

    This function follows the same structure as the conventional detector so future
    code can compare both models in a consistent way.

    Args:
        raw_transaction_df: Raw transaction data.

    Returns:
        DataFrame with the quantum-inspired predictions and scores for each transaction.
    """
    prepared = get_features_for_anomaly_detection(raw_transaction_df)
    transaction_data = prepared["transaction_data"].copy()
    feature_matrix = prepared["features"].copy()

    detector = QuantumInspiredDetector(similarity_threshold=0.4, random_state=42)
    detector.fit(feature_matrix)

    return detector.predict_transactions(feature_matrix, transaction_data)
