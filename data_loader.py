"""Data loading and preprocessing for the Qypher prototype.

This module is responsible for:
1. loading the transaction CSV from disk,
2. validating that required columns are present,
3. cleaning missing values,
4. creating useful time-based features,
5. turning categorical values into numbers for machine learning,
6. scaling numeric features for model input,
7. preserving transaction IDs and original transaction fields for later dashboard display.

This file is intentionally beginner-friendly and does not include the ML models themselves.
"""

import os
from typing import List, Dict, Any

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:  # pragma: no cover - fallback if sklearn is not installed yet
    ColumnTransformer = None
    SimpleImputer = None
    Pipeline = None
    OneHotEncoder = None
    StandardScaler = None


REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "amount",
    "location",
    "merchant_category",
    "payment_method",
    "device_id",
    "transaction_type",
    "hour",
    "is_new_device",
    "distance_from_usual",
    "failed_attempts",
]


def load_transactions_csv(csv_path: str) -> pd.DataFrame:
    """Load the CSV file and return a pandas DataFrame.

    Args:
        csv_path: Full path or relative path to the transactions CSV.

    Returns:
        A DataFrame containing the raw transaction data.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_required_columns(df)
    return df


def validate_required_columns(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """Check that all required transaction columns are present.

    Args:
        df: DataFrame to validate.
        required_columns: Optional custom list of required fields.

    Returns:
        True if all required columns are present.

    Raises:
        ValueError: If any required column is missing.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    return True


def _safe_fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values safely so ML code does not fail later.

    Numeric columns use the median. Categorical values use 'Unknown'.
    """
    cleaned = df.copy()

    for column in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            median_value = (
                cleaned[column].median()
                if cleaned[column].notna().any()
                else 0
            )
            cleaned[column] = cleaned[column].fillna(
                median_value if pd.notna(median_value) else 0
            )
        else:
            cleaned[column] = cleaned[column].fillna("Unknown")

    return cleaned


def _convert_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp into useful time-based features.

    We keep the original timestamp and also create:
    - hour_of_day
    - day_of_week

    These are helpful for detecting unusual transaction times.
    """
    processed = df.copy()

    if "timestamp" in processed.columns:
        processed["timestamp"] = pd.to_datetime(processed["timestamp"], errors="coerce")

        # Fill any missing timestamps with a safe default value.
        processed["timestamp"] = processed["timestamp"].fillna(pd.Timestamp("2026-01-01 00:00:00"))

        # Create new time features while preserving the original timestamp.
        processed["hour_of_day"] = processed["timestamp"].dt.hour
        processed["day_of_week"] = processed["timestamp"].dt.dayofweek

    return processed


def _convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean-like columns to numeric values for ML models.

    This makes fields like is_new_device usable in scikit-learn models.
    """
    processed = df.copy()

    if "is_new_device" in processed.columns:
        processed["is_new_device"] = _normalize_boolean(processed["is_new_device"])

    return processed


def _scale_numeric_features(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """Scale selected numeric columns so they are on a similar range.

    This helps machine learning algorithms behave more consistently.
    """
    scaled_df = df.copy()

    if not numeric_columns:
        return scaled_df

    columns_to_scale = [column for column in numeric_columns if column in scaled_df.columns]

    if not columns_to_scale:
        return scaled_df

    # Use scikit-learn StandardScaler if it is available.
    if StandardScaler is not None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(scaled_df[columns_to_scale].fillna(0))
        for index, column in enumerate(columns_to_scale):
            scaled_df[f"{column}_scaled"] = scaled_values[:, index]
    else:
        # Fallback scaling if sklearn is not installed yet.
        for column in columns_to_scale:
            mean_value = scaled_df[column].mean()
            std_value = scaled_df[column].std(ddof=0)
            if std_value == 0:
                std_value = 1
            scaled_df[f"{column}_scaled"] = (scaled_df[column] - mean_value) / std_value

    return scaled_df


EXCLUDED_MODEL_COLUMNS = {
    "transaction_id",
    "timestamp",
    "transaction_ref",
    "user_id",
    "device_id",
    "currency",
    "label",
    "new_device",
}

NUMERIC_MODEL_FEATURES = [
    "amount",
    "hour",
    "is_new_device",
    "distance_from_usual",
    "failed_attempts",
    "transactions_last_hour",
]

CATEGORICAL_MODEL_FEATURES = [
    "merchant_category",
    "transaction_type",
    "location",
    "payment_method",
]


def _normalize_boolean(series: pd.Series) -> pd.Series:
    """Convert common true/false values into numeric 0/1 values."""
    return series.map(normalize_boolean_value).astype(int)


def normalize_boolean_value(value: Any) -> bool:
    """Normalize boolean-like CSV and form values without Python string truthiness."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return value == 1

    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0", "", "unknown"}:
        return False
    return False


def _ensure_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dynamic transaction inputs before feature extraction."""
    processed = df.copy()

    if "timestamp" in processed.columns:
        processed["timestamp"] = pd.to_datetime(processed["timestamp"], errors="coerce")
        processed["timestamp"] = processed["timestamp"].fillna(pd.Timestamp("2026-01-01 00:00:00"))

    if "hour" not in processed.columns:
        if "timestamp" in processed.columns:
            processed["hour"] = processed["timestamp"].dt.hour.fillna(0).astype(int)
        else:
            processed["hour"] = 0

    if "transactions_last_hour" not in processed.columns:
        if "timestamp" in processed.columns:
            timestamps = processed["timestamp"]
            counts = []
            for value in timestamps:
                if pd.isna(value):
                    counts.append(0)
                    continue
                mask = (timestamps >= value - pd.Timedelta(1, unit="h")) & (timestamps <= value)
                counts.append(int(mask.sum()))
            processed["transactions_last_hour"] = counts
        else:
            processed["transactions_last_hour"] = 0

    if "new_device" in processed.columns and "is_new_device" not in processed.columns:
        processed["is_new_device"] = _normalize_boolean(processed["new_device"])

    if "is_new_device" in processed.columns:
        processed["is_new_device"] = _normalize_boolean(processed["is_new_device"])

    return processed


def _select_model_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Choose numeric and categorical features while excluding identifiers and labels."""
    processed = _ensure_model_columns(df)

    numeric_features = []
    for column in NUMERIC_MODEL_FEATURES:
        if column in processed.columns and pd.api.types.is_numeric_dtype(processed[column]):
            numeric_features.append(column)

    for column in processed.columns:
        if column in EXCLUDED_MODEL_COLUMNS or column in numeric_features:
            continue
        if pd.api.types.is_numeric_dtype(processed[column]):
            numeric_features.append(column)

    categorical_features = []
    for column in CATEGORICAL_MODEL_FEATURES:
        if column in processed.columns:
            categorical_features.append(column)

    for column in processed.columns:
        if column in EXCLUDED_MODEL_COLUMNS or column in numeric_features or column in categorical_features:
            continue
        if processed[column].dtype == "object" or pd.api.types.is_string_dtype(processed[column]):
            categorical_features.append(column)

    return numeric_features, categorical_features


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform the transaction data for machine learning."""
    validate_required_columns(df)

    processed = df.copy()
    numeric_columns = ("amount", "hour", "distance_from_usual", "failed_attempts")
    for column in numeric_columns:
        raw_values = processed[column]
        numeric_values = pd.to_numeric(raw_values, errors="coerce")
        invalid_values = raw_values.notna() & numeric_values.isna()
        if invalid_values.any():
            raise ValueError(f"Transaction column '{column}' contains non-numeric values.")
        if (numeric_values.abs() == float("inf")).any():
            raise ValueError(f"Transaction column '{column}' must contain finite values.")
        processed[column] = numeric_values

    for column in ("amount", "distance_from_usual", "failed_attempts"):
        if (processed[column].dropna() < 0).any():
            raise ValueError(f"Transaction column '{column}' cannot contain negative values.")
    for column in ("hour", "failed_attempts"):
        if (processed[column].dropna() % 1 != 0).any():
            raise ValueError(f"Transaction column '{column}' must contain whole numbers.")
    invalid_hours = processed["hour"].dropna().lt(0) | processed["hour"].dropna().gt(23)
    if invalid_hours.any():
        raise ValueError("Transaction column 'hour' must be between 0 and 23.")

    processed = _safe_fill_missing_values(processed)
    processed = _convert_time_features(processed)
    processed = _convert_boolean_columns(processed)
    processed = _ensure_model_columns(processed)

    numeric_columns_to_scale = [
        "amount",
        "distance_from_usual",
        "failed_attempts",
        "hour",
        "hour_of_day",
        "day_of_week",
        "transactions_last_hour",
    ]

    processed = _scale_numeric_features(processed, numeric_columns_to_scale)
    return processed


def get_features_for_anomaly_detection(df: pd.DataFrame) -> Dict[str, Any]:
    """Build the numeric-only feature matrix used by the anomaly model.

    This keeps the project output format unchanged while preventing identifier
    columns such as transaction_id, device_id, and user_id from being used as ML
    inputs. Categorical values are encoded through a sklearn ColumnTransformer.
    """
    validate_required_columns(df)
    if df.empty:
        raise ValueError("At least one transaction is required for anomaly detection.")

    processed = preprocess_transactions(df)
    numeric_features, categorical_features = _select_model_features(processed)

    numeric_frame = processed[numeric_features].copy() if numeric_features else pd.DataFrame(index=processed.index)
    for column in numeric_features:
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")
        median_value = numeric_frame[column].median()
        numeric_frame[column] = numeric_frame[column].fillna(median_value)

    categorical_frame = pd.DataFrame(index=processed.index)
    for column in categorical_features:
        categorical_frame[column] = processed[column].fillna("Unknown").astype(str)

    model_input = pd.concat([numeric_frame, categorical_frame], axis=1)

    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))

    if not transformers:
        features = pd.DataFrame(index=processed.index)
    else:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        encoded_values = preprocessor.fit_transform(model_input)
        if hasattr(encoded_values, "toarray"):
            encoded_values = encoded_values.toarray()
        feature_names = preprocessor.get_feature_names_out()
        features = pd.DataFrame(encoded_values, columns=feature_names, index=processed.index)

    return {
        "transaction_data": processed,
        "features": features,
        "transaction_ids": processed["transaction_id"].tolist() if "transaction_id" in processed.columns else processed.index.astype(str).tolist(),
    }
