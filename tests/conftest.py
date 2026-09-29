"""Test fixtures to keep the benchmark dataset and history isolated between tests."""

import os
from pathlib import Path

import pytest

from generate_data import main as generate_dataset
from history.store import HISTORY_PATH, ensure_history_file


@pytest.fixture(autouse=True)
def reset_project_data():
    """Isolate tests while restoring all CSV files they mutate."""
    csv_paths = [
        Path("data/transactions.csv"),
        Path("data/ground_truth.csv"),
        Path(HISTORY_PATH),
    ]
    snapshots = {
        path: path.read_bytes() if path.is_file() else None
        for path in csv_paths
    }

    try:
        generate_dataset()
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)
        ensure_history_file()
        yield
    finally:
        for path, original_bytes in snapshots.items():
            if original_bytes is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original_bytes)
