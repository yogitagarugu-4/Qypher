"""Test fixtures to keep the benchmark dataset and history isolated between tests."""

import os

import pytest

from generate_data import main as generate_dataset
from history.store import HISTORY_PATH, ensure_history_file


@pytest.fixture(autouse=True)
def reset_project_data():
    """Regenerate the canonical transaction dataset and clear history for each test."""
    generate_dataset()
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)
    ensure_history_file()

    yield

    generate_dataset()
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)
    ensure_history_file()
