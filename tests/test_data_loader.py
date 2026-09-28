import pathlib

from data_loader import load_transactions_csv, validate_required_columns


def test_transactions_csv_loads_and_has_required_columns():
    """Confirm that the generated transaction CSV can be loaded and validated."""
    project_root = pathlib.Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "transactions.csv"

    df = load_transactions_csv(str(csv_path))

    assert not df.empty
    assert list(df.columns) is not None
    assert validate_required_columns(df) is True
