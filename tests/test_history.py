from history.store import append_history_record, create_history_record, load_history, submit_new_transaction


def test_history_file_can_append_records():
    record = create_history_record(
        transaction_id="TXN_HISTORY_TEST",
        timestamp="2026-01-15 22:00:00",
        amount=55.20,
        signature_valid=True,
        conventional_label="NORMAL",
        quantum_label="NORMAL",
        conventional_score=0.30,
        quantum_score=0.02,
        trust_score=88.50,
        action="ALLOW",
        alert_message="No alert required.",
    )

    history_df = append_history_record(record)
    loaded = load_history()

    assert "transaction_id" in history_df.columns
    assert "transaction_id" in loaded.columns
    assert "TXN_HISTORY_TEST" in set(loaded["transaction_id"].astype(str).tolist())


def test_submit_new_transaction_records_to_history():
    new_transaction = {
        "transaction_id": "TXN_LIVE_SUBMIT_TEST",
        "timestamp": "2026-01-15 23:45:00",
        "amount": 240.00,
        "location": "Boston",
        "merchant_category": "Retail",
        "payment_method": "Credit Card",
        "device_id": "DEV-LIVE-99",
        "transaction_type": "purchase",
        "hour": 23,
        "is_new_device": False,
        "distance_from_usual": 30.0,
        "failed_attempts": 0,
    }

    result = submit_new_transaction(new_transaction)

    assert result["transaction_id"] == "TXN_LIVE_SUBMIT_TEST"
    assert result["signature_valid"] is True
    assert result["action"] in {"ALLOW", "MONITOR", "DENY", "QUARANTINE"}

    history = load_history()
    assert "TXN_LIVE_SUBMIT_TEST" in set(history["transaction_id"].astype(str).tolist())
