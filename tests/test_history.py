import history.store as history_store
from history.store import (
    append_history_record,
    append_history_records,
    create_history_record,
    load_history,
)


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


def test_single_transaction_entry_api_is_removed():
    assert not hasattr(history_store, "submit_new_transaction")
    assert not hasattr(history_store, "analyze_transaction")


def test_history_can_persist_a_batch_of_individual_transaction_results():
    records = [
        create_history_record(
            transaction_id=f"TXN_BATCH_{index}",
            timestamp="2026-01-15 10:00:00",
            amount=100 + index,
            signature_valid=True,
            conventional_label="NORMAL",
            quantum_label="NORMAL",
            conventional_score=0.1,
            quantum_score=0.02,
            trust_score=100 - index,
            action="ALLOW",
            alert_message="No alert required.",
            risk_level="TRUSTED",
            risk_factors=[{"key": "test", "evidence": "batch result"}],
            detection_method="Isolation Forest",
            account="QYP-ACCT-1042",
        )
        for index in range(10)
    ]

    appended = append_history_records(records)
    saved = appended[appended["transaction_id"].astype(str).str.startswith("TXN_BATCH_")]
    assert len(saved) == 10
    assert saved["trust_score"].notna().all()
    assert saved["risk_factors"].str.contains("batch result").all()
    assert saved["detection_method"].eq("Isolation Forest").all()
    assert saved["account"].eq("QYP-ACCT-1042").all()
