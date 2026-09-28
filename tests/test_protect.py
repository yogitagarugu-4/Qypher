from protect.signature_simulator import SignatureSimulator
from protect.verifier import verify_transaction_signature, verify_transaction_result


def test_signature_simulation_creates_valid_signature():
    transaction = {
        "transaction_id": "TXN_TEST_001",
        "timestamp": "2026-01-15 20:00:00",
        "amount": 120.50,
        "location": "New York",
        "merchant_category": "Dining",
        "payment_method": "Credit Card",
        "device_id": "DEV-TEST-1",
        "transaction_type": "purchase",
        "hour": 20,
        "is_new_device": False,
        "distance_from_usual": 12.0,
        "failed_attempts": 0,
    }

    result = SignatureSimulator().sign_transaction(transaction)

    assert result["transaction_id"] == "TXN_TEST_001"
    assert "signature" in result
    assert len(result["signature"]) > 0
    assert verify_transaction_signature(transaction, result["signature"]) is True

    verification = verify_transaction_result(transaction, result["signature"])
    assert verification["signature_valid"] is True
