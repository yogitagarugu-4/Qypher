from respond.policy_engine import decide_action, evaluate_action, evaluate_transaction
from respond.trust_score import assess_transaction_risk, calculate_trust_score


def test_trust_score_and_policy_engine_return_valid_decisions():
    trust_score = calculate_trust_score(
        signature_valid=True,
        conventional_label="NORMAL",
        quantum_label="NORMAL",
        amount=120.0,
        distance_from_usual=20.0,
        failed_attempts=0,
        is_new_device=False,
    )

    assert 0 <= trust_score <= 100

    action = decide_action(trust_score)
    assert action in {"ALLOW", "MONITOR", "DENY", "QUARANTINE"}

    result = evaluate_action(trust_score)
    assert result["action"] == action
    assert "reason" in result


def test_decision_thresholds_keep_deny_and_quarantine_separate():
    assert decide_action(81) == "ALLOW"
    assert decide_action(61) == "MONITOR"
    assert decide_action(60) == "DENY"
    assert decide_action(15) == "DENY"
    assert decide_action(14) == "QUARANTINE"


def test_transaction_decision_explains_detector_and_signature_evidence():
    result = evaluate_transaction(
        trust_score=35,
        signature_valid=False,
        conventional_label="ANOMALY",
        quantum_label="NORMAL",
        amount=750,
        distance_from_usual=250,
        failed_attempts=4,
        is_new_device=True,
    )

    assert result["action"] == "DENY"
    assert "signature INVALID" in result["reason"]
    assert "Isolation Forest flagged an anomaly" in result["reason"]
    assert "new device" in result["reason"]


def test_explainable_score_uses_only_supported_transaction_risk_factors():
    safe = assess_transaction_risk(
        {
            "amount": 120,
            "hour": 14,
            "distance_from_usual": 15,
            "failed_attempts": 0,
            "is_new_device": False,
        },
        "NORMAL",
        "NORMAL",
        True,
        reference_amount_median=200,
        reference_amount_mad=20,
        transactions_last_hour=1,
    )
    assert safe["trust_score"] == 100
    assert safe["risk_factors"] == []

    suspicious = assess_transaction_risk(
        {
            "amount": 5000,
            "hour": 2,
            "distance_from_usual": 250,
            "failed_attempts": 3,
            "is_new_device": True,
        },
        "NORMAL",
        "ANOMALY",
        True,
        reference_amount_median=200,
        reference_amount_mad=20,
        transactions_last_hour=5,
    )
    assert 0 <= suspicious["trust_score"] < 100
    assert {
        "amount", "time", "location", "failed_attempts", "device",
        "frequency", "quantum_inspired",
    }.issubset({factor["key"] for factor in suspicious["risk_factors"]})
