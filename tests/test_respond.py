from respond.policy_engine import decide_action, evaluate_action
from respond.trust_score import calculate_trust_score


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
