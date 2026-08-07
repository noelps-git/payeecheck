"""Unit tests for attribution.py — the three-layer attribution record."""
from attribution import build_attribution

CLEAN_TX = {"input_method": "type", "paste_trust_score": 0.9, "amount": 2500}
CLEAN_PAYEE = {"vpa_age_days": 900, "mule_flagged": False,
               "mule_flag_count": 0, "name_score": 0.98}
CLEAN_TEMPORAL = {"first_time_payee": False, "amount_vs_avg_ratio": 1.0}


def _signals(result, layer):
    return [s["signal"] for s in result["attribution"][layer]]


def test_clean_transaction_fires_no_signals():
    r = build_attribution(CLEAN_TX, CLEAN_PAYEE, CLEAN_TEMPORAL)
    assert r["total_signals_fired"] == 0
    assert r["risk_score"] == 0
    assert r["verdict"] == "pass"
    assert r["grounded_assertions"] == []


def test_empty_inputs_use_safe_defaults():
    r = build_attribution({}, {}, {})
    assert r["total_signals_fired"] == 0
    assert r["verdict"] == "pass"


def test_layer1_paste_and_high_value_signals():
    r = build_attribution(
        {"input_method": "paste", "paste_trust_score": 0.08, "amount": 250000},
        CLEAN_PAYEE, CLEAN_TEMPORAL)
    assert _signals(r, "L1") == ["clipboard_paste", "high_value"]
    paste, high = r["attribution"]["L1"]
    assert paste["value"] == 0.08 and paste["threshold"] == 0.30
    assert high["value"] == 250000
    assert "250,000" in high["assertion"]


def test_high_value_threshold_is_exclusive():
    r = build_attribution({"input_method": "type", "amount": 100000},
                          CLEAN_PAYEE, CLEAN_TEMPORAL)
    assert _signals(r, "L1") == []


def test_layer2_signals_fire_with_values_and_thresholds():
    r = build_attribution(
        CLEAN_TX,
        {"vpa_age_days": 4, "mule_flagged": True, "mule_flag_count": 3,
         "name_score": 0.40},
        CLEAN_TEMPORAL)
    assert _signals(r, "L2") == ["vpa_age", "mule_consortium", "name_mismatch"]
    age, mule, name = r["attribution"]["L2"]
    assert age["value"] == 4 and "4 days old" in age["assertion"]
    assert mule["value"] == 3 and "3 banks flagged" in mule["assertion"]
    assert name["value"] == 0.40 and "0.40" in name["assertion"]


def test_mule_flag_count_defaults_to_one_when_absent():
    r = build_attribution(CLEAN_TX, {"vpa_age_days": 900, "mule_flagged": True,
                                     "name_score": 0.99}, CLEAN_TEMPORAL)
    mule = r["attribution"]["L2"][0]
    assert mule["signal"] == "mule_consortium"
    assert mule["value"] == 1


def test_layer3_first_time_payee_and_unusual_amount():
    r = build_attribution(
        CLEAN_TX, CLEAN_PAYEE,
        {"first_time_payee": True, "amount_vs_avg_ratio": 4.5})
    assert _signals(r, "L3") == ["first_time_payee", "unusual_amount"]
    assert r["attribution"]["L3"][1]["value"] == 4.5
    assert "4.5x" in r["attribution"]["L3"][1]["assertion"]


def test_risk_score_is_layer_weighted():
    # One of two L1 signals only: 0.35 * (1/2) = 0.175 -> 18
    r = build_attribution({"input_method": "paste", "amount": 1000},
                          CLEAN_PAYEE, CLEAN_TEMPORAL)
    assert r["risk_score"] == 18
    assert r["verdict"] == "pass"


def test_warn_verdict_band():
    # All three L2 signals: 0.45 -> 45
    r = build_attribution(
        CLEAN_TX,
        {"vpa_age_days": 4, "mule_flagged": True, "mule_flag_count": 2,
         "name_score": 0.1},
        CLEAN_TEMPORAL)
    assert r["risk_score"] == 45
    assert r["verdict"] == "warn"


def test_every_signal_firing_blocks_and_caps_at_100():
    r = build_attribution(
        {"input_method": "paste", "paste_trust_score": 0.05, "amount": 500000},
        {"vpa_age_days": 1, "mule_flagged": True, "mule_flag_count": 4,
         "name_score": 0.1},
        {"first_time_payee": True, "amount_vs_avg_ratio": 9.0})
    assert r["total_signals_fired"] == 7
    assert r["risk_score"] == 100
    assert r["verdict"] == "block"
    assert r["grounded_assertions"] == (r["attribution"]["L1"] +
                                        r["attribution"]["L2"] +
                                        r["attribution"]["L3"])
    assert all({"signal", "value", "threshold", "assertion"} <= set(s)
               for s in r["grounded_assertions"])
