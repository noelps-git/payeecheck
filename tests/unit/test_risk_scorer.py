"""Unit tests for risk_engine/risk_scorer.py."""
import pytest

from risk_engine import risk_scorer
from risk_engine.risk_scorer import (BLOCK_THRESHOLD, WARN_THRESHOLD,
                                     run_lookalike_check, run_mule_check,
                                     run_velocity_check, score_transaction)

# Captured before the autouse stubbing fixture replaces them.
_REAL_NAME_MATCH = risk_scorer.run_name_match
_REAL_SANCTIONS_CHECK = risk_scorer.run_sanctions_check

CLEAN_TX = {
    "tx_id": "T1", "entered_name": "Suresh Kumar", "actual_name": "Suresh Kumar",
    "payee_vpa": "suresh.kumar@ybl", "amount": 2500.0, "input_method": "type",
    "vpa_age_days": 900, "prior_tx_count": 40, "unique_senders_7d": 2,
    "mule_flagged": False, "mule_flag_count": 0,
}


def tx(**overrides):
    return {**CLEAN_TX, **overrides}


@pytest.fixture(autouse=True)
def _stub_slow_signals(monkeypatch):
    """Keep unit tests independent of the TF-IDF corpus and sanctions lists."""
    monkeypatch.setattr(risk_scorer, "run_name_match",
                        lambda entered, actual: {
                            "match": "exact" if entered == actual else "no_match",
                            "score": 1.0 if entered == actual else 0.2,
                            "level": 3})
    monkeypatch.setattr(risk_scorer, "run_sanctions_check",
                        lambda name: {"flag": "clear", "reason": "stub",
                                      "sanctions_score": 0.0, "pep_score": 0.0})


# ── run_lookalike_check ──────────────────────────────────────────────
def test_lookalike_ignores_vpas_without_a_protected_brand():
    assert run_lookalike_check("suresh.kumar@ybl") == {
        "flag": "no_brand_detected", "is_lookalike": False}


def test_lookalike_accepts_registered_brand_vpa():
    result = run_lookalike_check("amazon.pay@apl")
    assert result == {"flag": "legitimate", "is_lookalike": False,
                      "brand": "amazon"}


def test_lookalike_high_confidence_on_brand_plus_scam_suffix():
    result = run_lookalike_check("paytm.support@axl")
    assert result["flag"] == "brand_impersonation_high_confidence"
    assert result["is_lookalike"] is True
    assert result["brand"] == "paytm"
    assert result["suspicious_suffixes"] == ["support"]


def test_lookalike_suspected_when_brand_used_without_scam_suffix():
    result = run_lookalike_check("paytm.rewards2024@axl")
    assert result["flag"] == "brand_impersonation_suspected"
    assert result["is_lookalike"] is True
    assert result["suspicious_suffixes"] == []


def test_lookalike_handles_vpa_without_psp_handle():
    assert run_lookalike_check("PAYTM-verify")["is_lookalike"] is True


# ── run_mule_check / run_velocity_check ──────────────────────────────
def test_mule_check_passes_through_transaction_flags():
    assert run_mule_check("a@ybl", True, 3) == {"flagged": True,
                                                "flag_count": 3}


def test_velocity_clean_account_scores_zero():
    result = run_velocity_check(vpa_age_days=900, prior_tx_count=50,
                                unique_senders_7d=2, amount=2500)
    assert result == {"velocity_score": 0, "reasons": [],
                      "vpa_age_days": 900}


def test_velocity_stacks_fresh_vpa_fanin_and_high_value():
    result = run_velocity_check(vpa_age_days=4, prior_tx_count=0,
                                unique_senders_7d=19, amount=60000)
    assert result["velocity_score"] == 90
    assert len(result["reasons"]) == 3


@pytest.mark.parametrize("kwargs,expected", [
    (dict(vpa_age_days=10, prior_tx_count=0, unique_senders_7d=1,
          amount=1000), 30),                       # fresh VPA only
    (dict(vpa_age_days=900, prior_tx_count=50, unique_senders_7d=8,
          amount=1000), 30),                       # fan-in only
    (dict(vpa_age_days=13, prior_tx_count=50, unique_senders_7d=1,
          amount=50001), 30),                      # high value to new VPA only
    (dict(vpa_age_days=14, prior_tx_count=50, unique_senders_7d=7,
          amount=100000), 0),                      # all just under thresholds
])
def test_velocity_thresholds(kwargs, expected):
    assert run_velocity_check(**kwargs)["velocity_score"] == expected


# ── score_transaction ────────────────────────────────────────────────
def test_clean_transaction_passes_with_no_signals():
    result = score_transaction(tx())
    assert result["tx_id"] == "T1"
    assert result["verdict"] == "pass"
    assert result["risk_score"] == 0
    assert result["total_signals_fired"] == 0
    assert set(result["module_outputs"]) == {"name_match", "lookalike", "mule",
                                             "velocity", "sanctions", "gat"}


def test_consortium_mule_flag_forces_a_block():
    result = score_transaction(tx(mule_flagged=True, mule_flag_count=3))
    assert result["risk_score"] >= 90
    assert result["verdict"] == "block"
    signals = [s["signal"] for s in result["attribution"]["L2"]]
    assert "mule_consortium" in signals


def test_single_bank_mule_flag_does_not_trigger_the_override():
    result = score_transaction(tx(mule_flagged=True, mule_flag_count=1))
    assert result["risk_score"] < 90


def test_sanctions_hit_forces_the_hardest_block(monkeypatch):
    monkeypatch.setattr(risk_scorer, "run_sanctions_check",
                        lambda name: {"flag": "sanctions_match",
                                      "reason": "100% match: Entity Alpha",
                                      "sanctions_score": 1.0, "pep_score": 0.0})
    result = score_transaction(tx())
    assert result["risk_score"] >= 95
    assert result["verdict"] == "block"
    sanction = [s for s in result["attribution"]["L2"]
                if s["signal"] == "sanctions_pep_match"][0]
    assert sanction["value"] == 1.0
    assert sanction["assertion"] == "100% match: Entity Alpha"


def test_pep_hit_is_attributed_without_the_hard_override(monkeypatch):
    monkeypatch.setattr(risk_scorer, "run_sanctions_check",
                        lambda name: {"flag": "pep_match", "reason": "PEP",
                                      "sanctions_score": 0.1, "pep_score": 0.9})
    result = score_transaction(tx())
    signals = [s["signal"] for s in result["attribution"]["L2"]]
    assert "sanctions_pep_match" in signals
    assert result["risk_score"] < 95


def test_lookalike_and_velocity_signals_are_appended_to_layer2():
    result = score_transaction(tx(payee_vpa="paytm.verify@axl",
                                  vpa_age_days=4, prior_tx_count=0,
                                  unique_senders_7d=19, amount=60000,
                                  input_method="paste"))
    signals = [s["signal"] for s in result["attribution"]["L2"]]
    assert "lookalike_vpa" in signals
    assert "velocity_anomaly" in signals
    lookalike = [s for s in result["attribution"]["L2"]
                 if s["signal"] == "lookalike_vpa"][0]
    assert "paytm" in lookalike["assertion"]
    velocity = [s for s in result["attribution"]["L2"]
                if s["signal"] == "velocity_anomaly"][0]
    assert velocity["value"] == 90
    assert result["total_signals_fired"] == 6


def test_gat_signal_wired_into_layer2_when_available(monkeypatch):
    monkeypatch.setattr(risk_scorer, "_gat_score_vpa",
                        lambda vpa: {"gat_score": 0.91, "is_available": True})
    result = score_transaction(tx())
    gat = [s for s in result["attribution"]["L2"]
           if s["signal"] == "gat_ring_score"][0]
    assert gat["value"] == 0.91
    assert gat["threshold"] == 0.55
    assert "91%" in gat["assertion"]


def test_gat_score_below_threshold_is_not_attributed(monkeypatch):
    monkeypatch.setattr(risk_scorer, "_gat_score_vpa",
                        lambda vpa: {"gat_score": 0.4, "is_available": True})
    signals = [s["signal"]
               for s in score_transaction(tx())["attribution"]["L2"]]
    assert "gat_ring_score" not in signals


def test_unavailable_gat_model_is_skipped(monkeypatch):
    monkeypatch.setattr(risk_scorer, "_gat_score_vpa",
                        lambda vpa: {"gat_score": 0.99, "is_available": False})
    signals = [s["signal"]
               for s in score_transaction(tx())["attribution"]["L2"]]
    assert "gat_ring_score" not in signals


def test_paste_and_first_time_payee_populate_layers_1_and_3():
    result = score_transaction(tx(input_method="paste", prior_tx_count=0,
                                  amount=150000))
    l1 = [s["signal"] for s in result["attribution"]["L1"]]
    l3 = [s["signal"] for s in result["attribution"]["L3"]]
    assert l1 == ["clipboard_paste", "high_value"]
    assert "first_time_payee" in l3
    assert "unusual_amount" in l3   # amount/10000 = 15 > 3


def test_true_label_is_passed_through_for_benchmark_rows():
    assert score_transaction(tx(label="mule"))["true_label"] == "mule"
    assert score_transaction(tx())["true_label"] is None


def test_verdict_bands_follow_the_documented_thresholds():
    assert WARN_THRESHOLD < BLOCK_THRESHOLD
    for transaction in (tx(), tx(mule_flagged=True, mule_flag_count=2),
                        tx(payee_vpa="paytm.verify@axl", vpa_age_days=2,
                           prior_tx_count=0, unique_senders_7d=20)):
        result = score_transaction(transaction)
        expected = ("block" if result["risk_score"] >= BLOCK_THRESHOLD else
                    "warn" if result["risk_score"] >= WARN_THRESHOLD else "pass")
        assert result["verdict"] == expected


def test_run_name_match_uses_the_tfidf_matcher():
    result = _REAL_NAME_MATCH("State Bank of India", "State Bank of India")
    assert result["match"] == "exact"
    assert result["level"] == 3
    assert result["score"] == pytest.approx(1.0, abs=0.05)


def test_run_sanctions_check_delegates_to_the_screening_module():
    result = _REAL_SANCTIONS_CHECK("Suresh Kumar")
    assert result["flag"] == "clear"
