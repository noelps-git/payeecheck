"""Unit tests for risk_engine/calibrate_weights.py."""
import pytest

from risk_engine import calibrate_weights, risk_scorer
from risk_engine.calibrate_weights import (_coerce, calibrate,
                                           load_scored_dataset,
                                           precision_recall_f1)


def _result(score, fraud, signals=()):
    return {"risk_score": score, "true_fraud": fraud,
            "attribution": {"L1": [{"signal": s} for s in signals],
                            "L2": [], "L3": []}}


# ── _coerce ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("key,raw,expected", [
    ("mule_flagged", "True", True),
    ("mule_flagged", "false", False),
    ("vpa_age_days", "12.0", 12),
    ("prior_tx_count", "", 0),
    ("unique_senders_7d", "not-a-number", 0),
    ("amount", "1234.5", 1234.5),
    ("name_score", "", 0.0),
    ("name_score", "n/a", 0.0),
    ("label", "mule", "mule"),
])
def test_coerce_casts_csv_strings_by_column(key, raw, expected):
    assert _coerce(key, raw) == expected


# ── precision_recall_f1 ──────────────────────────────────────────────
def test_confusion_matrix_and_metrics():
    results = [_result(90, True), _result(80, False),
               _result(10, True), _result(5, False)]
    prec, rec, f1, tp, fp, fn, tn = precision_recall_f1(results, threshold=50)
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)
    assert prec == 0.5 and rec == 0.5 and f1 == 0.5


def test_metrics_are_zero_when_nothing_is_predicted_fraud():
    prec, rec, f1, tp, fp, fn, tn = precision_recall_f1(
        [_result(10, True), _result(1, False)], threshold=90)
    assert (prec, rec, f1) == (0.0, 0.0, 0.0)
    assert (tp, fp, fn, tn) == (0, 0, 1, 1)


def test_threshold_is_inclusive():
    assert precision_recall_f1([_result(50, True)], threshold=50)[3] == 1


# ── load_scored_dataset ──────────────────────────────────────────────
def test_load_scored_dataset_labels_and_skips_broken_rows(make_csv,
                                                          monkeypatch, capsys):
    def _fake_score(tx):
        if tx["payee_vpa"] == "boom@ybl":
            raise ValueError("unscoreable row")
        return {"risk_score": 80, "attribution": {"L1": [], "L2": [], "L3": []}}

    monkeypatch.setattr(risk_scorer, "score_transaction", _fake_score)
    path = make_csv([
        {"payee_vpa": "a@ybl", "label": "mule"},
        {"payee_vpa": "b@ybl", "label": "clean"},
        {"payee_vpa": "boom@ybl", "label": "mule"},
    ])

    results = load_scored_dataset(path)
    capsys.readouterr()
    assert [r["true_fraud"] for r in results] == [True, False]


# ── calibrate ────────────────────────────────────────────────────────
def test_calibrate_reports_metrics_and_weight_recommendations(monkeypatch,
                                                              make_csv, capsys):
    scored = (
        # missed fraud — only vpa_age fires, so its weight should go up
        [_result(20, True, ["vpa_age"])] * 10 +
        # false alarms on clipboard_paste — its weight should go down
        [_result(90, False, ["clipboard_paste"])] * 10 +
        [_result(95, True, ["mule_consortium"])] * 10 +
        [_result(5, False)] * 10
    )
    monkeypatch.setattr(calibrate_weights, "load_scored_dataset",
                        lambda csv_path: scored)

    summary = calibrate(make_csv([{"payee_vpa": "a@ybl"}]))
    capsys.readouterr()

    assert 25 <= summary["threshold"] < 80
    assert 0.0 <= summary["precision"] <= 1.0
    assert 0.0 <= summary["recall"] <= 1.0
    assert summary["baseline_f1"] == pytest.approx(0.5, abs=0.01)
    assert summary["precision"] == pytest.approx(0.5)
    assert summary["recall"] == pytest.approx(0.5)

    weights = summary["recommended_weights"]
    assert set(weights) == set(risk_scorer.SIGNAL_WEIGHTS)
    assert weights["vpa_age"] > risk_scorer.SIGNAL_WEIGHTS["vpa_age"]
    assert weights["clipboard_paste"] < risk_scorer.SIGNAL_WEIGHTS["clipboard_paste"]
    assert all(0.05 <= w <= 1.0 for w in weights.values())
    assert "synthetic" in summary["honest_note"]


def test_calibrate_picks_the_best_threshold(monkeypatch, make_csv, capsys):
    scored = [_result(70, True)] * 5 + [_result(45, False)] * 5
    monkeypatch.setattr(calibrate_weights, "load_scored_dataset",
                        lambda csv_path: scored)
    summary = calibrate(make_csv([{"payee_vpa": "a@ybl"}]))
    capsys.readouterr()
    assert summary["threshold"] == 50   # first threshold separating the classes
    assert summary["baseline_f1"] == 1.0
