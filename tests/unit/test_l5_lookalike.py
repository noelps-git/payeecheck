"""Unit tests for matchers/l5_lookalike.py — the VPA look-alike classifier."""
import os
import pickle

import pytest

from matchers import l5_lookalike
from matchers.l5_lookalike import _generate_vpa_pairs, _vpa_features, classify_vpa

HAS_SCAM_KW, HAS_FAKE_PSP, HAS_REAL_BRAND, N_DOTS, HANDLE_LEN, NUMERIC, \
    BRAND_PLUS_SCAM = range(7)


@pytest.fixture(autouse=True)
def _isolate_model(tmp_path, monkeypatch):
    """Never read or write the developer's trained model file."""
    monkeypatch.setattr(l5_lookalike, "_MODEL", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(l5_lookalike, "_cache", None)
    yield
    monkeypatch.setattr(l5_lookalike, "_cache", None)


# ── _vpa_features ────────────────────────────────────────────────────
def test_features_flag_brand_plus_scam_keyword():
    feats = _vpa_features("paytm.verify@axl")
    assert feats[HAS_SCAM_KW] == 1.0
    assert feats[HAS_FAKE_PSP] == 1.0
    assert feats[HAS_REAL_BRAND] == 1.0
    assert feats[BRAND_PLUS_SCAM] == 1.0


def test_features_of_a_legitimate_merchant_vpa():
    feats = _vpa_features("amazon.pay@apl")
    assert feats[HAS_SCAM_KW] == 0.0
    assert feats[HAS_FAKE_PSP] == 0.0
    assert feats[HAS_REAL_BRAND] == 1.0
    assert feats[BRAND_PLUS_SCAM] == 0.0


def test_features_are_normalised_counts():
    feats = _vpa_features("a.b.c-123@ybl")
    assert feats[N_DOTS] == pytest.approx(2 / 5)
    assert feats[HANDLE_LEN] == pytest.approx(len("a.b.c-123") / 30)
    assert feats[NUMERIC] == pytest.approx(3 / 9)


def test_features_handle_vpa_without_psp():
    feats = _vpa_features("support")
    assert feats[HAS_SCAM_KW] == 1.0
    assert feats[HAS_FAKE_PSP] == 0.0
    assert feats[NUMERIC] == 0.0


def test_features_are_fixed_width():
    assert len(_vpa_features("")) == len(_vpa_features("x.y@z")) == 7


def test_scam_keyword_needs_a_token_boundary():
    # 'supportive' is one token and must not match the 'support' keyword
    assert _vpa_features("supportive@ybl")[HAS_SCAM_KW] == 0.0
    assert _vpa_features("acct.support@ybl")[HAS_SCAM_KW] == 1.0


# ── dataset assembly ─────────────────────────────────────────────────
def test_generate_vpa_pairs_labels_fraud_rows(make_csv):
    path = make_csv([
        {"payee_vpa": "paytm.verify@axl", "label": "lookalike"},
        {"payee_vpa": "hdfc.support@paytrn", "label": "clipboard_scam"},
        {"payee_vpa": "suresh.123@ybl", "label": "clean"},
        {"payee_vpa": "", "label": "lookalike"},        # skipped: no VPA
    ])
    X, y = _generate_vpa_pairs(path)
    assert X.shape == (3, 7)
    assert list(y) == [1, 1, 0]


# ── train / classify_vpa ─────────────────────────────────────────────
@pytest.fixture
def trained():
    """Train on the bundled synthetic dataset — no network, no model download."""
    return l5_lookalike.train()


def test_train_writes_a_model_file(trained):
    assert os.path.exists(l5_lookalike._MODEL)
    with open(l5_lookalike._MODEL, "rb") as f:
        assert pickle.load(f).predict_proba is not None


def test_classify_separates_impersonation_from_legitimate_vpas(trained,
                                                               monkeypatch):
    monkeypatch.setattr(l5_lookalike, "_cache", trained)
    fraud = classify_vpa("paytm.verify@axl")
    clean = classify_vpa("suresh.123@ybl")
    assert fraud["lookalike_prob"] > clean["lookalike_prob"]
    assert fraud["is_lookalike"] is True
    assert clean["is_lookalike"] is False
    assert fraud["algorithm"] == "l5_lookalike_logreg"
    assert fraud["features"] == {"has_scam_keyword": True,
                                 "has_fake_psp": True,
                                 "has_real_brand": True,
                                 "brand_plus_scam": True}


def test_classify_trains_on_first_use_when_no_model_exists():
    assert not os.path.exists(l5_lookalike._MODEL)
    assert 0.0 <= classify_vpa("amazon.pay@apl")["lookalike_prob"] <= 1.0
    assert l5_lookalike._cache is not None
    assert os.path.exists(l5_lookalike._MODEL)


def test_classify_reuses_the_cached_model(trained, monkeypatch):
    monkeypatch.setattr(l5_lookalike, "_cache", trained)
    classify_vpa("paytm.verify@axl")
    assert l5_lookalike._cache is trained
