"""Unit tests for risk_engine/bilinear_fusion.py."""
import numpy as np
import pytest

from risk_engine import bilinear_fusion
from risk_engine.bilinear_fusion import (D, SIGNAL_NAMES, BilinearFusion,
                                         _row_to_phi, get_fusion)

IDX = {name: i for i, name in enumerate(SIGNAL_NAMES)}


@pytest.fixture(autouse=True)
def _isolate_model(tmp_path, monkeypatch):
    monkeypatch.setattr(bilinear_fusion, "_MODEL", str(tmp_path / "bi.pkl"))
    monkeypatch.setattr(bilinear_fusion, "_cache", None)
    yield
    monkeypatch.setattr(bilinear_fusion, "_cache", None)


# ── _row_to_phi ──────────────────────────────────────────────────────
def test_phi_is_zero_for_a_clean_row():
    phi = _row_to_phi({"mule_flagged": "False", "mule_flag_count": "0",
                       "sanction_match": "False", "is_lookalike": "False",
                       "name_score": "1", "vpa_age_days": "1000",
                       "input_method": "type", "amount": "2500",
                       "prior_tx_count": "50", "unique_senders_7d": "0"})
    assert phi.shape == (D,)
    assert not phi.any()


def test_phi_of_an_empty_row_uses_documented_defaults():
    phi = _row_to_phi({})
    assert phi[IDX["first_time_payee"]] == 1.0   # prior_tx_count defaults to 0
    assert phi[IDX["name_mismatch"]] == 0.0
    assert phi[IDX["vpa_age"]] == 0.0            # vpa_age_days defaults to 1000


def test_phi_flags_consortium_mule_only_at_two_flags():
    common = {"mule_flagged": "True", "bank_flags": "2"}
    assert _row_to_phi({**common, "mule_flag_count": "2"})[IDX["mule_consortium"]] == 1.0
    assert _row_to_phi({**common, "mule_flag_count": "1"})[IDX["mule_consortium"]] == 0.0
    assert _row_to_phi({"mule_flagged": "False",
                        "mule_flag_count": "3"})[IDX["mule_consortium"]] == 0.0


def test_phi_encodes_the_remaining_signals():
    phi = _row_to_phi({"sanction_match": "True", "is_lookalike": "True",
                       "name_score": "0.2", "vpa_age_days": "100",
                       "input_method": "paste", "amount": "150000",
                       "prior_tx_count": "0", "unique_senders_7d": "30"})
    assert phi[IDX["sanctions_pep_match"]] == 1.0
    assert phi[IDX["lookalike_vpa"]] == 1.0
    assert phi[IDX["name_mismatch"]] == pytest.approx(0.8)
    assert phi[IDX["vpa_age"]] == pytest.approx(0.9)
    assert phi[IDX["clipboard_paste"]] == 1.0
    assert phi[IDX["high_value"]] == 1.0
    assert phi[IDX["first_time_payee"]] == 1.0
    assert phi[IDX["unusual_amount"]] == 1.0
    assert phi[IDX["velocity_anomaly"]] == pytest.approx(0.5)
    assert phi[IDX["velocity_pattern"]] == pytest.approx(0.6)
    assert phi[IDX["gat_ring_score"]] == 0.0   # not scored per row


def test_phi_velocity_features_saturate():
    phi = _row_to_phi({"unique_senders_7d": "500"})
    assert phi[IDX["velocity_anomaly"]] == 1.0
    assert phi[IDX["velocity_pattern"]] == 1.0


def test_phi_tolerates_blank_csv_cells():
    phi = _row_to_phi({"amount": "", "prior_tx_count": "",
                       "unique_senders_7d": "", "vpa_age_days": "",
                       "name_score": "", "mule_flag_count": ""})
    assert np.isfinite(phi).all()


# ── fuse / fit ───────────────────────────────────────────────────────
def test_untrained_fusion_falls_back_to_the_signal_mean():
    phi = np.zeros(D, dtype=np.float32)
    phi[0] = 1.0
    assert BilinearFusion().fuse(phi) == pytest.approx(1 / D)


def test_fit_learns_to_separate_fraud_from_clean():
    fraud = np.zeros((5, D), dtype=np.float32)
    fraud[:, IDX["mule_consortium"]] = 1.0
    clean = np.zeros((5, D), dtype=np.float32)
    clean[:, IDX["vpa_age"]] = 0.1

    model = BilinearFusion()
    model.fit(np.vstack([fraud, clean]),
              np.array([1] * 5 + [0] * 5, dtype=np.float32),
              epochs=30, lr=0.05)

    assert model.trained is True
    assert model.fuse(fraud[0]) > 0.5 > model.fuse(clean[0])
    assert np.allclose(model.W, model.W.T)   # bilinear form stays symmetric


def test_fuse_output_is_a_bounded_probability():
    model = BilinearFusion()
    model.trained = True
    model.W = np.full((D, D), 50.0, dtype=np.float32)
    model.b = 0.0
    assert model.fuse(np.ones(D, dtype=np.float32)) == 1.0
    model.W = np.full((D, D), -50.0, dtype=np.float32)
    assert model.fuse(np.ones(D, dtype=np.float32)) == 0.0


def test_train_load_roundtrip_and_row_scoring(make_csv, capsys):
    path = make_csv(
        [{"payee_vpa": "m@ybl", "mule_flagged": "True", "mule_flag_count": 3,
          "bank_flags": 3, "label": "mule_consortium"}] * 3 +
        [{"payee_vpa": "c@ybl", "label": "clean"}] * 3)
    model = BilinearFusion.train(path)
    capsys.readouterr()
    assert model.trained is True

    reloaded = BilinearFusion.load()
    assert np.allclose(reloaded.W, model.W)
    assert reloaded.b == model.b

    fraud_score = reloaded.score_from_row(
        {"mule_flagged": "True", "mule_flag_count": "3", "bank_flags": "3"})
    assert 0.0 <= fraud_score <= 1.0


def test_get_fusion_trains_once_then_caches(make_csv, monkeypatch):
    path = make_csv([{"payee_vpa": "a@ybl", "label": "clean"},
                     {"payee_vpa": "b@ybl", "label": "mule"}])
    monkeypatch.setattr(BilinearFusion, "train",
                        classmethod(lambda cls, csv_path: cls()))
    first = get_fusion()
    assert get_fusion() is first


def test_get_fusion_loads_an_existing_model(make_csv):
    path = make_csv([{"payee_vpa": "a@ybl", "label": "clean"},
                     {"payee_vpa": "b@ybl", "label": "mule"}])
    trained = BilinearFusion.train(path)
    bilinear_fusion._cache = None
    loaded = get_fusion()
    assert loaded.trained is True
    assert np.allclose(loaded.W, trained.W)
