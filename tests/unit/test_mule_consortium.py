"""Unit tests for risk_engine/mule_consortium.py."""
import pytest

from risk_engine import mule_consortium
from risk_engine.mule_consortium import ConsortiumDB, get_db


@pytest.fixture(autouse=True)
def _reset_db_cache():
    mule_consortium._db_cache = None
    yield
    mule_consortium._db_cache = None


@pytest.fixture
def db(make_csv):
    path = make_csv([
        {"payee_vpa": "coord@ybl", "mule_flagged": "True", "bank_flags": 3,
         "fraud_ring_id": "RING1"},
        {"payee_vpa": "satellite@ybl", "fraud_ring_id": "RING1"},
        {"payee_vpa": "single@ybl", "mule_flagged": "True", "bank_flags": 1},
        {"payee_vpa": "flagged_no_banks@ybl", "mule_flagged": "True",
         "bank_flags": 0},
        {"payee_vpa": "clean@ybl"},
        {"payee_vpa": ""},
    ])
    return ConsortiumDB.from_csv(path)


def test_from_csv_indexes_flags_and_rings(db):
    assert db.flags["coord@ybl"] == ["SYNTH_BANK_1", "SYNTH_BANK_2",
                                     "SYNTH_BANK_3"]
    assert "flagged_no_banks@ybl" not in db.flags   # flagged but zero banks
    assert "clean@ybl" not in db.flags
    assert db.rings == {"coord@ybl": "RING1", "satellite@ybl": "RING1"}
    assert db.ring_vpas["RING1"] == ["coord@ybl", "satellite@ybl"]


def test_lookup_confirms_consortium_at_two_banks(db):
    result = db.lookup("coord@ybl")
    assert result["vpa"] == "coord@ybl"
    assert result["direct_flag_count"] == 3
    assert result["consortium_confirmed"] is True
    assert result["ring_id"] == "RING1"
    assert result["effective_score"] == pytest.approx(0.75)
    assert result["source"] == "consortium_db_synthetic"


def test_single_bank_flag_is_not_consortium_confirmed(db):
    result = db.lookup("single@ybl")
    assert result["direct_flag_count"] == 1
    assert result["consortium_confirmed"] is False
    assert result["propagated_risk_score"] == 0.0
    assert result["effective_score"] == pytest.approx(0.25)


def test_unflagged_ring_member_inherits_propagated_risk(db):
    result = db.lookup("satellite@ybl")
    assert result["direct_flag_count"] == 0
    assert result["consortium_confirmed"] is False
    assert result["ring_id"] == "RING1"
    # 3 flags * 0.5 decay, capped at 1.0
    assert result["propagated_risk_score"] == 1.0
    assert result["effective_score"] == 1.0


def test_propagation_decays_with_flag_count(make_csv):
    path = make_csv([
        {"payee_vpa": "weak_coord@ybl", "mule_flagged": "True", "bank_flags": 1,
         "fraud_ring_id": "RING2"},
        {"payee_vpa": "satellite@ybl", "fraud_ring_id": "RING2"},
    ])
    db = ConsortiumDB.from_csv(path)
    assert db.lookup("satellite@ybl")["propagated_risk_score"] == 0.5
    assert db._propagate("satellite@ybl", decay=0.2) == 0.2


def test_lookup_of_unknown_vpa_is_all_clear(db):
    result = db.lookup("never.seen@ybl")
    assert result["direct_flag_count"] == 0
    assert result["flagging_banks"] == []
    assert result["ring_id"] is None
    assert result["effective_score"] == 0.0


def test_propagate_ignores_rings_with_no_flagged_members(make_csv):
    path = make_csv([
        {"payee_vpa": "a@ybl", "fraud_ring_id": "RING3"},
        {"payee_vpa": "b@ybl", "fraud_ring_id": "RING3"},
    ])
    db = ConsortiumDB.from_csv(path)
    assert db._propagate("a@ybl") == 0.0


def test_get_db_caches_instance(make_csv):
    path = make_csv([{"payee_vpa": "a@ybl"}])
    first = get_db(path)
    assert get_db(path) is first


def test_get_db_defaults_to_bundled_dataset():
    db = get_db()
    assert len(db.flags) > 0
