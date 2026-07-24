"""Unit tests for risk_engine/sanctions_screening.py and sanctions_loader.py."""
import json

import pytest

from risk_engine import sanctions_loader, sanctions_screening
from risk_engine.sanctions_screening import _norm, check_sanctions


@pytest.fixture
def seeded_lists(monkeypatch):
    monkeypatch.setattr(sanctions_screening, "_SANCTIONS",
                        ["Lashkar-e-Taiba", "Fictional Shell Company Gamma"])
    monkeypatch.setattr(sanctions_screening, "_PEP",
                        ["Example Former Minister Delta"])


# ── check_sanctions ──────────────────────────────────────────────────
def test_norm_lowercases_and_collapses_whitespace():
    assert _norm("  Suresh   KUMAR ") == "suresh kumar"


def test_exact_sanctions_name_is_flagged(seeded_lists):
    result = check_sanctions("lashkar-e-taiba")
    assert result["flag"] == "sanctions_match"
    assert result["sanctions_score"] == 1.0
    assert "Lashkar-e-Taiba" in result["reason"]
    assert result["algorithm"] == "fuzzy_token_sort"


def test_token_order_does_not_defeat_the_screen(seeded_lists):
    assert check_sanctions("Gamma Company Shell Fictional")["flag"] == \
        "sanctions_match"


def test_pep_match_when_no_sanctions_hit(seeded_lists):
    result = check_sanctions("Example Former Minister Delta")
    assert result["flag"] == "pep_match"
    assert result["pep_score"] == 1.0
    assert "PEP match" in result["reason"]


def test_sanctions_take_precedence_over_pep(monkeypatch):
    monkeypatch.setattr(sanctions_screening, "_SANCTIONS", ["Ravi Shankar"])
    monkeypatch.setattr(sanctions_screening, "_PEP", ["Ravi Shankar"])
    assert check_sanctions("Ravi Shankar")["flag"] == "sanctions_match"


def test_unrelated_name_is_clear(seeded_lists):
    result = check_sanctions("Suresh Kumar")
    assert result["flag"] == "clear"
    assert result["reason"] == "No match above threshold"
    assert result["sanctions_score"] < 0.85


def test_threshold_is_configurable(seeded_lists):
    near_miss = "Lashkar e Taib"
    assert check_sanctions(near_miss, threshold=0.99)["flag"] == "clear"
    assert check_sanctions(near_miss, threshold=0.5)["flag"] == "sanctions_match"


def test_bundled_list_metadata_is_reported():
    result = check_sanctions("Suresh Kumar")
    assert isinstance(result["using_real_data"], bool)
    assert result["data_source"]


# ── _load_lists ──────────────────────────────────────────────────────
def test_load_lists_reads_generated_data_file(tmp_path, monkeypatch):
    data_file = tmp_path / "sanctions_data.json"
    data_file.write_text(json.dumps({
        "sanctions": ["Entity One"], "pep": ["Person Two"],
        "counts": {"total_unique": 2}, "generated_at": "2026-01-01T00:00:00Z",
    }))
    monkeypatch.setattr(sanctions_screening, "_DATA_FILE", str(data_file))
    sanctions, pep, source, is_real = sanctions_screening._load_lists()
    assert sanctions == ["Entity One"]
    assert pep == ["Person Two"]
    assert is_real is True
    assert "2 entries" in source and "2026-01-01T00:00:00Z" in source


def test_load_lists_falls_back_to_illustrative_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(sanctions_screening, "_DATA_FILE",
                        str(tmp_path / "missing.json"))
    sanctions, pep, source, is_real = sanctions_screening._load_lists()
    assert sanctions == sanctions_screening._SANCTIONS_SEED
    assert pep == sanctions_screening._PEP_SEED
    assert is_real is False
    assert "illustrative seed" in source


# ── sanctions_loader ─────────────────────────────────────────────────
def test_rbi_seed_is_non_empty_and_unique():
    seed = sanctions_loader._rbi_seed()
    assert len(seed) == len(set(seed))
    assert "Lashkar-e-Taiba" in seed


def test_fetch_helpers_return_empty_without_requests(monkeypatch):
    monkeypatch.setattr(sanctions_loader, "REQUESTS_OK", False)
    assert sanctions_loader._fetch_ofac() == []
    assert sanctions_loader._fetch_un() == []


def test_fetch_helpers_swallow_network_errors(monkeypatch):
    class _Boom:
        @staticmethod
        def get(*args, **kwargs):
            raise RuntimeError("network down")

    monkeypatch.setattr(sanctions_loader, "REQUESTS_OK", True)
    monkeypatch.setattr(sanctions_loader, "requests", _Boom, raising=False)
    assert sanctions_loader._fetch_ofac() == []
    assert sanctions_loader._fetch_un() == []


def test_fetch_ofac_parses_sdn_csv(monkeypatch):
    class _Resp:
        text = ('ent_num,SDN_Name,SDN_Type\n'
                '1,"Entity Alpha",individual\n'
                '2,"Ab",individual\n'          # too short -> dropped
                '3\n')                          # malformed -> skipped

        def raise_for_status(self):
            pass

    monkeypatch.setattr(sanctions_loader, "REQUESTS_OK", True)
    monkeypatch.setattr(sanctions_loader, "requests",
                        type("R", (), {"get": staticmethod(lambda *a, **k: _Resp())}),
                        raising=False)
    assert sanctions_loader._fetch_ofac() == ["Entity Alpha"]


def test_fetch_un_joins_name_parts(monkeypatch):
    class _Resp:
        text = ("<FIRST_NAME>Alpha</FIRST_NAME><SECOND_NAME>Beta</SECOND_NAME>"
                "<THIRD_NAME>Gamma</THIRD_NAME>"
                "<FIRST_NAME>Delta</FIRST_NAME><SECOND_NAME></SECOND_NAME>"
                "<THIRD_NAME></THIRD_NAME>")

        def raise_for_status(self):
            pass

    monkeypatch.setattr(sanctions_loader, "REQUESTS_OK", True)
    monkeypatch.setattr(sanctions_loader, "requests",
                        type("R", (), {"get": staticmethod(lambda *a, **k: _Resp())}),
                        raising=False)
    assert sanctions_loader._fetch_un() == ["Alpha Beta Gamma", "Delta"]


def test_build_deduplicates_and_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(sanctions_loader, "_fetch_ofac",
                        lambda: ["Entity Alpha", "entity alpha "])
    monkeypatch.setattr(sanctions_loader, "_fetch_un", lambda: ["Entity Beta"])
    monkeypatch.setattr(sanctions_loader, "_rbi_seed", lambda: ["ISIS"])

    out = tmp_path / "out.json"
    sanctions_loader.build(str(out))

    data = json.loads(out.read_text())
    assert data["sanctions"] == ["Entity Alpha", "Entity Beta", "ISIS"]
    assert data["counts"] == {"ofac": 2, "un": 1, "rbi_seed": 1,
                              "total_unique": 3}
    assert data["pep"] == []
    assert data["generated_at"].endswith("Z")
