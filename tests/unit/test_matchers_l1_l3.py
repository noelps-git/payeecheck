"""Unit tests for the offline name matchers (L1 fuzzy, L2 phonetic, L3 TF-IDF).

L4/L5 are excluded on purpose: they download sentence-transformer weights.
"""
import pytest

from matchers import l1_fuzzy, l2_phonetic, l3_tfidf


# ── shared normalisation helpers ─────────────────────────────────────
@pytest.mark.parametrize("module", [l1_fuzzy, l2_phonetic, l3_tfidf])
def test_normalise_strips_punctuation_and_case(module):
    assert module._normalise("  M/s. SURESH   Kumar!  ") == "m s suresh kumar"


@pytest.mark.parametrize("module", [l1_fuzzy, l2_phonetic, l3_tfidf])
@pytest.mark.parametrize("name,expected", [
    ("Krishna Enterprises", True),
    ("Acme Pvt Ltd", True),
    ("Suresh Kumar", False),
    ("Limitless Kumar", False),   # token boundary, not substring
])
def test_is_business_detects_entity_tokens(module, name, expected):
    assert module._is_business(name) is expected


# ── L1 fuzzy ─────────────────────────────────────────────────────────
def test_l1_exact_match():
    result = l1_fuzzy.match("Suresh Kumar", "suresh kumar")
    assert result["match"] == "exact"
    assert result["score"] == 1.0
    assert result["entity_mismatch"] is False
    assert result["level"] == 1
    assert result["algorithm"] == "jaro_winkler + token_sort"


def test_l1_token_order_is_ignored():
    assert l1_fuzzy.match("Kumar Suresh", "Suresh Kumar")["match"] == "exact"


def test_l1_reports_entity_type_mismatch():
    result = l1_fuzzy.match("Suresh Kumar Pvt Ltd", "Suresh Kumar")
    assert result["entity_mismatch"] is True
    assert "entity type differs: business vs individual" in result["reason"]


def test_l1_cannot_resolve_abbreviations():
    result = l1_fuzzy.match("SBI", "State Bank of India")
    assert result["match"] == "no_match"
    assert result["reason"] == "Names too different to confirm"


def test_l1_close_match_on_similar_surname():
    result = l1_fuzzy.match("Anita Sharma", "Anita Verma")
    assert result["match"] == "close"
    assert 0.75 <= result["score"] < 0.95


# ── L2 phonetic ──────────────────────────────────────────────────────
def test_l2_boosts_transliteration_variants():
    result = l2_phonetic.match("Mohammed Riaz", "Mohammad Riaz")
    assert result["match"] in ("exact", "close")
    assert result["phonetic_score"] == 1.0
    assert "+0.15 boost" in result["reason"]
    assert result["level"] == 2


def test_l2_applies_no_boost_to_unrelated_names():
    result = l2_phonetic.match("Suresh Kumar", "Priya Nair")
    assert result["phonetic_score"] < 0.5
    assert "+0.00 boost" in result["reason"]
    assert result["match"] == "no_match"


def test_l2_phonetic_score_is_zero_for_empty_input():
    assert l2_phonetic._phonetic_token_score("", "Suresh") == 0.0
    assert l2_phonetic._phonetic_token_score("Suresh", "") == 0.0


def test_l2_still_fails_on_abbreviations():
    assert l2_phonetic.match("SBI", "State Bank of India")["match"] == "no_match"


def test_l2_scores_are_capped_at_one():
    assert l2_phonetic.match("Suresh Kumar", "Suresh Kumar")["score"] == 1.0


# ── L3 TF-IDF ────────────────────────────────────────────────────────
def test_l3_exact_match_reports_signals_and_corpus():
    result = l3_tfidf.match("State Bank of India", "State Bank of India")
    assert result["match"] == "exact"
    assert result["level"] == 3
    assert set(result["signals"]) == {"fuzzy", "tfidf", "ph_boost"}
    assert result["corpus_size"] > 0


def test_l3_recovers_partial_abbreviation_signal():
    result = l3_tfidf.match("SBI", "State Bank of India")
    assert result["signals"]["tfidf"] > 0
    assert result["match"] in ("close", "no_match")


def test_l3_flags_entity_mismatch():
    result = l3_tfidf.match("Krishna Enterprises", "Krishna Kumar")
    assert result["entity_mismatch"] is True


def test_l3_rejects_unrelated_names():
    assert l3_tfidf.match("Suresh Kumar", "Priya Nair")["match"] == "no_match"
