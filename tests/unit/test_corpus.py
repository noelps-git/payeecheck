"""Unit tests for matchers/corpus.py — the real seed corpus loader."""
import pytest

from matchers.corpus import (BANK_ALIASES, CORPORATE_GROUPS,
                             KNOWN_MERCHANT_VPAS, build_corpus,
                             load_from_mca_csv)


def test_build_corpus_is_sorted_deduplicated_and_complete():
    corpus = build_corpus()
    assert corpus == sorted(set(corpus))
    for canonical, aliases in BANK_ALIASES.items():
        assert canonical in corpus
        assert set(aliases) <= set(corpus)
    for name, _vpa, _psp in KNOWN_MERCHANT_VPAS:
        assert name in corpus
    for entities in CORPORATE_GROUPS.values():
        assert set(entities) <= set(corpus)


def test_load_from_mca_csv_reads_the_company_name_column(tmp_path):
    path = tmp_path / "mca.csv"
    path.write_text("CIN,COMPANY_NAME,COMPANY_STATUS\n"
                    "U1,Acme Industries Pvt Ltd,ACTIVE\n"
                    "U2,  Beta Traders  ,ACTIVE\n"
                    "U3,,ACTIVE\n")
    assert load_from_mca_csv(str(path)) == ["Acme Industries Pvt Ltd",
                                            "Beta Traders"]


def test_load_from_mca_csv_accepts_any_name_column(tmp_path):
    path = tmp_path / "mca.csv"
    path.write_text("cin,Registered Name\nU1,Gamma Holdings\n")
    assert load_from_mca_csv(str(path)) == ["Gamma Holdings"]


def test_load_from_mca_csv_raises_without_a_name_column(tmp_path):
    path = tmp_path / "mca.csv"
    path.write_text("cin,status\nU1,ACTIVE\n")
    with pytest.raises(ValueError, match="Could not find a name column"):
        load_from_mca_csv(str(path))


def test_load_from_mca_csv_falls_back_when_file_is_missing(tmp_path, capsys):
    assert load_from_mca_csv(str(tmp_path / "missing.csv")) == []
    assert "MCA file not found" in capsys.readouterr().out
