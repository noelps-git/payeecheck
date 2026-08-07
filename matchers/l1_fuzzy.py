"""
l1_fuzzy.py — Level 1: Fuzzy String Matching

Jaro-Winkler + Token Sort Ratio. Character-level comparison only.
No understanding of meaning — fails on abbreviations and transliteration.
See: PayeeCheck Engineering Playbook, Level 1.
"""
from rapidfuzz import fuzz
import re

BUSINESS_TOKENS = {
    "pvt", "ltd", "limited", "llp", "inc", "corp",
    "enterprises", "solutions", "services", "industries",
    "holdings", "group", "ngo", "trust", "foundation"
}


def _normalise(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _is_business(name: str) -> bool:
    return bool(set(_normalise(name).split()) & BUSINESS_TOKENS)


def match(entered: str, actual: str) -> dict:
    n_e = _normalise(entered)
    n_a = _normalise(actual)

    jaro = fuzz.WRatio(n_e, n_a) / 100
    token = fuzz.token_sort_ratio(n_e, n_a) / 100
    score = round(max(jaro, token), 2)

    entered_biz = _is_business(entered)
    actual_biz = _is_business(actual)
    entity_mismatch = entered_biz != actual_biz

    if score >= 0.95:
        return _result("exact", score, "Names match precisely", entity_mismatch)
    elif score >= 0.75 and not entity_mismatch:
        return _result("close", score, "Names are similar", entity_mismatch)
    elif score >= 0.55 and entity_mismatch:
        et = "business" if entered_biz else "individual"
        at = "business" if actual_biz else "individual"
        return _result(
            "close", score,
            f"Name matches but entity type differs: {et} vs {at}",
            entity_mismatch,
        )
    else:
        return _result("no_match", score, "Names too different to confirm", entity_mismatch)


def _result(match_type, score, reason, entity_mismatch):
    return {
        "match": match_type,
        "score": score,
        "reason": reason,
        "entity_mismatch": entity_mismatch,
        "level": 1,
        "algorithm": "jaro_winkler + token_sort",
    }


if __name__ == "__main__":
    print(match("Suresh Kumar Pvt Ltd", "Suresh Kumar"))
    print(match("SBI", "State Bank of India"))
