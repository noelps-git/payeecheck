"""
l1_fuzzy.py — Level 1: Fuzzy String Matching

Jaro-Winkler + Token Sort Ratio. Character-level comparison only.
No understanding of meaning — fails on abbreviations and transliteration.
See: PayeeCheck Engineering Playbook, Level 1.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.text import fuzzy_score, is_business, normalise


def match(entered: str, actual: str) -> dict:
    score = round(fuzzy_score(normalise(entered), normalise(actual)), 2)

    entered_biz = is_business(entered)
    actual_biz = is_business(actual)
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
