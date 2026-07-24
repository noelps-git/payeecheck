"""
l2_phonetic.py — Level 2: Phonetic Matching

Adds Double Metaphone as a boost signal on top of L1's fuzzy matching.
Solves transliteration variants (Mohammed/Mohammad) that L1 cannot.
See: PayeeCheck Engineering Playbook, Level 2.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.text import entity_mismatch as _entity_mismatch
from common.text import fuzzy_score, normalise, phonetic_token_score


def match(entered: str, actual: str) -> dict:
    fuzzy = fuzzy_score(normalise(entered), normalise(actual))
    ph_score = phonetic_token_score(entered, actual)
    ph_matched = ph_score >= 0.5
    boost = 0.15 if ph_matched else 0.0
    score = round(min(1.0, fuzzy + boost), 2)

    entity_mismatch = _entity_mismatch(entered, actual)
    ph_note = f"phonetic {ph_score:.2f}, +{boost:.2f} boost"

    if score >= 0.95:
        mt, reason = "exact", f"Precise match. {ph_note}"
    elif score >= 0.75 and not entity_mismatch:
        mt, reason = "close", f"Strong similarity. {ph_note}"
    elif score >= 0.55:
        mt, reason = "close", f"Partial match. {ph_note}"
    else:
        mt, reason = "no_match", f"Too different. {ph_note}"

    return {
        "match": mt,
        "score": score,
        "reason": reason,
        "entity_mismatch": entity_mismatch,
        "phonetic_score": ph_score,
        "level": 2,
        "algorithm": "jaro_winkler + double_metaphone",
    }


if __name__ == "__main__":
    print(match("Mohammed Riaz", "Mohammad Riaz"))
    print(match("SBI", "State Bank of India"))  # still fails — needs L3+
