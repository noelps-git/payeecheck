"""
l2_phonetic.py — Level 2: Phonetic Matching

Adds Double Metaphone as a boost signal on top of L1's fuzzy matching.
Solves transliteration variants (Mohammed/Mohammad) that L1 cannot.
See: PayeeCheck Engineering Playbook, Level 2.
"""
from rapidfuzz import fuzz
from metaphone import doublemetaphone
import re

BUSINESS_TOKENS = {
    "pvt", "ltd", "limited", "llp", "inc", "corp", "enterprises",
    "solutions", "services", "industries", "holdings", "group"
}


def _normalise(name):
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _is_business(name):
    return bool(set(_normalise(name).split()) & BUSINESS_TOKENS)


def _phonetic_token_score(name_a: str, name_b: str) -> float:
    tokens_a = _normalise(name_a).split()
    tokens_b = _normalise(name_b).split()
    if not tokens_a or not tokens_b:
        return 0.0

    def codes(tok):
        return set(c for c in doublemetaphone(tok) if c)

    hits = sum(
        any(codes(ta) & codes(tb) for tb in tokens_b)
        for ta in tokens_a
    )
    return hits / max(len(tokens_a), len(tokens_b))


def match(entered: str, actual: str) -> dict:
    n_e = _normalise(entered)
    n_a = _normalise(actual)

    fuzzy = max(fuzz.WRatio(n_e, n_a), fuzz.token_sort_ratio(n_e, n_a)) / 100
    ph_score = _phonetic_token_score(entered, actual)
    ph_matched = ph_score >= 0.5
    boost = 0.15 if ph_matched else 0.0
    score = round(min(1.0, fuzzy + boost), 2)

    entity_mismatch = _is_business(entered) != _is_business(actual)
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
