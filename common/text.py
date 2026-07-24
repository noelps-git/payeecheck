"""
common/text.py — Name normalisation and similarity primitives.

Shared by every name matcher (L1-L4) so the levels differ only in the
signal each one adds, not in how names are cleaned or how the baseline
fuzzy/phonetic signals are computed.
"""
import re

from metaphone import doublemetaphone
from rapidfuzz import fuzz

BUSINESS_TOKENS = {
    "pvt", "ltd", "limited", "llp", "inc", "corp",
    "enterprises", "solutions", "services", "industries",
    "holdings", "group", "ngo", "trust", "foundation",
}


def normalise(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = (name or "").lower()
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def is_business(name: str) -> bool:
    return bool(set(normalise(name).split()) & BUSINESS_TOKENS)


def entity_mismatch(entered: str, actual: str) -> bool:
    return is_business(entered) != is_business(actual)


def fuzzy_score(normalised_a: str, normalised_b: str) -> float:
    """Best of weighted-ratio and token-sort ratio, scaled to 0-1."""
    return max(
        fuzz.WRatio(normalised_a, normalised_b),
        fuzz.token_sort_ratio(normalised_a, normalised_b),
    ) / 100


def phonetic_codes(token: str) -> set:
    return set(c for c in doublemetaphone(token) if c)


def phonetic_boost(a: str, b: str, boost: float) -> float:
    """`boost` when the two names share any Double Metaphone code, else 0."""
    return boost if phonetic_codes(a) & phonetic_codes(b) else 0.0


def phonetic_token_score(name_a: str, name_b: str) -> float:
    """Fraction of tokens in the longer name with a phonetic counterpart."""
    tokens_a = normalise(name_a).split()
    tokens_b = normalise(name_b).split()
    if not tokens_a or not tokens_b:
        return 0.0
    hits = sum(
        any(phonetic_codes(ta) & phonetic_codes(tb) for tb in tokens_b)
        for ta in tokens_a
    )
    return hits / max(len(tokens_a), len(tokens_b))
