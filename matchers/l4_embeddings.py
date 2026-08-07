"""
l4_embeddings.py — Level 4: Embedding-Based Matching

Sentence-transformer embeddings + cosine similarity. The step-change from
syntax to semantics — the model knows 'SBI' and 'State Bank of India' are
related because it was trained on billions of text pairs.
See: PayeeCheck Engineering Playbook, Level 4.

This is the recommended production level for most PayeeCheck use cases.
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from metaphone import doublemetaphone
import re

# Load once at import time — NOT inside match(). Reused across all calls.
# 'all-MiniLM-L6-v2': 384-dim, fast, strong on short text like names.
# For Hindi-script names use 'paraphrase-multilingual-MiniLM-L12-v2' instead.
print("[l4_embeddings] Loading sentence-transformer model (first run downloads ~90MB)...")
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[l4_embeddings] Model loaded.")

BUSINESS_TOKENS = {
    "pvt", "ltd", "limited", "llp", "inc", "corp",
    "enterprises", "solutions", "services", "holdings", "group"
}


def _normalise(name):
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _is_business(name):
    return bool(set(_normalise(name).split()) & BUSINESS_TOKENS)


def _semantic(a, b):
    embs = _MODEL.encode([a, b], normalize_embeddings=True)
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])


def _ph_boost(a, b):
    ca = set(c for c in doublemetaphone(a) if c)
    cb = set(c for c in doublemetaphone(b) if c)
    return 0.08 if ca & cb else 0.0


def match(entered: str, actual: str) -> dict:
    n_e = _normalise(entered)
    n_a = _normalise(actual)

    sem = _semantic(entered, actual)
    fuz = max(fuzz.WRatio(n_e, n_a), fuzz.token_sort_ratio(n_e, n_a)) / 100
    ph = _ph_boost(entered, actual)

    # Semantic anchors the decision, fuzzy validates, phonetic gives a small boost
    score = round(min(1.0, 0.65 * sem + 0.30 * fuz + ph), 2)
    em = _is_business(entered) != _is_business(actual)
    sigs = {"semantic": round(sem, 3), "fuzzy": round(fuz, 3), "phonetic_boost": ph}

    if score >= 0.88:
        mt = "exact"
    elif score >= 0.70 and not em:
        mt = "close"
    elif score >= 0.55:
        mt = "close"
    else:
        mt = "no_match"

    return {
        "match": mt, "score": score, "signals": sigs,
        "entity_mismatch": em, "level": 4,
        "algorithm": "sentence_transformer + fuzzy + phonetic",
    }


if __name__ == "__main__":
    print(match("SBI", "State Bank of India"))
    print(match("Reliance Jio", "Reliance Industries"))
    print(match("HDFC Bank", "HDFC Life"))
