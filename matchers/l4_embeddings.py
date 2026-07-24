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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.text import entity_mismatch, fuzzy_score, normalise, phonetic_boost

# Load once at import time — NOT inside match(). Reused across all calls.
# 'all-MiniLM-L6-v2': 384-dim, fast, strong on short text like names.
# For Hindi-script names use 'paraphrase-multilingual-MiniLM-L12-v2' instead.
print("[l4_embeddings] Loading sentence-transformer model (first run downloads ~90MB)...")
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[l4_embeddings] Model loaded.")


def _semantic(a, b):
    embs = _MODEL.encode([a, b], normalize_embeddings=True)
    return float(cosine_similarity([embs[0]], [embs[1]])[0][0])


def match(entered: str, actual: str) -> dict:
    sem = _semantic(entered, actual)
    fuz = fuzzy_score(normalise(entered), normalise(actual))
    ph = phonetic_boost(entered, actual, 0.08)

    # Semantic anchors the decision, fuzzy validates, phonetic gives a small boost
    score = round(min(1.0, 0.65 * sem + 0.30 * fuz + ph), 2)
    em = entity_mismatch(entered, actual)
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
