"""
l5_siamese.py — Level 5: Fine-Tuned Siamese Network Matcher

Uses the model fine-tuned by train_l5.py. Falls back to the base
all-MiniLM-L6-v2 model (same as Level 4) with a clear warning if no
fine-tuned model is found yet — so this file never crashes, it just
tells you what to run first.
See: PayeeCheck Engineering Playbook, Level 5.
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import re
import os

BUSINESS_TOKENS = {
    "pvt", "ltd", "limited", "llp", "inc", "corp",
    "enterprises", "solutions", "services", "holdings", "group"
}

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "payeecheck_name_model")

if os.path.exists(_MODEL_PATH):
    print(f"[l5_siamese] Loading fine-tuned model from {_MODEL_PATH}")
    _MODEL = SentenceTransformer(_MODEL_PATH)
    _IS_FINETUNED = True
else:
    print("[l5_siamese] WARNING: No fine-tuned model found at "
          f"{_MODEL_PATH}")
    print("[l5_siamese] Run `python matchers/train_l5.py` first to create one.")
    print("[l5_siamese] Falling back to base all-MiniLM-L6-v2 (same as Level 4).")
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _IS_FINETUNED = False


def _normalise(name):
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _is_business(name):
    return bool(set(_normalise(name).split()) & BUSINESS_TOKENS)


def match(entered: str, actual: str) -> dict:
    embs = _MODEL.encode([entered, actual], normalize_embeddings=True)
    sem = float(cosine_similarity([embs[0]], [embs[1]])[0][0])

    n_e = _normalise(entered)
    n_a = _normalise(actual)
    fuz = max(fuzz.WRatio(n_e, n_a), fuzz.token_sort_ratio(n_e, n_a)) / 100

    # Fine-tuned semantic score is more trustworthy -> higher weight
    score = round(min(1.0, 0.80 * sem + 0.20 * fuz), 2)
    em = _is_business(entered) != _is_business(actual)
    sigs = {"semantic_finetuned": round(sem, 3), "fuzzy": round(fuz, 3)}

    if score >= 0.85:
        mt = "exact"
    elif score >= 0.65:
        mt = "close"
    else:
        mt = "no_match"

    return {
        "match": mt, "score": score, "signals": sigs,
        "entity_mismatch": em, "level": 5,
        "algorithm": "siamese_finetuned + fuzzy" if _IS_FINETUNED
                     else "siamese_NOT_finetuned_fallback + fuzzy",
        "is_finetuned": _IS_FINETUNED,
    }


if __name__ == "__main__":
    print(match("SBI", "State Bank of India"))
    print(match("HDFC Bank", "HDFC Life"))
