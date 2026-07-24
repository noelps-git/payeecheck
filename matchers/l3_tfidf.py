"""
l3_tfidf.py — Level 3: TF-IDF Vector Matching

Character n-gram TF-IDF + cosine similarity. Fit on the REAL seed corpus
(NPCI banks, RBI list, known UPI merchants) from corpus.py — not a toy list.
Partially solves abbreviations via character n-gram overlap.
See: PayeeCheck Engineering Playbook, Level 3.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corpus import build_corpus  # real public data — see corpus.py
from common.text import entity_mismatch, fuzzy_score, normalise, phonetic_boost

# Fit on the REAL corpus built from NPCI / RBI / known merchant data.
# Extend this by calling corpus.load_from_mca_csv() once you've downloaded
# a real MCA export — see corpus.py docstring for instructions.
_CORPUS = build_corpus()
_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                       min_df=1, sublinear_tf=True)
_vec.fit(_CORPUS)


def _tfidf(a, b):
    v = _vec.transform([a, b])
    return float(cosine_similarity(v[0], v[1])[0][0])


def match(entered: str, actual: str) -> dict:
    n_e = normalise(entered)
    n_a = normalise(actual)

    fuzzy = fuzzy_score(n_e, n_a)
    tfidf = _tfidf(n_e, n_a)
    ph = phonetic_boost(entered, actual, 0.10)

    score = round(min(1.0, 0.5 * tfidf + 0.4 * fuzzy + ph), 2)
    em = entity_mismatch(entered, actual)
    sigs = {"fuzzy": round(fuzzy, 2), "tfidf": round(tfidf, 2), "ph_boost": ph}

    if score >= 0.90:
        mt = "exact"
    elif score >= 0.65 and not em:
        mt = "close"
    elif score >= 0.50:
        mt = "close"
    else:
        mt = "no_match"

    return {
        "match": mt, "score": score, "signals": sigs,
        "entity_mismatch": em, "level": 3,
        "algorithm": "tfidf_char_ngram + fuzzy + phonetic",
        "corpus_size": len(_CORPUS),
    }


if __name__ == "__main__":
    print(f"Fitted on {len(_CORPUS)} real entity names")
    print(match("SBI", "State Bank of India"))
    print(match("Krishna Enterprises", "Krishna Solutions"))
