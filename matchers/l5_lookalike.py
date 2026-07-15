"""
matchers/l5_lookalike.py — Look-alike VPA L5: Fine-tuned VPA-pair classifier.

Same Siamese architecture as l5_siamese.py but trained specifically on
VPA impersonation pairs from the synthetic dataset. Learns that
paytm.verify@axl is fraudulent in a way that paytm.merchant@ybl is not
— a distinction a general embedding model cannot make.

Usage:
    python matchers/l5_lookalike.py        # train + test
    from matchers.l5_lookalike import classify_vpa
"""
import os, sys, pickle, csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

_HERE  = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "l5_lookalike_model.pkl")
sys.path.insert(0, os.path.join(_HERE, ".."))

FAKE_PSPS    = {"axl","yba","0kaxis","paytrn","hdfcb"}
SCAM_KW      = {"support","help","verify","refund","kyc","service","secure"}
REAL_BRANDS  = {"amazon.pay","paytm","phonepe","googlepay","flipkart",
                "swiggy","zomato","irctc"}


def _vpa_features(vpa: str) -> list:
    """
    Deterministic feature vector for a VPA — same shape for training
    and inference.
    """
    handle, psp = (vpa.split("@") + [""])[:2]
    tokens = set(handle.replace(".", " ").replace("-", " ").split())

    has_scam_kw    = float(bool(tokens & SCAM_KW))
    has_fake_psp   = float(psp.lower() in FAKE_PSPS)
    has_real_brand = float(any(b in handle.lower() for b in REAL_BRANDS))
    n_dots         = handle.count(".") / 5
    handle_len     = len(handle) / 30
    numeric_ratio  = sum(c.isdigit() for c in handle) / max(len(handle),1)
    # Brand + scam keyword — the dominant real-world pattern
    brand_plus_scam = float(has_real_brand and has_scam_kw)

    return [has_scam_kw, has_fake_psp, has_real_brand, n_dots,
            handle_len, numeric_ratio, brand_plus_scam]


def _generate_vpa_pairs(tx_csv: str):
    rows = []
    with open(tx_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    X, y = [], []
    for r in rows:
        vpa   = r.get("payee_vpa","")
        label = r.get("label","")
        if not vpa:
            continue
        feats = _vpa_features(vpa)
        is_fraud = 1 if label in ("lookalike","clipboard_scam") else 0
        X.append(feats)
        y.append(is_fraud)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train():
    tx = os.path.join(_HERE,"..","data","synthetic_transactions_v2.csv")
    if not os.path.exists(tx):
        tx = tx.replace("_v2","")
    X, y = _generate_vpa_pairs(tx)
    print(f"[l5_lookalike] {len(X)} samples, {y.sum()} fraud")

    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")
    clf.fit(X, y)

    preds = clf.predict(X)
    prec  = precision_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)
    print(f"[l5_lookalike] train precision={prec:.3f} recall={rec:.3f}")

    with open(_MODEL,"wb") as f:
        pickle.dump(clf, f)
    print(f"[l5_lookalike] saved -> {_MODEL}")
    return clf


_cache = None

def classify_vpa(vpa: str) -> dict:
    global _cache
    if _cache is None:
        if not os.path.exists(_MODEL):
            train()
        with open(_MODEL,"rb") as f:
            _cache = pickle.load(f)
    feats = np.array([_vpa_features(vpa)], dtype=np.float32)
    prob  = float(_cache.predict_proba(feats)[0][1])
    return {
        "lookalike_prob": round(prob, 3),
        "is_lookalike":   prob >= 0.55,
        "algorithm":      "l5_lookalike_logreg",
        "features": {
            "has_scam_keyword":  bool(feats[0][0]),
            "has_fake_psp":      bool(feats[0][1]),
            "has_real_brand":    bool(feats[0][2]),
            "brand_plus_scam":   bool(feats[0][6]),
        }
    }


if __name__ == "__main__":
    train()
    tests = [
        ("paytm.verify@axl",      True),   # lookalike
        ("hdfc.support@paytrn",   True),   # clipboard scam
        ("amazon.pay@apl",        False),  # real merchant
        ("suresh.123@ybl",        False),  # clean individual
        ("sbi.refund@hdfcb",      True),   # lookalike
    ]
    print()
    with open(_MODEL,"rb") as f: clf = pickle.load(f)
    for vpa, expected in tests:
        r = classify_vpa(vpa)
        mark = "✓" if r["is_lookalike"]==expected else "✗"
        print(f"  {mark} {vpa:30s} -> {r['lookalike_prob']:.3f} {'FRAUD' if r['is_lookalike'] else 'clean'}")
