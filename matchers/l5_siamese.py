"""
matchers/l5_siamese.py — Name Matching L5: Lightweight Siamese Network.

Trained on synthetic labelled name pairs using TF-IDF char n-gram features
and contrastive loss. No transformer dependency — NumPy + sklearn only.

HONEST NOTE: Trained on synthetic pairs. Retrain on real bank data for
production use.

Usage:
    python matchers/l5_siamese.py          # train + smoke test
    from matchers.l5_siamese import match  # inference
"""
import os, sys, pickle, argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_HERE  = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "l5_model.pkl")
sys.path.insert(0, os.path.join(_HERE, ".."))
from common.datasets import read_rows, transactions_csv


def _generate_pairs(tx_csv: str) -> list:
    import random
    rows = read_rows(tx_csv)

    pairs = []
    # Positives from dataset
    for r in rows:
        en, an = r.get("entered_name",""), r.get("actual_name","")
        if en and an:
            pairs.append((en, an, 1))
        if an:
            pairs.append((an, an, 1))

    # Hard positives: known abbreviation/transliteration pairs
    HARD = [
        ("SBI","State Bank of India"), ("HDFC","HDFC Bank"),
        ("PNB","Punjab National Bank"), ("BOB","Bank of Baroda"),
        ("Mohammed","Muhammad"), ("Kavya","Kavitha"),
        ("Amit Kumar","A. Kumar"), ("Pooja Sharma","P. Sharma"),
        ("Suresh","Suresh Kumar"), ("Raj","Rajesh"),
    ]
    for a, b in HARD:
        pairs.extend([(a,b,1),(b,a,1)])

    # Negatives: random cross-pairings
    random.seed(42)
    names = list(set(r.get("actual_name","") for r in rows if r.get("actual_name","")))
    for _ in range(len(pairs)):
        a, b = random.sample(names, 2)
        if a != b:
            pairs.append((a, b, 0))

    random.shuffle(pairs)
    return pairs


class LightSiamese:
    def __init__(self, dim=64, margin=0.4):
        self.dim = dim; self.margin = margin
        self.W = None; self.vectorizer = None

    def _vec(self, name):
        v = self.vectorizer.transform([name]).toarray().astype(np.float32).flatten()
        p = self.W @ v
        return p / (np.linalg.norm(p) + 1e-8)

    def _raw(self, name):
        return self.vectorizer.transform([name]).toarray().astype(np.float32).flatten()

    def fit(self, pairs, epochs=30, lr=0.005):
        all_names = list(set(n for a,b,_ in pairs for n in [a,b]))
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2,4),
            max_features=512, sublinear_tf=True
        )
        self.vectorizer.fit(all_names)
        np.random.seed(42)
        self.W = np.random.randn(self.dim, 512).astype(np.float32) * 0.05
        for ep in range(epochs):
            loss_sum = 0.0
            np.random.shuffle(pairs)
            for a, b, lbl in pairs:
                ra, rb = self._raw(a), self._raw(b)
                va = (self.W @ ra); va /= (np.linalg.norm(va)+1e-8)
                vb = (self.W @ rb); vb /= (np.linalg.norm(vb)+1e-8)
                sim = float(va @ vb)
                d   = 1.0 - sim
                if lbl == 1:
                    loss_sum += d**2; g = -2.0*d
                else:
                    margin_loss = max(0.0, self.margin - d)
                    loss_sum += margin_loss**2
                    g = 2.0*margin_loss if d < self.margin else 0.0
                if abs(g) < 1e-6:
                    continue
                # Correct outer product gradient: dL/dW
                grad = g * np.outer(va - sim*vb, ra) + g * np.outer(vb - sim*va, rb)
                self.W -= lr * grad * 0.1
            if ep % 10 == 0:
                print(f"  ep {ep:3d} loss={loss_sum/len(pairs):.4f}")

    def match(self, a, b):
        va, vb = self._vec(a), self._vec(b)
        sim = float(cosine_similarity(va.reshape(1,-1), vb.reshape(1,-1))[0,0])
        return {"score": round(sim,3), "match": sim>=0.65, "algorithm":"l5_siamese"}

    def save(self, path):
        with open(path,"wb") as f:
            pickle.dump({"W":self.W,"vectorizer":self.vectorizer,
                         "dim":self.dim,"margin":self.margin}, f)

    @classmethod
    def load(cls, path):
        with open(path,"rb") as f: d = pickle.load(f)
        m = cls(d["dim"], d["margin"])
        m.W = d["W"]; m.vectorizer = d["vectorizer"]
        return m


_cache = None

def match(name_a: str, name_b: str) -> dict:
    global _cache
    if _cache is None:
        if not os.path.exists(_MODEL):
            train()
        _cache = LightSiamese.load(_MODEL)
    return _cache.match(name_a, name_b)

def train():
    pairs = _generate_pairs(transactions_csv())
    print(f"[l5] {len(pairs)} pairs, {sum(l for _,_,l in pairs)} positive")
    m = LightSiamese(dim=64); m.fit(pairs, epochs=30, lr=0.01)
    m.save(_MODEL); print(f"[l5] saved -> {_MODEL}")

if __name__ == "__main__":
    train()
    tests = [
        ("SBI","State Bank of India"), ("HDFC","HDFC Bank"),
        ("Mohammed Kumar","Muhammad Kumar"),
        ("Suresh Kumar","Ramesh Sharma"),
        ("Paytm Support","Rohan Patel"),
    ]
    m = LightSiamese.load(_MODEL)
    print()
    for a,b in tests:
        r = m.match(a,b)
        print(f"  {a!r:30s} vs {b!r:30s} -> {r['score']:.3f} {'MATCH' if r['match'] else '    -'}")
