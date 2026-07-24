"""
risk_engine/bilinear_fusion.py — Bilinear Fusion Layer (SCAFDS Stage 5).

Replaces the hand-calibrated weighted sum in risk_scorer.py with a learned
bilinear interaction matrix that captures non-linear signal dependencies.

Key insight: vpa_age + fresh_vpa + high_value together are MORE suspicious
than the sum of their individual weights suggests. A bilinear layer learns
these multiplicative interactions from labelled outcome data.

Architecture:
    fused_score = sigmoid(phi^T @ W_bi @ phi + b)
    where phi is the fired-signal feature vector.

Training uses a proxy loss: binary cross-entropy against synthetic fraud
labels. Replace with real outcome labels once a bank pilot is live.

HONEST NOTE: The largest acknowledged architectural gap against SCAFDS.
This module closes the gap on synthetic data. Real performance requires
retraining on real labelled outcomes.

Usage:
    from risk_engine.bilinear_fusion import BilinearFusion
    bf = BilinearFusion.train("data/synthetic_transactions_v2.csv")
    score = bf.fuse(signal_vector)
"""
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.datasets import read_rows, transactions_csv

_HERE  = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "bilinear_model.pkl")

# Signal names in consistent order — the feature vector phi
SIGNAL_NAMES = [
    "mule_consortium", "sanctions_pep_match", "gat_ring_score",
    "lookalike_vpa",   "velocity_anomaly",    "name_mismatch",
    "vpa_age",         "clipboard_paste",     "high_value",
    "first_time_payee","unusual_amount",      "velocity_pattern",
]
D = len(SIGNAL_NAMES)

FRAUD_LABELS = {"mule","mule_consortium","lookalike","clipboard_scam",
                "ring_member","sanction"}


def _row_to_phi(row: dict) -> np.ndarray:
    """Convert a transaction row to a D-dimensional signal feature vector."""
    phi = np.zeros(D, dtype=np.float32)

    mf = str(row.get("mule_flagged","")).lower() == "true"
    fc = int(float(row.get("mule_flag_count", 0) or 0))
    bf = int(float(row.get("bank_flags", 0) or 0))

    phi[0] = float(mf and fc >= 2)           # mule_consortium
    phi[1] = float(str(row.get("sanction_match","")).lower()=="true")  # sanctions
    phi[2] = 0.0                               # gat (not scored per row)
    phi[3] = float(str(row.get("is_lookalike","")).lower()=="true")
    phi[4] = min(int(float(row.get("unique_senders_7d",0)or 0)),60)/60
    phi[5] = 1.0 - float(row.get("name_score","1") or 1)   # mismatch = high
    phi[6] = 1.0 - min(int(float(row.get("vpa_age_days",1000)or 1000)),1000)/1000
    phi[7] = float(row.get("input_method","type") == "paste")
    phi[8] = float(float(row.get("amount",0)or 0) > 100000)
    phi[9] = float(int(float(row.get("prior_tx_count",0)or 0)) == 0)
    phi[10]= float(float(row.get("amount",0)or 0) > 50000 and
                   int(float(row.get("prior_tx_count",0)or 0)) < 3)
    phi[11]= min(int(float(row.get("unique_senders_7d",0)or 0)),50)/50
    return phi


class BilinearFusion:
    def __init__(self):
        self.W  = None   # (D, D) bilinear matrix
        self.b  = 0.0
        self.trained = False

    def _sigmoid(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def fuse(self, phi: np.ndarray) -> float:
        """Bilinear fusion: phi^T W phi + b, passed through sigmoid."""
        if not self.trained:
            return float(phi.mean())
        score = float(phi @ self.W @ phi) + self.b
        return round(float(self._sigmoid(score)), 3)

    def fit(self, phis: np.ndarray, labels: np.ndarray,
            epochs=100, lr=0.005):
        np.random.seed(42)
        self.W = np.random.randn(D, D).astype(np.float32) * 0.01
        # Symmetrise — the bilinear form should be symmetric
        self.W = (self.W + self.W.T) / 2
        self.b = 0.0

        for ep in range(epochs):
            loss_sum = 0.0
            for phi, y in zip(phis, labels):
                s     = float(phi @ self.W @ phi) + self.b
                pred  = self._sigmoid(s)
                err   = pred - y
                loss_sum += -(y * np.log(pred + 1e-8) +
                              (1-y) * np.log(1-pred + 1e-8))
                # Gradient: dL/dW = err * (phi^T phi)
                outer = np.outer(phi, phi).astype(np.float32)
                self.W -= lr * err * outer
                self.W  = (self.W + self.W.T) / 2   # keep symmetric
                self.b -= lr * err

            if ep % 25 == 0:
                preds  = np.array([self._sigmoid(float(p@self.W@p)+self.b)
                                   for p in phis])
                pred_b = (preds >= 0.5).astype(int)
                acc    = (pred_b == labels).mean()
                print(f"  ep {ep:4d} loss={loss_sum/len(phis):.4f} acc={acc:.3f}")

        self.trained = True

    @classmethod
    def train(cls, csv_path: str) -> "BilinearFusion":
        phis, labels = [], []
        for row in read_rows(csv_path):
            phis.append(_row_to_phi(row))
            labels.append(1 if row.get("label","") in FRAUD_LABELS else 0)
        phis   = np.stack(phis)
        labels = np.array(labels, dtype=np.float32)
        print(f"[bilinear] {len(phis)} samples, {labels.sum():.0f} fraud ({labels.mean():.1%})")
        bf = cls()
        bf.fit(phis, labels, epochs=100, lr=0.005)
        with open(_MODEL,"wb") as f:
            pickle.dump({"W":bf.W,"b":bf.b,"trained":bf.trained}, f)
        print(f"[bilinear] model saved -> {_MODEL}")
        return bf

    @classmethod
    def load(cls) -> "BilinearFusion":
        with open(_MODEL,"rb") as f: d = pickle.load(f)
        bf = cls(); bf.W = d["W"]; bf.b = d["b"]; bf.trained = d["trained"]
        return bf

    def score_from_row(self, row: dict) -> float:
        phi = _row_to_phi(row)
        return self.fuse(phi)


_cache = None
def get_fusion() -> BilinearFusion:
    global _cache
    if _cache is None:
        if os.path.exists(_MODEL):
            _cache = BilinearFusion.load()
        else:
            _cache = BilinearFusion.train(transactions_csv())
    return _cache


if __name__ == "__main__":
    bf = BilinearFusion.train(transactions_csv())

    tests = [
        {"label":"mule_consortium","mule_flagged":"True","mule_flag_count":"3",
         "bank_flags":"3","sanction_match":"False","is_lookalike":"False",
         "name_score":"0.95","vpa_age_days":"45","input_method":"type",
         "amount":"80000","prior_tx_count":"5","unique_senders_7d":"30"},
        {"label":"clean","mule_flagged":"False","mule_flag_count":"0",
         "bank_flags":"0","sanction_match":"False","is_lookalike":"False",
         "name_score":"0.97","vpa_age_days":"800","input_method":"type",
         "amount":"2500","prior_tx_count":"200","unique_senders_7d":"2"},
        {"label":"lookalike","mule_flagged":"False","mule_flag_count":"0",
         "bank_flags":"0","sanction_match":"False","is_lookalike":"True",
         "name_score":"0.25","vpa_age_days":"5","input_method":"paste",
         "amount":"12000","prior_tx_count":"1","unique_senders_7d":"8"},
    ]
    print("\nSmoke test:")
    for t in tests:
        s = bf.score_from_row(t)
        print(f"  {t['label']:25s} -> fused score {s:.3f}")
