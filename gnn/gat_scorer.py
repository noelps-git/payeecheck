"""
gat_scorer.py — Lightweight inference wrapper for the trained GAT.

Loads gat_mule_detector.pt once at module import and exposes a single
score_vpa() function the Risk Scorer calls as a 7th signal.

The GAT produces a fraud probability (0.0–1.0) per node based on the
account's position in the shared-attribute entity graph. The risk scorer
treats this as an additional weighted signal, not a hard override.

HONEST NOTE: The model was trained on synthetic ring data. Retrain on
real consortium data once a bank pilot is live.
"""
import os, torch, numpy as np

_HERE   = os.path.dirname(__file__)
_WEIGHTS = os.path.join(_HERE, "gat_mule_detector.pt")

_model    = None
_graph    = None
_vpa_idx  = {}   # vpa -> node index for quick lookup

def _load():
    global _model, _graph, _vpa_idx
    if _model is not None:
        return True
    try:
        import sys
        sys.path.insert(0, os.path.join(_HERE, ".."))
        from gnn.gat_mule_detector import EdgeFeatureGAT, build_graph_from_rings
        from torch_geometric.data import Data

        tx_path   = os.path.join(_HERE, "..", "data", "synthetic_transactions_v2.csv")
        ring_path = os.path.join(_HERE, "..", "data", "synthetic_ring_accounts.csv")

        # Fall back to v1 dataset if v2 not present
        if not os.path.exists(tx_path):
            tx_path = os.path.join(_HERE, "..", "data", "synthetic_transactions.csv")

        if not os.path.exists(ring_path) or not os.path.exists(tx_path):
            return False

        _graph = build_graph_from_rings(ring_path, tx_path)

        # Read VPA index from ring CSV for lookup
        import csv
        with open(ring_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                vpa = row.get("payee_vpa", row.get("vpa", ""))
                if vpa:
                    _vpa_idx[vpa.lower()] = i

        m = EdgeFeatureGAT()
        m.load_state_dict(torch.load(_weights, map_location="cpu",
                                     weights_only=True))
        m.eval()
        _model = m
        return True
    except Exception as e:
        return False


def score_vpa(vpa: str, fallback: float = 0.1) -> dict:
    """
    Returns a GAT-based fraud probability for a given VPA.

    If the model cannot be loaded (missing weights, missing PyG), returns
    fallback with is_available=False so the Risk Scorer can degrade
    gracefully without crashing.
    """
    if not _load():
        return {"gat_score": fallback, "is_available": False,
                "note": "GAT model not loaded — run gat_mule_detector.py to train"}

    vpa_lower = vpa.lower()
    if vpa_lower not in _vpa_idx:
        # VPA not in training graph — return a neutral prior
        return {"gat_score": 0.15, "is_available": True,
                "note": "VPA not in entity graph — neutral prior returned"}

    from gnn.gat_mule_detector import score_with_gat
    probs = score_with_gat(_model, _graph)
    idx   = _vpa_idx[vpa_lower]
    if idx >= len(probs):
        return {"gat_score": 0.15, "is_available": True,
                "note": "Node index out of range — neutral prior returned"}

    prob = float(probs[idx][0])
    return {
        "gat_score":     round(prob, 3),
        "is_available":  True,
        "node_index":    idx,
        "note":          "EdgeFeatureGAT inference on shared-attribute graph",
    }


if __name__ == "__main__":
    r = score_vpa("pooja.455@okaxis")
    print(r)
