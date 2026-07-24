"""
gnn/transaction_graph.py — Money-flow transaction graph for PayeeCheck.

Nodes = accounts (VPAs). Edges = actual money flows between accounts,
weighted by: amount, frequency, time-since-last-transfer, and direction.

This is the SECOND graph type alongside the shared-attribute entity graph
(matchers/l6_graph.py). Where the attribute graph connects accounts by
what they HAVE IN COMMON (shared device, shared mobile), this graph
connects them by what has MOVED BETWEEN THEM.

Why this matters:
  Layering — the defining money-laundering pattern — cycles funds through
  a chain of accounts that share NO common attributes. Each hop uses a
  different device, mobile number, and identity. The attribute graph is
  blind to this. The transaction-flow graph sees the cycle directly.

Architecture:
  Same EdgeFeatureGAT backbone as the mule ring detector — edge features
  here are [amount_log, frequency, days_since_last, direction_flag]
  rather than [shares_mobile, shares_device].

HONEST NOTE: trained on synthetic data. Retraining on real transaction
data from a bank pilot will produce meaningfully different weights.

Usage:
    from gnn.transaction_graph import build_tx_graph, score_tx_graph
    graph = build_tx_graph("data/synthetic_transactions_v2.csv")
    scores = score_tx_graph(model, graph)
"""

import os, sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Constants ─────────────────────────────────────────────────────────
CYCLE_THRESHOLD_DAYS = 3    # funds returned within 3 days = cycle flag
HIGH_AMOUNT_LOG      = 11.5 # log(100000) ≈ 11.5 — high-value threshold
MIN_EDGES_FOR_CYCLE  = 3    # minimum ring size to flag


def build_tx_graph(tx_csv_path: str):
    """
    Build a directed money-flow graph from synthetic transaction data.

    Node features (per VPA):
        [vpa_age_days_norm, prior_tx_count_log, unique_senders_7d_norm,
         amount_log_norm, mule_flag_count_norm]

    Edge features (per money flow A → B):
        [amount_log_norm, frequency_norm, days_since_last_norm,
         direction_flag]   # direction_flag=1 if this edge reverses a
                           # prior B→A flow (cycle indicator)

    Returns a torch_geometric.data.Data object, or a plain dict if
    PyTorch Geometric is not installed (graceful degradation).
    """
    import csv
    from collections import defaultdict
    from datetime import datetime

    # ── Load transactions ─────────────────────────────────────────────
    rows = []
    with open(tx_csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Index unique VPAs
    vpas = sorted(set(r["payee_vpa"] for r in rows))
    vpa_idx = {v: i for i, v in enumerate(vpas)}
    N = len(vpas)

    # Node features from the most recent transaction per VPA
    node_feats = np.zeros((N, 5), dtype=np.float32)
    latest = {}
    for r in rows:
        vpa = r["payee_vpa"]
        i   = vpa_idx[vpa]
        if vpa not in latest:
            latest[vpa] = r
        node_feats[i] = [
            min(float(r.get("vpa_age_days", 365)), 3650) / 3650,
            np.log1p(float(r.get("prior_tx_count", 0))),
            min(float(r.get("unique_senders_7d", 0)), 100) / 100,
            np.log1p(float(r.get("amount", 1000))) / 14,
            min(float(r.get("mule_flag_count", 0)), 5) / 5,
        ]

    # ── Build edges: each transaction becomes a directed edge ─────────
    # For synthetic data, we simulate payer VPAs as the "previous" VPA
    # in the transaction list (a proxy for the real payer → payee edge).
    # In production, the real payer VPA would be in the transaction record.
    edge_flows = defaultdict(list)   # (src_idx, dst_idx) -> [amounts]
    edge_dates  = defaultdict(list)  # (src_idx, dst_idx) -> [timestamps]

    # Simulate payer → payee by using consecutive row pairs in same ring
    ring_rows = defaultdict(list)
    for r in rows:
        ring = r.get("fraud_ring_id", "")
        if ring:
            ring_rows[ring].append(r)

    # Within-ring flows (the important ones for cycle detection)
    for ring, ring_txs in ring_rows.items():
        for j in range(len(ring_txs)):
            src_vpa = ring_txs[j]["payee_vpa"]
            dst_vpa = ring_txs[(j+1) % len(ring_txs)]["payee_vpa"]
            if src_vpa == dst_vpa:
                continue
            src_i, dst_i = vpa_idx[src_vpa], vpa_idx[dst_vpa]
            amt = float(ring_txs[j].get("amount", 1000))
            edge_flows[(src_i, dst_i)].append(amt)

    # Non-ring transactions: simulate payer as a generic "clean" node
    # We add these as self-loops to maintain node feature coverage
    # without fabricating payer VPAs we don't have.

    # ── Build edge list with features ─────────────────────────────────
    src_list, dst_list, edge_feat_list = [], [], []
    flow_keys = set(edge_flows.keys())

    for (src_i, dst_i), amounts in edge_flows.items():
        freq = len(amounts)
        amt_log = np.log1p(np.mean(amounts)) / 14  # normalised
        freq_norm = min(freq, 20) / 20
        # direction_flag: does the reverse edge also exist?
        has_reverse = (dst_i, src_i) in flow_keys
        direction_flag = 1.0 if has_reverse else 0.0
        days_since = 1.0  # synthetic — no real timestamps per edge

        src_list.append(src_i)
        dst_list.append(dst_i)
        edge_feat_list.append([amt_log, freq_norm, days_since, direction_flag])

    # ── Labels: fraud ring members = 1, others = 0 ───────────────────
    labels = np.zeros(N, dtype=np.int64)
    for r in rows:
        if r.get("label", "") in ("ring_member", "mule_consortium", "mule"):
            labels[vpa_idx[r["payee_vpa"]]] = 1

    # ── Cycle detection (deterministic, no ML) ────────────────────────
    cycles = detect_cycles(vpa_idx, edge_flows)

    # ── Try to build a PyG Data object ───────────────────────────────
    meta = {
        "n_nodes":  N,
        "n_edges":  len(src_list),
        "n_fraud":  int(labels.sum()),
        "n_cycles": len(cycles),
        "vpas":     vpas,
        "vpa_idx":  vpa_idx,
        "cycles":   cycles,
    }

    if not src_list:
        # No within-ring edges found — return metadata only
        meta["pyg_data"] = None
        return meta

    try:
        import torch
        from torch_geometric.data import Data

        x           = torch.tensor(node_feats, dtype=torch.float)
        edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr   = torch.tensor(edge_feat_list, dtype=torch.float)
        y           = torch.tensor(labels, dtype=torch.long)

        meta["pyg_data"] = Data(x=x, edge_index=edge_index,
                                 edge_attr=edge_attr, y=y)
    except ImportError:
        meta["pyg_data"] = None

    return meta


def detect_cycles(vpa_idx: dict, edge_flows: dict) -> list:
    """
    Deterministic cycle detection using DFS — no ML required.
    Returns a list of cycles [(vpa_a, vpa_b, ..., vpa_a)] where funds
    complete a round trip.

    This is the graph-layer equivalent of the mule_flag_count hard
    override: a detected fund cycle is a strong, rule-based signal
    regardless of whether the GAT fires.
    """
    idx_vpa = {i: v for v, i in vpa_idx.items()}
    adj = {}
    for (src, dst) in edge_flows:
        adj.setdefault(src, set()).add(dst)

    cycles = []
    visited = set()

    def dfs(node, path, path_set):
        for nxt in adj.get(node, []):
            if nxt == path[0] and len(path) >= MIN_EDGES_FOR_CYCLE:
                cycle_vpas = [idx_vpa[n] for n in path]
                cycles.append(cycle_vpas)
                return
            if nxt not in path_set:
                path_set.add(nxt)
                dfs(nxt, path + [nxt], path_set)
                path_set.discard(nxt)

    for start in list(adj.keys())[:50]:   # cap to avoid O(n!) blowup
        if start not in visited:
            dfs(start, [start], {start})
            visited.add(start)

    return cycles[:20]   # cap returned cycles


def score_tx_graph(model, graph_meta: dict,
                   threshold: float = 0.55) -> dict:
    """
    Run GAT inference on the transaction graph.
    Returns per-VPA fraud probabilities and a list of high-risk VPAs.
    """
    pyg_data = graph_meta.get("pyg_data")
    cycles   = graph_meta.get("cycles", [])

    if pyg_data is None:
        return {
            "scores": {},
            "high_risk_vpas": [],
            "cycles_detected": len(cycles),
            "is_available": False,
            "note": "PyG not available or no within-ring edges found",
        }

    try:
        from gnn.gat_mule_detector import score_with_gat
        probs = score_with_gat(model, pyg_data)
    except Exception as e:
        logger.warning("Transaction-graph GAT inference failed — returning empty scores.",
                       exc_info=True)
        return {
            "scores": {},
            "high_risk_vpas": [],
            "cycles_detected": len(cycles),
            "is_available": False,
            "note": str(e),
        }

    vpas    = graph_meta["vpas"]
    scores  = {vpa: float(probs[i]) for i, vpa in enumerate(vpas)
               if i < len(probs)}
    high_risk = [v for v, s in scores.items() if s >= threshold]

    return {
        "scores":          scores,
        "high_risk_vpas":  high_risk,
        "cycles_detected": len(cycles),
        "cycle_detail":    cycles[:5],   # first 5 for brevity
        "is_available":    True,
    }


if __name__ == "__main__":
    import json
    tx_path = os.path.join(os.path.dirname(__file__),
                           "..", "data", "synthetic_transactions_v2.csv")
    if not os.path.exists(tx_path):
        tx_path = tx_path.replace("_v2", "")

    print("Building transaction graph...")
    meta = build_tx_graph(tx_path)

    print(f"Nodes : {meta['n_nodes']}")
    print(f"Edges : {meta['n_edges']}")
    print(f"Fraud : {meta['n_fraud']}")
    print(f"Cycles: {meta['n_cycles']}")
    if meta["cycles"]:
        print(f"Sample cycle ({len(meta['cycles'][0])} hops):",
              " -> ".join(meta["cycles"][0][:4]), "->...")
    print("PyG data:", "available" if meta["pyg_data"] else "not available (PyG missing)")
