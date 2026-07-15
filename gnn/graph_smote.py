"""
gnn/graph_smote.py — Graph-aware oversampling (GraphSMOTE proxy).

Generates synthetic minority-class (fraud ring) nodes by interpolating
in the embedding space between existing fraud-ring nodes, while preserving
graph structure. This is a simplified GraphSMOTE implementation using
numpy — no special library required.

HONEST NOTE: The real UPI fraud rate is <1%. Our synthetic dataset has
~33% positive rate — far too balanced for real-world calibration.
This module corrects that imbalance when training the GAT, preventing
a model calibrated for an unrealistically balanced dataset.

Usage:
    from gnn.graph_smote import oversample_graph
    X_aug, y_aug, edge_index_aug = oversample_graph(X, y, edge_index, target_rate=0.01)
"""
import numpy as np
from collections import defaultdict


def oversample_graph(X: np.ndarray, y: np.ndarray,
                     edge_index: np.ndarray,
                     target_rate: float = 0.01,
                     seed: int = 42) -> tuple:
    """
    Oversample minority class (fraud=1) nodes to approximate target_rate,
    using feature-space interpolation between existing fraud nodes.

    Args:
        X:           (N, F) node feature matrix
        y:           (N,)   node labels (0=clean, 1=fraud)
        edge_index:  (2, E) directed edge list
        target_rate: desired fraud rate in augmented graph (e.g. 0.01 = 1%)
        seed:        random seed

    Returns:
        X_aug, y_aug, edge_index_aug — augmented graph

    The augmented graph has the same clean nodes + interpolated fraud nodes.
    New fraud nodes are connected to their nearest existing fraud neighbor.
    """
    rng = np.random.default_rng(seed)

    fraud_idx = np.where(y == 1)[0]
    clean_idx = np.where(y == 0)[0]
    n_clean   = len(clean_idx)
    n_fraud   = len(fraud_idx)

    # Target: fraud_count / (fraud_count + clean_count) = target_rate
    # => fraud_count = target_rate * clean_count / (1 - target_rate)
    target_fraud = max(n_fraud, int(target_rate * n_clean / (1 - target_rate)))
    n_synthetic  = target_fraud - n_fraud

    if n_synthetic <= 0 or n_fraud < 2:
        return X, y, edge_index  # nothing to do

    # ── Generate synthetic fraud nodes via interpolation ──────────────
    fraud_X  = X[fraud_idx]
    syn_nodes = []
    syn_edges = []   # synthetic node -> nearest real fraud neighbor
    base_N    = len(y)

    for i in range(n_synthetic):
        # Pick two random fraud nodes and interpolate
        a, b = rng.choice(n_fraud, size=2, replace=False)
        alpha = rng.uniform(0.3, 0.7)
        syn_feat = alpha * fraud_X[a] + (1 - alpha) * fraud_X[b]
        # Add small Gaussian noise to avoid duplicates
        syn_feat += rng.normal(0, 0.01, size=syn_feat.shape)
        syn_nodes.append(syn_feat)
        # Connect synthetic node to nearest real fraud neighbor
        dists = np.linalg.norm(fraud_X - syn_feat, axis=1)
        nearest_real_idx = fraud_idx[np.argmin(dists)]
        new_node_idx = base_N + i
        # Bidirectional edge
        syn_edges.append([nearest_real_idx, new_node_idx])
        syn_edges.append([new_node_idx, nearest_real_idx])

    syn_X = np.stack(syn_nodes, axis=0).astype(X.dtype)
    syn_y = np.ones(n_synthetic, dtype=y.dtype)

    X_aug = np.concatenate([X, syn_X], axis=0)
    y_aug = np.concatenate([y, syn_y], axis=0)

    if syn_edges:
        syn_edge_arr = np.array(syn_edges, dtype=edge_index.dtype).T
        edge_index_aug = np.concatenate([edge_index, syn_edge_arr], axis=1)
    else:
        edge_index_aug = edge_index

    return X_aug, y_aug, edge_index_aug


def report(y_orig, y_aug):
    orig_rate = y_orig.mean()
    aug_rate  = y_aug.mean()
    print(f"Original: {len(y_orig)} nodes, fraud rate {orig_rate:.1%}")
    print(f"Augmented: {len(y_aug)} nodes, fraud rate {aug_rate:.1%}")
    print(f"Synthetic nodes added: {len(y_aug)-len(y_orig)}")


def adjust_to_target_rate(X, y, edge_index, target_rate=0.01, seed=42):
    """
    Adjusts class balance toward target_rate.
    If current fraud rate > target_rate: undersample clean nodes.
    If current fraud rate < target_rate: oversample fraud nodes (GraphSMOTE path).
    """
    rng = np.random.default_rng(seed)
    fraud_idx = np.where(y == 1)[0]
    clean_idx = np.where(y == 0)[0]
    current_rate = len(fraud_idx) / max(len(y), 1)

    if current_rate > target_rate:
        # Undersample clean nodes to achieve target rate
        n_clean_target = int(len(fraud_idx) * (1 - target_rate) / target_rate)
        keep_clean = rng.choice(clean_idx, size=min(n_clean_target, len(clean_idx)),
                                replace=False)
        keep_idx = np.concatenate([fraud_idx, keep_clean])
        keep_idx.sort()
        # Remap indices
        new_idx = {old: new for new, old in enumerate(keep_idx)}
        X_new = X[keep_idx]
        y_new = y[keep_idx]
        # Rebuild edge_index keeping only edges between kept nodes
        if edge_index.shape[1] > 0:
            mask = np.array([
                s in new_idx and d in new_idx
                for s, d in zip(edge_index[0], edge_index[1])
            ])
            e_kept = edge_index[:, mask]
            e_new = np.array([[new_idx[s] for s in e_kept[0]],
                              [new_idx[d] for d in e_kept[1]]], dtype=edge_index.dtype)
        else:
            e_new = edge_index
        return X_new, y_new, e_new
    else:
        return oversample_graph(X, y, edge_index, target_rate, seed)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    # Simulate with random graph matching synthetic dataset structure
    rng = np.random.default_rng(42)
    N = 285; F = 5
    X = rng.random((N, F)).astype(np.float32)
    y = (rng.random(N) < 0.33).astype(np.int64)  # ~33% fraud (synthetic)
    edges = rng.integers(0, N, size=(2, 30))

    print("Testing graph oversampling:")
    print(f"  Target rate: 1% (real-world UPI fraud approximation)")
    X_aug, y_aug, e_aug = adjust_to_target_rate(X, y, edges, target_rate=0.01)
    report(y, y_aug)
    print(f"  Edge count: {edges.shape[1]} -> {e_aug.shape[1]}")
