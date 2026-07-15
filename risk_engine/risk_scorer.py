"""
risk_scorer.py — Phase 2: The PayeeCheck Risk Scorer

This is the missing product core every Phase 1 playbook pointed to.
It consumes outputs from all five Phase 1 signal modules simultaneously,
populates the full three-layer attribution record in one pass, and
produces the single block/warn/pass verdict the UI displays.

Signal modules consumed:
    1. Name Matching (CoP)         -> Layer 2
    2. Look-alike VPA Detection    -> Layer 2
    3. Mule Account Database       -> Layer 2 (highest weight)
    4. Behavioural Signal SDK      -> Layer 1
    5. VPA Age + Velocity Engine   -> Layer 2
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from attribution import build_attribution


# ── Weight configuration — tunable, documented per-signal ────────────
# These weights determine how much each fired signal contributes to the
# final risk score. Calibrated qualitatively against the playbooks'
# "Typical Weight in Final Score" table — NOT learned from real data yet.
# Re-tune once you have labelled outcomes from production.
SIGNAL_WEIGHTS = {
    "mule_consortium":      1.00,
    "sanctions_pep_match":  1.00,
    "gat_ring_score":       0.75,   # GAT fraud probability (7th signal)
    "velocity_pattern":     0.55,
    "lookalike_vpa":        0.70,
    "name_mismatch":        0.45,
    "vpa_age":              0.35,
    "clipboard_paste":      0.20,
    "high_value":           0.15,
    "first_time_payee":     0.10,
    "unusual_amount":       0.20,
    "rapid_cross_app_paste": 0.30,
    "flow_speed_anomaly":   0.15,
    "velocity_anomaly":     0.40,
}

BLOCK_THRESHOLD = 75
WARN_THRESHOLD = 40


def run_name_match(entered_name: str, actual_name: str) -> dict:
    """
    Wraps matchers.l3_tfidf for benchmark/CI environments with no network
    access (TF-IDF needs no model download). In production with internet
    access, swap this single line to matchers.l4_embeddings — the
    function signature is identical, this is a one-line change.
    """
    from matchers.l3_tfidf import match
    return match(entered_name, actual_name)


def run_lookalike_check(vpa: str) -> dict:
    """
    Simplified inline look-alike check (the full L1-L5 detector lives in
    the Look-alike VPA playbook). Reused here at Level 3 (brand registry)
    depth — the highest-leverage, lowest-cost level per that playbook.
    """
    import re
    PROTECTED_BRANDS = {
        "amazon": ["amazon.pay@apl"], "paytm": ["paytm@paytm"],
        "phonepe": ["phonepe@ybl"], "googlepay": ["googlepay@okaxis"],
        "flipkart": ["flipkart@icici"], "swiggy": ["swiggy@ybl"],
        "zomato": ["zomato@paytm"],
    }
    SUSPICIOUS_SUFFIXES = {"support", "help", "verify", "refund", "kyc",
                            "service", "cashback", "secure"}
    handle = vpa.split("@")[0].lower() if "@" in vpa else vpa.lower()
    tokens = set(re.split(r"[._-]", handle))

    brand = next((b for b in PROTECTED_BRANDS if b in handle), None)
    if brand is None:
        return {"flag": "no_brand_detected", "is_lookalike": False}

    is_registered = vpa in PROTECTED_BRANDS[brand]
    suspicious = tokens & SUSPICIOUS_SUFFIXES

    if is_registered:
        return {"flag": "legitimate", "is_lookalike": False, "brand": brand}

    return {
        "flag": "brand_impersonation_high_confidence" if suspicious
                else "brand_impersonation_suspected",
        "is_lookalike": True, "brand": brand,
        "suspicious_suffixes": list(suspicious),
    }


def run_mule_check(vpa_or_account: str, mule_flagged: bool,
                     mule_flag_count: int) -> dict:
    """Simplified inline mule check — reuses transaction-level flags
    already present in the synthetic dataset / would come from
    mule_l1_static.check_account() in production."""
    return {
        "flagged": mule_flagged,
        "flag_count": mule_flag_count,
    }


def run_sanctions_check(payee_name: str) -> dict:
    """Wraps risk_engine.sanctions_screening — the 6th signal module."""
    from risk_engine.sanctions_screening import check_sanctions
    return check_sanctions(payee_name)


def run_velocity_check(vpa_age_days: int, prior_tx_count: int,
                         unique_senders_7d: int, amount: float) -> dict:
    """Inline Level 2 velocity scoring (rolling window logic) — see the
    VPA Age + Velocity playbook for the full implementation this mirrors."""
    score = 0
    reasons = []

    if vpa_age_days < 30 and prior_tx_count < 3:
        score += 30
        reasons.append(f"VPA is {vpa_age_days} days old with only "
                        f"{prior_tx_count} prior transactions")

    if unique_senders_7d >= 8:
        score += 30
        reasons.append(f"{unique_senders_7d} unique senders in 7 days")

    if vpa_age_days < 14 and amount > 50000:
        score += 30
        reasons.append(f"High-value transfer (Rs {amount:,.0f}) to a "
                        f"{vpa_age_days}-day-old VPA")

    return {"velocity_score": score, "reasons": reasons,
            "vpa_age_days": vpa_age_days}



# GAT ring scorer — loads once, degrades gracefully if PyG not installed
try:
    from gnn.gat_scorer import score_vpa as _gat_score_vpa
    _GAT_AVAILABLE = True
except Exception:
    _GAT_AVAILABLE = False
    def _gat_score_vpa(vpa, fallback=0.1):
        return {"gat_score": fallback, "is_available": False, "note": "GAT import failed"}

def score_transaction(tx: dict) -> dict:
    """
    The main Phase 2 entry point. Takes a transaction dict (matching the
    shape produced by data/generate_dataset.py, or a real transaction
    with the same fields) and returns a full verdict with attribution.
    """
    # ── Run all Phase 1 signal modules ───────────────────────────────
    name_result = run_name_match(tx["entered_name"], tx["actual_name"])
    lookalike_result = run_lookalike_check(tx["payee_vpa"])
    mule_result = run_mule_check(tx["payee_vpa"], tx["mule_flagged"],
                                   tx["mule_flag_count"])
    velocity_result = run_velocity_check(
        tx["vpa_age_days"], tx["prior_tx_count"],
        tx["unique_senders_7d"], tx["amount"]
    )
    sanctions_result = run_sanctions_check(tx["actual_name"])
    gat_result = _gat_score_vpa(tx["payee_vpa"])

    # ── Build the three-layer attribution record ─────────────────────
    transaction_signals = {
        "input_method": tx["input_method"],
        "paste_trust_score": 0.08 if tx["input_method"] == "paste" else 0.9,
        "amount": tx["amount"],
    }
    payee_signals = {
        "vpa_age_days": tx["vpa_age_days"],
        "mule_flagged": mule_result["flagged"],
        "mule_flag_count": mule_result["flag_count"],
        "name_score": name_result["score"],
    }
    temporal_signals = {
        "first_time_payee": tx["prior_tx_count"] == 0,
        "amount_vs_avg_ratio": (
            tx["amount"] / 10000 if tx["prior_tx_count"] < 3 else 1.0
        ),
    }

    base_attribution = build_attribution(
        transaction_signals, payee_signals, temporal_signals
    )

    # ── Extend attribution with look-alike and velocity signals ──────
    L2 = base_attribution["attribution"]["L2"]
    if lookalike_result["is_lookalike"]:
        L2.append({
            "signal": "lookalike_vpa",
            "value": 1.0,
            "threshold": 0.5,
            "assertion": (
                f"VPA impersonates protected brand "
                f"'{lookalike_result['brand']}': {lookalike_result['flag']}"
            ),
        })
    # Wire GAT score into L2 if above noise floor and model is available
    GAT_SIGNAL_THRESHOLD = 0.55
    if gat_result["is_available"] and gat_result["gat_score"] >= GAT_SIGNAL_THRESHOLD:
        L2.append({
            "signal": "gat_ring_score",
            "value":  gat_result["gat_score"],
            "threshold": GAT_SIGNAL_THRESHOLD,
            "assertion": (
                f"GAT ring fraud probability {gat_result['gat_score']:.0%} "
                f"— account appears in a connected fraud ring sub-graph"
            ),
        })

    if velocity_result["velocity_score"] >= 30:
        L2.append({
            "signal": "velocity_anomaly",
            "value": velocity_result["velocity_score"],
            "threshold": 30,
            "assertion": "; ".join(velocity_result["reasons"]),
        })
    if sanctions_result["flag"] in ("sanctions_match", "pep_match"):
        L2.append({
            "signal": "sanctions_pep_match",
            "value": max(sanctions_result["sanctions_score"],
                         sanctions_result["pep_score"]),
            "threshold": 0.85,
            "assertion": sanctions_result["reason"],
        })

    # ── Recompute final weighted risk score across ALL fired signals ─
    all_signals = (base_attribution["attribution"]["L1"] +
                   L2 +
                   base_attribution["attribution"]["L3"])

    weighted_sum = 0.0
    max_possible = 0.0
    for sig in all_signals:
        w = SIGNAL_WEIGHTS.get(sig["signal"], 0.25)
        weighted_sum += w
        max_possible += 1.0  # normalising constant per fired signal

    # Mule consortium flag is treated as a near-automatic block,
    # mirroring the playbook's "highest weight, overrides other signals"
    # Sanctions/PEP hits are treated as automatic blocks, same severity
    # tier as confirmed mule consortium flags — both are externally
    # confirmed facts, not probabilistic signals
    sanctions_override = sanctions_result["flag"] == "sanctions_match"
    mule_override = mule_result["flagged"] and mule_result["flag_count"] >= 2

    if max_possible > 0:
        risk_score = round(min(1.0, weighted_sum / max(max_possible, 3)) * 100)
    else:
        risk_score = 0

    if mule_override:
        risk_score = max(risk_score, 90)
    if sanctions_override:
        risk_score = max(risk_score, 95)

    verdict = ("block" if risk_score >= BLOCK_THRESHOLD else
               "warn" if risk_score >= WARN_THRESHOLD else "pass")

    return {
        "tx_id": tx.get("tx_id"),
        "risk_score": risk_score,
        "verdict": verdict,
        "module_outputs": {
            "name_match": name_result,
            "lookalike": lookalike_result,
            "mule": mule_result,
            "velocity": velocity_result,
            "sanctions": sanctions_result,
            "gat": gat_result,
        },
        "attribution": {
            "L1": base_attribution["attribution"]["L1"],
            "L2": L2,
            "L3": base_attribution["attribution"]["L3"],
        },
        "total_signals_fired": len(all_signals),
        "true_label": tx.get("label"),  # only present for benchmark data
    }


if __name__ == "__main__":
    sample_tx = {
        "tx_id": "DEMO1", "entered_name": "Suresh Kumar Pvt Ltd",
        "actual_name": "Suresh Kumar", "payee_vpa": "paytm.support@axl",
        "amount": 24000, "input_method": "paste", "vpa_age_days": 11,
        "prior_tx_count": 1, "unique_senders_7d": 12,
        "mule_flagged": True, "mule_flag_count": 3, "label": "mule",
    }
    import json
    print(json.dumps(score_transaction(sample_tx), indent=2, default=str))
