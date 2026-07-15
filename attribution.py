"""
attribution.py — Three-Layer Attribution Record

Every PayeeCheck verdict must be traceable to specific numerical evidence,
not just a risk score. Inspired by SCAFDS's forensic architecture, adapted
for Indian UPI / RBI Model Risk Management context.
See: PayeeCheck Engineering Playbook, "Three-Layer Attribution Record".
"""


def build_attribution(transaction: dict, payee: dict, temporal: dict) -> dict:
    """
    transaction: {'input_method': 'paste'|'type', 'paste_trust_score': float,
                   'amount': int}
    payee:       {'vpa_age_days': int, 'mule_flagged': bool,
                   'mule_flag_count': int, 'name_score': float}
    temporal:    {'first_time_payee': bool, 'amount_vs_avg_ratio': float}
    """
    L1, L2, L3 = [], [], []

    # ── Layer 1: Transaction signals ─────────────────────────────────
    if transaction.get("input_method") == "paste":
        L1.append({
            "signal": "clipboard_paste",
            "value": transaction.get("paste_trust_score", 0.0),
            "threshold": 0.30,
            "assertion": "VPA pasted from unverified source",
        })
    if transaction.get("amount", 0) > 100000:
        L1.append({
            "signal": "high_value",
            "value": transaction["amount"],
            "threshold": 100000,
            "assertion": f"High-value transfer: Rs {transaction['amount']:,}",
        })

    # ── Layer 2: Payee network signals ───────────────────────────────
    if payee.get("vpa_age_days", 999) < 30:
        L2.append({
            "signal": "vpa_age",
            "value": payee["vpa_age_days"],
            "threshold": 30,
            "assertion": f"VPA {payee['vpa_age_days']} days old",
        })
    if payee.get("mule_flagged"):
        L2.append({
            "signal": "mule_consortium",
            "value": payee.get("mule_flag_count", 1),
            "threshold": 1,
            "assertion": f"{payee.get('mule_flag_count', 1)} banks flagged this account",
        })
    if payee.get("name_score", 1.0) < 0.75:
        L2.append({
            "signal": "name_mismatch",
            "value": payee["name_score"],
            "threshold": 0.75,
            "assertion": f"Name match score: {payee['name_score']:.2f}",
        })

    # ── Layer 3: Temporal / behavioural signals ──────────────────────
    if temporal.get("first_time_payee"):
        L3.append({
            "signal": "first_time_payee",
            "value": 1,
            "threshold": 1,
            "assertion": "First transaction to this payee",
        })
    if temporal.get("amount_vs_avg_ratio", 1.0) > 3.0:
        ratio = temporal["amount_vs_avg_ratio"]
        L3.append({
            "signal": "unusual_amount",
            "value": ratio,
            "threshold": 3.0,
            "assertion": f"Amount {ratio:.1f}x payer average",
        })

    # ── Risk score: weighted sum of triggered signals ────────────────
    weights = {"L1": 0.35, "L2": 0.45, "L3": 0.20}
    max_signals = {"L1": 2, "L2": 3, "L3": 2}
    signal_counts = {"L1": len(L1), "L2": len(L2), "L3": len(L3)}

    risk_score = sum(
        weights[layer] * (signal_counts[layer] / max_signals[layer])
        for layer in ["L1", "L2", "L3"]
    )
    risk_score = round(min(1.0, risk_score) * 100)

    verdict = "block" if risk_score >= 75 else "warn" if risk_score >= 45 else "pass"

    return {
        "risk_score": risk_score,
        "verdict": verdict,
        "attribution": {"L1": L1, "L2": L2, "L3": L3},
        "grounded_assertions": L1 + L2 + L3,
        "total_signals_fired": len(L1) + len(L2) + len(L3),
    }


if __name__ == "__main__":
    result = build_attribution(
        transaction={"input_method": "paste", "paste_trust_score": 0.08, "amount": 24000},
        payee={"vpa_age_days": 11, "mule_flagged": True, "mule_flag_count": 3,
               "name_score": 0.40},
        temporal={"first_time_payee": True, "amount_vs_avg_ratio": 1.2},
    )
    import json
    print(json.dumps(result, indent=2))
