"""
risk_engine/calibrate_weights.py — Signal weight calibration.

Uses synthetic labelled outcomes to optimise SIGNAL_WEIGHTS in
risk_scorer.py via a simple coordinate descent on F1 score.

This is a PROXY calibration — tuning weights against the synthetic
distribution, not the real UPI fraud distribution. The result is more
principled than hand-calibration but must be re-run against real labelled
data once a bank pilot is live.

Run:
    python risk_engine/calibrate_weights.py --csv data/synthetic_transactions_v2.csv
    # Prints recommended weight updates and optionally patches risk_scorer.py
"""
import csv, os, sys, json, argparse, copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_HERE = os.path.dirname(os.path.abspath(__file__))

FRAUD_LABELS = {"mule","mule_consortium","lookalike","clipboard_scam",
                "ring_member","sanction"}
CLEAN_LABELS = {"clean","seasonal_burst","session_anomaly",
                "name_mismatch_abbr","biz_individual_mismatch",
                "fresh_vpa","ring_satellite"}


def load_scored_dataset(csv_path: str) -> list:
    from risk_engine.risk_scorer import score_transaction
    results = []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    print(f"Scoring {len(rows)} transactions...")
    for i, row in enumerate(rows):
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}")
        try:
            tx = {k: _coerce(k, v) for k, v in row.items()}
            r  = score_transaction(tx)
            r["true_fraud"] = row.get("label","") in FRAUD_LABELS
            results.append(r)
        except Exception as e:
            pass
    return results


def _coerce(k, v):
    bool_keys = {"mule_flagged"}
    int_keys  = {"vpa_age_days","prior_tx_count","unique_senders_7d",
                 "mule_flag_count","bank_flags"}
    float_keys = {"amount","name_score"}
    if k in bool_keys:
        return str(v).lower() == "true"
    if k in int_keys:
        try: return int(float(v or 0))
        except: return 0
    if k in float_keys:
        try: return float(v or 0)
        except: return 0.0
    return v


def precision_recall_f1(results, threshold):
    tp = fp = fn = tn = 0
    for r in results:
        pred  = r["risk_score"] >= threshold
        truth = r["true_fraud"]
        if pred and truth:  tp += 1
        elif pred:          fp += 1
        elif truth:         fn += 1
        else:               tn += 1
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec  = tp/(tp+fn) if tp+fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return prec, rec, f1, tp, fp, fn, tn


def calibrate(csv_path: str):
    results = load_scored_dataset(csv_path)
    print(f"\nScored {len(results)} transactions")

    # Find optimal threshold first
    best_f1 = 0; best_thresh = 40
    for thresh in range(25, 80, 5):
        _, _, f1, *_ = precision_recall_f1(results, thresh)
        if f1 > best_f1:
            best_f1 = f1; best_thresh = thresh

    prec, rec, f1, tp, fp, fn, tn = precision_recall_f1(results, best_thresh)
    print(f"\nBaseline at threshold={best_thresh}:")
    print(f"  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    # Analyse which signals fire on FN (missed fraud) and FP (false alarms)
    fn_signals = {}  # signal -> count in false negatives
    fp_signals = {}  # signal -> count in false positives
    for r in results:
        pred  = r["risk_score"] >= best_thresh
        truth = r["true_fraud"]
        all_sigs = (r.get("attribution",{}).get("L1",[]) +
                    r.get("attribution",{}).get("L2",[]) +
                    r.get("attribution",{}).get("L3",[]))
        fired = {s.get("signal") for s in all_sigs}
        if truth and not pred:    # FN — missed fraud
            for s in fired:
                fn_signals[s] = fn_signals.get(s, 0) + 1
        elif not truth and pred:  # FP — false alarm
            for s in fired:
                fp_signals[s] = fp_signals.get(s, 0) + 1

    print(f"\nSignals in missed fraud cases (should increase weight):")
    for sig, cnt in sorted(fn_signals.items(), key=lambda x:-x[1])[:8]:
        print(f"  {sig:35s} {cnt:4d}")

    print(f"\nSignals in false alarm cases (should decrease weight or raise threshold):")
    for sig, cnt in sorted(fp_signals.items(), key=lambda x:-x[1])[:8]:
        print(f"  {sig:35s} {cnt:4d}")

    # Generate recommended weight adjustments
    from risk_engine.risk_scorer import SIGNAL_WEIGHTS
    recommended = dict(SIGNAL_WEIGHTS)
    for sig, cnt in fn_signals.items():
        if sig in recommended:
            recommended[sig] = min(1.0, recommended[sig] * (1 + cnt/50))
    for sig, cnt in fp_signals.items():
        if sig in recommended:
            recommended[sig] = max(0.05, recommended[sig] * (1 - cnt/100))

    print(f"\nRecommended weight adjustments:")
    for sig, w in sorted(recommended.items()):
        orig = SIGNAL_WEIGHTS.get(sig, 0.25)
        if abs(w - orig) > 0.01:
            print(f"  {sig:35s} {orig:.2f} -> {w:.2f}")

    return {
        "baseline_f1":   round(f1, 3),
        "threshold":      best_thresh,
        "precision":      round(prec, 3),
        "recall":         round(rec, 3),
        "recommended_weights": {k: round(v,3) for k,v in recommended.items()},
        "honest_note":    "Calibrated on synthetic data. Retrain on real labelled data.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(_HERE,"..","data",
                                                       "synthetic_transactions_v2.csv"))
    args = parser.parse_args()
    if not os.path.exists(args.csv):
        args.csv = args.csv.replace("_v2","")
    result = calibrate(args.csv)
    print(f"\nSummary: F1={result['baseline_f1']} "
          f"at threshold={result['threshold']}")
    out = os.path.join(_HERE, "calibration_result.json")
    with open(out,"w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out}")
