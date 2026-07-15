"""
risk_engine/mule_consortium.py — Mule Database L3/L4: Consortium simulation
and graph-based ring propagation on synthetic data.

L3: Simulates a consortium shared database by treating all `bank_flags`
    across the synthetic dataset as coming from different banks. This
    gives us the multi-bank aggregation logic without needing real
    cross-bank legal agreements.

L4: Graph propagation — a confirmed flag at one account propagates to
    connected ring accounts with exponential decay.

Usage:
    from risk_engine.mule_consortium import ConsortiumDB
    db = ConsortiumDB.from_csv("data/synthetic_transactions_v2.csv")
    result = db.lookup("payee_vpa_here")
"""
import csv, os, json
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))


class ConsortiumDB:
    def __init__(self):
        self.flags   = {}  # vpa -> [bank_id, ...]
        self.rings   = {}  # vpa -> ring_id
        self.ring_vpas = defaultdict(list)  # ring_id -> [vpas]

    @classmethod
    def from_csv(cls, csv_path: str) -> "ConsortiumDB":
        db = cls()
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                vpa  = row.get("payee_vpa","")
                bf   = int(row.get("bank_flags", 0) or 0)
                ring = row.get("fraud_ring_id","")
                mf   = row.get("mule_flagged","").lower() == "true"
                if not vpa:
                    continue
                if mf and bf > 0:
                    # Simulate each flag as from a different synthetic bank
                    db.flags[vpa] = [f"SYNTH_BANK_{i+1}" for i in range(bf)]
                if ring:
                    db.rings[vpa] = ring
                    db.ring_vpas[ring].append(vpa)
        return db

    def _propagate(self, vpa: str, decay: float = 0.5) -> float:
        """
        L4: Propagate consortium risk score from ring coordinator to
        satellite accounts with exponential decay.
        """
        ring = self.rings.get(vpa)
        if not ring:
            return 0.0
        ring_members = self.ring_vpas[ring]
        max_score = 0.0
        for member in ring_members:
            if member == vpa:
                continue
            flags = self.flags.get(member, [])
            if flags:
                # Direct ring connection to a flagged member
                propagated = len(flags) * decay
                max_score = max(max_score, min(propagated, 1.0))
        return round(max_score, 3)

    def lookup(self, vpa: str) -> dict:
        flags         = self.flags.get(vpa, [])
        direct_count  = len(flags)
        prop_score    = self._propagate(vpa)
        ring          = self.rings.get(vpa)

        # Hard override threshold: 2+ banks = consortium confirmed
        consortium_confirmed = direct_count >= 2

        return {
            "vpa":                   vpa,
            "direct_flag_count":     direct_count,
            "flagging_banks":        flags,
            "consortium_confirmed":  consortium_confirmed,
            "ring_id":               ring,
            "propagated_risk_score": prop_score,
            "effective_score":       max(direct_count / 4.0, prop_score),
            "source":                "consortium_db_synthetic",
        }


_db_cache = None

def get_db(csv_path: str = None) -> ConsortiumDB:
    global _db_cache
    if _db_cache is None:
        if csv_path is None:
            csv_path = os.path.join(_HERE, "..", "data",
                                    "synthetic_transactions_v2.csv")
            if not os.path.exists(csv_path):
                csv_path = csv_path.replace("_v2", "")
        _db_cache = ConsortiumDB.from_csv(csv_path)
    return _db_cache


if __name__ == "__main__":
    db = get_db()
    print(f"Loaded: {len(db.flags)} flagged VPAs, {len(db.rings)} ring members")
    # Test a mule VPA and a clean one
    test_vpas = list(db.flags.keys())[:3] + list(
        set(db.rings.keys()) - set(db.flags.keys()))[:2]
    for vpa in test_vpas:
        r = db.lookup(vpa)
        print(f"  {vpa:30s} flags={r['direct_flag_count']} "
              f"consortium={r['consortium_confirmed']} "
              f"prop={r['propagated_risk_score']:.2f}")
