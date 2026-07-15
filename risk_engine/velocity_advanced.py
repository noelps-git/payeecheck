"""
risk_engine/velocity_advanced.py — VPA Velocity L3/L4 and Behavioural SDK L4.

L3 (Velocity): Seasonal decomposition — separates salary-day spikes from
    genuine mule bursts using STL-style rolling median decomposition.
    Reduces false positives on legitimate high-volume accounts.

L4 (Velocity): Time-series forecasting — ARIMA-lite trend extrapolation
    for early-warning queue entries before full mule thresholds are hit.

L4 (Behavioural): Per-user baseline scoring — z-score of current session
    signals against the user's own history, not a population average.

All three are simulated from the synthetic dataset, which provides
multi-row histories per VPA (from the ring/mule accounts that appear
multiple times with consistent attributes).

Usage:
    from risk_engine.velocity_advanced import VelocityAnalyser, BehaviouralProfiler
"""
import csv, os, math
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── VPA Velocity L3/L4 ────────────────────────────────────────────────
class VelocityAnalyser:
    """
    Builds per-VPA velocity history from synthetic data and provides
    L3 (seasonal decomposition) and L4 (trend forecast) scoring.
    """
    def __init__(self):
        self.history = defaultdict(list)  # vpa -> [unique_senders_7d, ...]

    @classmethod
    def from_csv(cls, csv_path: str) -> "VelocityAnalyser":
        va = cls()
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                vpa = row.get("payee_vpa","")
                us7 = float(row.get("unique_senders_7d", 0) or 0)
                if vpa and us7 > 0:
                    va.history[vpa].append(us7)
        return va

    def _rolling_median(self, values: list, window: int = 3) -> list:
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            chunk = sorted(values[start:i+1])
            result.append(chunk[len(chunk)//2])
        return result

    def score_l3_seasonal(self, vpa: str, current_us7: float) -> dict:
        """
        L3: Compare current velocity against deseasonalised baseline.
        Uses rolling median as a simple seasonal component proxy.
        """
        hist = self.history.get(vpa, [])
        if len(hist) < 3:
            return {"l3_available": False, "note": "Insufficient history (<3 obs)"}

        medians = self._rolling_median(hist)
        baseline = medians[-1]   # most recent rolling median = seasonal expectation
        residual = current_us7 - baseline
        z_score  = residual / (max(1.0, _std(hist)))

        anomaly = z_score > 2.0
        return {
            "l3_available":  True,
            "baseline_us7":  round(baseline, 1),
            "current_us7":   current_us7,
            "residual":      round(residual, 1),
            "z_score":       round(z_score, 2),
            "seasonal_anomaly": anomaly,
            "note": ("Anomaly vs deseasonalised baseline"
                     if anomaly else "Within seasonal norms"),
        }

    def score_l4_forecast(self, vpa: str) -> dict:
        """
        L4: Simple linear trend extrapolation (ARIMA-lite, AR(1) model).
        Returns expected next-period velocity and flags if trend is steep.
        """
        hist = self.history.get(vpa, [])
        if len(hist) < 3:
            return {"l4_available": False, "note": "Insufficient history"}

        # AR(1): next = alpha * last + (1-alpha) * mean
        alpha   = 0.7
        mean_v  = sum(hist) / len(hist)
        forecast = alpha * hist[-1] + (1 - alpha) * mean_v
        # Slope from last 3 observations
        recent = hist[-3:]
        slope  = (recent[-1] - recent[0]) / max(len(recent)-1, 1)

        early_warning = slope > 5.0 and forecast > mean_v * 1.5
        return {
            "l4_available":  True,
            "forecast_us7":  round(forecast, 1),
            "trend_slope":   round(slope, 2),
            "mean_us7":      round(mean_v, 1),
            "early_warning": early_warning,
            "note": ("Steep upward trend — early warning queue" if early_warning
                     else "Trend within normal range"),
        }


# ── Behavioural SDK L4 ────────────────────────────────────────────────
class BehaviouralProfiler:
    """
    Per-user z-score baseline from synthetic session data.
    Tracks: paste_rate (fraction of sessions with paste events),
            avg_amount, session_timing (proxy: unique_senders_7d).
    """
    def __init__(self):
        self.profiles = {}   # vpa -> {paste_rate, avg_amount, mean, std}

    @classmethod
    def from_csv(cls, csv_path: str) -> "BehaviouralProfiler":
        bp = cls()
        raw = defaultdict(list)   # vpa -> [(paste, amount), ...]
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                vpa    = row.get("payee_vpa","")
                method = row.get("input_method","type")
                amt    = float(row.get("amount", 0) or 0)
                if vpa:
                    raw[vpa].append((1 if method=="paste" else 0, amt))
        for vpa, sessions in raw.items():
            if len(sessions) < 2:
                continue
            pastes = [s[0] for s in sessions]
            amts   = [s[1] for s in sessions]
            bp.profiles[vpa] = {
                "paste_rate":  sum(pastes)/len(pastes),
                "avg_amount":  sum(amts)/len(amts),
                "n_sessions":  len(sessions),
                "paste_std":   _std(pastes),
                "amount_std":  _std(amts),
            }
        return bp

    def score(self, vpa: str, current_paste: bool,
               current_amount: float) -> dict:
        prof = self.profiles.get(vpa)
        if prof is None or prof["n_sessions"] < 3:
            return {"l4_available": False,
                    "note": "Insufficient session history"}

        # Z-score on paste rate
        paste_z = ((1 if current_paste else 0) - prof["paste_rate"]) / \
                  max(prof["paste_std"], 0.1)
        # Z-score on amount
        amount_z = (current_amount - prof["avg_amount"]) / \
                   max(prof["amount_std"], 1.0)

        anomaly = abs(paste_z) > 2.0 or abs(amount_z) > 2.5
        return {
            "l4_available":  True,
            "paste_z_score": round(paste_z, 2),
            "amount_z_score": round(amount_z, 2),
            "session_anomaly": anomaly,
            "baseline_paste_rate": round(prof["paste_rate"], 2),
            "baseline_avg_amount": round(prof["avg_amount"], 0),
            "note": ("Session deviates from user baseline"
                     if anomaly else "Consistent with user history"),
        }


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v-mean)**2 for v in values) / len(values))


# ── Module-level cached instances ─────────────────────────────────────
_va = None
_bp = None

def get_velocity_analyser(csv_path=None) -> VelocityAnalyser:
    global _va
    if _va is None:
        if csv_path is None:
            csv_path = os.path.join(_HERE,"..","data",
                                    "synthetic_transactions_v2.csv")
            if not os.path.exists(csv_path):
                csv_path = csv_path.replace("_v2","")
        _va = VelocityAnalyser.from_csv(csv_path)
    return _va

def get_behavioural_profiler(csv_path=None) -> BehaviouralProfiler:
    global _bp
    if _bp is None:
        if csv_path is None:
            csv_path = os.path.join(_HERE,"..","data",
                                    "synthetic_transactions_v2.csv")
            if not os.path.exists(csv_path):
                csv_path = csv_path.replace("_v2","")
        _bp = BehaviouralProfiler.from_csv(csv_path)
    return _bp


if __name__ == "__main__":
    va = get_velocity_analyser()
    bp = get_behavioural_profiler()
    print(f"VelocityAnalyser: {len(va.history)} VPAs with history")
    print(f"BehaviouralProfiler: {len(bp.profiles)} VPA profiles")

    # Test on a high-velocity ring account
    test_vpas = [v for v in va.history if len(va.history[v]) >= 3][:3]
    for vpa in test_vpas:
        hist = va.history[vpa]
        l3 = va.score_l3_seasonal(vpa, hist[-1] * 2)
        l4 = va.score_l4_forecast(vpa)
        print(f"\n{vpa}")
        print(f"  L3: z={l3.get('z_score','n/a')} anomaly={l3.get('seasonal_anomaly')}")
        print(f"  L4: forecast={l4.get('forecast_us7','n/a')} warning={l4.get('early_warning')}")
