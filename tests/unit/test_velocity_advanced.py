"""Unit tests for risk_engine/velocity_advanced.py."""
import pytest

from risk_engine import velocity_advanced as va
from risk_engine.velocity_advanced import (BehaviouralProfiler, VelocityAnalyser,
                                           _std, get_behavioural_profiler,
                                           get_velocity_analyser)


@pytest.fixture(autouse=True)
def _reset_module_cache():
    va._va = None
    va._bp = None
    yield
    va._va = None
    va._bp = None


# ── _std ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("values,expected", [
    ([], 0.0),
    ([5.0], 0.0),
    ([2.0, 2.0, 2.0], 0.0),
    ([1.0, 3.0], 1.0),
])
def test_std(values, expected):
    assert _std(values) == pytest.approx(expected)


# ── VelocityAnalyser ─────────────────────────────────────────────────
def test_from_csv_skips_rows_without_velocity(make_csv):
    path = make_csv([
        {"payee_vpa": "ring@ybl", "unique_senders_7d": 5},
        {"payee_vpa": "ring@ybl", "unique_senders_7d": 0},   # zero -> skipped
        {"payee_vpa": "", "unique_senders_7d": 9},           # no vpa -> skipped
        {"payee_vpa": "other@ybl", "unique_senders_7d": 3},
    ])
    analyser = VelocityAnalyser.from_csv(path)
    assert dict(analyser.history) == {"ring@ybl": [5.0], "other@ybl": [3.0]}


def test_rolling_median_uses_trailing_window():
    analyser = VelocityAnalyser()
    assert analyser._rolling_median([1, 9, 2, 3], window=3) == [1, 9, 2, 3]
    assert analyser._rolling_median([4, 4, 4, 40], window=2) == [4, 4, 4, 40]


def test_l3_unavailable_without_enough_history():
    analyser = VelocityAnalyser()
    analyser.history["fresh@ybl"] = [4.0, 5.0]
    result = analyser.score_l3_seasonal("fresh@ybl", 40.0)
    assert result == {"l3_available": False,
                      "note": "Insufficient history (<3 obs)"}


def test_l3_flags_burst_above_deseasonalised_baseline():
    analyser = VelocityAnalyser()
    analyser.history["mule@ybl"] = [4.0, 5.0, 6.0, 5.0]
    result = analyser.score_l3_seasonal("mule@ybl", 60.0)
    assert result["l3_available"] is True
    assert result["baseline_us7"] == 5.0
    assert result["residual"] == 55.0
    assert result["z_score"] > 2.0
    assert result["seasonal_anomaly"] is True
    assert result["note"] == "Anomaly vs deseasonalised baseline"


def test_l3_accepts_velocity_within_seasonal_norms():
    analyser = VelocityAnalyser()
    analyser.history["salary@ybl"] = [10.0, 30.0, 20.0, 25.0]
    result = analyser.score_l3_seasonal("salary@ybl", 26.0)
    assert result["seasonal_anomaly"] is False
    assert result["note"] == "Within seasonal norms"


def test_l4_unavailable_without_enough_history():
    analyser = VelocityAnalyser()
    assert analyser.score_l4_forecast("unknown@ybl") == {
        "l4_available": False, "note": "Insufficient history"}


def test_l4_forecast_is_ar1_of_last_and_mean():
    analyser = VelocityAnalyser()
    analyser.history["vpa@ybl"] = [5.0, 20.0, 60.0]
    result = analyser.score_l4_forecast("vpa@ybl")
    assert result["mean_us7"] == pytest.approx(28.3)
    assert result["forecast_us7"] == pytest.approx(50.5)  # .7*60 + .3*28.33
    assert result["trend_slope"] == pytest.approx(27.5)
    assert result["early_warning"] is True
    assert "early warning" in result["note"]


def test_l4_flat_history_is_not_an_early_warning():
    analyser = VelocityAnalyser()
    analyser.history["vpa@ybl"] = [20.0, 21.0, 20.0, 21.0]
    result = analyser.score_l4_forecast("vpa@ybl")
    assert result["early_warning"] is False
    assert result["note"] == "Trend within normal range"


# ── BehaviouralProfiler ──────────────────────────────────────────────
def test_profiler_ignores_vpas_with_a_single_session(make_csv):
    path = make_csv([
        {"payee_vpa": "solo@ybl", "amount": 100},
        {"payee_vpa": "repeat@ybl", "amount": 100, "input_method": "paste"},
        {"payee_vpa": "repeat@ybl", "amount": 300},
    ])
    profiler = BehaviouralProfiler.from_csv(path)
    assert set(profiler.profiles) == {"repeat@ybl"}
    prof = profiler.profiles["repeat@ybl"]
    assert prof["paste_rate"] == 0.5
    assert prof["avg_amount"] == 200.0
    assert prof["n_sessions"] == 2


def test_score_unavailable_below_three_sessions():
    profiler = BehaviouralProfiler()
    profiler.profiles["vpa@ybl"] = {"paste_rate": 0.0, "avg_amount": 100.0,
                                    "n_sessions": 2, "paste_std": 0.0,
                                    "amount_std": 0.0}
    assert profiler.score("vpa@ybl", True, 100.0)["l4_available"] is False
    assert profiler.score("missing@ybl", True, 100.0)["l4_available"] is False


def test_score_flags_paste_deviation_from_user_baseline():
    profiler = BehaviouralProfiler()
    profiler.profiles["vpa@ybl"] = {"paste_rate": 0.0, "avg_amount": 1000.0,
                                    "n_sessions": 10, "paste_std": 0.0,
                                    "amount_std": 100.0}
    result = profiler.score("vpa@ybl", current_paste=True, current_amount=1050)
    assert result["paste_z_score"] == 10.0      # std floored at 0.1
    assert result["session_anomaly"] is True
    assert result["note"] == "Session deviates from user baseline"
    assert result["baseline_paste_rate"] == 0.0
    assert result["baseline_avg_amount"] == 1000


def test_score_flags_amount_deviation_and_accepts_normal_sessions():
    profiler = BehaviouralProfiler()
    profiler.profiles["vpa@ybl"] = {"paste_rate": 0.5, "avg_amount": 1000.0,
                                    "n_sessions": 10, "paste_std": 0.5,
                                    "amount_std": 100.0}
    spike = profiler.score("vpa@ybl", current_paste=True, current_amount=2000)
    assert spike["amount_z_score"] == 10.0
    assert spike["session_anomaly"] is True

    normal = profiler.score("vpa@ybl", current_paste=True, current_amount=1050)
    assert normal["session_anomaly"] is False
    assert normal["note"] == "Consistent with user history"


# ── Module-level cached accessors ────────────────────────────────────
def test_cached_accessors_build_once(make_csv):
    path = make_csv([
        {"payee_vpa": "vpa@ybl", "unique_senders_7d": 4},
        {"payee_vpa": "vpa@ybl", "unique_senders_7d": 6},
    ])
    analyser = get_velocity_analyser(path)
    profiler = get_behavioural_profiler(path)
    assert get_velocity_analyser(path) is analyser
    assert get_behavioural_profiler(path) is profiler
    assert analyser.history["vpa@ybl"] == [4.0, 6.0]
    assert profiler.profiles["vpa@ybl"]["n_sessions"] == 2


def test_default_accessors_fall_back_to_bundled_dataset():
    assert len(get_velocity_analyser().history) > 0
    assert len(get_behavioural_profiler().profiles) > 0
