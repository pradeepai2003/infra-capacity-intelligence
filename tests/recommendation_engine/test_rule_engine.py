import pandas as pd

from src.recommendation_engine.rule_engine import (
    RecommendationType,
    RiskLevel,
    evaluate_compute,
    evaluate_network_spike,
    evaluate_storage,
)


def _forecast_df(peak_value: float, days: int = 28) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2025-01-01", periods=days, freq="D"),
            "yhat": [peak_value] * days,
        }
    )


def test_evaluate_storage_critical_when_forecast_exceeds_critical_threshold():
    recs = evaluate_storage(
        "storage-01",
        current_util_pct=60,
        forecast_4wk=_forecast_df(95),
        forecast_12wk=_forecast_df(97),
        critical_threshold=90,
        warning_threshold=75,
    )
    assert any(r.risk_level == RiskLevel.CRITICAL for r in recs)
    assert any(r.recommendation_type == RecommendationType.INCREASE_STORAGE for r in recs)


def test_evaluate_storage_no_action_when_below_thresholds():
    recs = evaluate_storage(
        "storage-02",
        current_util_pct=40,
        forecast_4wk=_forecast_df(50),
        forecast_12wk=_forecast_df(55),
        critical_threshold=90,
        warning_threshold=75,
    )
    assert all(r.recommendation_type == RecommendationType.NO_ACTION for r in recs)


def test_evaluate_compute_flags_chronic_waste():
    chronic_low_util = pd.Series([15.0] * 30)  # 30 days at 15% -> way below threshold
    rec = evaluate_compute("cluster-01", chronic_low_util, underutilization_threshold=30, chronic_days_threshold=21)
    assert rec.recommendation_type in (
        RecommendationType.DOWNSIZE_COMPUTE,
        RecommendationType.DECOMMISSION,
    )
    assert rec.risk_level == RiskLevel.WARNING


def test_evaluate_compute_flags_overutilization():
    high_util = pd.Series([92.0] * 10)
    rec = evaluate_compute("cluster-02", high_util, overutilization_threshold=85)
    assert rec.recommendation_type == RecommendationType.ENABLE_AUTOSCALING
    assert rec.risk_level == RiskLevel.CRITICAL


def test_evaluate_compute_no_action_for_healthy_utilization():
    healthy_util = pd.Series([55.0] * 10)
    rec = evaluate_compute("cluster-03", healthy_util)
    assert rec.recommendation_type == RecommendationType.NO_ACTION


def test_evaluate_network_spike_detects_spike():
    spiky = pd.Series([20.0] * 20 + [90.0] * 5)
    rec = evaluate_network_spike("network-01", spiky, spike_threshold_pct=85, min_spike_days=2)
    assert rec.recommendation_type == RecommendationType.ENABLE_AUTOSCALING


def test_evaluate_network_spike_no_action_when_stable():
    stable = pd.Series([30.0] * 20)
    rec = evaluate_network_spike("network-02", stable)
    assert rec.recommendation_type == RecommendationType.NO_ACTION
