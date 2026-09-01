import pandas as pd

from src.recommendation_engine.rule_engine import (
    RecommendationType,
    RiskLevel,
    evaluate_equalization,
    evaluate_mac_resource,
)


def _forecast_df(peak_value: float, days: int = 28) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2025-01-01", periods=days, freq="D"),
            "yhat": [peak_value] * days,
        }
    )


def test_evaluate_mac_resource_flags_overload_from_current_value():
    history = pd.Series([90.0] * 10)
    rec = evaluate_mac_resource("mac-01", "cpu", "Project Atlas", history, overloaded_threshold=85)
    assert rec.recommendation_type == RecommendationType.INCREASE_ALLOCATION
    assert rec.risk_level == RiskLevel.CRITICAL
    assert rec.mac_id == "mac-01"
    assert rec.project_name == "Project Atlas"


def test_evaluate_mac_resource_flags_overload_from_forecast():
    history = pd.Series([50.0] * 10)  # current is fine
    rec = evaluate_mac_resource(
        "mac-02",
        "disk",
        "Project Nova",
        history,
        forecast_4wk=_forecast_df(95),
        overloaded_threshold=85,
    )
    assert rec.recommendation_type == RecommendationType.INCREASE_ALLOCATION
    assert rec.forecasted_value == 95


def test_evaluate_mac_resource_flags_chronic_underload_as_reduce_allocation():
    history = pd.Series([15.0] * 30)  # 30 days well below threshold
    rec = evaluate_mac_resource(
        "mac-03",
        "ram",
        "Project Orion",
        history,
        underloaded_threshold=30,
        chronic_days_threshold=21,
    )
    assert rec.recommendation_type == RecommendationType.REDUCE_ALLOCATION
    assert rec.risk_level == RiskLevel.WARNING


def test_evaluate_mac_resource_no_action_for_healthy_utilization():
    history = pd.Series([55.0] * 10)
    rec = evaluate_mac_resource("mac-04", "cpu", "Project Zephyr", history)
    assert rec.recommendation_type == RecommendationType.NO_ACTION
    assert rec.risk_level == RiskLevel.INFO


def test_evaluate_mac_resource_skips_missing_forecast():
    history = pd.Series([50.0] * 10)
    rec = evaluate_mac_resource(
        "mac-05",
        "disk",
        "Project Falcon",
        history,
        forecast_4wk=None,
        forecast_12wk=pd.DataFrame(columns=["ds", "yhat"]),
    )
    assert rec.recommendation_type == RecommendationType.NO_ACTION


def test_evaluate_equalization_pairs_overloaded_with_underloaded():
    fleet = pd.DataFrame(
        {
            "mac_id": ["mac-01", "mac-02", "mac-03", "mac-04"],
            "project_name": ["Atlas", "Nova", "Orion", "Zephyr"],
            "utilization_pct": [90.0, 50.0, 50.0, 10.0],
        }
    )
    recs = evaluate_equalization(fleet, "cpu", deviation_threshold_pct=20)
    assert len(recs) == 1
    assert recs[0].overloaded_mac_id == "mac-01"
    assert recs[0].underloaded_mac_id == "mac-04"
    assert "mac-01" in recs[0].suggested_action
    assert "mac-04" in recs[0].suggested_action
    assert "wasted" not in recs[0].suggested_action.lower()


def test_evaluate_equalization_no_recommendations_when_balanced():
    fleet = pd.DataFrame(
        {
            "mac_id": ["mac-01", "mac-02", "mac-03"],
            "project_name": ["Atlas", "Nova", "Orion"],
            "utilization_pct": [48.0, 50.0, 52.0],
        }
    )
    recs = evaluate_equalization(fleet, "ram", deviation_threshold_pct=20)
    assert recs == []


def test_evaluate_equalization_handles_empty_fleet():
    recs = evaluate_equalization(pd.DataFrame(columns=["mac_id", "project_name", "utilization_pct"]), "cpu")
    assert recs == []


def test_evaluate_equalization_risk_escalates_with_larger_gap():
    fleet = pd.DataFrame(
        {
            "mac_id": ["mac-01", "mac-02"],
            "project_name": ["Atlas", "Nova"],
            "utilization_pct": [95.0, 5.0],
        }
    )
    recs = evaluate_equalization(fleet, "disk", deviation_threshold_pct=20)
    assert recs[0].risk_level == RiskLevel.CRITICAL
