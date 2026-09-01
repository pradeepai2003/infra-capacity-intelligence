import importlib

import pandas as pd

trends_module = importlib.import_module("src.databricks.notebooks.03_aggregation_trend_analysis")


def test_add_series_id_combines_mac_and_resource_type():
    df = pd.DataFrame({"mac_id": ["mac-01", "mac-02"], "resource_type": ["cpu", "ram"]})
    result = trends_module.add_series_id(df)
    assert result["series_id"].tolist() == ["mac-01__cpu", "mac-02__ram"]


def test_aggregate_daily_reduces_hourly_to_daily():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=48, freq="h"),
            "series_id": ["mac-01__cpu"] * 48,
            "value": range(48),
        }
    )
    daily = trends_module.aggregate_daily(df, "series_id", ["value"])
    assert len(daily) == 2  # 48 hours -> 2 days


def test_add_trend_indicators_adds_expected_columns():
    df = pd.DataFrame(
        {
            "series_id": ["mac-01__cpu"] * 10,
            "date": pd.date_range("2025-01-01", periods=10, freq="D"),
            "utilization_pct": range(10),
        }
    )
    result = trends_module.add_trend_indicators(df, "series_id", "utilization_pct")
    assert "utilization_pct_ma7" in result.columns
    assert "utilization_pct_ma30" in result.columns
    assert "utilization_pct_growth_rate" in result.columns


def test_process_mac_trends_end_to_end():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=48, freq="h"),
            "mac_id": ["mac-01"] * 48,
            "project_name": ["Project Atlas"] * 48,
            "resource_type": ["cpu"] * 48,
            "allocated_capacity": [10.0] * 48,
            "used_capacity": [5.0] * 48,
            "capacity_unit": ["cores"] * 48,
            "utilization_pct": [50.0] * 48,
        }
    )
    result = trends_module.process_mac_trends(df)
    assert "utilization_pct_ma7" in result.columns
    assert "mac_id" in result.columns
    assert "resource_type" in result.columns
    assert "project_name" in result.columns
    assert (result["project_name"] == "Project Atlas").all()
    assert len(result) == 2
