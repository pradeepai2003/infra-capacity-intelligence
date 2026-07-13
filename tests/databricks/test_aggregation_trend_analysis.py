import importlib

import pandas as pd

trends_module = importlib.import_module("src.databricks.notebooks.03_aggregation_trend_analysis")


def test_aggregate_daily_reduces_hourly_to_daily():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=48, freq="h"),
            "cluster_id": ["c1"] * 48,
            "value": range(48),
        }
    )
    daily = trends_module.aggregate_daily(df, "cluster_id", ["value"])
    assert len(daily) == 2  # 48 hours -> 2 days


def test_add_trend_indicators_adds_expected_columns():
    df = pd.DataFrame(
        {
            "cluster_id": ["c1"] * 10,
            "date": pd.date_range("2025-01-01", periods=10, freq="D"),
            "utilization": range(10),
        }
    )
    result = trends_module.add_trend_indicators(df, "cluster_id", "utilization")
    assert "utilization_ma7" in result.columns
    assert "utilization_ma30" in result.columns
    assert "utilization_growth_rate" in result.columns


def test_process_compute_trends_end_to_end():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=48, freq="H"),
            "cluster_id": ["c1"] * 48,
            "cpu_utilization_pct": [50.0] * 48,
            "memory_utilization_pct": [40.0] * 48,
            "cluster_utilization_pct": [45.0] * 48,
        }
    )
    result = trends_module.process_compute_trends(df)
    assert "cluster_utilization_pct_ma7" in result.columns
    assert len(result) == 2
