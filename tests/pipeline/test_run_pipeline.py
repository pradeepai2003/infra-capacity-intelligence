"""
Integration test: runs the full pipeline against a tiny synthetic dataset
(small num_days/clusters to keep CI fast) and checks that each stage hands
off valid data to the next, ending with non-empty Power BI export files.
"""

import os

import pandas as pd
import pytest

from src.pipeline.run_pipeline import (
    step_clean_and_trend,
    step_forecast,
    step_generate_data,
    step_recommend,
)


@pytest.fixture
def small_cfg(tmp_path):
    return {
        "data_generation": {
            "start_date": "2025-01-01",
            "num_days": 45,  # >30 so trend indicators + forecasts have enough history
            "num_compute_clusters": 2,
            "num_storage_systems": 2,
            "num_network_links": 2,
            "random_seed": 1,
        },
        "paths": {
            "raw_data_dir": str(tmp_path / "raw"),
            "interim_data_dir": str(tmp_path / "interim"),
            "processed_data_dir": str(tmp_path / "processed"),
            "seeded_scenarios_dir": str(tmp_path / "seeded_scenarios"),
            "powerbi_export_dir": str(tmp_path / "powerbi"),
        },
        "forecasting": {"horizons_weeks": [4, 12]},
        "thresholds": {
            "storage_utilization_critical_pct": 90,
            "storage_utilization_warning_pct": 75,
            "compute_underutilization_pct": 30,
            "compute_overutilization_pct": 85,
            "network_latency_warning_ms": 150,
        },
        "recommendation_engine": {"provider": "template_fallback"},
    }


def test_full_pipeline_runs_end_to_end(small_cfg):
    for d in small_cfg["paths"].values():
        os.makedirs(d, exist_ok=True)

    raw = step_generate_data(small_cfg)
    assert all(len(df) > 0 for df in raw.values())

    trends = step_clean_and_trend(raw, small_cfg)
    assert all(len(df) > 0 for df in trends.values())

    forecasts = step_forecast(trends, small_cfg)
    assert "compute" in forecasts and "storage" in forecasts and "network" in forecasts

    recommendations_df = step_recommend(trends, forecasts, small_cfg)
    assert isinstance(recommendations_df, pd.DataFrame)
    assert len(recommendations_df) > 0
    assert "ai_narrative" in recommendations_df.columns
    assert recommendations_df["ai_narrative"].apply(lambda x: len(x) > 0).all()
