"""
Integration test: runs the full pipeline against a tiny synthetic dataset
(small num_days/clusters to keep CI fast) and checks that each stage hands
off valid data to the next, ending with non-empty Power BI export files.
"""

import os

import pandas as pd
import pytest

from src.pipeline.run_pipeline import (
    load_config,
    run,
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


def test_load_config_reads_real_config_file():
    cfg = load_config("config/config.yaml")
    assert "data_generation" in cfg
    assert "paths" in cfg
    assert "forecasting" in cfg
    assert "thresholds" in cfg
    assert "recommendation_engine" in cfg
    assert cfg["forecasting"]["horizons_weeks"] == [4, 12]


def test_run_executes_full_pipeline_end_to_end(small_cfg, monkeypatch):
    """Exercises the top-level run() orchestrator (not just the individual steps),
    by monkeypatching load_config() to return a small, fast, tmp-path-scoped config.
    """
    import src.pipeline.run_pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "load_config", lambda path="config/config.yaml": small_cfg)

    run()

    powerbi_dir = small_cfg["paths"]["powerbi_export_dir"]
    assert os.path.exists(f"{powerbi_dir}/utilization_overview.csv")
    assert os.path.exists(f"{powerbi_dir}/recommendations.csv")
    assert os.path.exists(f"{powerbi_dir}/risk_summary.csv")

    processed_dir = small_cfg["paths"]["processed_data_dir"]
    assert os.path.exists(f"{processed_dir}/recommendations.csv")
