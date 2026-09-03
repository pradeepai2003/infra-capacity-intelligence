"""
Integration test: runs the full pipeline against a tiny synthetic dataset
(small num_days/macs to keep CI fast) and checks that each stage hands off
valid data to the next, ending with non-empty Power BI export files and a
dated snapshot.
"""

import os
from datetime import datetime

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
        "topic_id": "S3-P-07",
        "data_generation": {
            "start_date": "2025-01-01",
            "num_days": 45,  # >30 so trend indicators + forecasts have enough history
            "num_macs": 4,
            "project_pool": ["Project Atlas", "Project Nova", "Project Orion", "Project Zephyr"],
            "random_seed": 1,
        },
        "paths": {
            "raw_data_dir": str(tmp_path / "raw"),
            "interim_data_dir": str(tmp_path / "interim"),
            "processed_data_dir": str(tmp_path / "processed"),
            "seeded_scenarios_dir": str(tmp_path / "seeded_scenarios"),
            "powerbi_export_dir": str(tmp_path / "powerbi"),
            "snapshot_dir": str(tmp_path / "runs"),
            "onedrive_dir": "",
            "test_report_file": str(tmp_path / "test-report-that-does-not-exist.txt"),
            "coverage_xml_file": str(tmp_path / "coverage-that-does-not-exist.xml"),
        },
        "forecasting": {"horizons_weeks": [4, 12]},
        "thresholds": {
            "overloaded_utilization_pct": 85,
            "underloaded_utilization_pct": 30,
            "equalization_deviation_pct": 20,
        },
        "recommendation_engine": {"provider": "template_fallback"},
    }


def test_full_pipeline_runs_end_to_end(small_cfg):
    dir_keys = [
        "raw_data_dir",
        "interim_data_dir",
        "processed_data_dir",
        "seeded_scenarios_dir",
        "powerbi_export_dir",
        "snapshot_dir",
    ]
    for key in dir_keys:
        os.makedirs(small_cfg["paths"][key], exist_ok=True)

    raw_df = step_generate_data(small_cfg)
    assert len(raw_df) > 0
    assert set(raw_df["mac_id"].unique()) == {"mac-01", "mac-02", "mac-03", "mac-04"}

    trends = step_clean_and_trend(raw_df, small_cfg)
    assert len(trends) > 0
    assert "series_id" in trends.columns

    forecasts = step_forecast(trends, small_cfg)
    assert len(forecasts) > 0

    recommendations_df, equalization_df = step_recommend(trends, forecasts, small_cfg)
    assert isinstance(recommendations_df, pd.DataFrame)
    assert len(recommendations_df) > 0
    assert "ai_narrative" in recommendations_df.columns
    assert recommendations_df["ai_narrative"].apply(lambda x: len(x) > 0).all()
    assert isinstance(equalization_df, pd.DataFrame)

    processed_dir = small_cfg["paths"]["processed_data_dir"]
    assert os.path.exists(f"{processed_dir}/recommendations.csv")
    assert os.path.exists(f"{processed_dir}/equalization_summary.csv")


def test_load_config_reads_real_config_file():
    cfg = load_config("config/config.yaml")
    assert "data_generation" in cfg
    assert "paths" in cfg
    assert "forecasting" in cfg
    assert "thresholds" in cfg
    assert "recommendation_engine" in cfg
    assert cfg["data_generation"]["num_macs"] == 8
    assert cfg["forecasting"]["horizons_weeks"] == [4, 12]


def test_run_executes_full_pipeline_end_to_end(small_cfg, monkeypatch):
    """Exercises the top-level run() orchestrator (not just the individual steps),
    by monkeypatching load_config() to return a small, fast, tmp-path-scoped config.
    """
    import src.pipeline.run_pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "load_config", lambda path="config/config.yaml": small_cfg)

    run()

    powerbi_dir = small_cfg["paths"]["powerbi_export_dir"]
    assert os.path.exists(f"{powerbi_dir}/mac_utilization_overview.csv")
    assert os.path.exists(f"{powerbi_dir}/recommendations.csv")
    assert os.path.exists(f"{powerbi_dir}/equalization_summary.csv")
    assert os.path.exists(f"{powerbi_dir}/risk_summary.csv")

    processed_dir = small_cfg["paths"]["processed_data_dir"]
    assert os.path.exists(f"{processed_dir}/recommendations.csv")

    snapshot_dir = small_cfg["paths"]["snapshot_dir"]
    expected_folder = f"S3-P-07 - {datetime.now().strftime('%Y-%m-%d')}"
    assert os.path.isdir(os.path.join(snapshot_dir, expected_folder))
    assert os.path.exists(os.path.join(snapshot_dir, expected_folder, "powerbi", "recommendations.csv"))


def test_dotenv_is_loaded_on_module_import(tmp_path, monkeypatch):
    """Confirms a local .env file is actually picked up (not just documented) --
    writes a temp .env, points the working directory at it, and checks the
    variable becomes visible via os.getenv() after reloading the module.
    """
    import importlib
    import os

    env_file = tmp_path / ".env"
    env_file.write_text("TEST_DOTENV_MARKER=hello_from_dotenv\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TEST_DOTENV_MARKER", raising=False)

    import src.pipeline.run_pipeline as pipeline_module

    importlib.reload(pipeline_module)

    assert os.getenv("TEST_DOTENV_MARKER") == "hello_from_dotenv"


def test_generate_narratives_with_pacing_sleeps_between_gemini_calls(monkeypatch):
    import src.pipeline.run_pipeline as pipeline_module

    sleep_calls = []
    monkeypatch.setattr(pipeline_module.time, "sleep", lambda s: sleep_calls.append(s))

    def fake_generator(item, provider):
        return f"narrative-for-{item}"

    result = pipeline_module._generate_narratives_with_pacing(["a", "b", "c"], fake_generator, "gemini")

    assert result == ["narrative-for-a", "narrative-for-b", "narrative-for-c"]
    assert len(sleep_calls) == 2  # paced between items, not after the last one


def test_generate_narratives_with_pacing_skips_sleep_for_non_gemini_provider(monkeypatch):
    import src.pipeline.run_pipeline as pipeline_module

    sleep_calls = []
    monkeypatch.setattr(pipeline_module.time, "sleep", lambda s: sleep_calls.append(s))

    def fake_generator(item, provider):
        return f"narrative-for-{item}"

    result = pipeline_module._generate_narratives_with_pacing(["a", "b"], fake_generator, "template_fallback")

    assert result == ["narrative-for-a", "narrative-for-b"]
    assert sleep_calls == []
