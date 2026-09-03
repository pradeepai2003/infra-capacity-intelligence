"""
End-to-end pipeline orchestrator for Psiog's 8-Mac resource allocation
platform:

  1. Generate synthetic Mac allocation data (8 Macs x cpu/ram/disk)
  2. Run the Databricks-style cleansing/aggregation/time-series-prep steps
  3. Run forecasting (Linear Regression + Prophet) for every (mac, resource) series
  4. Run the rule engine (per-Mac) + the fleet equalization algorithm (cross-Mac),
     then generate AI narratives for both via Gemini/Ollama
  5. Export the final datasets for Power BI (per-Mac pages + the final
     Recommendation/Equalization dashboard)
  6. Bundle everything into a dated snapshot folder

Every stage writes its output to disk immediately and logs a
"[SAVED] ..." confirmation, so progress is visible in real time.
"""

from __future__ import annotations

import importlib
import logging
import os
import time

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.data_generation.generate_mac_allocation_metrics import generate_mac_allocation_metrics
from src.data_generation.scenario_seeder import generate_all_scenarios
from src.pipeline.io_utils import save_and_log
from src.pipeline.snapshot import create_dated_snapshot
from src.powerbi.dataset_export import export_for_powerbi
from src.recommendation_engine.ai_narrative_generator import (
    GEMINI_PACING_SECONDS,
    generate_equalization_narrative,
    generate_narrative,
)
from src.recommendation_engine.recommendation_schema import equalization_to_dataframe, recommendations_to_dataframe
from src.recommendation_engine.rule_engine import evaluate_equalization, evaluate_mac_resource
from src.forecasting.forecast_runner import run_forecast_for_all

# Loads variables from a local .env file (if present) into the process
# environment, so GEMINI_API_KEY / OLLAMA_HOST / etc. from config/.env.example
# work automatically without manually exporting them in every shell session.
# Safe no-op if no .env file exists (e.g. in CI, where secrets are injected
# directly as real environment variables instead).
load_dotenv()

# Notebook-style modules are prefixed with digits (01_, 02_, ...) to preserve
# their execution order when browsed in Databricks/VS Code. Python identifiers
# can't start with a digit, so they can't be imported with a normal `from ... import`
# statement -- we load them dynamically with importlib instead.
_cleansing = importlib.import_module("src.databricks.notebooks.02_data_cleansing")
_trends = importlib.import_module("src.databricks.notebooks.03_aggregation_trend_analysis")
_ts_prep = importlib.import_module("src.databricks.notebooks.04_time_series_prep")

clean_mac_allocation = _cleansing.clean_mac_allocation
process_mac_trends = _trends.process_mac_trends
prepare_series = _ts_prep.prepare_series

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _banner(step_text: str) -> None:
    logger.info("\n%s\n%s\n%s", "=" * 70, step_text, "=" * 70)


def _generate_narratives_with_pacing(items: list, generator_fn, provider: str) -> list[str]:
    """Calls generator_fn(item, provider=provider) for each item, pacing calls
    when using Gemini so a batch of 20-30 recommendations doesn't immediately
    trigger the free tier's requests-per-minute rate limit. No pacing needed
    for Ollama (local) or the template fallback (no network call at all).
    """
    narratives = []
    for i, item in enumerate(items):
        narratives.append(generator_fn(item, provider=provider))
        if provider == "gemini" and i < len(items) - 1:
            time.sleep(GEMINI_PACING_SECONDS)
    return narratives


def step_generate_data(cfg: dict) -> pd.DataFrame:
    _banner("STEP 1/6: Generating synthetic Mac allocation data")
    gen_cfg = cfg["data_generation"]
    raw_dir = cfg["paths"]["raw_data_dir"]

    df = generate_mac_allocation_metrics(
        start_date=gen_cfg["start_date"],
        num_days=gen_cfg["num_days"],
        num_macs=gen_cfg["num_macs"],
        project_pool=gen_cfg["project_pool"],
        seed=gen_cfg["random_seed"],
    )
    save_and_log(df, f"{raw_dir}/mac_allocation_metrics.csv", "Synthetic Mac allocation data")

    # generate_all_scenarios already saves + logs each of the 3 seeded scenario CSVs
    generate_all_scenarios(cfg["paths"]["seeded_scenarios_dir"])

    return df


def step_clean_and_trend(raw_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    _banner("STEP 2/6: Cleansing + trend analysis (Databricks-style)")
    processed_dir = cfg["paths"]["processed_data_dir"]

    cleaned = clean_mac_allocation(raw_df)
    save_and_log(cleaned, f"{processed_dir}/mac_allocation_cleaned.csv", "Cleaned Mac allocation data")

    trends = process_mac_trends(cleaned)
    save_and_log(trends, f"{processed_dir}/mac_allocation_trends.csv", "Mac allocation trend indicators")

    return trends


def step_forecast(trends: pd.DataFrame, cfg: dict) -> dict:
    _banner("STEP 3/6: Forecasting (Linear Regression + Prophet)")
    horizons = cfg["forecasting"]["horizons_weeks"]

    series_by_id = prepare_series(trends, "series_id", "date", "utilization_pct")
    forecasts = run_forecast_for_all(series_by_id, horizons)

    logger.info(
        "[FORECASTED] %d Mac x resource-type series, horizons=%s weeks",
        len(series_by_id),
        horizons,
    )
    return forecasts


def step_recommend(trends: pd.DataFrame, forecasts: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    _banner("STEP 4/6: Generating per-Mac recommendations + fleet equalization")
    thresholds = cfg["thresholds"]
    provider = cfg["recommendation_engine"]["provider"]

    # --- Per-(mac, resource_type) recommendations ---
    per_mac_recs = []
    for series_id, group in trends.groupby("series_id"):
        group = group.sort_values("date")
        mac_id = group["mac_id"].iloc[0]
        resource_type = group["resource_type"].iloc[0]
        project_name = group["project_name"].iloc[0]

        f4 = forecasts.get(series_id, {}).get("prophet", {}).get(4)
        f12 = forecasts.get(series_id, {}).get("prophet", {}).get(12)

        rec = evaluate_mac_resource(
            mac_id,
            resource_type,
            project_name,
            group["utilization_pct"],
            forecast_4wk=f4,
            forecast_12wk=f12,
            overloaded_threshold=thresholds["overloaded_utilization_pct"],
            underloaded_threshold=thresholds["underloaded_utilization_pct"],
        )
        per_mac_recs.append(rec)

    recommendations_df = recommendations_to_dataframe(per_mac_recs)
    logger.info("[AI NARRATIVE] generating narratives for %d recommendations via '%s'...", len(per_mac_recs), provider)
    recommendations_df["ai_narrative"] = _generate_narratives_with_pacing(per_mac_recs, generate_narrative, provider)

    processed_dir = cfg["paths"]["processed_data_dir"]
    save_and_log(recommendations_df, f"{processed_dir}/recommendations.csv", "Per-Mac recommendations")

    # --- Fleet-wide equalization, one pass per resource type ---
    equalization_recs = []
    latest_date = trends["date"].max()
    for resource_type, group in trends[trends["date"] == latest_date].groupby("resource_type"):
        fleet_snapshot = group[["mac_id", "project_name", "utilization_pct"]]
        equalization_recs.extend(
            evaluate_equalization(fleet_snapshot, resource_type, thresholds["equalization_deviation_pct"])
        )

    equalization_df = equalization_to_dataframe(equalization_recs)
    if equalization_recs:
        equalization_df["ai_narrative"] = _generate_narratives_with_pacing(
            equalization_recs, generate_equalization_narrative, provider
        )
    else:
        equalization_df["ai_narrative"] = []
    save_and_log(equalization_df, f"{processed_dir}/equalization_summary.csv", "Fleet equalization recommendations")

    critical_count = int((recommendations_df["risk_level"] == "critical").sum()) if not recommendations_df.empty else 0
    warning_count = int((recommendations_df["risk_level"] == "warning").sum()) if not recommendations_df.empty else 0
    logger.info(
        "[SUMMARY] %d critical, %d warning per-Mac recommendation(s); %d fleet-rebalancing suggestion(s)",
        critical_count,
        warning_count,
        len(equalization_recs),
    )

    return recommendations_df, equalization_df


def run() -> None:
    cfg = load_config()
    dir_keys = [
        "raw_data_dir",
        "interim_data_dir",
        "processed_data_dir",
        "seeded_scenarios_dir",
        "powerbi_export_dir",
        "snapshot_dir",
    ]
    for key in dir_keys:
        os.makedirs(cfg["paths"][key], exist_ok=True)

    raw_df = step_generate_data(cfg)
    trends = step_clean_and_trend(raw_df, cfg)
    forecasts = step_forecast(trends, cfg)
    recommendations_df, equalization_df = step_recommend(trends, forecasts, cfg)

    _banner("STEP 5/6: Exporting Power BI datasets")
    export_for_powerbi(trends, recommendations_df, equalization_df, cfg["paths"]["powerbi_export_dir"])

    _banner("STEP 6/6: Creating dated run snapshot")
    create_dated_snapshot(cfg)

    logger.info(
        "\nPipeline complete. %d per-Mac recommendations, %d fleet-rebalancing suggestions.\n",
        len(recommendations_df),
        len(equalization_df),
    )


if __name__ == "__main__":  # pragma: no cover
    run()
