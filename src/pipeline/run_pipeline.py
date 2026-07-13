"""
End-to-end pipeline orchestrator:

  1. Generate synthetic data (or, in future, ingest real Azure Monitor data)
  2. Run the Databricks-style cleansing/aggregation/time-series-prep steps
  3. Run forecasting (Linear Regression + Prophet) for every resource
  4. Run the rule engine + AI narrative generator to produce recommendations
  5. Export the final dataset for Power BI

Designed to be called both as a script (`python -m src.pipeline.run_pipeline`)
and from the GitHub Actions data-pipeline workflow.
"""

from __future__ import annotations

import importlib
import logging
import os

import pandas as pd
import yaml

from src.data_generation.generate_compute_metrics import generate_compute_metrics
from src.data_generation.generate_network_metrics import generate_network_metrics
from src.data_generation.generate_storage_metrics import generate_storage_metrics
from src.data_generation.scenario_seeder import generate_all_scenarios
from src.forecasting.forecast_runner import run_forecast_for_all
from src.powerbi.dataset_export import export_for_powerbi
from src.recommendation_engine.ai_narrative_generator import generate_narrative
from src.recommendation_engine.recommendation_schema import recommendations_to_dataframe
from src.recommendation_engine.rule_engine import evaluate_compute, evaluate_network_spike, evaluate_storage

# Notebook-style modules are prefixed with digits (01_, 02_, ...) to preserve
# their execution order when browsed in Databricks/VS Code. Python identifiers
# can't start with a digit, so they can't be imported with a normal `from ... import`
# statement -- we load them dynamically with importlib instead.
_cleansing = importlib.import_module("src.databricks.notebooks.02_data_cleansing")
_trends = importlib.import_module("src.databricks.notebooks.03_aggregation_trend_analysis")
_ts_prep = importlib.import_module("src.databricks.notebooks.04_time_series_prep")

clean_compute = _cleansing.clean_compute
clean_storage = _cleansing.clean_storage
clean_network = _cleansing.clean_network

process_compute_trends = _trends.process_compute_trends
process_storage_trends = _trends.process_storage_trends
process_network_trends = _trends.process_network_trends

prepare_series = _ts_prep.prepare_series

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def step_generate_data(cfg: dict) -> dict[str, pd.DataFrame]:
    logger.info("Step 1/5: generating synthetic data")
    gen_cfg = cfg["data_generation"]
    compute = generate_compute_metrics(
        gen_cfg["start_date"], gen_cfg["num_days"], gen_cfg["num_compute_clusters"], seed=gen_cfg["random_seed"]
    )
    storage = generate_storage_metrics(
        gen_cfg["start_date"], gen_cfg["num_days"], gen_cfg["num_storage_systems"], seed=gen_cfg["random_seed"]
    )
    network = generate_network_metrics(
        gen_cfg["start_date"], gen_cfg["num_days"], gen_cfg["num_network_links"], seed=gen_cfg["random_seed"]
    )
    generate_all_scenarios(cfg["paths"]["seeded_scenarios_dir"])
    return {"compute": compute, "storage": storage, "network": network}


def step_clean_and_trend(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger.info("Step 2/5: cleansing + trend analysis (Databricks-style)")
    compute_clean = clean_compute(raw["compute"])
    storage_clean = clean_storage(raw["storage"])
    network_clean = clean_network(raw["network"])

    return {
        "compute": process_compute_trends(compute_clean),
        "storage": process_storage_trends(storage_clean),
        "network": process_network_trends(network_clean),
    }


def step_forecast(trends: dict[str, pd.DataFrame], cfg: dict) -> dict:
    logger.info("Step 3/5: forecasting")
    horizons = cfg["forecasting"]["horizons_weeks"]

    compute_series = prepare_series(trends["compute"], "cluster_id", "date", "cluster_utilization_pct")
    storage_series = prepare_series(trends["storage"], "storage_id", "date", "disk_utilization_pct")
    network_series = prepare_series(trends["network"], "link_id", "date", "throughput_mbps")

    return {
        "compute": run_forecast_for_all(compute_series, horizons),
        "storage": run_forecast_for_all(storage_series, horizons),
        "network": run_forecast_for_all(network_series, horizons),
        "raw_series": {"compute": compute_series, "storage": storage_series, "network": network_series},
    }


def step_recommend(trends: dict[str, pd.DataFrame], forecasts: dict, cfg: dict) -> pd.DataFrame:
    logger.info("Step 4/5: generating recommendations + AI narratives")
    thresholds = cfg["thresholds"]
    provider = cfg["recommendation_engine"]["provider"]
    all_recs = []

    for storage_id, group in trends["storage"].groupby("storage_id"):
        current = float(group["disk_utilization_pct"].iloc[-1])
        f4 = forecasts["storage"].get(storage_id, {}).get("prophet", {}).get(4)
        f12 = forecasts["storage"].get(storage_id, {}).get("prophet", {}).get(12)
        all_recs.extend(
            evaluate_storage(
                storage_id,
                current,
                f4,
                f12,
                thresholds["storage_utilization_critical_pct"],
                thresholds["storage_utilization_warning_pct"],
            )
        )

    for cluster_id, group in trends["compute"].groupby("cluster_id"):
        all_recs.append(
            evaluate_compute(
                cluster_id,
                group["cluster_utilization_pct"],
                thresholds["compute_underutilization_pct"],
                thresholds["compute_overutilization_pct"],
            )
        )

    for link_id, group in trends["network"].groupby("link_id"):
        throughput_pct = group["throughput_mbps"] / group["throughput_mbps"].max() * 100
        all_recs.append(evaluate_network_spike(link_id, throughput_pct))

    df = recommendations_to_dataframe(all_recs)
    df["ai_narrative"] = [generate_narrative(r, provider=provider) for r in all_recs]
    return df


def run() -> None:
    cfg = load_config()
    for d in cfg["paths"].values():
        os.makedirs(d, exist_ok=True)

    raw = step_generate_data(cfg)
    trends = step_clean_and_trend(raw)
    forecasts = step_forecast(trends, cfg)
    recommendations_df = step_recommend(trends, forecasts, cfg)

    logger.info("Step 5/5: exporting for Power BI")
    export_for_powerbi(trends, recommendations_df, cfg["paths"]["powerbi_export_dir"])

    logger.info("Pipeline complete. %d recommendations generated.", len(recommendations_df))


if __name__ == "__main__":
    run()
