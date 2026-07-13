"""
End-to-end pipeline orchestrator:

  1. Generate synthetic data (or, in future, ingest real Azure Monitor data)
  2. Run the Databricks-style cleansing/aggregation/time-series-prep steps
  3. Run forecasting (Linear Regression + Prophet) for every resource
  4. Run the rule engine + AI narrative generator to produce recommendations
  5. Export the final dataset for Power BI

Every stage below writes its output to disk immediately AND prints a
"[SAVED] ..." confirmation line to the terminal (via src.pipeline.io_utils),
so you can see exactly what file was written, where, and how many rows it
contains, as each step completes.

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
from src.pipeline.io_utils import save_and_log
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _banner(step_text: str) -> None:
    logger.info("\n%s\n%s\n%s", "=" * 70, step_text, "=" * 70)


def step_generate_data(cfg: dict) -> dict[str, pd.DataFrame]:
    _banner("STEP 1/5: Generating synthetic data")
    gen_cfg = cfg["data_generation"]
    raw_dir = cfg["paths"]["raw_data_dir"]

    compute = generate_compute_metrics(
        gen_cfg["start_date"], gen_cfg["num_days"], gen_cfg["num_compute_clusters"], seed=gen_cfg["random_seed"]
    )
    save_and_log(compute, f"{raw_dir}/compute_metrics.csv", "Synthetic compute metrics")

    storage = generate_storage_metrics(
        gen_cfg["start_date"], gen_cfg["num_days"], gen_cfg["num_storage_systems"], seed=gen_cfg["random_seed"]
    )
    save_and_log(storage, f"{raw_dir}/storage_metrics.csv", "Synthetic storage metrics")

    network = generate_network_metrics(
        gen_cfg["start_date"], gen_cfg["num_days"], gen_cfg["num_network_links"], seed=gen_cfg["random_seed"]
    )
    save_and_log(network, f"{raw_dir}/network_metrics.csv", "Synthetic network metrics")

    # generate_all_scenarios already saves + logs each of the 3 seeded scenario CSVs
    generate_all_scenarios(cfg["paths"]["seeded_scenarios_dir"])

    return {"compute": compute, "storage": storage, "network": network}


def step_clean_and_trend(raw: dict[str, pd.DataFrame], cfg: dict) -> dict[str, pd.DataFrame]:
    _banner("STEP 2/5: Cleansing + trend analysis (Databricks-style)")
    processed_dir = cfg["paths"]["processed_data_dir"]

    compute_clean = clean_compute(raw["compute"])
    save_and_log(compute_clean, f"{processed_dir}/compute_cleaned.csv", "Cleaned compute data")

    storage_clean = clean_storage(raw["storage"])
    save_and_log(storage_clean, f"{processed_dir}/storage_cleaned.csv", "Cleaned storage data")

    network_clean = clean_network(raw["network"])
    save_and_log(network_clean, f"{processed_dir}/network_cleaned.csv", "Cleaned network data")

    compute_trends = process_compute_trends(compute_clean)
    save_and_log(compute_trends, f"{processed_dir}/compute_trends.csv", "Compute trend indicators")

    storage_trends = process_storage_trends(storage_clean)
    save_and_log(storage_trends, f"{processed_dir}/storage_trends.csv", "Storage trend indicators")

    network_trends = process_network_trends(network_clean)
    save_and_log(network_trends, f"{processed_dir}/network_trends.csv", "Network trend indicators")

    return {"compute": compute_trends, "storage": storage_trends, "network": network_trends}


def step_forecast(trends: dict[str, pd.DataFrame], cfg: dict) -> dict:
    _banner("STEP 3/5: Forecasting (Linear Regression + Prophet)")
    horizons = cfg["forecasting"]["horizons_weeks"]

    compute_series = prepare_series(trends["compute"], "cluster_id", "date", "cluster_utilization_pct")
    storage_series = prepare_series(trends["storage"], "storage_id", "date", "disk_utilization_pct")
    network_series = prepare_series(trends["network"], "link_id", "date", "throughput_mbps")

    forecasts = {
        "compute": run_forecast_for_all(compute_series, horizons),
        "storage": run_forecast_for_all(storage_series, horizons),
        "network": run_forecast_for_all(network_series, horizons),
        "raw_series": {"compute": compute_series, "storage": storage_series, "network": network_series},
    }

    logger.info(
        "[FORECASTED] %d compute + %d storage + %d network resources, horizons=%s weeks",
        len(compute_series),
        len(storage_series),
        len(network_series),
        horizons,
    )

    return forecasts


def step_recommend(trends: dict[str, pd.DataFrame], forecasts: dict, cfg: dict) -> pd.DataFrame:
    _banner("STEP 4/5: Generating recommendations + AI narratives")
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

    logger.info("[AI NARRATIVE] generating narratives for %d recommendations via '%s'...", len(all_recs), provider)
    df["ai_narrative"] = [generate_narrative(r, provider=provider) for r in all_recs]

    processed_dir = cfg["paths"]["processed_data_dir"]
    save_and_log(df, f"{processed_dir}/recommendations.csv", "AI recommendations")

    critical_count = int((df["risk_level"] == "critical").sum())
    warning_count = int((df["risk_level"] == "warning").sum())
    logger.info("[SUMMARY] %d critical, %d warning recommendation(s) generated", critical_count, warning_count)

    return df


def run() -> None:
    cfg = load_config()
    for d in cfg["paths"].values():
        os.makedirs(d, exist_ok=True)

    raw = step_generate_data(cfg)
    trends = step_clean_and_trend(raw, cfg)
    forecasts = step_forecast(trends, cfg)
    recommendations_df = step_recommend(trends, forecasts, cfg)

    _banner("STEP 5/5: Exporting Power BI datasets")
    export_for_powerbi(trends, recommendations_df, cfg["paths"]["powerbi_export_dir"])

    logger.info("\nPipeline complete. %d total recommendations generated.\n", len(recommendations_df))


if __name__ == "__main__":
    run()
