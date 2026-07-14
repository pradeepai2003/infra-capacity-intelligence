# Databricks notebook source
"""
04_time_series_prep
----------------------
Final step of the Databricks pipeline: reshapes trend-annotated data into the
clean per-resource time series format expected by the forecasting engine
(one row per resource per day, with a numeric `t` index for regression models).
"""

# COMMAND ----------

from __future__ import annotations

import pandas as pd

# COMMAND ----------


def prepare_series(df: pd.DataFrame, id_col: str, date_col: str, target_col: str) -> dict[str, pd.DataFrame]:
    """Split a long-format trend DataFrame into one time series per resource ID,
    with columns renamed to `ds`/`y` (Prophet convention) plus a numeric `t` index
    (for linear regression).
    """
    series_by_id = {}
    for resource_id, group in df.groupby(id_col):
        s = group[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"}).reset_index(drop=True)
        s["t"] = range(len(s))
        series_by_id[resource_id] = s
    return series_by_id


def prepare_all(processed_dir: str = "data/processed") -> dict[str, dict[str, pd.DataFrame]]:
    compute = pd.read_csv(f"{processed_dir}/compute_trends.csv", parse_dates=["date"])
    storage = pd.read_csv(f"{processed_dir}/storage_trends.csv", parse_dates=["date"])
    network = pd.read_csv(f"{processed_dir}/network_trends.csv", parse_dates=["date"])

    return {
        "compute": prepare_series(compute, "cluster_id", "date", "cluster_utilization_pct"),
        "storage": prepare_series(storage, "storage_id", "date", "disk_utilization_pct"),
        "network": prepare_series(network, "link_id", "date", "throughput_mbps"),
    }


# COMMAND ----------

if __name__ == "__main__":  # pragma: no cover
    prepared = prepare_all()
    for resource_type, series_dict in prepared.items():
        print(f"{resource_type}: prepared {len(series_dict)} time series")
