# Databricks notebook source
"""
04_time_series_prep
----------------------
Final step of the Databricks pipeline: reshapes trend-annotated data into the
clean per-series time series format expected by the forecasting engine (one
row per series per day, with a numeric `t` index for regression models).

Each (mac_id, resource_type) pair is treated as one independent series via
the "series_id" column produced in step 03 (e.g. "mac-03__cpu"), so the
forecasting engine below never needs to know it's dealing with Macs
specifically -- it just forecasts N independent series.
"""

# COMMAND ----------

from __future__ import annotations

import pandas as pd

# COMMAND ----------


def prepare_series(df: pd.DataFrame, id_col: str, date_col: str, target_col: str) -> dict[str, pd.DataFrame]:
    """Split a long-format trend DataFrame into one time series per series_id,
    with columns renamed to `ds`/`y` (Prophet convention) plus a numeric `t` index
    (for linear regression).
    """
    series_by_id = {}
    for series_id, group in df.groupby(id_col):
        s = group[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"}).reset_index(drop=True)
        s["t"] = range(len(s))
        series_by_id[series_id] = s
    return series_by_id


def prepare_all(processed_dir: str = "data/processed") -> dict[str, pd.DataFrame]:
    df = pd.read_csv(f"{processed_dir}/mac_allocation_trends.csv", parse_dates=["date"])
    return prepare_series(df, "series_id", "date", "utilization_pct")


# COMMAND ----------

if __name__ == "__main__":  # pragma: no cover
    prepared = prepare_all()
    print(f"Prepared {len(prepared)} Mac x resource-type time series")
