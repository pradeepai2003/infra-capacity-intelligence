# Databricks notebook source
"""
03_aggregation_trend_analysis
-------------------------------
Aggregates cleaned Mac allocation data to daily granularity per (mac, resource
type) and computes rolling trend indicators (7-day / 30-day moving averages,
day-over-day growth rate) used both for dashboarding and as forecasting
features.

Internally, each (mac_id, resource_type) pair is combined into a single
"series_id" (e.g. "mac-03__cpu") so the same generic aggregation/trend
helpers work regardless of how many Macs or resource types exist -- mac_id,
resource_type, and project_name are split/reattached afterward for
readability in the output.
"""

# COMMAND ----------

from __future__ import annotations

import pandas as pd

# COMMAND ----------


def add_series_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["series_id"] = df["mac_id"] + "__" + df["resource_type"]
    return df


def aggregate_daily(df: pd.DataFrame, id_col: str, value_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = df.groupby([id_col, "date"], as_index=False)[value_cols].mean()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values([id_col, "date"]).reset_index(drop=True)


def add_trend_indicators(df: pd.DataFrame, id_col: str, value_col: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{value_col}_ma7"] = df.groupby(id_col)[value_col].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df[f"{value_col}_ma30"] = df.groupby(id_col)[value_col].transform(lambda s: s.rolling(30, min_periods=1).mean())
    df[f"{value_col}_growth_rate"] = df.groupby(id_col)[value_col].transform(lambda s: s.pct_change().fillna(0))
    return df


def process_mac_trends(df: pd.DataFrame) -> pd.DataFrame:
    df_with_series = add_series_id(df)
    daily = aggregate_daily(df_with_series, "series_id", ["utilization_pct", "used_capacity", "allocated_capacity"])
    trended = add_trend_indicators(daily, "series_id", "utilization_pct")

    trended[["mac_id", "resource_type"]] = trended["series_id"].str.split("__", expand=True)
    project_lookup = df.drop_duplicates("mac_id").set_index("mac_id")["project_name"]
    trended["project_name"] = trended["mac_id"].map(project_lookup)
    return trended


# COMMAND ----------

if __name__ == "__main__":  # pragma: no cover
    df = pd.read_csv("data/processed/mac_allocation_cleaned.csv", parse_dates=["timestamp"])
    process_mac_trends(df).to_csv("data/processed/mac_allocation_trends.csv", index=False)
    print("Trend analysis complete.")
