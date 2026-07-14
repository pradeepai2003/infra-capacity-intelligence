# Databricks notebook source
"""
03_aggregation_trend_analysis
-------------------------------
Aggregates cleaned data to daily granularity per resource and computes
rolling trend indicators (7-day / 30-day moving averages, day-over-day
growth rate) used both for dashboarding and as forecasting features.
"""

# COMMAND ----------

from __future__ import annotations

import pandas as pd

# COMMAND ----------


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


def process_compute_trends(df: pd.DataFrame) -> pd.DataFrame:
    daily = aggregate_daily(
        df, "cluster_id", ["cpu_utilization_pct", "memory_utilization_pct", "cluster_utilization_pct"]
    )
    return add_trend_indicators(daily, "cluster_id", "cluster_utilization_pct")


def process_storage_trends(df: pd.DataFrame) -> pd.DataFrame:
    daily = aggregate_daily(
        df, "storage_id", ["disk_used_gb", "disk_total_gb", "disk_utilization_pct", "io_utilization_pct"]
    )
    return add_trend_indicators(daily, "storage_id", "disk_utilization_pct")


def process_network_trends(df: pd.DataFrame) -> pd.DataFrame:
    daily = aggregate_daily(df, "link_id", ["bandwidth_mbps", "throughput_mbps", "latency_ms"])
    return add_trend_indicators(daily, "link_id", "throughput_mbps")


# COMMAND ----------

if __name__ == "__main__":  # pragma: no cover
    compute = pd.read_csv("data/processed/compute_cleaned.csv", parse_dates=["timestamp"])
    storage = pd.read_csv("data/processed/storage_cleaned.csv", parse_dates=["timestamp"])
    network = pd.read_csv("data/processed/network_cleaned.csv", parse_dates=["timestamp"])

    process_compute_trends(compute).to_csv("data/processed/compute_trends.csv", index=False)
    process_storage_trends(storage).to_csv("data/processed/storage_trends.csv", index=False)
    process_network_trends(network).to_csv("data/processed/network_trends.csv", index=False)
    print("Trend analysis complete.")
