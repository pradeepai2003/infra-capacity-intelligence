# Databricks notebook source
"""
02_data_cleansing
-------------------
Cleans the ingested interim data: deduplicates, handles missing values,
clips out-of-range readings, and drops sensor/collection glitches
(e.g. negative latency, utilization > 100%).
"""

# COMMAND ----------

from __future__ import annotations

import pandas as pd

# COMMAND ----------


def clean_compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["timestamp", "cluster_id"])
    df = df.dropna(subset=["cpu_utilization_pct", "memory_utilization_pct"])
    for col in ["cpu_utilization_pct", "memory_utilization_pct", "cluster_utilization_pct"]:
        df[col] = df[col].clip(lower=0, upper=100)
    return df.sort_values(["cluster_id", "timestamp"]).reset_index(drop=True)


def clean_storage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["timestamp", "storage_id"])
    df = df.dropna(subset=["disk_used_gb", "disk_total_gb"])
    df = df[df["disk_used_gb"] <= df["disk_total_gb"]]
    df["disk_utilization_pct"] = (df["disk_used_gb"] / df["disk_total_gb"] * 100).round(2)
    df["io_utilization_pct"] = df["io_utilization_pct"].clip(lower=0, upper=100)
    return df.sort_values(["storage_id", "timestamp"]).reset_index(drop=True)


def clean_network(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["timestamp", "link_id"])
    df = df.dropna(subset=["throughput_mbps", "latency_ms"])
    df = df[df["latency_ms"] >= 0]
    df["throughput_mbps"] = df["throughput_mbps"].clip(lower=0)
    return df.sort_values(["link_id", "timestamp"]).reset_index(drop=True)


CLEANERS = {"compute": clean_compute, "storage": clean_storage, "network": clean_network}


def clean_all(interim_dir: str = "data/interim", processed_dir: str = "data/processed") -> dict[str, pd.DataFrame]:
    cleaned = {}
    for name, cleaner_fn in CLEANERS.items():
        df = pd.read_csv(f"{interim_dir}/{name}_ingested.csv", parse_dates=["timestamp"])
        cleaned_df = cleaner_fn(df)
        cleaned_df.to_csv(f"{processed_dir}/{name}_cleaned.csv", index=False)
        cleaned[name] = cleaned_df
    return cleaned


# COMMAND ----------

if __name__ == "__main__":
    cleaned = clean_all()
    for name, df in cleaned.items():
        print(f"Cleaned {name}: {len(df)} rows")
