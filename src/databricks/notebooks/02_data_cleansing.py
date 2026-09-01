# Databricks notebook source
"""
02_data_cleansing
-------------------
Cleans the ingested Mac allocation data: deduplicates, handles missing
values, clips out-of-range readings, and recomputes utilization_pct from
used/allocated capacity to guard against any upstream drift between the two.
"""

# COMMAND ----------

from __future__ import annotations

import pandas as pd

# COMMAND ----------


def clean_mac_allocation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["timestamp", "mac_id", "resource_type"])
    df = df.dropna(subset=["used_capacity", "allocated_capacity"])
    df = df[df["used_capacity"] <= df["allocated_capacity"]]
    df["utilization_pct"] = (df["used_capacity"] / df["allocated_capacity"] * 100).clip(0, 100).round(2)
    return df.sort_values(["mac_id", "resource_type", "timestamp"]).reset_index(drop=True)


def clean_all(interim_dir: str = "data/interim", processed_dir: str = "data/processed") -> dict[str, pd.DataFrame]:
    df = pd.read_csv(f"{interim_dir}/mac_allocation_ingested.csv", parse_dates=["timestamp"])
    cleaned_df = clean_mac_allocation(df)
    cleaned_df.to_csv(f"{processed_dir}/mac_allocation_cleaned.csv", index=False)
    return {"mac_allocation": cleaned_df}


# COMMAND ----------

if __name__ == "__main__":  # pragma: no cover
    cleaned = clean_all()
    for name, df in cleaned.items():
        print(f"Cleaned {name}: {len(df)} rows")
