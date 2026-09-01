# Databricks notebook source
"""
01_ingest_raw_data
-------------------
Reads the raw synthetic (or eventually real, Jamf/MDM-collected) Mac
allocation CSV from the raw data landing zone and validates it against the
shared schema before handing it to the cleansing stage.

Runs standalone with plain pandas when not on a Databricks cluster, so it can
be unit tested locally / in CI. When executed inside Databricks, swap the
pandas read/write calls for spark.read.csv / df.write.format("delta") as
noted in the comments below.
"""

# COMMAND ----------

from __future__ import annotations

import pandas as pd

from src.data_generation.schema import validate_mac_allocation

# COMMAND ----------


def ingest_csv(path: str) -> pd.DataFrame:
    """Read a CSV and validate it against the Mac allocation schema before returning it.

    On Databricks, replace with:
        df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    validate_mac_allocation(df)
    return df


def ingest_all(raw_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    return {"mac_allocation": ingest_csv(f"{raw_dir}/mac_allocation_metrics.csv")}


def write_interim(datasets: dict[str, pd.DataFrame], interim_dir: str = "data/interim") -> None:
    """Persist ingested data to the interim zone. Replace with Delta write on Databricks:
    df.write.format("delta").mode("overwrite").save(f"{interim_dir}/{name}")
    """
    for name, df in datasets.items():
        df.to_csv(f"{interim_dir}/{name}_ingested.csv", index=False)


# COMMAND ----------

if __name__ == "__main__":  # pragma: no cover
    datasets = ingest_all()
    write_interim(datasets)
    for name, df in datasets.items():
        print(f"Ingested {name}: {len(df)} rows")
