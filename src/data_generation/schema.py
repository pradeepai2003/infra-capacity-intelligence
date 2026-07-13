"""
Shared schema definitions and validation helpers for synthetic infrastructure
utilization data. Keeping these in one place ensures the Databricks cleansing
notebooks and the forecasting engine agree on column names/types.
"""

from __future__ import annotations

import pandas as pd

COMPUTE_COLUMNS = [
    "timestamp",
    "cluster_id",
    "cpu_utilization_pct",
    "memory_utilization_pct",
    "cluster_utilization_pct",
]

STORAGE_COLUMNS = [
    "timestamp",
    "storage_id",
    "disk_used_gb",
    "disk_total_gb",
    "disk_utilization_pct",
    "io_utilization_pct",
]

NETWORK_COLUMNS = [
    "timestamp",
    "link_id",
    "bandwidth_mbps",
    "throughput_mbps",
    "latency_ms",
]


class SchemaValidationError(ValueError):
    """Raised when a generated or ingested DataFrame does not match the expected schema."""


def validate_columns(df: pd.DataFrame, expected_columns: list[str], name: str) -> None:
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise SchemaValidationError(f"{name} is missing required columns: {sorted(missing)}")


def validate_compute(df: pd.DataFrame) -> None:
    validate_columns(df, COMPUTE_COLUMNS, "compute metrics")
    if (df["cpu_utilization_pct"] < 0).any() or (df["cpu_utilization_pct"] > 100).any():
        raise SchemaValidationError("cpu_utilization_pct out of bounds [0, 100]")


def validate_storage(df: pd.DataFrame) -> None:
    validate_columns(df, STORAGE_COLUMNS, "storage metrics")
    if (df["disk_used_gb"] > df["disk_total_gb"]).any():
        raise SchemaValidationError("disk_used_gb cannot exceed disk_total_gb")


def validate_network(df: pd.DataFrame) -> None:
    validate_columns(df, NETWORK_COLUMNS, "network metrics")
    if (df["latency_ms"] < 0).any():
        raise SchemaValidationError("latency_ms cannot be negative")
