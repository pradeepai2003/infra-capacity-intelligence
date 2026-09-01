"""
Shared schema definitions and validation helpers for synthetic Mac resource
allocation data. Every Mac (mac-01 .. mac-08) is tracked across 3 resource
types (cpu, ram, disk) as a single long-format table, so the Databricks
cleansing notebooks, forecasting engine, and recommendation engine all agree
on one column set.
"""

from __future__ import annotations

import pandas as pd

MAC_ALLOCATION_COLUMNS = [
    "timestamp",
    "mac_id",
    "project_name",
    "resource_type",
    "allocated_capacity",
    "used_capacity",
    "capacity_unit",
    "utilization_pct",
]

RESOURCE_TYPES = ("cpu", "ram", "disk")
CAPACITY_UNITS = {"cpu": "cores", "ram": "GB", "disk": "GB"}


class SchemaValidationError(ValueError):
    """Raised when a generated or ingested DataFrame does not match the expected schema."""


def validate_columns(df: pd.DataFrame, expected_columns: list[str], name: str) -> None:
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise SchemaValidationError(f"{name} is missing required columns: {sorted(missing)}")


def validate_mac_allocation(df: pd.DataFrame) -> None:
    validate_columns(df, MAC_ALLOCATION_COLUMNS, "Mac allocation data")

    if (df["utilization_pct"] < 0).any() or (df["utilization_pct"] > 100).any():
        raise SchemaValidationError("utilization_pct out of bounds [0, 100]")

    if (df["used_capacity"] > df["allocated_capacity"]).any():
        raise SchemaValidationError("used_capacity cannot exceed allocated_capacity")

    invalid_types = set(df["resource_type"].unique()) - set(RESOURCE_TYPES)
    if invalid_types:
        raise SchemaValidationError(f"Unknown resource_type value(s): {sorted(invalid_types)}")
