import pandas as pd
import pytest

from src.data_generation.schema import (
    SchemaValidationError,
    validate_columns,
    validate_mac_allocation,
)


def _valid_row(**overrides):
    row = {
        "timestamp": pd.to_datetime(["2025-01-01"]),
        "mac_id": ["mac-01"],
        "project_name": ["Project Atlas"],
        "resource_type": ["cpu"],
        "allocated_capacity": [10.0],
        "used_capacity": [5.0],
        "capacity_unit": ["cores"],
        "utilization_pct": [50.0],
    }
    row.update(overrides)
    return pd.DataFrame(row)


def test_validate_columns_raises_on_missing_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(SchemaValidationError, match="missing required columns"):
        validate_columns(df, ["a", "b", "c"], "test data")


def test_validate_columns_passes_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    validate_columns(df, ["a", "b"], "test data")  # should not raise


def test_validate_mac_allocation_passes_on_valid_data():
    validate_mac_allocation(_valid_row())  # should not raise


def test_validate_mac_allocation_raises_on_missing_columns():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01"]), "mac_id": ["mac-01"]})
    with pytest.raises(SchemaValidationError, match="missing required columns"):
        validate_mac_allocation(df)


def test_validate_mac_allocation_raises_on_out_of_bounds_utilization():
    df = _valid_row(utilization_pct=[150.0])
    with pytest.raises(SchemaValidationError, match="utilization_pct out of bounds"):
        validate_mac_allocation(df)


def test_validate_mac_allocation_raises_when_used_exceeds_allocated():
    df = _valid_row(used_capacity=[20.0], allocated_capacity=[10.0])
    with pytest.raises(SchemaValidationError, match="cannot exceed"):
        validate_mac_allocation(df)


def test_validate_mac_allocation_raises_on_unknown_resource_type():
    df = _valid_row(resource_type=["gpu"])
    with pytest.raises(SchemaValidationError, match="Unknown resource_type"):
        validate_mac_allocation(df)
