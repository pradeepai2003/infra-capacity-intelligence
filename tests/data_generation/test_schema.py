import pandas as pd
import pytest

from src.data_generation.schema import (
    SchemaValidationError,
    validate_columns,
    validate_compute,
    validate_network,
    validate_storage,
)


def test_validate_columns_raises_on_missing_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(SchemaValidationError, match="missing required columns"):
        validate_columns(df, ["a", "b", "c"], "test data")


def test_validate_columns_passes_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    validate_columns(df, ["a", "b"], "test data")  # should not raise


def test_validate_compute_raises_on_out_of_bounds_cpu():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01"]),
            "cluster_id": ["c1"],
            "cpu_utilization_pct": [150.0],  # out of bounds
            "memory_utilization_pct": [50.0],
            "cluster_utilization_pct": [50.0],
        }
    )
    with pytest.raises(SchemaValidationError, match="cpu_utilization_pct out of bounds"):
        validate_compute(df)


def test_validate_compute_raises_on_missing_columns():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01"]), "cluster_id": ["c1"]})
    with pytest.raises(SchemaValidationError, match="missing required columns"):
        validate_compute(df)


def test_validate_storage_raises_when_used_exceeds_total():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01"]),
            "storage_id": ["s1"],
            "disk_used_gb": [1200.0],
            "disk_total_gb": [1000.0],
            "disk_utilization_pct": [120.0],
            "io_utilization_pct": [50.0],
        }
    )
    with pytest.raises(SchemaValidationError, match="cannot exceed"):
        validate_storage(df)


def test_validate_network_raises_on_negative_latency():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01"]),
            "link_id": ["n1"],
            "bandwidth_mbps": [1000],
            "throughput_mbps": [500.0],
            "latency_ms": [-5.0],
        }
    )
    with pytest.raises(SchemaValidationError, match="cannot be negative"):
        validate_network(df)
