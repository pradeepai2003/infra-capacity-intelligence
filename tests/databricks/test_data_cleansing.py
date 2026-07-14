import importlib

import numpy as np
import pandas as pd
import pytest

cleansing = importlib.import_module("src.databricks.notebooks.02_data_cleansing")


@pytest.fixture
def dirty_compute_df():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"]),
            "cluster_id": ["c1", "c1", "c1", "c1"],
            "cpu_utilization_pct": [50.0, 50.0, 120.0, np.nan],  # duplicate + out-of-range + missing
            "memory_utilization_pct": [40.0, 40.0, 60.0, 70.0],
            "cluster_utilization_pct": [45.0, 45.0, 90.0, 75.0],
        }
    )


@pytest.fixture
def dirty_storage_df():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "storage_id": ["s1", "s1"],
            "disk_used_gb": [600.0, 1100.0],  # second row exceeds total -> should be dropped
            "disk_total_gb": [1000.0, 1000.0],
            "disk_utilization_pct": [60.0, 110.0],
            "io_utilization_pct": [50.0, 150.0],  # out of range -> should be clipped
        }
    )


@pytest.fixture
def dirty_network_df():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "link_id": ["n1", "n1"],
            "bandwidth_mbps": [1000, 1000],
            "throughput_mbps": [500.0, -10.0],  # negative throughput -> clipped to 0
            "latency_ms": [20.0, -5.0],  # negative latency row -> should be dropped
        }
    )


def test_clean_compute_removes_duplicates_and_missing(dirty_compute_df):
    cleaned = cleansing.clean_compute(dirty_compute_df)
    assert len(cleaned) == 2  # duplicate removed, NaN row removed
    assert cleaned["cpu_utilization_pct"].between(0, 100).all()


def test_clean_storage_drops_invalid_rows(dirty_storage_df):
    cleaned = cleansing.clean_storage(dirty_storage_df)
    assert (cleaned["disk_used_gb"] <= cleaned["disk_total_gb"]).all()
    assert len(cleaned) == 1


def test_clean_network_drops_negative_latency(dirty_network_df):
    cleaned = cleansing.clean_network(dirty_network_df)
    assert (cleaned["latency_ms"] >= 0).all()
    assert (cleaned["throughput_mbps"] >= 0).all()
    assert len(cleaned) == 1


def test_clean_all_reads_interim_and_writes_processed(tmp_path):
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    interim_dir.mkdir()
    processed_dir.mkdir()

    compute_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "cluster_id": ["c1", "c1"],
            "cpu_utilization_pct": [40.0, 45.0],
            "memory_utilization_pct": [35.0, 42.0],
            "cluster_utilization_pct": [38.0, 44.0],
        }
    )
    storage_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "storage_id": ["s1", "s1"],
            "disk_used_gb": [500.0, 520.0],
            "disk_total_gb": [1000.0, 1000.0],
            "disk_utilization_pct": [50.0, 52.0],
            "io_utilization_pct": [30.0, 32.0],
        }
    )
    network_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "link_id": ["n1", "n1"],
            "bandwidth_mbps": [1000, 1000],
            "throughput_mbps": [300.0, 310.0],
            "latency_ms": [20.0, 22.0],
        }
    )

    compute_df.to_csv(interim_dir / "compute_ingested.csv", index=False)
    storage_df.to_csv(interim_dir / "storage_ingested.csv", index=False)
    network_df.to_csv(interim_dir / "network_ingested.csv", index=False)

    result = cleansing.clean_all(interim_dir=str(interim_dir), processed_dir=str(processed_dir))

    assert set(result.keys()) == {"compute", "storage", "network"}
    assert (processed_dir / "compute_cleaned.csv").exists()
    assert (processed_dir / "storage_cleaned.csv").exists()
    assert (processed_dir / "network_cleaned.csv").exists()
    assert len(result["compute"]) == 2
