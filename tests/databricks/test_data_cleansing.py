import importlib

import numpy as np
import pandas as pd
import pytest

cleansing = importlib.import_module("src.databricks.notebooks.02_data_cleansing")


@pytest.fixture
def dirty_mac_df():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"]),
            "mac_id": ["mac-01", "mac-01", "mac-01", "mac-01"],
            "project_name": ["Project Atlas"] * 4,
            "resource_type": ["cpu", "cpu", "cpu", "cpu"],
            "allocated_capacity": [10.0, 10.0, 10.0, 10.0],
            "used_capacity": [5.0, 5.0, 15.0, np.nan],  # duplicate + exceeds allocated + missing
            "capacity_unit": ["cores"] * 4,
            "utilization_pct": [50.0, 50.0, 150.0, np.nan],
        }
    )


def test_clean_mac_allocation_removes_duplicates_and_missing(dirty_mac_df):
    cleaned = cleansing.clean_mac_allocation(dirty_mac_df)
    # duplicate row removed, exceeds-allocated row dropped, NaN row dropped -> 1 remains
    assert len(cleaned) == 1
    assert cleaned["utilization_pct"].between(0, 100).all()


def test_clean_mac_allocation_recomputes_utilization_pct():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01"]),
            "mac_id": ["mac-01"],
            "project_name": ["Project Atlas"],
            "resource_type": ["ram"],
            "allocated_capacity": [64.0],
            "used_capacity": [32.0],
            "capacity_unit": ["GB"],
            "utilization_pct": [999.0],  # deliberately wrong -- should be recomputed
        }
    )
    cleaned = cleansing.clean_mac_allocation(df)
    assert cleaned["utilization_pct"].iloc[0] == 50.0


def test_clean_all_reads_interim_and_writes_processed(tmp_path):
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    interim_dir.mkdir()
    processed_dir.mkdir()

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "mac_id": ["mac-01", "mac-01"],
            "project_name": ["Project Atlas", "Project Atlas"],
            "resource_type": ["cpu", "cpu"],
            "allocated_capacity": [10.0, 10.0],
            "used_capacity": [5.0, 6.0],
            "capacity_unit": ["cores", "cores"],
            "utilization_pct": [50.0, 60.0],
        }
    )
    df.to_csv(interim_dir / "mac_allocation_ingested.csv", index=False)

    result = cleansing.clean_all(interim_dir=str(interim_dir), processed_dir=str(processed_dir))

    assert "mac_allocation" in result
    assert (processed_dir / "mac_allocation_cleaned.csv").exists()
    assert len(result["mac_allocation"]) == 2
