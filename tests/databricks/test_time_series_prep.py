import importlib

import pandas as pd

ts_prep = importlib.import_module("src.databricks.notebooks.04_time_series_prep")


def test_prepare_series_splits_by_resource_id():
    df = pd.DataFrame(
        {
            "cluster_id": ["c1", "c1", "c2", "c2"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]),
            "cluster_utilization_pct": [40.0, 45.0, 60.0, 62.0],
        }
    )
    result = ts_prep.prepare_series(df, "cluster_id", "date", "cluster_utilization_pct")

    assert set(result.keys()) == {"c1", "c2"}
    assert list(result["c1"].columns) == ["ds", "y", "t"]
    assert result["c1"]["t"].tolist() == [0, 1]
    assert result["c1"]["y"].tolist() == [40.0, 45.0]


def test_prepare_all_reads_and_prepares_all_three_resource_types(tmp_path):
    dates = pd.date_range("2025-01-01", periods=3, freq="D")

    compute = pd.DataFrame({"cluster_id": ["c1"] * 3, "date": dates, "cluster_utilization_pct": [40.0, 42.0, 44.0]})
    storage = pd.DataFrame({"storage_id": ["s1"] * 3, "date": dates, "disk_utilization_pct": [60.0, 62.0, 64.0]})
    network = pd.DataFrame({"link_id": ["n1"] * 3, "date": dates, "throughput_mbps": [300.0, 310.0, 320.0]})

    compute.to_csv(tmp_path / "compute_trends.csv", index=False)
    storage.to_csv(tmp_path / "storage_trends.csv", index=False)
    network.to_csv(tmp_path / "network_trends.csv", index=False)

    result = ts_prep.prepare_all(processed_dir=str(tmp_path))

    assert set(result.keys()) == {"compute", "storage", "network"}
    assert "c1" in result["compute"]
    assert "s1" in result["storage"]
    assert "n1" in result["network"]
    assert result["storage"]["s1"]["y"].tolist() == [60.0, 62.0, 64.0]
