import importlib

import pandas as pd

ts_prep = importlib.import_module("src.databricks.notebooks.04_time_series_prep")


def test_prepare_series_splits_by_series_id():
    df = pd.DataFrame(
        {
            "series_id": ["mac-01__cpu", "mac-01__cpu", "mac-02__ram", "mac-02__ram"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]),
            "utilization_pct": [40.0, 45.0, 60.0, 62.0],
        }
    )
    result = ts_prep.prepare_series(df, "series_id", "date", "utilization_pct")

    assert set(result.keys()) == {"mac-01__cpu", "mac-02__ram"}
    assert list(result["mac-01__cpu"].columns) == ["ds", "y", "t"]
    assert result["mac-01__cpu"]["t"].tolist() == [0, 1]
    assert result["mac-01__cpu"]["y"].tolist() == [40.0, 45.0]


def test_prepare_all_reads_and_prepares_all_series(tmp_path):
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "series_id": ["mac-01__cpu"] * 3 + ["mac-02__ram"] * 3,
            "date": list(dates) * 2,
            "utilization_pct": [40.0, 42.0, 44.0, 60.0, 62.0, 64.0],
        }
    )
    df.to_csv(tmp_path / "mac_allocation_trends.csv", index=False)

    result = ts_prep.prepare_all(processed_dir=str(tmp_path))

    assert set(result.keys()) == {"mac-01__cpu", "mac-02__ram"}
    assert result["mac-02__ram"]["y"].tolist() == [60.0, 62.0, 64.0]
