from src.data_generation.generate_storage_metrics import generate_storage_metrics
from src.data_generation.schema import STORAGE_COLUMNS


def test_generate_storage_metrics_shape():
    df = generate_storage_metrics(start_date="2025-01-01", num_days=20, num_systems=4, seed=1)
    assert list(df.columns) == STORAGE_COLUMNS
    assert len(df) == 20 * 4


def test_generate_storage_metrics_used_never_exceeds_total():
    df = generate_storage_metrics(start_date="2025-01-01", num_days=60, num_systems=4, seed=1)
    assert (df["disk_used_gb"] <= df["disk_total_gb"]).all()


def test_generate_storage_metrics_utilization_bounds():
    df = generate_storage_metrics(start_date="2025-01-01", num_days=60, num_systems=4, seed=1)
    assert df["disk_utilization_pct"].between(0, 100).all()


def test_generate_storage_metrics_has_growth_trend():
    df = generate_storage_metrics(start_date="2025-01-01", num_days=100, num_systems=4, seed=1)
    first_system = df["storage_id"].iloc[0]
    series = df[df["storage_id"] == first_system].sort_values("timestamp")
    assert series["disk_utilization_pct"].iloc[-1] > series["disk_utilization_pct"].iloc[0]
