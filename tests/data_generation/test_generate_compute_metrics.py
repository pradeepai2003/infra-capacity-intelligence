import pandas as pd

from src.data_generation.generate_compute_metrics import generate_compute_metrics
from src.data_generation.schema import COMPUTE_COLUMNS


def test_generate_compute_metrics_shape():
    df = generate_compute_metrics(start_date="2025-01-01", num_days=10, num_clusters=3, seed=1)
    assert list(df.columns) == COMPUTE_COLUMNS
    assert len(df) == 10 * 24 * 3  # hourly data x days x clusters


def test_generate_compute_metrics_value_bounds():
    df = generate_compute_metrics(start_date="2025-01-01", num_days=10, num_clusters=2, seed=1)
    assert df["cpu_utilization_pct"].between(0, 100).all()
    assert df["memory_utilization_pct"].between(0, 100).all()
    assert df["cluster_utilization_pct"].between(0, 100).all()


def test_generate_compute_metrics_deterministic_with_seed():
    df1 = generate_compute_metrics(start_date="2025-01-01", num_days=5, num_clusters=2, seed=42)
    df2 = generate_compute_metrics(start_date="2025-01-01", num_days=5, num_clusters=2, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_compute_metrics_has_underutilized_cluster():
    df = generate_compute_metrics(start_date="2025-01-01", num_days=30, num_clusters=6, seed=1)
    cluster_means = df.groupby("cluster_id")["cpu_utilization_pct"].mean()
    assert (cluster_means < 40).any(), "Expected at least one chronically underutilized cluster"
