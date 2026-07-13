from src.data_generation.generate_network_metrics import generate_network_metrics
from src.data_generation.schema import NETWORK_COLUMNS


def test_generate_network_metrics_shape():
    df = generate_network_metrics(start_date="2025-01-01", num_days=5, num_links=3, seed=1)
    assert list(df.columns) == NETWORK_COLUMNS
    assert len(df) == 5 * 24 * 3


def test_generate_network_metrics_latency_non_negative():
    df = generate_network_metrics(start_date="2025-01-01", num_days=10, num_links=2, seed=1)
    assert (df["latency_ms"] >= 0).all()


def test_generate_network_metrics_has_seasonal_spike():
    df = generate_network_metrics(start_date="2025-01-01", num_days=60, num_links=2, seed=1)
    first_link = df["link_id"].iloc[0]
    series = df[df["link_id"] == first_link].sort_values("timestamp").reset_index(drop=True)
    throughput_pct = series["throughput_mbps"] / series["bandwidth_mbps"] * 100
    assert throughput_pct.max() > throughput_pct.iloc[:100].mean() + 15, "Expected a distinguishable spike window"
