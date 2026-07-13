from src.data_generation.scenario_seeder import (
    seed_capacity_shortfall,
    seed_chronic_waste,
    seed_seasonal_spike,
)


def test_capacity_shortfall_breaches_critical_threshold():
    df = seed_capacity_shortfall()
    assert df["disk_utilization_pct"].max() >= 90


def test_chronic_waste_stays_below_underutilization_threshold():
    df = seed_chronic_waste()
    assert df["cpu_utilization_pct"].mean() < 30


def test_seasonal_spike_has_a_spike_window():
    df = seed_seasonal_spike()
    bandwidth = df["bandwidth_mbps"].iloc[0]
    throughput_pct = df["throughput_mbps"] / bandwidth * 100
    baseline = throughput_pct.iloc[:24].mean()
    assert throughput_pct.max() > baseline + 40


def test_seeded_scenarios_have_required_columns():
    assert {"timestamp", "storage_id", "disk_utilization_pct"}.issubset(seed_capacity_shortfall().columns)
    assert {"timestamp", "cluster_id", "cpu_utilization_pct"}.issubset(seed_chronic_waste().columns)
    assert {"timestamp", "link_id", "throughput_mbps"}.issubset(seed_seasonal_spike().columns)
