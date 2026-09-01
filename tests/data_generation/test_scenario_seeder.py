from src.data_generation.scenario_seeder import (
    seed_capacity_shortfall,
    seed_chronic_waste,
    seed_seasonal_spike,
)


def test_capacity_shortfall_breaches_critical_threshold():
    df = seed_capacity_shortfall()
    assert df["utilization_pct"].max() >= 90
    assert (df["resource_type"] == "disk").all()


def test_chronic_waste_stays_below_underutilization_threshold():
    df = seed_chronic_waste()
    assert df["utilization_pct"].mean() < 30
    assert set(df["resource_type"].unique()) == {"cpu", "ram", "disk"}


def test_seasonal_spike_has_a_spike_window():
    df = seed_seasonal_spike()
    baseline = df["utilization_pct"].iloc[:24].mean()
    assert df["utilization_pct"].max() > baseline + 40
    assert (df["resource_type"] == "cpu").all()


def test_seeded_scenarios_have_required_columns():
    required = {"timestamp", "mac_id", "project_name", "resource_type", "utilization_pct"}
    assert required.issubset(seed_capacity_shortfall().columns)
    assert required.issubset(seed_chronic_waste().columns)
    assert required.issubset(seed_seasonal_spike().columns)
