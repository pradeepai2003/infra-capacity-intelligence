import pandas as pd

from src.data_generation.generate_mac_allocation_metrics import generate_mac_allocation_metrics
from src.data_generation.schema import MAC_ALLOCATION_COLUMNS, RESOURCE_TYPES

PROJECT_POOL = ["Project Atlas", "Project Nova", "Project Orion", "Project Zephyr"]


def test_generate_mac_allocation_metrics_shape():
    df = generate_mac_allocation_metrics(
        start_date="2025-01-01", num_days=10, num_macs=4, project_pool=PROJECT_POOL, seed=1
    )
    assert list(df.columns) == MAC_ALLOCATION_COLUMNS
    # 4 macs x 3 resource types x 10 days x 24 hours
    assert len(df) == 4 * 3 * 10 * 24


def test_generate_mac_allocation_metrics_value_bounds():
    df = generate_mac_allocation_metrics(
        start_date="2025-01-01", num_days=10, num_macs=4, project_pool=PROJECT_POOL, seed=1
    )
    assert df["utilization_pct"].between(0, 100).all()
    assert (df["used_capacity"] <= df["allocated_capacity"]).all()
    assert set(df["resource_type"].unique()) == set(RESOURCE_TYPES)


def test_generate_mac_allocation_metrics_deterministic_with_seed():
    df1 = generate_mac_allocation_metrics(
        start_date="2025-01-01", num_days=5, num_macs=4, project_pool=PROJECT_POOL, seed=42
    )
    df2 = generate_mac_allocation_metrics(
        start_date="2025-01-01", num_days=5, num_macs=4, project_pool=PROJECT_POOL, seed=42
    )
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_mac_allocation_metrics_covers_all_macs():
    df = generate_mac_allocation_metrics(
        start_date="2025-01-01", num_days=5, num_macs=8, project_pool=PROJECT_POOL, seed=1
    )
    expected_macs = {f"mac-{i+1:02d}" for i in range(8)}
    assert set(df["mac_id"].unique()) == expected_macs


def test_generate_mac_allocation_metrics_has_overloaded_and_underloaded_macs():
    # With num_macs=8, the generator deliberately makes every 4th Mac overloaded
    # and every 3rd (non-overloaded) Mac chronically underloaded.
    df = generate_mac_allocation_metrics(
        start_date="2025-01-01",
        num_days=30,
        num_macs=8,
        project_pool=[
            "Project Atlas",
            "Project Nova",
            "Project Orion",
            "Project Zephyr",
            "Project Falcon",
            "Project Comet",
            "Project Vega",
            "Project Lumen",
        ],
        seed=1,
    )
    mac_means = df.groupby("mac_id")["utilization_pct"].mean()
    assert (mac_means > 65).any(), "Expected at least one consistently high-utilization Mac"
    assert (mac_means < 35).any(), "Expected at least one consistently low-utilization Mac"


def test_generate_mac_allocation_metrics_project_assignment():
    df = generate_mac_allocation_metrics(
        start_date="2025-01-01", num_days=2, num_macs=4, project_pool=PROJECT_POOL, seed=1
    )
    # Each Mac should have exactly one project assigned (consistent across its rows)
    projects_per_mac = df.groupby("mac_id")["project_name"].nunique()
    assert (projects_per_mac == 1).all()
