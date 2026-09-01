"""
Injects the three seeded validation scenarios required by the approved
proposal, reimplemented against the Mac-allocation model:

1. capacity_shortfall  -> a Mac's disk allocation trending toward exhaustion
2. chronic_waste        -> a Mac chronically under-used across all resources
                           (over-allocated relative to actual need)
3. seasonal_spike       -> a Mac's CPU spiking during a release/crunch window
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_generation.schema import CAPACITY_UNITS
from src.pipeline.io_utils import save_and_log

logger = logging.getLogger(__name__)


def seed_capacity_shortfall(start_date: str = "2025-01-01", num_days: int = 90) -> pd.DataFrame:
    """A Mac's disk usage growing fast enough to breach 90% within the 12-week horizon."""
    timestamps = pd.date_range(start=start_date, periods=num_days, freq="D")
    allocated_capacity = 1024  # GB
    used_pct = 55 + np.linspace(0, 40, num_days)  # 55% -> 95% over the window
    used_capacity = used_pct / 100 * allocated_capacity

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "mac_id": "mac-shortfall-scenario",
            "project_name": "Project Nova",
            "resource_type": "disk",
            "allocated_capacity": allocated_capacity,
            "used_capacity": used_capacity.round(2),
            "capacity_unit": CAPACITY_UNITS["disk"],
            "utilization_pct": used_pct.round(2),
        }
    )


def seed_chronic_waste(start_date: str = "2025-01-01", num_days: int = 90) -> pd.DataFrame:
    """A Mac flatlined at ~15% utilization across CPU/RAM/disk -- over-allocated,
    not delivering value for the resources it's been given.
    """
    timestamps = pd.date_range(start=start_date, periods=num_days * 24, freq="h")
    rng = np.random.default_rng(7)
    frames = []
    hw = {"cpu": 12, "ram": 64, "disk": 2048}
    for resource_type in ("cpu", "ram", "disk"):
        util = np.clip(15 + rng.normal(0, 2, len(timestamps)), 1, 100)
        allocated = hw[resource_type]
        used = np.clip(util / 100 * allocated, 0, allocated)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "mac_id": "mac-waste-scenario",
                    "project_name": "Project Vega",
                    "resource_type": resource_type,
                    "allocated_capacity": allocated,
                    "used_capacity": used.round(2),
                    "capacity_unit": CAPACITY_UNITS[resource_type],
                    "utilization_pct": util.round(2),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def seed_seasonal_spike(start_date: str = "2025-01-01", num_days: int = 60) -> pd.DataFrame:
    """A Mac's CPU idle most of the time with a sharp 5-day spike (e.g. a release week)."""
    timestamps = pd.date_range(start=start_date, periods=num_days * 24, freq="h")
    n = len(timestamps)
    util = np.full(n, 25.0)

    spike_start, spike_end = int(n * 0.5), int(n * 0.5) + 24 * 5
    util[spike_start:spike_end] = 92.0

    allocated_capacity = 10  # cores
    used_capacity = util / 100 * allocated_capacity

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "mac_id": "mac-spike-scenario",
            "project_name": "Project Falcon",
            "resource_type": "cpu",
            "allocated_capacity": allocated_capacity,
            "used_capacity": used_capacity.round(2),
            "capacity_unit": CAPACITY_UNITS["cpu"],
            "utilization_pct": util.round(2),
        }
    )


SCENARIO_GENERATORS = {
    "capacity_shortfall": seed_capacity_shortfall,
    "chronic_waste": seed_chronic_waste,
    "seasonal_spike": seed_seasonal_spike,
}


def generate_all_scenarios(output_dir: str = "data/seeded_scenarios") -> dict[str, pd.DataFrame]:
    results = {}
    for name, generator_fn in SCENARIO_GENERATORS.items():
        df = generator_fn()
        save_and_log(df, f"{output_dir}/{name}.csv", f"Seeded scenario '{name}'")
        results[name] = df
    return results


if __name__ == "__main__":  # pragma: no cover
    generated = generate_all_scenarios()
    for name, df in generated.items():
        print(f"Seeded scenario '{name}': {len(df)} rows -> data/seeded_scenarios/{name}.csv")
