"""
Injects the three seeded validation scenarios required by the problem statement:

1. capacity_shortfall  -> a storage system trending toward exhaustion within the forecast horizon
2. chronic_waste        -> a compute cluster running at ~20% utilization for a long stretch
3. seasonal_spike       -> a network link with a sharp, temporary demand spike

These scenarios are written out separately so the recommendation engine's
"Seeded Scenario Validation" step can be tested deterministically, independent
of the full randomly generated dataset.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.pipeline.io_utils import save_and_log

logger = logging.getLogger(__name__)


def seed_capacity_shortfall(start_date: str = "2025-01-01", num_days: int = 90) -> pd.DataFrame:
    """A storage system growing fast enough to breach 90% within the 12-week horizon."""
    timestamps = pd.date_range(start=start_date, periods=num_days, freq="D")
    total_gb = 1000
    used_pct = 55 + np.linspace(0, 40, num_days)  # 55% -> 95% over the window
    used_gb = used_pct / 100 * total_gb

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "storage_id": "storage-shortfall-scenario",
            "disk_used_gb": used_gb.round(2),
            "disk_total_gb": total_gb,
            "disk_utilization_pct": used_pct.round(2),
            "io_utilization_pct": np.clip(used_pct * 0.4 + 20, 0, 100).round(2),
        }
    )


def seed_chronic_waste(start_date: str = "2025-01-01", num_days: int = 90) -> pd.DataFrame:
    """A compute cluster flatlined at ~20% utilization -> should trigger downsize/decommission."""
    timestamps = pd.date_range(start=start_date, periods=num_days * 24, freq="h")
    rng = np.random.default_rng(7)
    cpu = np.clip(20 + rng.normal(0, 2, len(timestamps)), 1, 100)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "cluster_id": "cluster-waste-scenario",
            "cpu_utilization_pct": cpu.round(2),
            "memory_utilization_pct": (cpu * 0.9).round(2),
            "cluster_utilization_pct": cpu.round(2),
        }
    )


def seed_seasonal_spike(start_date: str = "2025-01-01", num_days: int = 60) -> pd.DataFrame:
    """A network link idle most of the time with a sharp 5-day demand spike."""
    timestamps = pd.date_range(start=start_date, periods=num_days * 24, freq="h")
    n = len(timestamps)
    throughput_pct = np.full(n, 25.0)

    spike_start, spike_end = int(n * 0.5), int(n * 0.5) + 24 * 5
    throughput_pct[spike_start:spike_end] = 92.0

    bandwidth = 2000
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "link_id": "network-spike-scenario",
            "bandwidth_mbps": bandwidth,
            "throughput_mbps": (throughput_pct / 100 * bandwidth).round(2),
            "latency_ms": (20 + throughput_pct * 1.2).round(2),
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


if __name__ == "__main__":
    generated = generate_all_scenarios()
    for name, df in generated.items():
        print(f"Seeded scenario '{name}': {len(df)} rows -> data/seeded_scenarios/{name}.csv")
