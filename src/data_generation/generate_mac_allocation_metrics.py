"""
Generates synthetic resource-allocation utilization data for Psiog's 8 Macs,
tracked across 3 resource types (cpu, ram, disk).

Each Mac is assigned a realistic Apple Silicon hardware profile (CPU cores,
RAM, disk) and a project it's primarily running. Utilization patterns vary
deliberately across the fleet so downstream forecasting/recommendation logic
has real signal to work with: some Macs run consistently overloaded, some
sit chronically under-used (over-allocated relative to actual need), and
most fall in a healthy band -- exactly the imbalance the equalization
algorithm is meant to detect and correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generation.schema import CAPACITY_UNITS, MAC_ALLOCATION_COLUMNS, RESOURCE_TYPES, validate_mac_allocation

# Realistic Apple Silicon build-farm / dev-Mac hardware tiers
HARDWARE_TIERS = [
    {"cpu_cores": 8, "ram_gb": 16, "disk_gb": 512},
    {"cpu_cores": 10, "ram_gb": 32, "disk_gb": 1024},
    {"cpu_cores": 12, "ram_gb": 64, "disk_gb": 2048},
]


def _daily_seasonality(hours: np.ndarray) -> np.ndarray:
    """Business-hours-heavy sinusoidal pattern, peaking mid-day."""
    return 15 * np.sin((hours - 6) / 24 * 2 * np.pi) + 15


def _generate_resource_series(
    rng: np.random.Generator,
    timestamps: pd.DatetimeIndex,
    hours: np.ndarray,
    base_load: float,
    seasonal_amplitude: float,
    drift_max: float,
) -> np.ndarray:
    seasonal = _daily_seasonality(hours) * seasonal_amplitude
    drift = np.linspace(0, drift_max, len(timestamps))
    noise = rng.normal(0, 4, size=len(timestamps))
    return np.clip(base_load + seasonal + drift + noise, 1, 100)


def generate_mac_allocation_metrics(
    start_date: str,
    num_days: int,
    num_macs: int,
    project_pool: list[str],
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, periods=num_days * 24, freq=freq)
    hours = timestamps.hour.values

    frames = []
    for i in range(num_macs):
        mac_id = f"mac-{i+1:02d}"
        project_name = project_pool[i % len(project_pool)]
        hw = HARDWARE_TIERS[i % len(HARDWARE_TIERS)]

        # Deliberately vary the fleet: every 4th Mac runs chronically overloaded,
        # every 3rd (but not overloaded) runs chronically under-used/over-allocated,
        # the rest sit in a healthy band. This is what gives the equalization
        # algorithm real imbalance to detect.
        if i % 4 == 0:
            profile = "overloaded"
            base_load, seasonal_amp, drift_max = 68, 1.3, 15
        elif i % 3 == 0:
            profile = "underloaded"
            base_load, seasonal_amp, drift_max = 14, 0.3, 0
        else:
            profile = "healthy"
            base_load, seasonal_amp, drift_max = rng.uniform(42, 60), 1.0, rng.uniform(3, 10)

        for resource_type in RESOURCE_TYPES:
            # Slight per-resource variation so cpu/ram/disk aren't identical curves
            resource_jitter = rng.uniform(-6, 6)
            util_pct = _generate_resource_series(
                rng,
                timestamps,
                hours,
                base_load=base_load + resource_jitter,
                seasonal_amplitude=seasonal_amp,
                drift_max=drift_max,
            )

            allocated_capacity = (
                hw["cpu_cores"]
                if resource_type == "cpu"
                else (hw["ram_gb"] if resource_type == "ram" else hw["disk_gb"])
            )
            used_capacity = np.clip(util_pct / 100 * allocated_capacity, 0, allocated_capacity)

            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "mac_id": mac_id,
                        "project_name": project_name,
                        "resource_type": resource_type,
                        "allocated_capacity": allocated_capacity,
                        "used_capacity": used_capacity.round(2),
                        "capacity_unit": CAPACITY_UNITS[resource_type],
                        "utilization_pct": util_pct.round(2),
                    }
                )
            )
        _ = profile  # profile is implicit in the generated values; kept for readability

    df = pd.concat(frames, ignore_index=True)[MAC_ALLOCATION_COLUMNS]
    validate_mac_allocation(df)
    return df


if __name__ == "__main__":  # pragma: no cover
    df = generate_mac_allocation_metrics(
        start_date="2025-01-01",
        num_days=365,
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
    )
    df.to_csv("data/raw/mac_allocation_metrics.csv", index=False)
    print(f"Generated {len(df)} Mac allocation rows -> data/raw/mac_allocation_metrics.csv")
