"""
Generates synthetic compute (CPU / memory / cluster) utilization time-series data.

Simulates realistic patterns: daily seasonality, slow upward drift, weekday/weekend
effects, and random noise -- plus a couple of deliberately under-utilized clusters
so the recommendation engine has something to flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generation.schema import COMPUTE_COLUMNS, validate_compute


def _daily_seasonality(hours: np.ndarray) -> np.ndarray:
    """Business-hours-heavy sinusoidal pattern, peaking mid-day."""
    return 15 * np.sin((hours - 6) / 24 * 2 * np.pi) + 15


def generate_compute_metrics(
    start_date: str,
    num_days: int,
    num_clusters: int,
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, periods=num_days * 24, freq=freq)
    hours = timestamps.hour.values

    frames = []
    for i in range(num_clusters):
        cluster_id = f"cluster-{i+1:02d}"

        # Every 3rd cluster is deliberately chronically under-utilized (waste scenario):
        # low base load, muted daily swing, and no growth drift, so its mean utilization
        # stays clearly and consistently below normal clusters.
        is_waste_cluster = i % 3 == 0
        base_load = 12 if is_waste_cluster else rng.uniform(45, 65)
        seasonal_amplitude = 0.3 if is_waste_cluster else 1.0

        # Slow upward drift over the period (simulates organic growth) - waste clusters don't grow
        drift = np.zeros(len(timestamps)) if is_waste_cluster else np.linspace(0, rng.uniform(5, 20), len(timestamps))

        seasonal = _daily_seasonality(hours) * seasonal_amplitude
        noise = rng.normal(0, 4, size=len(timestamps))

        cpu = np.clip(base_load + seasonal + drift + noise, 1, 100)
        memory = np.clip(cpu * rng.uniform(0.85, 1.05) + rng.normal(0, 3, len(timestamps)), 1, 100)
        cluster_util = np.clip((cpu + memory) / 2 + rng.normal(0, 2, len(timestamps)), 1, 100)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "cluster_id": cluster_id,
                    "cpu_utilization_pct": cpu.round(2),
                    "memory_utilization_pct": memory.round(2),
                    "cluster_utilization_pct": cluster_util.round(2),
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)[COMPUTE_COLUMNS]
    validate_compute(df)
    return df


if __name__ == "__main__":
    df = generate_compute_metrics(start_date="2025-01-01", num_days=365, num_clusters=5)
    df.to_csv("data/raw/compute_metrics.csv", index=False)
    print(f"Generated {len(df)} compute metric rows -> data/raw/compute_metrics.csv")
