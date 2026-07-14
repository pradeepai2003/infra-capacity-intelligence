"""
Generates synthetic network utilization time-series data (bandwidth, throughput,
latency), including a seasonal demand spike window to validate the
"seasonal_spike" recommendation scenario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generation.schema import NETWORK_COLUMNS, validate_network


def generate_network_metrics(
    start_date: str,
    num_days: int,
    num_links: int,
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, periods=num_days * 24, freq=freq)
    n = len(timestamps)

    # Seasonal spike window: e.g. a two-week period ~70% through the series
    spike_start = int(n * 0.65)
    spike_end = spike_start + 24 * 14
    spike_mask = np.zeros(n, dtype=bool)
    spike_mask[spike_start:spike_end] = True

    frames = []
    for i in range(num_links):
        link_id = f"network-{i+1:02d}"
        bandwidth_capacity = rng.choice([1000, 2000, 5000])  # Mbps

        base_throughput_pct = rng.uniform(30, 50)
        noise = rng.normal(0, 5, n)
        spike_boost = np.where(spike_mask, rng.uniform(25, 45), 0)

        throughput_pct = np.clip(base_throughput_pct + noise + spike_boost, 1, 100)
        throughput = throughput_pct / 100 * bandwidth_capacity

        base_latency = rng.uniform(15, 40)
        latency = base_latency + (throughput_pct / 100) * rng.uniform(80, 150) + rng.normal(0, 3, n)
        latency = np.clip(latency, 1, None)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "link_id": link_id,
                    "bandwidth_mbps": bandwidth_capacity,
                    "throughput_mbps": throughput.round(2),
                    "latency_ms": latency.round(2),
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)[NETWORK_COLUMNS]
    validate_network(df)
    return df


if __name__ == "__main__":  # pragma: no cover
    df = generate_network_metrics(start_date="2025-01-01", num_days=365, num_links=3)
    df.to_csv("data/raw/network_metrics.csv", index=False)
    print(f"Generated {len(df)} network metric rows -> data/raw/network_metrics.csv")
