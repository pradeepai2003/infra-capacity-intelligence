"""
Generates synthetic storage utilization time-series data with steady
month-over-month growth trends, so the forecasting engine has a realistic
"storage exhaustion" curve to predict against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generation.schema import STORAGE_COLUMNS, validate_storage


def generate_storage_metrics(
    start_date: str,
    num_days: int,
    num_systems: int,
    freq: str = "D",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, periods=num_days, freq=freq)

    frames = []
    for i in range(num_systems):
        storage_id = f"storage-{i+1:02d}"
        total_gb = rng.choice([500, 1000, 2000, 5000])

        # One system is on an aggressive growth curve -> will breach threshold (shortfall scenario)
        growth_rate = rng.uniform(0.15, 0.35) if i == 0 else rng.uniform(0.02, 0.08)
        starting_used_pct = rng.uniform(30, 50)

        days = np.arange(num_days)
        used_pct = starting_used_pct + growth_rate * days + rng.normal(0, 0.5, num_days).cumsum() * 0.01
        used_pct = np.clip(used_pct, 1, 99.5)
        used_gb = used_pct / 100 * total_gb

        io_util = np.clip(30 + used_pct * 0.3 + rng.normal(0, 5, num_days), 1, 100)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "storage_id": storage_id,
                    "disk_used_gb": used_gb.round(2),
                    "disk_total_gb": total_gb,
                    "disk_utilization_pct": used_pct.round(2),
                    "io_utilization_pct": io_util.round(2),
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)[STORAGE_COLUMNS]
    validate_storage(df)
    return df


if __name__ == "__main__":
    df = generate_storage_metrics(start_date="2025-01-01", num_days=365, num_systems=4)
    df.to_csv("data/raw/storage_metrics.csv", index=False)
    print(f"Generated {len(df)} storage metric rows -> data/raw/storage_metrics.csv")
