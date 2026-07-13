"""
Exports the final, dashboard-ready datasets for Power BI Desktop.

Power BI Desktop reads flat files (CSV/Parquet) directly via "Get Data",
so this step just needs to write clean, well-typed, denormalized tables --
one per dashboard concern -- into src/powerbi/. Power BI's own refresh
schedule (or a manual "Refresh" click) picks up new files from there.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def export_for_powerbi(
    trends: dict[str, pd.DataFrame],
    recommendations_df: pd.DataFrame,
    output_dir: str = "src/powerbi",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 1. Utilization overview (compute + storage + network unioned into one long table)
    compute = trends["compute"].rename(
        columns={"cluster_id": "resource_id", "cluster_utilization_pct": "utilization_pct"}
    )
    compute["resource_type"] = "compute"

    storage = trends["storage"].rename(columns={"storage_id": "resource_id", "disk_utilization_pct": "utilization_pct"})
    storage["resource_type"] = "storage"

    network = trends["network"].rename(columns={"link_id": "resource_id"})
    network["utilization_pct"] = (network["throughput_mbps"] / network["bandwidth_mbps"] * 100).round(2)
    network["resource_type"] = "network"

    common_cols = ["resource_id", "resource_type", "date", "utilization_pct"]
    overview = pd.concat(
        [compute[common_cols], storage[common_cols], network[common_cols]],
        ignore_index=True,
    )
    overview.to_csv(f"{output_dir}/utilization_overview.csv", index=False)

    # 2. Recommendations + AI narratives (drives the "Recommendation summaries" panel)
    recommendations_df.to_csv(f"{output_dir}/recommendations.csv", index=False)

    # 3. Risk indicators summary (counts by risk level, for KPI cards)
    risk_summary = (
        recommendations_df.groupby(["resource_type", "risk_level"]).size().reset_index(name="count")
        if not recommendations_df.empty
        else pd.DataFrame(columns=["resource_type", "risk_level", "count"])
    )
    risk_summary.to_csv(f"{output_dir}/risk_summary.csv", index=False)

    logger.info("Power BI datasets exported to %s", output_dir)
