"""
Exports the final, dashboard-ready datasets for Power BI Desktop.

Four CSVs are produced:
  - mac_utilization_overview.csv  -> per-Mac, per-resource-type utilization over time
                                      (drives the 8 individual per-Mac visual pages)
  - recommendations.csv            -> per-(mac, resource) recommendations + AI narrative
  - equalization_summary.csv       -> fleet-wide rebalancing recommendations + AI narrative
                                      (drives the final "Recommendation Dashboard" page)
  - risk_summary.csv               -> recommendation counts by risk level, for KPI cards
"""

from __future__ import annotations

import logging

import pandas as pd

from src.pipeline.io_utils import save_and_log

logger = logging.getLogger(__name__)


def export_for_powerbi(
    mac_trends: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    equalization_df: pd.DataFrame,
    output_dir: str = "src/powerbi",
) -> None:
    # 1. Per-Mac utilization overview (drives the 8 individual per-Mac pages)
    overview = mac_trends[["mac_id", "project_name", "resource_type", "date", "utilization_pct"]].copy()
    save_and_log(overview, f"{output_dir}/mac_utilization_overview.csv", "Power BI per-Mac utilization overview")

    # 2. Per-(mac, resource) recommendations + AI narratives
    save_and_log(recommendations_df, f"{output_dir}/recommendations.csv", "Power BI recommendations table")

    # 3. Fleet-wide equalization recommendations -- the final Recommendation Dashboard input
    save_and_log(equalization_df, f"{output_dir}/equalization_summary.csv", "Power BI equalization summary")

    # 4. Risk indicators summary (counts by risk level, for KPI cards)
    risk_summary = (
        recommendations_df.groupby(["resource_type", "risk_level"]).size().reset_index(name="count")
        if not recommendations_df.empty
        else pd.DataFrame(columns=["resource_type", "risk_level", "count"])
    )
    save_and_log(risk_summary, f"{output_dir}/risk_summary.csv", "Power BI risk summary")
