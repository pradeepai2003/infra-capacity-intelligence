"""
Rule-based recommendation engine for the Mac fleet, covering two distinct
concerns:

1. Per-(mac, resource type) evaluation -- is this Mac overloaded or
   over-allocated (chronically under-used) on this specific resource?
2. Fleet-wide equalization -- across all 8 Macs, which ones are carrying too
   much load and which are sitting idle, and how should that be rebalanced?

Terminology note: underused resources are deliberately described as
"over-allocated" rather than "wasted" throughout -- the framing is that the
Mac was given more capacity than its project needs, not that the resource
itself is worthless. The goal in all cases is saving time, cost, and
resources: right-sizing allocations and redistributing load rather than
provisioning new hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class RecommendationType(str, Enum):
    INCREASE_ALLOCATION = "increase_allocation"
    REDUCE_ALLOCATION = "reduce_allocation"
    REBALANCE_BETWEEN_MACS = "rebalance_between_macs"
    NO_ACTION = "no_action"


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Recommendation:
    mac_id: str
    resource_type: str
    project_name: str
    recommendation_type: RecommendationType
    risk_level: RiskLevel
    current_value: float
    forecasted_value: float
    forecast_horizon_weeks: int
    details: dict = field(default_factory=dict)


@dataclass
class EqualizationRecommendation:
    resource_type: str
    overloaded_mac_id: str
    overloaded_project: str
    overloaded_utilization_pct: float
    underloaded_mac_id: str
    underloaded_project: str
    underloaded_utilization_pct: float
    fleet_average_utilization_pct: float
    suggested_action: str
    risk_level: RiskLevel


def evaluate_mac_resource(
    mac_id: str,
    resource_type: str,
    project_name: str,
    utilization_history_pct: pd.Series,
    forecast_4wk: pd.DataFrame = None,
    forecast_12wk: pd.DataFrame = None,
    overloaded_threshold: float = 85,
    underloaded_threshold: float = 30,
    chronic_days_threshold: int = 21,
) -> Recommendation:
    """Evaluate a single (mac, resource_type) series and return one recommendation."""
    current_value = float(utilization_history_pct.iloc[-1])
    days_below_threshold = int((utilization_history_pct < underloaded_threshold).sum())

    peak_forecast = None
    horizon_weeks = 0
    for weeks, forecast in [(4, forecast_4wk), (12, forecast_12wk)]:
        if forecast is None or forecast.empty:
            continue
        peak = float(forecast["yhat"].max())
        if peak_forecast is None or peak > peak_forecast:
            peak_forecast = peak
            horizon_weeks = weeks

    forecasted_value = peak_forecast if peak_forecast is not None else current_value

    if forecasted_value >= overloaded_threshold or current_value >= overloaded_threshold:
        return Recommendation(
            mac_id=mac_id,
            resource_type=resource_type,
            project_name=project_name,
            recommendation_type=RecommendationType.INCREASE_ALLOCATION,
            risk_level=RiskLevel.CRITICAL,
            current_value=current_value,
            forecasted_value=forecasted_value,
            forecast_horizon_weeks=horizon_weeks,
            details={"threshold": overloaded_threshold},
        )

    if days_below_threshold >= chronic_days_threshold:
        return Recommendation(
            mac_id=mac_id,
            resource_type=resource_type,
            project_name=project_name,
            recommendation_type=RecommendationType.REDUCE_ALLOCATION,
            risk_level=RiskLevel.WARNING,
            current_value=current_value,
            forecasted_value=forecasted_value,
            forecast_horizon_weeks=0,
            details={"days_below_threshold": days_below_threshold, "threshold": underloaded_threshold},
        )

    return Recommendation(
        mac_id=mac_id,
        resource_type=resource_type,
        project_name=project_name,
        recommendation_type=RecommendationType.NO_ACTION,
        risk_level=RiskLevel.INFO,
        current_value=current_value,
        forecasted_value=forecasted_value,
        forecast_horizon_weeks=0,
    )


def evaluate_equalization(
    fleet_snapshot: pd.DataFrame,
    resource_type: str,
    deviation_threshold_pct: float = 20,
) -> list[EqualizationRecommendation]:
    """
    Compare current utilization across all Macs for one resource type, and
    recommend rebalancing between the most overloaded and most underloaded
    Macs when the spread is wide enough to matter.

    Args:
        fleet_snapshot: DataFrame with columns [mac_id, project_name, utilization_pct]
            -- one row per Mac, for a single resource_type, at the latest timestamp
        resource_type: "cpu" | "ram" | "disk" (for labeling only)
        deviation_threshold_pct: how far above/below the fleet average a Mac
            must be to count as overloaded/underloaded

    Returns:
        A list of pairwise rebalancing recommendations (may be empty if the
        fleet is already reasonably balanced).
    """
    if fleet_snapshot.empty:
        return []

    fleet_avg = float(fleet_snapshot["utilization_pct"].mean())

    overloaded = fleet_snapshot[fleet_snapshot["utilization_pct"] >= fleet_avg + deviation_threshold_pct]
    underloaded = fleet_snapshot[fleet_snapshot["utilization_pct"] <= fleet_avg - deviation_threshold_pct]

    overloaded = overloaded.sort_values("utilization_pct", ascending=False).reset_index(drop=True)
    underloaded = underloaded.sort_values("utilization_pct", ascending=True).reset_index(drop=True)

    recommendations = []
    pair_count = min(len(overloaded), len(underloaded))
    for i in range(pair_count):
        over_row = overloaded.iloc[i]
        under_row = underloaded.iloc[i]
        gap = over_row["utilization_pct"] - under_row["utilization_pct"]
        risk = RiskLevel.CRITICAL if gap >= deviation_threshold_pct * 2 else RiskLevel.WARNING

        recommendations.append(
            EqualizationRecommendation(
                resource_type=resource_type,
                overloaded_mac_id=over_row["mac_id"],
                overloaded_project=over_row["project_name"],
                overloaded_utilization_pct=round(float(over_row["utilization_pct"]), 2),
                underloaded_mac_id=under_row["mac_id"],
                underloaded_project=under_row["project_name"],
                underloaded_utilization_pct=round(float(under_row["utilization_pct"]), 2),
                fleet_average_utilization_pct=round(fleet_avg, 2),
                suggested_action=(
                    f"Shift a portion of {resource_type.upper()} load from {over_row['mac_id']} "
                    f"({over_row['project_name']}, {round(over_row['utilization_pct'], 1)}%) to "
                    f"{under_row['mac_id']} ({under_row['project_name']}, "
                    f"{round(under_row['utilization_pct'], 1)}%) to bring both closer to the fleet "
                    f"average of {round(fleet_avg, 1)}% -- avoids over-provisioning new hardware and "
                    f"reduces idle allocation, saving time, cost, and resources."
                ),
                risk_level=risk,
            )
        )

    return recommendations
