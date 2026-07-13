"""
Rule-based recommendation engine. Converts forecast output + current
utilization into structured, actionable recommendations, per the thresholds
defined in config/config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class RecommendationType(str, Enum):
    INCREASE_STORAGE = "increase_storage_capacity"
    DOWNSIZE_COMPUTE = "downsize_underutilized_compute"
    DECOMMISSION = "decommission_orphaned_infrastructure"
    ENABLE_AUTOSCALING = "enable_temporary_autoscaling"
    NO_ACTION = "no_action"


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Recommendation:
    resource_id: str
    resource_type: str
    recommendation_type: RecommendationType
    risk_level: RiskLevel
    current_value: float
    forecasted_value: float
    forecast_horizon_weeks: int
    details: dict = field(default_factory=dict)


def evaluate_storage(
    resource_id: str,
    current_util_pct: float,
    forecast_4wk: pd.DataFrame,
    forecast_12wk: pd.DataFrame,
    critical_threshold: float = 90,
    warning_threshold: float = 75,
) -> list[Recommendation]:
    recs: list[Recommendation] = []

    for weeks, forecast in [(4, forecast_4wk), (12, forecast_12wk)]:
        if forecast is None or forecast.empty:
            continue
        peak = float(forecast["yhat"].max())

        if peak >= critical_threshold:
            recs.append(
                Recommendation(
                    resource_id=resource_id,
                    resource_type="storage",
                    recommendation_type=RecommendationType.INCREASE_STORAGE,
                    risk_level=RiskLevel.CRITICAL,
                    current_value=current_util_pct,
                    forecasted_value=peak,
                    forecast_horizon_weeks=weeks,
                    details={"threshold": critical_threshold},
                )
            )
            break  # no need to also report the longer/shorter horizon once critical is found
        elif peak >= warning_threshold:
            recs.append(
                Recommendation(
                    resource_id=resource_id,
                    resource_type="storage",
                    recommendation_type=RecommendationType.INCREASE_STORAGE,
                    risk_level=RiskLevel.WARNING,
                    current_value=current_util_pct,
                    forecasted_value=peak,
                    forecast_horizon_weeks=weeks,
                    details={"threshold": warning_threshold},
                )
            )

    if not recs:
        recs.append(
            Recommendation(
                resource_id=resource_id,
                resource_type="storage",
                recommendation_type=RecommendationType.NO_ACTION,
                risk_level=RiskLevel.INFO,
                current_value=current_util_pct,
                forecasted_value=current_util_pct,
                forecast_horizon_weeks=0,
            )
        )
    return recs


def evaluate_compute(
    resource_id: str,
    utilization_history_pct: pd.Series,
    underutilization_threshold: float = 30,
    overutilization_threshold: float = 85,
    chronic_days_threshold: int = 21,
) -> Recommendation:
    avg_util = float(utilization_history_pct.mean())
    days_below_threshold = int((utilization_history_pct < underutilization_threshold).sum())

    if days_below_threshold >= chronic_days_threshold:
        rec_type = (
            RecommendationType.DECOMMISSION
            if avg_util < underutilization_threshold * 0.5
            else RecommendationType.DOWNSIZE_COMPUTE
        )
        return Recommendation(
            resource_id=resource_id,
            resource_type="compute",
            recommendation_type=rec_type,
            risk_level=RiskLevel.WARNING,
            current_value=avg_util,
            forecasted_value=avg_util,
            forecast_horizon_weeks=0,
            details={"days_below_threshold": days_below_threshold},
        )

    if avg_util >= overutilization_threshold:
        return Recommendation(
            resource_id=resource_id,
            resource_type="compute",
            recommendation_type=RecommendationType.ENABLE_AUTOSCALING,
            risk_level=RiskLevel.CRITICAL,
            current_value=avg_util,
            forecasted_value=avg_util,
            forecast_horizon_weeks=0,
        )

    return Recommendation(
        resource_id=resource_id,
        resource_type="compute",
        recommendation_type=RecommendationType.NO_ACTION,
        risk_level=RiskLevel.INFO,
        current_value=avg_util,
        forecasted_value=avg_util,
        forecast_horizon_weeks=0,
    )


def evaluate_network_spike(
    resource_id: str,
    throughput_pct_series: pd.Series,
    spike_threshold_pct: float = 85,
    min_spike_days: int = 2,
) -> Recommendation:
    spike_days = int((throughput_pct_series >= spike_threshold_pct).sum())
    avg_util = float(throughput_pct_series.mean())

    if spike_days >= min_spike_days:
        return Recommendation(
            resource_id=resource_id,
            resource_type="network",
            recommendation_type=RecommendationType.ENABLE_AUTOSCALING,
            risk_level=RiskLevel.WARNING,
            current_value=avg_util,
            forecasted_value=float(throughput_pct_series.max()),
            forecast_horizon_weeks=0,
            details={"spike_days": spike_days},
        )

    return Recommendation(
        resource_id=resource_id,
        resource_type="network",
        recommendation_type=RecommendationType.NO_ACTION,
        risk_level=RiskLevel.INFO,
        current_value=avg_util,
        forecasted_value=avg_util,
        forecast_horizon_weeks=0,
    )
