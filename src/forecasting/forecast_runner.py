"""
Orchestrates the forecasting engine: takes the prepared per-resource time
series (from src/databricks/notebooks/04_time_series_prep.py) and produces
4-week and 12-week forecasts using both Linear Regression and Prophet,
picking the best available result per resource.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.forecasting.linear_regression_model import forecast_linear
from src.forecasting.prophet_model import forecast_prophet

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS_WEEKS = [4, 12]


def run_forecast_for_resource(series: pd.DataFrame, horizons_weeks: list[int] = None) -> dict:
    """
    Args:
        series: DataFrame with columns ['ds', 'y', 't']
        horizons_weeks: list of forecast horizons in weeks, e.g. [4, 12]

    Returns:
        {
          "linear_regression": {4: df, 12: df},
          "prophet": {4: df, 12: df}
        }
    """
    horizons_weeks = horizons_weeks or DEFAULT_HORIZONS_WEEKS
    results: dict = {"linear_regression": {}, "prophet": {}}

    for weeks in horizons_weeks:
        horizon_days = weeks * 7
        try:
            results["linear_regression"][weeks] = forecast_linear(series, horizon_days)
        except Exception as exc:  # noqa: BLE001
            logger.error("Linear regression forecast failed: %s", exc)

        try:
            results["prophet"][weeks] = forecast_prophet(series[["ds", "y"]], horizon_days)
        except Exception as exc:  # noqa: BLE001
            logger.error("Prophet forecast failed: %s", exc)

    return results


def run_forecast_for_all(series_by_id: dict[str, pd.DataFrame], horizons_weeks: list[int] = None) -> dict[str, dict]:
    return {
        resource_id: run_forecast_for_resource(series, horizons_weeks) for resource_id, series in series_by_id.items()
    }
