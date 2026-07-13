"""
Prophet-based forecasting model. Captures weekly seasonality and trend
changepoints better than plain linear regression, which matters for
detecting seasonal usage spikes.

Prophet is an optional dependency: if it isn't installed (e.g. lightweight CI
environments), forecast_prophet transparently falls back to the linear model
so the pipeline never hard-fails.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    PROPHET_AVAILABLE = False


def forecast_prophet(series: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Args:
        series: DataFrame with columns ['ds', 'y']
        horizon_days: number of days to forecast forward

    Returns:
        DataFrame with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    """
    if len(series) < 2:
        raise ValueError("Need at least 2 data points to fit a forecast.")

    def _fallback_to_linear() -> pd.DataFrame:
        logger.warning("Prophet unavailable or failed at runtime; falling back to linear regression forecast.")
        from src.forecasting.linear_regression_model import forecast_linear

        series_with_t = series.copy()
        series_with_t["t"] = range(len(series_with_t))
        return forecast_linear(series_with_t, horizon_days)

    if not PROPHET_AVAILABLE:
        return _fallback_to_linear()

    try:
        model = Prophet(
            weekly_seasonality=True,
            daily_seasonality=False,
            yearly_seasonality=len(series) >= 365,
            interval_width=0.95,
        )
        model.fit(series[["ds", "y"]])

        future = model.make_future_dataframe(periods=horizon_days, include_history=False)
        forecast = model.predict(future)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception as exc:  # noqa: BLE001 - Prophet/Stan backend issues surface as various runtime errors
        logger.warning("Prophet fit/predict failed (%s); falling back to linear regression.", exc)
        return _fallback_to_linear()


def forecast_multi_horizon(series: pd.DataFrame, horizons_weeks: list[int]) -> dict[int, pd.DataFrame]:
    # Prophet fits once and can just be truncated per horizon for efficiency.
    max_horizon_days = max(horizons_weeks) * 7
    full_forecast = forecast_prophet(series, max_horizon_days)

    results = {}
    for weeks in horizons_weeks:
        days = weeks * 7
        results[weeks] = full_forecast.head(days).reset_index(drop=True)
    return results
