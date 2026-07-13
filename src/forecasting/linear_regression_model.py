"""
Simple, fast linear-trend forecasting model. Used as a lightweight baseline
and for resources with short/noisy histories where Prophet may overfit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def forecast_linear(series: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Args:
        series: DataFrame with columns ['ds', 'y', 't'] (t = numeric day index)
        horizon_days: number of days to forecast forward

    Returns:
        DataFrame with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    """
    if len(series) < 2:
        raise ValueError("Need at least 2 data points to fit a linear regression forecast.")

    X = series[["t"]].values
    y = series["y"].values

    model = LinearRegression()
    model.fit(X, y)

    residual_std = float(np.std(y - model.predict(X)))

    last_t = series["t"].max()
    last_date = pd.to_datetime(series["ds"].max())
    future_t = np.arange(last_t + 1, last_t + 1 + horizon_days).reshape(-1, 1)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    yhat = model.predict(future_t)

    return pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": yhat,
            "yhat_lower": yhat - 1.96 * residual_std,
            "yhat_upper": yhat + 1.96 * residual_std,
        }
    )


def forecast_multi_horizon(series: pd.DataFrame, horizons_weeks: list[int]) -> dict[int, pd.DataFrame]:
    return {weeks: forecast_linear(series, horizon_days=weeks * 7) for weeks in horizons_weeks}
