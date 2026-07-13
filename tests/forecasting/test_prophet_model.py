import numpy as np
import pandas as pd
import pytest

from src.forecasting.prophet_model import PROPHET_AVAILABLE, forecast_multi_horizon, forecast_prophet


@pytest.fixture
def sample_series():
    dates = pd.date_range("2025-01-01", periods=90, freq="D")
    values = 50 + np.linspace(0, 20, 90) + 3 * np.sin(np.arange(90) / 7 * 2 * np.pi)
    return pd.DataFrame({"ds": dates, "y": values})


def test_forecast_prophet_returns_expected_columns(sample_series):
    result = forecast_prophet(sample_series, horizon_days=28)
    assert list(result.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(result) == 28


def test_forecast_prophet_raises_on_insufficient_data():
    tiny_series = pd.DataFrame({"ds": ["2025-01-01"], "y": [50.0]})
    if PROPHET_AVAILABLE:
        with pytest.raises(ValueError):
            forecast_prophet(tiny_series, horizon_days=7)
    else:
        # Falls back to linear regression, which also requires >= 2 points
        with pytest.raises(ValueError):
            forecast_prophet(tiny_series, horizon_days=7)


def test_forecast_multi_horizon_returns_all_requested_horizons(sample_series):
    result = forecast_multi_horizon(sample_series, horizons_weeks=[4, 12])
    assert set(result.keys()) == {4, 12}
    assert len(result[4]) == 28
    assert len(result[12]) == 84
