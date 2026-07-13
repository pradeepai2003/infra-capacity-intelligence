import numpy as np
import pandas as pd
import pytest

from src.forecasting.linear_regression_model import forecast_linear, forecast_multi_horizon


@pytest.fixture
def sample_series():
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    values = np.linspace(50, 80, 60)  # clear upward trend
    return pd.DataFrame({"ds": dates, "y": values, "t": range(60)})


def test_forecast_linear_returns_expected_columns(sample_series):
    result = forecast_linear(sample_series, horizon_days=28)
    assert list(result.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(result) == 28


def test_forecast_linear_continues_upward_trend(sample_series):
    result = forecast_linear(sample_series, horizon_days=7)
    assert result["yhat"].iloc[-1] > sample_series["y"].iloc[-1]


def test_forecast_linear_confidence_bounds_valid(sample_series):
    result = forecast_linear(sample_series, horizon_days=14)
    assert (result["yhat_lower"] <= result["yhat"]).all()
    assert (result["yhat"] <= result["yhat_upper"]).all()


def test_forecast_linear_raises_on_insufficient_data():
    tiny_series = pd.DataFrame({"ds": ["2025-01-01"], "y": [50.0], "t": [0]})
    with pytest.raises(ValueError):
        forecast_linear(tiny_series, horizon_days=7)


def test_forecast_multi_horizon_returns_all_requested_horizons(sample_series):
    result = forecast_multi_horizon(sample_series, horizons_weeks=[4, 12])
    assert set(result.keys()) == {4, 12}
    assert len(result[4]) == 28
    assert len(result[12]) == 84
