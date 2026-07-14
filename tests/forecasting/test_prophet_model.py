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


def test_forecast_prophet_falls_back_when_prophet_not_available(monkeypatch, sample_series):
    import src.forecasting.prophet_model as prophet_module

    monkeypatch.setattr(prophet_module, "PROPHET_AVAILABLE", False)
    result = prophet_module.forecast_prophet(sample_series, horizon_days=14)
    assert list(result.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(result) == 14


def test_forecast_prophet_success_path_with_mocked_model(monkeypatch, sample_series):
    """Mocks a working Prophet model so the try-block success path (fit/predict)
    is exercised even in environments where the real Stan backend isn't installed.
    """
    import src.forecasting.prophet_model as prophet_module

    horizon_days = 10
    fake_forecast = pd.DataFrame(
        {
            "ds": pd.date_range(sample_series["ds"].max() + pd.Timedelta(days=1), periods=horizon_days, freq="D"),
            "yhat": [60.0] * horizon_days,
            "yhat_lower": [55.0] * horizon_days,
            "yhat_upper": [65.0] * horizon_days,
            "extra_col": [0] * horizon_days,  # Prophet's real output has many more columns than we need
        }
    )

    class FakeModel:
        def fit(self, df):
            return self

        def make_future_dataframe(self, periods, include_history):
            return pd.DataFrame({"ds": fake_forecast["ds"]})

        def predict(self, future):
            return fake_forecast

    monkeypatch.setattr(prophet_module, "PROPHET_AVAILABLE", True)
    monkeypatch.setattr(prophet_module, "Prophet", lambda **kwargs: FakeModel())

    result = prophet_module.forecast_prophet(sample_series, horizon_days=horizon_days)
    assert list(result.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(result) == horizon_days
    assert (result["yhat"] == 60.0).all()
