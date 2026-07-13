import numpy as np
import pandas as pd

from src.forecasting.forecast_runner import run_forecast_for_all, run_forecast_for_resource


def _make_series(n=60):
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    values = np.linspace(40, 70, n)
    return pd.DataFrame({"ds": dates, "y": values, "t": range(n)})


def test_run_forecast_for_resource_returns_both_models():
    result = run_forecast_for_resource(_make_series(), horizons_weeks=[4])
    assert "linear_regression" in result
    assert "prophet" in result
    assert 4 in result["linear_regression"]
    assert 4 in result["prophet"]


def test_run_forecast_for_all_handles_multiple_resources():
    series_by_id = {"cluster-01": _make_series(), "cluster-02": _make_series()}
    result = run_forecast_for_all(series_by_id, horizons_weeks=[4])
    assert set(result.keys()) == {"cluster-01", "cluster-02"}


def test_run_forecast_for_resource_does_not_raise_on_bad_input():
    # A malformed series should be handled gracefully (errors logged, not raised)
    bad_series = pd.DataFrame({"ds": [], "y": [], "t": []})
    result = run_forecast_for_resource(bad_series, horizons_weeks=[4])
    assert result["linear_regression"] == {}
