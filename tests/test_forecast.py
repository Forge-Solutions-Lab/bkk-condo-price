"""Unit tests for the Bangkok residential price index trend forecast in src/evaluate/forecast.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate.forecast import BASE_YEAR, FIRST_ORIGIN, backtest, fit_trend, predict_index


def _synthetic_exponential_index(growth_rate: float = 0.04, start: int = 1995, end: int = BASE_YEAR) -> pd.Series:
    years = np.arange(start, end + 1)
    values = 100.0 * ((1.0 + growth_rate) ** (years - start))
    return pd.Series(values, index=years)


def test_fit_trend_recovers_exact_exponential_growth() -> None:
    index = _synthetic_exponential_index(growth_rate=0.05)
    weights = fit_trend(index, last_year=BASE_YEAR, window=None)
    predicted = predict_index(weights, np.array([BASE_YEAR, BASE_YEAR + 1, BASE_YEAR + 5]))
    assert predicted[1] / predicted[0] == pytest.approx(1.05, rel=1e-6)
    assert predicted[2] / predicted[0] == pytest.approx(1.05 ** 5, rel=1e-6)


def test_fit_trend_window_tracks_recent_regime() -> None:
    years = np.arange(2000, BASE_YEAR + 1)
    values = []
    level = 100.0
    for y in years:
        # Flat before 2017, then steady 6% annual growth from 2017 onward.
        if y > 2016:
            level *= 1.06
        values.append(level)
    index = pd.Series(values, index=years)

    recent_w = fit_trend(index, last_year=BASE_YEAR, window=10)
    full_w = fit_trend(index, last_year=BASE_YEAR, window=None)
    recent_growth = float(np.exp(recent_w[0]) - 1)
    full_growth = float(np.exp(full_w[0]) - 1)
    assert recent_growth == pytest.approx(0.06, abs=1e-4)
    assert recent_growth > full_growth


def test_backtest_has_zero_error_on_noiseless_exponential_series() -> None:
    index = _synthetic_exponential_index(growth_rate=0.035, start=1991, end=BASE_YEAR)
    errors = backtest(index, window=10)
    assert len(errors) > 0
    assert errors["origin"].min() == FIRST_ORIGIN
    assert errors["horizon"].between(1, 10).all()
    np.testing.assert_allclose(errors["log_error"].to_numpy(), 0.0, atol=1e-10)
