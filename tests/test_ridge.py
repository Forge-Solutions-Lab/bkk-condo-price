"""Unit tests for pure NumPy Ridge Regression and Bangkok-bounded IDW neighbourhood pricing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.evaluate.mse_experiment import CleanNeighbourhoodPrice, is_valid_bkk_coords
from src.models import linear_regression as lr
from src.models import ridge_regression as ridge


@pytest.fixture
def synthetic_regression_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(250, 6))
    y = X @ np.array([2.0, -1.5, 0.8, 0.0, 3.1, -0.4]) + 11.5 + rng.normal(scale=0.2, size=250)
    return X, y


@pytest.mark.parametrize("lam", [0.1, 1.0, 25.0, 100.0])
def test_ridge_matches_sklearn(synthetic_regression_data, lam: float) -> None:
    X, y = synthetic_regression_data
    W = ridge.fit(X, y, lam=lam)
    reference = Ridge(alpha=lam, fit_intercept=True).fit(X, y)
    np.testing.assert_allclose(W[:-1], reference.coef_, rtol=1e-8)
    np.testing.assert_allclose(W[-1], reference.intercept_, rtol=1e-8)
    np.testing.assert_allclose(ridge.predict(X, W), reference.predict(X), rtol=1e-8)


def test_ridge_zero_lambda_matches_linear_regression(synthetic_regression_data) -> None:
    X, y = synthetic_regression_data
    W_ridge = ridge.fit(X, y, lam=0.0)
    W_ols = lr.fit(X, y)
    np.testing.assert_allclose(W_ridge, W_ols, rtol=1e-8)


def test_is_valid_bkk_coords_rejects_corrupted_pins() -> None:
    lats = pd.Series([13.7367, 100.5578, 13.0937, np.nan])
    lons = pd.Series([100.5608, 100.5558, 100.9135, 100.5600])
    assert list(is_valid_bkk_coords(lats, lons)) == [True, False, False, False]


def test_clean_neighbourhood_filters_bad_coords_and_excludes_self() -> None:
    train = pd.DataFrame([
        # Valid Bangkok tower A (near Asoke)
        {"project_name": "Tower A", "zone_no": 1, "latitude": 13.7370, "longitude": 100.5600, "price_per_sqm": 200_000.0},
        # Valid Bangkok tower B (very close to A)
        {"project_name": "Tower B", "zone_no": 1, "latitude": 13.7380, "longitude": 100.5610, "price_per_sqm": 220_000.0},
        # Valid Bangkok tower C (farther away, cheaper)
        {"project_name": "Tower C", "zone_no": 2, "latitude": 13.8800, "longitude": 100.6200, "price_per_sqm": 70_000.0},
        # Corrupted pin (lat=100.55 like project_id=3928) — must be excluded from lookup and fall back to zone 1 median
        {"project_name": "Bad Pin", "zone_no": 1, "latitude": 100.5578, "longitude": 100.5558, "price_per_sqm": 50_000.0},
    ])
    model = CleanNeighbourhoodPrice.fit(train, k=2, smooth=0.004)
    # Only 3 valid Bangkok towers should remain in the index
    assert len(model.keys) == 3
    assert "bad pin" not in set(model.keys)

    # Tower A excludes itself, so IDW is dominated by nearby Tower B (220k) over distant Tower C (70k)
    vals = model.transform(train)
    assert np.exp(vals[0]) > 180_000.0
    assert np.exp(vals[0]) < 220_000.0
    # Bad Pin falls back to zone 1 training median of log(price_per_sqm)
    expected_zone1_fallback = float(np.log(train.loc[train["zone_no"] == 1, "price_per_sqm"]).median())
    assert vals[3] == pytest.approx(expected_zone1_fallback)
