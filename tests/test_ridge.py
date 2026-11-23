"""Unit tests for pure NumPy Ridge Regression, percentage RMSE metrics, and leak-free V2 unseen features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.evaluate.mse_experiment import (
    CleanNeighbourhoodPrice,
    UnseenFeaturePipeline,
    brand_tokens,
    full_metrics,
    is_valid_bkk_coords,
)
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
        {"project_name": "Tower A", "zone_no": 1, "latitude": 13.7370, "longitude": 100.5600, "price_per_sqm": 200_000.0},
        {"project_name": "Tower B", "zone_no": 1, "latitude": 13.7380, "longitude": 100.5610, "price_per_sqm": 220_000.0},
        {"project_name": "Tower C", "zone_no": 2, "latitude": 13.8800, "longitude": 100.6200, "price_per_sqm": 70_000.0},
        {"project_name": "Bad Pin", "zone_no": 1, "latitude": 100.5578, "longitude": 100.5558, "price_per_sqm": 50_000.0},
    ])
    model = CleanNeighbourhoodPrice.fit(train, k=2, smooth=0.004)
    assert len(model.keys) == 3
    assert "bad pin" not in set(model.keys)

    vals = model.transform(train)
    assert np.exp(vals[0]) > 180_000.0
    assert np.exp(vals[0]) < 220_000.0
    expected_zone1_fallback = float(np.log(train.loc[train["zone_no"] == 1, "price_per_sqm"]).median())
    assert vals[3] == pytest.approx(expected_zone1_fallback)


def test_full_metrics_computes_rmspe_and_nrmse_pct() -> None:
    actual = np.array([2_000_000.0, 4_000_000.0])
    predicted = np.array([2_200_000.0, 3_600_000.0])  # +10% and -10% relative errors
    m = full_metrics(actual, predicted)
    assert m["rmspe_pct"] == pytest.approx(10.0)
    expected_rmse = np.sqrt((200_000.0 ** 2 + 400_000.0 ** 2) / 2.0)
    assert m["rmse_thb"] == pytest.approx(expected_rmse)
    assert m["nrmse_pct"] == pytest.approx(100.0 * expected_rmse / 3_000_000.0)


def test_brand_tokens_extracts_series_prefixes() -> None:
    assert brand_tokens("ลุมพินี เพลส พระราม 9") == ("ลุมพินี", "ลุมพินี เพลส")
    assert brand_tokens("รีเจ้นท์ โฮม 15 แจ้งวัฒนะ") == ("รีเจ้นท์", "รีเจ้นท์ โฮม")
    assert brand_tokens("") == ("", "")


def test_unseen_feature_pipeline_has_zero_self_building_leakage() -> None:
    base = pd.DataFrame([
        {
            "listing_id": "1", "project_name": "โนเบิล สเตท 39", "zone_no": 1, "proj_district_th": "วัฒนา",
            "area_sqm": 35.0, "bedrooms": 1.0, "bathrooms": 1.0, "floor": 12.0, "is_new_unit": False,
            "price_thb": 7_000_000.0, "price_per_sqm": 200_000.0, "complete_year": 2022.0,
            "total_unit": 350.0, "total_floor": 38.0, "total_building": 1.0, "facility_count": 8.0,
            "has_pool": 1.0, "has_gym": 1.0, "nearest_rail_km": 0.4, "nearest_rail_line": "bts",
            "nearest_mall_km": 0.6, "latitude": 13.732, "longitude": 100.571, "has_building_info": True,
        },
        {
            "listing_id": "2", "project_name": "โนเบิล รีมิกซ์ 36", "zone_no": 1, "proj_district_th": "คลองเตย",
            "area_sqm": 45.0, "bedrooms": 1.0, "bathrooms": 1.0, "floor": 10.0, "is_new_unit": False,
            "price_thb": 6_750_000.0, "price_per_sqm": 150_000.0, "complete_year": 2012.0,
            "total_unit": 500.0, "total_floor": 33.0, "total_building": 1.0, "facility_count": 6.0,
            "has_pool": 1.0, "has_gym": 1.0, "nearest_rail_km": 0.2, "nearest_rail_line": "bts",
            "nearest_mall_km": 0.5, "latitude": 13.723, "longitude": 100.577, "has_building_info": True,
        },
        {
            "listing_id": "3", "project_name": "ลุมพินี วิลล์ อ่อนนุช", "zone_no": 8, "proj_district_th": "วัฒนา",
            "area_sqm": 30.0, "bedrooms": 1.0, "bathrooms": 1.0, "floor": 6.0, "is_new_unit": False,
            "price_thb": 2_400_000.0, "price_per_sqm": 80_000.0, "complete_year": 2015.0,
            "total_unit": 900.0, "total_floor": 8.0, "total_building": 4.0, "facility_count": 5.0,
            "has_pool": 1.0, "has_gym": 1.0, "nearest_rail_km": 1.0, "nearest_rail_line": "bts",
            "nearest_mall_km": 1.2, "latitude": 13.708, "longitude": 100.601, "has_building_info": True,
        },
        {
            "listing_id": "4", "project_name": "ลุมพินี เพลส รัชดา", "zone_no": 12, "proj_district_th": "ห้วยขวาง",
            "area_sqm": 32.0, "bedrooms": 1.0, "bathrooms": 1.0, "floor": 15.0, "is_new_unit": False,
            "price_thb": 3_200_000.0, "price_per_sqm": 100_000.0, "complete_year": 2016.0,
            "total_unit": 700.0, "total_floor": 25.0, "total_building": 2.0, "facility_count": 6.0,
            "has_pool": 1.0, "has_gym": 1.0, "nearest_rail_km": 0.5, "nearest_rail_line": "mrt",
            "nearest_mall_km": 0.8, "latitude": 13.765, "longitude": 100.573, "has_building_info": True,
        },
    ])
    perturbed = base.copy()
    perturbed.loc[0, "price_thb"] = 21_000_000.0
    perturbed.loc[0, "price_per_sqm"] = 600_000.0

    X_base = UnseenFeaturePipeline.fit(base).transform(base, leave_one_building_out=True)
    X_pert = UnseenFeaturePipeline.fit(perturbed).transform(perturbed, leave_one_building_out=True)

    for col in ("neigh_idw_k5", "neigh_idw_k15", "neigh_comparable", "district_log_ppsqm"):
        assert X_base.loc[0, col] == pytest.approx(X_pert.loc[0, col])
