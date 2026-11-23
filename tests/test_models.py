import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

from src.features.build import SpatialPriceEncoder, attach_rich_buildings
from src.models import ffbp
from src.models import fuzzy_cmeans as fcm
from src.models import kmeans as km
from src.models import knn_regression as knn
from src.models import linear_regression as lr


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.5, -2.0, 0.5, 3.0]) + 4.0 + rng.normal(scale=0.1, size=200)
    return X, y


@pytest.fixture
def blobs():
    rng = np.random.default_rng(1)
    return np.vstack([rng.normal(loc=c, scale=0.3, size=(60, 2)) for c in ([0, 0], [5, 5], [0, 6])])


def test_linear_regression_matches_sklearn(data):
    X, y = data
    W = lr.fit(X, y)
    reference = LinearRegression().fit(X, y)
    np.testing.assert_allclose(W[:-1], reference.coef_, rtol=1e-8)
    np.testing.assert_allclose(W[-1], reference.intercept_, rtol=1e-8)
    np.testing.assert_allclose(lr.predict(X, W), reference.predict(X), rtol=1e-8)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("weight", ["uniform", "distance"])
def test_knn_matches_sklearn(data, metric, weight):
    X, y = data
    Z = X[:50] + 0.05
    ours = knn.knn_predict(X, y, Z, k=7, metric=metric, weight=weight)
    reference = KNeighborsRegressor(n_neighbors=7, metric=metric, weights=weight, algorithm="brute").fit(X, y)
    np.testing.assert_allclose(ours, reference.predict(Z), rtol=1e-10)


def test_kmeans_matches_sklearn(blobs):
    C, idx, inertia = km.kmeans(blobs, 3)
    reference = KMeans(n_clusters=3, n_init=10, random_state=3).fit(blobs)
    assert inertia == pytest.approx(reference.inertia_, rel=1e-6)
    # same partition up to label renaming: every sklearn cluster maps to exactly one of ours
    for k in range(3):
        assert len(set(idx[reference.labels_ == k])) == 1


def test_kmeans_manhattan_finds_the_three_blobs(blobs):
    C, idx, _ = km.kmeans(blobs, 3, metric="manhattan")
    for k in range(3):
        assert len(set(idx[k * 60:(k + 1) * 60])) == 1
    assert len(set(idx)) == 3


def test_ffbp_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([0, 1, 1, 0], dtype=float)
    W0, W1 = ffbp.ffbp(X, Y, H=4, lr=1.0, epochs=20000, seed=0)
    np.testing.assert_allclose(np.round(ffbp.predict(X, W0, W1)), Y)


def test_ffbp_fits_a_smooth_function(data):
    X, y = data
    y = (y - y.mean()) / y.std()
    W0, W1 = ffbp.ffbp(X, y, H=8, lr=0.5, epochs=3000)
    assert np.mean((ffbp.predict(X, W0, W1) - y) ** 2) < 0.01


def test_fuzzy_cmeans_memberships_sum_to_one_and_find_the_blobs(blobs):
    C, mu = fcm.fuzzy_cmeans(blobs, 3)
    np.testing.assert_allclose(mu.sum(axis=1), 1.0)
    idx = np.argmax(mu, axis=1)
    for k in range(3):
        assert len(set(idx[k * 60:(k + 1) * 60])) == 1
    assert len(set(idx)) == 3


def test_fuzzy_cmeans_centres_match_kmeans_on_separated_blobs(blobs):
    C, _ = fcm.fuzzy_cmeans(blobs, 3)
    reference = KMeans(n_clusters=3, n_init=10, random_state=3).fit(blobs).cluster_centers_
    # every fuzzy centre lies close to one K-Means centre
    for c in C:
        assert np.min(np.linalg.norm(reference - c, axis=1)) < 0.1


def test_ridge_matches_sklearn(data):
    X, y = data
    for alpha in (0.5, 10.0, 100.0):
        W = lr.fit_ridge(X, y, alpha=alpha)
        reference = Ridge(alpha=alpha, solver="cholesky").fit(X, y)
        np.testing.assert_allclose(W[:-1], reference.coef_, rtol=1e-8)
        np.testing.assert_allclose(W[-1], reference.intercept_, rtol=1e-8)
        np.testing.assert_allclose(lr.predict(X, W), reference.predict(X), rtol=1e-8)


def test_ffbp_stable_fits_smooth_function(data):
    X, y = data
    y = (y - y.mean()) / y.std()
    W0, W1 = ffbp.ffbp_stable(X, y, H=10, lr=0.35, epochs=2500, weight_decay=1e-3, seed=0)
    assert np.mean((ffbp.predict(X, W0, W1) - y) ** 2) < 0.02


def test_attach_rich_buildings_sanitizes_bad_coords():
    import pandas as pd

    listings = pd.DataFrame([
        {"project_name": "Good Condo", "zone_no": 1},
        {"project_name": "Bad Lat Condo", "zone_no": 1},
    ])
    projects = pd.DataFrame([
        {"project_name": "Good Condo", "latitude": 13.74, "longitude": 100.56, "complete_year": 2020,
         "total_unit": 500, "total_floor": 30, "total_building": 1, "facility_count": 10,
         "has_pool": 1, "has_gym": 1, "has_parking": 1, "nearest_rail_km": 0.3,
         "nearest_rail_line": "bts", "nearest_mall_km": 0.5, "nearest_hospital_km": 1.0},
        {"project_name": "Bad Lat Condo", "latitude": 100.55, "longitude": 100.55, "complete_year": 2018,
         "total_unit": 300, "total_floor": 20, "total_building": 1, "facility_count": 8,
         "has_pool": 1, "has_gym": 1, "has_parking": 1, "nearest_rail_km": 0.6,
         "nearest_rail_line": "mrt", "nearest_mall_km": 0.8, "nearest_hospital_km": 1.2},
    ])
    merged = attach_rich_buildings(listings, projects)
    assert pd.notna(merged.loc[0, "latitude"])
    assert pd.isna(merged.loc[1, "latitude"])
    assert pd.isna(merged.loc[1, "longitude"])


def test_spatial_encoder_excludes_own_building():
    import pandas as pd

    train = pd.DataFrame([
        {"project_name": "A", "zone_no": 1, "latitude": 13.740, "longitude": 100.560, "price_per_sqm": 200000.0},
        {"project_name": "B", "zone_no": 1, "latitude": 13.741, "longitude": 100.561, "price_per_sqm": 100000.0},
        {"project_name": "C", "zone_no": 1, "latitude": 13.742, "longitude": 100.562, "price_per_sqm": 100000.0},
    ])
    enc = SpatialPriceEncoder.fit(train, k_neighbors=2)
    out = enc.transform(train, leave_one_project_out=True)
    # Row 0 (building A, price 200k) must ONLY average B & C (100k), never seeing its own 200k!
    assert np.isclose(np.exp(out.loc[0, "local_log_pps_k8"]), 100000.0, rtol=1e-6)


def test_spatial_encoder_lopo_excludes_own_project_from_zone_mean():
    import pandas as pd

    train = pd.DataFrame([
        {"project_name": "A", "zone_no": 1, "latitude": 13.740, "longitude": 100.560, "price_per_sqm": 300000.0},
        {"project_name": "B", "zone_no": 1, "latitude": 13.741, "longitude": 100.561, "price_per_sqm": 100000.0},
        {"project_name": "C", "zone_no": 1, "latitude": 13.742, "longitude": 100.562, "price_per_sqm": 100000.0},
    ])
    enc = SpatialPriceEncoder.fit(train, k_neighbors=2, zone_smooth=0.0)
    out = enc.transform(train, leave_one_project_out=True)
    # With zone_smooth=0, row 0's zone mean should average only B and C (100k)
    assert np.isclose(np.exp(out.loc[0, "zone_log_pps"]), 100000.0, rtol=1e-6)


def test_rich_feature_builder_handles_missing_building_and_unseen_zone():
    import pandas as pd
    from src.features.build import RichFeatureBuilder, ZoneClustering

    zone_table = pd.DataFrame([
        {"zone_no": 1, "tier": 1, "median_price_per_sqm": 100000.0},
        {"zone_no": 2, "tier": 2, "median_price_per_sqm": 200000.0},
    ])
    clustering = ZoneClustering(zone_table=zone_table, n_clusters=2)
    train = pd.DataFrame([
        {
            "project_name": "A", "zone_no": 1, "area_sqm": 35.0, "bedrooms": 1, "bathrooms": 1,
            "floor": 10, "is_new_unit": 0, "price_per_sqm": 100000.0, "latitude": 13.74, "longitude": 100.56,
            "complete_year": 2020, "total_unit": 400, "total_floor": 25, "total_building": 1,
            "facility_count": 10, "has_pool": 1, "has_gym": 1, "has_parking": 1,
            "nearest_rail_km": 0.4, "nearest_rail_line": "bts", "nearest_mall_km": 0.6,
            "nearest_hospital_km": 1.2, "has_building_info": True,
        },
        {
            "project_name": "B", "zone_no": 2, "area_sqm": 55.0, "bedrooms": 2, "bathrooms": 2,
            "floor": 18, "is_new_unit": 1, "price_per_sqm": 200000.0, "latitude": 13.73, "longitude": 100.53,
            "complete_year": 2023, "total_unit": 600, "total_floor": 40, "total_building": 2,
            "facility_count": 14, "has_pool": 1, "has_gym": 1, "has_parking": 1,
            "nearest_rail_km": 0.2, "nearest_rail_line": "mrt", "nearest_mall_km": 0.3,
            "nearest_hospital_km": 0.8, "has_building_info": True,
        },
    ])
    test = pd.DataFrame([
        {
            "project_name": "Unknown", "zone_no": 99, "area_sqm": 40.0, "bedrooms": 1, "bathrooms": 1,
            "floor": 5, "is_new_unit": 0, "price_per_sqm": 120000.0, "latitude": np.nan, "longitude": np.nan,
            "complete_year": np.nan, "total_unit": np.nan, "total_floor": np.nan, "total_building": np.nan,
            "facility_count": np.nan, "has_pool": np.nan, "has_gym": np.nan, "has_parking": np.nan,
            "nearest_rail_km": np.nan, "nearest_rail_line": None, "nearest_mall_km": np.nan,
            "nearest_hospital_km": np.nan, "has_building_info": False,
        },
    ])
    builder = RichFeatureBuilder.fit(train, clustering, k_neighbors=2)
    x_tr = builder.transform(train, leave_one_project_out=True)
    x_te = builder.transform(test, leave_one_project_out=False)
    assert list(x_tr.columns) == list(x_te.columns)
    assert np.all(np.isfinite(x_tr.to_numpy()))
    assert np.all(np.isfinite(x_te.to_numpy()))


def test_bundle_prediction_sanity():
    from pathlib import Path
    import pickle
    from app.streamlit_app import build_input, predict

    bundle_path = Path(__file__).resolve().parents[1] / "models" / "model_bundle.pkl"
    if not bundle_path.exists():
        pytest.skip("model_bundle.pkl not built yet")
    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    zone_no = int(bundle["zone_table"]["zone_no"].iloc[0])
    listing = build_input(bundle, zone_no, 35.0, 1, 1, 10, False, None, 2018, 0.5)
    result = predict(bundle, listing)
    assert result["ensemble"] is not None
    ens = result["ensemble"]
    assert ens["low"] < ens["price"] < ens["high"]
    pps = ens["price"] / 35.0
    assert 20_000 < pps < 1_000_000
