import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

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


def test_linear_regression_ridge_matches_sklearn(data):
    X, y = data
    W = lr.fit(X, y, l2=2.5)
    reference = Ridge(alpha=2.5).fit(X, y)
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


def test_neighbourhood_price_ignores_own_building():
    import pandas as pd
    from src.features.build import NeighbourhoodPrice
    train = pd.DataFrame({
        "project_name": ["A", "A", "B", "C"], "zone_no": [1, 1, 1, 1],
        "latitude": [13.70, 13.70, 13.71, 13.90], "longitude": [100.50, 100.50, 100.50, 100.50],
        "price_per_sqm": [200000.0, 200000.0, 100000.0, 50000.0],
    })
    level = NeighbourhoodPrice.fit(train, k=1).transform(train)["neighbourhood_log_price_per_sqm"]
    # A's nearest other building is B, never A itself
    assert level.iloc[0] == pytest.approx(np.log(100000.0))
    assert level.iloc[2] == pytest.approx(np.log(200000.0))


def test_clean_coordinates_nulls_out_of_bangkok():
    import pandas as pd
    from src.features.build import clean_coordinates

    df = pd.DataFrame({
        "latitude": [13.75, 100.55, 13.09],
        "longitude": [100.55, 100.55, 100.91],
    })
    cleaned = clean_coordinates(df)
    assert cleaned["latitude"].iloc[0] == pytest.approx(13.75)
    assert pd.isna(cleaned["latitude"].iloc[1]) and pd.isna(cleaned["longitude"].iloc[1])
    assert pd.isna(cleaned["latitude"].iloc[2]) and pd.isna(cleaned["longitude"].iloc[2])


def test_manual_building_input_uses_median_for_mall_km():
    import pandas as pd
    from src.features.build import BuildingFeatures

    train = pd.DataFrame({
        "complete_year": [2018.0, 2020.0],
        "total_unit": [300.0, 500.0],
        "total_floor": [20.0, 30.0],
        "facility_count": [8.0, 10.0],
        "nearest_rail_km": [0.4, 0.8],
        "nearest_mall_km": [0.5, 0.7],
        "has_building_info": [True, True],
    })
    bf = BuildingFeatures.fit(train)
    manual = pd.DataFrame({
        "complete_year": [2019.0],
        "total_unit": [np.nan],
        "total_floor": [np.nan],
        "facility_count": [np.nan],
        "nearest_rail_km": [0.5],
        "nearest_mall_km": [np.nan],
        "has_building_info": [True],
        "manual_building_input": [True],
    })
    transformed = bf.transform(manual)
    assert transformed["mall_km"].iloc[0] == pytest.approx(0.6)
    assert transformed["has_building_info"].iloc[0] == pytest.approx(1.0)


def test_multiscale_neighbourhood_ignores_own_building_and_bad_coords():
    import pandas as pd
    from src.features.build import NB_AGE_SLOPE, MultiScaleNeighbourhood

    train = pd.DataFrame({
        "project_name": ["A", "A", "B", "C", "Bad"],
        "zone_no": [1, 1, 1, 1, 1],
        "latitude": [13.73, 13.73, 13.74, 13.85, 100.55],
        "longitude": [100.56, 100.56, 100.56, 100.56, 100.56],
        "complete_year": [2020.0, 2020.0, 2015.0, 2010.0, 2018.0],
        "price_per_sqm": [200000.0, 200000.0, 100000.0, 50000.0, 999999.0],
    })
    nb = MultiScaleNeighbourhood.fit(train, k_local=1, k_broad=2, k_age=1)
    out = nb.transform(train)
    # A ignores itself and Bad (corrupt lat=100.55), so its nearest building is B; B is 5 years
    # older than A, so its level is raised by the age slope before it is used for A
    assert out["nb_idw_k5"].iloc[0] == pytest.approx(np.log(100000.0) + 5 * NB_AGE_SLOPE)
    # A (age 6) minus B (age 11) = -5 years newer than neighbor
    assert out["nb_age_diff"].iloc[0] == pytest.approx(-5.0)
    # Bad falls back to zone median price/sqm and 0.0 age diff
    assert out["nb_age_diff"].iloc[4] == pytest.approx(0.0)


def test_multiscale_neighbourhood_counts_other_river_bank_as_farther():
    import pandas as pd
    from src.features.build import MultiScaleNeighbourhood

    # T (Rat Burana, west bank) is 1 km from R (Rama 3, east bank) and 2 km from W (west bank)
    train = pd.DataFrame({
        "project_name": ["R", "W"],
        "zone_no": [31, 35],
        "latitude": [13.69, 13.69],
        "longitude": [100.52 + 1 / (111 * np.cos(np.radians(13.75))), 100.52 - 2 / (111 * np.cos(np.radians(13.75)))],
        "complete_year": [2015.0, 2015.0],
        "price_per_sqm": [150000.0, 60000.0],
    })
    target = pd.DataFrame({"project_name": ["T"], "zone_no": [35], "latitude": [13.69], "longitude": [100.52],
                           "complete_year": [2015.0], "price_per_sqm": [1.0]})
    near_only = MultiScaleNeighbourhood.fit(train, k_local=1, k_broad=2, k_age=1, river_km=0.0).transform(target)
    with_river = MultiScaleNeighbourhood.fit(train, k_local=1, k_broad=2, k_age=1).transform(target)
    assert near_only["nb_idw_k5"].iloc[0] == pytest.approx(np.log(150000.0))
    assert with_river["nb_idw_k5"].iloc[0] == pytest.approx(np.log(60000.0))


def test_attach_buildings_infers_missing_total_floor_from_unit_floors():
    import pandas as pd
    from src.features.build import attach_buildings

    projects = pd.DataFrame({"project_name": ["New Tower", "Old"], "total_floor": [0, 8],
                             "latitude": [13.75, 13.76], "longitude": [100.55, 100.56]})
    listings = pd.DataFrame({"project_name": ["New Tower", "New Tower", "Old"], "floor": [12.0, 34.5, 5.0]})
    out = attach_buildings(listings, projects)
    # page says 0 floors, but a unit is on floor 34.5: the tower has at least 35
    assert out["total_floor"].tolist() == [35.0, 35.0, 8.0]
