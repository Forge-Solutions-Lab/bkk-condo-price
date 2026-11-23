import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

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
