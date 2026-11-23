"""Train/test splits, zone clustering with K-Means, and model feature matrices.

Zone statistics, cluster centres, project price levels and scaling parameters are always
computed from the training split only, so nothing from the test split leaks into features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split

from src.models import linear_regression as lr
from src.models.fuzzy_cmeans import fuzzy_cmeans
from src.models.kmeans import kmeans

LOG = logging.getLogger(__name__)

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
# Listing count is deliberately not a clustering feature: it reflects how many pages we
# saved per zone (collection effort), not a property of the location.
ZONE_STAT_COLUMNS = ["median_log_price_per_sqm", "median_area_sqm", "share_new_unit"]
# Tiers should summarise 38 zones into a few location classes; beyond 8 groups a cluster
# holds only a handful of zones and stops being a summary.
K_RANGE = range(2, 9)
BASE_YEAR = 2026  # year the listings were collected
NEARBY_LIMIT_KM = 3.0


def split_random(listings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    zone_counts = listings["zone_no"].map(listings["zone_no"].value_counts())
    # stratifying by zone keeps every zone in the training data, so each zone gets a cluster
    stratify = listings["zone_no"] if (zone_counts >= 2).all() and listings["zone_no"].nunique() > 1 else None
    train, test = train_test_split(listings, test_size=TEST_SIZE, random_state=SEED, stratify=stratify)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_by_project(listings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # every unit of a building lands on the same side, so the test set holds only
    # buildings the model has never seen (closer to valuing a new project)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_index, test_index = next(splitter.split(listings, groups=listings["project_name"]))
    return listings.iloc[train_index].reset_index(drop=True), listings.iloc[test_index].reset_index(drop=True)


def zone_statistics(train: pd.DataFrame) -> pd.DataFrame:
    grouped = train.groupby(["zone_no", "zone_slug", "zone_name_th", "district_th"])
    stats = grouped.agg(
        median_price_per_sqm=("price_per_sqm", "median"),
        median_area_sqm=("area_sqm", "median"),
        listing_count=("listing_id", "count"),
        share_new_unit=("is_new_unit", "mean"),
    ).reset_index()
    stats["median_log_price_per_sqm"] = np.log(stats["median_price_per_sqm"])
    return stats


@dataclass
class ZoneClustering:
    zone_table: pd.DataFrame
    n_clusters: int
    metric: str = "euclidean"
    scan: list[dict[str, float]] = field(default_factory=list)

    def tier_of(self, zone_no: pd.Series) -> pd.Series:
        return zone_no.map(self.zone_table.set_index("zone_no")["tier"]).astype(int)


def cluster_zones(stats: pd.DataFrame, n_clusters: int, metric: str = "euclidean") -> tuple[ZoneClustering, float, float]:
    table = stats.copy()
    values = table[ZONE_STAT_COLUMNS].to_numpy()
    scaled = (values - values.mean(axis=0)) / values.std(axis=0)
    _, clusters, inertia = kmeans(scaled, n_clusters, metric=metric, seed=SEED)
    table["cluster"] = clusters
    # rename clusters so tier 1 is the cheapest location group and tier k the most expensive
    order = table.groupby("cluster")["median_price_per_sqm"].median().sort_values().index
    table["tier"] = table["cluster"].map({cluster: rank + 1 for rank, cluster in enumerate(order)})
    silhouette_metric = "manhattan" if metric == "manhattan" else "euclidean"
    has_groups = table["cluster"].nunique() > 1
    silhouette = float(silhouette_score(scaled, table["cluster"], metric=silhouette_metric)) if has_groups else -1.0
    return ZoneClustering(zone_table=table, n_clusters=n_clusters, metric=metric), silhouette, float(inertia)


def fuzzy_zone_memberships(stats: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Fuzzy C-Means on the same zone statistics as K-Means. Each zone gets a degree of
    membership in every tier instead of exactly one tier, so a zone between two price
    levels can be partly both."""
    values = stats[ZONE_STAT_COLUMNS].to_numpy()
    scaled = (values - values.mean(axis=0)) / values.std(axis=0)
    _, mu = fuzzy_cmeans(scaled, n_clusters, seed=SEED)
    # order clusters from cheapest to most expensive, as for the K-Means tiers
    price = stats["median_log_price_per_sqm"].to_numpy()
    order = np.argsort([np.average(price, weights=mu[:, k]) for k in range(n_clusters)])
    table = stats[["zone_no", "zone_slug", "median_price_per_sqm"]].copy()
    for tier, k in enumerate(order, start=1):
        table[f"member_{tier}"] = mu[:, k]
    return table


def membership_frame(listings: pd.DataFrame, memberships: pd.DataFrame) -> pd.DataFrame:
    # tier 1 is left out as the reference level, like the one-hot K-Means tiers
    columns = [column for column in memberships.columns if column.startswith("member_")][1:]
    table = memberships.set_index("zone_no")[columns]
    return table.reindex(listings["zone_no"].to_numpy()).reset_index(drop=True)


def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(100 * np.mean(np.abs(predicted - actual) / actual))


def choose_zone_clustering(train: pd.DataFrame, metric: str = "euclidean") -> ZoneClustering:
    """Pick k by the cross-validated error of the main Linear Regression, and record
    silhouette and inertia for every k so the choice can be shown and questioned."""
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scan = []
    for k in K_RANGE:
        fold_errors = []
        for train_index, valid_index in folds.split(train):
            fold_train, fold_valid = train.iloc[train_index], train.iloc[valid_index]
            clustering, _, _ = cluster_zones(zone_statistics(fold_train), k, metric)
            fold_valid = fold_valid[fold_valid["zone_no"].isin(clustering.zone_table["zone_no"])]
            W = lr.fit(feature_frame(fold_train, clustering).to_numpy(), np.log(fold_train["price_thb"].to_numpy()))
            predicted = np.exp(lr.predict(feature_frame(fold_valid, clustering).to_numpy(), W))
            fold_errors.append(mean_absolute_percentage_error(fold_valid["price_thb"].to_numpy(), predicted))
        _, silhouette, inertia = cluster_zones(zone_statistics(train), k, metric)
        scan.append({"k": k, "cv_mape_pct": float(np.mean(fold_errors)), "silhouette": silhouette, "inertia": inertia})
    best_k = int(min(scan, key=lambda row: row["cv_mape_pct"])["k"])
    clustering, silhouette, _ = cluster_zones(zone_statistics(train), best_k, metric)
    clustering.scan = scan
    LOG.info("zone clustering (%s): k=%s (by CV error), silhouette=%.3f", metric, best_k, silhouette)
    return clustering


def project_key(names: pd.Series) -> pd.Series:
    return names.fillna("").str.strip().str.lower()


@dataclass
class ProjectPriceLevels:
    """Mean log price per sqm of each building in the training data, with the zone median
    as fallback for buildings that have no other listing."""

    sums: pd.Series
    counts: pd.Series
    zone_fallback: pd.Series

    @classmethod
    def fit(cls, train: pd.DataFrame) -> ProjectPriceLevels:
        log_price_per_sqm = np.log(train["price_per_sqm"])
        keys = project_key(train["project_name"])
        return cls(
            sums=log_price_per_sqm.groupby(keys).sum(),
            counts=log_price_per_sqm.groupby(keys).count(),
            zone_fallback=log_price_per_sqm.groupby(train["zone_no"]).median(),
        )

    def transform(self, listings: pd.DataFrame, leave_one_out: bool) -> pd.DataFrame:
        keys = project_key(listings["project_name"])
        sums = keys.map(self.sums).fillna(0.0).to_numpy()
        counts = keys.map(self.counts).fillna(0).to_numpy()
        if leave_one_out:
            # a training row must not see its own price, otherwise the feature leaks the target
            sums = sums - np.log(listings["price_per_sqm"].to_numpy())
            counts = counts - 1
        fallback = listings["zone_no"].map(self.zone_fallback).to_numpy()
        has_history = counts > 0
        level = np.where(has_history, sums / np.maximum(counts, 1), fallback)
        return pd.DataFrame({"project_log_price_per_sqm": level, "has_project_history": has_history.astype(float)})


def attach_buildings(listings: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    """Add each listing's building data (from its project page) by project name."""
    table = projects.assign(key=project_key(projects["project_name"])).drop_duplicates("key")
    columns = ["complete_year", "total_unit", "total_floor", "facility_count", "nearest_rail_km", "nearest_mall_km"]
    merged = listings.assign(key=project_key(listings["project_name"])).merge(table[["key", *columns]], on="key", how="left")
    merged["has_building_info"] = merged["key"].isin(table["key"])
    return merged.drop(columns="key")


@dataclass
class BuildingFeatures:
    """Building age, size, facilities and distance to rail and malls. A listing whose
    project page we did not save gets the training median and has_building_info = 0."""

    medians: dict[str, float]

    @staticmethod
    def raw(listings: pd.DataFrame) -> pd.DataFrame:
        # the site lists places within about 3 km; no rail station or mall listed means
        # the nearest one is farther than that, so the gap is capped at 3 km, not imputed
        return pd.DataFrame({
            "building_age": (BASE_YEAR - listings["complete_year"]).clip(lower=0),
            "log_total_unit": np.log(listings["total_unit"].where(listings["total_unit"] > 0)),
            "total_floor": listings["total_floor"].where(listings["total_floor"] > 0),
            "facility_count": listings["facility_count"],
            "rail_km": listings["nearest_rail_km"].where(~listings["has_building_info"], listings["nearest_rail_km"].fillna(NEARBY_LIMIT_KM)),
            "mall_km": listings["nearest_mall_km"].where(~listings["has_building_info"], listings["nearest_mall_km"].fillna(NEARBY_LIMIT_KM)),
        }).astype(float)

    @classmethod
    def fit(cls, train: pd.DataFrame) -> BuildingFeatures:
        return cls(medians=cls.raw(train).median().to_dict())

    def transform(self, listings: pd.DataFrame) -> pd.DataFrame:
        frame = self.raw(listings).fillna(self.medians)
        frame["has_building_info"] = listings["has_building_info"].to_numpy(dtype=float)
        return frame


def feature_frame(
    listings: pd.DataFrame,
    clustering: ZoneClustering,
    use_location: bool = True,
    project_levels: pd.DataFrame | None = None,
    buildings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "log_area_sqm": np.log(listings["area_sqm"].to_numpy(dtype=float)),
        "bedrooms": listings["bedrooms"].to_numpy(dtype=float),
        "bathrooms": listings["bathrooms"].to_numpy(dtype=float),
        "floor": listings["floor"].to_numpy(dtype=float),
        "is_new_unit": listings["is_new_unit"].to_numpy(dtype=float),
    })
    if use_location and clustering.n_clusters > 1:
        tiers = clustering.tier_of(listings["zone_no"]).to_numpy()
        # one-hot without tier 1, which becomes the reference level of the regression
        for tier in range(2, clustering.n_clusters + 1):
            frame[f"tier_{tier}"] = (tiers == tier).astype(float)
    if project_levels is not None:
        frame[project_levels.columns] = project_levels.to_numpy()
    if buildings is not None:
        frame[buildings.columns] = buildings.to_numpy()
    return frame


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> Standardizer:
        std = values.std(axis=0)
        return cls(mean=values.mean(axis=0), std=np.where(std == 0, 1.0, std))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std
