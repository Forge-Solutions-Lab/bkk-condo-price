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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split

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
# Valid Bangkok bounding box (matches scripts/check_data.py)
BKK_LAT_BOUNDS = (13.5, 14.0)
BKK_LON_BOUNDS = (100.3, 100.95)
# One degree of latitude is ~111 km; longitude shrinks by cos(13.75 deg) at Bangkok
KM_PER_LAT_DEG = 111.0
LONGITUDE_SCALE = float(np.cos(np.radians(13.75)))
IDW_EPS_KM = 0.25
# Thonburi zones (west bank of the Chao Phraya). Prices differ sharply across the river even a
# kilometre apart (Rat Burana vs Rama 3, Charoen Nakhon vs Silom), so in the neighbourhood price
# a building on the other bank counts as if it were this much farther away.
WEST_BANK_ZONES = frozenset({32, 33, 34, 35, 36, 37, 38})
# Extra distance for a neighbour on the other bank of the Chao Phraya. Off: the bank is taken from
# the zone, and zones such as 29 (Rama 8) have buildings on both banks, so the penalty pushed real
# neighbours away. Grouped CV (3 seeds) with 0 km: zone bias 17.5% vs 27.9% at 5 km, same MAPE.
RIVER_CROSSING_KM = 0.0
# Before averaging, each neighbour's price level is moved to the unit's building: newer, taller
# and larger-unit buildings sell for more per sqm within the same area. Slopes in log price per
# sqm, close to a within-zone building regression on the development rows and chosen by grouped
# CV in src/evaluate/benchmark.py (never on the locked holdout).
NB_AGE_SLOPE = 0.025        # per year of building age
NB_FLOORS_SLOPE = 0.08      # per log(1 + total floors)
NB_LARGE_AREA_SLOPE = 0.08  # per log area above the neighbour's median unit
NB_ADJUST_CAP = 0.45
# Ploenchit / Siam / Asok central business district reference point
CBD_LAT = 13.743
CBD_LON = 100.549


def cv_splits(
    frame: pd.DataFrame | np.ndarray,
    groups: pd.Series | np.ndarray | None = None,
    n_splits: int = CV_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use GroupKFold when evaluating unseen-building splits so units from the same project
    never straddle training and validation folds."""
    if groups is not None:
        return list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(frame, groups=groups))
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(frame))


def clean_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace out-of-Bangkok coordinates (data entry / geocoding errors) with NaN."""
    cleaned = frame.copy()
    if "latitude" not in cleaned.columns or "longitude" not in cleaned.columns:
        return cleaned
    lat = pd.to_numeric(cleaned["latitude"], errors="coerce")
    lon = pd.to_numeric(cleaned["longitude"], errors="coerce")
    valid = lat.between(*BKK_LAT_BOUNDS) & lon.between(*BKK_LON_BOUNDS)
    cleaned["latitude"] = lat.where(valid)
    cleaned["longitude"] = lon.where(valid)
    return cleaned


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


def choose_zone_clustering(
    train: pd.DataFrame,
    metric: str = "euclidean",
    groups: pd.Series | np.ndarray | None = None,
) -> ZoneClustering:
    """Pick k by the cross-validated error of the main Linear Regression, and record
    silhouette and inertia for every k so the choice can be shown and questioned."""
    folds = cv_splits(train, groups=groups)
    scan = []
    for k in K_RANGE:
        fold_errors = []
        for train_index, valid_index in folds:
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
        # a listing without a project name is not part of any building
        train = train[project_key(train["project_name"]) != ""]
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


# within a building with 3+ listings, a price per sqm this far from the others is treated as a
# typing error when building the history (a factor of 1.8 either way)
BUILDING_OUTLIER_FACTOR = 1.8
# weight of the median in the building level; the rest is the mean
BUILDING_MEDIAN_WEIGHT = 0.5
# how many listings' worth of weight the unseen-building estimate gets (chosen by grouped CV in
# src/evaluate/benchmark.py on the development rows only)
KNOWN_PRIOR_STRENGTH = 0.5


def drop_building_outliers(history: pd.DataFrame) -> pd.DataFrame:
    keys = project_key(history["project_name"])
    log_ppsqm = np.log(history["price_per_sqm"])
    grouped = log_ppsqm.groupby(keys)
    count = grouped.transform("size")
    others = (grouped.transform("sum") - log_ppsqm) / (count - 1).clip(lower=1)
    odd = (keys != "") & (count >= 3) & ((log_ppsqm - others).abs() > np.log(BUILDING_OUTLIER_FACTOR))
    return history[~odd]


@dataclass
class KnownBuildingPrice:
    """Price of a unit in a building that already has listings in the data.

    1. Building level: log price per sqm of the building's other listings, half mean and half
       median so one odd listing moves it less.
    2. Shrinkage: with few listings the level is noisy, so it is pulled toward the unseen-building
       estimate for the unit, weighted as if that estimate were `prior_strength` more listings:
       (n * level + m * prior) / (n + m).
    3. Within-building adjustment: Linear Regression on how the unit differs from the rest of its
       building (floor, log area, new unit) explains what is left.
    The history may include listings from pages beyond the per-zone cap: they only change the
    level of their own building, not any zone-level statistic, so they cannot tilt one zone."""

    history: pd.DataFrame
    W_adjust: np.ndarray
    prior_strength: float = KNOWN_PRIOR_STRENGTH
    median_weight: float = BUILDING_MEDIAN_WEIGHT

    @staticmethod
    def make_history(rows: pd.DataFrame) -> pd.DataFrame:
        rows = drop_building_outliers(rows[project_key(rows["project_name"]) != ""])
        return pd.DataFrame({
            "key": project_key(rows["project_name"]).to_numpy(),
            "listing_id": rows["listing_id"].astype(str).to_numpy(),
            "log_ppsqm": np.log(rows["price_per_sqm"].to_numpy()),
            "floor": rows["floor"].to_numpy(dtype=float),
            "log_area": np.log(rows["area_sqm"].to_numpy()),
        })

    def building_stats(self, listings: pd.DataFrame, leave_one_out: bool) -> pd.DataFrame:
        """n, level, mean floor and mean log area of each unit's building; a training unit never
        sees its own listing."""
        keys = project_key(listings["project_name"]).to_numpy()
        # a unit typed into the web app has no listing id
        ids = (listings["listing_id"].astype(str).to_numpy() if "listing_id" in listings
               else np.full(len(listings), ""))
        by_key = {key: (frame["listing_id"].to_numpy(), frame["log_ppsqm"].to_numpy(), frame["floor"].to_numpy(),
                        frame["log_area"].to_numpy()) for key, frame in self.history.groupby("key")}
        out = np.full((len(listings), 4), np.nan)
        out[:, 0] = 0
        for row, (key, listing_id) in enumerate(zip(keys, ids)):
            if key not in by_key:
                continue
            history_ids, values, floors, log_areas = by_key[key]
            keep = history_ids != listing_id if leave_one_out else np.ones(len(values), dtype=bool)
            if not keep.any():
                continue
            kept = values[keep]
            level = (1 - self.median_weight) * kept.mean() + self.median_weight * np.median(kept)
            out[row] = [keep.sum(), level, np.nanmean(floors[keep]), log_areas[keep].mean()]
        return pd.DataFrame(out, columns=["n", "level", "mean_floor", "mean_log_area"])

    @staticmethod
    def adjustment_features(listings: pd.DataFrame, stats: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            listings["floor"].to_numpy(dtype=float) - stats["mean_floor"].to_numpy(),
            np.log(listings["area_sqm"].to_numpy()) - stats["mean_log_area"].to_numpy(),
            listings["is_new_unit"].to_numpy(dtype=float),
        ])

    def base(self, stats: pd.DataFrame, prior_log_ppsqm: np.ndarray) -> np.ndarray:
        n, m = stats["n"].to_numpy(), self.prior_strength
        return (n * stats["level"].to_numpy() + m * prior_log_ppsqm) / (n + m)

    @classmethod
    def fit(cls, train: pd.DataFrame, prior_log_ppsqm: np.ndarray, extra_history: pd.DataFrame | None = None,
            prior_strength: float = KNOWN_PRIOR_STRENGTH, median_weight: float = BUILDING_MEDIAN_WEIGHT) -> KnownBuildingPrice:
        rows = train if extra_history is None else pd.concat([train, extra_history], ignore_index=True)
        model = cls(history=cls.make_history(rows), W_adjust=np.zeros(4), prior_strength=prior_strength,
                    median_weight=median_weight)
        stats = model.building_stats(train, leave_one_out=True)
        has = stats["n"].to_numpy() > 0
        x = cls.adjustment_features(train, stats)[has]
        x = np.nan_to_num(x)
        residual = np.log(train["price_per_sqm"].to_numpy())[has] - model.base(stats, prior_log_ppsqm)[has]
        model.W_adjust = lr.fit(x, residual)
        return model

    def predict(self, listings: pd.DataFrame, prior_log_ppsqm: np.ndarray) -> np.ndarray:
        """Price in baht, NaN for a unit whose building has no listing in the history."""
        stats = self.building_stats(listings, leave_one_out=False)
        has = stats["n"].to_numpy() > 0
        log_ppsqm = self.base(stats, prior_log_ppsqm) + lr.predict(np.nan_to_num(self.adjustment_features(listings, stats)), self.W_adjust)
        return np.where(has, np.exp(log_ppsqm) * listings["area_sqm"].to_numpy(), np.nan)


# a building completed in this year or later counts as a recent launch
RECENT_BUILDING_YEAR = 2013
RECENT_NEIGHBOURS = 5


@dataclass
class RecentNeighbourLevel:
    """Inverse-distance level of log price per sqm of the nearest other buildings completed since
    RECENT_BUILDING_YEAR. A new project is priced against recent launches nearby more than against
    old stock, which the age-adjusted all-building level understates in mixed old/new areas."""

    keys: np.ndarray
    coordinates: np.ndarray
    levels: np.ndarray
    k: int = RECENT_NEIGHBOURS

    @classmethod
    def fit(cls, train: pd.DataFrame) -> RecentNeighbourLevel:
        located = clean_coordinates(train).dropna(subset=["latitude", "longitude"])
        located = located[_column(located, "complete_year") >= RECENT_BUILDING_YEAR]
        grouped = located.groupby(project_key(located["project_name"]))
        buildings = grouped.agg(latitude=("latitude", "first"), longitude=("longitude", "first"))
        buildings["level"] = grouped["price_per_sqm"].apply(lambda prices: float(np.median(np.log(prices))))
        return cls(keys=buildings.index.to_numpy(),
                   coordinates=buildings[["latitude", "longitude"]].to_numpy() * [KM_PER_LAT_DEG, KM_PER_LAT_DEG * LONGITUDE_SCALE],
                   levels=buildings["level"].to_numpy())

    def transform(self, listings: pd.DataFrame) -> np.ndarray:
        """NaN where the unit has no coordinates or no recent building is known."""
        cleaned = clean_coordinates(listings)
        keys = project_key(cleaned["project_name"]).to_numpy()
        points = cleaned[["latitude", "longitude"]].to_numpy(dtype=float) * [KM_PER_LAT_DEG, KM_PER_LAT_DEG * LONGITUDE_SCALE]
        out = np.full(len(cleaned), np.nan)
        for row, (key, point) in enumerate(zip(keys, points)):
            if np.isnan(point).any() or len(self.levels) == 0:
                continue
            distance = np.sqrt(np.sum((self.coordinates - point) ** 2, axis=1))
            distance[self.keys == key] = np.inf
            nearest = np.argsort(distance)[: self.k]
            nearest = nearest[np.isfinite(distance[nearest])]
            if len(nearest):
                weight = 1.0 / (distance[nearest] + IDW_EPS_KM)
                out[row] = float(np.sum(weight * self.levels[nearest]) / np.sum(weight))
        return out


# Spatial spread features from jimmy's SpatialPriceEncoder (feature/model-stability-tuning): the
# same inverse-distance weighting as there (8 nearest other buildings, power 1.5, 0.25 km floor,
# weight grows with the square root of a building's listings up to 5).
SPREAD_NEIGHBOURS = 8
SPREAD_POWER = 1.5
SPREAD_FLOOR_KM = 0.25
SPREAD_DEFAULT_STD = 0.20
SPREAD_DEFAULT_NEAREST_KM = 2.0


@dataclass
class NeighbourSpread:
    """How mixed and how isolated a building's surroundings are: the weighted standard deviation of
    the nearest other buildings' log price per sqm (a mixed area makes the neighbourhood level less
    reliable) and the log distance to the nearest other building."""

    keys: np.ndarray
    coordinates: np.ndarray
    levels: np.ndarray
    counts: np.ndarray

    @classmethod
    def fit(cls, train: pd.DataFrame) -> NeighbourSpread:
        located = clean_coordinates(train).dropna(subset=["latitude", "longitude"])
        log_ppsqm = np.log(located["price_per_sqm"].astype(float))
        buildings = located.assign(_lp=log_ppsqm).groupby(project_key(located["project_name"])).agg(
            latitude=("latitude", "first"), longitude=("longitude", "first"), level=("_lp", "mean"), count=("_lp", "size"))
        return cls(keys=buildings.index.to_numpy(),
                   coordinates=buildings[["latitude", "longitude"]].to_numpy() * [KM_PER_LAT_DEG, KM_PER_LAT_DEG * LONGITUDE_SCALE],
                   levels=buildings["level"].to_numpy(), counts=buildings["count"].to_numpy(dtype=float))

    def transform(self, listings: pd.DataFrame) -> pd.DataFrame:
        cleaned = clean_coordinates(listings)
        keys = project_key(cleaned["project_name"]).to_numpy()
        points = cleaned[["latitude", "longitude"]].to_numpy(dtype=float) * [KM_PER_LAT_DEG, KM_PER_LAT_DEG * LONGITUDE_SCALE]
        spread = np.full(len(cleaned), SPREAD_DEFAULT_STD)
        nearest_km = np.full(len(cleaned), SPREAD_DEFAULT_NEAREST_KM)
        for row, (key, point) in enumerate(zip(keys, points)):
            if np.isnan(point).any() or len(self.levels) == 0:
                continue
            distance = np.sqrt(np.sum((self.coordinates - point) ** 2, axis=1))
            distance[self.keys == key] = np.inf
            nearest = np.argsort(distance)[:SPREAD_NEIGHBOURS]
            nearest = nearest[np.isfinite(distance[nearest])]
            if not len(nearest):
                continue
            nearest_km[row] = distance[nearest[0]]
            weight = (np.sqrt(np.minimum(self.counts[nearest], 5.0))
                      / np.maximum(distance[nearest], SPREAD_FLOOR_KM) ** SPREAD_POWER)
            mean = np.sum(weight * self.levels[nearest]) / np.sum(weight)
            spread[row] = np.sqrt(np.sum(weight * (self.levels[nearest] - mean) ** 2) / np.sum(weight))
        return pd.DataFrame({"nb_spread": spread, "log_nearest_building_km": np.log1p(nearest_km)})


# words that start many project names without naming the developer's series
GENERIC_NAME_WORDS = frozenset({"เดอะ", "เดอ", "เลอ", "ดิ", "the", "บ้าน", "คอนโด"})
# how many buildings' worth of weight the "no premium" prior gets (chosen by grouped CV from 1 and 3)
BRAND_PRIOR_BUILDINGS = 3.0


def project_brand(names: pd.Series) -> pd.Series:
    """Developer series from the project name: its first word ("ไอดีโอ", "ลุมพินี"), or the first two
    when the first is generic ("เดอะ ไลน์")."""
    def first_words(name: str) -> str:
        words = [word for word in str(name).replace("-", " ").split() if word]
        if not words:
            return ""
        return " ".join(words[:2]) if words[0].lower() in GENERIC_NAME_WORDS and len(words) > 1 else words[0]
    return names.fillna("").map(first_words)


@dataclass
class BrandPremium:
    """How much a developer series sells above or below its neighbourhood: for each other building
    of the same brand, its log price per sqm minus its neighbourhood level, averaged and shrunk
    toward 0 by `prior_buildings`. A building never sees its own premium, so a new project of a known
    series gets the series' premium and nothing from its own price."""

    buildings: pd.DataFrame  # brand, key, residual per training building
    prior_buildings: float = BRAND_PRIOR_BUILDINGS

    @classmethod
    def fit(cls, train: pd.DataFrame, neighbourhood_level: np.ndarray,
            prior_buildings: float = BRAND_PRIOR_BUILDINGS) -> BrandPremium:
        frame = pd.DataFrame({
            "key": project_key(train["project_name"]).to_numpy(),
            "brand": project_brand(train["project_name"]).to_numpy(),
            "residual": np.log(train["price_per_sqm"].to_numpy()) - neighbourhood_level,
        })
        frame = frame[(frame["key"] != "") & (frame["brand"] != "")]
        return cls(buildings=frame.groupby(["brand", "key"], as_index=False)["residual"].mean(),
                   prior_buildings=prior_buildings)

    def transform(self, listings: pd.DataFrame) -> np.ndarray:
        keys = project_key(listings["project_name"]).reset_index(drop=True)
        brands = project_brand(listings["project_name"]).reset_index(drop=True)
        grouped = self.buildings.groupby("brand")["residual"]
        sums = brands.map(grouped.sum()).fillna(0.0).to_numpy()
        counts = brands.map(grouped.size()).fillna(0).to_numpy()
        own = keys.map(self.buildings.set_index("key")["residual"]).to_numpy()
        has_own = ~np.isnan(own)
        sums = np.where(has_own, sums - np.nan_to_num(own), sums)
        counts = np.where(has_own, counts - 1, counts)
        weight = counts + self.prior_buildings
        return np.where((brands.to_numpy() != "") & (weight > 0), sums / np.maximum(weight, 1e-12), 0.0)


def attach_buildings(listings: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    """Add each listing's building data (from its project page) by project name, cleaning
    out-of-Bangkok coordinates so geocoding errors do not corrupt distance features."""
    table = clean_coordinates(projects).assign(key=project_key(projects["project_name"])).drop_duplicates("key")
    candidate_columns = [
        "complete_year", "total_unit", "total_floor", "total_building", "facility_count",
        "has_pool", "has_gym", "has_parking", "nearest_rail_km", "nearest_rail_line",
        "nearest_mall_km", "nearest_hospital_km", "latitude", "longitude",
    ]
    columns = [col for col in candidate_columns if col in table.columns]
    merged = listings.assign(key=project_key(listings["project_name"])).merge(table[["key", *columns]], on="key", how="left")
    merged["has_building_info"] = merged["key"].isin(table["key"])
    if "total_floor" in merged.columns and "floor" in merged.columns:
        merged["total_floor"] = infer_total_floor(merged)
    return merged.drop(columns="key")


def infer_total_floor(listings: pd.DataFrame) -> pd.Series:
    """Project pages of new towers often give 0 floors. A listed unit on floor 35 shows the building
    has at least 35, so the highest unit floor seen in the building replaces a missing or smaller
    count (only unit floors are used, never prices)."""
    keys = project_key(listings["project_name"])
    unit_floor = listings["floor"].astype(float)
    highest = unit_floor.groupby(keys).transform("max").where(keys != "", unit_floor)
    highest = np.ceil(np.fmax(highest, unit_floor))
    reported = listings["total_floor"].astype(float)
    use_highest = highest.notna() & (reported.isna() | (reported <= 0) | (highest > reported))
    return reported.where(~use_highest, highest)


def _nearby_checked_mask(listings: pd.DataFrame) -> pd.Series:
    """True when the listing came from a saved project page whose nearby-places list was checked.
    Manual web inputs set manual_building_input=True so unentered mall/hospital distances stay NaN
    and fall back to the training median instead of being capped at 3 km."""
    if "manual_building_input" in listings.columns:
        return listings["has_building_info"] & (~listings["manual_building_input"].fillna(False).astype(bool))
    return listings["has_building_info"]


@dataclass
class BuildingFeatures:
    """Building age, size, facilities and distance to rail and malls. A listing whose
    project page we did not save gets the training median and has_building_info = 0."""

    medians: dict[str, float]

    @staticmethod
    def raw(listings: pd.DataFrame) -> pd.DataFrame:
        # the site lists places within about 3 km; for a real saved project page, no rail station
        # or mall listed means the nearest one is farther than 3 km, so the gap is capped at 3 km
        nearby_checked = _nearby_checked_mask(listings)
        return pd.DataFrame({
            "building_age": (BASE_YEAR - listings["complete_year"]).clip(lower=0),
            "log_total_unit": np.log(listings["total_unit"].where(listings["total_unit"] > 0)),
            "total_floor": listings["total_floor"].where(listings["total_floor"] > 0),
            "facility_count": listings["facility_count"],
            "rail_km": listings["nearest_rail_km"].where(~nearby_checked, listings["nearest_rail_km"].fillna(NEARBY_LIMIT_KM)),
            "mall_km": listings["nearest_mall_km"].where(~nearby_checked, listings["nearest_mall_km"].fillna(NEARBY_LIMIT_KM)),
        }).astype(float)

    @classmethod
    def fit(cls, train: pd.DataFrame) -> BuildingFeatures:
        return cls(medians=cls.raw(train).median().to_dict())

    def transform(self, listings: pd.DataFrame) -> pd.DataFrame:
        frame = self.raw(listings).fillna(self.medians)
        frame["has_building_info"] = listings["has_building_info"].to_numpy(dtype=float)
        return frame


@dataclass
class ExtendedBuildingFeatures:
    """Base building features plus domain features derived from already-parsed project data:
    building density (units per floor), high-rise flag, relative floor height, room spaciousness,
    non-linear rail/age terms, CBD distance, rail line tier (BTS/MRT), hospital distance, and
    core amenities (pool, gym, parking)."""

    base: BuildingFeatures
    medians: dict[str, float]

    @staticmethod
    def _extra_raw(listings: pd.DataFrame, base_frame: pd.DataFrame) -> pd.DataFrame:
        nearby_checked = _nearby_checked_mask(listings)
        cleaned = clean_coordinates(listings)
        lat = cleaned["latitude"].to_numpy(dtype=float) if "latitude" in cleaned.columns else np.full(len(listings), np.nan)
        lon = cleaned["longitude"].to_numpy(dtype=float) if "longitude" in cleaned.columns else np.full(len(listings), np.nan)
        cbd_km = np.sqrt(((lat - CBD_LAT) * KM_PER_LAT_DEG) ** 2 + ((lon - CBD_LON) * KM_PER_LAT_DEG * LONGITUDE_SCALE) ** 2)

        total_floor = listings["total_floor"].where(listings["total_floor"] > 0)
        total_unit = listings["total_unit"].where(listings["total_unit"] > 0)
        units_per_floor_log = np.log(np.maximum(total_unit / total_floor, 1.0))
        is_high_rise = (total_floor > 8).where(total_floor.notna())
        floor_ratio = (listings["floor"].astype(float) / total_floor.astype(float)).clip(lower=0.0, upper=1.0)
        area_per_room_log = np.log(listings["area_sqm"].to_numpy(dtype=float) / (listings["bedrooms"].to_numpy(dtype=float) + 1.0))

        if "nearest_hospital_km" in listings.columns:
            hospital_km = listings["nearest_hospital_km"].where(
                ~nearby_checked, listings["nearest_hospital_km"].fillna(NEARBY_LIMIT_KM)
            )
        else:
            hospital_km = pd.Series(np.nan, index=listings.index)

        if "nearest_rail_line" in listings.columns:
            rail_line = listings["nearest_rail_line"].fillna("").astype(str)
            is_bts = (rail_line == "bts").astype(float).where(nearby_checked)
            is_mrt = (rail_line == "mrt").astype(float).where(nearby_checked)
        else:
            is_bts = pd.Series(np.nan, index=listings.index)
            is_mrt = pd.Series(np.nan, index=listings.index)

        amenities = {}
        for col in ("has_pool", "has_gym", "has_parking"):
            if col in listings.columns:
                amenities[col] = listings[col].where(nearby_checked).astype(float)
            else:
                amenities[col] = pd.Series(np.nan, index=listings.index)

        return pd.DataFrame({
            "units_per_floor_log": units_per_floor_log,
            "is_high_rise": is_high_rise.astype(float),
            "floor_ratio": floor_ratio,
            "area_per_room_log": area_per_room_log,
            "log_rail_km": np.log1p(base_frame["rail_km"].to_numpy(dtype=float)),
            "log_age": np.log1p(base_frame["building_age"].to_numpy(dtype=float)),
            "log_cbd_km": np.log1p(cbd_km),
            "hospital_km": hospital_km,
            "is_bts": is_bts,
            "is_mrt": is_mrt,
            **amenities,
        }).astype(float)

    @classmethod
    def fit(cls, train: pd.DataFrame) -> ExtendedBuildingFeatures:
        base = BuildingFeatures.fit(train)
        base_frame = base.transform(train)
        extra = cls._extra_raw(train, base_frame)
        return cls(base=base, medians=extra.median().to_dict())

    def transform(self, listings: pd.DataFrame) -> pd.DataFrame:
        base_frame = self.base.transform(listings).reset_index(drop=True)
        extra_frame = self._extra_raw(listings.reset_index(drop=True), base_frame).fillna(self.medians)
        return pd.concat([base_frame, extra_frame], axis=1)


@dataclass
class NeighbourhoodPrice:
    """Distance-weighted mean log price per sqm of the k nearest other buildings in the
    training data, found by validated Bangkok coordinates. The unit's own building is never
    used, so a training row cannot see its own price, and the value means the same for a
    new building."""

    keys: np.ndarray
    coordinates: np.ndarray
    levels: np.ndarray
    zone_fallback: pd.Series
    k: int = 10

    @classmethod
    def fit(cls, train: pd.DataFrame, k: int = 10) -> NeighbourhoodPrice:
        located = clean_coordinates(train).dropna(subset=["latitude", "longitude"])
        buildings = located.groupby(project_key(located["project_name"])).agg(
            latitude=("latitude", "first"), longitude=("longitude", "first"),
            level=("price_per_sqm", lambda prices: float(np.log(prices).mean())),
        )
        return cls(
            keys=buildings.index.to_numpy(),
            coordinates=buildings[["latitude", "longitude"]].to_numpy() * [KM_PER_LAT_DEG, KM_PER_LAT_DEG * LONGITUDE_SCALE],
            levels=buildings["level"].to_numpy(),
            zone_fallback=np.log(train["price_per_sqm"]).groupby(train["zone_no"]).median(),
            k=k,
        )

    def transform(self, listings: pd.DataFrame) -> pd.DataFrame:
        cleaned = clean_coordinates(listings)
        keys = project_key(cleaned["project_name"]).to_numpy()
        points = cleaned[["latitude", "longitude"]].to_numpy(dtype=float) * [KM_PER_LAT_DEG, KM_PER_LAT_DEG * LONGITUDE_SCALE]
        levels = np.full(len(cleaned), np.nan)
        for row, (key, point) in enumerate(zip(keys, points)):
            if np.isnan(point).any() or len(self.coordinates) == 0:
                continue
            distance = np.sqrt(np.sum((self.coordinates - point) ** 2, axis=1))
            distance[self.keys == key] = np.inf
            order = np.argsort(distance)[: self.k]
            finite = np.isfinite(distance[order])
            if not np.any(finite):
                continue
            order = order[finite]
            weights = 1.0 / (distance[order] + IDW_EPS_KM)
            levels[row] = float(np.sum(weights * self.levels[order]) / np.sum(weights))
        # no valid coordinates: use the zone median, the same information the K-Means tiers start from
        fallback = cleaned["zone_no"].map(self.zone_fallback).fillna(float(self.zone_fallback.median())).to_numpy()
        return pd.DataFrame({"neighbourhood_log_price_per_sqm": np.where(np.isnan(levels), fallback, levels)})


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame[name].astype(float) if name in frame.columns else pd.Series(np.nan, index=frame.index)


@dataclass
class MultiScaleNeighbourhood:
    """Multi-scale IDW neighbourhood context from validated Bangkok coordinates:
    - nb_idw_k5: hyper-local price level (5 nearest other buildings)
    - nb_idw_k20: broader corridor price level (20 nearest other buildings)
    - nb_age_diff: building age minus IDW age of 10 nearest other buildings.

    Each neighbour's level (half mean, half median log price per sqm, so one odd listing moves it
    less) is first adjusted to the unit's building for age, height and unit size, and a neighbour
    across the Chao Phraya counts as `river_km` farther away."""

    keys: np.ndarray
    coordinates: np.ndarray
    levels: np.ndarray
    ages: np.ndarray
    zone_fallback: pd.Series
    k_local: int = 5
    k_broad: int = 20
    k_age: int = 10
    west_bank: np.ndarray | None = None
    log_floors: np.ndarray | None = None
    log_areas: np.ndarray | None = None
    river_km: float = RIVER_CROSSING_KM
    age_slope: float = NB_AGE_SLOPE
    floors_slope: float = NB_FLOORS_SLOPE
    area_slope: float = NB_LARGE_AREA_SLOPE

    @classmethod
    def fit(cls, train: pd.DataFrame, k_local: int = 5, k_broad: int = 20, k_age: int = 10,
            river_km: float = RIVER_CROSSING_KM, age_slope: float = NB_AGE_SLOPE,
            floors_slope: float = NB_FLOORS_SLOPE, area_slope: float = NB_LARGE_AREA_SLOPE) -> MultiScaleNeighbourhood:
        located = clean_coordinates(train).dropna(subset=["latitude", "longitude"])
        total_floor = _column(located, "total_floor")
        located = located.assign(
            _age=(BASE_YEAR - _column(located, "complete_year")).clip(lower=0),
            _log_floors=np.log1p(total_floor.where(total_floor > 0)),
            _log_area=np.log(_column(located, "area_sqm")),
            _log_ppsqm=np.log(located["price_per_sqm"].astype(float)),
        )
        buildings = located.groupby(project_key(located["project_name"])).agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            zone_no=("zone_no", "first"),
            level_mean=("_log_ppsqm", "mean"),
            level_median=("_log_ppsqm", "median"),
            age=("_age", "first"),
            log_floors=("_log_floors", "first"),
            log_area=("_log_area", "median"),
        )
        return cls(
            keys=buildings.index.to_numpy(),
            coordinates=buildings[["latitude", "longitude"]].to_numpy() * [KM_PER_LAT_DEG, KM_PER_LAT_DEG * LONGITUDE_SCALE],
            levels=(0.5 * buildings["level_mean"] + 0.5 * buildings["level_median"]).to_numpy(),
            ages=buildings["age"].to_numpy(dtype=float),
            zone_fallback=np.log(train["price_per_sqm"]).groupby(train["zone_no"]).median(),
            k_local=k_local,
            k_broad=k_broad,
            k_age=k_age,
            west_bank=buildings["zone_no"].isin(WEST_BANK_ZONES).to_numpy(),
            log_floors=buildings["log_floors"].to_numpy(dtype=float),
            log_areas=buildings["log_area"].to_numpy(dtype=float),
            river_km=river_km,
            age_slope=age_slope,
            floors_slope=floors_slope,
            area_slope=area_slope,
        )

    def _adjustment(self, age: float, log_floors: float, log_area: float) -> np.ndarray:
        """Log price per sqm to add to each neighbour's level so it describes the unit's building."""
        adjust = np.zeros(len(self.levels))
        if not np.isnan(age):
            adjust += self.age_slope * np.nan_to_num(self.ages - age)
        if self.log_floors is not None and not np.isnan(log_floors):
            adjust += self.floors_slope * np.nan_to_num(log_floors - self.log_floors)
        if self.log_areas is not None and not np.isnan(log_area):
            adjust += self.area_slope * np.nan_to_num(np.maximum(0.0, log_area - self.log_areas))
        return np.clip(adjust, -NB_ADJUST_CAP, NB_ADJUST_CAP)

    def transform(self, listings: pd.DataFrame) -> pd.DataFrame:
        cleaned = clean_coordinates(listings)
        keys = project_key(cleaned["project_name"]).to_numpy()
        points = cleaned[["latitude", "longitude"]].to_numpy(dtype=float) * [KM_PER_LAT_DEG, KM_PER_LAT_DEG * LONGITUDE_SCALE]
        listing_age = (BASE_YEAR - _column(cleaned, "complete_year")).clip(lower=0).to_numpy(dtype=float)
        total_floor = _column(cleaned, "total_floor")
        listing_log_floors = np.log1p(total_floor.where(total_floor > 0)).to_numpy(dtype=float)
        listing_log_area = np.log(_column(cleaned, "area_sqm")).to_numpy(dtype=float)
        listing_west = cleaned["zone_no"].isin(WEST_BANK_ZONES).to_numpy()

        local_level = np.full(len(cleaned), np.nan)
        broad_level = np.full(len(cleaned), np.nan)
        age_diff = np.full(len(cleaned), np.nan)

        for row, (key, point) in enumerate(zip(keys, points)):
            if np.isnan(point).any() or len(self.coordinates) == 0:
                continue
            distance = np.sqrt(np.sum((self.coordinates - point) ** 2, axis=1))
            if self.west_bank is not None:
                distance = distance + self.river_km * (self.west_bank != listing_west[row])
            distance[self.keys == key] = np.inf
            order = np.argsort(distance)
            levels = self.levels + self._adjustment(listing_age[row], listing_log_floors[row], listing_log_area[row])

            for k, target in ((self.k_local, local_level), (self.k_broad, broad_level)):
                idx = order[:k]
                idx = idx[np.isfinite(distance[idx])]
                if len(idx):
                    w = 1.0 / (distance[idx] + IDW_EPS_KM)
                    target[row] = float(np.sum(w * levels[idx]) / np.sum(w))

            if not np.isnan(listing_age[row]):
                idx_age = order[: self.k_age]
                idx_age = idx_age[np.isfinite(distance[idx_age]) & ~np.isnan(self.ages[idx_age])]
                if len(idx_age):
                    w_age = 1.0 / (distance[idx_age] + IDW_EPS_KM)
                    nb_age = float(np.sum(w_age * self.ages[idx_age]) / np.sum(w_age))
                    age_diff[row] = listing_age[row] - nb_age

        fallback = cleaned["zone_no"].map(self.zone_fallback).fillna(float(self.zone_fallback.median())).to_numpy()
        return pd.DataFrame({
            "nb_idw_k5": np.where(np.isnan(local_level), fallback, local_level),
            "nb_idw_k20": np.where(np.isnan(broad_level), fallback, broad_level),
            "nb_age_diff": np.where(np.isnan(age_diff), 0.0, age_diff),
        })


def feature_frame(
    listings: pd.DataFrame,
    clustering: ZoneClustering,
    use_location: bool = True,
    project_levels: pd.DataFrame | None = None,
    buildings: pd.DataFrame | None = None,
    neighbourhood: pd.DataFrame | None = None,
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
    if neighbourhood is not None:
        frame[neighbourhood.columns] = neighbourhood.to_numpy()
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
