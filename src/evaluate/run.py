"""Train every model on the same split, evaluate on held-out data, and save metrics,
figures and the model bundle used by the web app.

Two evaluation splits are reported:
- random split: test units may come from buildings that also appear in training
  (like valuing a unit in a building with other listings on the market);
- project split: test buildings never appear in training (like valuing a new building).

Distance experiment: K-Means is run with Euclidean and Manhattan distance, and KNN with
both distances and both voting rules (equal vote, distance-weighted vote). The setting
used for the main models is the one with the lowest cross-validated error on the
training data; the test set is never used to choose.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from src.features.build import (
    SEED,
    BuildingFeatures,
    ExtendedBuildingFeatures,
    MultiScaleNeighbourhood,
    NeighbourhoodPrice,
    KnownBuildingPrice,
    ProjectPriceLevels,
    Standardizer,
    ZoneClustering,
    attach_buildings,
    choose_zone_clustering,
    cv_splits,
    feature_frame,
    fuzzy_zone_memberships,
    membership_frame,
    project_key,
    split_by_project,
    split_random,
    zone_statistics,
)
from src.models import ffbp
from src.models import linear_regression as lr
from src.models.knn_regression import knn_predict

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
PROJECTS_CSV = PROJECT_ROOT / "data" / "interim" / "projects.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
BUNDLE_PATH = PROJECT_ROOT / "models" / "model_bundle.pkl"

MAX_PAGES_PER_ZONE = 2
K_CANDIDATES = range(1, 31)
CV_FOLDS = 5
LEARNING_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
FILTER_AREA_TOLERANCE = 0.10
METRICS = ["euclidean", "manhattan"]
WEIGHTS = ["uniform", "distance"]
NEIGHBOUR_COUNTS = [5, 10, 20]
RIDGE_CANDIDATES = [0.0, 1.0, 5.0]
BLEND_WEIGHTS = [0.0, 0.25, 0.35, 0.5, 0.65, 0.75, 1.0]
# bands of the predicted price, the value the web app knows before the real price
PRICE_BANDS = [0, 3e6, 5e6, 8e6, 15e6, np.inf]
PRICE_BAND_LABELS = ["< 3 ล้าน", "3-5 ล้าน", "5-8 ล้าน", "8-15 ล้าน", "> 15 ล้าน"]
FFBP_SETTINGS = [{"H": H, "epochs": epochs} for H in (4, 8, 16) for epochs in (3000, 10000)]
GOLD = "#FFC000"
WHITE = "#FFFFFF"
ASH = "#7D7D7D"
CHARCOAL = "#202020"

MAIN_LR = "Linear Regression (K-Means tiers)"
MAIN_KNN = "KNN Regression (K-Means tiers)"
PROJECT_LR = "Linear Regression + project price level"
BUILDING_LR = "Linear Regression + building features"
BUILDING_PROJECT_LR = "Linear Regression + building + project price level"
BUILDING_KNN = "KNN Regression + building features"
MAIN_FFBP = "FFBP neural network (K-Means tiers)"
BUILDING_FFBP = "FFBP neural network + building features"
FUZZY_LR = "Linear Regression (Fuzzy C-Means memberships)"
NEIGHBOUR_LR = "Linear Regression + building + neighbourhood price"
NEIGHBOUR_FFBP = "FFBP neural network + building + neighbourhood price"
NEIGHBOUR_GB = "Gradient Boosting + building + neighbourhood price (reference)"
BLEND = "Blend: Linear Regression + {partner} (building + neighbourhood price)"
FUZZY_BUILDING_LR = "Linear Regression + building features (Fuzzy C-Means memberships)"
EXTENDED_LR = "Linear Regression + extended building + multi-scale IDW"
EXTENDED_GB = "Gradient Boosting (log price/sqm) + extended + IDW (reference)"
EXTENDED_RF = "Random Forest (log price/sqm) + extended + IDW (reference)"
EXTENDED_BLEND = "Blend: LR + GB (extended + IDW, CV-weighted)"
EXTENDED_PROJECT_LR = "Linear Regression + extended + project price level"


def price_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    absolute_percent = np.abs(errors) / actual
    return {
        "mae_thb": float(np.mean(np.abs(errors))),
        "rmse_thb": float(np.sqrt(np.mean(errors**2))),
        "mape_pct": float(100 * np.mean(absolute_percent)),
        "median_ape_pct": float(100 * np.median(absolute_percent)),
        "r2": float(1 - np.sum(errors**2) / np.sum((actual - actual.mean()) ** 2)),
        "within_10pct": float(100 * np.mean(absolute_percent <= 0.10)),
        "within_20pct": float(100 * np.mean(absolute_percent <= 0.20)),
    }


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))


def sota_style_metrics(test: pd.DataFrame, predicted_price: np.ndarray) -> dict[str, float]:
    """Metrics in the form used by earlier Bangkok studies: price per sqm, and price per
    sqm averaged to project level (those studies predict one value per project)."""
    actual = test["price_per_sqm"].to_numpy()
    predicted = predicted_price / test["area_sqm"].to_numpy()
    projects = pd.DataFrame({"project": test["project_name"], "actual": actual, "predicted": predicted})
    projects = projects.groupby("project").median()
    return {
        "price_per_sqm_r2": r_squared(actual, predicted),
        "project_price_per_sqm_r2": r_squared(projects["actual"].to_numpy(), projects["predicted"].to_numpy()),
        "project_relative_error_pct": float(100 * np.mean(np.abs(projects["predicted"] - projects["actual"]) / projects["actual"])),
    }


def website_filter_prediction(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, float]:
    """What a buyer gets by filtering a listing site: median price of listings in the same
    zone with the same bedrooms and an area within 10%. When nothing matches, the zone's
    median price per sqm times the area is used, and the match rate is reported."""
    predicted, matched = [], 0
    zone_price = train.groupby("zone_no")["price_per_sqm"].median()
    for row in test.itertuples():
        same = train[(train["zone_no"] == row.zone_no) & (train["bedrooms"] == row.bedrooms)
                     & train["area_sqm"].between(row.area_sqm * (1 - FILTER_AREA_TOLERANCE), row.area_sqm * (1 + FILTER_AREA_TOLERANCE))]
        if len(same):
            predicted.append(same["price_thb"].median())
            matched += 1
        else:
            predicted.append(zone_price[row.zone_no] * row.area_sqm)
    return np.array(predicted), 100 * matched / len(test)


def choose_knn_k(
    features: np.ndarray,
    log_price: np.ndarray,
    metric: str,
    weight: str,
    groups: pd.Series | np.ndarray | None = None,
) -> tuple[int, list[dict[str, float]]]:
    folds = cv_splits(features, groups=groups)
    curve = []
    for k in K_CANDIDATES:
        fold_errors = []
        for train_index, valid_index in folds:
            scaler = Standardizer.fit(features[train_index])
            X, Z = scaler.transform(features[train_index]), scaler.transform(features[valid_index])
            predicted = np.exp(knn_predict(X, log_price[train_index], Z, k, metric, weight))
            actual = np.exp(log_price[valid_index])
            fold_errors.append(100 * np.mean(np.abs(predicted - actual) / actual))
        curve.append({"k": k, "cv_mape_pct": float(np.mean(fold_errors))})
    best = min(curve, key=lambda row: row["cv_mape_pct"])
    return int(best["k"]), curve


@dataclass
class FittedKnn:
    """KNN keeps the scaled training rows; prediction is a vote of the k nearest of them."""

    X: np.ndarray
    y: np.ndarray
    scaler: Standardizer
    k: int
    metric: str
    weight: str
    cv_mape_pct: float
    curve: list[dict[str, float]]

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        log_price: np.ndarray,
        metric: str,
        weight: str,
        groups: pd.Series | np.ndarray | None = None,
    ) -> FittedKnn:
        k, curve = choose_knn_k(features, log_price, metric, weight, groups=groups)
        scaler = Standardizer.fit(features)
        cv_mape = min(row["cv_mape_pct"] for row in curve)
        return cls(scaler.transform(features), log_price, scaler, k, metric, weight, cv_mape, curve)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.exp(knn_predict(self.X, self.y, self.scaler.transform(features), self.k, self.metric, self.weight))


@dataclass
class FittedFfbp:
    """Neural network on standardised features and standardised log price. Hidden size and
    number of epochs are chosen by 5-fold cross-validation on the training data."""

    W0: np.ndarray
    W1: np.ndarray
    scaler: Standardizer
    y_mean: float
    y_std: float
    setting: dict[str, int]
    cv_mape_pct: float

    @staticmethod
    def train(features: np.ndarray, log_price: np.ndarray, setting: dict[str, int]) -> tuple[np.ndarray, np.ndarray, Standardizer, float, float]:
        scaler = Standardizer.fit(features)
        y_mean, y_std = float(log_price.mean()), float(log_price.std())
        W0, W1 = ffbp.ffbp(scaler.transform(features), (log_price - y_mean) / y_std, H=setting["H"],
                           epochs=setting["epochs"], seed=SEED)
        return W0, W1, scaler, y_mean, y_std

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        log_price: np.ndarray,
        groups: pd.Series | np.ndarray | None = None,
    ) -> FittedFfbp:
        folds = cv_splits(features, groups=groups)
        scores = []
        for setting in FFBP_SETTINGS:
            fold_errors = []
            for train_index, valid_index in folds:
                W0, W1, scaler, y_mean, y_std = cls.train(features[train_index], log_price[train_index], setting)
                predicted = np.exp(ffbp.predict(scaler.transform(features[valid_index]), W0, W1) * y_std + y_mean)
                actual = np.exp(log_price[valid_index])
                fold_errors.append(100 * np.mean(np.abs(predicted - actual) / actual))
            scores.append((float(np.mean(fold_errors)), setting))
        cv_mape, setting = min(scores, key=lambda score: score[0])
        return cls(*cls.train(features, log_price, setting), setting=setting, cv_mape_pct=cv_mape)

    @classmethod
    def fit_with_neighbourhood(
        cls,
        train: pd.DataFrame,
        clustering: ZoneClustering,
        neighbour_k: int,
        groups: pd.Series | np.ndarray | None = None,
    ) -> FittedFfbp:
        """Tune FFBP with neighbourhood price refit strictly inside every CV fold (no leakage)."""
        folds = cv_splits(train, groups=groups)
        fold_data = []
        for train_index, valid_index in folds:
            fold_train, fold_valid = train.iloc[train_index], train.iloc[valid_index]
            fold_buildings = BuildingFeatures.fit(fold_train)
            fold_nb = NeighbourhoodPrice.fit(fold_train, neighbour_k)
            x_fit = feature_frame(fold_train, clustering, buildings=fold_buildings.transform(fold_train),
                                  neighbourhood=fold_nb.transform(fold_train)).to_numpy()
            x_valid = feature_frame(fold_valid, clustering, buildings=fold_buildings.transform(fold_valid),
                                    neighbourhood=fold_nb.transform(fold_valid)).to_numpy()
            fold_data.append((x_fit, np.log(fold_train["price_thb"].to_numpy()), x_valid, fold_valid["price_thb"].to_numpy()))
        scores = []
        for setting in FFBP_SETTINGS:
            fold_errors = []
            for x_fit, y_fit, x_valid, actual in fold_data:
                W0, W1, scaler, y_mean, y_std = cls.train(x_fit, y_fit, setting)
                predicted = np.exp(ffbp.predict(scaler.transform(x_valid), W0, W1) * y_std + y_mean)
                fold_errors.append(mape_pct(actual, predicted))
            scores.append((float(np.mean(fold_errors)), setting))
        cv_mape, setting = min(scores, key=lambda score: score[0])
        full_buildings = BuildingFeatures.fit(train)
        full_nb = NeighbourhoodPrice.fit(train, neighbour_k)
        x_full = feature_frame(train, clustering, buildings=full_buildings.transform(train),
                               neighbourhood=full_nb.transform(train)).to_numpy()
        return cls(*cls.train(x_full, np.log(train["price_thb"].to_numpy()), setting), setting=setting, cv_mape_pct=cv_mape)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.exp(ffbp.predict(self.scaler.transform(features), self.W0, self.W1) * self.y_std + self.y_mean)


def mape_pct(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(100 * np.mean(np.abs(predicted - actual) / actual))


def make_extended_gb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=2.0,
        learning_rate=0.05,
        max_iter=250,
        random_state=SEED,
    )


def choose_neighbour_count(
    train: pd.DataFrame,
    clustering: ZoneClustering,
    buildings: BuildingFeatures | None = None,
    groups: pd.Series | np.ndarray | None = None,
) -> tuple[int, list[dict[str, float]]]:
    """k nearest buildings for the neighbourhood price, chosen by the CV error of Linear Regression.
    Both BuildingFeatures and NeighbourhoodPrice are rebuilt inside every fold from training rows only."""
    _ = buildings  # kept optional for backward compatibility; refit per fold below to avoid leakage
    folds = cv_splits(train, groups=groups)
    scan = []
    for k in NEIGHBOUR_COUNTS:
        errors = []
        for train_index, valid_index in folds:
            fold_train, fold_valid = train.iloc[train_index], train.iloc[valid_index]
            fold_buildings = BuildingFeatures.fit(fold_train)
            neighbourhood = NeighbourhoodPrice.fit(fold_train, k)
            x_fit = feature_frame(fold_train, clustering, buildings=fold_buildings.transform(fold_train),
                                  neighbourhood=neighbourhood.transform(fold_train)).to_numpy()
            x_valid = feature_frame(fold_valid, clustering, buildings=fold_buildings.transform(fold_valid),
                                    neighbourhood=neighbourhood.transform(fold_valid)).to_numpy()
            W = lr.fit(x_fit, np.log(fold_train["price_thb"].to_numpy()))
            errors.append(mape_pct(fold_valid["price_thb"].to_numpy(), np.exp(lr.predict(x_valid, W))))
        scan.append({"k": k, "cv_mape_pct": float(np.mean(errors))})
    return int(min(scan, key=lambda row: row["cv_mape_pct"])["k"]), scan


def choose_blend_partner(
    train: pd.DataFrame,
    clustering: ZoneClustering,
    neighbour_k: int,
    ffbp_setting: dict[str, int],
    groups: pd.Series | np.ndarray | None = None,
) -> tuple[str, dict[str, float]]:
    """Average Linear Regression with a second model (geometric mean of the two prices).
    BuildingFeatures and NeighbourhoodPrice are refit inside every CV fold to prevent leakage."""
    folds = cv_splits(train, groups=groups)
    errors: dict[str, list[float]] = {"Linear Regression alone": [], "FFBP": [], "Gradient Boosting": []}
    for train_index, valid_index in folds:
        fold_train, fold_valid = train.iloc[train_index], train.iloc[valid_index]
        fold_buildings = BuildingFeatures.fit(fold_train)
        fold_nb = NeighbourhoodPrice.fit(fold_train, neighbour_k)
        x_fit = feature_frame(fold_train, clustering, buildings=fold_buildings.transform(fold_train),
                              neighbourhood=fold_nb.transform(fold_train)).to_numpy()
        x_valid = feature_frame(fold_valid, clustering, buildings=fold_buildings.transform(fold_valid),
                                neighbourhood=fold_nb.transform(fold_valid)).to_numpy()
        y_fit = np.log(fold_train["price_thb"].to_numpy())
        actual = fold_valid["price_thb"].to_numpy()
        log_lr = lr.predict(x_valid, lr.fit(x_fit, y_fit))
        W0, W1, scaler, y_mean, y_std = FittedFfbp.train(x_fit, y_fit, ffbp_setting)
        log_ffbp = ffbp.predict(scaler.transform(x_valid), W0, W1) * y_std + y_mean
        log_gb = HistGradientBoostingRegressor(random_state=SEED).fit(x_fit, y_fit).predict(x_valid)
        errors["Linear Regression alone"].append(mape_pct(actual, np.exp(log_lr)))
        errors["FFBP"].append(mape_pct(actual, np.exp((log_lr + log_ffbp) / 2)))
        errors["Gradient Boosting"].append(mape_pct(actual, np.exp((log_lr + log_gb) / 2)))
    scores = {name: float(np.mean(values)) for name, values in errors.items()}
    partner = min(("FFBP", "Gradient Boosting"), key=scores.get)
    return partner, scores


def choose_extended_settings(
    train: pd.DataFrame,
    clustering: ZoneClustering,
    groups: pd.Series | np.ndarray | None = None,
) -> dict[str, object]:
    """Select Ridge L2 penalty for extended LR and blend weight w_lr (between extended LR and
    log-price/sqm Gradient Boosting) strictly via leak-free 5-fold CV."""
    folds = cv_splits(train, groups=groups)
    oof_lr: dict[float, list[np.ndarray]] = {l2: [] for l2 in RIDGE_CANDIDATES}
    oof_gb: list[np.ndarray] = []
    oof_actual: list[np.ndarray] = []

    for train_index, valid_index in folds:
        fold_train, fold_valid = train.iloc[train_index], train.iloc[valid_index]
        ext_bf = ExtendedBuildingFeatures.fit(fold_train)
        ms_nb = MultiScaleNeighbourhood.fit(fold_train)
        x_fit = feature_frame(fold_train, clustering, buildings=ext_bf.transform(fold_train),
                              neighbourhood=ms_nb.transform(fold_train)).to_numpy()
        x_valid = feature_frame(fold_valid, clustering, buildings=ext_bf.transform(fold_valid),
                                neighbourhood=ms_nb.transform(fold_valid)).to_numpy()
        y_fit_log = np.log(fold_train["price_thb"].to_numpy())
        y_fit_ppsqm = np.log(fold_train["price_per_sqm"].to_numpy())
        log_area_valid = np.log(fold_valid["area_sqm"].to_numpy())
        for l2 in RIDGE_CANDIDATES:
            oof_lr[l2].append(lr.predict(x_valid, lr.fit(x_fit, y_fit_log, l2=l2)))
        gb_fold = make_extended_gb().fit(x_fit, y_fit_ppsqm)
        oof_gb.append(gb_fold.predict(x_valid) + log_area_valid)
        oof_actual.append(fold_valid["price_thb"].to_numpy())

    l2_scores = {
        l2: float(np.mean([mape_pct(a, np.exp(p)) for a, p in zip(oof_actual, oof_lr[l2])]))
        for l2 in RIDGE_CANDIDATES
    }
    best_l2 = min(l2_scores, key=l2_scores.get)
    gb_cv_mape = float(np.mean([mape_pct(a, np.exp(p)) for a, p in zip(oof_actual, oof_gb)]))
    weight_scores = {
        w: float(np.mean([
            mape_pct(a, np.exp(w * p_lr + (1.0 - w) * p_gb))
            for a, p_lr, p_gb in zip(oof_actual, oof_lr[best_l2], oof_gb)
        ]))
        for w in BLEND_WEIGHTS
    }
    best_w_lr = min(weight_scores, key=weight_scores.get)
    return {
        "l2": float(best_l2),
        "w_lr": float(best_w_lr),
        "lr_cv_mape_pct": l2_scores[best_l2],
        "gb_cv_mape_pct": gb_cv_mape,
        "blend_cv_mape_pct": weight_scores[best_w_lr],
        "l2_scan": {str(k): v for k, v in l2_scores.items()},
        "weight_scan": {str(k): v for k, v in weight_scores.items()},
    }


def best_zone_clustering(
    train: pd.DataFrame,
    groups: pd.Series | np.ndarray | None = None,
) -> tuple[ZoneClustering, list[dict[str, object]]]:
    candidates = [choose_zone_clustering(train, metric, groups=groups) for metric in METRICS]
    rows = []
    for clustering in candidates:
        chosen = next(row for row in clustering.scan if row["k"] == clustering.n_clusters)
        rows.append({"metric": clustering.metric, "k": clustering.n_clusters,
                     "cv_mape_pct": chosen["cv_mape_pct"], "silhouette": chosen["silhouette"]})
    best = min(candidates, key=lambda clustering: next(
        row["cv_mape_pct"] for row in clustering.scan if row["k"] == clustering.n_clusters))
    return best, rows


def best_knn(
    features: np.ndarray,
    log_price: np.ndarray,
    groups: pd.Series | np.ndarray | None = None,
) -> tuple[FittedKnn, list[FittedKnn]]:
    candidates = [FittedKnn.fit(features, log_price, metric, weight, groups=groups) for metric in METRICS for weight in WEIGHTS]
    return min(candidates, key=lambda model: model.cv_mape_pct), candidates


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame, group_cv: bool = False) -> dict[str, object]:
    groups = train["project_name"] if group_cv else None
    clustering, kmeans_experiment = best_zone_clustering(train, groups=groups)
    test = test[test["zone_no"].isin(clustering.zone_table["zone_no"])].reset_index(drop=True)
    y_train_log = np.log(train["price_thb"].to_numpy())
    y_train_ppsqm_log = np.log(train["price_per_sqm"].to_numpy())
    y_test = test["price_thb"].to_numpy()
    log_area_test = np.log(test["area_sqm"].to_numpy())

    x_train = feature_frame(train, clustering).to_numpy()
    x_test = feature_frame(test, clustering).to_numpy()
    levels = ProjectPriceLevels.fit(train)
    xp_train_frame = feature_frame(train, clustering, project_levels=levels.transform(train, leave_one_out=True))
    xp_train = xp_train_frame.to_numpy()
    xp_test = feature_frame(test, clustering, project_levels=levels.transform(test, leave_one_out=False)).to_numpy()
    buildings = BuildingFeatures.fit(train)
    xb_train_frame = feature_frame(train, clustering, buildings=buildings.transform(train))
    xb_train = xb_train_frame.to_numpy()
    xb_test = feature_frame(test, clustering, buildings=buildings.transform(test)).to_numpy()
    xbp_train_frame = feature_frame(train, clustering, project_levels=levels.transform(train, leave_one_out=True),
                                    buildings=buildings.transform(train))
    xbp_train = xbp_train_frame.to_numpy()
    xbp_test = feature_frame(test, clustering, project_levels=levels.transform(test, leave_one_out=False),
                             buildings=buildings.transform(test)).to_numpy()
    x_train_no_location = feature_frame(train, clustering, use_location=False).to_numpy()
    x_test_no_location = feature_frame(test, clustering, use_location=False).to_numpy()

    W = lr.fit(x_train, y_train_log)
    W_project = lr.fit(xp_train, y_train_log)
    W_no_location = lr.fit(x_train_no_location, y_train_log)
    knn, knn_candidates = best_knn(x_train, y_train_log, groups=groups)
    knn_project = FittedKnn.fit(xp_train, y_train_log, knn.metric, knn.weight, groups=groups)
    W_building = lr.fit(xb_train, y_train_log)
    W_building_project = lr.fit(xbp_train, y_train_log)
    knn_building = FittedKnn.fit(xb_train, y_train_log, knn.metric, knn.weight, groups=groups)
    forest_building = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(xb_train, y_train_log)
    boosting_building = HistGradientBoostingRegressor(random_state=SEED).fit(xb_train, y_train_log)
    boosting_building_project = HistGradientBoostingRegressor(random_state=SEED).fit(xbp_train, y_train_log)
    # Fuzzy C-Means with the same number of groups that K-Means chose, so only the
    # hard-versus-soft assignment differs between the two
    memberships = fuzzy_zone_memberships(zone_statistics(train), clustering.n_clusters)
    xf_train = pd.concat([feature_frame(train, clustering, use_location=False), membership_frame(train, memberships)], axis=1)
    xf_test = pd.concat([feature_frame(test, clustering, use_location=False), membership_frame(test, memberships)], axis=1)
    xfb_train = pd.concat([xf_train, buildings.transform(train)], axis=1).to_numpy()
    xfb_test = pd.concat([xf_test, buildings.transform(test)], axis=1).to_numpy()
    W_fuzzy = lr.fit(xf_train.to_numpy(), y_train_log)
    W_fuzzy_building = lr.fit(xfb_train, y_train_log)
    neighbour_count, neighbour_scan = choose_neighbour_count(train, clustering, groups=groups)
    neighbourhood = NeighbourhoodPrice.fit(train, neighbour_count)
    xn_train_frame = feature_frame(train, clustering, buildings=buildings.transform(train),
                                   neighbourhood=neighbourhood.transform(train))
    xn_train = xn_train_frame.to_numpy()
    xn_test = feature_frame(test, clustering, buildings=buildings.transform(test),
                            neighbourhood=neighbourhood.transform(test)).to_numpy()
    W_neighbour = lr.fit(xn_train, y_train_log)
    network_neighbour = FittedFfbp.fit_with_neighbourhood(train, clustering, neighbour_count, groups=groups)
    boosting_neighbour = HistGradientBoostingRegressor(random_state=SEED).fit(xn_train, y_train_log)
    partner, blend_scores = choose_blend_partner(train, clustering, neighbour_count, network_neighbour.setting, groups=groups)
    log_lr_neighbour = lr.predict(xn_test, W_neighbour)
    log_partner = (np.log(network_neighbour.predict(xn_test)) if partner == "FFBP"
                   else boosting_neighbour.predict(xn_test))
    blend_name = BLEND.format(partner=partner)

    # Extended building features + multi-scale IDW neighbourhood + CV-selected Ridge L2 & blend weight
    ext_settings = choose_extended_settings(train, clustering, groups=groups)
    ext_buildings = ExtendedBuildingFeatures.fit(train)
    ms_neighbourhood = MultiScaleNeighbourhood.fit(train)
    xe_train_frame = feature_frame(train, clustering, buildings=ext_buildings.transform(train),
                                   neighbourhood=ms_neighbourhood.transform(train))
    xe_train = xe_train_frame.to_numpy()
    xe_test = feature_frame(test, clustering, buildings=ext_buildings.transform(test),
                            neighbourhood=ms_neighbourhood.transform(test)).to_numpy()
    W_extended = lr.fit(xe_train, y_train_log, l2=ext_settings["l2"])
    gb_extended = make_extended_gb().fit(xe_train, y_train_ppsqm_log)
    rf_extended = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1).fit(
        xe_train, y_train_ppsqm_log
    )
    log_lr_extended = lr.predict(xe_test, W_extended)
    log_gb_extended = gb_extended.predict(xe_test) + log_area_test
    log_rf_extended = rf_extended.predict(xe_test) + log_area_test
    w_lr = ext_settings["w_lr"]
    log_blend_extended = w_lr * log_lr_extended + (1.0 - w_lr) * log_gb_extended

    xep_train_frame = feature_frame(train, clustering, project_levels=levels.transform(train, leave_one_out=True),
                                    buildings=ext_buildings.transform(train),
                                    neighbourhood=ms_neighbourhood.transform(train))
    xep_test = feature_frame(test, clustering, project_levels=levels.transform(test, leave_one_out=False),
                             buildings=ext_buildings.transform(test),
                             neighbourhood=ms_neighbourhood.transform(test)).to_numpy()
    W_extended_project = lr.fit(xep_train_frame.to_numpy(), y_train_log, l2=ext_settings["l2"])

    network = FittedFfbp.fit(x_train, y_train_log, groups=groups)
    network_building = FittedFfbp.fit(xb_train, y_train_log, groups=groups)
    boosting = HistGradientBoostingRegressor(random_state=SEED).fit(x_train, y_train_log)
    boosting_project = HistGradientBoostingRegressor(random_state=SEED).fit(xp_train, y_train_log)
    forest = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(x_train, y_train_log)
    forest_project = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(xp_train, y_train_log)
    zone_price = train.groupby("zone_no")["price_per_sqm"].median()
    filter_price, filter_match_pct = website_filter_prediction(train, test)

    predictions = {
        "Website filter: same zone, bedrooms, area +-10%": filter_price,
        "Baseline: zone median price/sqm x area": test["zone_no"].map(zone_price).to_numpy() * test["area_sqm"].to_numpy(),
        "Linear Regression (no location)": np.exp(lr.predict(x_test_no_location, W_no_location)),
        MAIN_LR: np.exp(lr.predict(x_test, W)),
        MAIN_KNN: knn.predict(x_test),
        MAIN_FFBP: network.predict(x_test),
        "Random Forest (K-Means tiers, reference)": np.exp(forest.predict(x_test)),
        "Gradient Boosting (K-Means tiers, reference)": np.exp(boosting.predict(x_test)),
        PROJECT_LR: np.exp(lr.predict(xp_test, W_project)),
        "KNN Regression + project price level": knn_project.predict(xp_test),
        "Random Forest + project price level (reference)": np.exp(forest_project.predict(xp_test)),
        "Gradient Boosting + project price level (reference)": np.exp(boosting_project.predict(xp_test)),
        FUZZY_LR: np.exp(lr.predict(xf_test.to_numpy(), W_fuzzy)),
        BUILDING_LR: np.exp(lr.predict(xb_test, W_building)),
        FUZZY_BUILDING_LR: np.exp(lr.predict(xfb_test, W_fuzzy_building)),
        NEIGHBOUR_LR: np.exp(log_lr_neighbour),
        NEIGHBOUR_FFBP: network_neighbour.predict(xn_test),
        NEIGHBOUR_GB: np.exp(boosting_neighbour.predict(xn_test)),
        blend_name: np.exp((log_lr_neighbour + log_partner) / 2),
        EXTENDED_LR: np.exp(log_lr_extended),
        EXTENDED_GB: np.exp(log_gb_extended),
        EXTENDED_RF: np.exp(log_rf_extended),
        EXTENDED_BLEND: np.exp(log_blend_extended),
        BUILDING_KNN: knn_building.predict(xb_test),
        BUILDING_FFBP: network_building.predict(xb_test),
        "Random Forest + building features (reference)": np.exp(forest_building.predict(xb_test)),
        "Gradient Boosting + building features (reference)": np.exp(boosting_building.predict(xb_test)),
        BUILDING_PROJECT_LR: np.exp(lr.predict(xbp_test, W_building_project)),
        EXTENDED_PROJECT_LR: np.exp(lr.predict(xep_test, W_extended_project)),
        "Gradient Boosting + building + project price level (reference)": np.exp(boosting_building_project.predict(xbp_test)),
    }
    knn_experiment = [
        {"metric": model.metric, "weight": model.weight, "k": model.k, "cv_mape_pct": model.cv_mape_pct,
         "test_mape_pct": price_metrics(y_test, model.predict(x_test))["mape_pct"]}
        for model in knn_candidates
    ]
    return {
        "train": train, "test": test, "clustering": clustering, "levels": levels,
        "W": W, "W_project": W_project, "knn": knn, "knn_project": knn_project,
        "W_building": W_building, "W_building_project": W_building_project, "knn_building": knn_building,
        "ffbp": {name: {"hidden": model.setting["H"], "epochs": model.setting["epochs"], "cv_mape_pct": model.cv_mape_pct}
                 for name, model in ((MAIN_FFBP, network), (BUILDING_FFBP, network_building), (NEIGHBOUR_FFBP, network_neighbour))},
        "buildings": buildings, "building_feature_names": list(xb_train_frame.columns),
        "memberships": memberships,
        "neighbourhood": neighbourhood, "neighbour_scan": neighbour_scan, "W_neighbour": W_neighbour,
        "neighbour_feature_names": list(xn_train_frame.columns),
        "network_neighbour": network_neighbour, "boosting_neighbour": boosting_neighbour,
        "blend": {"name": blend_name, "partner": partner, "cv_mape_pct": blend_scores, "neighbour_k": neighbour_count},
        "extended_buildings": ext_buildings, "multiscale_neighbourhood": ms_neighbourhood,
        "extended_feature_names": list(xe_train_frame.columns),
        "W_extended": W_extended, "gb_extended": gb_extended,
        "extended_blend": ext_settings,
        "W_extended_project": W_extended_project,
        "extended_project_feature_names": list(xep_train_frame.columns),
        "building_project_feature_names": list(xbp_train_frame.columns),
        "residual_std_log_building": float(np.std(y_train_log - lr.predict(xb_train, W_building))),
        "residual_std_log_building_project": float(np.std(y_train_log - lr.predict(xbp_train, W_building_project))),
        "test_building_share_pct": float(100 * test["has_building_info"].mean()),
        "feature_names": list(feature_frame(train, clustering).columns),
        "project_feature_names": list(xp_train_frame.columns),
        "predictions": predictions,
        "results": {name: price_metrics(y_test, predicted) for name, predicted in predictions.items()},
        "sota_style": {name: sota_style_metrics(test, predicted) for name, predicted in predictions.items()},
        "filter_match_pct": filter_match_pct,
        "kmeans_experiment": kmeans_experiment,
        "knn_experiment": knn_experiment,
        "y_test": y_test,
        "residual_std_log": float(np.std(y_train_log - lr.predict(x_train, W))),
        "residual_std_log_project": float(np.std(y_train_log - lr.predict(xp_train, W_project))),
    }


def learning_curve(train: pd.DataFrame, test: pd.DataFrame, clustering: ZoneClustering) -> list[dict[str, float]]:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train))
    x_test = feature_frame(test, clustering).to_numpy()
    rows = []
    for fraction in LEARNING_FRACTIONS:
        subset = train.iloc[order[: int(fraction * len(train))]]
        subset = subset[subset["zone_no"].isin(clustering.zone_table["zone_no"])]
        W = lr.fit(feature_frame(subset, clustering).to_numpy(), np.log(subset["price_thb"].to_numpy()))
        predicted = np.exp(lr.predict(x_test, W))
        rows.append({"fraction": fraction, "train_rows": len(subset),
                     "mape_pct": price_metrics(test["price_thb"].to_numpy(), predicted)["mape_pct"]})
    return rows




def style_axes(axes: plt.Axes, title: str) -> None:
    axes.set_facecolor("#000000")
    axes.set_title(title.upper(), color=WHITE, fontsize=12, loc="left")
    axes.tick_params(colors=ASH)
    for spine in axes.spines.values():
        spine.set_color(CHARCOAL)
    axes.xaxis.label.set_color(ASH)
    axes.yaxis.label.set_color(ASH)


def new_figure(width: float = 8, height: float = 5) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(width, height), facecolor="#000000")


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / name, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_cluster_scan(clustering: ZoneClustering) -> None:
    scan = pd.DataFrame(clustering.scan)
    figure, axes = new_figure()
    axes.plot(scan["k"], scan["cv_mape_pct"], color=GOLD, marker="o", label="LR cross-validated MAPE")
    twin = axes.twinx()
    twin.plot(scan["k"], scan["silhouette"], color=WHITE, marker="s", label="silhouette")
    twin.tick_params(colors=WHITE)
    twin.set_ylabel("silhouette", color=WHITE)
    axes.axvline(clustering.n_clusters, color=GOLD, linestyle="--", linewidth=1)
    style_axes(axes, f"K-Means ({clustering.metric}) on 38 zones: chosen k = {clustering.n_clusters}")
    axes.set_xlabel("number of clusters k")
    axes.set_ylabel("cross-validated MAPE (%)", color=GOLD)
    save_figure(figure, "kmeans_k_selection.png")


def plot_zone_tiers(clustering: ZoneClustering) -> None:
    table = clustering.zone_table.sort_values("median_price_per_sqm")
    figure, axes = new_figure(9, max(4, 0.28 * len(table)))
    shades = plt.cm.YlOrBr(np.linspace(0.3, 0.95, max(clustering.n_clusters, 1)))
    axes.barh(table["zone_slug"], table["median_price_per_sqm"] / 1000, color=[shades[t - 1] for t in table["tier"]])
    for position, (value, tier) in enumerate(zip(table["median_price_per_sqm"] / 1000, table["tier"])):
        axes.text(value, position, f"  T{tier}", va="center", color=WHITE, fontsize=7)
    style_axes(axes, "Median asking price per sqm by zone (T = K-Means tier)")
    axes.set_xlabel("thousand THB per sqm")
    axes.tick_params(axis="y", labelsize=7)
    save_figure(figure, "zone_tiers.png")


def plot_actual_vs_predicted(actual: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    figure, axes = new_figure(7, 7)
    for name, color in ((EXTENDED_LR, GOLD), (EXTENDED_BLEND, WHITE)):
        axes.scatter(actual / 1e6, predictions[name] / 1e6, s=8, alpha=0.45, color=color, label=name)
    limit = np.percentile(actual / 1e6, 98)
    axes.plot([0, limit], [0, limit], color=ASH, linestyle="--")
    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    style_axes(axes, "Actual vs predicted price (random-split test set)")
    axes.set_xlabel("actual price (million THB)")
    axes.set_ylabel("predicted price (million THB)")
    axes.legend(facecolor=CHARCOAL, labelcolor=WHITE, edgecolor=CHARCOAL, fontsize=8)
    save_figure(figure, "actual_vs_predicted.png")


def plot_knn_curve(curve: list[dict[str, float]], best_k: int) -> None:
    table = pd.DataFrame(curve)
    figure, axes = new_figure()
    axes.plot(table["k"], table["cv_mape_pct"], color=WHITE, marker="o", markersize=3)
    axes.axvline(best_k, color=GOLD, linestyle="--")
    style_axes(axes, f"KNN: 5-fold CV error by k (chosen k = {best_k})")
    axes.set_xlabel("k neighbours")
    axes.set_ylabel("CV MAPE (%)")
    save_figure(figure, "knn_k_selection.png")


def plot_model_comparison(random_results: dict[str, dict[str, float]], project_results: dict[str, dict[str, float]]) -> None:
    names = list(random_results)
    positions = np.arange(len(names))
    figure, axes = new_figure(12, 0.42 * len(names) + 1.6)
    rand_vals = [random_results[n]["mape_pct"] for n in names]
    proj_vals = [project_results[n]["mape_pct"] for n in names]
    axes.barh(positions - 0.2, rand_vals, height=0.4, color=GOLD, label="random split")
    axes.barh(positions + 0.2, proj_vals, height=0.4, color=ASH, label="unseen buildings")
    for position, r_val, p_val in zip(positions, rand_vals, proj_vals):
        axes.text(r_val + 0.3, position - 0.2, f"{r_val:.1f}%", va="center", color=WHITE, fontsize=8)
        axes.text(p_val + 0.3, position + 0.2, f"{p_val:.1f}%", va="center", color=WHITE, fontsize=8)
    axes.set_xlim(0, max(rand_vals + proj_vals) * 1.15)
    axes.set_yticks(positions, names, fontsize=8)
    axes.invert_yaxis()
    style_axes(axes, "Test MAPE (%) by model (lower is better)")
    axes.set_xlabel("MAPE (%)")
    axes.legend(facecolor=CHARCOAL, labelcolor=WHITE, edgecolor=CHARCOAL, loc="upper center",
                bbox_to_anchor=(0.5, -0.10), ncol=2)
    save_figure(figure, "model_comparison.png")


def plot_knn_experiment(random_rows: list[dict[str, object]], project_rows: list[dict[str, object]]) -> None:
    labels = [f"{row['metric']} / {row['weight']} vote (k={row['k']})" for row in random_rows]
    positions = np.arange(len(labels))
    figure, axes = new_figure(9, 4)
    axes.barh(positions - 0.2, [row["test_mape_pct"] for row in random_rows], height=0.4, color=GOLD, label="random split")
    axes.barh(positions + 0.2, [row["test_mape_pct"] for row in project_rows], height=0.4, color=ASH, label="unseen buildings")
    for position, random_row, project_row in zip(positions, random_rows, project_rows):
        axes.text(random_row["test_mape_pct"], position - 0.2, f" {random_row['test_mape_pct']:.1f}%", va="center", color=WHITE, fontsize=8)
        axes.text(project_row["test_mape_pct"], position + 0.2, f" {project_row['test_mape_pct']:.1f}%", va="center", color=WHITE, fontsize=8)
    axes.set_yticks(positions, labels, fontsize=9)
    axes.invert_yaxis()
    style_axes(axes, "KNN: distance and voting rule (test MAPE)")
    axes.set_xlabel("MAPE (%)")
    axes.legend(facecolor=CHARCOAL, labelcolor=WHITE, edgecolor=CHARCOAL, loc="upper center",
                bbox_to_anchor=(0.5, -0.18), ncol=2)
    save_figure(figure, "knn_distance_voting.png")


def zone_errors(test: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    """Error of one model per zone: median signed error shows if a zone is priced too low or too
    high on average, MAPE shows how far off a single unit usually is."""
    error = (predicted - test["price_thb"].to_numpy()) / test["price_thb"].to_numpy()
    frame = pd.DataFrame({"zone_no": test["zone_no"], "zone_slug": test["zone_slug"], "error": error})
    return frame.groupby(["zone_no", "zone_slug"]).agg(
        test_units=("error", "size"),
        median_signed_error_pct=("error", lambda e: float(100 * np.median(e))),
        mape_pct=("error", lambda e: float(100 * np.mean(np.abs(e)))),
        within_20pct=("error", lambda e: float(100 * np.mean(np.abs(e) <= 0.2))),
    ).reset_index()


def price_band_errors(predicted: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    error = (predicted - actual) / actual
    band = pd.cut(predicted, PRICE_BANDS, labels=PRICE_BAND_LABELS, right=False)
    return pd.DataFrame({"price_band": band, "error": error}).groupby("price_band", observed=False).agg(
        test_units=("error", "size"),
        median_signed_error_pct=("error", lambda e: float(100 * np.median(e)) if len(e) else np.nan),
        mape_pct=("error", lambda e: float(100 * np.mean(np.abs(e))) if len(e) else np.nan),
        within_20pct=("error", lambda e: float(100 * np.mean(np.abs(e) <= 0.2)) if len(e) else np.nan),
    ).reset_index()


def reliability(test: pd.DataFrame, predicted: np.ndarray) -> dict[str, object]:
    """Test-set accuracy of one model, overall, per zone and per predicted price band, so the web
    app can say how often a price like this one landed within 20% of the real price."""
    actual = test["price_thb"].to_numpy()
    error = np.abs(predicted - actual) / actual
    return {
        "test_units": len(test),
        "mape_pct": float(100 * error.mean()),
        "within_20pct": float(100 * np.mean(error <= 0.2)),
        "zones": zone_errors(test, predicted),
        "bands": price_band_errors(predicted, actual),
    }


def fuzzy_summary(memberships: pd.DataFrame, clustering: ZoneClustering) -> dict[str, object]:
    columns = [column for column in memberships.columns if column.startswith("member_")]
    strongest = memberships[columns].max(axis=1)
    fuzzy_tier = memberships[columns].to_numpy().argmax(axis=1) + 1
    kmeans_tier = clustering.tier_of(memberships["zone_no"]).to_numpy()
    return {
        "k": len(columns), "m": 2,
        "zones_same_tier_as_kmeans": int((fuzzy_tier == kmeans_tier).sum()),
        "zones": len(memberships),
        # a zone whose largest membership is below 0.6 sits between price levels
        "zones_between_tiers": memberships.loc[strongest < 0.6, "zone_slug"].tolist(),
    }


def plot_fuzzy_memberships(memberships: pd.DataFrame) -> None:
    columns = [column for column in memberships.columns if column.startswith("member_")]
    figure, axes = new_figure(9, max(4, 0.28 * len(memberships)))
    shades = plt.cm.YlOrBr(np.linspace(0.3, 0.95, len(columns)))
    left = np.zeros(len(memberships))
    for column, shade in zip(columns, shades):
        axes.barh(memberships["zone_slug"], memberships[column], left=left, color=shade, label=column.replace("member_", "tier "))
        left += memberships[column].to_numpy()
    style_axes(axes, "Fuzzy C-Means: membership of each zone in each tier")
    axes.set_xlabel("membership (sums to 1 per zone)")
    axes.tick_params(axis="y", labelsize=7)
    axes.legend(facecolor=CHARCOAL, labelcolor=WHITE, edgecolor=CHARCOAL, loc="upper center",
                bbox_to_anchor=(0.5, -0.06), ncol=len(columns), fontsize=7)
    save_figure(figure, "fuzzy_memberships.png")


def plot_learning_curve(rows: list[dict[str, float]]) -> None:
    table = pd.DataFrame(rows)
    figure, axes = new_figure()
    axes.plot(table["train_rows"], table["mape_pct"], color=GOLD, marker="o")
    style_axes(axes, "Learning curve: Linear Regression error vs training size")
    axes.set_xlabel("training listings")
    axes.set_ylabel("test MAPE (%)")
    save_figure(figure, "learning_curve.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    all_listings = pd.read_csv(CLEAN_CSV, dtype={"listing_id": str})
    # a cleared project name (zone name typed as project) is read back from CSV as NaN
    all_listings["project_name"] = all_listings["project_name"].fillna("")
    projects = pd.read_csv(PROJECTS_CSV)
    # Same collection rule for every zone: zone 01 was saved to 38 pages, others to 2,
    # so without this cap one zone would dominate the training data.
    listings = all_listings[all_listings["page"] <= MAX_PAGES_PER_ZONE].reset_index(drop=True)
    listings = attach_buildings(listings, projects)
    LOG.info("building data found for %.1f%% of listings", 100 * listings["has_building_info"].mean())
    LOG.info("using %s of %s clean listings (pages <= %s per zone)", len(listings), len(all_listings), MAX_PAGES_PER_ZONE)

    main_run = evaluate_split(*split_random(listings), group_cv=False)
    project_run = evaluate_split(*split_by_project(listings), group_cv=True)
    curve = learning_curve(main_run["train"], main_run["test"], main_run["clustering"])

    clustering = main_run["clustering"]
    knn = main_run["knn"]
    coefficients = {
        name: {"coef_log": float(coef), "effect_pct": float(100 * (np.exp(coef) - 1))}
        for name, coef in zip(main_run["feature_names"], main_run["W"][:-1])
    }
    metrics = {
        "rows": {
            "clean_all_pages": len(all_listings), "used": len(listings),
            "train_random": len(main_run["train"]), "test_random": len(main_run["test"]),
            "train_project": len(project_run["train"]), "test_project": len(project_run["test"]),
            "zones": int(listings["zone_no"].nunique()), "projects": int(listings["project_name"].nunique()),
            "listings_with_building_info": int(listings["has_building_info"].sum()),
            "projects_with_building_info": int(listings.loc[listings["has_building_info"], "project_name"].nunique()),
        },
        "kmeans": {"k": clustering.n_clusters, "metric": clustering.metric, "scan": clustering.scan},
        "knn": {"k": knn.k, "metric": knn.metric, "weight": knn.weight, "cv_curve": knn.curve,
                "k_project": main_run["knn_project"].k},
        "distance_experiment": {
            "kmeans_random_split": main_run["kmeans_experiment"],
            "kmeans_project_split": project_run["kmeans_experiment"],
            "knn_random_split": main_run["knn_experiment"],
            "knn_project_split": project_run["knn_experiment"],
        },
        "fuzzy_cmeans": fuzzy_summary(main_run["memberships"], clustering),
        "neighbourhood_random_split": {"k": main_run["blend"]["neighbour_k"], "scan": main_run["neighbour_scan"]},
        "neighbourhood_project_split": {"k": project_run["blend"]["neighbour_k"], "scan": project_run["neighbour_scan"]},
        "blend_random_split": main_run["blend"],
        "blend_project_split": project_run["blend"],
        "extended_blend_random_split": main_run["extended_blend"],
        "extended_blend_project_split": project_run["extended_blend"],
        "ffbp_random_split": main_run["ffbp"],
        "ffbp_project_split": project_run["ffbp"],
        "features": main_run["feature_names"],
        "building_features": main_run["building_feature_names"],
        "extended_features": main_run["extended_feature_names"],
        "building_linear_regression": {
            name: {"coef_log": float(coef), "effect_pct": float(100 * (np.exp(coef) - 1))}
            for name, coef in zip(main_run["building_feature_names"], main_run["W_building"][:-1])
        },
        "extended_linear_regression": {
            name: {"coef_log": float(coef), "effect_pct": float(100 * (np.exp(coef) - 1))}
            for name, coef in zip(main_run["extended_feature_names"], main_run["W_extended"][:-1])
        },
        "linear_regression": {"intercept_log": float(main_run["W"][-1]), "coefficients": coefficients},
        "test_results_random_split": main_run["results"],
        "test_results_project_split": project_run["results"],
        "sota_style_random_split": main_run["sota_style"],
        "sota_style_project_split": project_run["sota_style"],
        "website_filter_match_pct": {"random_split": main_run["filter_match_pct"],
                                     "project_split": project_run["filter_match_pct"]},
        "learning_curve": curve,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clustering.zone_table.to_csv(REPORTS_DIR / "zone_clusters.csv", index=False)
    memberships = main_run["memberships"].merge(clustering.zone_table[["zone_no", "tier"]], on="zone_no")
    memberships.sort_values("median_price_per_sqm").to_csv(REPORTS_DIR / "zone_fuzzy_memberships.csv", index=False)
    plot_fuzzy_memberships(memberships)
    for split_name, run in (("random_split", main_run), ("project_split", project_run)):
        errors = zone_errors(run["test"], run["predictions"][EXTENDED_BLEND])
        errors.to_csv(REPORTS_DIR / f"zone_errors_{split_name}.csv", index=False)

    plot_cluster_scan(clustering)
    plot_zone_tiers(clustering)
    plot_actual_vs_predicted(main_run["y_test"], main_run["predictions"])
    plot_knn_curve(knn.curve, knn.k)
    plot_model_comparison(main_run["results"], project_run["results"])
    plot_learning_curve(curve)
    plot_knn_experiment(main_run["knn_experiment"], project_run["knn_experiment"])

    # Fit the unseen-building extended blend on main_run["train"] with the GroupCV-selected
    # hyperparameters from project_run so deployed weights and web_reliability match 100%.
    unseen_l2 = project_run["extended_blend"]["l2"]
    unseen_w_lr = project_run["extended_blend"]["w_lr"]
    xe_web_frame = feature_frame(
        main_run["train"], clustering,
        buildings=main_run["extended_buildings"].transform(main_run["train"]),
        neighbourhood=main_run["multiscale_neighbourhood"].transform(main_run["train"]),
    )
    W_unseen_web = lr.fit(xe_web_frame.to_numpy(), np.log(main_run["train"]["price_thb"].to_numpy()), l2=unseen_l2)
    gb_unseen_web = make_extended_gb().fit(xe_web_frame.to_numpy(), np.log(main_run["train"]["price_per_sqm"].to_numpy()))

    # Known-building model for the web app: the building's own listings (including pages beyond
    # the per-zone cap, which only move their own building's level), shrunk toward the
    # unseen-building blend above and adjusted for floor and size within the building.
    def unseen_log_ppsqm(frame: pd.DataFrame) -> np.ndarray:
        x = feature_frame(frame, clustering, buildings=main_run["extended_buildings"].transform(frame),
                          neighbourhood=main_run["multiscale_neighbourhood"].transform(frame))[xe_web_frame.columns].to_numpy()
        log_lr = lr.predict(x, W_unseen_web) - np.log(frame["area_sqm"].to_numpy())
        return unseen_w_lr * log_lr + (1.0 - unseen_w_lr) * gb_unseen_web.predict(x)

    extra_history = attach_buildings(all_listings[all_listings["page"] > MAX_PAGES_PER_ZONE].reset_index(drop=True), projects)
    extra_history = extra_history[~extra_history["listing_id"].isin(main_run["test"]["listing_id"])]
    known_model = KnownBuildingPrice.fit(main_run["train"], unseen_log_ppsqm(main_run["train"]), extra_history)
    known_test_price = known_model.predict(main_run["test"], unseen_log_ppsqm(main_run["test"]))
    known = np.isfinite(known_test_price)
    web_reliability = {
        "project_level": reliability(main_run["test"][known], known_test_price[known]),
        # a building the app cannot price from its own listings is an unseen building: the
        # by-project split measures the exact extended unseen-building blend deployed in the app
        "blend_unseen": reliability(project_run["test"], project_run["predictions"][EXTENDED_BLEND]),
    }
    for name, table in web_reliability.items():
        table["bands"].to_csv(REPORTS_DIR / f"price_band_errors_{name}.csv", index=False)
    metrics["web_reliability"] = {name: {key: value for key, value in table.items() if key not in ("zones", "bands")}
                                  for name, table in web_reliability.items()}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    # Exclude fake unlinked project names that equal a zone name (no real project page)
    zone_names = set(listings["zone_name_th"].dropna().str.strip())
    real_listing_projects = {
        name.strip() for name in main_run["train"].loc[main_run["train"]["has_building_info"], "project_name"].dropna()
    }
    parsed_projects = {name.strip() for name in projects["project_name"].dropna()}
    clean_project_names = sorted((real_listing_projects | parsed_projects) - zone_names - {""})
    # Map each project key -> most frequent zone_no in listings so the web app can detect zone mismatches
    project_zones = (
        listings.assign(_key=project_key(listings["project_name"]))
        .groupby("_key")["zone_no"]
        .agg(lambda s: int(s.mode().iloc[0]))
        .to_dict()
    )

    reference_rows = main_run["train"][["project_name", "zone_name_th", "district_th", "room_type", "bedrooms",
                                        "bathrooms", "floor", "area_sqm", "price_thb"]].reset_index(drop=True)
    bundle = {
        "zone_table": clustering.zone_table,
        "n_clusters": clustering.n_clusters,
        "feature_names": main_run["feature_names"],
        "project_feature_names": main_run["project_feature_names"],
        "W": main_run["W"],
        "W_project": main_run["W_project"],
        "project_levels": main_run["levels"],
        # plain values, not the FittedKnn object: run.py executes as __main__, so a pickled
        # instance of its class could not be loaded by the web app
        "knn": {"X": knn.X, "y": knn.y, "scaler": knn.scaler, "k": knn.k, "metric": knn.metric, "weight": knn.weight},
        # earlier main web model (building features only), kept for reference
        "W_building": main_run["W_building"],
        "W_building_project": main_run["W_building_project"],
        "building_features": main_run["buildings"],
        "building_feature_names": main_run["building_feature_names"],
        "building_project_feature_names": main_run["building_project_feature_names"],
        "residual_std_log_building": main_run["residual_std_log_building"],
        "residual_std_log_building_project": main_run["residual_std_log_building_project"],
        "blend": {
            "name": EXTENDED_BLEND,
            "partner": "Gradient Boosting",
            "mode": "extended_multiscale",
            "w_lr": unseen_w_lr,
            "l2": unseen_l2,
            "extended_buildings": main_run["extended_buildings"],
            "multiscale_neighbourhood": main_run["multiscale_neighbourhood"],
            "W_extended": W_unseen_web,
            "gb_extended": gb_unseen_web,
            "feature_names": list(xe_web_frame.columns),
            # keep legacy single-scale neighbourhood object for reference / backward compatibility
            "neighbourhood": main_run["neighbourhood"],
        },
        "known_building": known_model,
        "reliability": web_reliability,
        "knn_building": {key: getattr(main_run["knn_building"], key) for key in ("X", "y", "scaler", "k", "metric", "weight")},
        "projects": projects,
        "project_zones": project_zones,
        "reference_rows": reference_rows,
        "project_names": clean_project_names,
        "test_results": main_run["results"],
        "test_results_project_split": project_run["results"],
        "residual_std_log": main_run["residual_std_log"],
        "residual_std_log_project": main_run["residual_std_log_project"],
    }
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BUNDLE_PATH.open("wb") as handle:
        pickle.dump(bundle, handle)
    for name in main_run["results"]:
        LOG.info("%-52s random MAPE %5.1f%% | unseen buildings MAPE %5.1f%%", name,
                 main_run["results"][name]["mape_pct"], project_run["results"][name]["mape_pct"])


if __name__ == "__main__":
    main()
