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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GroupKFold, KFold

from src.features.build import (
    SEED,
    BuildingFeatures,
    ProjectPriceLevels,
    RichFeatureBuilder,
    Standardizer,
    ZoneClustering,
    attach_rich_buildings,
    choose_zone_clustering,
    cluster_zones,
    feature_frame,
    fuzzy_zone_memberships,
    mean_absolute_percentage_error,
    membership_frame,
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
FFBP_SETTINGS = [{"H": H, "epochs": epochs} for H in (4, 8, 16) for epochs in (3000, 10000)]
GOLD = "#FFC000"
WHITE = "#FFFFFF"
ASH = "#7D7D7D"
CHARCOAL = "#202020"

MAIN_LR = "Linear Regression (K-Means tiers)"
MAIN_KNN = "KNN Regression (K-Means tiers)"
PROJECT_LR = "Linear Regression + project price level"
BUILDING_LR = "Linear Regression + building features"
BUILDING_KNN = "KNN Regression + building features"
MAIN_FFBP = "FFBP neural network (K-Means tiers)"
BUILDING_FFBP = "FFBP neural network + building features"
FUZZY_LR = "Linear Regression (Fuzzy C-Means memberships)"
FUZZY_BUILDING_LR = "Linear Regression + building features (Fuzzy C-Means memberships)"
STABLE_RIDGE = "Ridge Regression + rich spatial/building (NumPy)"
STABLE_HUBER = "Huber Regression + rich spatial/building"
STABLE_FFBP = "FFBP Stabilized Ensemble + rich spatial/building (NumPy)"
STABLE_HGB = "Gradient Boosting (L1, 5-seed) + rich spatial/building"
STABLE_ENSEMBLE = "Stabilized Ensemble (CV-weighted, log price/sqm)"

RIDGE_ALPHAS = (5.0, 15.0, 40.0, 80.0, 150.0)
STABILITY_SEEDS = (42, 7, 13, 21, 99)
STABLE_HGB_CONFIG = {
    "max_iter": 350,
    "learning_rate": 0.035,
    "max_leaf_nodes": 16,
    "max_depth": 5,
    "min_samples_leaf": 20,
    "l2_regularization": 6.0,
    "max_features": 0.75,
}


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


def choose_knn_k(features: np.ndarray, log_price: np.ndarray, metric: str, weight: str) -> tuple[int, list[dict[str, float]]]:
    folds = list(KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(features))
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
    def fit(cls, features: np.ndarray, log_price: np.ndarray, metric: str, weight: str) -> FittedKnn:
        k, curve = choose_knn_k(features, log_price, metric, weight)
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
    def fit(cls, features: np.ndarray, log_price: np.ndarray) -> FittedFfbp:
        folds = list(KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(features))
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

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.exp(ffbp.predict(self.scaler.transform(features), self.W0, self.W1) * self.y_std + self.y_mean)


def best_zone_clustering(train: pd.DataFrame) -> tuple[ZoneClustering, list[dict[str, object]]]:
    candidates = [choose_zone_clustering(train, metric) for metric in METRICS]
    rows = []
    for clustering in candidates:
        chosen = next(row for row in clustering.scan if row["k"] == clustering.n_clusters)
        rows.append({"metric": clustering.metric, "k": clustering.n_clusters,
                     "cv_mape_pct": chosen["cv_mape_pct"], "silhouette": chosen["silhouette"]})
    best = min(candidates, key=lambda clustering: next(
        row["cv_mape_pct"] for row in clustering.scan if row["k"] == clustering.n_clusters))
    return best, rows


def best_knn(features: np.ndarray, log_price: np.ndarray) -> tuple[FittedKnn, list[FittedKnn]]:
    candidates = [FittedKnn.fit(features, log_price, metric, weight) for metric in METRICS for weight in WEIGHTS]
    return min(candidates, key=lambda model: model.cv_mape_pct), candidates


def price_from_log_pps(log_pps: np.ndarray, area_sqm: np.ndarray) -> np.ndarray:
    return np.exp(log_pps) * area_sqm


def fit_stabilized_suite(
    train: pd.DataFrame,
    test: pd.DataFrame,
    clustering: ZoneClustering,
    n_seeds: int = 5,
) -> dict[str, object]:
    """Train stabilized models on log(price_per_sqm) using leak-free GroupKFold CV on train."""
    LOG.info("fitting stabilized suite (%d train rows, %d test rows)...", len(train), len(test))
    y_train_pps = np.log(train["price_per_sqm"].to_numpy(dtype=float))
    area_train = train["area_sqm"].to_numpy(dtype=float)
    price_train = train["price_thb"].to_numpy(dtype=float)
    area_test = test["area_sqm"].to_numpy(dtype=float)

    folds = list(GroupKFold(n_splits=CV_FOLDS).split(train, groups=train["project_name"]))
    fold_data = []
    for train_index, valid_index in folds:
        f_tr = train.iloc[train_index].reset_index(drop=True)
        f_va = train.iloc[valid_index].reset_index(drop=True)
        f_cl, _, _ = cluster_zones(zone_statistics(f_tr), clustering.n_clusters, metric=clustering.metric)
        fb = RichFeatureBuilder.fit(f_tr, f_cl)
        X_tr = fb.transform(f_tr, leave_one_project_out=True).to_numpy()
        X_va = fb.transform(f_va, leave_one_project_out=False).to_numpy()
        sc = Standardizer.fit(X_tr)
        fold_data.append((train_index, valid_index, X_tr, X_va, sc.transform(X_tr), sc.transform(X_va)))

    best_alpha, best_alpha_mape = RIDGE_ALPHAS[0], float("inf")
    for alpha in RIDGE_ALPHAS:
        fold_errors = [
            mean_absolute_percentage_error(
                price_train[va],
                price_from_log_pps(lr.predict(X_va_s, lr.fit_ridge(X_tr_s, y_train_pps[tr], alpha=alpha)), area_train[va]),
            )
            for tr, va, _, _, X_tr_s, X_va_s in fold_data
        ]
        score = float(np.mean(fold_errors))
        if score < best_alpha_mape:
            best_alpha, best_alpha_mape = float(alpha), score

    model_names = ["ridge", "huber", "ffbp", "hgb_l1", "hgb_l2", "et"]
    oofs = {name: np.zeros(len(train)) for name in model_names}
    for tr, va, X_tr, X_va, X_tr_s, X_va_s in fold_data:
        yt = y_train_pps[tr]
        ym, ys = float(yt.mean()), float(yt.std())
        oofs["ridge"][va] = lr.predict(X_va_s, lr.fit_ridge(X_tr_s, yt, alpha=best_alpha))
        oofs["huber"][va] = HuberRegressor(alpha=best_alpha, max_iter=500).fit(X_tr_s, yt).predict(X_va_s)
        oofs["ffbp"][va] = np.mean([
            ffbp.predict(X_va_s, *ffbp.ffbp_stable(X_tr_s, (yt - ym) / ys, H=12, epochs=2200, seed=SEED + s)) * ys + ym
            for s in range(3)
        ], axis=0)
        oofs["hgb_l1"][va] = HistGradientBoostingRegressor(
            loss="absolute_error", random_state=SEED, **STABLE_HGB_CONFIG
        ).fit(X_tr, yt).predict(X_va)
        oofs["hgb_l2"][va] = HistGradientBoostingRegressor(
            loss="squared_error", random_state=SEED, **STABLE_HGB_CONFIG
        ).fit(X_tr, yt).predict(X_va)
        oofs["et"][va] = ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=3, max_features=0.7, random_state=SEED, n_jobs=-1
        ).fit(X_tr, yt).predict(X_va)

    oof_matrix = np.column_stack([oofs[name] for name in model_names])
    weights_vec = np.ones(len(model_names)) / len(model_names)
    best_obj = mean_absolute_percentage_error(price_train, price_from_log_pps(oof_matrix @ weights_vec, area_train))
    rng = np.random.default_rng(SEED)
    for step_size in (0.15, 0.08, 0.03):
        for _ in range(400):
            candidate = np.clip(weights_vec + rng.normal(0.0, step_size, size=len(model_names)), 0.0, None)
            if candidate.sum() == 0:
                continue
            candidate /= candidate.sum()
            # Mild L2 weight diversity penalty reduces ensemble variance across splits
            obj = (
                mean_absolute_percentage_error(price_train, price_from_log_pps(oof_matrix @ candidate, area_train))
                + 0.15 * float(np.sum(candidate**2))
            )
            if obj < best_obj:
                best_obj, weights_vec = obj, candidate

    oof_blend_log_pps = oof_matrix @ weights_vec
    oof_mape_pct = mean_absolute_percentage_error(price_train, price_from_log_pps(oof_blend_log_pps, area_train))
    residual_std_log = float(np.std(y_train_pps - oof_blend_log_pps))
    weights = {name: float(weight) for name, weight in zip(model_names, weights_vec)}

    rich_builder = RichFeatureBuilder.fit(train, clustering)
    X_tr = rich_builder.transform(train, leave_one_project_out=True).to_numpy()
    X_te = rich_builder.transform(test, leave_one_project_out=False).to_numpy()
    scaler = Standardizer.fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    ffbp_y_mean, ffbp_y_std = float(y_train_pps.mean()), float(y_train_pps.std())

    ridge_W = lr.fit_ridge(X_tr_s, y_train_pps, alpha=best_alpha)
    huber_model = HuberRegressor(alpha=best_alpha, max_iter=500).fit(X_tr_s, y_train_pps)
    ffbp_models = [
        ffbp.ffbp_stable(X_tr_s, (y_train_pps - ffbp_y_mean) / ffbp_y_std, H=12, epochs=2500, seed=SEED + s)
        for s in range(n_seeds)
    ]
    hgb_l1_models = [
        HistGradientBoostingRegressor(loss="absolute_error", random_state=SEED + s, **STABLE_HGB_CONFIG).fit(X_tr, y_train_pps)
        for s in range(n_seeds)
    ]
    hgb_l2_models = [
        HistGradientBoostingRegressor(loss="squared_error", random_state=SEED + s, **STABLE_HGB_CONFIG).fit(X_tr, y_train_pps)
        for s in range(n_seeds)
    ]
    et_model = ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=0.7, random_state=SEED, n_jobs=-1
    ).fit(X_tr, y_train_pps)

    test_log_pps = {
        "ridge": lr.predict(X_te_s, ridge_W),
        "huber": huber_model.predict(X_te_s),
        "ffbp": np.mean([ffbp.predict(X_te_s, W0, W1) * ffbp_y_std + ffbp_y_mean for W0, W1 in ffbp_models], axis=0),
        "hgb_l1": np.mean([model.predict(X_te) for model in hgb_l1_models], axis=0),
        "hgb_l2": np.mean([model.predict(X_te) for model in hgb_l2_models], axis=0),
        "et": et_model.predict(X_te),
    }
    blend_log_pps = sum(weights[name] * test_log_pps[name] for name in model_names)

    return {
        "predictions": {
            STABLE_RIDGE: price_from_log_pps(test_log_pps["ridge"], area_test),
            STABLE_HUBER: price_from_log_pps(test_log_pps["huber"], area_test),
            STABLE_FFBP: price_from_log_pps(test_log_pps["ffbp"], area_test),
            STABLE_HGB: price_from_log_pps(test_log_pps["hgb_l1"], area_test),
            STABLE_ENSEMBLE: price_from_log_pps(blend_log_pps, area_test),
        },
        "meta": {
            "ridge_alpha": best_alpha,
            "oof_group_cv_mape_pct": oof_mape_pct,
            "residual_std_log": residual_std_log,
            "weights": weights,
            "n_seeds": n_seeds,
            "n_features": int(X_tr.shape[1]),
        },
        "bundle": {
            "rich_builder": rich_builder,
            "scaler": scaler,
            "ridge_W": ridge_W,
            "huber": huber_model,
            "ffbp_models": ffbp_models,
            "ffbp_y_mean": ffbp_y_mean,
            "ffbp_y_std": ffbp_y_std,
            "hgb_l1_models": hgb_l1_models,
            "hgb_l2_models": hgb_l2_models,
            "et_model": et_model,
            "weights": weights,
            "model_names": model_names,
            "alpha": best_alpha,
            "oof_mape_pct": oof_mape_pct,
            "residual_std_log": residual_std_log,
        },
    }


def plot_stability_benchmark(report: dict[str, object]) -> None:
    models = [
        BUILDING_LR,
        "Gradient Boosting + building features (reference)",
        STABLE_RIDGE,
        STABLE_FFBP,
        STABLE_HGB,
        STABLE_ENSEMBLE,
    ]
    short_labels = [
        "Baseline LR + bldg",
        "Baseline HGB + bldg",
        "Ridge + rich (NumPy)",
        "FFBP Stable + rich (NumPy)",
        "HGB L1 5-seed + rich",
        "Stabilized Ensemble",
    ]
    figure, axes_list = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="#000000", sharey=True)
    panels = [
        (axes_list[0], report["random_split"], "Random split (5 seeds)"),
        (axes_list[1], report["project_split"], "Unseen buildings (5 seeds)"),
    ]
    positions = np.arange(len(models))
    for axes, summary, title in panels:
        means = [summary[m]["mean_mape_pct"] for m in models]
        stds = [summary[m]["std_mape_pct"] for m in models]
        colors = [ASH, ASH, WHITE, WHITE, WHITE, GOLD]
        axes.barh(positions, means, xerr=stds, color=colors, height=0.58,
                  error_kw={"ecolor": "#FF6B6B", "elinewidth": 1.6, "capsize": 4})
        for pos, mean_v, std_v in zip(positions, means, stds):
            axes.text(mean_v + std_v + 0.35, pos, f"{mean_v:.1f}% ± {std_v:.2f}%", va="center", color=WHITE, fontsize=8)
        axes.set_yticks(positions, short_labels, fontsize=8.5)
        axes.invert_yaxis()
        axes.set_xlim(0, max(m + s for m, s in zip(means, stds)) + 6.5)
        style_axes(axes, f"Stability & accuracy: {title}")
        axes.set_xlabel("MAPE (%) — mean ± std across 5 splits (lower is better)")
    save_figure(figure, "model_stability.png")


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    clustering, kmeans_experiment = best_zone_clustering(train)
    test = test[test["zone_no"].isin(clustering.zone_table["zone_no"])].reset_index(drop=True)
    y_train_log = np.log(train["price_thb"].to_numpy())
    y_test = test["price_thb"].to_numpy()

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
    knn, knn_candidates = best_knn(x_train, y_train_log)
    knn_project = FittedKnn.fit(xp_train, y_train_log, knn.metric, knn.weight)
    W_building = lr.fit(xb_train, y_train_log)
    W_building_project = lr.fit(xbp_train, y_train_log)
    knn_building = FittedKnn.fit(xb_train, y_train_log, knn.metric, knn.weight)
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
    network = FittedFfbp.fit(x_train, y_train_log)
    network_building = FittedFfbp.fit(xb_train, y_train_log)
    boosting = HistGradientBoostingRegressor(random_state=SEED).fit(x_train, y_train_log)
    boosting_project = HistGradientBoostingRegressor(random_state=SEED).fit(xp_train, y_train_log)
    forest = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(x_train, y_train_log)
    forest_project = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(xp_train, y_train_log)
    zone_price = train.groupby("zone_no")["price_per_sqm"].median()
    filter_price, filter_match_pct = website_filter_prediction(train, test)
    stabilized = fit_stabilized_suite(train, test, clustering, n_seeds=5)

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
        BUILDING_KNN: knn_building.predict(xb_test),
        BUILDING_FFBP: network_building.predict(xb_test),
        "Random Forest + building features (reference)": np.exp(forest_building.predict(xb_test)),
        "Gradient Boosting + building features (reference)": np.exp(boosting_building.predict(xb_test)),
        "Linear Regression + building + project price level": np.exp(lr.predict(xbp_test, W_building_project)),
        "Gradient Boosting + building + project price level (reference)": np.exp(boosting_building_project.predict(xbp_test)),
        **stabilized["predictions"],
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
                 for name, model in ((MAIN_FFBP, network), (BUILDING_FFBP, network_building))},
        "buildings": buildings, "building_feature_names": list(xb_train_frame.columns),
        "memberships": memberships,
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
        "stabilized": stabilized,
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
    for name, color in ((MAIN_LR, GOLD), (MAIN_KNN, WHITE)):
        axes.scatter(actual / 1e6, predictions[name] / 1e6, s=8, alpha=0.45, color=color, label=name)
    limit = np.percentile(actual / 1e6, 98)
    axes.plot([0, limit], [0, limit], color=ASH, linestyle="--")
    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    style_axes(axes, "Actual vs predicted price (random-split test set)")
    axes.set_xlabel("actual price (million THB)")
    axes.set_ylabel("predicted price (million THB)")
    axes.legend(facecolor=CHARCOAL, labelcolor=WHITE, edgecolor=CHARCOAL)
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
    figure, axes = new_figure(10, 0.42 * len(names) + 1.5)
    axes.barh(positions - 0.2, [random_results[n]["mape_pct"] for n in names], height=0.4, color=GOLD, label="random split")
    axes.barh(positions + 0.2, [project_results[n]["mape_pct"] for n in names], height=0.4, color=ASH, label="unseen buildings")
    for position, name in zip(positions, names):
        axes.text(random_results[name]["mape_pct"], position - 0.2, f" {random_results[name]['mape_pct']:.1f}%", va="center", color=WHITE, fontsize=8)
        axes.text(project_results[name]["mape_pct"], position + 0.2, f" {project_results[name]['mape_pct']:.1f}%", va="center", color=WHITE, fontsize=8)
    axes.set_yticks(positions, names, fontsize=8)
    axes.invert_yaxis()
    style_axes(axes, "Mean absolute percentage error on test set (lower is better)")
    axes.set_xlabel("MAPE (%)")
    axes.legend(facecolor=CHARCOAL, labelcolor=WHITE, edgecolor=CHARCOAL, loc="upper center",
                bbox_to_anchor=(0.5, -0.12), ncol=2)
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
    projects = pd.read_csv(PROJECTS_CSV)
    # Same collection rule for every zone: zone 01 was saved to 38 pages, others to 2,
    # so without this cap one zone would dominate the training data.
    listings = all_listings[all_listings["page"] <= MAX_PAGES_PER_ZONE].reset_index(drop=True)
    listings = attach_rich_buildings(listings, projects)
    LOG.info("building data found for %.1f%% of listings", 100 * listings["has_building_info"].mean())
    LOG.info("using %s of %s clean listings (pages <= %s per zone)", len(listings), len(all_listings), MAX_PAGES_PER_ZONE)

    main_run = evaluate_split(*split_random(listings))
    project_run = evaluate_split(*split_by_project(listings))
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
        "ffbp_random_split": main_run["ffbp"],
        "ffbp_project_split": project_run["ffbp"],
        "features": main_run["feature_names"],
        "building_features": main_run["building_feature_names"],
        "building_linear_regression": {
            name: {"coef_log": float(coef), "effect_pct": float(100 * (np.exp(coef) - 1))}
            for name, coef in zip(main_run["building_feature_names"], main_run["W_building"][:-1])
        },
        "linear_regression": {"intercept_log": float(main_run["W"][-1]), "coefficients": coefficients},
        "test_results_random_split": main_run["results"],
        "test_results_project_split": project_run["results"],
        "sota_style_random_split": main_run["sota_style"],
        "sota_style_project_split": project_run["sota_style"],
        "website_filter_match_pct": {"random_split": main_run["filter_match_pct"],
                                     "project_split": project_run["filter_match_pct"]},
        "stabilized_ensemble": {
            "random_split": main_run["stabilized"]["meta"],
            "project_split": project_run["stabilized"]["meta"],
        },
        "learning_curve": curve,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    clustering.zone_table.to_csv(REPORTS_DIR / "zone_clusters.csv", index=False)
    memberships = main_run["memberships"].merge(clustering.zone_table[["zone_no", "tier"]], on="zone_no")
    memberships.sort_values("median_price_per_sqm").to_csv(REPORTS_DIR / "zone_fuzzy_memberships.csv", index=False)
    plot_fuzzy_memberships(memberships)

    plot_cluster_scan(clustering)
    plot_zone_tiers(clustering)
    plot_actual_vs_predicted(main_run["y_test"], main_run["predictions"])
    plot_knn_curve(knn.curve, knn.k)
    plot_model_comparison(main_run["results"], project_run["results"])
    plot_learning_curve(curve)
    plot_knn_experiment(main_run["knn_experiment"], project_run["knn_experiment"])

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
        # main web model: Linear Regression with building features (lowest error on unseen buildings)
        "W_building": main_run["W_building"],
        "W_building_project": main_run["W_building_project"],
        "building_features": main_run["buildings"],
        "building_feature_names": main_run["building_feature_names"],
        "building_project_feature_names": main_run["building_project_feature_names"],
        "residual_std_log_building": main_run["residual_std_log_building"],
        "residual_std_log_building_project": main_run["residual_std_log_building_project"],
        "knn_building": {key: getattr(main_run["knn_building"], key) for key in ("X", "y", "scaler", "k", "metric", "weight")},
        "projects": projects,
        "reference_rows": reference_rows,
        "project_names": sorted(set(main_run["train"]["project_name"].dropna()) | set(projects["project_name"].str.strip())),
        "test_results": main_run["results"],
        "test_results_project_split": project_run["results"],
        "residual_std_log": main_run["residual_std_log"],
        "residual_std_log_project": main_run["residual_std_log_project"],
        "stabilized_ensemble": main_run["stabilized"]["bundle"],
    }
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BUNDLE_PATH.open("wb") as handle:
        pickle.dump(bundle, handle)
    for name in main_run["results"]:
        LOG.info("%-52s random MAPE %5.1f%% | unseen buildings MAPE %5.1f%%", name,
                 main_run["results"][name]["mape_pct"], project_run["results"][name]["mape_pct"])


if __name__ == "__main__":
    main()
