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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from src.features.build import (
    SEED,
    ProjectPriceLevels,
    Standardizer,
    ZoneClustering,
    choose_zone_clustering,
    feature_frame,
    split_by_project,
    split_random,
)
from src.models import linear_regression as lr
from src.models.knn_regression import knn_predict

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
BUNDLE_PATH = PROJECT_ROOT / "models" / "model_bundle.pkl"

MAX_PAGES_PER_ZONE = 2
K_CANDIDATES = range(1, 31)
CV_FOLDS = 5
LEARNING_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
METRICS = ["euclidean", "manhattan"]
WEIGHTS = ["uniform", "distance"]
GOLD = "#FFC000"
WHITE = "#FFFFFF"
ASH = "#7D7D7D"
CHARCOAL = "#202020"

MAIN_LR = "Linear Regression (K-Means tiers)"
MAIN_KNN = "KNN Regression (K-Means tiers)"
PROJECT_LR = "Linear Regression + project price level"


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
    x_train_no_location = feature_frame(train, clustering, use_location=False).to_numpy()
    x_test_no_location = feature_frame(test, clustering, use_location=False).to_numpy()

    W = lr.fit(x_train, y_train_log)
    W_project = lr.fit(xp_train, y_train_log)
    W_no_location = lr.fit(x_train_no_location, y_train_log)
    knn, knn_candidates = best_knn(x_train, y_train_log)
    knn_project = FittedKnn.fit(xp_train, y_train_log, knn.metric, knn.weight)
    boosting = HistGradientBoostingRegressor(random_state=SEED).fit(x_train, y_train_log)
    boosting_project = HistGradientBoostingRegressor(random_state=SEED).fit(xp_train, y_train_log)
    zone_price = train.groupby("zone_no")["price_per_sqm"].median()

    predictions = {
        "Baseline: zone median price/sqm x area": test["zone_no"].map(zone_price).to_numpy() * test["area_sqm"].to_numpy(),
        "Linear Regression (no location)": np.exp(lr.predict(x_test_no_location, W_no_location)),
        MAIN_LR: np.exp(lr.predict(x_test, W)),
        MAIN_KNN: knn.predict(x_test),
        "Gradient Boosting (K-Means tiers, reference)": np.exp(boosting.predict(x_test)),
        PROJECT_LR: np.exp(lr.predict(xp_test, W_project)),
        "KNN Regression + project price level": knn_project.predict(xp_test),
        "Gradient Boosting + project price level (reference)": np.exp(boosting_project.predict(xp_test)),
    }
    knn_experiment = [
        {"metric": model.metric, "weight": model.weight, "k": model.k, "cv_mape_pct": model.cv_mape_pct,
         "test_mape_pct": price_metrics(y_test, model.predict(x_test))["mape_pct"]}
        for model in knn_candidates
    ]
    return {
        "train": train, "test": test, "clustering": clustering, "levels": levels,
        "W": W, "W_project": W_project, "knn": knn, "knn_project": knn_project,
        "feature_names": list(feature_frame(train, clustering).columns),
        "project_feature_names": list(xp_train_frame.columns),
        "predictions": predictions,
        "results": {name: price_metrics(y_test, predicted) for name, predicted in predictions.items()},
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
    figure, axes = new_figure(10, 5.5)
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
    # Same collection rule for every zone: zone 01 was saved to 38 pages, others to 2,
    # so without this cap one zone would dominate the training data.
    listings = all_listings[all_listings["page"] <= MAX_PAGES_PER_ZONE].reset_index(drop=True)
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
        "features": main_run["feature_names"],
        "linear_regression": {"intercept_log": float(main_run["W"][-1]), "coefficients": coefficients},
        "test_results_random_split": main_run["results"],
        "test_results_project_split": project_run["results"],
        "learning_curve": curve,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    clustering.zone_table.to_csv(REPORTS_DIR / "zone_clusters.csv", index=False)

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
        "reference_rows": reference_rows,
        "project_names": sorted(main_run["train"]["project_name"].dropna().unique().tolist()),
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
