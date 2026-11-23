"""Forward feature selection for KNN regression with building features.

Adding building attributes improves Linear Regression on unseen buildings (28.3% -> 23.4%
MAPE), whereas KNN barely improves (31.1% -> 30.0%) because Euclidean distance weights
every feature equally. Here we greedily add one feature at a time using 5-fold
cross-validation on the training set only (never touching the test set to choose),
stopping when no additional feature lowers CV MAPE.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from src.evaluate.run import ASH, CHARCOAL, GOLD, WHITE, save_figure, style_axes
from src.features.build import (
    CV_FOLDS,
    SEED,
    BuildingFeatures,
    Standardizer,
    attach_buildings,
    cluster_zones,
    feature_frame,
    mean_absolute_percentage_error,
    split_by_project,
    split_random,
    zone_statistics,
)
from src.models.knn_regression import knn_predict

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
PROJECTS_CSV = PROJECT_ROOT / "data" / "interim" / "projects.csv"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "knn_feature_selection.json"

MAX_PAGES_PER_ZONE = 2
N_CLUSTERS = 8
METRIC = "euclidean"
WEIGHT = "distance"
SELECTION_K = 21
K_CANDIDATES = range(1, 31)
UNSEEN_BUILDINGS_BASELINE_MAPE = 30.0


def make_folds(train: pd.DataFrame, group_by_project: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    if group_by_project:
        splitter = GroupKFold(n_splits=CV_FOLDS)
        return list(splitter.split(train, groups=train["project_name"]))
    splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    return list(splitter.split(train))


def cv_mape(
    features: np.ndarray,
    log_price: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    k: int,
) -> float:
    fold_errors = []
    for train_index, valid_index in folds:
        scaler = Standardizer.fit(features[train_index])
        X = scaler.transform(features[train_index])
        Z = scaler.transform(features[valid_index])
        predicted = np.exp(knn_predict(X, log_price[train_index], Z, k=k, metric=METRIC, weight=WEIGHT))
        actual = np.exp(log_price[valid_index])
        fold_errors.append(mean_absolute_percentage_error(actual, predicted))
    return float(np.mean(fold_errors))


def choose_k(
    features: np.ndarray,
    log_price: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[int, float]:
    best_k, best_mape = SELECTION_K, float("inf")
    for k in K_CANDIDATES:
        score = cv_mape(features, log_price, folds, k)
        if score < best_mape:
            best_k, best_mape = k, score
    return best_k, best_mape


def evaluate_on_test(
    x_train: np.ndarray,
    y_train_log: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k: int,
) -> float:
    scaler = Standardizer.fit(x_train)
    predicted = np.exp(
        knn_predict(scaler.transform(x_train), y_train_log, scaler.transform(x_test), k=k, metric=METRIC, weight=WEIGHT)
    )
    return mean_absolute_percentage_error(y_test, predicted)


def forward_select(
    train_frame: pd.DataFrame,
    y_train_log: np.ndarray,
    test_frame: pd.DataFrame,
    y_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    split_name: str,
) -> dict[str, object]:
    """Greedily add features that lower 5-fold CV MAPE on train; stop when no feature helps.

    Test MAPE is computed only for reporting after each step is chosen and is never used
    to pick features or decide when to stop.
    """
    all_columns = list(train_frame.columns)
    selected: list[str] = []
    remaining = list(all_columns)
    best_cv = float("inf")
    steps: list[dict[str, object]] = []

    while remaining:
        best_candidate = None
        best_candidate_cv = float("inf")
        for candidate in remaining:
            cols = [*selected, candidate]
            score = cv_mape(train_frame[cols].to_numpy(), y_train_log, folds, k=SELECTION_K)
            if score < best_candidate_cv:
                best_candidate = candidate
                best_candidate_cv = score

        if best_candidate is None or best_candidate_cv >= best_cv:
            LOG.info("[%s] stopped at %d features (best CV MAPE %.2f%%)", split_name, len(selected), best_cv)
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_cv = best_candidate_cv
        step_test_mape = evaluate_on_test(
            train_frame[selected].to_numpy(), y_train_log, test_frame[selected].to_numpy(), y_test, k=SELECTION_K
        )
        steps.append({
            "step": len(selected),
            "added_feature": best_candidate,
            "selected_features": list(selected),
            "cv_mape_pct": best_cv,
            "test_mape_pct": step_test_mape,
        })
        LOG.info(
            "[%s] step %2d: +%-18s | CV MAPE %5.2f%% | test MAPE %5.2f%%",
            split_name, len(selected), best_candidate, best_cv, step_test_mape,
        )

    best_k_selected, tuned_cv_mape = choose_k(train_frame[selected].to_numpy(), y_train_log, folds)
    tuned_test_mape = evaluate_on_test(
        train_frame[selected].to_numpy(), y_train_log, test_frame[selected].to_numpy(), y_test, k=best_k_selected
    )

    best_k_all, all_cv_mape = choose_k(train_frame[all_columns].to_numpy(), y_train_log, folds)
    all_test_mape = evaluate_on_test(
        train_frame[all_columns].to_numpy(), y_train_log, test_frame[all_columns].to_numpy(), y_test, k=best_k_all
    )

    return {
        "candidate_features": all_columns,
        "selected_features": selected,
        "n_selected": len(selected),
        "n_total": len(all_columns),
        "selection_k": SELECTION_K,
        "steps": steps,
        "selected_tuned": {
            "k": best_k_selected,
            "cv_mape_pct": tuned_cv_mape,
            "test_mape_pct": tuned_test_mape,
        },
        "all_features_baseline": {
            "k": best_k_all,
            "cv_mape_pct": all_cv_mape,
            "test_mape_pct": all_test_mape,
        },
    }


def run_split(train: pd.DataFrame, test: pd.DataFrame, split_name: str, group_by_project: bool) -> dict[str, object]:
    clustering, _, _ = cluster_zones(zone_statistics(train), N_CLUSTERS, metric=METRIC)
    test = test[test["zone_no"].isin(clustering.zone_table["zone_no"])].reset_index(drop=True)

    buildings = BuildingFeatures.fit(train)
    train_frame = feature_frame(train, clustering, buildings=buildings.transform(train))
    test_frame = feature_frame(test, clustering, buildings=buildings.transform(test))

    y_train_log = np.log(train["price_thb"].to_numpy())
    y_test = test["price_thb"].to_numpy()
    folds = make_folds(train, group_by_project=group_by_project)

    return forward_select(train_frame, y_train_log, test_frame, y_test, folds, split_name)


def plot_selection(random_result: dict[str, object], project_result: dict[str, object]) -> None:
    figure, axes_list = plt.subplots(1, 2, figsize=(12, 5), facecolor="#000000", sharey=True)
    panels = [
        (axes_list[0], random_result, "Random split"),
        (axes_list[1], project_result, "Unseen buildings (project split)"),
    ]
    for axes, result, title in panels:
        steps = result["steps"]
        counts = [row["step"] for row in steps]
        cv_scores = [row["cv_mape_pct"] for row in steps]
        test_scores = [row["test_mape_pct"] for row in steps]
        all_test = result["all_features_baseline"]["test_mape_pct"]

        axes.plot(counts, cv_scores, color=GOLD, marker="o", linewidth=2, label="5-fold CV MAPE (train)")
        axes.plot(counts, test_scores, color=WHITE, marker="s", linestyle="--", linewidth=1.8, label="Test MAPE")
        axes.axhline(all_test, color=ASH, linestyle=":", linewidth=1.3, label=f"All {result['n_total']} features test ({all_test:.1f}%)")

        for x, y, row in zip(counts, cv_scores, steps):
            axes.annotate(
                row["added_feature"],
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=6.5,
                color=ASH,
                rotation=25,
            )

        style_axes(axes, f"KNN forward selection: {title}")
        axes.set_xlabel("number of features")
        axes.set_xticks(counts)
        axes.legend(facecolor=CHARCOAL, labelcolor=WHITE, edgecolor=CHARCOAL, fontsize=8)

    axes_list[0].set_ylabel("MAPE (%)")
    save_figure(figure, "knn_feature_selection.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    all_listings = pd.read_csv(CLEAN_CSV, dtype={"listing_id": str})
    projects = pd.read_csv(PROJECTS_CSV)
    listings = all_listings[all_listings["page"] <= MAX_PAGES_PER_ZONE].reset_index(drop=True)
    listings = attach_buildings(listings, projects)

    random_result = run_split(*split_random(listings), split_name="random_split", group_by_project=False)
    project_result = run_split(*split_by_project(listings), split_name="project_split", group_by_project=True)

    selected_unseen_mape = project_result["selected_tuned"]["test_mape_pct"]
    output = {
        "method": "forward_feature_selection",
        "metric": METRIC,
        "weight": WEIGHT,
        "cv_folds": CV_FOLDS,
        "random_split": random_result,
        "project_split": project_result,
        "summary": {
            "unseen_buildings_benchmark_mape_pct": UNSEEN_BUILDINGS_BASELINE_MAPE,
            "unseen_buildings_selected_test_mape_pct": selected_unseen_mape,
            "improves_over_30pct_unseen_buildings": bool(selected_unseen_mape < UNSEEN_BUILDINGS_BASELINE_MAPE),
        },
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    plot_selection(random_result, project_result)

    LOG.info(
        "random split: selected %d/%d features -> test MAPE %.2f%% (all features %.2f%%)",
        random_result["n_selected"], random_result["n_total"],
        random_result["selected_tuned"]["test_mape_pct"],
        random_result["all_features_baseline"]["test_mape_pct"],
    )
    LOG.info(
        "unseen buildings: selected %d/%d features -> test MAPE %.2f%% (all features %.2f%%, benchmark %.1f%%)",
        project_result["n_selected"], project_result["n_total"],
        selected_unseen_mape,
        project_result["all_features_baseline"]["test_mape_pct"],
        UNSEEN_BUILDINGS_BASELINE_MAPE,
    )


if __name__ == "__main__":
    main()
