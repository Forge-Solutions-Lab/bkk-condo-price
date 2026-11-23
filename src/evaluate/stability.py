"""Multi-seed stability benchmark comparing baseline vs stabilized models across 5 splits."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.evaluate.run import (
    BUILDING_LR,
    CLEAN_CSV,
    MAX_PAGES_PER_ZONE,
    PROJECTS_CSV,
    REPORTS_DIR,
    STABILITY_SEEDS,
    fit_stabilized_suite,
    plot_stability_benchmark,
    price_metrics,
)
from src.features.build import (
    SEED,
    TEST_SIZE,
    BuildingFeatures,
    attach_rich_buildings,
    cluster_zones,
    feature_frame,
    zone_statistics,
)
from src.models import linear_regression as lr

LOG = logging.getLogger(__name__)
OUTPUT_JSON = REPORTS_DIR / "stability_report.json"


def run_stability_benchmark(listings: pd.DataFrame) -> dict[str, object]:
    """Measure MAPE and R^2 mean +- std across independent splits on both split types."""
    report: dict[str, object] = {"seeds": list(STABILITY_SEEDS)}
    for split_mode in ("random_split", "project_split"):
        per_model: dict[str, list[dict[str, float]]] = {}
        for seed in STABILITY_SEEDS:
            if split_mode == "random_split":
                zone_counts = listings["zone_no"].map(listings["zone_no"].value_counts())
                stratify = listings["zone_no"] if (zone_counts >= 2).all() else None
                tr, te = train_test_split(listings, test_size=TEST_SIZE, random_state=seed, stratify=stratify)
                tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
            else:
                splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
                ti, vi = next(splitter.split(listings, groups=listings["project_name"]))
                tr, te = listings.iloc[ti].reset_index(drop=True), listings.iloc[vi].reset_index(drop=True)

            clustering, _, _ = cluster_zones(zone_statistics(tr), 8, metric="euclidean")
            te = te[te["zone_no"].isin(clustering.zone_table["zone_no"])].reset_index(drop=True)
            y_tr_log = np.log(tr["price_thb"].to_numpy(dtype=float))
            y_te = te["price_thb"].to_numpy(dtype=float)

            bldg = BuildingFeatures.fit(tr)
            xb_tr = feature_frame(tr, clustering, buildings=bldg.transform(tr)).to_numpy()
            xb_te = feature_frame(te, clustering, buildings=bldg.transform(te)).to_numpy()
            base_preds = {
                BUILDING_LR: np.exp(lr.predict(xb_te, lr.fit(xb_tr, y_tr_log))),
                "Gradient Boosting + building features (reference)": np.exp(
                    HistGradientBoostingRegressor(random_state=SEED).fit(xb_tr, y_tr_log).predict(xb_te)
                ),
            }
            stabilized = fit_stabilized_suite(tr, te, clustering, n_seeds=5)
            all_preds = {**base_preds, **stabilized["predictions"]}
            for name, pred in all_preds.items():
                metrics = price_metrics(y_te, pred)
                per_model.setdefault(name, []).append({
                    "seed": int(seed),
                    "mape_pct": metrics["mape_pct"],
                    "r2": metrics["r2"],
                    "within_20pct": metrics["within_20pct"],
                })

        summary = {}
        for name, rows in per_model.items():
            mapes = [row["mape_pct"] for row in rows]
            r2s = [row["r2"] for row in rows]
            w20s = [row["within_20pct"] for row in rows]
            summary[name] = {
                "mean_mape_pct": float(np.mean(mapes)),
                "std_mape_pct": float(np.std(mapes)),
                "mean_r2": float(np.mean(r2s)),
                "std_r2": float(np.std(r2s)),
                "mean_within_20pct": float(np.mean(w20s)),
                "per_seed": rows,
            }
        report[split_mode] = summary
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    all_listings = pd.read_csv(CLEAN_CSV, dtype={"listing_id": str})
    projects = pd.read_csv(PROJECTS_CSV)
    listings = attach_rich_buildings(
        all_listings[all_listings["page"] <= MAX_PAGES_PER_ZONE].reset_index(drop=True), projects
    )
    report = run_stability_benchmark(listings)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    plot_stability_benchmark(report)
    for split_mode in ("random_split", "project_split"):
        LOG.info("=== %s ===", split_mode)
        for name, row in report[split_mode].items():
            LOG.info(
                "  %-56s MAPE %5.2f%% +- %4.2f%% | R2 %.3f +- %.3f",
                name, row["mean_mape_pct"], row["std_mape_pct"], row["mean_r2"], row["std_r2"],
            )


if __name__ == "__main__":
    main()
