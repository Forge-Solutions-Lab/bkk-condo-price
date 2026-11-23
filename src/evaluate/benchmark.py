"""Benchmark that does not flatter the models.

The single seed-42 test sets in run.py have been looked at while choosing models, so their numbers
are optimistic. This benchmark:
- locks a final holdout (whole buildings for the unseen-building case, plus a share of units in
  other buildings for the known-building case) that is scored only with --final;
- scores every decision on the rest with grouped 5-fold CV repeated over several seeds;
- keeps near-duplicate listings (the same unit posted twice) in the same fold, so a unit is never
  tested against its own twin;
- compares each web model with a simple baseline it has to beat.

Run: python -m src.evaluate.benchmark          (CV on the development rows)
     python -m src.evaluate.benchmark --final  (also scores the locked holdout into benchmark_holdout.json;
                                           run it once, at the end, never to choose a model)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.evaluate.run import (
    CLEAN_CSV, MAX_PAGES_PER_ZONE, PROJECTS_CSV, REPORTS_DIR, best_zone_clustering, choose_neighbour_count,
)
from src.features.build import (
    BuildingFeatures, NeighbourhoodPrice, ProjectPriceLevels, ZoneClustering, attach_buildings, feature_frame,
    project_key,
)
from src.models import linear_regression as lr

LOG = logging.getLogger(__name__)
CV_FOLDS = 5
CV_SEEDS = (11, 22, 33)
HOLDOUT_SEED = 2026
HOLDOUT_BUILDING_SHARE = 0.15
HOLDOUT_KNOWN_UNIT_SHARE = 0.10
# same unit posted twice: same building, area within 1 sqm, same floor, price within 3%
TWIN_AREA_SQM = 1.0
TWIN_PRICE_SHARE = 0.03
LARGE_UNIT_SQM = 120
MIN_ZONE_UNITS = 20
GB_SEED = 42


def load_listings() -> pd.DataFrame:
    listings = pd.read_csv(CLEAN_CSV, dtype={"listing_id": str})
    listings = listings[listings["page"] <= MAX_PAGES_PER_ZONE].reset_index(drop=True)
    listings = attach_buildings(listings, pd.read_csv(PROJECTS_CSV))
    listings["building"] = project_key(listings["project_name"]).to_numpy()
    listings["twin_group"] = twin_groups(listings)
    return listings


def twin_groups(listings: pd.DataFrame) -> np.ndarray:
    """Group id per row; near-duplicate listings of one unit share an id."""
    parent = np.arange(len(listings))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys = project_key(listings["project_name"]).to_numpy()
    area, floor, price = (listings[c].to_numpy(dtype=float) for c in ("area_sqm", "floor", "price_thb"))
    for key in pd.unique(keys):
        rows = np.flatnonzero(keys == key)
        for a, i in enumerate(rows):
            for j in rows[a + 1:]:
                same_floor = floor[i] == floor[j] or (np.isnan(floor[i]) and np.isnan(floor[j]))
                if (abs(area[i] - area[j]) <= TWIN_AREA_SQM and same_floor
                        and abs(price[i] - price[j]) <= TWIN_PRICE_SHARE * price[i]):
                    parent[root(j)] = root(i)
    return np.array([root(i) for i in range(len(listings))])


def group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold number per row; every row of a group lands in the same fold (shuffled by seed)."""
    unique = pd.unique(groups)
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of = dict(zip(unique[order], np.arange(len(unique)) % n_folds))
    return np.array([fold_of[g] for g in groups])


def split_holdout(listings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(development, unseen-building holdout, known-building holdout)."""
    rng = np.random.default_rng(HOLDOUT_SEED)
    buildings = pd.unique(listings["building"])
    held_buildings = set(rng.choice(buildings, int(HOLDOUT_BUILDING_SHARE * len(buildings)), replace=False))
    in_unseen = listings["building"].isin(held_buildings).to_numpy()
    rest = listings[~in_unseen]
    groups = pd.unique(rest["twin_group"])
    held_groups = set(rng.choice(groups, int(HOLDOUT_KNOWN_UNIT_SHARE * len(groups)), replace=False))
    in_known = listings["twin_group"].isin(held_groups).to_numpy() & ~in_unseen
    return (listings[~in_unseen & ~in_known].reset_index(drop=True),
            listings[in_unseen].reset_index(drop=True), listings[in_known].reset_index(drop=True))


@dataclass
class WebModels:
    """The two models the web app uses, fitted the same way as in run.py."""

    clustering: ZoneClustering
    buildings: BuildingFeatures
    levels: ProjectPriceLevels
    neighbourhood: NeighbourhoodPrice
    W_project: np.ndarray
    W_neighbour: np.ndarray
    boosting: HistGradientBoostingRegressor
    zone_price_per_sqm: pd.Series

    @classmethod
    def fit(cls, train: pd.DataFrame) -> WebModels:
        clustering, _ = best_zone_clustering(train)
        buildings = BuildingFeatures.fit(train)
        levels = ProjectPriceLevels.fit(train)
        log_price = np.log(train["price_thb"].to_numpy())
        x_project = feature_frame(train, clustering, project_levels=levels.transform(train, leave_one_out=True),
                                  buildings=buildings.transform(train)).to_numpy()
        k, _ = choose_neighbour_count(train, clustering, buildings)
        neighbourhood = NeighbourhoodPrice.fit(train, k)
        x_neighbour = feature_frame(train, clustering, buildings=buildings.transform(train),
                                    neighbourhood=neighbourhood.transform(train)).to_numpy()
        return cls(
            clustering=clustering, buildings=buildings, levels=levels, neighbourhood=neighbourhood,
            W_project=lr.fit(x_project, log_price), W_neighbour=lr.fit(x_neighbour, log_price),
            boosting=HistGradientBoostingRegressor(random_state=GB_SEED).fit(x_neighbour, log_price),
            zone_price_per_sqm=train.groupby("zone_no")["price_per_sqm"].median(),
        )

    def predict(self, listings: pd.DataFrame) -> pd.DataFrame:
        buildings = self.buildings.transform(listings)
        levels = self.levels.transform(listings, leave_one_out=False)
        x_project = feature_frame(listings, self.clustering, project_levels=levels, buildings=buildings).to_numpy()
        x_neighbour = feature_frame(listings, self.clustering, buildings=buildings,
                                    neighbourhood=self.neighbourhood.transform(listings)).to_numpy()
        log_lr = lr.predict(x_neighbour, self.W_neighbour)
        known = levels["has_project_history"].to_numpy() > 0
        area = listings["area_sqm"].to_numpy()
        return pd.DataFrame({
            "blend": np.exp((log_lr + self.boosting.predict(x_neighbour)) / 2),
            "project_level": np.where(known, np.exp(lr.predict(x_project, self.W_project)), np.nan),
            # baselines the models must beat
            "zone_median": listings["zone_no"].map(self.zone_price_per_sqm).to_numpy() * area,
            "same_building_mean": np.where(known, np.exp(levels["project_log_price_per_sqm"].to_numpy()) * area, np.nan),
            "known_building": known,
        })


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ape = np.abs(predicted - actual) / actual
    return {"units": int(len(actual)), "mape_pct": float(100 * ape.mean()), "median_ape_pct": float(100 * np.median(ape)),
            "within_20pct": float(100 * np.mean(ape <= 0.2)), "mae_thb": float(np.mean(np.abs(predicted - actual)))}


def out_of_fold(dev: pd.DataFrame, group_column: str, seed: int) -> pd.DataFrame:
    folds = group_folds(dev[group_column].to_numpy(), CV_FOLDS, seed)
    parts = []
    for fold in range(CV_FOLDS):
        train, valid = dev[folds != fold], dev[folds == fold]
        models = WebModels.fit(train)
        valid = valid[valid["zone_no"].isin(models.clustering.zone_table["zone_no"])]
        parts.append(pd.concat([valid.reset_index(drop=True), models.predict(valid)], axis=1))
    return pd.concat(parts, ignore_index=True)


def case_scores(frame: pd.DataFrame, model: str, baseline: str, rows: np.ndarray) -> dict[str, object]:
    actual = frame["price_thb"].to_numpy()[rows]
    result = {"model": score(actual, frame[model].to_numpy()[rows]),
              "baseline": score(actual, frame[baseline].to_numpy()[rows])}
    large = frame["area_sqm"].to_numpy()[rows] > LARGE_UNIT_SQM
    result["large_units"] = score(actual[large], frame[model].to_numpy()[rows][large])
    error = (frame[model].to_numpy()[rows] - actual) / actual
    zones = pd.DataFrame({"zone_no": frame["zone_no"].to_numpy()[rows], "error": error}).groupby("zone_no")["error"]
    bias = zones.median()[zones.size() >= MIN_ZONE_UNITS]
    result["worst_zone_bias_pct"] = float(100 * bias.abs().max())
    result["worst_zone"] = int(bias.abs().idxmax())
    return result


def summarise(per_seed: list[dict[str, object]]) -> dict[str, object]:
    """Mean and spread over seeds of every number in the per-seed results."""
    def walk(items: list[object]) -> object:
        if isinstance(items[0], dict):
            return {key: walk([item[key] for item in items]) for key in items[0]}
        values = np.array(items, dtype=float)
        return {"mean": float(values.mean()), "std": float(values.std())}
    return walk(per_seed)


def development_cv(dev: pd.DataFrame) -> dict[str, object]:
    unseen, known = [], []
    for seed in CV_SEEDS:
        LOG.info("seed %s: unseen-building folds", seed)
        frame = out_of_fold(dev, "building", seed)
        unseen.append(case_scores(frame, "blend", "zone_median", np.ones(len(frame), dtype=bool)))
        LOG.info("seed %s: known-building folds", seed)
        frame = out_of_fold(dev, "twin_group", seed)
        known.append(case_scores(frame, "project_level", "same_building_mean", frame["known_building"].to_numpy()))
    return {"unseen_building": summarise(unseen), "known_building": summarise(known)}


def final_holdout(dev: pd.DataFrame, unseen: pd.DataFrame, known: pd.DataFrame) -> dict[str, object]:
    models = WebModels.fit(dev)
    zones = models.clustering.zone_table["zone_no"]
    unseen, known = unseen[unseen["zone_no"].isin(zones)], known[known["zone_no"].isin(zones)]
    unseen = pd.concat([unseen.reset_index(drop=True), models.predict(unseen)], axis=1)
    known = pd.concat([known.reset_index(drop=True), models.predict(known)], axis=1)
    return {
        "unseen_building": case_scores(unseen, "blend", "zone_median", np.ones(len(unseen), dtype=bool)),
        "known_building": case_scores(known, "project_level", "same_building_mean", known["known_building"].to_numpy()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true", help="also score the locked holdout")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dev, unseen, known = split_holdout(load_listings())
    LOG.info("development %s rows | holdout: unseen buildings %s rows, known buildings %s rows",
             len(dev), len(unseen), len(known))
    report: dict[str, object] = {
        "rows": {"development": len(dev), "holdout_unseen": len(unseen), "holdout_known": len(known)},
        "cv": development_cv(dev),
    }
    (REPORTS_DIR / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.final:
        holdout = final_holdout(dev, unseen, known)
        (REPORTS_DIR / "benchmark_holdout.json").write_text(json.dumps(holdout, indent=2, sort_keys=True), encoding="utf-8")
    for case, values in report["cv"].items():
        model, base = values["model"], values["baseline"]
        LOG.info("%-16s MAPE %.1f%% (+-%.1f) within 20%% %.1f%% | baseline MAPE %.1f%% | large units within 20%% %.1f%%",
                 case, model["mape_pct"]["mean"], model["mape_pct"]["std"], model["within_20pct"]["mean"],
                 base["mape_pct"]["mean"], values["large_units"]["within_20pct"]["mean"])


if __name__ == "__main__":
    main()
