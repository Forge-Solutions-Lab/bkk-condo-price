"""Benchmark that does not flatter the models.

The single seed-42 test sets in run.py have been looked at while choosing models, so their numbers
are optimistic. This benchmark:
- locks a final holdout (whole buildings for the unseen-building case, plus a share of units in
  other buildings for the known-building case) that is scored only with --final;
- scores every decision on the rest with grouped 5-fold CV repeated over several seeds;
- keeps near-duplicate listings (the same unit posted twice) in the same fold, so a unit is never
  tested against its own twin (src/clean/clean.py now removes most of them before this);
- picks the holdout by a hash of the building name / listing id, so later data cleaning does not
  reshuffle which rows are locked away;
- lets listings from pages beyond the per-zone cap into the known-building history only, never
  into validation or the unseen-building model;
- compares each web model with a simple baseline it has to beat.

Run: python -m src.evaluate.benchmark          (CV on the development rows)
     python -m src.evaluate.benchmark --final  (also scores the locked holdout into benchmark_holdout.json;
                                           run it once, at the end, never to choose a model)
"""

from __future__ import annotations

import argparse
import json
import zlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean.clean import twin_groups
from src.evaluate.run import (
    CLEAN_CSV, MAX_PAGES_PER_ZONE, PROJECTS_CSV, REPORTS_DIR, best_zone_clustering,
    choose_extended_settings,
)
from src.features.build import (
    BuildingFeatures, ExtendedBuildingFeatures, KnownBuildingPrice, MultiScaleNeighbourhood, ProjectPriceLevels,
    ZoneClustering, attach_buildings, feature_frame, project_key,
)
from src.models import linear_regression as lr
from src.models.unseen_ensemble import UnseenBuildingEnsemble

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


def prepare(rows: pd.DataFrame) -> pd.DataFrame:
    # a cleared project name (zone name typed as project) is read back from CSV as NaN
    rows = rows.assign(project_name=rows["project_name"].fillna(""))
    rows = attach_buildings(rows.reset_index(drop=True), pd.read_csv(PROJECTS_CSV))
    keys = project_key(rows["project_name"])
    # a listing without a project name is its own one-unit "building" for grouping
    rows["building"] = np.where(keys == "", "__listing_" + rows["listing_id"].astype(str), keys)
    rows["twin_group"] = twin_groups(rows)
    return rows


def load_listings() -> pd.DataFrame:
    listings = pd.read_csv(CLEAN_CSV, dtype={"listing_id": str})
    return prepare(listings[listings["page"] <= MAX_PAGES_PER_ZONE])


def load_extra_rows() -> pd.DataFrame:
    """Listings from pages beyond the per-zone cap (zones 1 and 2 were saved to more pages)."""
    listings = pd.read_csv(CLEAN_CSV, dtype={"listing_id": str})
    return prepare(listings[listings["page"] > MAX_PAGES_PER_ZONE])


def group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold number per row; every row of a group lands in the same fold (shuffled by seed)."""
    unique = pd.unique(groups)
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of = dict(zip(unique[order], np.arange(len(unique)) % n_folds))
    return np.array([fold_of[g] for g in groups])


def stable_share(values: pd.Series, salt: str) -> np.ndarray:
    """A number in [0, 1) per value that depends only on the value, so a row stays in or out of
    the holdout whatever other rows are added or removed."""
    return np.array([zlib.crc32(f"{salt}:{value}".encode()) % 10_000 / 10_000 for value in values])


def split_holdout(listings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(development, unseen-building holdout, known-building holdout)."""
    in_unseen = stable_share(listings["building"], "building") < HOLDOUT_BUILDING_SHARE
    # a unit and its twins share the smallest listing id of the group
    group_id = listings.groupby("twin_group")["listing_id"].transform("min")
    in_known = (stable_share(group_id, "unit") < HOLDOUT_KNOWN_UNIT_SHARE) & ~in_unseen
    return (listings[~in_unseen & ~in_known].reset_index(drop=True),
            listings[in_unseen].reset_index(drop=True), listings[in_known].reset_index(drop=True))


def usable_extra(extra: pd.DataFrame, held_out: pd.DataFrame) -> pd.DataFrame:
    """Extra-page listings that may join a training history: none from a building that is being
    tested as unseen, and none of the tested units themselves."""
    return extra[~extra["building"].isin(held_out["building"]) & ~extra["listing_id"].isin(held_out["listing_id"])]


@dataclass
class WebModels:
    """The two models the web app uses, fitted the same way as in run.py: the known-building
    model and the unseen-building ensemble (src/models/unseen_ensemble.py)."""

    clustering: ZoneClustering
    buildings: BuildingFeatures
    extended_buildings: ExtendedBuildingFeatures
    levels: ProjectPriceLevels
    neighbourhood: MultiScaleNeighbourhood
    W_project: np.ndarray
    unseen: UnseenBuildingEnsemble
    zone_price_per_sqm: pd.Series
    known_model: KnownBuildingPrice
    history_levels: ProjectPriceLevels

    @classmethod
    def fit(cls, train: pd.DataFrame, extra_history: pd.DataFrame | None = None) -> WebModels:
        groups = train["building"]
        clustering, _ = best_zone_clustering(train, groups=groups)
        buildings = BuildingFeatures.fit(train)
        levels = ProjectPriceLevels.fit(train)
        log_price = np.log(train["price_thb"].to_numpy())
        x_project = feature_frame(train, clustering, project_levels=levels.transform(train, leave_one_out=True),
                                  buildings=buildings.transform(train)).to_numpy()
        ext_settings = choose_extended_settings(train, clustering, groups=groups)
        unseen = UnseenBuildingEnsemble.fit(train, clustering, l2=ext_settings["l2"])
        models = cls(
            clustering=clustering, buildings=buildings, extended_buildings=unseen.extended_buildings,
            levels=levels, neighbourhood=unseen.neighbourhood,
            W_project=lr.fit(x_project, log_price),
            unseen=unseen,
            zone_price_per_sqm=train.groupby("zone_no")["price_per_sqm"].median(),
            known_model=None, history_levels=None,
        )
        history = train if extra_history is None else pd.concat([train, extra_history], ignore_index=True)
        models.history_levels = ProjectPriceLevels.fit(history)
        models.known_model = KnownBuildingPrice.fit(train, models.unseen_log_ppsqm(train), extra_history)
        return models

    def unseen_log_ppsqm(self, listings: pd.DataFrame) -> np.ndarray:
        """Log price per sqm from the unseen-building ensemble, the prior for the known-building model."""
        return np.log(self.unseen.predict(listings, self.clustering) / listings["area_sqm"].to_numpy())

    def predict(self, listings: pd.DataFrame) -> pd.DataFrame:
        buildings = self.buildings.transform(listings)
        levels = self.levels.transform(listings, leave_one_out=False)
        x_project = feature_frame(listings, self.clustering, project_levels=levels, buildings=buildings).to_numpy()
        area = listings["area_sqm"].to_numpy()
        blend = self.unseen.predict(listings, self.clustering)
        known_price = self.known_model.predict(listings, np.log(blend / area))
        history = self.history_levels.transform(listings, leave_one_out=False)
        known = np.isfinite(known_price)
        return pd.DataFrame({
            "blend": blend,
            "project_level": known_price,
            "project_level_lr": np.where(levels["has_project_history"].to_numpy() > 0,
                                         np.exp(lr.predict(x_project, self.W_project)), np.nan),
            # baselines the models must beat; same history as the known-building model
            "zone_median": listings["zone_no"].map(self.zone_price_per_sqm).to_numpy() * area,
            "same_building_mean": np.where(known, np.exp(history["project_log_price_per_sqm"].to_numpy()) * area, np.nan),
            "known_building": known,
        })


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ape = np.abs(predicted - actual) / actual
    return {"units": int(len(actual)), "mape_pct": float(100 * ape.mean()), "median_ape_pct": float(100 * np.median(ape)),
            "within_20pct": float(100 * np.mean(ape <= 0.2)), "mae_thb": float(np.mean(np.abs(predicted - actual)))}


def out_of_fold(dev: pd.DataFrame, extra: pd.DataFrame, group_column: str, seed: int) -> pd.DataFrame:
    folds = group_folds(dev[group_column].to_numpy(), CV_FOLDS, seed)
    parts = []
    for fold in range(CV_FOLDS):
        train, valid = dev[folds != fold], dev[folds == fold]
        models = WebModels.fit(train, usable_extra(extra, valid))
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


def development_cv(dev: pd.DataFrame, extra: pd.DataFrame) -> dict[str, object]:
    unseen, known = [], []
    for seed in CV_SEEDS:
        LOG.info("seed %s: unseen-building folds", seed)
        frame = out_of_fold(dev, extra, "building", seed)
        unseen.append(case_scores(frame, "blend", "zone_median", np.ones(len(frame), dtype=bool)))
        LOG.info("seed %s: known-building folds", seed)
        frame = out_of_fold(dev, extra, "twin_group", seed)
        known.append(case_scores(frame, "project_level", "same_building_mean", frame["known_building"].to_numpy()))
    return {"unseen_building": summarise(unseen), "known_building": summarise(known)}


def final_holdout(dev: pd.DataFrame, extra: pd.DataFrame, unseen: pd.DataFrame, known: pd.DataFrame) -> dict[str, object]:
    models = WebModels.fit(dev, usable_extra(extra, known))
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
    # extra-page rows of a locked unseen-holdout building would make it a known building
    extra = usable_extra(load_extra_rows(), unseen)
    LOG.info("development %s rows | holdout: unseen buildings %s rows, known buildings %s rows",
             len(dev), len(unseen), len(known))
    report: dict[str, object] = {
        "rows": {"development": len(dev), "holdout_unseen": len(unseen), "holdout_known": len(known),
                 "extra_history": len(extra)},
        "cv": development_cv(dev, extra),
    }
    (REPORTS_DIR / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.final:
        holdout = final_holdout(dev, extra, unseen, known)
        (REPORTS_DIR / "benchmark_holdout.json").write_text(json.dumps(holdout, indent=2, sort_keys=True), encoding="utf-8")
    for case, values in report["cv"].items():
        model, base = values["model"], values["baseline"]
        LOG.info("%-16s MAPE %.1f%% (+-%.1f) within 20%% %.1f%% | baseline MAPE %.1f%% | large units within 20%% %.1f%%",
                 case, model["mape_pct"]["mean"], model["mape_pct"]["std"], model["within_20pct"]["mean"],
                 base["mape_pct"]["mean"], values["large_units"]["within_20pct"]["mean"])


if __name__ == "__main__":
    main()
