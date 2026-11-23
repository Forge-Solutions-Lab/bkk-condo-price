"""Price of a unit in a building the data has never seen.

Three models on the same extended building + neighbourhood features, averaged in log price:
- Linear Regression (Ridge) on log price,
- Gradient Boosting on log price per sqm,
- Extra Trees on log price per sqm.
Tree models cannot predict above the prices they were trained on, so they pull large luxury
units down; Linear Regression extrapolates. The weight of Linear Regression therefore grows with
the unit's size, from LR_WEIGHT at 80 sqm to LR_WEIGHT_LARGE at 150 sqm and above.

Features also include the developer series' price premium over its neighbourhood
(BrandPremium in src/features/build.py), learnt only from other buildings of the same series.

Weights and the size ramp were chosen by grouped CV on the development rows in
src/evaluate/benchmark.py, never on the locked holdout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from src.features.build import (
    SEED, BrandPremium, ExtendedBuildingFeatures, MultiScaleNeighbourhood, NeighbourSpread, RecentNeighbourLevel,
    ZoneClustering, feature_frame,
)
from src.models import linear_regression as lr

LR_WEIGHT, GB_WEIGHT, ET_WEIGHT = 0.40, 0.35, 0.25
LR_WEIGHT_LARGE = 0.80
RAMP_START_SQM, RAMP_END_SQM = 80.0, 150.0
# Linear Regression gets a separate slope for log area above this size
LARGE_AREA_HINGE_SQM = 80.0


def make_gb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=5, min_samples_leaf=20, l2_regularization=2.0,
        learning_rate=0.05, max_iter=250, random_state=SEED,
    )


def make_et() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.7, random_state=SEED, n_jobs=-1)


def lr_weight(area_sqm: np.ndarray) -> np.ndarray:
    share = np.clip((np.asarray(area_sqm, dtype=float) - RAMP_START_SQM) / (RAMP_END_SQM - RAMP_START_SQM), 0.0, 1.0)
    return LR_WEIGHT + share * (LR_WEIGHT_LARGE - LR_WEIGHT)


@dataclass
class UnseenBuildingEnsemble:
    extended_buildings: ExtendedBuildingFeatures
    neighbourhood: MultiScaleNeighbourhood
    W_lr: np.ndarray
    gb: HistGradientBoostingRegressor
    et: ExtraTreesRegressor
    l2: float
    feature_names: list[str]
    brand: BrandPremium | None = None
    recent: RecentNeighbourLevel | None = None
    spread: NeighbourSpread | None = None

    @staticmethod
    def _frame(listings: pd.DataFrame, clustering: ZoneClustering, extended_buildings: ExtendedBuildingFeatures,
               neighbourhood: MultiScaleNeighbourhood, brand: BrandPremium | None = None,
               recent: RecentNeighbourLevel | None = None, spread: NeighbourSpread | None = None) -> pd.DataFrame:
        frame = feature_frame(listings, clustering, buildings=extended_buildings.transform(listings),
                              neighbourhood=neighbourhood.transform(listings))
        frame["large_area_hinge"] = np.maximum(0.0, frame["log_area_sqm"] - np.log(LARGE_AREA_HINGE_SQM))
        if brand is not None:
            frame["brand_premium"] = brand.transform(listings)
        if recent is not None:
            # gap between recent launches nearby and all nearby buildings; 0 where unknown
            frame["nb_recent_gap"] = np.nan_to_num(recent.transform(listings) - frame["nb_idw_k5"].to_numpy())
        if spread is not None:
            for column, values in spread.transform(listings).items():
                frame[column] = values.to_numpy()
        return frame

    @classmethod
    def fit(cls, train: pd.DataFrame, clustering: ZoneClustering, l2: float = 5.0) -> UnseenBuildingEnsemble:
        extended_buildings = ExtendedBuildingFeatures.fit(train)
        neighbourhood = MultiScaleNeighbourhood.fit(train)
        brand = BrandPremium.fit(train, neighbourhood.transform(train)["nb_idw_k5"].to_numpy())
        recent = RecentNeighbourLevel.fit(train)
        spread = NeighbourSpread.fit(train)
        frame = cls._frame(train, clustering, extended_buildings, neighbourhood, brand, recent, spread)
        x = frame.to_numpy()
        log_ppsqm = np.log(train["price_per_sqm"].to_numpy())
        return cls(
            extended_buildings=extended_buildings, neighbourhood=neighbourhood,
            W_lr=lr.fit(x, np.log(train["price_thb"].to_numpy()), l2=l2),
            gb=make_gb().fit(x, log_ppsqm), et=make_et().fit(x, log_ppsqm),
            l2=float(l2), feature_names=list(frame.columns), brand=brand, recent=recent, spread=spread,
        )

    def features(self, listings: pd.DataFrame, clustering: ZoneClustering) -> np.ndarray:
        frame = self._frame(listings, clustering, self.extended_buildings, self.neighbourhood, self.brand,
                            self.recent, self.spread)
        return frame[self.feature_names].to_numpy()

    def predict_parts(self, listings: pd.DataFrame, clustering: ZoneClustering) -> pd.DataFrame:
        """Log price from each model, the tree average and the size-weighted blend."""
        x = self.features(listings, clustering)
        area = listings["area_sqm"].to_numpy(dtype=float)
        log_lr = lr.predict(x, self.W_lr)
        log_gb = self.gb.predict(x) + np.log(area)
        log_et = self.et.predict(x) + np.log(area)
        log_trees = (GB_WEIGHT * log_gb + ET_WEIGHT * log_et) / (GB_WEIGHT + ET_WEIGHT)
        w = lr_weight(area)
        return pd.DataFrame({"log_lr": log_lr, "log_gb": log_gb, "log_et": log_et, "log_trees": log_trees,
                             "lr_weight": w, "log_blend": w * log_lr + (1.0 - w) * log_trees})

    def predict(self, listings: pd.DataFrame, clustering: ZoneClustering) -> np.ndarray:
        return np.exp(self.predict_parts(listings, clustering)["log_blend"].to_numpy())
