"""Standalone MSE-reduction experiment building on origin/main without editing existing files.

Key improvements evaluated against the exact same split_random, split_by_project, and
5-fold GroupKFold (unseen-building) protocols:
1. Bangkok-bounded inverse-distance weighted (IDW) neighbourhood pricing, filtering out
   corrupted coordinates outside Bangkok (lat 13.5-14.0, lon 100.3-100.95).
2. Predicting log(price_per_sqm) * area_sqm instead of log(price_thb) to avoid exponential
   area-slope blow-up on large luxury units.
3. Deduplicated location representation + compact structural ratios already in projects.csv
   (units/floor density, relative floor height, high-rise boundary, central BTS/MRT rail,
   area per bedroom).
4. Pure NumPy L2-regularised Ridge Regression (src/models/ridge_regression.py) and
   regularised multi-family ensemble (Ridge + Tuned HistGB + ExtraTrees) with CV tuning
   explicitly minimising THB MSE/RMSE on training folds only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

from src.evaluate.run import (
    ASH,
    CHARCOAL,
    GOLD,
    MAX_PAGES_PER_ZONE,
    WHITE,
    new_figure,
    price_metrics,
    style_axes,
)
from src.features.build import (
    LONGITUDE_SCALE,
    SEED,
    BuildingFeatures,
    NeighbourhoodPrice,
    Standardizer,
    cluster_zones,
    feature_frame,
    project_key,
    split_by_project,
    split_random,
    zone_statistics,
)
from src.models import linear_regression as lr
from src.models import ridge_regression as ridge

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
PROJECTS_CSV = PROJECT_ROOT / "data" / "interim" / "projects.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

BKK_LAT_BOUNDS = (13.5, 14.0)
BKK_LON_BOUNDS = (100.3, 100.95)
IDW_K = 12
IDW_SMOOTH = 0.004  # ~440 m distance floor so adjacent towers do not dominate with infinite weight
RIDGE_LAMBDAS = [1.0, 5.0, 15.0, 40.0, 100.0, 250.0, 500.0]
CV_FOLDS = 5
EXTRA_PROJECT_COLUMNS = [
    "complete_year",
    "total_unit",
    "total_floor",
    "total_building",
    "facility_count",
    "nearest_rail_km",
    "nearest_rail_line",
    "nearest_mall_km",
    "latitude",
    "longitude",
]


def is_valid_bkk_coords(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    lat = pd.to_numeric(latitude, errors="coerce")
    lon = pd.to_numeric(longitude, errors="coerce")
    return lat.between(*BKK_LAT_BOUNDS) & lon.between(*BKK_LON_BOUNDS)


def attach_project_fields(listings: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    """Join all needed project columns by normalised project name without mutating build.py."""
    table = projects.assign(key=project_key(projects["project_name"])).drop_duplicates("key")
    merged = listings.assign(key=project_key(listings["project_name"])).merge(
        table[["key", *EXTRA_PROJECT_COLUMNS]], on="key", how="left"
    )
    merged["has_building_info"] = merged["key"].isin(table["key"])
    return merged.drop(columns="key")


@dataclass
class CleanNeighbourhoodPrice:
    """Inverse-distance weighted mean log(price_per_sqm) of the k nearest other Bangkok
    buildings in training data. Coordinates outside Bangkok bounds are excluded so bad map
    pins cannot corrupt distance lookups."""

    keys: np.ndarray
    coordinates: np.ndarray
    levels: np.ndarray
    zone_fallback: pd.Series
    global_fallback: float
    k: int = IDW_K
    smooth: float = IDW_SMOOTH

    @classmethod
    def fit(cls, train: pd.DataFrame, k: int = IDW_K, smooth: float = IDW_SMOOTH) -> CleanNeighbourhoodPrice:
        valid = is_valid_bkk_coords(train["latitude"], train["longitude"])
        located = train.loc[valid]
        buildings = located.groupby(project_key(located["project_name"])).agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            level=("price_per_sqm", lambda prices: float(np.log(prices).mean())),
        )
        zone_fallback = np.log(train["price_per_sqm"]).groupby(train["zone_no"]).median()
        return cls(
            keys=buildings.index.to_numpy(),
            coordinates=buildings[["latitude", "longitude"]].to_numpy(dtype=float) * [1.0, LONGITUDE_SCALE],
            levels=buildings["level"].to_numpy(dtype=float),
            zone_fallback=zone_fallback,
            global_fallback=float(zone_fallback.median()),
            k=k,
            smooth=smooth,
        )

    def transform(self, listings: pd.DataFrame) -> np.ndarray:
        keys = project_key(listings["project_name"]).to_numpy()
        valid = is_valid_bkk_coords(listings["latitude"], listings["longitude"]).to_numpy()
        lat = pd.to_numeric(listings["latitude"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(listings["longitude"], errors="coerce").to_numpy(dtype=float)
        points = np.column_stack([lat, lon * LONGITUDE_SCALE])
        fallback = listings["zone_no"].map(self.zone_fallback).fillna(self.global_fallback).to_numpy(dtype=float)

        values = np.full(len(listings), np.nan)
        for row, (key, point, ok) in enumerate(zip(keys, points, valid)):
            if not ok or len(self.coordinates) == 0:
                continue
            dist2 = np.sum((self.coordinates - point) ** 2, axis=1)
            dist2[self.keys == key] = np.inf
            finite = np.isfinite(dist2)
            if not np.any(finite):
                continue
            count = min(self.k, int(finite.sum()))
            idx = np.argsort(dist2)[:count]
            weights = 1.0 / (np.sqrt(dist2[idx]) + self.smooth)
            values[row] = float(np.sum(weights * self.levels[idx]) / np.sum(weights))
        return np.where(np.isnan(values), fallback, values)


@dataclass
class ImprovedFeaturePipeline:
    """Deduplicated feature matrix for log(price_per_sqm) prediction. Replaces redundant
    K-Means tier dummies with Bangkok-bounded IDW neighbourhood pricing and adds compact
    structural ratios already present in projects.csv."""

    buildings: BuildingFeatures
    neighbourhood: CleanNeighbourhoodPrice
    medians: dict[str, float]

    @staticmethod
    def _build_raw(
        listings: pd.DataFrame,
        buildings: BuildingFeatures,
        neighbourhood: CleanNeighbourhoodPrice,
    ) -> pd.DataFrame:
        b = buildings.transform(listings)
        area = listings["area_sqm"].to_numpy(dtype=float)
        bedrooms = listings["bedrooms"].to_numpy(dtype=float)
        bathrooms = listings["bathrooms"].to_numpy(dtype=float)
        floor = listings["floor"].to_numpy(dtype=float)
        total_floor = listings["total_floor"].where(listings["total_floor"] > 0).to_numpy(dtype=float)
        total_unit = listings["total_unit"].where(listings["total_unit"] > 0).to_numpy(dtype=float)
        rail_line = listings["nearest_rail_line"].fillna("").astype(str)

        return pd.DataFrame({
            "log_area_sqm": np.log(area),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floor": floor,
            "is_new_unit": listings["is_new_unit"].to_numpy(dtype=float),
            "building_age": b["building_age"].to_numpy(dtype=float),
            "log_total_unit": b["log_total_unit"].to_numpy(dtype=float),
            "total_floor": b["total_floor"].to_numpy(dtype=float),
            "facility_count": b["facility_count"].to_numpy(dtype=float),
            "rail_km": b["rail_km"].to_numpy(dtype=float),
            "mall_km": b["mall_km"].to_numpy(dtype=float),
            "has_building_info": b["has_building_info"].to_numpy(dtype=float),
            "floor_ratio": np.clip(floor / total_floor, 0.0, 1.2),
            "log_units_per_floor": np.log1p(total_unit / total_floor),
            "is_high_rise": (total_floor > 8).astype(float),
            "is_central_rail": rail_line.isin(["bts", "mrt"]).astype(float).to_numpy(),
            "log_area_per_br": np.log(area / np.maximum(bedrooms, 1.0)),
            "clean_neigh_idw": neighbourhood.transform(listings),
        })

    @classmethod
    def fit(cls, train: pd.DataFrame) -> ImprovedFeaturePipeline:
        buildings = BuildingFeatures.fit(train)
        neighbourhood = CleanNeighbourhoodPrice.fit(train)
        raw = cls._build_raw(train, buildings, neighbourhood)
        return cls(buildings=buildings, neighbourhood=neighbourhood, medians=raw.median().to_dict())

    def transform(self, listings: pd.DataFrame) -> pd.DataFrame:
        return self._build_raw(listings, self.buildings, self.neighbourhood).fillna(self.medians)


def full_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    metrics = price_metrics(actual, predicted)
    metrics["mse_thb2"] = float(np.mean((predicted - actual) ** 2))
    return metrics


def _make_hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=3.0,
        random_state=SEED,
    )


def _make_et(n_estimators: int = 300) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1,
    )


def fit_and_predict_improved(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_cv: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Fit pure NumPy Ridge, Regularised HistGB, ExtraTrees, and CV-MSE-optimal blend."""
    pipe = ImprovedFeaturePipeline.fit(train)
    X_tr_df = pipe.transform(train)
    X_te_df = pipe.transform(test)
    X_tr = X_tr_df.to_numpy()
    X_te = X_te_df.to_numpy()

    y_tr_sqm = np.log(train["price_per_sqm"].to_numpy())
    area_tr = train["area_sqm"].to_numpy()
    area_te = test["area_sqm"].to_numpy()
    actual_tr = train["price_thb"].to_numpy()

    if group_cv:
        folds = list(GroupKFold(n_splits=CV_FOLDS).split(X_tr, groups=train["project_name"]))
    else:
        folds = list(KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(X_tr))

    # 1. Select Ridge lambda minimising THB RMSE via CV
    best_lam, best_ridge_rmse = 100.0, np.inf
    for lam in RIDGE_LAMBDAS:
        oof = np.zeros(len(X_tr))
        for tr_i, va_i in folds:
            sc = Standardizer.fit(X_tr[tr_i])
            W = ridge.fit(sc.transform(X_tr[tr_i]), y_tr_sqm[tr_i], lam=lam)
            oof[va_i] = ridge.predict(sc.transform(X_tr[va_i]), W)
        rmse = float(np.sqrt(np.mean((np.exp(oof) * area_tr - actual_tr) ** 2)))
        if rmse < best_ridge_rmse:
            best_ridge_rmse, best_lam = rmse, lam

    # 2. Out-of-fold predictions for (Ridge, Tuned HGB, ExtraTrees) to choose MSE-optimal blend
    oof_r = np.zeros(len(X_tr))
    oof_h = np.zeros(len(X_tr))
    oof_e = np.zeros(len(X_tr))
    for tr_i, va_i in folds:
        sc = Standardizer.fit(X_tr[tr_i])
        W = ridge.fit(sc.transform(X_tr[tr_i]), y_tr_sqm[tr_i], lam=best_lam)
        oof_r[va_i] = ridge.predict(sc.transform(X_tr[va_i]), W)
        oof_h[va_i] = _make_hgb().fit(X_tr_df.iloc[tr_i], y_tr_sqm[tr_i]).predict(X_tr_df.iloc[va_i])
        oof_e[va_i] = _make_et(n_estimators=150).fit(X_tr_df.iloc[tr_i], y_tr_sqm[tr_i]).predict(X_tr_df.iloc[va_i])

    best_weights, best_blend_rmse = (0.4, 0.3, 0.3), np.inf
    grid = np.linspace(0.0, 1.0, 21)
    for wr in grid:
        for wh in grid:
            we = round(1.0 - wr - wh, 6)
            if we < -1e-9:
                continue
            we = max(0.0, we)
            pred = np.exp(wr * oof_r + wh * oof_h + we * oof_e) * area_tr
            rmse = float(np.sqrt(np.mean((pred - actual_tr) ** 2)))
            if rmse < best_blend_rmse:
                best_blend_rmse = rmse
                best_weights = (round(float(wr), 2), round(float(wh), 2), round(float(we), 2))

    # 3. Fit final models on full training split
    sc_full = Standardizer.fit(X_tr)
    W_full = ridge.fit(sc_full.transform(X_tr), y_tr_sqm, lam=best_lam)
    hgb_full = _make_hgb().fit(X_tr_df, y_tr_sqm)
    et_full = _make_et(n_estimators=300).fit(X_tr_df, y_tr_sqm)

    log_r = ridge.predict(sc_full.transform(X_te), W_full)
    log_h = hgb_full.predict(X_te_df)
    log_e = et_full.predict(X_te_df)
    wr, wh, we = best_weights

    preds = {
        "Pure NumPy Ridge (clean IDW, log ppsqm)": np.exp(log_r) * area_te,
        "Tuned HistGB (clean IDW, log ppsqm)": np.exp(log_h) * area_te,
        "ExtraTrees (clean IDW, log ppsqm)": np.exp(log_e) * area_te,
        "Balanced Blend (0.4 Ridge + 0.3 HGB + 0.3 ET)": np.exp(0.4 * log_r + 0.3 * log_h + 0.3 * log_e) * area_te,
        "CV-MSE Optimal Blend (Ridge + HGB + ET)": np.exp(wr * log_r + wh * log_h + we * log_e) * area_te,
    }
    meta = {
        "ridge_lambda": best_lam,
        "cv_ridge_rmse_thb": best_ridge_rmse,
        "cv_blend_weights": {"ridge": wr, "hgb": wh, "et": we},
        "cv_blend_rmse_thb": best_blend_rmse,
        "feature_names": list(X_tr_df.columns),
    }
    return preds, meta


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reproduce exact origin/main baselines on the same train/test split for fair comparison."""
    clustering, _, _ = cluster_zones(zone_statistics(train), 8, "euclidean")
    y_tr_log = np.log(train["price_thb"].to_numpy())
    bf = BuildingFeatures.fit(train)
    xb_tr = feature_frame(train, clustering, buildings=bf.transform(train)).to_numpy()
    xb_te = feature_frame(test, clustering, buildings=bf.transform(test)).to_numpy()

    np_orig = NeighbourhoodPrice.fit(train, k=10)
    xn_tr = feature_frame(train, clustering, buildings=bf.transform(train), neighbourhood=np_orig.transform(train)).to_numpy()
    xn_te = feature_frame(test, clustering, buildings=bf.transform(test), neighbourhood=np_orig.transform(test)).to_numpy()

    p_lr_b = np.exp(lr.predict(xb_te, lr.fit(xb_tr, y_tr_log)))
    log_lr_n = lr.predict(xn_te, lr.fit(xn_tr, y_tr_log))
    log_gb_n = HistGradientBoostingRegressor(random_state=SEED).fit(xn_tr, y_tr_log).predict(xn_te)
    return {
        "Baseline: LR + building (origin/main)": p_lr_b,
        "Baseline: LR + bld + neigh (origin/main)": np.exp(log_lr_n),
        "Baseline: Blend LR + GB (origin/main PR #8)": np.exp(0.5 * log_lr_n + 0.5 * log_gb_n),
    }


def evaluate_group_cv(listings: pd.DataFrame) -> dict[str, dict[str, float]]:
    """5-fold GroupKFold CV across all buildings (every listing evaluated on unseen buildings)."""
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(listings, groups=listings["project_name"]))
    bucket: dict[str, list[np.ndarray]] = {}
    actuals: list[np.ndarray] = []

    for fold_idx, (tr_i, te_i) in enumerate(folds, start=1):
        tr = listings.iloc[tr_i].reset_index(drop=True)
        te = listings.iloc[te_i].reset_index(drop=True)
        te = te[te["zone_no"].isin(set(tr["zone_no"].unique()))].reset_index(drop=True)
        actuals.append(te["price_thb"].to_numpy())

        base_p = baseline_predictions(tr, te)
        imp_p, _ = fit_and_predict_improved(tr, te, group_cv=True)
        for name, arr in {**base_p, **imp_p}.items():
            bucket.setdefault(name, []).append(arr)
        LOG.info("GroupKFold fold %d/%d complete (%d test rows)", fold_idx, CV_FOLDS, len(te))

    actual_all = np.concatenate(actuals)
    return {name: full_metrics(actual_all, np.concatenate(parts)) for name, parts in bucket.items()}


def plot_mse_comparison(
    random_res: dict[str, dict[str, float]],
    project_res: dict[str, dict[str, float]],
    group_cv_res: dict[str, dict[str, float]],
) -> None:
    names = list(random_res)
    positions = np.arange(len(names))
    figure, axes = new_figure(11, 0.55 * len(names) + 1.8)
    h = 0.26
    axes.barh(positions - h, [random_res[n]["rmse_thb"] / 1e6 for n in names], height=h, color=GOLD, label="Random split (seed=42)")
    axes.barh(positions, [project_res[n]["rmse_thb"] / 1e6 for n in names], height=h, color=WHITE, label="Unseen buildings (seed=42)")
    axes.barh(positions + h, [group_cv_res[n]["rmse_thb"] / 1e6 for n in names], height=h, color=ASH, label="5-fold GroupKFold CV (all buildings)")
    for pos, name in zip(positions, names):
        axes.text(random_res[name]["rmse_thb"] / 1e6 + 0.05, pos - h, f"{random_res[name]['rmse_thb']/1e6:.2f}M", va="center", color=GOLD, fontsize=7.5)
        axes.text(project_res[name]["rmse_thb"] / 1e6 + 0.05, pos, f"{project_res[name]['rmse_thb']/1e6:.2f}M", va="center", color=WHITE, fontsize=7.5)
        axes.text(group_cv_res[name]["rmse_thb"] / 1e6 + 0.05, pos + h, f"{group_cv_res[name]['rmse_thb']/1e6:.2f}M", va="center", color=ASH, fontsize=7.5)
    axes.set_yticks(positions, names, fontsize=8.5)
    axes.invert_yaxis()
    style_axes(axes, "RMSE (million THB, lower is better) — baseline vs MSE-improved models")
    axes.set_xlabel("RMSE (million THB)")
    axes.legend(facecolor=CHARCOAL, labelcolor=WHITE, edgecolor=CHARCOAL, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "mse_comparison.png", dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def render_markdown(
    random_res: dict[str, dict[str, float]],
    project_res: dict[str, dict[str, float]],
    group_cv_res: dict[str, dict[str, float]],
    random_meta: dict[str, object],
    project_meta: dict[str, object],
) -> str:
    def _table(title: str, results: dict[str, dict[str, float]], ref_key: str) -> list[str]:
        ref_mse = results[ref_key]["mse_thb2"]
        rows = [
            f"## {title}",
            "",
            "| Model | RMSE (M THB) | MSE (T THB²) | vs PR #8 MSE | MAE (M THB) | MAPE (%) | R² | Within 20% |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, m in sorted(results.items(), key=lambda kv: kv[1]["rmse_thb"]):
            delta = (m["mse_thb2"] / ref_mse - 1.0) * 100.0
            sign = "+" if delta > 0 else ""
            rows.append(
                f"| {name} | {m['rmse_thb']/1e6:.2f} | {m['mse_thb2']/1e12:.2f} | "
                f"{sign}{delta:.1f}% | {m['mae_thb']/1e6:.2f} | {m['mape_pct']:.1f}% | "
                f"{m['r2']:.4f} | {m['within_20pct']:.1f}% |"
            )
        rows.append("")
        return rows

    ref = "Baseline: Blend LR + GB (origin/main PR #8)"
    lines = [
        "# MSE Improvement Experiment Report",
        "",
        "Compares `origin/main` (PR #8) baselines against the MSE-improved pipeline on the **exact same splits**",
        "(zero test leakage, zero changes to existing repo files):",
        "",
        "- **Bangkok-bounded IDW Neighbourhood Price**: filters out corrupted coordinates outside Bangkok (`lat 13.5–14.0`, `lon 100.3–100.95`, excluding `project_id=3928` and `3953`) and weights nearby buildings by inverse distance.",
        "- **`log(price_per_sqm)` Target**: predicts `exp(y_hat) * area_sqm` to prevent exponential area-slope blow-up on large luxury condos.",
        "- **Compact Structural Features**: adds `floor_ratio`, `log_units_per_floor`, `is_high_rise`, `is_central_rail`, `log_area_per_br` and removes redundant K-Means tier double-counting.",
        "- **Pure NumPy Ridge Regression** (`src/models/ridge_regression.py`) + **Regularised HistGB** + **ExtraTrees** blended with CV-selected weights minimising THB RMSE/MSE.",
        "",
        f"- **Random split CV selection**: Ridge `lam={random_meta['ridge_lambda']}`, blend weights `{random_meta['cv_blend_weights']}`",
        f"- **Project split CV selection**: Ridge `lam={project_meta['ridge_lambda']}`, blend weights `{project_meta['cv_blend_weights']}`",
        "",
    ]
    lines.extend(_table("1. Random Split (exact seed=42 test set)", random_res, ref))
    lines.extend(_table("2. Project Split — Unseen Buildings (exact seed=42 test set)", project_res, ref))
    lines.extend(_table("3. 5-Fold GroupKFold CV — All Unseen Buildings (3,264 listings)", group_cv_res, ref))
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    all_listings = pd.read_csv(CLEAN_CSV, dtype={"listing_id": str})
    projects = pd.read_csv(PROJECTS_CSV)
    listings = all_listings[all_listings["page"] <= MAX_PAGES_PER_ZONE].reset_index(drop=True)
    listings = attach_project_fields(listings, projects)

    # 1. Exact split_random
    tr_r, te_r = split_random(listings)
    te_r = te_r[te_r["zone_no"].isin(set(tr_r["zone_no"].unique()))].reset_index(drop=True)
    r_base = baseline_predictions(tr_r, te_r)
    r_imp, r_meta = fit_and_predict_improved(tr_r, te_r, group_cv=False)
    random_results = {name: full_metrics(te_r["price_thb"].to_numpy(), p) for name, p in {**r_base, **r_imp}.items()}

    # 2. Exact split_by_project
    tr_p, te_p = split_by_project(listings)
    te_p = te_p[te_p["zone_no"].isin(set(tr_p["zone_no"].unique()))].reset_index(drop=True)
    p_base = baseline_predictions(tr_p, te_p)
    p_imp, p_meta = fit_and_predict_improved(tr_p, te_p, group_cv=True)
    project_results = {name: full_metrics(te_p["price_thb"].to_numpy(), p) for name, p in {**p_base, **p_imp}.items()}

    # 3. Full 5-fold GroupKFold CV
    group_cv_results = evaluate_group_cv(listings)

    payload = {
        "random_split_meta": r_meta,
        "project_split_meta": p_meta,
        "random_split": random_results,
        "project_split": project_results,
        "group_cv_5fold": group_cv_results,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "mse_improvement.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (REPORTS_DIR / "mse_improvement.md").write_text(
        render_markdown(random_results, project_results, group_cv_results, r_meta, p_meta),
        encoding="utf-8",
    )
    plot_mse_comparison(random_results, project_results, group_cv_results)

    for label, res in [
        ("RANDOM SPLIT", random_results),
        ("PROJECT SPLIT", project_results),
        ("5-FOLD GROUP CV", group_cv_results),
    ]:
        LOG.info("=== %s ===", label)
        for name, m in sorted(res.items(), key=lambda kv: kv[1]["rmse_thb"]):
            LOG.info(
                "%-48s | RMSE %5.2fM | MSE %5.2fT | MAE %5.2fM | MAPE %5.1f%% | R2 %.4f",
                name, m["rmse_thb"] / 1e6, m["mse_thb2"] / 1e12, m["mae_thb"] / 1e6, m["mape_pct"], m["r2"],
            )


if __name__ == "__main__":
    main()
