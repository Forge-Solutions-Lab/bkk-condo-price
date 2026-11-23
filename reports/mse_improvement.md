# MSE & Unseen-Building Accuracy Improvement Report (V2)

Compares `origin/main` (PR #8) baselines against the V2 unseen-building pipeline on the **exact same splits**
(zero test leakage, zero modifications to existing `main` files):

- **Percentage RMSE Metrics**: Added **`RMSPE (%)`** (Root Mean Squared Percentage Error) and **`NRMSE (%)`** (RMSE / mean price × 100%) alongside THB `RMSE` and `MSE`.
- **Leave-One-Building-Out Brand/Series Premium (`brand_prem_t1`, `brand_prem_t2`)**: Captures developer/series price premium above neighbourhood from `project_name` prefixes (matching 76.4% of unseen buildings) with Bayesian smoothing and strict zero self-building leakage.
- **Age-Adjusted & Comparable-Building Spatial IDW (`neigh_age_adj`, `neigh_comparable`)**: Filters corrupted coordinates outside Bangkok (`lat 13.5–14.0`, `lon 100.3–100.95`, excluding `project_id=3928` & `3953`), separates building depreciation from land value, and matches nearby buildings of similar age & height.
- **Building-Balanced Multi-Loss Ensemble**: Pure NumPy Ridge (`src/models/ridge_regression.py`, tuned via CV `log-RMSE`) + HistGB-L2 + HistGB-MAE (`loss='absolute_error'`) + ExtraTrees weighted by `1 / sqrt(n_units_in_building)`.

- **Random split CV selection**: Ridge `lam=5.0`, CV-RMSPE blend weights `{'ridge': 0.0, 'hgb_l2': 0.3, 'hgb_mae': 0.0, 'et': 0.7}`
- **Project split CV selection**: Ridge `lam=5.0`, CV-RMSPE blend weights `{'ridge': 0.3, 'hgb_l2': 0.7, 'hgb_mae': 0.0, 'et': 0.0}`

## 1. Project Split — Unseen Buildings (exact seed=42 test set, sorted by RMSE)

| Model | RMSE (M THB) | NRMSE (%) | RMSPE (%) | MSE (T THB²) | vs PR #8 MSE | MAE (M THB) | MAPE (%) | R² | Within 20% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 HistGB-MAE (brand + comp IDW, bld-w) | 3.39 | 49.5% | 29.5% | 11.51 | -40.1% | 1.37 | 18.3% | 0.8809 | 68.8% |
| Baseline: LR + building (origin/main) | 3.49 | 51.0% | 31.5% | 12.20 | -36.6% | 1.53 | 23.4% | 0.8738 | 53.9% |
| V2 Unseen Ensemble (Ridge + HGB-L2/MAE + ET) | 4.01 | 58.6% | 28.0% | 16.12 | -16.2% | 1.39 | 18.1% | 0.8332 | 70.8% |
| V2 CV-RMSPE Optimal Ensemble | 4.04 | 59.0% | 28.2% | 16.36 | -14.9% | 1.35 | 17.7% | 0.8308 | 70.3% |
| V2 HistGB-L2 (brand + comp IDW, bld-w) | 4.08 | 59.5% | 29.0% | 16.62 | -13.6% | 1.39 | 18.0% | 0.8281 | 68.6% |
| Baseline: Blend LR + GB (origin/main PR #8) | 4.38 | 64.0% | 29.9% | 19.23 | 0.0% | 1.62 | 20.4% | 0.8011 | 65.5% |
| V2 Pure NumPy Ridge (brand + age-adj IDW) | 4.50 | 65.6% | 28.1% | 20.23 | +5.2% | 1.43 | 19.1% | 0.7907 | 67.1% |
| V2 ExtraTrees (brand + comp IDW, bld-w) | 4.72 | 68.9% | 30.0% | 22.31 | +16.0% | 1.70 | 19.7% | 0.7692 | 67.4% |
| Baseline: LR + bld + neigh (origin/main) | 5.35 | 78.0% | 31.5% | 28.60 | +48.8% | 1.65 | 21.9% | 0.7041 | 60.0% |

## 2. 5-Fold GroupKFold CV — All Unseen Buildings (3,264 listings, sorted by RMSPE %)

| Model | RMSE (M THB) | NRMSE (%) | RMSPE (%) | MSE (T THB²) | vs PR #8 MSE | MAE (M THB) | MAPE (%) | R² | Within 20% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 HistGB-L2 (brand + comp IDW, bld-w) | 6.01 | 86.9% | 24.7% | 36.09 | +19.0% | 1.50 | 17.8% | 0.7822 | 66.7% |
| V2 CV-RMSPE Optimal Ensemble | 5.63 | 81.5% | 24.8% | 31.74 | +4.7% | 1.48 | 17.7% | 0.8084 | 67.0% |
| V2 Unseen Ensemble (Ridge + HGB-L2/MAE + ET) | 5.48 | 79.3% | 24.9% | 30.04 | -1.0% | 1.47 | 17.7% | 0.8187 | 67.0% |
| V2 HistGB-MAE (brand + comp IDW, bld-w) | 5.76 | 83.4% | 25.3% | 33.23 | +9.6% | 1.54 | 18.1% | 0.7994 | 66.4% |
| V2 ExtraTrees (brand + comp IDW, bld-w) | 5.85 | 84.6% | 25.6% | 34.25 | +12.9% | 1.56 | 18.3% | 0.7933 | 65.4% |
| Baseline: Blend LR + GB (origin/main PR #8) | 5.51 | 79.6% | 26.8% | 30.33 | 0.0% | 1.57 | 19.5% | 0.8169 | 62.6% |
| V2 Pure NumPy Ridge (brand + age-adj IDW) | 5.16 | 74.6% | 26.8% | 26.62 | -12.2% | 1.50 | 19.0% | 0.8393 | 63.1% |
| Baseline: LR + bld + neigh (origin/main) | 4.98 | 71.9% | 29.2% | 24.75 | -18.4% | 1.64 | 21.4% | 0.8506 | 58.3% |
| Baseline: LR + building (origin/main) | 5.69 | 82.2% | 30.8% | 32.33 | +6.6% | 1.85 | 23.1% | 0.8049 | 53.5% |

## 3. Random Split (exact seed=42 test set, sorted by RMSE)

| Model | RMSE (M THB) | NRMSE (%) | RMSPE (%) | MSE (T THB²) | vs PR #8 MSE | MAE (M THB) | MAPE (%) | R² | Within 20% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 ExtraTrees (brand + comp IDW, bld-w) | 3.39 | 49.0% | 20.7% | 11.47 | -39.5% | 1.12 | 13.6% | 0.9099 | 79.8% |
| V2 CV-RMSPE Optimal Ensemble | 3.49 | 50.6% | 20.9% | 12.20 | -35.7% | 1.15 | 13.7% | 0.9041 | 80.0% |
| V2 HistGB-L2 (brand + comp IDW, bld-w) | 3.94 | 57.1% | 22.7% | 15.56 | -18.0% | 1.28 | 14.9% | 0.8778 | 75.6% |
| V2 Unseen Ensemble (Ridge + HGB-L2/MAE + ET) | 3.99 | 57.7% | 22.1% | 15.90 | -16.2% | 1.26 | 14.8% | 0.8751 | 78.0% |
| V2 HistGB-MAE (brand + comp IDW, bld-w) | 4.04 | 58.5% | 22.5% | 16.35 | -13.8% | 1.24 | 14.8% | 0.8715 | 76.3% |
| Baseline: Blend LR + GB (origin/main PR #8) | 4.36 | 63.0% | 23.6% | 18.97 | 0.0% | 1.40 | 16.3% | 0.8510 | 72.5% |
| V2 Pure NumPy Ridge (brand + age-adj IDW) | 4.43 | 64.1% | 24.4% | 19.63 | +3.5% | 1.41 | 16.8% | 0.8458 | 70.5% |
| Baseline: LR + bld + neigh (origin/main) | 4.83 | 69.8% | 28.8% | 23.29 | +22.8% | 1.68 | 20.4% | 0.8170 | 61.5% |
| Baseline: LR + building (origin/main) | 5.20 | 75.3% | 30.2% | 27.04 | +42.6% | 1.76 | 21.3% | 0.7876 | 61.1% |
