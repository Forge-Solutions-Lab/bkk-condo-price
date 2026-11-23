# MSE Improvement Experiment Report

Compares `origin/main` (PR #8) baselines against the MSE-improved pipeline on the **exact same splits**
(zero test leakage, zero changes to existing repo files):

- **Bangkok-bounded IDW Neighbourhood Price**: filters out corrupted coordinates outside Bangkok (`lat 13.5–14.0`, `lon 100.3–100.95`, excluding `project_id=3928` and `3953`) and weights nearby buildings by inverse distance.
- **`log(price_per_sqm)` Target**: predicts `exp(y_hat) * area_sqm` to prevent exponential area-slope blow-up on large luxury condos.
- **Compact Structural Features**: adds `floor_ratio`, `log_units_per_floor`, `is_high_rise`, `is_central_rail`, `log_area_per_br` and removes redundant K-Means tier double-counting.
- **Pure NumPy Ridge Regression** (`src/models/ridge_regression.py`) + **Regularised HistGB** + **ExtraTrees** blended with CV-selected weights minimising THB RMSE/MSE.

- **Random split CV selection**: Ridge `lam=1.0`, blend weights `{'ridge': 0.55, 'hgb': 0.45, 'et': 0.0}`
- **Project split CV selection**: Ridge `lam=1.0`, blend weights `{'ridge': 1.0, 'hgb': 0.0, 'et': 0.0}`

## 1. Random Split (exact seed=42 test set)

| Model | RMSE (M THB) | MSE (T THB²) | vs PR #8 MSE | MAE (M THB) | MAPE (%) | R² | Within 20% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ExtraTrees (clean IDW, log ppsqm) | 3.39 | 11.48 | -39.5% | 1.10 | 13.6% | 0.9098 | 80.5% |
| Balanced Blend (0.4 Ridge + 0.3 HGB + 0.3 ET) | 3.75 | 14.06 | -25.9% | 1.24 | 15.5% | 0.8896 | 73.3% |
| CV-MSE Optimal Blend (Ridge + HGB + ET) | 3.96 | 15.69 | -17.3% | 1.32 | 16.9% | 0.8768 | 71.1% |
| Tuned HistGB (clean IDW, log ppsqm) | 4.00 | 15.99 | -15.7% | 1.20 | 14.9% | 0.8743 | 75.7% |
| Baseline: Blend LR + GB (origin/main PR #8) | 4.36 | 18.97 | 0.0% | 1.40 | 16.3% | 0.8510 | 72.5% |
| Pure NumPy Ridge (clean IDW, log ppsqm) | 4.52 | 20.43 | +7.7% | 1.59 | 20.5% | 0.8395 | 61.7% |
| Baseline: LR + bld + neigh (origin/main) | 4.83 | 23.29 | +22.8% | 1.68 | 20.4% | 0.8170 | 61.5% |
| Baseline: LR + building (origin/main) | 5.20 | 27.04 | +42.6% | 1.76 | 21.3% | 0.7876 | 61.1% |

## 2. Project Split — Unseen Buildings (exact seed=42 test set)

| Model | RMSE (M THB) | MSE (T THB²) | vs PR #8 MSE | MAE (M THB) | MAPE (%) | R² | Within 20% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline: LR + building (origin/main) | 3.49 | 12.20 | -36.6% | 1.53 | 23.4% | 0.8738 | 53.9% |
| ExtraTrees (clean IDW, log ppsqm) | 4.37 | 19.10 | -0.7% | 1.74 | 21.7% | 0.8024 | 62.6% |
| Baseline: Blend LR + GB (origin/main PR #8) | 4.38 | 19.23 | 0.0% | 1.62 | 20.4% | 0.8011 | 65.5% |
| Balanced Blend (0.4 Ridge + 0.3 HGB + 0.3 ET) | 4.60 | 21.20 | +10.3% | 1.65 | 20.6% | 0.7807 | 64.3% |
| Tuned HistGB (clean IDW, log ppsqm) | 4.91 | 24.13 | +25.5% | 1.78 | 20.9% | 0.7503 | 60.1% |
| Baseline: LR + bld + neigh (origin/main) | 5.35 | 28.60 | +48.8% | 1.65 | 21.9% | 0.7041 | 60.0% |
| Pure NumPy Ridge (clean IDW, log ppsqm) | 5.60 | 31.35 | +63.0% | 1.72 | 22.7% | 0.6757 | 57.8% |
| CV-MSE Optimal Blend (Ridge + HGB + ET) | 5.60 | 31.35 | +63.0% | 1.72 | 22.7% | 0.6757 | 57.8% |

## 3. 5-Fold GroupKFold CV — All Unseen Buildings (3,264 listings)

| Model | RMSE (M THB) | MSE (T THB²) | vs PR #8 MSE | MAE (M THB) | MAPE (%) | R² | Within 20% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure NumPy Ridge (clean IDW, log ppsqm) | 4.85 | 23.55 | -22.4% | 1.57 | 21.2% | 0.8578 | 58.5% |
| Baseline: LR + bld + neigh (origin/main) | 4.98 | 24.75 | -18.4% | 1.64 | 21.4% | 0.8506 | 58.3% |
| CV-MSE Optimal Blend (Ridge + HGB + ET) | 5.04 | 25.35 | -16.4% | 1.57 | 21.1% | 0.8470 | 58.7% |
| Balanced Blend (0.4 Ridge + 0.3 HGB + 0.3 ET) | 5.49 | 30.12 | -0.7% | 1.52 | 18.9% | 0.8182 | 63.5% |
| Baseline: Blend LR + GB (origin/main PR #8) | 5.51 | 30.33 | 0.0% | 1.57 | 19.5% | 0.8169 | 62.6% |
| Baseline: LR + building (origin/main) | 5.69 | 32.33 | +6.6% | 1.85 | 23.1% | 0.8049 | 53.5% |
| Tuned HistGB (clean IDW, log ppsqm) | 5.98 | 35.74 | +17.8% | 1.57 | 18.8% | 0.7843 | 63.0% |
| ExtraTrees (clean IDW, log ppsqm) | 6.15 | 37.80 | +24.6% | 1.64 | 19.4% | 0.7718 | 63.5% |
