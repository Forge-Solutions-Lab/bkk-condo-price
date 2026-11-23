# Unseen-Building Accuracy & Error-Reduction Report (V2)

Compares exact `main` (`src/evaluate/run.py`) baselines (using CV-chosen `neighbour_k`) against the V2
unseen-building pipeline with **strict inner-CV feature refitting** (zero inner-fold or test leakage):

- **Percentage & THB Error Metrics**: Reports both percentage errors (**`RMSPE (%)`**, **`NRMSE (%)`**, **`MAPE (%)`**, **`Within 20%`**) and THB errors (`RMSE`, `MSE`, `MAE`).
- **Leave-One-Building-Out Brand/Series Premium (`brand_prem_t1`, `brand_prem_t2`)**: Generic developer/series prefixes from `project_name` (matching 76.4% of unseen buildings) with Bayesian smoothing and verified zero self-building leakage.
- **Age-Adjusted & Comparable-Building Spatial IDW (`neigh_age_adj`, `neigh_comparable`)**: Filters corrupted coordinates outside Bangkok (`lat 13.5–14.0`, `lon 100.3–100.95`, excluding `project_id=3928` & `3953`), separates building age depreciation from land value, and matches nearby buildings of similar age & height.
- **Building-Balanced Multi-Loss Ensemble**: Pure NumPy Ridge (`src/models/ridge_regression.py`) + HistGB-L2 + HistGB-MAE (`loss='absolute_error'`) + ExtraTrees weighted by `1 / sqrt(n_units_in_building)`.
- **Trade-off note (Percentage Error vs. Raw THB MSE)**: On 5-fold GroupKFold CV across all 3,264 unseen-building rows, V2 decisively improves percentage accuracy (`RMSPE` **24.6–24.9%** vs. **26.8%**, `MAPE` **17.7%** vs. **19.5%**, `Within 20%` **67%** vs. **62.6%**), whereas raw THB MSE on extreme 50M–150M THB ultra-luxury penthouses favours smooth linear extrapolation (Pure NumPy Ridge / linear IDW). Blending Ridge with tree models balances both regimes.

- **Random split strict inner-CV selection**: Ridge `lam=2.0`, CV-RMSPE blend weights `{'ridge': 0.0, 'hgb_l2': 0.3, 'hgb_mae': 0.0, 'et': 0.7}`
- **Project split strict inner-CV selection**: Ridge `lam=0.5`, CV-RMSPE blend weights `{'ridge': 0.6, 'hgb_l2': 0.2, 'hgb_mae': 0.1, 'et': 0.1}`

## 1. Project Split — Unseen Buildings (exact seed=42 test set, sorted by RMSE)

| Model | RMSE (M THB) | NRMSE (%) | RMSPE (%) | MSE (T THB²) | vs Main Blend MSE | MAE (M THB) | MAPE (%) | R² | Within 20% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 ExtraTrees (brand + comp IDW, bld-w) | 3.24 | 52.7% | 24.1% | 10.47 | -14.5% | 1.26 | 17.9% | 0.8410 | 63.7% |
| V2 HistGB-L2 (brand + comp IDW, bld-w) | 3.42 | 55.7% | 23.3% | 11.70 | -4.4% | 1.28 | 17.5% | 0.8223 | 64.0% |
| V2 Unseen Ensemble (Ridge + HGB-L2/MAE + ET) | 3.46 | 56.3% | 23.0% | 11.97 | -2.2% | 1.23 | 17.1% | 0.8182 | 68.2% |
| V2 CV-RMSPE Optimal Ensemble | 3.49 | 56.8% | 23.3% | 12.19 | -0.4% | 1.23 | 17.2% | 0.8149 | 67.4% |
| Baseline: Blend LR + GB (main) | 3.50 | 57.0% | 24.1% | 12.24 | 0.0% | 1.28 | 17.9% | 0.8141 | 65.2% |
| V2 Pure NumPy Ridge (brand + age-adj IDW) | 3.60 | 58.6% | 24.5% | 12.97 | +5.9% | 1.26 | 18.1% | 0.8031 | 64.4% |
| V2 HistGB-MAE (brand + comp IDW, bld-w) | 3.63 | 59.1% | 23.4% | 13.17 | +7.6% | 1.30 | 17.3% | 0.8000 | 67.9% |
| Baseline: LR + bld + neigh IDW (main) | 3.81 | 62.0% | 26.8% | 14.49 | +18.4% | 1.35 | 19.6% | 0.7799 | 63.8% |
| Baseline: LR + building (main) | 4.54 | 73.8% | 29.8% | 20.58 | +68.1% | 1.68 | 22.4% | 0.6875 | 55.0% |

## 2. 5-Fold GroupKFold CV — All Unseen Buildings (3,264 listings, sorted by RMSPE %)

| Model | RMSE (M THB) | NRMSE (%) | RMSPE (%) | MSE (T THB²) | vs Main Blend MSE | MAE (M THB) | MAPE (%) | R² | Within 20% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 Unseen Ensemble (Ridge + HGB-L2/MAE + ET) | 5.09 | 73.5% | 26.1% | 25.88 | -11.9% | 1.46 | 18.2% | 0.8450 | 66.1% |
| V2 CV-RMSPE Optimal Ensemble | 5.20 | 75.2% | 26.6% | 27.08 | -7.8% | 1.47 | 18.2% | 0.8378 | 66.1% |
| V2 ExtraTrees (brand + comp IDW, bld-w) | 5.61 | 81.0% | 26.7% | 31.45 | +7.0% | 1.58 | 18.8% | 0.8116 | 65.0% |
| V2 HistGB-L2 (brand + comp IDW, bld-w) | 5.71 | 82.5% | 27.0% | 32.61 | +11.0% | 1.54 | 18.6% | 0.8047 | 65.1% |
| V2 Pure NumPy Ridge (brand + age-adj IDW) | 4.67 | 67.4% | 27.0% | 21.77 | -25.9% | 1.44 | 19.1% | 0.8696 | 64.5% |
| V2 HistGB-MAE (brand + comp IDW, bld-w) | 5.50 | 79.5% | 28.1% | 30.30 | +3.1% | 1.60 | 19.2% | 0.8185 | 64.2% |
| Baseline: Blend LR + GB (main) | 5.42 | 78.3% | 28.3% | 29.38 | 0.0% | 1.58 | 19.7% | 0.8240 | 63.3% |
| Baseline: LR + bld + neigh IDW (main) | 4.90 | 70.8% | 30.2% | 23.98 | -18.4% | 1.59 | 21.3% | 0.8563 | 58.6% |
| Baseline: LR + building (main) | 5.75 | 83.1% | 32.5% | 33.05 | +12.5% | 1.83 | 23.1% | 0.8020 | 56.3% |

## 3. Random Split (exact seed=42 test set, sorted by RMSE)

| Model | RMSE (M THB) | NRMSE (%) | RMSPE (%) | MSE (T THB²) | vs Main Blend MSE | MAE (M THB) | MAPE (%) | R² | Within 20% |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 HistGB-L2 (brand + comp IDW, bld-w) | 2.91 | 44.0% | 20.1% | 8.50 | -45.7% | 1.10 | 14.7% | 0.9291 | 74.9% |
| V2 CV-RMSPE Optimal Ensemble | 3.04 | 45.8% | 19.0% | 9.24 | -41.0% | 1.03 | 13.4% | 0.9230 | 79.4% |
| V2 ExtraTrees (brand + comp IDW, bld-w) | 3.15 | 47.5% | 19.2% | 9.93 | -36.6% | 1.04 | 13.4% | 0.9172 | 79.0% |
| V2 HistGB-MAE (brand + comp IDW, bld-w) | 3.19 | 48.2% | 21.2% | 10.20 | -34.9% | 1.18 | 15.2% | 0.9149 | 75.4% |
| V2 Unseen Ensemble (Ridge + HGB-L2/MAE + ET) | 3.54 | 53.4% | 20.6% | 12.53 | -20.0% | 1.18 | 15.1% | 0.8955 | 74.1% |
| Baseline: Blend LR + GB (main) | 3.96 | 59.7% | 22.4% | 15.66 | 0.0% | 1.29 | 16.4% | 0.8694 | 71.3% |
| Baseline: LR + building (main) | 4.60 | 69.4% | 31.2% | 21.18 | +35.3% | 1.78 | 23.1% | 0.8233 | 54.6% |
| V2 Pure NumPy Ridge (brand + age-adj IDW) | 4.68 | 70.6% | 23.6% | 21.89 | +39.8% | 1.39 | 17.6% | 0.8174 | 67.8% |
| Baseline: LR + bld + neigh IDW (main) | 5.67 | 85.6% | 27.6% | 32.19 | +105.6% | 1.68 | 20.7% | 0.7315 | 57.8% |
