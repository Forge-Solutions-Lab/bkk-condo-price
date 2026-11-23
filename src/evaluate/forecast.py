"""Price trend forecast to 2036 from the Bangkok residential property price index.

The condo valuation model knows prices on one day only (23 Nov 2026). To say what a unit
may cost in a later year, we multiply today's estimate by how much the market index is
forecast to grow. The trend is a Linear Regression of log(index) on time, fitted on the
most recent W years. W is not chosen by hand: a backtest replays the method from many past
start years, forecasts 1-10 years ahead, and keeps the window with the smallest error.
The same backtest errors give the uncertainty band shown for each forecast year.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.evaluate.run import ASH, CHARCOAL, GOLD, WHITE, new_figure, save_figure, style_axes
from src.models import linear_regression as lr

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_CSV = PROJECT_ROOT / "data" / "external" / "QTHN628BIS.csv"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "forecast.json"

BASE_YEAR = 2026
TARGET_YEAR = 2036
HORIZON = TARGET_YEAR - BASE_YEAR
WINDOWS = [5, 10, 15, 20, None]  # years of history used for the trend; None = all years
FIRST_ORIGIN = 2001
BAND_PERCENTILE = 90


def yearly_index() -> pd.Series:
    quarterly = pd.read_csv(INDEX_CSV, parse_dates=["observation_date"])
    quarterly["year"] = quarterly["observation_date"].dt.year
    # BASE_YEAR has only the quarters published so far; their mean is today's level
    return quarterly.groupby("year")["QTHN628BIS"].mean()


def fit_trend(index: pd.Series, last_year: int, window: int | None) -> np.ndarray:
    history = index.loc[: last_year]
    if window is not None:
        history = history.loc[last_year - window + 1:]
    years = history.index.to_numpy(dtype=float)[:, None]
    return lr.fit(years, np.log(history.to_numpy()))


def predict_index(W: np.ndarray, years: np.ndarray) -> np.ndarray:
    return np.exp(lr.predict(years.astype(float)[:, None], W))


def backtest(index: pd.Series, window: int | None) -> pd.DataFrame:
    """Forecast from every past origin year and record the log error h years ahead.
    Forecasts are rescaled to pass through the origin year's actual level, as in use."""
    rows = []
    for origin in range(FIRST_ORIGIN, BASE_YEAR):
        W = fit_trend(index, origin, window)
        for h in range(1, HORIZON + 1):
            year = origin + h
            if year > BASE_YEAR:
                break
            growth = predict_index(W, np.array([year]))[0] / predict_index(W, np.array([origin]))[0]
            forecast = index[origin] * growth
            rows.append({"origin": origin, "horizon": h, "log_error": float(np.log(forecast / index[year]))})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    index = yearly_index()
    results = {}
    for window in WINDOWS:
        errors = backtest(index, window)
        results[window] = {"errors": errors, "mape_pct": float(100 * np.mean(np.abs(np.expm1(errors["log_error"]))))}
        LOG.info("window %-4s backtest MAPE %.2f%% over %s forecasts", window or "all", results[window]["mape_pct"], len(errors))
    best_window = min(results, key=lambda window: results[window]["mape_pct"])
    errors = results[best_window]["errors"]

    W = fit_trend(index, BASE_YEAR, best_window)
    years = np.arange(BASE_YEAR, TARGET_YEAR + 1)
    growth = predict_index(W, years) / predict_index(W, np.array([BASE_YEAR]))[0]
    band = errors.groupby("horizon")["log_error"].apply(lambda e: float(np.percentile(np.abs(e), BAND_PERCENTILE)))
    table = []
    for year, factor in zip(years, growth):
        h = int(year - BASE_YEAR)
        # beyond the longest horizon the backtest could check, reuse the widest observed band
        width = float(band.get(h, band.iloc[-1])) if h > 0 else 0.0
        table.append({"year": int(year), "factor": float(factor), "low": float(factor * np.exp(-width)),
                      "high": float(factor * np.exp(width)), "backtest_count": int((errors["horizon"] == h).sum())})

    annual_growth = float(np.exp(W[0]) - 1)
    output = {
        "source": "BIS residential property prices for Bangkok, nominal, 2010=100 (FRED series QTHN628BIS)",
        "base_year": BASE_YEAR,
        "window_years": best_window or "all",
        "annual_growth_pct": 100 * annual_growth,
        "band_percentile": BAND_PERCENTILE,
        "backtest_mape_pct_by_window": {str(window or "all"): results[window]["mape_pct"] for window in WINDOWS},
        "backtest_mape_pct_by_horizon": {
            int(h): float(100 * np.mean(np.abs(np.expm1(group)))) for h, group in errors.groupby("horizon")["log_error"]
        },
        "forecast": table,
        "index_history": {int(year): float(value) for year, value in index.items()},
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    plot(index, table, best_window, annual_growth)
    LOG.info("chosen window %s years, trend %.2f%% per year, factor %s: %.3f (%.3f-%.3f)", best_window or "all",
             100 * annual_growth, TARGET_YEAR, table[-1]["factor"], table[-1]["low"], table[-1]["high"])


def plot(index: pd.Series, table: list[dict[str, float]], window: int | None, annual_growth: float) -> None:
    figure, axes = new_figure(9, 5)
    axes.plot(index.index, index.values, color=WHITE, label="index (actual)")
    years = [row["year"] for row in table]
    base = index[BASE_YEAR]
    axes.plot(years, [base * row["factor"] for row in table], color=GOLD, linestyle="--", label="forecast")
    axes.fill_between(years, [base * row["low"] for row in table], [base * row["high"] for row in table],
                      color=GOLD, alpha=0.2, label=f"{BAND_PERCENTILE}% backtest band")
    style_axes(axes, f"Bangkok residential price index: trend {100 * annual_growth:.1f}%/yr (last {window or 'all'} years)")
    axes.set_xlabel("year")
    axes.set_ylabel("index (2010 = 100)")
    axes.legend(facecolor=CHARCOAL, labelcolor=WHITE, edgecolor=CHARCOAL, loc="upper left")
    axes.tick_params(colors=ASH)
    save_figure(figure, "price_index_forecast.png")


if __name__ == "__main__":
    main()
