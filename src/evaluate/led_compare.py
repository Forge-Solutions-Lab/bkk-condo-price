"""Compare asking prices (Livinginsider) with court appraisal prices (Legal Execution Department)
by district, for the Discuss section. LED data is not used to train any model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import pandas as pd

from src.evaluate.run import ASH, CHARCOAL, GOLD, WHITE, new_figure, save_figure, style_axes

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LED_CSV = PROJECT_ROOT / "data" / "interim" / "led_raw.csv"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
OUTPUT_CSV = PROJECT_ROOT / "reports" / "led_vs_listing_by_district.csv"
THAI_FONTS = ["Thonburi", "Tahoma", "Ayuthaya", "Noto Sans Thai"]
MIN_UNITS = 3
MIN_AREA_SQM = 15.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    led = pd.read_csv(LED_CSV)
    led = led[(led["area_sqm"] >= MIN_AREA_SQM) & (led["appraisal_thb"] > 0)].copy()
    led["price_per_sqm"] = led["appraisal_thb"] / led["area_sqm"]
    listings = pd.read_csv(CLEAN_CSV)
    led_by_district = led.groupby("district_th")["price_per_sqm"].agg(led_units="count", led_median="median")
    listing_by_district = listings.groupby("district_th")["price_per_sqm"].agg(
        listing_units="count", listing_median="median"
    )
    both = led_by_district.join(listing_by_district, how="inner")
    both = both[(both["led_units"] >= MIN_UNITS) & (both["listing_units"] >= MIN_UNITS)].copy()
    both["led_to_listing_ratio"] = both["led_median"] / both["listing_median"]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    both.round(3).to_csv(OUTPUT_CSV)
    LOG.info("districts in both sources: %s, median ratio %.2f", len(both), both["led_to_listing_ratio"].median())
    if both.empty:
        return
    both = both.sort_values("listing_median")
    positions = np.arange(len(both))
    figure, axes = new_figure(9, max(4, 0.45 * len(both)))
    axes.barh(positions - 0.2, both["listing_median"] / 1000, height=0.4, color=GOLD, label="asking price (Livinginsider)")
    axes.barh(positions + 0.2, both["led_median"] / 1000, height=0.4, color=ASH, label="appraisal (LED auction)")
    # district names are Thai and matplotlib's default font has no Thai glyphs
    available = {font.name for font in font_manager.fontManager.ttflist}
    thai_font = next((name for name in THAI_FONTS if name in available), None)
    axes.set_yticks(positions, both.index.tolist(), fontfamily=thai_font)
    style_axes(axes, "Median price per sqm: asking vs court appraisal")
    axes.set_xlabel("thousand THB per sqm")
    axes.legend(facecolor=CHARCOAL, labelcolor=WHITE, edgecolor=CHARCOAL)
    save_figure(figure, "led_vs_listing.png")


if __name__ == "__main__":
    main()
