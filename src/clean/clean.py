"""Label and clean the parsed listings.

Every rule that changes or removes rows is counted, so the cleaning steps can be shown
in the presentation with real numbers.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "interim" / "listings_raw.csv"
ZONE_LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "zone_to_district.csv"
LABELED_CSV = PROJECT_ROOT / "data" / "labeled" / "listings_labeled.csv"
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "listings_clean.csv"
CLEANING_LOG = PROJECT_ROOT / "reports" / "cleaning_log.json"

STUDIO_TEXT = "สตูดิโอ"
DOWN_PAYMENT_BADGE = "ขายดาวน์"
NEW_UNIT_BADGE = "มือ 1"
SALE_AND_RENT_BADGE = "ขายและเช่า"

# Sanity bounds for a Bangkok condo unit. Values outside are almost always typing errors
# (for example price entered in thousands, or area entered in square wa).
MIN_AREA_SQM = 15.0
MAX_AREA_SQM = 500.0
MIN_PRICE_THB = 500_000.0
IQR_MULTIPLIER = 1.5
# the same unit posted again by another agent: same building, floor and bedrooms, area within
# 1 sqm and price within 3% (an exact re-post is already removed by the rule before it)
TWIN_AREA_SQM = 1.0
TWIN_PRICE_SHARE = 0.03


def parse_price(text: str) -> float:
    cleaned = str(text).replace("฿", "").replace(",", "").strip().upper()
    match = re.fullmatch(r"([\d.]+)\s*([MK]?)", cleaned)
    if match is None:
        return np.nan
    value = float(match.group(1))
    multiplier = {"M": 1_000_000.0, "K": 1_000.0, "": 1.0}[match.group(2)]
    return value * multiplier


def parse_area(text: str) -> float:
    match = re.search(r"[\d.]+", str(text).replace(",", ""))
    return float(match.group(0)) if match else np.nan


def parse_bedrooms(text: str) -> float:
    value = str(text).strip()
    if value == STUDIO_TEXT:
        return 0.0
    return float(value) if value.isdigit() else np.nan


def parse_floor(text: str) -> tuple[float, bool]:
    numbers = [float(number) for number in re.findall(r"\d+", str(text))]
    if not numbers:
        return np.nan, False
    # The site often shows a floor band such as "11-20" instead of the exact floor.
    # The band midpoint keeps the ordering information that matters for price.
    is_range = len(numbers) >= 2
    return (sum(numbers[:2]) / 2.0 if is_range else numbers[0]), is_range


def label_listings(raw: pd.DataFrame, zone_labels: pd.DataFrame) -> pd.DataFrame:
    labeled = raw.copy()
    labeled["price_thb"] = labeled["price_raw"].map(parse_price)
    labeled["area_sqm"] = labeled["area_raw"].map(parse_area)
    labeled["bedrooms"] = labeled["bedrooms_raw"].map(parse_bedrooms)
    labeled["bathrooms"] = pd.to_numeric(labeled["bathrooms_raw"], errors="coerce")
    floors = labeled["floor_raw"].map(parse_floor)
    labeled["floor"] = floors.map(lambda item: item[0])
    labeled["floor_is_range"] = floors.map(lambda item: item[1])
    badges = labeled["badges"].fillna("")
    labeled["is_studio"] = labeled["bedrooms"] == 0
    labeled["is_new_unit"] = badges.str.contains(NEW_UNIT_BADGE, regex=False)
    labeled["is_down_payment"] = badges.str.contains(DOWN_PAYMENT_BADGE, regex=False)
    labeled["is_sale_and_rent"] = badges.str.contains(SALE_AND_RENT_BADGE, regex=False)
    labeled["room_type"] = pd.cut(
        labeled["bedrooms"], bins=[-1, 0, 1, 2, np.inf], labels=["studio", "1br", "2br", "3br_plus"]
    ).astype(str)
    labeled = labeled.merge(
        zone_labels[["zone_no", "zone_name_th", "district_th", "label_confidence"]], on="zone_no", how="left"
    )
    labeled["price_per_sqm"] = labeled["price_thb"] / labeled["area_sqm"]
    return labeled


def normalised_name(names: pd.Series) -> pd.Series:
    return names.fillna("").str.replace(r"[\s\-]+", "", regex=True).str.lower()


def twin_groups(listings: pd.DataFrame) -> np.ndarray:
    """Group id per row; listings of one unit posted more than once share an id. Rows without a
    project name are never grouped, because there is no building to compare within."""
    parent = np.arange(len(listings))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys = normalised_name(listings["project_name"]).to_numpy()
    area, floor, price, bedrooms = (listings[c].to_numpy(dtype=float) for c in ("area_sqm", "floor", "price_thb", "bedrooms"))
    for key in pd.unique(keys[keys != ""]):
        rows = np.flatnonzero(keys == key)
        for a, i in enumerate(rows):
            for j in rows[a + 1:]:
                same_floor = floor[i] == floor[j] or (np.isnan(floor[i]) and np.isnan(floor[j]))
                if (same_floor and bedrooms[i] == bedrooms[j] and abs(area[i] - area[j]) <= TWIN_AREA_SQM
                        and abs(price[i] - price[j]) <= TWIN_PRICE_SHARE * price[i]):
                    parent[root(j)] = root(i)
    return np.array([root(i) for i in range(len(listings))])


def unlink_area_names(frame: pd.DataFrame, log: list[dict[str, object]]) -> pd.DataFrame:
    """Some agents type the zone name (for example "มีนบุรี-ร่มเกล้า") as the project. Those rows
    are different buildings, so the name is cleared instead of treating them as one building."""
    is_area_name = normalised_name(frame["project_name"]) == normalised_name(frame["zone_name_th"])
    frame = frame.copy()
    frame["project_is_area_name"] = is_area_name
    frame.loc[is_area_name, "project_name"] = ""
    log.append({"rule": "project name is the zone name: name cleared (rows kept)", "changed": int(is_area_name.sum()),
                "remaining": len(frame)})
    LOG.info("project name is the zone name: cleared %s", int(is_area_name.sum()))
    return frame


def drop_rows(frame: pd.DataFrame, mask: pd.Series, rule: str, log: list[dict[str, object]]) -> pd.DataFrame:
    log.append({"rule": rule, "removed": int(mask.sum()), "remaining": int((~mask).sum())})
    LOG.info("%s: removed %s, remaining %s", rule, int(mask.sum()), int((~mask).sum()))
    return frame.loc[~mask].copy()


def clean_listings(labeled: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    log: list[dict[str, object]] = [{"rule": "start", "removed": 0, "remaining": len(labeled)}]
    frame = drop_rows(labeled, labeled.duplicated("listing_id"), "duplicate listing id across pages", log)
    frame = drop_rows(
        frame,
        frame.duplicated(["project_name", "area_sqm", "price_thb", "floor_raw", "bedrooms"]),
        "same unit re-posted (project, area, price, floor, bedrooms)",
        log,
    )
    frame = unlink_area_names(frame, log)
    frame = drop_rows(frame, pd.Series(twin_groups(frame), index=frame.index).duplicated(),
                      "same unit posted again (same building, floor, bedrooms; area +-1 sqm; price +-3%)", log)
    frame = drop_rows(frame, frame["is_down_payment"], "down-payment listing (price is not full price)", log)
    required = ["price_thb", "area_sqm", "bedrooms", "bathrooms", "district_th"]
    frame = drop_rows(frame, frame[required].isna().any(axis=1), "missing required field", log)
    frame = drop_rows(
        frame,
        ~frame["area_sqm"].between(MIN_AREA_SQM, MAX_AREA_SQM),
        f"area outside {MIN_AREA_SQM:g}-{MAX_AREA_SQM:g} sqm",
        log,
    )
    frame = drop_rows(frame, frame["price_thb"] < MIN_PRICE_THB, f"price below {MIN_PRICE_THB:,.0f} THB", log)
    # Price per sqm differs several times between central and suburban zones, so the
    # outlier fence is computed inside each zone; a global fence would cut whole zones.
    log_price_per_sqm = np.log(frame["price_per_sqm"])
    grouped = log_price_per_sqm.groupby(frame["zone_no"])
    q1 = grouped.transform(lambda values: values.quantile(0.25))
    q3 = grouped.transform(lambda values: values.quantile(0.75))
    spread = IQR_MULTIPLIER * (q3 - q1)
    outlier = (log_price_per_sqm < q1 - spread) | (log_price_per_sqm > q3 + spread)
    frame = drop_rows(frame, outlier, "price per sqm outlier (1.5 IQR on log scale, per zone)", log)
    frame["floor"] = frame["floor"].fillna(frame["floor"].median())
    return frame, log


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw = pd.read_csv(RAW_CSV, dtype={"listing_id": str})
    zone_labels = pd.read_csv(ZONE_LABELS_CSV)
    labeled = label_listings(raw, zone_labels)
    LABELED_CSV.parent.mkdir(parents=True, exist_ok=True)
    labeled.drop(columns=["listing_url"]).to_csv(LABELED_CSV, index=False)
    clean, log = clean_listings(labeled)
    columns = [
        "listing_id", "zone_no", "zone_slug", "page", "zone_name_th", "district_th", "label_confidence",
        "project_name", "project_is_area_name", "room_type", "bedrooms", "bathrooms", "floor", "floor_is_range", "area_sqm",
        "is_new_unit", "is_sale_and_rent", "price_thb", "price_per_sqm",
    ]
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean[columns].to_csv(CLEAN_CSV, index=False)
    CLEANING_LOG.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("clean rows=%s zones=%s -> %s", len(clean), clean["zone_no"].nunique(), CLEAN_CSV)


if __name__ == "__main__":
    main()
