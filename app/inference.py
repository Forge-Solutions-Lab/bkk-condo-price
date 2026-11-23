"""Pure-Python inference and presentation helpers shared by Streamlit and the Vercel FastAPI backend."""

from __future__ import annotations

from functools import lru_cache
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The pickled models are instances of classes in src/, so the project root must be importable.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build import (  # noqa: E402
    BASE_YEAR, ZoneClustering, attach_buildings, clean_coordinates, feature_frame, project_key,
)
from src.models import ffbp  # noqa: E402
from src.models import linear_regression as lr  # noqa: E402
from src.models.knn_regression import knn, neighbors  # noqa: E402

BUNDLE_PATH = PROJECT_ROOT / "models" / "model_bundle.pkl"
FORECAST_PATH = PROJECT_ROOT / "reports" / "forecast.json"
BEDROOM_OPTIONS = {
    "สตูดิโอ": 0,
    "1 ห้องนอน": 1,
    "2 ห้องนอน": 2,
    "3 ห้องนอน": 3,
    "4 ห้องนอน": 4,
    "5 ห้องนอน": 5,
}
SIMILAR_COUNT = 5
NO_PROJECT = "ไม่ระบุโครงการ"
# A zone with fewer test units than this has no reliable zone-level accuracy figure.
MIN_ZONE_TEST_UNITS = 10
# A predicted-price band whose test accuracy is below this gets a warning.
WEAK_BAND_WITHIN_20 = 60.0
# A zone whose median error is beyond this is priced too low or too high on average.
ZONE_BIAS_PCT = 10.0
# "New unit" in a building finished this many years ago or more is unusual.
NEW_UNIT_MAX_BUILDING_AGE = 3
# Zones with prominent 30-year leasehold luxury condos (priced well below neighbouring freehold).
LEASEHOLD_ADVISORY_ZONES = {5}


@lru_cache(maxsize=1)
def load_bundle_cached() -> dict[str, Any]:
    with BUNDLE_PATH.open("rb") as handle:
        return pickle.load(handle)


@lru_cache(maxsize=1)
def load_forecast_cached() -> dict[str, Any] | None:
    if not FORECAST_PATH.exists():
        return None
    return json.loads(FORECAST_PATH.read_text(encoding="utf-8"))


def format_baht(value: float) -> str:
    return f"{value:,.0f} บาท"


def zone_label(row: pd.Series) -> str:
    return f"{row['zone_name_th']} (เขต{row['district_th']})"


def zone_labels_map(bundle: dict[str, Any]) -> dict[int, str]:
    zones = bundle["zone_table"].sort_values("zone_no")
    return {int(row["zone_no"]): zone_label(row) for _, row in zones.iterrows()}


def zone_centres(bundle: dict[str, Any]) -> pd.DataFrame:
    """Median coordinates of each zone's buildings. A unit with no project has no building
    coordinates; without a stand-in location its neighbourhood features look like those of the few
    training listings with no project, and KNN matched cheap units from other zones."""
    cached = bundle.get("_zone_centres")
    if cached is None:
        located = clean_coordinates(bundle["projects"]).dropna(subset=["latitude", "longitude"])
        zone_of = project_key(located["project_name"]).map(bundle.get("project_zones", {}))
        located = located.assign(zone_no=zone_of).dropna(subset=["zone_no"])
        cached = located.groupby(located["zone_no"].astype(int))[["latitude", "longitude"]].median()
        bundle["_zone_centres"] = cached
    return cached


def build_input(
    bundle: dict[str, Any],
    zone_no: int,
    area: float,
    bedrooms: int,
    bathrooms: int,
    floor: int,
    is_new: bool,
    project: str | None,
    complete_year: int,
    rail_km: float,
) -> pd.DataFrame:
    effective_zone = int(zone_no)
    if project:
        expected_zone = bundle.get("project_zones", {}).get(project_key(pd.Series([project])).iloc[0])
        if expected_zone is not None and int(expected_zone) != effective_zone:
            effective_zone = int(expected_zone)
    listing = pd.DataFrame([{
        "zone_no": effective_zone,
        "requested_zone_no": int(zone_no),
        "area_sqm": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floor": floor,
        "is_new_unit": is_new,
        "project_name": project or "",
    }])
    listing = attach_buildings(listing, bundle["projects"])
    if not listing["has_building_info"].iloc[0]:
        # No saved project page: use what the user typed; other building fields (including mall_km)
        # fall back to the training medians via manual_building_input=True instead of 3 km cap.
        listing[["complete_year", "nearest_rail_km"]] = [complete_year, rail_km]
        listing["has_building_info"] = True
        listing["manual_building_input"] = True
    else:
        listing["manual_building_input"] = False
    centres = zone_centres(bundle)
    no_location = listing[["latitude", "longitude"]].isna().any(axis=1).iloc[0]
    listing["location_from_zone"] = bool(no_location and effective_zone in centres.index)
    if listing["location_from_zone"].iloc[0]:
        listing[["latitude", "longitude"]] = centres.loc[effective_zone].to_numpy()
    return listing


def similar_rooms(bundle: dict[str, Any], knn_model: dict[str, Any], z: np.ndarray, zone_no: int) -> pd.DataFrame:
    """The KNN's closest training rooms within the unit's own zone (the whole data if the zone has
    too few rooms), so the table shows comparable rooms a buyer would recognise."""
    rows = bundle["reference_rows"]
    zone_name = bundle["zone_table"].set_index("zone_no")["zone_name_th"].get(zone_no)
    in_zone = np.flatnonzero((rows["zone_name_th"] == zone_name).to_numpy())
    if len(in_zone) < SIMILAR_COUNT:
        in_zone = np.arange(len(rows))
    index, _ = neighbors(knn_model["X"][in_zone], z, SIMILAR_COUNT, knn_model["metric"])
    return rows.iloc[in_zone[index]]


def input_warnings(
    listing: pd.DataFrame,
    zone_labels: dict[int, str] | None = None,
) -> list[str]:
    row = listing.iloc[0]
    warnings: list[str] = []
    if "requested_zone_no" in listing.columns and int(row["requested_zone_no"]) != int(row["zone_no"]):
        chosen, actual = int(row["requested_zone_no"]), int(row["zone_no"])
        chosen_name = zone_labels.get(chosen, f"โซน {chosen}") if zone_labels else f"โซน {chosen}"
        actual_name = zone_labels.get(actual, f"โซน {actual}") if zone_labels else f"โซน {actual}"
        warnings.append(
            f"โครงการที่เลือกอยู่ใน '{actual_name}' (ไม่ใช่ '{chosen_name}') ระบบจึงปรับใช้โซนจริงของโครงการให้อัตโนมัติ"
        )
    if row["bedrooms"] >= 2 and row["area_sqm"] < 40:
        warnings.append("พื้นที่น้อยมากสำหรับจำนวนห้องนอนนี้ ผลอาจคลาดเคลื่อนสูง")
    if row["bathrooms"] > row["bedrooms"] + 2:
        warnings.append("จำนวนห้องน้ำมากกว่าห้องนอนผิดปกติ ผลอาจคลาดเคลื่อนสูง")
    if row["is_new_unit"] and pd.notna(row["complete_year"]) and BASE_YEAR - row["complete_year"] >= NEW_UNIT_MAX_BUILDING_AGE:
        warnings.append(
            f"เลือกห้องมือ 1 แต่ตึกสร้างเสร็จปี {int(row['complete_year'])} ซึ่งไม่ค่อยพบ "
            "ตรวจข้อมูลอีกครั้ง ถ้าเป็นห้องมือ 2 ให้เอาเครื่องหมายออก"
        )
    if int(row["zone_no"]) in LEASEHOLD_ADVISORY_ZONES:
        warnings.append(
            "ย่านวิทยุ-เพลินจิต-หลังสวน มีหลายโครงการเป็นสิทธิการเช่า (Leasehold 30 ปี) ซึ่งราคาต่ำกว่าแบบขายขาด "
            "(Freehold) ราว 30-50% โมเดลยังไม่ได้แยกประเภทกรรมสิทธิ์ หากเป็นห้อง Leasehold ควรเผื่อส่วนลดเพิ่ม"
        )
    return warnings


def blend_price(
    bundle: dict[str, Any],
    listing: pd.DataFrame,
    clustering: ZoneClustering,
    buildings: pd.DataFrame,
) -> dict[str, float]:
    """Weighted geometric mean of Linear Regression and the CV-chosen partner, using extended
    building features and multi-scale IDW neighbourhood price (matching src/evaluate/run.py)."""
    blend = bundle["blend"]
    if blend.get("mode") == "unseen_ensemble":
        parts = blend["ensemble"].predict_parts(listing, clustering).iloc[0]
        return {
            "blend": float(np.exp(parts["log_blend"])),
            "linear": float(np.exp(parts["log_lr"])),
            "partner": float(np.exp(parts["log_trees"])),
            "lr_weight": float(parts["lr_weight"]),
        }
    if blend.get("mode") == "extended_multiscale":
        ext_buildings = blend["extended_buildings"].transform(listing)
        neighbourhood = blend["multiscale_neighbourhood"].transform(listing)
        features = feature_frame(listing, clustering, buildings=ext_buildings, neighbourhood=neighbourhood)
        features = features[blend["feature_names"]].to_numpy()
        log_linear = float(lr.predict(features, blend["W_extended"])[0])
        log_partner = float(blend["gb_extended"].predict(features)[0]) + float(np.log(listing["area_sqm"].iloc[0]))
        w_lr = float(blend["w_lr"])
        return {
            "blend": float(np.exp(w_lr * log_linear + (1.0 - w_lr) * log_partner)),
            "linear": float(np.exp(log_linear)),
            "partner": float(np.exp(log_partner)),
        }
    neighbourhood = blend["neighbourhood"].transform(listing)
    features = feature_frame(listing, clustering, buildings=buildings, neighbourhood=neighbourhood)
    features = features[blend["feature_names"]].to_numpy()
    log_linear = float(lr.predict(features, blend["W_neighbour"])[0])
    if blend["partner"] == "FFBP":
        network = blend["ffbp"]
        scaled = network["scaler"].transform(features)
        log_partner = float(ffbp.predict(scaled, network["W0"], network["W1"])[0] * network["y_std"] + network["y_mean"])
    else:
        log_partner = float(blend["boosting"].predict(features)[0])
    return {
        "blend": float(np.exp((log_linear + log_partner) / 2)),
        "linear": float(np.exp(log_linear)),
        "partner": float(np.exp(log_partner)),
    }


def reliability_of(bundle: dict[str, Any], model: str, price: float, zone_no: int) -> dict[str, Any]:
    """Test-set accuracy for this kind of estimate: overall, for its predicted-price band and for its zone."""
    table = bundle["reliability"][model]
    bands = table["bands"]
    band_index = int(np.searchsorted([3e6, 5e6, 8e6, 15e6], price, side="right"))
    zone = table["zones"][table["zones"]["zone_no"] == zone_no]
    return {
        "test_units": table["test_units"],
        "within_20pct": table["within_20pct"],
        "band": bands.iloc[band_index],
        "zone": zone.iloc[0] if len(zone) and zone.iloc[0]["test_units"] >= MIN_ZONE_TEST_UNITS else None,
    }


def predict(bundle: dict[str, Any], listing: pd.DataFrame) -> dict[str, Any]:
    clustering = ZoneClustering(zone_table=bundle["zone_table"], n_clusters=bundle["n_clusters"])
    buildings = bundle["building_features"].transform(listing)
    building_features = feature_frame(listing, clustering, buildings=buildings)[bundle["building_feature_names"]].to_numpy()
    if "knn_extended" in bundle:
        # Same features as the unseen-building ensemble; the vote is on log price per sqm.
        knn_model = bundle["knn_extended"]
        knn_features = bundle["blend"]["ensemble"].features(listing, clustering)
        log_area = float(np.log(listing["area_sqm"].iloc[0]))
    else:
        knn_model, knn_features, log_area = bundle["knn_building"], building_features, 0.0
    z = knn_model["scaler"].transform(knn_features)[0]
    log_knn = knn(knn_model["X"], knn_model["y"], z, knn_model["k"], knn_model["metric"], knn_model["weight"]) + log_area
    vote = "ถ่วงน้ำหนักตามระยะ" if knn_model["weight"] == "distance" else "โหวตเท่ากัน"
    blended = blend_price(bundle, listing, clustering, buildings)
    # Building's own listings, shrunk toward the unseen-building blend (src/features/build.py).
    area = float(listing["area_sqm"].iloc[0])
    known_price = bundle["known_building"].predict(listing, np.array([np.log(blended["blend"] / area)]))[0]
    project_price = float(known_price) if np.isfinite(known_price) else None
    # A building with other listings in the data is priced from them (most accurate on known
    # buildings); any other building is an unseen building, where the blend is most accurate.
    if project_price is not None:
        main_model, price = "project_level", project_price
    else:
        main_model, price = "blend_unseen", blended["blend"]
    has_coordinates = bool(listing[["latitude", "longitude"]].notna().all(axis=1).iloc[0])
    return {
        "price": price,
        "main_model": main_model,
        "blend": blended,
        "project_price": project_price,
        "has_coordinates": has_coordinates,
        "knn": float(np.exp(log_knn)),
        "knn_setting": f"{knn_model['metric']}, {vote}, k={knn_model['k']}",
        "tier": int(clustering.tier_of(listing["zone_no"]).iloc[0]),
        "neighbours": similar_rooms(bundle, knn_model, z, int(listing["zone_no"].iloc[0])),
        "location_from_zone": bool(listing["location_from_zone"].iloc[0]),
        "building": listing.iloc[0],
        "reliability": reliability_of(bundle, main_model, price, int(listing["zone_no"].iloc[0])),
    }


def main_model_label(result: dict[str, Any], bundle: dict[str, Any]) -> tuple[str, str]:
    if result["main_model"] == "project_level":
        known = bundle["known_building"]
        count = int(known.building_stats(result["building"].to_frame().T, leave_one_out=False)["n"].iloc[0])
        return (
            "ราคาห้องอื่นในโครงการเดียวกัน + Bayesian shrinkage",
            f"ตึกนี้มีประกาศห้องอื่นในข้อมูล {count} ห้อง ระบบใช้ราคาต่อตร.ม. ของห้องเหล่านั้นเป็นหลัก "
            "(ตึกที่มีประกาศน้อยจะถูกดึงเข้าหาราคาประเมินจากย่านมากขึ้น) แล้วปรับตามชั้นและขนาดห้อง",
        )
    blend = bundle["blend"]
    if result.get("location_from_zone"):
        return (
            f"เฉลี่ย Linear Regression + {blend['partner']} (ข้อมูลตึก + ราคาย่าน)",
            "ไม่ระบุโครงการ ระบบใช้ตำแหน่งกลางของโซนที่เลือกแทนตำแหน่งตึก และใช้ปีที่สร้างเสร็จกับระยะถึงรถไฟฟ้าที่กรอก "
            "ถ้าเลือกโครงการ ราคาจะแม่นขึ้น",
        )
    if blend.get("mode") == "unseen_ensemble":
        w_pct = int(round(100 * result["blend"].get("lr_weight", blend["w_lr"])))
        basis = (
            f"ตึกนี้ไม่มีประกาศห้องอื่นในข้อมูล ระบบใช้ข้อมูลตึกแบบละเอียดและราคาย่านแบบถ่วงน้ำหนักระยะ "
            f"(5 และ 20 ตึกใกล้สุด ปรับตามอายุ/ความสูงตึก และนับตึกฝั่งตรงข้ามแม่น้ำเจ้าพระยาเป็นระยะไกลขึ้น) "
            f"ถ่วงน้ำหนัก LR {w_pct}% + {blend['partner']} {100 - w_pct}% (ห้องยิ่งใหญ่ ยิ่งให้น้ำหนัก LR มาก)"
            if result["has_coordinates"]
            else "ไม่ระบุโครงการ และไม่ทราบตำแหน่งตึก ระบบจึงใช้ราคากลางของโซนแทนราคาตึกใกล้เคียง"
        )
        return f"เฉลี่ย Linear Regression + {blend['partner']} (ข้อมูลตึกละเอียด + ราคาย่าน IDW)", basis
    if blend.get("mode") == "extended_multiscale":
        w_pct = int(round(100 * blend["w_lr"]))
        basis = (
            f"ตึกนี้ไม่มีประกาศห้องอื่นในข้อมูล ระบบใช้ข้อมูลตึกแบบละเอียดและราคาย่านแบบถ่วงน้ำหนักระยะ "
            f"(5 และ 20 ตึกใกล้สุด) ถ่วงน้ำหนัก LR {w_pct}% + {blend['partner']} {100 - w_pct}%"
            if result["has_coordinates"]
            else "ไม่ระบุโครงการ และไม่ทราบตำแหน่งตึก ระบบจึงใช้ราคากลางของโซนแทนราคาตึกใกล้เคียง"
        )
        return f"เฉลี่ย Linear Regression + {blend['partner']} (ข้อมูลตึกละเอียด + ราคาย่าน IDW)", basis
    k = blend["neighbourhood"].k
    basis = (
        f"ตึกนี้ไม่มีประกาศห้องอื่นในข้อมูล ระบบใช้ราคาต่อตร.ม. ของ {k} ตึกที่อยู่ใกล้ที่สุดแทน"
        if result["has_coordinates"]
        else "ไม่ระบุโครงการ และไม่ทราบตำแหน่งตึก ระบบจึงใช้ราคากลางของโซนแทนราคาตึกใกล้เคียง"
    )
    return f"เฉลี่ย Linear Regression + {blend['partner']} (ข้อมูลตึก + ราคาย่าน)", basis


def reliability_details(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return plain-language test-set reliability lines and any reliability warnings."""
    reliability = result["reliability"]
    band = reliability["band"]
    case = "ห้องในตึกที่มีประกาศอื่นในข้อมูล" if result["main_model"] == "project_level" else "ห้องในตึกที่ระบบไม่เคยเห็น"
    lines = [
        f"ทดสอบกับ{case} {int(reliability['test_units']):,} ห้องที่ซ่อนไว้ตอนฝึก: ทายห่างจากราคาจริงไม่เกิน 20% "
        f"ได้ {float(reliability['within_20pct']) / 10:.1f} ใน 10 ห้อง",
        f"เฉพาะห้องที่ทายได้ราคาช่วง {band['price_band']} ({int(band['test_units'])} ห้อง): "
        f"ไม่เกิน 20% ได้ {float(band['within_20pct']) / 10:.1f} ใน 10 ห้อง",
    ]
    warnings: list[str] = []
    if float(band["within_20pct"]) < WEAK_BAND_WITHIN_20:
        warnings.append(f"ห้องราคาช่วง {band['price_band']} ระบบทายพลาดบ่อยกว่าช่วงอื่น ใช้ราคานี้เป็นแนวทางเท่านั้น")
    zone = reliability["zone"]
    if zone is None:
        lines.append("โซนนี้มีห้องทดสอบน้อย บอกความแม่นยำรายโซนไม่ได้")
    else:
        lines.append(
            f"ในโซนนี้ ({int(zone['test_units'])} ห้องทดสอบ): ไม่เกิน 20% ได้ {float(zone['within_20pct']) / 10:.1f} ใน 10 ห้อง"
        )
        bias = float(zone["median_signed_error_pct"])
        if abs(bias) >= ZONE_BIAS_PCT:
            direction = "ต่ำกว่า" if bias < 0 else "สูงกว่า"
            warnings.append(f"ในโซนนี้ระบบมักทาย{direction}ราคาประกาศจริงประมาณ {abs(bias):.0f}%")
    return lines, warnings


def model_quality_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    unseen = bundle["test_results_project_split"]
    return [
        {
            "model": name,
            "mape_random_pct": round(float(values["mape_pct"]), 1),
            "mape_unseen_pct": round(float(unseen[name]["mape_pct"]), 1),
            "mae_million_thb": round(float(values["mae_thb"]) / 1e6, 2),
            "r2": round(float(values["r2"]), 3),
            "within_10pct": round(float(values["within_10pct"]), 1),
        }
        for name, values in bundle["test_results"].items()
    ]


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        num = float(value)
        return None if (math.isnan(num) or math.isinf(num)) else num
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def serialize_metadata(bundle: dict[str, Any], forecast: dict[str, Any] | None) -> dict[str, Any]:
    zones_df = bundle["zone_table"].sort_values("zone_no")
    zones = [
        {
            "zone_no": int(row["zone_no"]),
            "label": zone_label(row),
            "zone_name_th": str(row["zone_name_th"]),
            "district_th": str(row["district_th"]),
        }
        for _, row in zones_df.iterrows()
    ]
    forecast_meta = None
    if forecast:
        years = [int(item["year"]) for item in forecast["forecast"]]
        forecast_meta = {
            "available": True,
            "base_year": int(forecast["base_year"]),
            "min_year": min(years),
            "max_year": max(years),
            "annual_growth_pct": float(forecast["annual_growth_pct"]),
        }
    return _json_safe({
        "zones": zones,
        "projects": list(bundle["project_names"]),
        "bedroom_options": [{"label": label, "value": val} for label, val in BEDROOM_OPTIONS.items()],
        "no_project_label": NO_PROJECT,
        "n_clusters": int(bundle["n_clusters"]),
        "defaults": {
            "zone_no": zones[0]["zone_no"] if zones else 1,
            "area_sqm": 35.0,
            "bedrooms": 1,
            "bathrooms": 1,
            "floor": 10,
            "complete_year": 2016,
            "rail_km": 0.5,
            "is_new": False,
            "year": forecast_meta["base_year"] if forecast_meta else 2026,
        },
        "forecast": forecast_meta,
        "model_quality": model_quality_rows(bundle),
    })


def serialize_prediction(
    bundle: dict[str, Any],
    forecast: dict[str, Any] | None,
    *,
    zone_no: int,
    area_sqm: float,
    bedrooms: int,
    bathrooms: int,
    floor: int,
    is_new: bool,
    project: str | None,
    complete_year: int,
    rail_km: float,
    year: int = 2026,
) -> dict[str, Any]:
    labels = zone_labels_map(bundle)
    clean_project = project.strip() if project else None
    if not clean_project or clean_project == NO_PROJECT:
        clean_project = None
    listing = build_input(
        bundle, zone_no, area_sqm, bedrooms, bathrooms, floor, is_new, clean_project, complete_year, rail_km
    )
    warnings = input_warnings(listing, zone_labels=labels)
    result = predict(bundle, listing)
    main_title, main_basis = main_model_label(result, bundle)
    rel_lines, rel_warnings = reliability_details(result)

    building = result["building"]
    rail_val = _json_safe(building.get("nearest_rail_km"))
    complete_val = _json_safe(building.get("complete_year"))
    total_unit_val = _json_safe(building.get("total_unit"))
    total_floor_val = _json_safe(building.get("total_floor"))

    blended = result["blend"]
    partner = bundle["blend"]["partner"]
    model_rows: list[dict[str, Any]] = [
        {"model": "Linear Regression + ข้อมูลตึก + ราคาย่าน", "price": float(blended["linear"])},
        {"model": f"{partner} + ข้อมูลตึก + ราคาย่าน", "price": float(blended["partner"])},
        {"model": "เฉลี่ย 2 ตัวบน", "price": float(blended["blend"])},
    ]
    if result["project_price"] is not None:
        model_rows.append({"model": "ราคาห้องอื่นในโครงการ + shrinkage", "price": float(result["project_price"])})
    model_rows.append({
        "model": "KNN Regression + ข้อมูลตึก + ราคาย่าน (ราคาต่อ ตร.ม.)",
        "price": float(result["knn"]),
    })

    neighbours_rows = [
        {
            "project_name": _json_safe(row.get("project_name")),
            "zone_name_th": _json_safe(row.get("zone_name_th")),
            "district_th": _json_safe(row.get("district_th")),
            "room_type": _json_safe(row.get("room_type")),
            "bedrooms": _json_safe(row.get("bedrooms")),
            "bathrooms": _json_safe(row.get("bathrooms")),
            "floor": _json_safe(row.get("floor")),
            "area_sqm": _json_safe(row.get("area_sqm")),
            "price_thb": _json_safe(row.get("price_thb")),
        }
        for _, row in result["neighbours"].iterrows()
    ]

    future = None
    if forecast and year > int(forecast["base_year"]):
        matching = [item for item in forecast["forecast"] if int(item["year"]) == int(year)]
        if matching:
            target_row = matching[0]
            horizon = int(year) - int(forecast["base_year"])
            horizon_error = forecast["backtest_mape_pct_by_horizon"].get(str(horizon))
            price_today = float(result["price"])
            future = {
                "year": int(year),
                "base_year": int(forecast["base_year"]),
                "horizon_years": horizon,
                "price": price_today * float(target_row["factor"]),
                "low": price_today * float(target_row["low"]),
                "high": price_today * float(target_row["high"]),
                "annual_growth_pct": float(forecast["annual_growth_pct"]),
                "backtest_mape_pct": float(horizon_error) if horizon_error is not None else None,
                "series": [
                    {
                        "year": int(item["year"]),
                        "price": price_today * float(item["factor"]),
                        "low": price_today * float(item["low"]),
                        "high": price_today * float(item["high"]),
                    }
                    for item in forecast["forecast"]
                    if int(item["year"]) <= int(year)
                ],
            }

    price = float(result["price"])
    # every forecast year, so the page can follow the year slider without asking the server again
    forecast_series = None
    if forecast:
        forecast_series = {
            "base_year": int(forecast["base_year"]),
            "annual_growth_pct": float(forecast["annual_growth_pct"]),
            "years": [
                {
                    "year": int(item["year"]),
                    "price": price * float(item["factor"]),
                    "low": price * float(item["low"]),
                    "high": price * float(item["high"]),
                    "backtest_mape_pct": forecast["backtest_mape_pct_by_horizon"].get(
                        str(int(item["year"]) - int(forecast["base_year"]))),
                }
                for item in forecast["forecast"]
            ],
        }
    return _json_safe({
        "input_warnings": warnings,
        "main_model": result["main_model"],
        "main_title": main_title,
        "main_basis": main_basis,
        "price": price,
        "price_per_sqm": price / float(area_sqm),
        "range_low": price / 1.2,
        "range_high": price / 0.8,
        "knn": float(result["knn"]),
        "knn_setting": result["knn_setting"],
        "tier": int(result["tier"]),
        "n_clusters": int(bundle["n_clusters"]),
        "effective_zone_no": int(building["zone_no"]),
        "reliability_lines": rel_lines,
        "reliability_warnings": rel_warnings,
        "building": {
            "complete_year": int(complete_val) if complete_val is not None else None,
            "nearest_rail_km": float(rail_val) if rail_val is not None else None,
            "total_unit": int(total_unit_val) if total_unit_val is not None and total_unit_val > 0 else None,
            "total_floor": int(total_floor_val) if total_floor_val is not None and total_floor_val > 0 else None,
            "has_coordinates": bool(result["has_coordinates"]),
            "location_from_zone": bool(result["location_from_zone"]),
        },
        "model_breakdown": model_rows,
        "neighbours": neighbours_rows,
        "future": future,
        "forecast_series": forecast_series,
    })
