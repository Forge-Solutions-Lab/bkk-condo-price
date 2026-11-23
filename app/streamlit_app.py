"""Web demo: enter a condo's details and get a price estimate from the trained models."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# the pickled models are instances of classes in src/, so the project root must be importable
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build import BASE_YEAR, ZoneClustering, attach_buildings, feature_frame, project_key  # noqa: E402
from src.models import ffbp  # noqa: E402
from src.models import linear_regression as lr  # noqa: E402
from src.models.knn_regression import knn, neighbors  # noqa: E402

BUNDLE_PATH = PROJECT_ROOT / "models" / "model_bundle.pkl"
FORECAST_PATH = PROJECT_ROOT / "reports" / "forecast.json"
THEME_PATH = Path(__file__).with_name("theme.css")
BEDROOM_OPTIONS = {"สตูดิโอ": 0, "1 ห้องนอน": 1, "2 ห้องนอน": 2, "3 ห้องนอน": 3, "4 ห้องนอน": 4, "5 ห้องนอน": 5}
SIMILAR_COUNT = 5
NO_PROJECT = "ไม่ระบุโครงการ"
# a zone with fewer test units than this has no reliable zone-level accuracy figure
MIN_ZONE_TEST_UNITS = 10
# a predicted-price band whose test accuracy is below this gets a warning
WEAK_BAND_WITHIN_20 = 60.0
# a zone whose median error is beyond this is priced too low or too high on average
ZONE_BIAS_PCT = 10.0
# "new unit" in a building finished this many years ago or more is unusual
NEW_UNIT_MAX_BUILDING_AGE = 3
# zones with prominent 30-year leasehold luxury condos (priced well below neighbouring freehold)
LEASEHOLD_ADVISORY_ZONES = {5}


@st.cache_resource
def load_bundle() -> dict:
    with BUNDLE_PATH.open("rb") as handle:
        return pickle.load(handle)


def format_baht(value: float) -> str:
    return f"{value:,.0f} บาท"


def zone_label(row: pd.Series) -> str:
    return f"{row['zone_name_th']} (เขต{row['district_th']})"


def build_input(
    bundle: dict, zone_no: int, area: float, bedrooms: int, bathrooms: int, floor: int, is_new: bool,
    project: str | None, complete_year: int, rail_km: float,
) -> pd.DataFrame:
    effective_zone = int(zone_no)
    if project:
        expected_zone = bundle.get("project_zones", {}).get(project_key(pd.Series([project])).iloc[0])
        if expected_zone is not None and int(expected_zone) != effective_zone:
            effective_zone = int(expected_zone)
    listing = pd.DataFrame([{
        "zone_no": effective_zone, "requested_zone_no": int(zone_no),
        "area_sqm": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "floor": floor, "is_new_unit": is_new, "project_name": project or "",
    }])
    listing = attach_buildings(listing, bundle["projects"])
    if not listing["has_building_info"].iloc[0]:
        # no saved project page: use what the user typed; other building fields (including mall_km)
        # fall back to the training medians via manual_building_input=True instead of 3 km cap
        listing[["complete_year", "nearest_rail_km"]] = [complete_year, rail_km]
        listing["has_building_info"] = True
        listing["manual_building_input"] = True
    else:
        listing["manual_building_input"] = False
    return listing


def input_warnings(
    listing: pd.DataFrame,
    zone_labels: dict[int, str] | None = None,
) -> list[str]:
    row = listing.iloc[0]
    warnings = []
    if "requested_zone_no" in listing.columns and int(row["requested_zone_no"]) != int(row["zone_no"]):
        chosen, actual = int(row["requested_zone_no"]), int(row["zone_no"])
        chosen_name = zone_labels.get(chosen, f"โซน {chosen}") if zone_labels else f"โซน {chosen}"
        actual_name = zone_labels.get(actual, f"โซน {actual}") if zone_labels else f"โซน {actual}"
        warnings.append(f"โครงการที่เลือกอยู่ใน '{actual_name}' (ไม่ใช่ '{chosen_name}') ระบบจึงปรับใช้โซนจริงของโครงการให้อัตโนมัติ")
    if row["bedrooms"] >= 2 and row["area_sqm"] < 40:
        warnings.append("พื้นที่น้อยมากสำหรับจำนวนห้องนอนนี้ ผลอาจคลาดเคลื่อนสูง")
    if row["bathrooms"] > row["bedrooms"] + 2:
        warnings.append("จำนวนห้องน้ำมากกว่าห้องนอนผิดปกติ ผลอาจคลาดเคลื่อนสูง")
    if row["is_new_unit"] and pd.notna(row["complete_year"]) and BASE_YEAR - row["complete_year"] >= NEW_UNIT_MAX_BUILDING_AGE:
        warnings.append(f"เลือกห้องมือ 1 แต่ตึกสร้างเสร็จปี {int(row['complete_year'])} ซึ่งไม่ค่อยพบ "
                        "ตรวจข้อมูลอีกครั้ง ถ้าเป็นห้องมือ 2 ให้เอาเครื่องหมายออก")
    if int(row["zone_no"]) in LEASEHOLD_ADVISORY_ZONES:
        warnings.append("ย่านวิทยุ-เพลินจิต-หลังสวน มีหลายโครงการเป็นสิทธิการเช่า (Leasehold 30 ปี) ซึ่งราคาต่ำกว่าแบบขายขาด "
                        "(Freehold) ราว 30-50% โมเดลยังไม่ได้แยกประเภทกรรมสิทธิ์ หากเป็นห้อง Leasehold ควรเผื่อส่วนลดเพิ่ม")
    return warnings


def blend_price(bundle: dict, listing: pd.DataFrame, clustering: ZoneClustering, buildings: pd.DataFrame) -> dict[str, float]:
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


def reliability_of(bundle: dict, model: str, price: float, zone_no: int) -> dict[str, object]:
    """Test-set accuracy for this kind of estimate: overall, for its predicted-price band and for its zone."""
    table = bundle["reliability"][model]
    bands = table["bands"]
    band_index = int(np.searchsorted([3e6, 5e6, 8e6, 15e6], price, side="right"))
    zone = table["zones"][table["zones"]["zone_no"] == zone_no]
    return {
        "test_units": table["test_units"], "within_20pct": table["within_20pct"],
        "band": bands.iloc[band_index],
        "zone": zone.iloc[0] if len(zone) and zone.iloc[0]["test_units"] >= MIN_ZONE_TEST_UNITS else None,
    }


def predict(bundle: dict, listing: pd.DataFrame) -> dict[str, object]:
    clustering = ZoneClustering(zone_table=bundle["zone_table"], n_clusters=bundle["n_clusters"])
    buildings = bundle["building_features"].transform(listing)
    building_features = feature_frame(listing, clustering, buildings=buildings)[bundle["building_feature_names"]].to_numpy()
    knn_model = bundle["knn_building"]
    z = knn_model["scaler"].transform(building_features)[0]
    neighbour_index, _ = neighbors(knn_model["X"], z, SIMILAR_COUNT, knn_model["metric"])
    log_knn = knn(knn_model["X"], knn_model["y"], z, knn_model["k"], knn_model["metric"], knn_model["weight"])
    vote = "ถ่วงน้ำหนักตามระยะ" if knn_model["weight"] == "distance" else "โหวตเท่ากัน"
    blended = blend_price(bundle, listing, clustering, buildings)
    # building's own listings, shrunk toward the unseen-building blend (src/features/build.py)
    area = float(listing["area_sqm"].iloc[0])
    known_price = bundle["known_building"].predict(listing, np.array([np.log(blended["blend"] / area)]))[0]
    project_price = float(known_price) if np.isfinite(known_price) else None
    # a building with other listings in the data is priced from them (most accurate on known
    # buildings); any other building is an unseen building, where the blend is most accurate
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
        "neighbours": bundle["reference_rows"].iloc[neighbour_index],
        "building": listing.iloc[0],
        "reliability": reliability_of(bundle, main_model, price, int(listing["zone_no"].iloc[0])),
    }


def render_header() -> None:
    st.markdown('<div class="micro-label">Bangkok condo price estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">BKK Condo Price</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">ประเมินราคาคอนโดในกรุงเทพฯ จากประกาศขายที่เก็บ ทำ label และทำความสะอาดเอง '
        'ด้วย K-Means + Linear Regression (พร้อมข้อมูลตึกและราคาย่าน) เทียบกับ KNN Regression</div>',
        unsafe_allow_html=True,
    )


def main_model_label(result: dict[str, object], bundle: dict) -> tuple[str, str]:
    if result["main_model"] == "project_level":
        known = bundle["known_building"]
        count = int(known.building_stats(result["building"].to_frame().T, leave_one_out=False)["n"].iloc[0])
        return ("ราคาห้องอื่นในโครงการเดียวกัน + Bayesian shrinkage",
                f"ตึกนี้มีประกาศห้องอื่นในข้อมูล {count} ห้อง ระบบใช้ราคาต่อตร.ม. ของห้องเหล่านั้นเป็นหลัก "
                "(ตึกที่มีประกาศน้อยจะถูกดึงเข้าหาราคาประเมินจากย่านมากขึ้น) แล้วปรับตามชั้นและขนาดห้อง")
    blend = bundle["blend"]
    if blend.get("mode") == "unseen_ensemble":
        w_pct = int(round(100 * result["blend"].get("lr_weight", blend["w_lr"])))
        basis = (f"ตึกนี้ไม่มีประกาศห้องอื่นในข้อมูล ระบบใช้ข้อมูลตึกแบบละเอียดและราคาย่านแบบถ่วงน้ำหนักระยะ "
                 f"(5 และ 20 ตึกใกล้สุด ปรับตามอายุ/ความสูงตึก และนับตึกฝั่งตรงข้ามแม่น้ำเจ้าพระยาเป็นระยะไกลขึ้น) "
                 f"ถ่วงน้ำหนัก LR {w_pct}% + {blend['partner']} {100 - w_pct}% (ห้องยิ่งใหญ่ ยิ่งให้น้ำหนัก LR มาก)"
                 if result["has_coordinates"] else
                 "ไม่ระบุโครงการ ระบบไม่รู้ตำแหน่งตึก จึงใช้ราคากลางของโซนแทนราคาตึกใกล้เคียง")
        return f"เฉลี่ย Linear Regression + {blend['partner']} (ข้อมูลตึกละเอียด + ราคาย่าน IDW)", basis
    if blend.get("mode") == "extended_multiscale":
        w_pct = int(round(100 * blend["w_lr"]))
        basis = (f"ตึกนี้ไม่มีประกาศห้องอื่นในข้อมูล ระบบใช้ข้อมูลตึกแบบละเอียดและราคาย่านแบบถ่วงน้ำหนักระยะ "
                 f"(5 และ 20 ตึกใกล้สุด) ถ่วงน้ำหนัก LR {w_pct}% + {blend['partner']} {100 - w_pct}%"
                 if result["has_coordinates"] else
                 "ไม่ระบุโครงการ ระบบไม่รู้ตำแหน่งตึก จึงใช้ราคากลางของโซนแทนราคาตึกใกล้เคียง")
        return f"เฉลี่ย Linear Regression + {blend['partner']} (ข้อมูลตึกละเอียด + ราคาย่าน IDW)", basis
    k = blend["neighbourhood"].k
    basis = (f"ตึกนี้ไม่มีประกาศห้องอื่นในข้อมูล ระบบใช้ราคาต่อตร.ม. ของ {k} ตึกที่อยู่ใกล้ที่สุดแทน"
             if result["has_coordinates"] else
             "ไม่ระบุโครงการ ระบบไม่รู้ตำแหน่งตึก จึงใช้ราคากลางของโซนแทนราคาตึกใกล้เคียง")
    return f"เฉลี่ย Linear Regression + {blend['partner']} (ข้อมูลตึก + ราคาย่าน)", basis


def render_reliability(result: dict[str, object]) -> list[str]:
    """Plain-language accuracy of this estimate from the test set, plus warnings where it is weak."""
    reliability = result["reliability"]
    band = reliability["band"]
    case = "ห้องในตึกที่มีประกาศอื่นในข้อมูล" if result["main_model"] == "project_level" else "ห้องในตึกที่ระบบไม่เคยเห็น"
    lines = [
        f"ทดสอบกับ{case} {reliability['test_units']:,} ห้องที่ซ่อนไว้ตอนฝึก: ทายห่างจากราคาจริงไม่เกิน 20% "
        f"ได้ {reliability['within_20pct'] / 10:.1f} ใน 10 ห้อง",
        f"เฉพาะห้องที่ทายได้ราคาช่วง {band['price_band']} ({int(band['test_units'])} ห้อง): "
        f"ไม่เกิน 20% ได้ {band['within_20pct'] / 10:.1f} ใน 10 ห้อง",
    ]
    warnings = []
    if band["within_20pct"] < WEAK_BAND_WITHIN_20:
        warnings.append(f"ห้องราคาช่วง {band['price_band']} ระบบทายพลาดบ่อยกว่าช่วงอื่น ใช้ราคานี้เป็นแนวทางเท่านั้น")
    zone = reliability["zone"]
    if zone is None:
        lines.append("โซนนี้มีห้องทดสอบน้อย บอกความแม่นยำรายโซนไม่ได้")
    else:
        lines.append(f"ในโซนนี้ ({int(zone['test_units'])} ห้องทดสอบ): ไม่เกิน 20% ได้ {zone['within_20pct'] / 10:.1f} ใน 10 ห้อง")
        bias = zone["median_signed_error_pct"]
        if abs(bias) >= ZONE_BIAS_PCT:
            direction = "ต่ำกว่า" if bias < 0 else "สูงกว่า"
            warnings.append(f"ในโซนนี้ระบบมักทาย{direction}ราคาประกาศจริงประมาณ {abs(bias):.0f}%")
    st.markdown(
        '<div class="panel-deep"><div class="micro-label">ราคานี้เชื่อได้แค่ไหน</div>'
        + "".join(f'<div class="value-note">{line}</div>' for line in lines)
        + '</div>',
        unsafe_allow_html=True,
    )
    return warnings


def render_result(result: dict[str, object], area: float, bundle: dict) -> None:
    title, basis = main_model_label(result, bundle)
    price = result["price"]
    left, right = st.columns([3, 2])
    with left:
        # within 20% of the real price means the real price is between price / 1.2 and price / 0.8
        st.markdown(
            f'<div class="panel"><div class="micro-label">{title} (โมเดลหลัก)</div>'
            f'<div class="price-main">{format_baht(price)}</div>'
            f'<div class="value-note">ถ้าทายพลาดไม่เกิน 20% ราคาจริงอยู่ระหว่าง {format_baht(price / 1.2)} - '
            f'{format_baht(price / 0.8)} | {price / area:,.0f} บาท/ตร.ม.</div>'
            f'<div class="value-note">{basis}</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="panel"><div class="micro-label">KNN Regression (เปรียบเทียบ)</div>'
            f'<div class="price-second">{format_baht(result["knn"])}</div>'
            f'<div class="value-note">ระยะ {result["knn_setting"]}</div>'
            f'<div class="value-note"><span class="tier-badge">ระดับทำเล {result["tier"]} / {bundle["n_clusters"]}</span>'
            ' จาก K-Means (1 = ราคาต่อตร.ม. ต่ำสุด)</div></div>',
            unsafe_allow_html=True,
        )
    for warning in render_reliability(result):
        st.warning(warning)
    building = result["building"]
    rail = building["nearest_rail_km"]
    rail_text = f"{rail:.1f} กม." if pd.notna(rail) else "เกิน 3 กม."
    year_text = f"{int(building['complete_year'])}" if pd.notna(building["complete_year"]) else "ไม่ทราบ"
    st.markdown(
        f'<div class="value-note">ข้อมูลตึกที่ใช้: สร้างเสร็จปี {year_text} | ระยะถึงรถไฟฟ้า {rail_text}'
        + (f' | {int(building["total_unit"])} ยูนิต {int(building["total_floor"])} ชั้น'
           if pd.notna(building.get("total_unit")) and building.get("total_unit", 0) > 0 else '')
        + '</div>',
        unsafe_allow_html=True,
    )
    blended = result["blend"]
    partner = bundle["blend"]["partner"]
    rows = [
        {"โมเดล": "Linear Regression + ข้อมูลตึก + ราคาย่าน", "ราคา (บาท)": blended["linear"]},
        {"โมเดล": f"{partner} + ข้อมูลตึก + ราคาย่าน", "ราคา (บาท)": blended["partner"]},
        {"โมเดล": "เฉลี่ย 2 ตัวบน", "ราคา (บาท)": blended["blend"]},
    ]
    if result["project_price"] is not None:
        rows.append({"โมเดล": "ราคาห้องอื่นในโครงการ + shrinkage", "ราคา (บาท)": result["project_price"]})
    rows.append({"โมเดล": "KNN Regression + ข้อมูลตึก", "ราคา (บาท)": result["knn"]})
    st.markdown("### ราคาจากแต่ละโมเดล")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"ราคา (บาท)": st.column_config.NumberColumn(format="%,.0f")})
    st.markdown("### ห้องที่คล้ายกันที่สุดในข้อมูล")
    neighbours = result["neighbours"].rename(columns={
        "project_name": "โครงการ", "zone_name_th": "โซน", "district_th": "เขต", "room_type": "ประเภท",
        "bedrooms": "ห้องนอน", "bathrooms": "ห้องน้ำ", "floor": "ชั้น", "area_sqm": "ตร.ม.", "price_thb": "ราคา (บาท)",
    })
    st.dataframe(neighbours, hide_index=True, use_container_width=True,
                 column_config={"ราคา (บาท)": st.column_config.NumberColumn(format="%,.0f")})


@st.cache_resource
def load_forecast() -> dict | None:
    return json.loads(FORECAST_PATH.read_text(encoding="utf-8")) if FORECAST_PATH.exists() else None


def render_future_price(price_today: float, year: int, forecast: dict) -> None:
    row = next(item for item in forecast["forecast"] if item["year"] == year)
    error = forecast["backtest_mape_pct_by_horizon"].get(str(year - forecast["base_year"]))
    st.markdown(
        f'<div class="panel-deep"><div class="micro-label">คาดการณ์ราคาปี {year} (แนวโน้มตลาด)</div>'
        f'<div class="price-second">{format_baht(price_today * row["factor"])}</div>'
        f'<div class="value-note">ช่วงที่น่าจะเป็น {format_baht(price_today * row["low"])} - '
        f'{format_baht(price_today * row["high"])} | แนวโน้ม {forecast["annual_growth_pct"]:.1f}% ต่อปี '
        f'จากดัชนีราคาที่อยู่อาศัยกรุงเทพฯ 1991-2026 | ทดสอบย้อนหลังทายล่วงหน้า {year - forecast["base_year"]} ปี '
        f'คลาดเคลื่อนเฉลี่ย {error:.1f}%</div></div>',
        unsafe_allow_html=True,
    )
    table = pd.DataFrame([
        {"ปี": str(item["year"]), "ราคาคาดการณ์": price_today * item["factor"],
         "ต่ำ": price_today * item["low"], "สูง": price_today * item["high"]}
        for item in forecast["forecast"] if item["year"] <= year
    ]).set_index("ปี")
    st.line_chart(table, color=["#FFC000", "#7D7D7D", "#FFFFFF"])


def render_model_quality(bundle: dict) -> None:
    st.markdown("### ความแม่นยำบนชุดทดสอบ")
    unseen = bundle["test_results_project_split"]
    rows = [
        {"โมเดล": name, "MAPE สุ่มแบ่ง (%)": round(values["mape_pct"], 1),
         "MAPE ตึกที่ไม่เคยเห็น (%)": round(unseen[name]["mape_pct"], 1),
         "MAE (ล้านบาท)": round(values["mae_thb"] / 1e6, 2), "R²": round(values["r2"], 3),
         "คลาดเคลื่อนไม่เกิน 10% (%)": round(values["within_10pct"], 1)}
        for name, values in bundle["test_results"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="BKK Condo Price", layout="wide")
    st.markdown(f"<style>{THEME_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    render_header()
    if not BUNDLE_PATH.exists():
        st.error("ยังไม่มีโมเดล ให้รัน python -m src.evaluate.run ก่อน")
        return
    bundle = load_bundle()
    zones = bundle["zone_table"].sort_values("zone_no")
    labels = {int(row["zone_no"]): zone_label(row) for _, row in zones.iterrows()}
    with st.form("estimate"):
        zone_col, area_col = st.columns(2)
        zone_no = zone_col.selectbox("โซน / เขต", list(labels), format_func=labels.get)
        area = area_col.number_input("พื้นที่ (ตร.ม.)", min_value=15.0, max_value=500.0, value=35.0, step=1.0)
        bed_col, bath_col, floor_col = st.columns(3)
        bedrooms = BEDROOM_OPTIONS[bed_col.selectbox("ห้องนอน", list(BEDROOM_OPTIONS), index=1)]
        bathrooms = bath_col.selectbox("ห้องน้ำ", [1, 2, 3, 4, 5], index=0)
        floor = floor_col.number_input("ชั้น", min_value=1, max_value=80, value=10, step=1)
        project_options = [NO_PROJECT] + bundle["project_names"]
        project_choice = st.selectbox("โครงการ (ไม่บังคับ พิมพ์ค้นหาได้ ถ้าเลือก ระบบใช้ข้อมูลตึกของโครงการให้)",
                                      project_options, index=0)
        year_col, rail_col = st.columns(2)
        complete_year = year_col.number_input("ปีที่ตึกสร้างเสร็จ (ใช้เมื่อไม่ระบุโครงการ)", min_value=1980,
                                              max_value=2030, value=2016, step=1)
        rail_km = rail_col.number_input("ระยะถึงสถานีรถไฟฟ้า กม. (ใช้เมื่อไม่ระบุโครงการ)", min_value=0.0,
                                        max_value=3.0, value=0.5, step=0.1)
        is_new = st.checkbox("ห้องมือ 1 (ยังไม่เคยมีผู้อยู่)")
        forecast = load_forecast()
        year = st.slider("ปีที่ต้องการดูราคา", 2026, 2036, 2026) if forecast else 2026
        submitted = st.form_submit_button("ประเมินราคา")
    if submitted:
        project = None if project_choice == NO_PROJECT else project_choice
        listing = build_input(bundle, zone_no, area, bedrooms, bathrooms, floor, is_new, project,
                              complete_year, rail_km)
        for warning in input_warnings(listing, zone_labels=labels):
            st.warning(warning)
        result = predict(bundle, listing)
        render_result(result, area, bundle)
        if year > 2026:
            render_future_price(result["price"], year, forecast)
    render_model_quality(bundle)
    st.markdown(
        '<div class="value-note">ข้อมูล: ประกาศขายคอนโดบน Livinginsider ที่บันทึกด้วยมือเมื่อ 23 พ.ย. 2569 '
        'ราคาเป็นราคาประกาศขาย ไม่ใช่ราคาซื้อขายจริง</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
