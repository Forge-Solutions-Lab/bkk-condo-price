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

from src.features.build import ZoneClustering, attach_buildings, feature_frame  # noqa: E402
from src.models import linear_regression as lr  # noqa: E402
from src.models.knn_regression import knn, neighbors  # noqa: E402

BUNDLE_PATH = PROJECT_ROOT / "models" / "model_bundle.pkl"
FORECAST_PATH = PROJECT_ROOT / "reports" / "forecast.json"
THEME_PATH = Path(__file__).with_name("theme.css")
BEDROOM_OPTIONS = {"สตูดิโอ": 0, "1 ห้องนอน": 1, "2 ห้องนอน": 2, "3 ห้องนอน": 3, "4 ห้องนอน": 4, "5 ห้องนอน": 5}
SIMILAR_COUNT = 5
NO_PROJECT = "ไม่ระบุโครงการ"


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
    listing = pd.DataFrame([{
        "zone_no": zone_no, "area_sqm": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "floor": floor, "is_new_unit": is_new, "project_name": project or "",
    }])
    listing = attach_buildings(listing, bundle["projects"])
    if not listing["has_building_info"].iloc[0]:
        # no saved project page: use what the user typed; other building fields fall back
        # to the training medians inside BuildingFeatures
        listing[["complete_year", "nearest_rail_km"]] = [complete_year, rail_km]
        listing["has_building_info"] = True
    return listing


def input_warnings(area: float, bedrooms: int, bathrooms: int) -> list[str]:
    warnings = []
    if bedrooms >= 2 and area < 40:
        warnings.append("พื้นที่น้อยมากสำหรับจำนวนห้องนอนนี้ ผลอาจคลาดเคลื่อนสูง")
    if bathrooms > bedrooms + 2:
        warnings.append("จำนวนห้องน้ำมากกว่าห้องนอนผิดปกติ ผลอาจคลาดเคลื่อนสูง")
    return warnings


def predict(bundle: dict, listing: pd.DataFrame) -> dict[str, object]:
    clustering = ZoneClustering(zone_table=bundle["zone_table"], n_clusters=bundle["n_clusters"])
    buildings = bundle["building_features"].transform(listing)
    features = feature_frame(listing, clustering, buildings=buildings)[bundle["building_feature_names"]].to_numpy()
    log_linear = float(lr.predict(features, bundle["W_building"])[0])
    knn_model = bundle["knn_building"]
    z = knn_model["scaler"].transform(features)[0]
    neighbour_index, _ = neighbors(knn_model["X"], z, SIMILAR_COUNT, knn_model["metric"])
    log_knn = knn(knn_model["X"], knn_model["y"], z, knn_model["k"], knn_model["metric"], knn_model["weight"])
    vote = "ถ่วงน้ำหนักตามระยะ" if knn_model["weight"] == "distance" else "โหวตเท่ากัน"
    spread = bundle["residual_std_log_building"]
    result: dict[str, object] = {
        "linear": float(np.exp(log_linear)),
        "knn": float(np.exp(log_knn)),
        "knn_setting": f"{knn_model['metric']}, {vote}, k={knn_model['k']}",
        "low": float(np.exp(log_linear - spread)),
        "high": float(np.exp(log_linear + spread)),
        "tier": int(clustering.tier_of(listing["zone_no"]).iloc[0]),
        "neighbours": bundle["reference_rows"].iloc[neighbour_index],
        "project": None,
        "building": listing.iloc[0],
    }
    if listing["project_name"].iloc[0]:
        levels = bundle["project_levels"].transform(listing, leave_one_out=False)
        project_features = feature_frame(listing, clustering, project_levels=levels, buildings=buildings)
        project_features = project_features[bundle["building_project_feature_names"]]
        log_project = float(lr.predict(project_features.to_numpy(), bundle["W_building_project"])[0])
        project_spread = bundle["residual_std_log_building_project"]
        result["project"] = {
            "price": float(np.exp(log_project)),
            "low": float(np.exp(log_project - project_spread)),
            "high": float(np.exp(log_project + project_spread)),
        }
    return result


def render_header() -> None:
    st.markdown('<div class="micro-label">Bangkok condo price estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">BKK Condo Price</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">ประเมินราคาคอนโดในกรุงเทพฯ จากประกาศขายที่เก็บ ทำ label และทำความสะอาดเอง '
        'ด้วย K-Means + Linear Regression (พร้อมข้อมูลตึก) เทียบกับ KNN Regression</div>',
        unsafe_allow_html=True,
    )


def render_result(result: dict[str, object], area: float, n_clusters: int) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            '<div class="panel"><div class="micro-label">Linear Regression + ข้อมูลตึก (โมเดลหลัก)</div>'
            f'<div class="price-main">{format_baht(result["linear"])}</div>'
            f'<div class="value-note">ช่วงที่น่าจะเป็น {format_baht(result["low"])} - {format_baht(result["high"])}'
            f' | {result["linear"] / area:,.0f} บาท/ตร.ม.</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="panel"><div class="micro-label">KNN Regression (เปรียบเทียบ)</div>'
            f'<div class="price-second">{format_baht(result["knn"])}</div>'
            f'<div class="value-note">ระยะ {result["knn_setting"]}</div>'
            f'<div class="value-note"><span class="tier-badge">ระดับทำเล {result["tier"]} / {n_clusters}</span>'
            ' จาก K-Means (1 = ราคาต่อตร.ม. ต่ำสุด)</div></div>',
            unsafe_allow_html=True,
        )
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
    project = result["project"]
    if project:
        st.markdown(
            '<div class="panel-deep"><div class="micro-label">Linear Regression + ระดับราคาของโครงการ (ต่อยอด)</div>'
            f'<div class="price-second">{format_baht(project["price"])}</div>'
            f'<div class="value-note">ช่วงที่น่าจะเป็น {format_baht(project["low"])} - {format_baht(project["high"])}'
            ' | ใช้ราคาต่อตร.ม. ของห้องอื่นในโครงการเดียวกันจากข้อมูลที่เก็บมา</div></div>',
            unsafe_allow_html=True,
        )
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
        for warning in input_warnings(area, bedrooms, bathrooms):
            st.warning(warning)
        project = None if project_choice == NO_PROJECT else project_choice
        listing = build_input(bundle, zone_no, area, bedrooms, bathrooms, floor, is_new, project,
                              complete_year, rail_km)
        result = predict(bundle, listing)
        render_result(result, area, bundle["n_clusters"])
        if year > 2026:
            render_future_price(result["linear"], year, forecast)
    render_model_quality(bundle)
    st.markdown(
        '<div class="value-note">ข้อมูล: ประกาศขายคอนโดบน Livinginsider ที่บันทึกด้วยมือเมื่อ 23 พ.ย. 2569 '
        'ราคาเป็นราคาประกาศขาย ไม่ใช่ราคาซื้อขายจริง</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
