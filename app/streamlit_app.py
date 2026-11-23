"""Web demo: enter a condo's details and get a price estimate from the trained models."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# the pickled models are instances of classes in src/, so the project root must be importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import (  # noqa: E402
    BEDROOM_OPTIONS,
    BUNDLE_PATH,
    FORECAST_PATH,
    LEASEHOLD_ADVISORY_ZONES,
    MIN_ZONE_TEST_UNITS,
    NEW_UNIT_MAX_BUILDING_AGE,
    NO_PROJECT,
    SIMILAR_COUNT,
    WEAK_BAND_WITHIN_20,
    ZONE_BIAS_PCT,
    blend_price,
    build_input,
    format_baht,
    input_warnings,
    load_bundle_cached,
    load_forecast_cached,
    main_model_label,
    model_quality_rows,
    predict,
    reliability_details,
    reliability_of,
    zone_label,
)

THEME_PATH = Path(__file__).with_name("theme.css")


@st.cache_resource
def load_bundle() -> dict:
    return load_bundle_cached()


def render_header() -> None:
    st.markdown('<div class="micro-label">Bangkok condo price estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">BKK Condo Price</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">ประเมินราคาคอนโดในกรุงเทพฯ จากประกาศขายที่เก็บ ทำ label และทำความสะอาดเอง '
        'ด้วย K-Means + Linear Regression (พร้อมข้อมูลตึกและราคาย่าน) เทียบกับ KNN Regression</div>',
        unsafe_allow_html=True,
    )


def render_reliability(result: dict[str, object]) -> list[str]:
    """Plain-language accuracy of this estimate from the test set, plus warnings where it is weak."""
    lines, warnings = reliability_details(result)
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
    rows.append({"โมเดล": "KNN Regression + ข้อมูลตึก + ราคาย่าน (ราคาต่อ ตร.ม.)", "ราคา (บาท)": result["knn"]})
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
    return load_forecast_cached()


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
    rows = [
        {
            "โมเดล": row["model"],
            "MAPE สุ่มแบ่ง (%)": row["mape_random_pct"],
            "MAPE ตึกที่ไม่เคยเห็น (%)": row["mape_unseen_pct"],
            "MAE (ล้านบาท)": row["mae_million_thb"],
            "R²": row["r2"],
            "คลาดเคลื่อนไม่เกิน 10% (%)": row["within_10pct"],
        }
        for row in model_quality_rows(bundle)
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
