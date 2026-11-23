"""Unit tests for scripts/parse_projects.py using small synthetic HTML fixtures.

All fixtures are generated inline so tests never depend on data/raw/ or real saved pages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.parse_projects import parse_file


def make_flight_html(payloads: list[dict], source_url: str = "") -> str:
    """Wrap dicts into a minimal Next.js flight-data HTML document (compact JSON)."""
    header = f"<!-- saved from url=({len(source_url):04d}){source_url} -->\n" if source_url else ""
    scripts = "\n".join(
        f"<script>self.__next_f.push([1,{json.dumps(json.dumps(item, ensure_ascii=False, separators=(',', ':')), ensure_ascii=False)}])</script>"
        for item in payloads
    )
    return f"<!doctype html>\n{header}<html><body>{scripts}</body></html>\n"


def test_parse_file_extracts_project_name_year_and_rail_distance(tmp_path: Path) -> None:
    listing_chunk = {
        "location": {"latitude": 13.7367, "longitude": 100.5608},
        "identity": {
            "id": 9999,
            "name": "ตัวอย่าง คอนโด สุขุมวิท",
            "slug": "sample-condo-sukhumvit",
        },
        "overview": {
            "complete_year": 2021,
            "total_unit": 350,
            "total_floor": 28,
            "total_building": 1,
        },
    }
    project_chunk = {
        "location": {"address": "ซอยสุขุมวิท 24 แขวงคลองตัน เขตคลองเตย กรุงเทพมหานคร"},
        "amenities": {
            "facilities": {
                "items": [
                    {"name": "สระว่ายน้ำ"},
                    {"name": "ฟิตเนส"},
                    {"name": "ที่จอดรถ"},
                    {"name": "สวนหย่อม"},
                ]
            },
            "nearby": {
                "items": [
                    {"category": "transit", "distance": 1.1, "icon": "icon_tran_mrt.svg"},
                    {"category": "transit", "distance": 0.4, "icon": "icon_tran_bts.svg"},
                    {"category": "mall", "distance": 0.6, "icon": "icon_mall.svg"},
                    {"category": "hospital", "distance": 1.8, "icon": "icon_hospital.svg"},
                    {"category": "school", "distance": 0.9, "icon": "icon_school.svg"},
                ]
            },
        },
    }
    source_url = "https://www.livinginsider.com/detail/condo-for-sale-sample-condo-sukhumvit-1bedroom-100001"
    html_file = tmp_path / "sample_project.html"
    html_file.write_text(make_flight_html([listing_chunk, project_chunk], source_url=source_url), encoding="utf-8")

    result = parse_file(html_file)

    assert result is not None
    assert result["project_id"] == 9999
    assert result["project_name"] == "ตัวอย่าง คอนโด สุขุมวิท"
    assert result["project_slug"] == "sample-condo-sukhumvit"
    assert result["complete_year"] == 2021
    assert result["total_unit"] == 350
    assert result["total_floor"] == 28
    assert result["total_building"] == 1
    assert result["latitude"] == pytest.approx(13.7367)
    assert result["longitude"] == pytest.approx(100.5608)
    assert result["district_th"] == "คลองเตย"
    assert result["facility_count"] == 4
    assert result["has_pool"] == 1
    assert result["has_gym"] == 1
    assert result["has_parking"] == 1
    assert result["nearest_rail_km"] == pytest.approx(0.4)
    assert result["nearest_rail_line"] == "bts"
    assert result["nearest_mall_km"] == pytest.approx(0.6)
    assert result["nearest_hospital_km"] == pytest.approx(1.8)
    assert result["nearest_school_km"] == pytest.approx(0.9)
    assert result["source_url"] == source_url
    assert result["source_file"] == "sample_project.html"


def test_parse_file_handles_missing_and_null_transit_distances(tmp_path: Path) -> None:
    payload = {
        "identity": {"id": 8888, "name": "คอนโด ชานเมือง", "slug": "suburb-condo"},
        "overview": {"complete_year": 2019},
        "amenities": {
            "facilities": {"items": []},
            "nearby": {
                "items": [
                    {"category": "transit", "distance": None, "icon": "icon_tran_mrt_pink.svg"},
                    {"category": "transit", "distance": 0.75, "icon": "icon_tran_mrt_pink.svg"},
                ]
            },
        },
    }
    html_file = tmp_path / "null_transit.html"
    html_file.write_text(make_flight_html([payload]), encoding="utf-8")

    result = parse_file(html_file)

    assert result is not None
    assert result["project_name"] == "คอนโด ชานเมือง"
    assert result["complete_year"] == 2019
    assert result["nearest_rail_km"] == pytest.approx(0.75)
    assert result["nearest_rail_line"] == "mrt_pink"
    assert result["facility_count"] == 0
    assert result["has_pool"] == 0


def test_parse_file_without_transit_sets_none(tmp_path: Path) -> None:
    payload = {
        "identity": {"id": 7777, "name": "คอนโด ไกลรถไฟฟ้า", "slug": "far-condo"},
        "overview": {"complete_year": 2015},
        "amenities": {"facilities": {"items": []}, "nearby": {"items": []}},
    }
    html_file = tmp_path / "no_transit.html"
    html_file.write_text(make_flight_html([payload]), encoding="utf-8")

    result = parse_file(html_file)

    assert result is not None
    assert result["project_name"] == "คอนโด ไกลรถไฟฟ้า"
    assert result["complete_year"] == 2015
    assert result["nearest_rail_km"] is None
    assert result["nearest_rail_line"] == ""


@pytest.mark.parametrize(
    "html_content",
    [
        "",
        "<html><body><h1>404 Not Found</h1></body></html>",
        make_flight_html([{"page": "home", "banners": []}]),
    ],
)
def test_parse_file_returns_none_without_project_data(tmp_path: Path, html_content: str) -> None:
    html_file = tmp_path / "empty_or_other.html"
    html_file.write_text(html_content, encoding="utf-8")

    assert parse_file(html_file) is None
