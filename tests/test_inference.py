"""Tests for shared pure-Python inference and JSON serialization used by the Vercel API."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from app.inference import (
    BUNDLE_PATH,
    _json_safe,
    filter_projects,
    format_baht,
    input_warnings,
    list_zones,
    load_bundle_cached,
    load_forecast_cached,
    serialize_metadata,
    serialize_prediction,
)


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="model_bundle.pkl not generated")
def test_serialize_metadata_strict_json() -> None:
    bundle = load_bundle_cached()
    forecast = load_forecast_cached()
    meta = serialize_metadata(bundle, forecast)
    encoded = json.dumps(meta, allow_nan=False)
    decoded = json.loads(encoded)
    assert len(decoded["zones"]) == len(bundle["zone_table"])
    assert len(decoded["projects"]) == len(bundle["project_names"])
    assert len(decoded["model_quality"]) == len(bundle["test_results"])
    assert decoded["forecast"]["available"] is True
    assert decoded["forecast"]["min_year"] == 2026
    assert decoded["forecast"]["max_year"] == 2036


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="model_bundle.pkl not generated")
def test_serialize_prediction_default_and_future_strict_json() -> None:
    bundle = load_bundle_cached()
    forecast = load_forecast_cached()
    default_zone = int(bundle["zone_table"].sort_values("zone_no").iloc[0]["zone_no"])
    payload = serialize_prediction(
        bundle,
        forecast,
        zone_no=default_zone,
        area_sqm=35.0,
        bedrooms=1,
        bathrooms=1,
        floor=10,
        is_new=False,
        project=None,
        complete_year=2016,
        rail_km=0.5,
        year=2036,
    )
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["main_model"] == "blend_unseen"
    assert decoded["price"] > 0
    assert decoded["knn"] > 0
    assert len(decoded["neighbours"]) == 5
    assert len(decoded["reliability_lines"]) >= 2
    assert decoded["future"] is not None
    assert decoded["future"]["year"] == 2036
    assert len(decoded["future"]["series"]) == 11


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="model_bundle.pkl not generated")
def test_serialize_prediction_known_project_auto_syncs_zone() -> None:
    bundle = load_bundle_cached()
    forecast = load_forecast_cached()
    sample_project = next(
        name for name in bundle["project_names"]
        if bundle["project_zones"].get(name.strip().lower(), 25) != 25
    )
    expected_zone = bundle["project_zones"][sample_project.strip().lower()]
    payload = serialize_prediction(
        bundle,
        forecast,
        zone_no=25,
        area_sqm=35.0,
        bedrooms=1,
        bathrooms=1,
        floor=10,
        is_new=False,
        project=sample_project,
        complete_year=2016,
        rail_km=0.5,
        year=2026,
    )
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["effective_zone_no"] == expected_zone
    assert any("ปรับใช้โซนจริง" in w for w in decoded["input_warnings"])
    assert decoded["future"] is None



def test_format_baht_and_json_safe_edges() -> None:
    assert format_baht(5_743_183.4) == "5,743,183 บาท"
    cleaned = _json_safe({
        "nan": float("nan"),
        "inf": float("inf"),
        "np_int": np.int64(7),
        "np_float": np.float64(2.5),
        "np_bool": np.bool_(True),
        "nested": [math.nan, np.int32(3)],
    })
    assert cleaned == {"nan": None, "inf": None, "np_int": 7, "np_float": 2.5, "np_bool": True, "nested": [None, 3]}
    json.dumps(cleaned, allow_nan=False)


def test_input_warnings_covers_all_guard_branches() -> None:
    listing = pd.DataFrame([{
        "zone_no": 5,
        "requested_zone_no": 1,
        "area_sqm": 30.0,
        "bedrooms": 2,
        "bathrooms": 5,
        "floor": 10,
        "is_new_unit": True,
        "complete_year": 2015,
    }])
    warnings = input_warnings(listing, zone_labels={1: "โซนหนึ่ง", 5: "วิทยุ-เพลินจิต"})
    assert any("ปรับใช้โซนจริง" in w for w in warnings)
    assert any("พื้นที่น้อยมาก" in w for w in warnings)
    assert any("ห้องน้ำมากกว่าห้องนอน" in w for w in warnings)
    assert any("ห้องมือ 1" in w for w in warnings)
    assert any("Leasehold" in w for w in warnings)


def test_filter_projects_and_list_zones_helpers() -> None:
    fake_bundle = {
        "project_names": ["Ashton Asoke", "Ideo Q Siam", "Ashton Morph 38"],
        "zone_table": pd.DataFrame([
            {"zone_no": 2, "zone_name_th": "สีลม", "district_th": "บางรัก"},
            {"zone_no": 1, "zone_name_th": "อโศก", "district_th": "วัฒนา"},
        ]),
    }
    assert filter_projects(fake_bundle, "ashton", limit=1) == ["Ashton Asoke"]
    assert filter_projects(fake_bundle, "ASHTON", limit=10) == ["Ashton Asoke", "Ashton Morph 38"]
    zones = list_zones(fake_bundle)
    assert [z["zone_no"] for z in zones] == [1, 2]
    assert zones[0]["label"] == "อโศก (เขตวัฒนา)"
