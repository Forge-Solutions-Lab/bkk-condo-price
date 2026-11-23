"""Tests for shared pure-Python inference and JSON serialization used by the Vercel API."""

from __future__ import annotations

import json

import pytest

from app.inference import (
    BUNDLE_PATH,
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
