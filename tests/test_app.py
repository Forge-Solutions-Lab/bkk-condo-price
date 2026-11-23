"""Unit and smoke tests for app/streamlit_app.py input guards and prediction pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from app.streamlit_app import build_input, input_warnings, load_bundle, predict, BUNDLE_PATH


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="model_bundle.pkl not generated")
def test_app_predict_manual_input_and_leasehold_advisory() -> None:
    bundle = load_bundle()
    # Zone 5 = Witthayu / Ploenchit / Langsuan (leasehold advisory zone)
    listing = build_input(
        bundle, zone_no=5, area=55.0, bedrooms=1, bathrooms=1, floor=15,
        is_new=False, project=None, complete_year=2018, rail_km=0.4,
    )
    assert int(listing["requested_zone_no"].iloc[0]) == 5
    assert bool(listing["manual_building_input"].iloc[0]) is True
    # Verify mall_km falls back to training median rather than the 3.0 km cap
    bf = bundle["building_features"].transform(listing)
    assert bf["mall_km"].iloc[0] == pytest.approx(bundle["building_features"].medians["mall_km"])

    warnings = input_warnings(listing)
    assert any("Leasehold" in w for w in warnings)

    result = predict(bundle, listing)
    assert result["main_model"] == "blend_unseen"
    assert result["price"] > 0
    assert result["blend"]["linear"] > 0 and result["blend"]["partner"] > 0


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="model_bundle.pkl not generated")
def test_app_auto_syncs_mismatched_zone_for_known_project() -> None:
    bundle = load_bundle()
    assert bundle["project_zones"], "bundle should map project keys to their primary zone"
    # Pick a real project whose zone != 25 (Min Buri)
    sample_project = next(
        name for name in bundle["project_names"]
        if bundle["project_zones"].get(name.strip().lower(), 25) != 25
    )
    expected_zone = bundle["project_zones"][sample_project.strip().lower()]
    listing = build_input(
        bundle, zone_no=25, area=35.0, bedrooms=1, bathrooms=1, floor=10,
        is_new=False, project=sample_project, complete_year=2016, rail_km=0.5,
    )
    assert int(listing["requested_zone_no"].iloc[0]) == 25
    assert int(listing["zone_no"].iloc[0]) == expected_zone
    warnings = input_warnings(listing)
    assert any("ปรับใช้โซนจริง" in w for w in warnings)

    result = predict(bundle, listing)
    assert result["price"] > 0
