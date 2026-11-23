"""End-to-end checks of both web front ends: the Streamlit page (driven with AppTest, the way a
person clicks it) and the Vercel FastAPI backend, plus a sweep over every zone and input edge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.inference import BUNDLE_PATH, load_bundle_cached, load_forecast_cached, serialize_prediction

pytestmark = pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="model_bundle.pkl not generated")

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
# KNN and the main model use different methods; a factor of 1.6 either way flags the old bug
# (KNN at 0.46x in central zones when no project was chosen) without failing on honest spread
MAX_KNN_FACTOR = 1.6


def default_request(bundle, **changes):
    request = dict(zone_no=int(bundle["zone_table"].sort_values("zone_no").iloc[0]["zone_no"]), area_sqm=35.0,
                   bedrooms=1, bathrooms=1, floor=10, is_new=False, project=None, complete_year=2016,
                   rail_km=0.5, year=2026)
    request.update(changes)
    return request


# ---------- Streamlit page ----------

@pytest.fixture
def page():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP, default_timeout=120)


def markdown_text(at) -> str:
    return "\n".join(block.value for block in at.markdown)


def test_streamlit_shows_nothing_before_the_button(page):
    page.run()
    assert not page.exception
    assert "ราคาวันนี้" not in markdown_text(page)


def test_streamlit_year_slider_updates_without_pressing_the_button_again(page):
    page.run()
    page.button[0].click().run()
    assert not page.exception
    estimate = page.session_state["last_estimate"]
    text = markdown_text(page)
    assert "ราคาวันนี้ (ปี 2026)" in text
    assert "คาดการณ์ราคาปี" not in text

    page.slider(key="year").set_value(2036).run()
    text = markdown_text(page)
    assert "คาดการณ์ราคาปี 2036" in text
    # the estimate is reused, not recomputed, and today's price stays the same
    assert page.session_state["last_estimate"] is estimate
    forecast = load_forecast_cached()
    factor = next(item["factor"] for item in forecast["forecast"] if item["year"] == 2036)
    expected = f'{estimate["result"]["price"] * factor:,.0f} บาท'
    assert expected in text

    page.slider(key="year").set_value(2030).run()
    assert "คาดการณ์ราคาปี 2030" in markdown_text(page)
    page.slider(key="year").set_value(2026).run()
    assert "คาดการณ์ราคาปี" not in markdown_text(page)


def test_streamlit_new_session_starts_empty(page):
    page.run()
    page.button[0].click().run()
    from streamlit.testing.v1 import AppTest
    fresh = AppTest.from_file(APP, default_timeout=120).run()
    assert "last_estimate" not in fresh.session_state
    assert fresh.slider(key="year").value == 2026


def test_streamlit_known_project(page):
    page.run()
    bundle = load_bundle_cached()
    known = set(bundle["known_building"].history["key"])
    project = next(name for name in bundle["project_names"] if name.strip().lower() in known)
    page.selectbox[3].set_value(project)
    page.button[0].click().run()
    assert not page.exception
    assert "ราคาห้องอื่นในโครงการเดียวกัน" in markdown_text(page)


# ---------- shared inference, every zone and input edge ----------

@pytest.fixture(scope="module")
def bundle():
    return load_bundle_cached()


def test_every_zone_without_a_project(bundle):
    forecast = load_forecast_cached()
    names = bundle["zone_table"].set_index("zone_no")["zone_name_th"]
    for zone in bundle["zone_table"]["zone_no"]:
        result = serialize_prediction(bundle, forecast, **default_request(bundle, zone_no=int(zone)))
        json.dumps(result, allow_nan=False)
        assert result["building"]["location_from_zone"]
        ratio = result["knn"] / result["price"]
        assert 1 / MAX_KNN_FACTOR <= ratio <= MAX_KNN_FACTOR, (zone, ratio)
        assert all(row["zone_name_th"] == names[int(zone)] for row in result["neighbours"]), zone


@pytest.mark.parametrize("changes", [
    dict(area_sqm=15.0, bedrooms=0), dict(area_sqm=500.0, bedrooms=5, bathrooms=5), dict(floor=1),
    dict(floor=80), dict(is_new=True), dict(complete_year=1980, rail_km=3.0), dict(complete_year=2030, rail_km=0.0),
])
def test_input_edges_give_finite_prices(bundle, changes):
    forecast = load_forecast_cached()
    result = serialize_prediction(bundle, forecast, **default_request(bundle, **changes))
    json.dumps(result, allow_nan=False)
    assert np.isfinite(result["price"]) and result["price"] > 0
    assert 15_000 <= result["price_per_sqm"] <= 450_000


def test_forecast_series_covers_every_year(bundle):
    forecast = load_forecast_cached()
    result = serialize_prediction(bundle, forecast, **default_request(bundle))
    years = [row["year"] for row in result["forecast_series"]["years"]]
    assert years == list(range(2026, 2037))
    factors = {item["year"]: item["factor"] for item in forecast["forecast"]}
    for row in result["forecast_series"]["years"]:
        assert row["price"] == pytest.approx(result["price"] * factors[row["year"]])


# ---------- Vercel FastAPI backend ----------

@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from api.index import app
    return TestClient(app)


def test_api_meta_and_predict(client, bundle):
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert len(meta["zones"]) == len(bundle["zone_table"])
    response = client.post("/api/predict", json=default_request(bundle))
    assert response.status_code == 200
    data = response.json()
    assert data["forecast_series"]["base_year"] == 2026
    assert data["building"]["location_from_zone"] is True


def test_api_rejects_bad_input(client, bundle):
    assert client.post("/api/predict", json=default_request(bundle, area_sqm=5)).status_code == 422
    assert client.post("/api/predict", json=default_request(bundle, zone_no=199)).status_code == 422
