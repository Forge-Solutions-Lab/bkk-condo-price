"""FastAPI endpoint tests for api/index.py (health, meta headers, zones, projects, validation)."""

from __future__ import annotations

import pytest

from app.inference import BUNDLE_PATH

pytestmark = pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="model_bundle.pkl not generated")


@pytest.fixture(scope="module")
def api_client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from api.index import app

    return TestClient(app)


def test_health_reports_diagnostics(api_client) -> None:
    res = api_client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["bundle_exists"] is True
    assert body["bundle_size_mb"] > 0
    assert "python_version" in body


def test_meta_sets_revalidating_cache_control(api_client) -> None:
    res = api_client.get("/api/meta")
    assert res.status_code == 200
    cc = res.headers.get("cache-control", "")
    assert "max-age=0" in cc
    assert "must-revalidate" in cc


def test_zones_and_projects_lookup_endpoints(api_client) -> None:
    zones_res = api_client.get("/api/zones")
    assert zones_res.status_code == 200
    zones_body = zones_res.json()
    assert zones_body["count"] == len(zones_body["zones"])
    assert zones_body["count"] > 0

    proj_res = api_client.get("/api/projects", params={"q": "ashton", "limit": 5})
    assert proj_res.status_code == 200
    proj_body = proj_res.json()
    assert proj_body["count"] <= 5
    assert all("ashton" in p.lower() for p in proj_body["projects"])


def test_validation_error_returns_thai_detail(api_client) -> None:
    res = api_client.post(
        "/api/predict",
        json={
            "zone_no": 1,
            "area_sqm": 5.0,
            "bedrooms": 1,
            "bathrooms": 1,
            "floor": 10,
            "complete_year": 2016,
            "rail_km": 0.5,
            "is_new": False,
            "year": 2026,
        },
    )
    assert res.status_code == 422
    body = res.json()
    assert "ข้อมูลไม่ถูกต้อง" in body["detail"]
    assert "พื้นที่" in body["detail"]
