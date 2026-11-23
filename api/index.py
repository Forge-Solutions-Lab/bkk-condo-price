"""Vercel FastAPI backend for BKK Condo Price Estimator."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import (  # noqa: E402
    BUNDLE_PATH,
    FORECAST_PATH,
    filter_projects,
    list_zones,
    load_bundle_cached,
    load_forecast_cached,
    serialize_metadata,
    serialize_prediction,
)

logger = logging.getLogger("bkk_condo_api")

app = FastAPI(
    title="BKK Condo Price API",
    description="Bangkok condo price estimator powered by K-Means, Linear Regression, Boosting/ExtraTrees, and KNN.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    zone_no: int = Field(..., ge=1, le=200)
    area_sqm: float = Field(35.0, ge=15.0, le=500.0)
    bedrooms: int = Field(1, ge=0, le=5)
    bathrooms: int = Field(1, ge=1, le=5)
    floor: int = Field(10, ge=1, le=80)
    project: str | None = Field(default=None, max_length=200)
    complete_year: int = Field(2016, ge=1980, le=2030)
    rail_km: float = Field(0.5, ge=0.0, le=3.0)
    is_new: bool = False
    year: int = Field(2026, ge=2026, le=2036)


def _require_bundle() -> dict[str, Any]:
    if not BUNDLE_PATH.exists():
        logger.error("Model bundle missing at %s", BUNDLE_PATH)
        raise HTTPException(
            status_code=503,
            detail="ยังไม่มีโมเดล ให้รัน python -m src.evaluate.run ก่อน",
        )
    try:
        return load_bundle_cached()
    except Exception as exc:
        logger.exception("Failed to load model bundle")
        raise HTTPException(status_code=500, detail=f"ไม่สามารถโหลดโมเดลได้: {exc}") from exc


@app.get("/api/health")
def health() -> dict[str, Any]:
    bundle_exists = BUNDLE_PATH.exists()
    bundle_size_mb = round(BUNDLE_PATH.stat().st_size / (1024 * 1024), 2) if bundle_exists else None
    return {
        "status": "ok",
        "bundle_exists": bundle_exists,
        "bundle_size_mb": bundle_size_mb,
        "forecast_exists": FORECAST_PATH.exists(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


@app.get("/api/meta")
def get_meta(response: Response) -> dict[str, Any]:
    bundle = _require_bundle()
    forecast = load_forecast_cached()
    response.headers["Cache-Control"] = (
        "public, max-age=0, s-maxage=3600, stale-while-revalidate=86400, must-revalidate"
    )
    return serialize_metadata(bundle, forecast)


@app.get("/api/zones")
def get_zones(response: Response) -> dict[str, Any]:
    bundle = _require_bundle()
    zones = list_zones(bundle)
    response.headers["Cache-Control"] = (
        "public, max-age=0, s-maxage=3600, stale-while-revalidate=86400, must-revalidate"
    )
    return {"count": len(zones), "zones": zones}


@app.get("/api/projects")
def get_projects(
    response: Response,
    q: str = Query(default="", max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    bundle = _require_bundle()
    matches = filter_projects(bundle, query=q, limit=limit)
    response.headers["Cache-Control"] = (
        "public, max-age=0, s-maxage=300, stale-while-revalidate=3600, must-revalidate"
    )
    return {
        "query": q,
        "count": len(matches),
        "total_projects": len(bundle["project_names"]),
        "projects": matches,
    }


@app.post("/api/predict")
def post_predict(payload: PredictRequest) -> JSONResponse:
    bundle = _require_bundle()
    forecast = load_forecast_cached()
    valid_zones = set(int(z) for z in bundle["zone_table"]["zone_no"].tolist())
    if payload.zone_no not in valid_zones:
        raise HTTPException(status_code=422, detail=f"ไม่พบโซนหมายเลข {payload.zone_no}")
    try:
        result = serialize_prediction(
            bundle,
            forecast,
            zone_no=payload.zone_no,
            area_sqm=payload.area_sqm,
            bedrooms=payload.bedrooms,
            bathrooms=payload.bathrooms,
            floor=payload.floor,
            is_new=payload.is_new,
            project=payload.project,
            complete_year=payload.complete_year,
            rail_km=payload.rail_km,
            year=payload.year,
        )
        return JSONResponse(content=result, headers={"Cache-Control": "no-store"})
    except Exception as exc:
        logger.exception("Prediction failed for payload=%s", payload.model_dump())
        raise HTTPException(status_code=500, detail=f"ไม่สามารถประเมินราคาได้: {exc}") from exc


@app.get("/")
def serve_index() -> FileResponse:
    index_path = PROJECT_ROOT / "public" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)
