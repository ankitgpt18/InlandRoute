"""
AIDSTL Project — Health Check Routes
=====================================
Liveness, readiness, and component-level health endpoints.

Routes
------
  GET /health                — overall application health (liveness)
  GET /health/models         — ML model loading status
  GET /health/gee            — Google Earth Engine connectivity
  GET /health/db             — PostgreSQL + PostGIS connectivity
  GET /health/redis          — Redis connectivity
  GET /metrics               — Prometheus metrics (redirects to instrumentator)

Design notes
------------
  - /health returns 200 even when sub-components are degraded, so that
    container orchestrators (Kubernetes, ECS) keep the pod running and
    allow it to self-heal.  A 503 is returned only when the application
    itself is in a completely broken state.
  - /health/models, /health/gee etc. return component-specific detail
    and use appropriate HTTP status codes (503 on failure).
  - All responses follow a consistent envelope:
      { "status": "healthy"|"degraded"|"unhealthy", ... }
"""

from __future__ import annotations

import asyncio
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
from app.core.config import get_settings
from app.core.database import check_db_health
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["Health"])

settings = get_settings()

# Record the time at which this module was imported (≈ app startup time)
_APP_START_TIME: float = time.monotonic()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    status: str  # "healthy" | "degraded" | "unhealthy" | "unknown"
    latency_ms: float | None = None
    detail: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Overall application health response envelope."""

    status: str
    version: str
    environment: str
    uptime_seconds: float
    timestamp: str
    components: dict[str, ComponentHealth]


class ModelHealthResponse(BaseModel):
    """Detailed ML model loading status."""

    status: str
    models_loaded: bool
    model_version: str
    device: str
    components: dict[str, bool]
    inference_count: int
    avg_inference_ms: float
    cache_ttl_seconds: int


class GEEHealthResponse(BaseModel):
    """Google Earth Engine connectivity status."""

    status: str
    initialised: bool
    mock_mode: bool
    detail: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Dependency — lazy model service import to avoid circular imports
# ---------------------------------------------------------------------------


async def _get_model_service_optional():
    """Return ModelService if available, else None."""
    try:
        from app.services.model_service import ModelService

        return await ModelService.get_instance()
    except Exception:
        return None


async def _get_gee_service_optional():
    """Return GEEService singleton if available, else None."""
    try:
        from app.services.gee_service import get_gee_service

        return get_gee_service()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uptime() -> float:
    """Return application uptime in seconds."""
    return round(time.monotonic() - _APP_START_TIME, 2)


async def _ping_redis() -> ComponentHealth:
    """Ping the Redis server and return a ComponentHealth result."""
    t0 = time.perf_counter()
    try:
        client = aioredis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        await client.ping()
        await client.close()
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return ComponentHealth(
            status="healthy",
            latency_ms=latency_ms,
            detail={"url": settings.REDIS_URL.split("@")[-1]},
        )
    except Exception as exc:
        return ComponentHealth(
            status="unhealthy",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            error=str(exc),
        )


async def _ping_db() -> ComponentHealth:
    """Check PostgreSQL + PostGIS connectivity."""
    t0 = time.perf_counter()
    try:
        result = await check_db_health()
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        is_healthy = result.get("status") == "healthy"
        return ComponentHealth(
            status="healthy" if is_healthy else "unhealthy",
            latency_ms=latency_ms,
            detail={k: v for k, v in result.items() if k not in ("status",)},
            error=result.get("error"),
        )
    except Exception as exc:
        return ComponentHealth(
            status="unhealthy",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# GET /health  — overall liveness probe
# ---------------------------------------------------------------------------


@router.get(
    "",
    summary="Application health — liveness probe",
    description=(
        "Returns the overall health of the AIDSTL API. "
        "A 200 response indicates the application process is alive. "
        "Sub-component failures are reported in the response body but do "
        "not change the HTTP status code unless ALL components fail."
    ),
    response_model=HealthResponse,
    responses={
        200: {"description": "Application is healthy or degraded"},
        503: {"description": "Application is entirely unavailable"},
    },
)
async def health_check() -> JSONResponse:
    """
    Liveness probe used by Kubernetes / ECS health checks.

    Checks:
    - Database connectivity (PostgreSQL + PostGIS)
    - Redis connectivity
    - Model loading status (non-blocking)

    Returns a ``HealthResponse`` with an aggregated status:
    - ``"healthy"``  — all components OK
    - ``"degraded"`` — one or more non-critical components failing
    - ``"unhealthy"``— critical components (DB) failing → HTTP 503
    """
    # Run component checks concurrently
    db_health, redis_health = await asyncio.gather(
        _ping_db(),
        _ping_redis(),
        return_exceptions=False,
    )

    # Model service status (non-blocking — don't fail liveness for model issues)
    model_status = "unknown"
    try:
        from app.services.model_service import ModelService

        svc = await ModelService.get_instance()
        info = svc.health_status()
        model_status = "healthy" if info.get("models_loaded") else "degraded"
    except Exception:
        model_status = "unknown"

    components: dict[str, ComponentHealth] = {
        "database": db_health,
        "redis": redis_health,
        "models": ComponentHealth(status=model_status),
    }

    # Determine overall status
    if db_health.status == "unhealthy":
        overall = "unhealthy"
    elif any(c.status in ("unhealthy", "degraded") for c in components.values()):
        overall = "degraded"
    else:
        overall = "healthy"

    body = HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        uptime_seconds=_uptime(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )

    http_status = (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if overall == "unhealthy"
        else status.HTTP_200_OK
    )

    return JSONResponse(content=body.model_dump(), status_code=http_status)


# ---------------------------------------------------------------------------
# GET /health/models  — ML model loading status
# ---------------------------------------------------------------------------


@router.get(
    "/models",
    summary="ML model loading status",
    description=(
        "Returns the loading state and performance metrics of the "
        "TFT + Swin Transformer ensemble and the navigability classifier."
    ),
    response_model=ModelHealthResponse,
    responses={
        200: {"description": "Models are loaded and operational"},
        503: {"description": "One or more models failed to load"},
    },
)
async def health_models() -> JSONResponse:
    """
    Detailed ML model health check.

    Reports:
    - Whether models are loaded
    - PyTorch device in use (cpu / cuda)
    - Per-component loading status (TFT, Swin, classifier, scaler, SHAP)
    - Running inference statistics (count, average latency)
    - Redis cache TTL configuration

    Returns HTTP 503 if models are not yet loaded.
    """
    try:
        from app.services.model_service import ModelService

        svc = await ModelService.get_instance()
        info = svc.health_status()

        body = ModelHealthResponse(
            status="healthy" if info.get("models_loaded") else "degraded",
            models_loaded=bool(info.get("models_loaded")),
            model_version=str(info.get("model_version", "unknown")),
            device=str(info.get("device", "cpu")),
            components={
                "tft": bool(info.get("tft_loaded")),
                "swin": bool(info.get("swin_loaded")),
                "navigability_classifier": bool(info.get("classifier_loaded")),
                "feature_scaler": bool(info.get("scaler_loaded")),
                "shap_explainer": bool(info.get("shap_explainer_loaded")),
            },
            inference_count=int(info.get("inference_count", 0)),
            avg_inference_ms=float(info.get("avg_inference_ms", 0.0)),
            cache_ttl_seconds=int(info.get("cache_ttl_seconds", 0)),
        )

        http_status = (
            status.HTTP_200_OK
            if body.models_loaded
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return JSONResponse(content=body.model_dump(), status_code=http_status)

    except Exception as exc:
        body = ModelHealthResponse(
            status="unhealthy",
            models_loaded=False,
            model_version="unknown",
            device="unknown",
            components={
                "tft": False,
                "swin": False,
                "navigability_classifier": False,
                "feature_scaler": False,
                "shap_explainer": False,
            },
            inference_count=0,
            avg_inference_ms=0.0,
            cache_ttl_seconds=0,
        )
        return JSONResponse(
            content={**body.model_dump(), "error": str(exc)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


# ---------------------------------------------------------------------------
# GET /health/gee  — Google Earth Engine connectivity
# ---------------------------------------------------------------------------


@router.get(
    "/gee",
    summary="Google Earth Engine connectivity",
    description=(
        "Verifies that the GEE service account is authenticated and the "
        "Earth Engine API is reachable. Returns mock_mode=true when the "
        "earthengine-api package is not installed."
    ),
    response_model=GEEHealthResponse,
    responses={
        200: {"description": "GEE is reachable"},
        503: {"description": "GEE authentication or connectivity failed"},
    },
)
async def health_gee() -> JSONResponse:
    """
    Google Earth Engine health check.

    Verifies:
    - GEE service account credentials are loaded
    - A trivial server-side computation succeeds
    - Reports ``mock_mode=true`` in development environments

    Returns HTTP 503 if GEE is uninitialised or unreachable.
    """
    try:
        from app.services.gee_service import get_gee_service

        gee_svc = get_gee_service()
        result = await gee_svc.health_check()

        initialised = result.get("status") != "uninitialised"
        mock_mode = bool(result.get("mock_mode", False))
        is_healthy = result.get("status") == "healthy"

        body = GEEHealthResponse(
            status=result.get("status", "unknown"),
            initialised=initialised,
            mock_mode=mock_mode,
            detail={k: v for k, v in result.items() if k not in ("status", "error")},
            error=result.get("error"),
        )

        http_status = (
            status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return JSONResponse(content=body.model_dump(), status_code=http_status)

    except Exception as exc:
        body = GEEHealthResponse(
            status="unhealthy",
            initialised=False,
            mock_mode=False,
            error=str(exc),
        )
        return JSONResponse(
            content=body.model_dump(),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


# ---------------------------------------------------------------------------
# GET /health/db  — Database connectivity
# ---------------------------------------------------------------------------


@router.get(
    "/db",
    summary="Database connectivity",
    description=(
        "Checks PostgreSQL + PostGIS availability and reports pool statistics."
    ),
    responses={
        200: {"description": "Database is reachable"},
        503: {"description": "Database is unreachable"},
    },
)
async def health_db() -> JSONResponse:
    """
    PostgreSQL + PostGIS health check.

    Executes a lightweight ``SELECT PostGIS_Full_Version()`` query and
    returns connection pool statistics.

    Returns HTTP 503 when the database is unreachable.
    """
    t0 = time.perf_counter()
    result = await check_db_health()
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    is_healthy = result.get("status") == "healthy"

    body = {
        **result,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    http_status = (
        status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(content=body, status_code=http_status)


# ---------------------------------------------------------------------------
# GET /health/redis  — Redis connectivity
# ---------------------------------------------------------------------------


@router.get(
    "/redis",
    summary="Redis connectivity",
    description="Pings the Redis server and returns latency.",
    responses={
        200: {"description": "Redis is reachable"},
        503: {"description": "Redis is unreachable"},
    },
)
async def health_redis() -> JSONResponse:
    """
    Redis health check.

    Issues a ``PING`` command and measures round-trip latency.
    Returns HTTP 503 when Redis is unreachable.
    """
    result = await _ping_redis()
    body = {
        "status": result.status,
        "latency_ms": result.latency_ms,
        "detail": result.detail,
        "error": result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    http_status = (
        status.HTTP_200_OK
        if result.status == "healthy"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(content=body, status_code=http_status)


# ---------------------------------------------------------------------------
# GET /health/ready  — Kubernetes readiness probe
# ---------------------------------------------------------------------------


@router.get(
    "/ready",
    summary="Readiness probe",
    description=(
        "Returns 200 only when the application is fully initialised and "
        "ready to serve traffic (models loaded, DB reachable). "
        "Used by Kubernetes readiness probes."
    ),
    responses={
        200: {"description": "Application is ready"},
        503: {"description": "Application is not yet ready"},
    },
)
async def readiness_probe() -> JSONResponse:
    """
    Kubernetes readiness probe.

    The application is considered 'ready' when:
    1. The database is reachable.
    2. ML models have been loaded (or at least the load was attempted).

    Returns HTTP 503 until both conditions are met, preventing traffic
    from being routed to this instance while it is still initialising.
    """
    db_health, redis_health = await asyncio.gather(
        _ping_db(),
        _ping_redis(),
    )

    models_ready = False
    try:
        from app.services.model_service import ModelService

        svc = await ModelService.get_instance()
        models_ready = svc.health_status().get("models_loaded", False)
    except Exception:
        models_ready = False

    ready = (
        db_health.status == "healthy"
        and redis_health.status == "healthy"
        and models_ready
    )

    body = {
        "ready": ready,
        "checks": {
            "database": db_health.status,
            "redis": redis_health.status,
            "models": "loaded" if models_ready else "not_loaded",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    http_status = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=body, status_code=http_status)


# ---------------------------------------------------------------------------
# GET /health/live  — Kubernetes liveness probe (ultra-lightweight)
# ---------------------------------------------------------------------------


@router.get(
    "/live",
    summary="Liveness probe (lightweight)",
    description=(
        "Ultra-lightweight liveness check that returns 200 as long as the "
        "Python process is alive.  Does NOT check any external dependencies."
    ),
    responses={200: {"description": "Process is alive"}},
)
async def liveness_probe() -> JSONResponse:
    """
    Minimal Kubernetes liveness probe.

    Returns immediately without any I/O.  Used as the container-level
    liveness probe to detect deadlocked or frozen processes.
    """
    return JSONResponse(
        content={
            "alive": True,
            "uptime_seconds": _uptime(),
            "pid": os.getpid(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        status_code=status.HTTP_200_OK,
    )


# ---------------------------------------------------------------------------
# GET /health/info  — Application metadata
# ---------------------------------------------------------------------------


@router.get(
    "/info",
    summary="Application information",
    description="Returns build metadata, configuration summary, and study-area info.",
)
async def app_info() -> JSONResponse:
    """
    Application metadata endpoint.

    Returns non-sensitive configuration details useful for debugging
    and client-side feature detection.
    """
    return JSONResponse(
        content={
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "uptime_seconds": _uptime(),
            "study_areas": {
                "NW-1": {
                    "name": "National Waterway 1 — Ganga",
                    "route": "Varanasi → Haldia",
                    "length_km": 1620,
                    "bbox": settings.NW1_BBOX,
                },
                "NW-2": {
                    "name": "National Waterway 2 — Brahmaputra",
                    "route": "Dhubri → Sadiya",
                    "length_km": 891,
                    "bbox": settings.NW2_BBOX,
                },
            },
            "model": {
                "ensemble": "TFT + Swin Transformer",
                "classifier": "LightGBM",
                "segment_length_km": settings.SEGMENT_LENGTH_KM,
            },
            "navigability_thresholds": {
                "navigable": {
                    "min_depth_m": settings.DEPTH_NAVIGABLE_MIN,
                    "min_width_m": settings.WIDTH_NAVIGABLE_MIN,
                },
                "conditional": {
                    "min_depth_m": settings.DEPTH_CONDITIONAL_MIN,
                    "min_width_m": settings.WIDTH_CONDITIONAL_MIN,
                },
            },
            "api_docs": "/docs",
            "openapi_schema": "/openapi.json",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
