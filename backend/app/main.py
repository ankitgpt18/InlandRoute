"""
AIDSTL Project — FastAPI Application Entry Point
=================================================
"Predicting Inland Waterway Navigability Using Satellite Remote Sensing
 and Deep Learning"

Study areas
-----------
  NW-1 : Ganga       — Varanasi → Haldia   (~1,620 km)
  NW-2 : Brahmaputra — Dhubri  → Sadiya    (~891 km)

This module wires together:
  - FastAPI application factory with lifespan context manager
  - CORS middleware (configurable origins)
  - Prometheus metrics instrumentation
  - Structlog structured logging
  - All API routers under /api/v1
  - Global exception handlers
  - OpenAPI documentation at /docs and /redoc
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import get_settings
from app.core.database import close_db, init_db

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

settings = get_settings()

# ---------------------------------------------------------------------------
# Logging configuration (structlog)
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """
    Configure structlog for structured, levelled logging.

    In production (LOG_FORMAT=json), emits newline-delimited JSON that
    is suitable for ingestion by Datadog, Loki, Cloud Logging, etc.

    In development (LOG_FORMAT=console), emits colourised human-readable
    output using structlog's ConsoleRenderer.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_FORMAT == "json":
        processors: list[Any] = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
        renderer = structlog.processors.JSONRenderer()
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so third-party libraries (uvicorn,
    # SQLAlchemy, etc.) emit through the same pipeline.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Suppress noisy third-party loggers in production
    if not settings.DEBUG:
        for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_logging()
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup & shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Startup sequence
    ----------------
    1.  Initialise the PostgreSQL + PostGIS connection pool.
    2.  Load ML model artefacts (TFT, Swin Transformer, classifier, scaler).
    3.  Authenticate with Google Earth Engine (GEE).

    Shutdown sequence
    -----------------
    1.  Release the database connection pool.
    2.  Flush Redis connections (via ModelService and AlertService).
    3.  Free GPU memory (if CUDA was used).
    """
    startup_start = time.monotonic()
    logger.info(
        "AIDSTL API starting up",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
    )

    # ── 1. Database ────────────────────────────────────────────────────────
    try:
        await init_db()
        logger.info("PostgreSQL + PostGIS connection pool initialised.")
    except Exception as exc:
        logger.error(
            "Database initialisation failed — the API will start without DB support.",
            error=str(exc),
        )
        # Non-fatal in development; fatal in production
        if settings.is_production:
            raise

    # ── 2. ML Models ───────────────────────────────────────────────────────
    try:
        from app.services.model_service import ModelService

        model_svc = await ModelService.get_instance()
        await model_svc.load_models()
        status_info = model_svc.health_status()
        logger.info(
            "ML models loaded",
            version=status_info.get("model_version"),
            device=status_info.get("device"),
            tft=status_info.get("tft_loaded"),
            swin=status_info.get("swin_loaded"),
            classifier=status_info.get("classifier_loaded"),
        )
    except Exception as exc:
        logger.warning(
            "ML model loading encountered an error — stub models will be used.",
            error=str(exc),
            exc_info=not settings.is_production,
        )
        # In production, we allow startup to proceed but flag the degraded state;
        # individual prediction requests will surface the error gracefully.

    # ── 3. GEE ─────────────────────────────────────────────────────────────
    try:
        from app.services.gee_service import get_gee_service

        gee_svc = get_gee_service()
        await gee_svc.initialize()
        health = await gee_svc.health_check()
        logger.info(
            "GEE service initialised",
            status=health.get("status"),
            mock_mode=health.get("mock_mode"),
        )
    except Exception as exc:
        logger.warning(
            "GEE initialisation failed — feature extraction will use mock data.",
            error=str(exc),
        )
        # GEE is optional; the API can still serve cached or synthetic predictions.

    elapsed = round(time.monotonic() - startup_start, 2)
    logger.info("AIDSTL API startup complete", elapsed_seconds=elapsed)

    # ── Hand control to FastAPI ────────────────────────────────────────────
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("AIDSTL API shutting down …")

    # Close model service (Redis + GPU)
    try:
        from app.services.model_service import ModelService

        svc = await ModelService.get_instance()
        await svc.close()
        logger.info("ModelService closed.")
    except Exception as exc:
        logger.warning("ModelService shutdown error.", error=str(exc))

    # Close alert service (Redis)
    try:
        from app.services.alert_service import get_alert_service

        alert_svc = await get_alert_service()
        await alert_svc.close()
        logger.info("AlertService closed.")
    except Exception as exc:
        logger.warning("AlertService shutdown error.", error=str(exc))

    # Close database pool
    try:
        await close_db()
        logger.info("Database pool closed.")
    except Exception as exc:
        logger.warning("Database shutdown error.", error=str(exc))

    logger.info("AIDSTL API shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Returns a fully configured :class:`FastAPI` application with:
    - Lifespan startup / shutdown hooks
    - CORS, GZip, and TrustedHost middleware
    - Prometheus metrics instrumentation
    - All API routers mounted under /api/v1
    - Global exception handlers
    - OpenAPI documentation at /docs

    Returns
    -------
    FastAPI
        The configured application instance.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        description=_OPENAPI_DESCRIPTION,
        version=settings.APP_VERSION,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 2,
            "syntaxHighlight.theme": "obsidian",
            "docExpansion": "none",
            "tryItOutEnabled": True,
        },
        contact={
            "name": "AIDSTL Research Team",
            "email": "aidstl@example.ac.in",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=_OPENAPI_TAGS,
    )

    # ── Middleware (order matters — outermost executes first on request) ───
    _add_middleware(app)

    # ── Exception handlers ─────────────────────────────────────────────────
    _add_exception_handlers(app)

    # ── Routers ────────────────────────────────────────────────────────────
    _mount_routers(app)

    # ── Prometheus metrics ─────────────────────────────────────────────────
    if settings.ENABLE_METRICS:
        _setup_prometheus(app)

    # ── Root redirect ──────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        """Root endpoint — redirects clients to the API documentation."""
        return JSONResponse(
            content={
                "message": f"Welcome to {settings.APP_NAME} v{settings.APP_VERSION}",
                "description": (
                    "Predicting Inland Waterway Navigability Using "
                    "Satellite Remote Sensing and Deep Learning"
                ),
                "study_areas": {
                    "NW-1": "Ganga — Varanasi → Haldia (~1,620 km)",
                    "NW-2": "Brahmaputra — Dhubri → Sadiya (~891 km)",
                },
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health",
                "api_prefix": "/api/v1",
            }
        )

    return app


# ---------------------------------------------------------------------------
# Middleware registration
# ---------------------------------------------------------------------------


def _add_middleware(app: FastAPI) -> None:
    """Register all middleware with the application."""

    # TrustedHost — guards against HTTP Host header injection
    if settings.ALLOWED_HOSTS and not settings.DEBUG:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )

    # CORS — allow configured origins (strict in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=settings.ALLOW_CREDENTIALS,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "X-Request-ID",
            "X-AIDSTL-Signature",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Process-Time",
            "X-Cache-Status",
            "Content-Disposition",
        ],
        max_age=600,  # seconds to cache CORS preflight responses
    )

    # GZip — compress responses larger than 1 KB
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # Request timing & structured logging middleware
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Any) -> Response:
        """
        Log every HTTP request with timing and structured fields.

        Adds the following response headers:
        - ``X-Request-ID`` — a per-request trace ID (echo from request or generated)
        - ``X-Process-Time`` — server-side processing time in milliseconds
        """
        import uuid as _uuid

        # Extract or generate request ID for distributed tracing
        request_id = request.headers.get("X-Request-ID", str(_uuid.uuid4())[:8])

        # Bind request context to structlog so all log lines in this request
        # automatically include the request_id, method, and path.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        t0 = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            logger.error(
                "Unhandled exception in request pipeline",
                exc_info=True,
                error=str(exc),
            )
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=_error_envelope(
                    "INTERNAL_SERVER_ERROR",
                    "An unexpected error occurred. Please try again later.",
                    request_id=request_id,
                ),
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Attach trace headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed_ms}ms"

        # Log the completed request
        log_fn = logger.warning if response.status_code >= 400 else logger.info
        log_fn(
            "HTTP request completed",
            status_code=response.status_code,
            duration_ms=elapsed_ms,
            request_id=request_id,
        )

        structlog.contextvars.clear_contextvars()
        return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def _error_envelope(
    error_code: str,
    message: str,
    detail: Any = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build a consistent error response envelope."""
    body: dict[str, Any] = {
        "error": {
            "code": error_code,
            "message": message,
        },
        "success": False,
    }
    if detail is not None:
        body["error"]["detail"] = detail
    if request_id is not None:
        body["request_id"] = request_id
    return body


def _add_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions with a consistent JSON envelope."""
        request_id = request.headers.get("X-Request-ID", "")
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=str(request.url.path),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_envelope(
                error_code=_status_to_code(exc.status_code),
                message=str(exc.detail),
                request_id=request_id or None,
            ),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors with structured field-level detail."""
        request_id = request.headers.get("X-Request-ID", "")
        errors = exc.errors()

        # Build a friendlier error list
        field_errors = []
        for err in errors:
            loc = " → ".join(str(p) for p in err.get("loc", []))
            field_errors.append(
                {
                    "field": loc,
                    "message": err.get("msg", ""),
                    "type": err.get("type", ""),
                    "input": err.get("input"),
                }
            )

        logger.warning(
            "Request validation failed",
            path=str(request.url.path),
            method=request.method,
            errors=field_errors,
            request_id=request_id,
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_envelope(
                error_code="VALIDATION_ERROR",
                message=(
                    f"Request validation failed: "
                    f"{len(field_errors)} field error(s). "
                    "Check 'detail' for per-field descriptions."
                ),
                detail=field_errors,
                request_id=request_id or None,
            ),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError as a 422 Unprocessable Entity."""
        request_id = request.headers.get("X-Request-ID", "")
        logger.warning(
            "ValueError in request handler",
            path=str(request.url.path),
            error=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_envelope(
                error_code="INVALID_INPUT",
                message=str(exc),
                request_id=request_id or None,
            ),
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(
        request: Request, exc: RuntimeError
    ) -> JSONResponse:
        """Handle RuntimeError (e.g. service not initialised) as 503."""
        request_id = request.headers.get("X-Request-ID", "")
        logger.error(
            "RuntimeError in request handler",
            path=str(request.url.path),
            error=str(exc),
            request_id=request_id,
            exc_info=settings.DEBUG,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_envelope(
                error_code="SERVICE_UNAVAILABLE",
                message=(
                    str(exc)
                    if settings.DEBUG
                    else "A required service is temporarily unavailable. "
                    "Please try again shortly."
                ),
                request_id=request_id or None,
            ),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for any unhandled exceptions."""
        request_id = request.headers.get("X-Request-ID", "")
        logger.error(
            "Unhandled exception",
            path=str(request.url.path),
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc() if settings.DEBUG else None,
            request_id=request_id,
            exc_info=True,
        )
        detail = traceback.format_exc() if settings.DEBUG else None
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_envelope(
                error_code="INTERNAL_SERVER_ERROR",
                message=(
                    f"[DEBUG] {type(exc).__name__}: {exc}"
                    if settings.DEBUG
                    else "An unexpected internal error occurred. "
                    "Our team has been notified."
                ),
                detail=detail,
                request_id=request_id or None,
            ),
        )


def _status_to_code(status_code: int) -> str:
    """Map an HTTP status code integer to a readable error code string."""
    return {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        408: "REQUEST_TIMEOUT",
        409: "CONFLICT",
        410: "GONE",
        422: "UNPROCESSABLE_ENTITY",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_SERVER_ERROR",
        501: "NOT_IMPLEMENTED",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT",
    }.get(status_code, f"HTTP_{status_code}")


# ---------------------------------------------------------------------------
# Router mounting
# ---------------------------------------------------------------------------


def _mount_routers(app: FastAPI) -> None:
    """Mount all sub-routers onto the application."""

    # Health endpoints (mounted at root — no /api/v1 prefix — for
    # compatibility with Kubernetes liveness/readiness probe configuration)
    from app.api.routes.health import router as health_router

    app.include_router(health_router)

    # All domain routers are versioned under /api/v1
    from app.api.routes.alerts import router as alerts_router
    from app.api.routes.analytics import router as analytics_router
    from app.api.routes.navigability import router as navigability_router
    from app.api.routes.segments import router as segments_router

    API_PREFIX = "/api/v1"

    app.include_router(navigability_router, prefix=API_PREFIX)
    app.include_router(segments_router, prefix=API_PREFIX)
    app.include_router(alerts_router, prefix=API_PREFIX)
    app.include_router(analytics_router, prefix=API_PREFIX)

    logger.debug(
        "All routers mounted",
        prefix=API_PREFIX,
        routers=[
            f"{API_PREFIX}{navigability_router.prefix}",
            f"{API_PREFIX}{segments_router.prefix}",
            f"{API_PREFIX}{alerts_router.prefix}",
            f"{API_PREFIX}{analytics_router.prefix}",
            health_router.prefix,
        ],
    )


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


def _setup_prometheus(app: FastAPI) -> None:
    """
    Instrument the application with Prometheus metrics.

    Exposes the following metrics (among others added by the instrumentator):
    - ``http_requests_total``         — total request count by method/path/status
    - ``http_request_duration_seconds`` — request latency histogram
    - ``http_requests_in_progress``   — currently active requests gauge

    Metrics are exposed at ``GET /metrics``.
    """
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[
            "/metrics",
            "/health/live",
            "/health/ready",
            "/openapi.json",
            "/favicon.ico",
        ],
        inprogress_name="aidstl_http_requests_in_progress",
        inprogress_labels=True,
    )

    instrumentator.instrument(app)
    instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)

    logger.info("Prometheus metrics instrumentation enabled at /metrics.")


# ---------------------------------------------------------------------------
# OpenAPI metadata
# ---------------------------------------------------------------------------

_OPENAPI_DESCRIPTION = """
## AIDSTL — Inland Waterway Navigability Prediction API

**Predicting Inland Waterway Navigability Using Satellite Remote Sensing and Deep Learning**

### Study Areas

| ID    | Waterway       | Route                     | Length   |
|-------|----------------|---------------------------|----------|
| NW-1  | Ganga          | Varanasi → Haldia         | ~1,620 km |
| NW-2  | Brahmaputra    | Dhubri → Sadiya           | ~891 km  |

Rivers are divided into **5-km analysis units** for spatial predictions.

### Navigability Classification

| Class         | Depth (m) | Width (m) | Description                        |
|---------------|-----------|-----------|-------------------------------------|
| `navigable`   | ≥ 3.0     | ≥ 50      | Full commercial navigation possible |
| `conditional` | ≥ 1.5     | ≥ 25      | Restricted / shallow-draft only     |
| `non_navigable` | < 1.5   | < 25      | Navigation not recommended          |

### ML Architecture

The prediction pipeline combines:

1. **Temporal Fusion Transformer (TFT)** — depth time-series regression
   Input: 12-month sequence of Sentinel-2 spectral features + hydrological ancillaries
   Output: depth point estimate + 90% credible interval (q10, q90)

2. **Swin Transformer** — water extent segmentation
   Input: Sentinel-2 multispectral image patches (10 bands, 64×64 px)
   Output: water fraction + channel width estimate

3. **Ensemble** — weighted combination (TFT 65%, Swin 35%)

4. **LightGBM Classifier** — 3-class navigability classification
   Input: [depth, width, 25 spectral features]
   Output: class probabilities + SHAP feature importances

### Data Sources

- **Copernicus Sentinel-2** L2A (Surface Reflectance, via Google Earth Engine)
- **CWC** Central Water Commission daily gauge observations
- **IWAI** Inland Waterways Authority of India navigability records
- **IMD** India Meteorological Department precipitation grids

### Authentication

This development instance does not require authentication.
Production deployments use OAuth 2.0 Bearer tokens.

---
*Developed by the AIDSTL Research Team.*
"""

_OPENAPI_TAGS = [
    {
        "name": "Navigability",
        "description": (
            "Core navigability prediction endpoints — maps, calendars, "
            "depth profiles, and ML inference."
        ),
    },
    {
        "name": "Segments",
        "description": (
            "River segment data, historical navigability records, and "
            "Sentinel-2 spectral feature retrieval."
        ),
    },
    {
        "name": "Alerts",
        "description": (
            "Risk alert generation, management, acknowledgement, and "
            "webhook subscriptions for real-time notifications."
        ),
    },
    {
        "name": "Analytics",
        "description": (
            "Multi-year trend analysis, seasonal pattern detection, "
            "ML model performance metrics, and feature importance."
        ),
    },
    {
        "name": "Health",
        "description": (
            "Liveness, readiness, and component-level health checks "
            "for Kubernetes probes and operational monitoring."
        ),
    },
]

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app: FastAPI = create_app()

# ---------------------------------------------------------------------------
# Uvicorn entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        # Graceful shutdown timeout
        timeout_graceful_shutdown=30,
    )
