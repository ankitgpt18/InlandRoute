"""
AIDSTL Project — Navigability API Routes
=========================================
REST endpoints for inland waterway navigability predictions.

Endpoints
---------
  GET  /api/v1/navigability/{waterway_id}/map
  GET  /api/v1/navigability/{waterway_id}/calendar
  GET  /api/v1/navigability/{waterway_id}/depth-profile
  GET  /api/v1/navigability/{waterway_id}/stats
  POST /api/v1/navigability/predict
  POST /api/v1/navigability/predict/batch

Study areas
-----------
  NW-1 : Ganga       — Varanasi → Haldia   (~1,620 km)
  NW-2 : Brahmaputra — Dhubri  → Sadiya    (~891 km)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from app.core.config import get_settings
from app.models.schemas.navigability import (
    BatchPredictionRequest,
    DepthProfile,
    HistoricalComparison,
    NavigabilityMap,
    NavigabilityPrediction,
    PredictionRequest,
    SeasonalCalendar,
    TaskStatus,
    WaterwayID,
    WaterwayStats,
)
from app.services.model_service import ModelService
from app.services.navigability_service import NavigabilityService
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Query,
    status,
)
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/navigability",
    tags=["Navigability"],
    responses={
        404: {"description": "Waterway or segment not found"},
        422: {"description": "Validation error — check request parameters"},
        503: {"description": "ML models not loaded or GEE unavailable"},
    },
)

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_nav_service: Optional[NavigabilityService] = None


def get_nav_service() -> NavigabilityService:
    """FastAPI dependency — returns the singleton NavigabilityService."""
    global _nav_service
    if _nav_service is None:
        _nav_service = NavigabilityService()
    return _nav_service


async def get_model_service() -> ModelService:
    """FastAPI dependency — returns the singleton ModelService."""
    return await ModelService.get_instance()


# ---------------------------------------------------------------------------
# Path / query parameter types
# ---------------------------------------------------------------------------

WaterwayPathParam = Annotated[
    str,
    Path(
        title="Waterway ID",
        description="National Waterway identifier — 'NW-1' (Ganga) or 'NW-2' (Brahmaputra).",
        pattern=r"^NW-[12]$",
        examples=["NW-1", "NW-2"],
    ),
]

MonthQueryParam = Annotated[
    int,
    Query(
        ge=1,
        le=12,
        title="Month",
        description="Calendar month (1 = January … 12 = December).",
        examples=[6],
    ),
]

YearQueryParam = Annotated[
    int,
    Query(
        ge=2015,
        le=2100,
        title="Year",
        description="Calendar year (≥ 2015, the start of the Sentinel-2 data record).",
        examples=[2024],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_month_year() -> tuple[int, int]:
    """Return (month, year) for the current UTC date."""
    now = datetime.now(timezone.utc)
    return now.month, now.year


def _validate_waterway(waterway_id: str) -> str:
    """Raise 404 if the waterway is not supported."""
    if waterway_id not in settings.SUPPORTED_WATERWAYS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Waterway '{waterway_id}' is not supported. "
                f"Supported waterways: {settings.SUPPORTED_WATERWAYS}"
            ),
        )
    return waterway_id


# ---------------------------------------------------------------------------
# In-memory task registry (replace with Celery task IDs in production)
# ---------------------------------------------------------------------------

_batch_tasks: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# GET /navigability/{waterway_id}/map
# ---------------------------------------------------------------------------


@router.get(
    "/{waterway_id}/map",
    response_model=NavigabilityMap,
    summary="Get Navigability Map",
    description=(
        "Return a full segment-level navigability map for the specified "
        "waterway, month, and year. Each 5-km segment includes predicted "
        "depth, width, navigability class, and risk score.\n\n"
        "Results are cached in Redis for 6 hours."
    ),
    responses={
        200: {
            "description": "Navigability map returned successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "waterway_id": "NW-1",
                        "month": 7,
                        "year": 2024,
                        "total_segments": 324,
                        "navigable_count": 210,
                        "overall_navigability_pct": 64.8,
                    }
                }
            },
        },
    },
)
async def get_navigability_map(
    waterway_id: WaterwayPathParam,
    month: MonthQueryParam = None,  # type: ignore[assignment]
    year: YearQueryParam = None,  # type: ignore[assignment]
    force_refresh: bool = Query(
        False,
        description="Bypass the Redis cache and recompute predictions from scratch.",
    ),
    nav_service: NavigabilityService = Depends(get_nav_service),
) -> NavigabilityMap:
    """
    Retrieve the navigability prediction map for a full waterway.

    - **waterway_id**: `NW-1` (Ganga: Varanasi–Haldia) or `NW-2` (Brahmaputra: Dhubri–Sadiya)
    - **month**: calendar month 1–12 (defaults to current month)
    - **year**: calendar year ≥ 2015 (defaults to current year)
    - **force_refresh**: set to `true` to bypass the 6-hour Redis cache
    """
    _validate_waterway(waterway_id)

    if month is None or year is None:
        cur_month, cur_year = _current_month_year()
        month = month or cur_month
        year = year or cur_year

    try:
        nav_map = await nav_service.get_navigability_map(
            waterway_id=waterway_id,
            month=month,
            year=year,
            force_refresh=force_refresh,
        )
        return nav_map
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Error building navigability map for %s %02d/%d: %s",
            waterway_id,
            month,
            year,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to compute navigability map: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# GET /navigability/{waterway_id}/calendar
# ---------------------------------------------------------------------------


@router.get(
    "/{waterway_id}/calendar",
    response_model=SeasonalCalendar,
    summary="Get Seasonal Calendar",
    description=(
        "Return a 12-month navigability calendar for every 5-km segment "
        "of the waterway for a given year. Useful for long-range operational "
        "planning and identifying peak navigation windows.\n\n"
        "Results are cached in Redis for 24 hours."
    ),
)
async def get_seasonal_calendar(
    waterway_id: WaterwayPathParam,
    year: YearQueryParam = None,  # type: ignore[assignment]
    force_refresh: bool = Query(False, description="Bypass the 24-hour Redis cache."),
    nav_service: NavigabilityService = Depends(get_nav_service),
) -> SeasonalCalendar:
    """
    Retrieve the full 12-month navigability seasonal calendar.

    - **waterway_id**: `NW-1` or `NW-2`
    - **year**: calendar year ≥ 2015 (defaults to current year)
    - **force_refresh**: set to `true` to recompute all 12 months
    """
    _validate_waterway(waterway_id)

    if year is None:
        _, year = _current_month_year()

    try:
        calendar = await nav_service.get_seasonal_calendar(
            waterway_id=waterway_id,
            year=year,
            force_refresh=force_refresh,
        )
        return calendar
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Error building seasonal calendar for %s %d: %s",
            waterway_id,
            year,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to compute seasonal calendar: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# GET /navigability/{waterway_id}/depth-profile
# ---------------------------------------------------------------------------


@router.get(
    "/{waterway_id}/depth-profile",
    response_model=DepthProfile,
    summary="Get Longitudinal Depth Profile",
    description=(
        "Return a longitudinal depth profile along the entire waterway — "
        "predicted depth and 90% credible interval at each 5-km segment "
        "midpoint, ordered upstream to downstream. Includes bottleneck "
        "(critical segment) identification."
    ),
)
async def get_depth_profile(
    waterway_id: WaterwayPathParam,
    month: MonthQueryParam = None,  # type: ignore[assignment]
    year: YearQueryParam = None,  # type: ignore[assignment]
    force_refresh: bool = Query(False, description="Bypass the 6-hour Redis cache."),
    nav_service: NavigabilityService = Depends(get_nav_service),
) -> DepthProfile:
    """
    Retrieve the longitudinal depth profile for route planning.

    - **waterway_id**: `NW-1` or `NW-2`
    - **month**: calendar month 1–12 (defaults to current month)
    - **year**: calendar year ≥ 2015 (defaults to current year)
    """
    _validate_waterway(waterway_id)

    if month is None or year is None:
        cur_month, cur_year = _current_month_year()
        month = month or cur_month
        year = year or cur_year

    try:
        profile = await nav_service.get_depth_profile(
            waterway_id=waterway_id,
            month=month,
            year=year,
            force_refresh=force_refresh,
        )
        return profile
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Error building depth profile for %s %02d/%d: %s",
            waterway_id,
            month,
            year,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to compute depth profile: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# GET /navigability/{waterway_id}/stats
# ---------------------------------------------------------------------------


@router.get(
    "/{waterway_id}/stats",
    response_model=WaterwayStats,
    summary="Get Annual Waterway Statistics",
    description=(
        "Return comprehensive annual operational statistics for a National "
        "Waterway — navigability percentages, depth/width trends, best/worst "
        "months, and year-on-year comparisons."
    ),
)
async def get_waterway_stats(
    waterway_id: WaterwayPathParam,
    year: YearQueryParam = None,  # type: ignore[assignment]
    force_refresh: bool = Query(False, description="Bypass the 12-hour Redis cache."),
    nav_service: NavigabilityService = Depends(get_nav_service),
) -> WaterwayStats:
    """
    Retrieve annual navigability statistics for a National Waterway.

    - **waterway_id**: `NW-1` or `NW-2`
    - **year**: calendar year ≥ 2015 (defaults to current year)
    """
    _validate_waterway(waterway_id)

    if year is None:
        _, year = _current_month_year()

    try:
        stats = await nav_service.get_waterway_stats(
            waterway_id=waterway_id,
            year=year,
            force_refresh=force_refresh,
        )
        return stats
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Error building waterway stats for %s %d: %s",
            waterway_id,
            year,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to compute waterway statistics: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# GET /navigability/{waterway_id}/historical-comparison
# ---------------------------------------------------------------------------


@router.get(
    "/{waterway_id}/historical-comparison",
    response_model=HistoricalComparison,
    summary="Compare with Historical Baseline",
    description=(
        "Compare the current month's predicted navigability conditions "
        "against a multi-year historical baseline (default: 5 years). "
        "Returns depth and navigability anomalies, trend direction, and "
        "the full historical time series."
    ),
)
async def get_historical_comparison(
    waterway_id: WaterwayPathParam,
    month: MonthQueryParam = None,  # type: ignore[assignment]
    year: YearQueryParam = None,  # type: ignore[assignment]
    base_years: int = Query(
        5,
        ge=1,
        le=15,
        description="Number of historical years to include in the baseline.",
    ),
    force_refresh: bool = Query(False, description="Bypass the 24-hour Redis cache."),
    nav_service: NavigabilityService = Depends(get_nav_service),
) -> HistoricalComparison:
    """
    Retrieve anomaly comparison of current vs. historical conditions.

    - **waterway_id**: `NW-1` or `NW-2`
    - **month**: calendar month 1–12 (defaults to current month)
    - **year**: calendar year ≥ 2015 (defaults to current year)
    - **base_years**: number of years for the baseline (1–15, default 5)
    """
    _validate_waterway(waterway_id)

    if month is None or year is None:
        cur_month, cur_year = _current_month_year()
        month = month or cur_month
        year = year or cur_year

    if year - base_years < 2015:
        base_years = max(1, year - 2015)

    try:
        comparison = await nav_service.compare_with_historical(
            waterway_id=waterway_id,
            month=month,
            year=year,
            base_years=base_years,
            force_refresh=force_refresh,
        )
        return comparison
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Error computing historical comparison for %s %02d/%d: %s",
            waterway_id,
            month,
            year,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to compute historical comparison: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# POST /navigability/predict  (single segment)
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=NavigabilityPrediction,
    status_code=status.HTTP_200_OK,
    summary="Predict Single Segment Navigability",
    description=(
        "Run the TFT + Swin Transformer ensemble inference for a single "
        "river segment and return a full navigability prediction including "
        "depth estimate, width, classification, risk score, and optionally "
        "SHAP feature-importance values."
    ),
    responses={
        200: {"description": "Prediction generated successfully."},
        422: {"description": "Validation error in request body."},
        503: {"description": "ML models not loaded or GEE service unavailable."},
    },
)
async def predict_single_segment(
    request: PredictionRequest,
    model_service: ModelService = Depends(get_model_service),
) -> NavigabilityPrediction:
    """
    Predict navigability for a single river segment.

    **Request body fields:**
    - `segment.segment_id` — unique segment ID, e.g. `"NW-1-042"`
    - `segment.waterway_id` — `"NW-1"` or `"NW-2"`
    - `segment.month` / `segment.year` — temporal context
    - `segment.features` — optional pre-computed spectral features;
      if omitted, fetched from Google Earth Engine
    - `return_shap` — include SHAP feature-contribution values
    - `return_features` — include the spectral feature vector
    - `force_refresh` — bypass the 6-hour prediction cache
    """
    seg = request.segment

    # Validate waterway
    if seg.waterway_id not in settings.SUPPORTED_WATERWAYS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Waterway '{seg.waterway_id}' is not supported. "
                f"Supported: {settings.SUPPORTED_WATERWAYS}"
            ),
        )

    # Build feature dict from request
    segment_features: dict[str, Any] = {
        "segment_id": seg.segment_id,
        "waterway_id": seg.waterway_id,
        "month": seg.month,
        "year": seg.year,
        "geometry": seg.geometry
        or {
            "type": "Point",
            "coordinates": [85.0, 24.0],  # fallback centroid
        },
    }

    # Merge pre-computed spectral features if provided
    if seg.features is not None:
        feat_dict = seg.features.model_dump(exclude_none=True)
        segment_features.update(feat_dict)
        # Map schema field names to band-key names expected by the model
        band_mapping = {
            "b2_blue": "blue",
            "b3_green": "green",
            "b4_red": "red",
            "b8_nir": "nir",
            "b11_swir1": "swir1",
            "b12_swir2": "swir2",
        }
        for schema_key, band_key in band_mapping.items():
            if schema_key in feat_dict and band_key not in segment_features:
                segment_features[band_key] = feat_dict[schema_key]

    # Hydrological ancillary inputs
    if seg.gauge_discharge_m3s is not None:
        segment_features["gauge_discharge_m3s"] = seg.gauge_discharge_m3s
    if seg.gauge_water_level_m is not None:
        segment_features["gauge_water_level_m"] = seg.gauge_water_level_m
    if seg.precipitation_mm is not None:
        segment_features["precipitation_mm"] = seg.precipitation_mm

    try:
        prediction = await model_service.predict_segment(
            segment_features=segment_features,
            patches=None,
            compute_shap=request.return_shap,
            force_refresh=request.force_refresh,
        )

        # Strip features from response if not requested
        if not request.return_features:
            prediction = prediction.model_copy(
                update={"features": prediction.features.__class__()}
            )

        return prediction

    except Exception as exc:
        logger.exception(
            "Prediction failed for segment %s: %s",
            seg.segment_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# POST /navigability/predict/batch
# ---------------------------------------------------------------------------


@router.post(
    "/predict/batch",
    response_model=list[NavigabilityPrediction] | TaskStatus,
    status_code=status.HTTP_200_OK,
    summary="Batch Navigability Prediction",
    description=(
        "Run navigability predictions for multiple river segments. "
        "Batches ≤ `async_threshold` segments (default 50) are processed "
        "synchronously and the full prediction list is returned immediately. "
        "Larger batches are submitted to a Celery background worker and a "
        "`TaskStatus` response is returned with a `task_id` for polling."
    ),
)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(get_model_service),
) -> list[NavigabilityPrediction] | TaskStatus:
    """
    Batch navigability prediction endpoint.

    **Synchronous mode** (≤ `async_threshold` segments):
      Returns `list[NavigabilityPrediction]` immediately.

    **Asynchronous mode** (> `async_threshold` segments):
      Returns `TaskStatus` with `task_id`.
      Poll `GET /api/v1/tasks/{task_id}` for progress and results.
    """
    # Validate all waterway IDs
    invalid_waterways = [
        seg.waterway_id
        for seg in request.segments
        if seg.waterway_id not in settings.SUPPORTED_WATERWAYS
    ]
    if invalid_waterways:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported waterway ID(s): {list(set(str(w) for w in invalid_waterways))}. "
                f"Supported: {settings.SUPPORTED_WATERWAYS}"
            ),
        )

    n_segments = len(request.segments)

    # --- Asynchronous path (large batches) ---
    if n_segments > request.async_threshold:
        task_id = str(uuid.uuid4())
        task_record: dict[str, Any] = {
            "task_id": task_id,
            "status": "PENDING",
            "progress_pct": 0.0,
            "message": f"Batch of {n_segments} segments queued for processing.",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
            "result_url": f"/api/v1/tasks/{task_id}",
        }
        _batch_tasks[task_id] = task_record

        # Schedule processing as a background task
        background_tasks.add_task(
            _process_batch_background,
            task_id=task_id,
            request=request,
            model_service=model_service,
        )

        logger.info(
            "Batch prediction task %s submitted: %d segments.", task_id, n_segments
        )
        return JSONResponse(  # type: ignore[return-value]
            status_code=status.HTTP_202_ACCEPTED,
            content=task_record,
        )

    # --- Synchronous path (small batches) ---
    segment_dicts: list[dict[str, Any]] = []
    for seg in request.segments:
        seg_dict: dict[str, Any] = {
            "segment_id": seg.segment_id,
            "waterway_id": seg.waterway_id,
            "month": seg.month,
            "year": seg.year,
            "geometry": seg.geometry
            or {
                "type": "Point",
                "coordinates": [85.0, 24.0],
            },
        }
        if seg.features is not None:
            seg_dict.update(seg.features.model_dump(exclude_none=True))
        segment_dicts.append(seg_dict)

    try:
        predictions = await model_service.predict_batch(
            segments=segment_dicts,
            patches_list=None,
            compute_shap=request.return_shap,
            force_refresh=request.force_refresh,
        )

        if not request.return_features:
            predictions = [
                p.model_copy(update={"features": p.features.__class__()})
                for p in predictions
            ]

        logger.info(
            "Synchronous batch prediction complete: %d segments.", len(predictions)
        )
        return predictions

    except Exception as exc:
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Batch prediction failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# GET /navigability/tasks/{task_id}  (batch job polling)
# ---------------------------------------------------------------------------


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatus,
    summary="Poll Batch Task Status",
    description=(
        "Poll the status and progress of a large batch prediction task "
        "that was submitted asynchronously. Returns `TaskStatus` with "
        "progress percentage and, when complete, a URL to retrieve results."
    ),
)
async def get_task_status(
    task_id: str = Path(
        ...,
        title="Task ID",
        description="UUID returned by the batch prediction endpoint.",
    ),
) -> TaskStatus:
    """
    Poll the status of an asynchronous batch prediction task.

    - Returns `PENDING` while the task is queued.
    - Returns `PROGRESS` while inference is running (with `progress_pct`).
    - Returns `SUCCESS` when complete (check `result_url` for results).
    - Returns `FAILURE` if the task encountered an error.
    """
    record = _batch_tasks.get(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found. It may have expired or the ID is incorrect.",
        )
    return TaskStatus(**record)


# ---------------------------------------------------------------------------
# Background task helper
# ---------------------------------------------------------------------------


async def _process_batch_background(
    task_id: str,
    request: BatchPredictionRequest,
    model_service: ModelService,
) -> None:
    """
    Execute a large batch prediction in the background.

    Updates the in-memory task registry with progress and final results.
    In production, replace this with a Celery task for multi-process
    distribution and persistent result storage.
    """
    task = _batch_tasks.get(task_id)
    if task is None:
        return

    task["status"] = "STARTED"
    task["message"] = "Inference pipeline initialised."
    task["progress_pct"] = 5.0

    try:
        n = len(request.segments)
        batch_size = settings.INFERENCE_BATCH_SIZE
        all_predictions: list[NavigabilityPrediction] = []

        for i in range(0, n, batch_size):
            chunk = request.segments[i : i + batch_size]
            chunk_dicts: list[dict[str, Any]] = []
            for seg in chunk:
                seg_dict: dict[str, Any] = {
                    "segment_id": seg.segment_id,
                    "waterway_id": seg.waterway_id,
                    "month": seg.month,
                    "year": seg.year,
                    "geometry": seg.geometry
                    or {
                        "type": "Point",
                        "coordinates": [85.0, 24.0],
                    },
                }
                if seg.features is not None:
                    seg_dict.update(seg.features.model_dump(exclude_none=True))
                chunk_dicts.append(seg_dict)

            chunk_preds = await model_service.predict_batch(
                segments=chunk_dicts,
                patches_list=None,
                compute_shap=request.return_shap,
                force_refresh=request.force_refresh,
            )
            all_predictions.extend(chunk_preds)

            progress = min(95.0, 5.0 + 90.0 * (i + len(chunk)) / n)
            task["status"] = "PROGRESS"
            task["progress_pct"] = round(progress, 1)
            task["message"] = (
                f"Processed {i + len(chunk)}/{n} segments ({progress:.0f}% complete)."
            )

            # Yield control briefly between chunks
            await asyncio.sleep(0)

        # Store results (simplified: store count; in production store in Redis/S3)
        task["status"] = "SUCCESS"
        task["progress_pct"] = 100.0
        task["message"] = (
            f"Batch complete — {len(all_predictions)} predictions generated."
        )
        task["completed_at"] = datetime.now(timezone.utc).isoformat()
        task["result_count"] = len(all_predictions)
        logger.info(
            "Background batch task %s completed: %d predictions.",
            task_id,
            len(all_predictions),
        )

    except Exception as exc:
        task["status"] = "FAILURE"
        task["error"] = str(exc)
        task["message"] = f"Batch prediction failed: {exc}"
        task["completed_at"] = datetime.now(timezone.utc).isoformat()
        logger.exception("Background batch task %s failed: %s", task_id, exc)
