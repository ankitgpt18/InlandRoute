"""
AIDSTL Project — River Segments API Routes
==========================================
REST endpoints for river segment data, historical navigability records,
and spectral feature retrieval.

Endpoints
---------
  GET /api/v1/segments/{waterway_id}                     — all segments with latest navigability
  GET /api/v1/segments/{segment_id}/history              — historical navigability data
  GET /api/v1/segments/{segment_id}/features             — spectral features for a month/year
  GET /api/v1/segments/{segment_id}/profile              — single segment full profile
  GET /api/v1/segments/{waterway_id}/geojson             — GeoJSON FeatureCollection export

Study areas
-----------
  NW-1 : Ganga       — Varanasi → Haldia   (~1,620 km, ~324 × 5-km segments)
  NW-2 : Brahmaputra — Dhubri  → Sadiya    (~891 km,  ~178 × 5-km segments)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from app.core.config import get_settings
from app.models.schemas.navigability import (
    NavigabilityClass,
    NavigabilityPrediction,
    RiverSegmentResponse,
    SpectralFeatures,
    WaterwayID,
)
from app.services.gee_service import GEEService, get_gee_service
from app.services.model_service import ModelService
from app.services.navigability_service import (
    NavigabilityService,
    _generate_synthetic_segments,
)
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/segments",
    tags=["Segments"],
    responses={
        404: {"description": "Segment or waterway not found"},
        422: {"description": "Validation error — check request parameters"},
        503: {"description": "Service unavailable"},
    },
)

# ---------------------------------------------------------------------------
# Response schemas (route-specific)
# ---------------------------------------------------------------------------


class SegmentSummary(BaseModel):
    """Lightweight segment summary returned in list endpoints."""

    segment_id: str
    waterway_id: str
    segment_index: int
    chainage_start_km: float
    chainage_end_km: float
    length_km: float
    sinuosity: float
    geometry: dict[str, Any]
    centroid_lon: float
    centroid_lat: float

    # Latest navigability prediction (populated when available)
    latest_depth_m: Optional[float] = None
    latest_width_m: Optional[float] = None
    latest_navigability_class: Optional[str] = None
    latest_risk_score: Optional[float] = None
    latest_confidence: Optional[float] = None
    prediction_month: Optional[int] = None
    prediction_year: Optional[int] = None


class SegmentListResponse(BaseModel):
    """Response for the full segment list of a waterway."""

    waterway_id: str
    total_segments: int
    month: int
    year: int
    segments: list[SegmentSummary]
    generated_at: str


class HistoricalRecord(BaseModel):
    """A single historical navigability record for a segment."""

    segment_id: str
    year: int
    month: int
    month_name: str
    predicted_depth_m: float
    depth_lower_ci: float
    depth_upper_ci: float
    width_m: float
    navigability_class: str
    navigability_probability: float
    risk_score: float
    confidence: float
    model_version: str


class SegmentHistoryResponse(BaseModel):
    """Historical navigability records for a single segment."""

    segment_id: str
    waterway_id: str
    years_requested: int
    total_records: int
    records: list[HistoricalRecord]
    mean_depth_m: Optional[float] = None
    mean_risk_score: Optional[float] = None
    trend_direction: str = "stable"  # "improving" | "stable" | "deteriorating"
    generated_at: str


class SegmentFeaturesResponse(BaseModel):
    """Spectral and derived features for a single segment."""

    segment_id: str
    waterway_id: str
    month: int
    year: int
    source: str = "gee"  # "gee" | "cached" | "synthetic"
    features: SpectralFeatures
    raw_bands: dict[str, Optional[float]]
    derived_indices: dict[str, Optional[float]]
    scene_count: int = 0
    cloud_cover_pct: Optional[float] = None
    generated_at: str


class SegmentGeoJSONResponse(BaseModel):
    """GeoJSON FeatureCollection of all segments in a waterway."""

    type: str = "FeatureCollection"
    features: list[dict[str, Any]]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_nav_service: Optional[NavigabilityService] = None


def get_nav_service() -> NavigabilityService:
    global _nav_service
    if _nav_service is None:
        _nav_service = NavigabilityService()
    return _nav_service


async def get_model_service() -> ModelService:
    return await ModelService.get_instance()


def get_gee() -> GEEService:
    return get_gee_service()


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

SegmentPathParam = Annotated[
    str,
    Path(
        title="Segment ID",
        description="Unique segment identifier, e.g. 'NW-1-042' or 'NW-2-107'.",
        pattern=r"^NW-[12]-\d{3,4}$",
        examples=["NW-1-042", "NW-2-107"],
    ),
]

MonthQueryParam = Annotated[
    int,
    Query(
        ge=1,
        le=12,
        title="Month",
        description="Calendar month (1–12). Defaults to current month.",
        examples=[6],
    ),
]

YearQueryParam = Annotated[
    int,
    Query(
        ge=2015,
        le=2100,
        title="Year",
        description="Calendar year ≥ 2015. Defaults to current year.",
        examples=[2024],
    ),
]

# ---------------------------------------------------------------------------
# Month name lookup
# ---------------------------------------------------------------------------

_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_month_year() -> tuple[int, int]:
    now = datetime.now(timezone.utc)
    return now.month, now.year


def _validate_waterway(waterway_id: str) -> str:
    if waterway_id not in settings.SUPPORTED_WATERWAYS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Waterway '{waterway_id}' is not supported. "
                f"Supported waterways: {settings.SUPPORTED_WATERWAYS}"
            ),
        )
    return waterway_id


def _parse_segment_id(segment_id: str) -> tuple[str, int]:
    """Split 'NW-1-042' → ('NW-1', 42). Raises 422 on invalid format."""
    parts = segment_id.rsplit("-", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid segment ID '{segment_id}'. "
                "Expected format: 'NW-X-NNN' (e.g. 'NW-1-042')."
            ),
        )
    waterway_id = parts[0]
    if waterway_id not in settings.SUPPORTED_WATERWAYS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Waterway '{waterway_id}' embedded in segment ID '{segment_id}' "
                f"is not supported. Supported: {settings.SUPPORTED_WATERWAYS}"
            ),
        )
    return waterway_id, int(parts[1])


def _segment_dict_to_summary(
    seg: dict[str, Any],
    prediction: Optional[NavigabilityPrediction] = None,
) -> SegmentSummary:
    """Convert a synthetic segment dict (+ optional prediction) to SegmentSummary."""
    geom = seg.get("geometry", {"type": "LineString", "coordinates": []})

    # Derive centroid from first coordinate of the LineString
    coords = geom.get("coordinates", [])
    if coords:
        centroid_lon = float(coords[0][0]) if len(coords) > 0 else 0.0
        centroid_lat = float(coords[0][1]) if len(coords) > 0 else 0.0
    else:
        centroid_lon = 0.0
        centroid_lat = 0.0

    summary = SegmentSummary(
        segment_id=seg["segment_id"],
        waterway_id=seg["waterway_id"],
        segment_index=int(seg.get("segment_index", 0)),
        chainage_start_km=float(seg.get("chainage_start_km", 0.0)),
        chainage_end_km=float(seg.get("chainage_end_km", 5.0)),
        length_km=float(seg.get("length_km", 5.0)),
        sinuosity=float(seg.get("sinuosity", 1.0)),
        geometry=geom,
        centroid_lon=round(centroid_lon, 6),
        centroid_lat=round(centroid_lat, 6),
    )

    if prediction is not None:
        summary.latest_depth_m = prediction.predicted_depth_m
        summary.latest_width_m = prediction.width_m
        summary.latest_navigability_class = str(prediction.navigability_class)
        summary.latest_risk_score = prediction.risk_score
        summary.latest_confidence = prediction.confidence
        summary.prediction_month = prediction.month
        summary.prediction_year = prediction.year

    return summary


# ---------------------------------------------------------------------------
# GET /segments/{waterway_id}  — all segments with latest navigability
# ---------------------------------------------------------------------------


@router.get(
    "/{waterway_id}",
    response_model=SegmentListResponse,
    summary="List All Segments with Latest Navigability",
    description=(
        "Return all 5-km river segments for a waterway together with their "
        "most recent navigability prediction. Segments are ordered upstream "
        "to downstream (by segment index / chainage).\n\n"
        "Pass `month` and `year` to retrieve predictions for a specific period; "
        "otherwise the current month/year is used."
    ),
)
async def list_segments(
    waterway_id: WaterwayPathParam,
    month: MonthQueryParam = None,  # type: ignore[assignment]
    year: YearQueryParam = None,  # type: ignore[assignment]
    navigability_class: Optional[str] = Query(
        None,
        description=(
            "Filter segments by navigability class. "
            "One of: 'navigable', 'conditional', 'non_navigable'."
        ),
        pattern="^(navigable|conditional|non_navigable)$",
    ),
    min_risk_score: float = Query(
        0.0,
        ge=0.0,
        le=1.0,
        description="Return only segments with risk_score ≥ this value.",
    ),
    limit: int = Query(
        500,
        ge=1,
        le=1000,
        description="Maximum number of segments to return.",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of segments to skip (for pagination).",
    ),
    nav_service: NavigabilityService = Depends(get_nav_service),
) -> SegmentListResponse:
    """
    List all river segments for a waterway with their current navigability status.

    - **waterway_id**: `NW-1` (Ganga) or `NW-2` (Brahmaputra)
    - **month** / **year**: temporal context for the prediction (defaults to current)
    - **navigability_class**: optional filter — `navigable | conditional | non_navigable`
    - **min_risk_score**: optional risk-score lower bound filter
    - **limit** / **offset**: pagination controls
    """
    _validate_waterway(waterway_id)

    if month is None or year is None:
        cur_month, cur_year = _current_month_year()
        month = month or cur_month
        year = year or cur_year

    try:
        # Fetch the full navigability map (cached after first call)
        nav_map = await nav_service.get_navigability_map(
            waterway_id=waterway_id,
            month=month,
            year=year,
        )
    except Exception as exc:
        logger.exception(
            "Failed to fetch nav map for segment listing (%s %02d/%d): %s",
            waterway_id,
            month,
            year,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not retrieve navigability data: {exc}",
        ) from exc

    # Build a lookup: segment_id → NavigabilityPrediction
    pred_by_id: dict[str, NavigabilityPrediction] = {
        p.segment_id: p for p in nav_map.predictions
    }

    # Generate synthetic segment metadata (geometry, chainage etc.)
    synthetic_segs = _generate_synthetic_segments(waterway_id, month, year)
    synthetic_by_id: dict[str, dict[str, Any]] = {
        s["segment_id"]: s for s in synthetic_segs
    }

    # Build summaries
    summaries: list[SegmentSummary] = []
    for seg_id, pred in pred_by_id.items():
        seg_dict = synthetic_by_id.get(
            seg_id,
            {
                "segment_id": seg_id,
                "waterway_id": waterway_id,
                "segment_index": 0,
                "chainage_start_km": 0.0,
                "chainage_end_km": 5.0,
                "length_km": 5.0,
                "sinuosity": 1.0,
                "geometry": pred.geometry,
            },
        )
        summary = _segment_dict_to_summary(seg_dict, pred)
        summaries.append(summary)

    # Sort by chainage (upstream → downstream)
    summaries.sort(key=lambda s: s.chainage_start_km)

    # Apply filters
    if navigability_class:
        summaries = [
            s for s in summaries if s.latest_navigability_class == navigability_class
        ]
    if min_risk_score > 0.0:
        summaries = [
            s for s in summaries if (s.latest_risk_score or 0.0) >= min_risk_score
        ]

    total = len(summaries)

    # Paginate
    summaries = summaries[offset : offset + limit]

    return SegmentListResponse(
        waterway_id=waterway_id,
        total_segments=total,
        month=month,
        year=year,
        segments=summaries,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /segments/{segment_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/{segment_id}/history",
    response_model=SegmentHistoryResponse,
    summary="Get Segment Historical Navigability",
    description=(
        "Retrieve historical navigability predictions for a single river "
        "segment over a multi-year window. Returns monthly records ordered "
        "chronologically (oldest first) together with trend statistics.\n\n"
        "The `years` parameter controls how many years of history to retrieve "
        "(default 3, maximum 10)."
    ),
)
async def get_segment_history(
    segment_id: SegmentPathParam,
    years: int = Query(
        3,
        ge=1,
        le=10,
        description="Number of years of historical data to retrieve.",
    ),
    months: Optional[str] = Query(
        None,
        description=(
            "Comma-separated list of months to include (e.g. '6,7,8,9' for monsoon). "
            "Omit to include all 12 months."
        ),
    ),
    nav_service: NavigabilityService = Depends(get_nav_service),
) -> SegmentHistoryResponse:
    """
    Retrieve multi-year historical navigability data for a single segment.

    - **segment_id**: e.g. `NW-1-042`
    - **years**: number of historical years (1–10, default 3)
    - **months**: optional comma-separated list of months to include
    """
    waterway_id, seg_index = _parse_segment_id(segment_id)

    # Parse months filter
    month_filter: Optional[list[int]] = None
    if months:
        try:
            month_filter = [int(m.strip()) for m in months.split(",")]
            invalid = [m for m in month_filter if not 1 <= m <= 12]
            if invalid:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid month values: {invalid}. Must be 1–12.",
                )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not parse months parameter: {exc}",
            ) from exc

    now = datetime.now(timezone.utc)
    end_year = now.year
    start_year = max(2015, end_year - years)

    records: list[HistoricalRecord] = []
    all_depths: list[float] = []
    all_risks: list[float] = []

    # Iterate over years and months
    for yr in range(start_year, end_year + 1):
        months_to_fetch = month_filter or list(range(1, 13))

        for mo in months_to_fetch:
            # Skip future months
            if yr == now.year and mo > now.month:
                continue

            try:
                nav_map = await nav_service.get_navigability_map(
                    waterway_id=waterway_id,
                    month=mo,
                    year=yr,
                    force_refresh=False,
                )
            except Exception as exc:
                logger.debug(
                    "Could not fetch historical map for %s %s %02d/%d: %s",
                    segment_id,
                    waterway_id,
                    mo,
                    yr,
                    exc,
                )
                continue

            # Find the prediction for our segment
            seg_pred: Optional[NavigabilityPrediction] = next(
                (p for p in nav_map.predictions if p.segment_id == segment_id),
                None,
            )
            if seg_pred is None:
                continue

            records.append(
                HistoricalRecord(
                    segment_id=segment_id,
                    year=yr,
                    month=mo,
                    month_name=_MONTH_NAMES[mo - 1],
                    predicted_depth_m=seg_pred.predicted_depth_m,
                    depth_lower_ci=seg_pred.depth_lower_ci,
                    depth_upper_ci=seg_pred.depth_upper_ci,
                    width_m=seg_pred.width_m,
                    navigability_class=str(seg_pred.navigability_class),
                    navigability_probability=seg_pred.navigability_probability,
                    risk_score=seg_pred.risk_score,
                    confidence=seg_pred.confidence,
                    model_version=seg_pred.model_version,
                )
            )
            all_depths.append(seg_pred.predicted_depth_m)
            all_risks.append(seg_pred.risk_score)

    # Sort chronologically
    records.sort(key=lambda r: (r.year, r.month))

    # Compute trend direction using linear regression on risk scores
    trend_direction = "stable"
    import numpy as np  # local import — numpy is always available

    if len(all_risks) >= 3:
        xs = np.arange(len(all_risks), dtype=float)
        ys = np.array(all_risks, dtype=float)
        slope = float(np.polyfit(xs, ys, 1)[0])
        if slope > 0.02:
            trend_direction = "deteriorating"
        elif slope < -0.02:
            trend_direction = "improving"

    return SegmentHistoryResponse(
        segment_id=segment_id,
        waterway_id=waterway_id,
        years_requested=years,
        total_records=len(records),
        records=records,
        mean_depth_m=round(float(np.mean(all_depths)), 3) if all_depths else None,
        mean_risk_score=round(float(np.mean(all_risks)), 4) if all_risks else None,
        trend_direction=trend_direction,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /segments/{segment_id}/features
# ---------------------------------------------------------------------------


@router.get(
    "/{segment_id}/features",
    response_model=SegmentFeaturesResponse,
    summary="Get Segment Spectral Features",
    description=(
        "Retrieve Sentinel-2 derived spectral features (band reflectances, "
        "water indices, bathymetric ratios) for a single river segment at "
        "a given month and year.\n\n"
        "Features are extracted from a cloud-masked monthly median composite "
        "via Google Earth Engine. When GEE is unavailable or running in mock "
        "mode, synthetic features are returned instead."
    ),
)
async def get_segment_features(
    segment_id: SegmentPathParam,
    month: MonthQueryParam = None,  # type: ignore[assignment]
    year: YearQueryParam = None,  # type: ignore[assignment]
    buffer_km: float = Query(
        2.0,
        ge=0.5,
        le=10.0,
        description="Buffer radius (km) around the segment centreline for the GEE extraction AOI.",
    ),
    gee_service: GEEService = Depends(get_gee),
) -> SegmentFeaturesResponse:
    """
    Retrieve Sentinel-2 spectral features for a single river segment.

    - **segment_id**: e.g. `NW-1-042`
    - **month** / **year**: temporal context (defaults to current)
    - **buffer_km**: AOI buffer radius around the centreline (default 2 km)
    """
    waterway_id, _ = _parse_segment_id(segment_id)

    if month is None or year is None:
        cur_month, cur_year = _current_month_year()
        month = month or cur_month
        year = year or cur_year

    # Determine geometry for the segment from synthetic data
    synthetic_segs = _generate_synthetic_segments(waterway_id, month, year)
    seg_data = next((s for s in synthetic_segs if s["segment_id"] == segment_id), None)
    geometry: Optional[dict[str, Any]] = seg_data.get("geometry") if seg_data else None

    # Fetch features from GEE (or mock)
    source = "gee"
    try:
        raw_features = await gee_service.extract_segment_features(
            segment_id=segment_id,
            month=month,
            year=year,
            geometry=geometry,
            buffer_km=buffer_km,
        )
        if getattr(gee_service, "_mock_mode", False):
            source = "synthetic"
    except Exception as exc:
        logger.warning(
            "GEE feature extraction failed for %s %02d/%d (%s). "
            "Falling back to synthetic features.",
            segment_id,
            month,
            year,
            exc,
        )
        source = "synthetic"
        # Produce synthetic fallback features
        seed_str = f"{segment_id}-{month}-{year}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
        import numpy as np

        rng = np.random.default_rng(seed)
        monsoon_boost = 0.2 if month in (6, 7, 8, 9) else 0.0
        raw_features = {
            "blue": float(rng.uniform(0.03, 0.08)),
            "green": float(rng.uniform(0.05, 0.15)),
            "red": float(rng.uniform(0.04, 0.12)),
            "red_edge_1": float(rng.uniform(0.06, 0.14)),
            "red_edge_2": float(rng.uniform(0.08, 0.18)),
            "red_edge_3": float(rng.uniform(0.10, 0.22)),
            "nir": float(rng.uniform(0.12, 0.30)),
            "nir_narrow": float(rng.uniform(0.11, 0.28)),
            "swir1": float(rng.uniform(0.04, 0.10)),
            "swir2": float(rng.uniform(0.02, 0.07)),
            "mndwi": float(min(1.0, max(-1.0, rng.uniform(0.1, 0.6) + monsoon_boost))),
            "ndwi": float(min(1.0, max(-1.0, rng.uniform(0.0, 0.5) + monsoon_boost))),
            "ndvi": float(rng.uniform(-0.1, 0.4)),
            "awei_nsh": float(rng.uniform(-0.2, 0.8) + monsoon_boost),
            "ndti": float(min(1.0, max(-1.0, rng.uniform(-0.1, 0.3)))),
            "stumpf_bg": float(rng.uniform(0.9, 1.2)),
        }

    # Get scene count from GEE (non-blocking best-effort)
    scene_count = 0
    try:
        scene_count = await gee_service.get_scene_count(
            waterway_id=waterway_id, month=month, year=year
        )
    except Exception:
        pass

    # Separate raw band reflectances from derived indices
    band_keys = {
        "blue",
        "green",
        "red",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
        "nir",
        "nir_narrow",
        "swir1",
        "swir2",
    }
    raw_bands: dict[str, Optional[float]] = {k: raw_features.get(k) for k in band_keys}
    derived_indices: dict[str, Optional[float]] = {
        k: v for k, v in raw_features.items() if k not in band_keys
    }

    # Build SpectralFeatures schema
    spectral = SpectralFeatures(
        mndwi=raw_features.get("mndwi"),
        ndwi=raw_features.get("ndwi"),
        awei_sh=raw_features.get("awei_sh"),
        awei_ns=raw_features.get("awei_nsh"),
        stumpf_ratio=raw_features.get("stumpf_bg"),
        turbidity_index=raw_features.get("ndti"),
        b2_blue=raw_features.get("blue"),
        b3_green=raw_features.get("green"),
        b4_red=raw_features.get("red"),
        b8_nir=raw_features.get("nir"),
        b11_swir1=raw_features.get("swir1"),
        b12_swir2=raw_features.get("swir2"),
        ndvi=raw_features.get("ndvi"),
        water_pixel_fraction=raw_features.get("water_fraction"),
        ndwi_trend_3m=raw_features.get("ndwi_trend_3m"),
        ndwi_anomaly=raw_features.get("ndwi_anomaly"),
    )

    return SegmentFeaturesResponse(
        segment_id=segment_id,
        waterway_id=waterway_id,
        month=month,
        year=year,
        source=source,
        features=spectral,
        raw_bands=raw_bands,
        derived_indices=derived_indices,
        scene_count=scene_count,
        cloud_cover_pct=raw_features.get("cloud_cover_pct"),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /segments/{segment_id}/profile
# ---------------------------------------------------------------------------


@router.get(
    "/{segment_id}/profile",
    response_model=NavigabilityPrediction,
    summary="Get Full Segment Prediction Profile",
    description=(
        "Return the complete navigability prediction profile for a single "
        "river segment at a given month and year, including all spectral "
        "features, class probabilities, confidence intervals, and "
        "optionally SHAP feature-importance values."
    ),
)
async def get_segment_profile(
    segment_id: SegmentPathParam,
    month: MonthQueryParam = None,  # type: ignore[assignment]
    year: YearQueryParam = None,  # type: ignore[assignment]
    return_shap: bool = Query(
        False,
        description="Include SHAP feature-importance values in the response.",
    ),
    force_refresh: bool = Query(
        False,
        description="Bypass the Redis prediction cache and recompute.",
    ),
    model_service: ModelService = Depends(get_model_service),
) -> NavigabilityPrediction:
    """
    Retrieve a complete navigability prediction for one river segment.

    - **segment_id**: e.g. `NW-1-042`
    - **month** / **year**: temporal context (defaults to current)
    - **return_shap**: include SHAP values for explainability
    - **force_refresh**: bypass the 6-hour prediction cache
    """
    waterway_id, seg_index = _parse_segment_id(segment_id)

    if month is None or year is None:
        cur_month, cur_year = _current_month_year()
        month = month or cur_month
        year = year or cur_year

    # Build segment features dict from synthetic data
    synthetic_segs = _generate_synthetic_segments(waterway_id, month, year)
    seg_data = next((s for s in synthetic_segs if s["segment_id"] == segment_id), None)

    if seg_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Segment '{segment_id}' not found in waterway '{waterway_id}'. "
                f"Segment index {seg_index} may be out of range for this waterway."
            ),
        )

    try:
        prediction = await model_service.predict_segment(
            segment_features=seg_data,
            patches=None,
            compute_shap=return_shap,
            force_refresh=force_refresh,
        )
        return prediction
    except Exception as exc:
        logger.exception(
            "Failed to compute prediction for segment %s %02d/%d: %s",
            segment_id,
            month,
            year,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction failed for segment '{segment_id}': {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# GET /segments/{waterway_id}/geojson
# ---------------------------------------------------------------------------


@router.get(
    "/{waterway_id}/geojson",
    summary="Export Segments as GeoJSON",
    description=(
        "Export all river segments for a waterway as a GeoJSON FeatureCollection. "
        "Each feature includes the segment geometry (LineString) and navigability "
        "prediction attributes as GeoJSON properties.\n\n"
        "Useful for loading directly into QGIS, Mapbox, or Leaflet."
    ),
    responses={
        200: {
            "content": {"application/geo+json": {}},
            "description": "GeoJSON FeatureCollection of river segments.",
        }
    },
)
async def export_segments_geojson(
    waterway_id: WaterwayPathParam,
    month: MonthQueryParam = None,  # type: ignore[assignment]
    year: YearQueryParam = None,  # type: ignore[assignment]
    nav_service: NavigabilityService = Depends(get_nav_service),
) -> JSONResponse:
    """
    Export all segments as a GeoJSON FeatureCollection.

    Each GeoJSON feature has properties:
    - `segment_id`, `waterway_id`, `chainage_start_km`, `chainage_end_km`
    - `navigability_class`, `predicted_depth_m`, `width_m`, `risk_score`
    - `confidence`, `navigability_probability`
    - Colour coding: `fill_color` (hex) based on navigability class
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
        )
    except Exception as exc:
        logger.exception("GeoJSON export failed for %s: %s", waterway_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not retrieve navigability data: {exc}",
        ) from exc

    # Navigability class → hex colour mapping (for frontend visualisation)
    _class_colours: dict[str, str] = {
        "navigable": "#22c55e",  # green-500
        "conditional": "#f59e0b",  # amber-500
        "non_navigable": "#ef4444",  # red-500
    }

    # Build synthetic segment metadata for chainage info
    synthetic_segs = _generate_synthetic_segments(waterway_id, month, year)
    synthetic_by_id: dict[str, dict[str, Any]] = {
        s["segment_id"]: s for s in synthetic_segs
    }

    features: list[dict[str, Any]] = []
    for pred in sorted(nav_map.predictions, key=lambda p: p.segment_id):
        seg = synthetic_by_id.get(pred.segment_id, {})
        nav_class_str = str(pred.navigability_class)

        feature: dict[str, Any] = {
            "type": "Feature",
            "geometry": pred.geometry,
            "properties": {
                "segment_id": pred.segment_id,
                "waterway_id": str(pred.waterway_id),
                "chainage_start_km": seg.get("chainage_start_km", 0.0),
                "chainage_end_km": seg.get("chainage_end_km", 5.0),
                "length_km": seg.get("length_km", 5.0),
                "month": pred.month,
                "year": pred.year,
                "predicted_depth_m": pred.predicted_depth_m,
                "depth_lower_ci": pred.depth_lower_ci,
                "depth_upper_ci": pred.depth_upper_ci,
                "width_m": pred.width_m,
                "navigability_class": nav_class_str,
                "navigability_probability": pred.navigability_probability,
                "risk_score": pred.risk_score,
                "confidence": pred.confidence,
                "fill_color": _class_colours.get(nav_class_str, "#94a3b8"),
                "fill_opacity": 0.7,
                "model_version": pred.model_version,
            },
        }
        features.append(feature)

    geojson_body: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "waterway_id": waterway_id,
            "month": month,
            "year": year,
            "total_segments": nav_map.total_segments,
            "navigable_count": nav_map.navigable_count,
            "conditional_count": nav_map.conditional_count,
            "non_navigable_count": nav_map.non_navigable_count,
            "overall_navigability_pct": nav_map.overall_navigability_pct,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        },
    }

    return JSONResponse(
        content=geojson_body,
        media_type="application/geo+json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="aidstl_{waterway_id}_{year}{month:02d}.geojson"'
            ),
            "Cache-Control": "public, max-age=21600",  # 6 h
        },
    )


# ---------------------------------------------------------------------------
# GET /segments/{waterway_id}/summary-stats
# ---------------------------------------------------------------------------


@router.get(
    "/{waterway_id}/summary-stats",
    summary="Get Quick Segment Count Statistics",
    description=(
        "Return lightweight summary statistics about segment navigability "
        "for a waterway/month/year — counts and percentages only, without "
        "the full prediction list. Ideal for dashboard KPI cards."
    ),
)
async def get_segment_summary_stats(
    waterway_id: WaterwayPathParam,
    month: MonthQueryParam = None,  # type: ignore[assignment]
    year: YearQueryParam = None,  # type: ignore[assignment]
    nav_service: NavigabilityService = Depends(get_nav_service),
) -> JSONResponse:
    """
    Return quick-access segment navigability statistics.

    Returns a compact JSON object with counts and percentages — no individual
    segment data is included.  Suitable for populating dashboard metrics.
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
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not retrieve navigability data: {exc}",
        ) from exc

    total = max(nav_map.total_segments, 1)

    return JSONResponse(
        content={
            "waterway_id": waterway_id,
            "month": month,
            "year": year,
            "month_name": _MONTH_NAMES[month - 1],
            "total_segments": nav_map.total_segments,
            "navigable": {
                "count": nav_map.navigable_count,
                "pct": round(100.0 * nav_map.navigable_count / total, 2),
                "length_km": nav_map.navigable_length_km,
            },
            "conditional": {
                "count": nav_map.conditional_count,
                "pct": round(100.0 * nav_map.conditional_count / total, 2),
                "length_km": nav_map.conditional_length_km,
            },
            "non_navigable": {
                "count": nav_map.non_navigable_count,
                "pct": round(100.0 * nav_map.non_navigable_count / total, 2),
                "length_km": nav_map.non_navigable_length_km,
            },
            "mean_depth_m": nav_map.mean_depth_m,
            "mean_width_m": nav_map.mean_width_m,
            "mean_risk_score": nav_map.mean_risk_score,
            "overall_navigability_pct": nav_map.overall_navigability_pct,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
