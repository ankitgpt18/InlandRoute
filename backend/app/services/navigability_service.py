"""
AIDSTL Project — Navigability Business Logic Service
=====================================================
Core orchestration layer that combines GEE feature extraction, ML model
inference, spatial analysis, and caching to produce navigability products.

Responsibilities
----------------
  get_navigability_map      — full segment-level map for a waterway/month/year
  get_seasonal_calendar     — 12-month outlook for every segment
  get_risk_alerts           — segments exceeding the risk threshold
  get_depth_profile         — longitudinal depth profile
  get_waterway_stats        — annual statistics summary
  compare_with_historical   — anomaly vs. multi-year baseline

All public methods are async.  Heavy computation (model inference, GEE
calls) is delegated to ModelService and GEEService respectively and runs
in thread-pool executors so the FastAPI event loop is never blocked.

Caching
-------
Results are cached in Redis with configurable TTLs:
  Navigability map   → 6 h  (CACHE_TTL_SECONDS)
  Seasonal calendar  → 24 h
  Depth profile      → 6 h
  Waterway stats     → 12 h
  Historical compare → 24 h
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import redis.asyncio as aioredis
from app.core.config import get_settings
from app.models.schemas.navigability import (
    AlertSeverity,
    AlertType,
    DepthProfile,
    DepthProfilePoint,
    HistoricalComparison,
    HistoricalDataPoint,
    MonthlyOutlook,
    MonthlyStats,
    NavigabilityClass,
    NavigabilityMap,
    NavigabilityPrediction,
    RiskAlert,
    Season,
    SeasonalCalendar,
    SegmentSeasonalOutlook,
    WaterwayID,
    WaterwayStats,
)
from app.services.gee_service import GEEService, get_gee_service
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Constants
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

# Approximate segment count per waterway (5 km segments)
_WATERWAY_SEGMENT_COUNTS: dict[str, int] = {
    "NW-1": 324,  # 1620 km / 5 km
    "NW-2": 178,  # 891 km / 5 km
}

# Approximate total lengths (km)
_WATERWAY_LENGTHS: dict[str, float] = {
    "NW-1": 1620.0,
    "NW-2": 891.0,
}

# Cache TTLs (seconds)
_TTL_MAP = 6 * 3600  # 6 h
_TTL_CALENDAR = 24 * 3600  # 24 h
_TTL_DEPTH = 6 * 3600  # 6 h
_TTL_STATS = 12 * 3600  # 12 h
_TTL_HISTORICAL = 24 * 3600  # 24 h

# Risk-score threshold above which an alert is generated
_DEFAULT_RISK_THRESHOLD: float = settings.RISK_ALERT_THRESHOLD

# Percentage of waterway that must be navigable to include a month in
# "best_navigation_months"
_BEST_MONTH_NAVIGABLE_PCT: float = 80.0


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(*parts: Any) -> str:
    """Build a deterministic, short Redis cache key from arbitrary parts."""
    raw = ":".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return f"aidstl:svc:{digest}"


class _CacheClient:
    """Thin async Redis wrapper used by NavigabilityService."""

    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._client: Optional[aioredis.Redis] = None  # type: ignore[type-arg]

    async def _get_client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        if self._client is None:
            self._client = aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                retry_on_timeout=True,
            )
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        try:
            client = await self._get_client()
            raw = await client.get(key)
            if raw is not None:
                return json.loads(raw)
        except Exception as exc:
            logger.debug("Cache GET miss/error (key=%s): %s", key, exc)
        return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        try:
            client = await self._get_client()
            await client.set(key, json.dumps(value, default=str), ex=ttl)
        except Exception as exc:
            logger.debug("Cache SET error (key=%s): %s", key, exc)

    async def delete(self, key: str) -> None:
        try:
            client = await self._get_client()
            await client.delete(key)
        except Exception as exc:
            logger.debug("Cache DELETE error (key=%s): %s", key, exc)

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None


# ---------------------------------------------------------------------------
# Synthetic segment generator
# ---------------------------------------------------------------------------


def _generate_synthetic_segments(
    waterway_id: str,
    month: int,
    year: int,
) -> list[dict[str, Any]]:
    """
    Generate plausible synthetic segment feature dicts for a full waterway.

    Used when the database has no persisted segments (development / demo
    mode).  Values are seeded deterministically so they are reproducible
    across repeated calls.
    """
    n_segs = _WATERWAY_SEGMENT_COUNTS.get(waterway_id, 100)
    total_km = _WATERWAY_LENGTHS.get(waterway_id, 500.0)
    seg_len_km = total_km / n_segs

    # Rough bounding box centre for centreline approximation
    bbox = settings.get_bbox_for_waterway(waterway_id)
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    monsoon = month in (6, 7, 8, 9)
    post_monsoon = month in (10, 11)
    pre_monsoon = month in (3, 4, 5)

    segments: list[dict[str, Any]] = []

    for idx in range(n_segs):
        seed_str = f"{waterway_id}-{idx}-{month}-{year}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)

        # Progress along the river (0 → 1)
        t = idx / max(n_segs - 1, 1)

        # Depth seasonality: monsoon deepest, pre-monsoon shallowest
        base_depth = rng.uniform(1.5, 6.0)
        if monsoon:
            depth_multiplier = rng.uniform(1.3, 1.8)
        elif post_monsoon:
            depth_multiplier = rng.uniform(1.1, 1.4)
        elif pre_monsoon:
            depth_multiplier = rng.uniform(0.7, 1.0)
        else:  # winter
            depth_multiplier = rng.uniform(0.8, 1.1)

        depth = float(base_depth * depth_multiplier)
        depth = round(max(0.1, depth), 2)

        # Width also varies seasonally
        base_width = rng.uniform(50.0, 600.0)
        width = float(base_width * (0.7 + 0.5 * depth_multiplier))
        width = round(max(5.0, width), 1)

        mndwi = float(
            np.clip(rng.uniform(0.1, 0.7) + (0.15 if monsoon else 0.0), -1, 1)
        )

        # Segment centreline geometry (simplified LineString along bbox)
        lon_start = min_lon + lon_range * t
        lon_end = min_lon + lon_range * min(t + seg_len_km / (lon_range * 111.0), 1.0)
        lat = min_lat + lat_range * (0.3 + 0.4 * rng.random())

        geometry = {
            "type": "LineString",
            "coordinates": [
                [round(lon_start, 6), round(lat, 6)],
                [round(lon_end, 6), round(lat + rng.uniform(-0.01, 0.01), 6)],
            ],
        }

        segments.append(
            {
                "segment_id": f"{waterway_id}-{idx:03d}",
                "waterway_id": waterway_id,
                "segment_index": idx,
                "month": month,
                "year": year,
                "geometry": geometry,
                "chainage_start_km": round(idx * seg_len_km, 3),
                "chainage_end_km": round((idx + 1) * seg_len_km, 3),
                "length_km": round(seg_len_km, 3),
                "sinuosity": float(rng.uniform(1.0, 1.8)),
                # Pre-computed spectral features (bypasses GEE in dev mode)
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
                "mndwi": mndwi,
                "ndwi": float(np.clip(mndwi - 0.05, -1, 1)),
                "ndvi": float(rng.uniform(-0.1, 0.4)),
                "turbidity_index": float(rng.uniform(0.0, 0.5)),
                "stumpf_ratio": float(rng.uniform(0.9, 1.2)),
                "water_pixel_fraction": float(
                    np.clip(mndwi + rng.uniform(-0.1, 0.1), 0, 1)
                ),
                "cloud_cover_pct": float(rng.uniform(0, 30)),
                # Override depth/width so the model uses them as context
                "_depth_override": depth,
                "_width_override": width,
            }
        )

    return segments


# ---------------------------------------------------------------------------
# Prediction post-processing helpers
# ---------------------------------------------------------------------------


def _map_month_to_season(month: int) -> Season:
    if month in (3, 4, 5):
        return Season.PRE_MONSOON
    if month in (6, 7, 8, 9):
        return Season.MONSOON
    if month in (10, 11):
        return Season.POST_MONSOON
    return Season.WINTER


def _summarise_predictions(
    predictions: list[NavigabilityPrediction],
    seg_len_km: float = 5.0,
) -> dict[str, Any]:
    """Compute aggregate statistics over a list of predictions."""
    if not predictions:
        return {
            "total_segments": 0,
            "navigable_count": 0,
            "conditional_count": 0,
            "non_navigable_count": 0,
            "navigable_length_km": 0.0,
            "conditional_length_km": 0.0,
            "non_navigable_length_km": 0.0,
            "mean_depth_m": None,
            "mean_width_m": None,
            "mean_risk_score": None,
            "overall_navigability_pct": 0.0,
        }

    navigable = [
        p for p in predictions if p.navigability_class == NavigabilityClass.NAVIGABLE
    ]
    conditional = [
        p for p in predictions if p.navigability_class == NavigabilityClass.CONDITIONAL
    ]
    non_nav = [
        p
        for p in predictions
        if p.navigability_class == NavigabilityClass.NON_NAVIGABLE
    ]

    total = len(predictions)
    nav_len = len(navigable) * seg_len_km
    cond_len = len(conditional) * seg_len_km
    non_nav_len = len(non_nav) * seg_len_km
    total_len = total * seg_len_km

    depths = [p.predicted_depth_m for p in predictions]
    widths = [p.width_m for p in predictions]
    risks = [p.risk_score for p in predictions]

    return {
        "total_segments": total,
        "navigable_count": len(navigable),
        "conditional_count": len(conditional),
        "non_navigable_count": len(non_nav),
        "navigable_length_km": round(nav_len, 2),
        "conditional_length_km": round(cond_len, 2),
        "non_navigable_length_km": round(non_nav_len, 2),
        "mean_depth_m": round(float(np.mean(depths)), 3) if depths else None,
        "mean_width_m": round(float(np.mean(widths)), 2) if widths else None,
        "mean_risk_score": round(float(np.mean(risks)), 4) if risks else None,
        "overall_navigability_pct": round(
            100.0 * nav_len / total_len if total_len > 0 else 0.0, 2
        ),
    }


# ---------------------------------------------------------------------------
# NavigabilityService
# ---------------------------------------------------------------------------


class NavigabilityService:
    """
    Core orchestration service for navigability prediction products.

    Wires together GEEService (feature extraction) and ModelService
    (ML inference) to produce the full set of navigability outputs
    exposed by the REST API.

    All public methods are async-safe.  Expensive computations are cached
    in Redis to avoid redundant GEE/inference calls within the same TTL
    window.

    Usage
    -----
    ::

        svc = NavigabilityService()
        nav_map = await svc.get_navigability_map("NW-1", month=6, year=2024)
    """

    def __init__(
        self,
        model_service: Optional[ModelService] = None,
        gee_service: Optional[GEEService] = None,
    ) -> None:
        self._model_service = model_service
        self._gee_service = gee_service or get_gee_service()
        self._cache = _CacheClient(settings.REDIS_URL)

    # ------------------------------------------------------------------
    # Lazy service accessors
    # ------------------------------------------------------------------

    async def _get_model_service(self) -> ModelService:
        if self._model_service is None:
            self._model_service = await ModelService.get_instance()
        return self._model_service

    # ------------------------------------------------------------------
    # Internal: segment prediction pipeline
    # ------------------------------------------------------------------

    async def _predict_all_segments(
        self,
        waterway_id: str,
        month: int,
        year: int,
        force_refresh: bool = False,
    ) -> list[NavigabilityPrediction]:
        """
        Fetch features for every segment in a waterway and run batch inference.

        Returns segments ordered upstream → downstream (by segment index).
        """
        # Generate synthetic segment metadata (replace with DB query in production)
        segment_dicts = _generate_synthetic_segments(waterway_id, month, year)

        model_svc = await self._get_model_service()

        # Inject GEE features when not running in synthetic/dev mode
        # In production: call await self._gee_service.extract_batch_segment_features(...)
        # and merge results into segment_dicts.

        predictions = await model_svc.predict_batch(
            segments=segment_dicts,
            patches_list=None,
            compute_shap=False,
            force_refresh=force_refresh,
        )

        # Sort by segment index to guarantee upstream → downstream ordering
        predictions.sort(key=lambda p: int(p.segment_id.split("-")[-1]))
        return predictions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_navigability_map(
        self,
        waterway_id: str,
        month: int,
        year: int,
        force_refresh: bool = False,
    ) -> NavigabilityMap:
        """
        Build the complete navigability map for a waterway at a given month/year.

        Parameters
        ----------
        waterway_id : str
            ``"NW-1"`` or ``"NW-2"``.
        month : int
            Calendar month (1–12).
        year : int
            Calendar year.
        force_refresh : bool
            Bypass Redis cache and recompute.

        Returns
        -------
        NavigabilityMap
            Full segment-level map with summary statistics.
        """
        cache_key = _cache_key("nav_map", waterway_id, month, year)

        if not force_refresh:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                logger.debug(
                    "Cache hit — nav_map %s %02d/%d.", waterway_id, month, year
                )
                return NavigabilityMap(**cached)

        logger.info(
            "Computing navigability map for %s %s/%d …",
            waterway_id,
            _MONTH_NAMES[month - 1],
            year,
        )

        predictions = await self._predict_all_segments(
            waterway_id, month, year, force_refresh
        )

        seg_len_km = _WATERWAY_LENGTHS.get(waterway_id, 500.0) / max(
            len(predictions), 1
        )
        summary = _summarise_predictions(predictions, seg_len_km)

        now = datetime.now(timezone.utc)
        cache_expires = now.replace(
            hour=(now.hour + _TTL_MAP // 3600) % 24, minute=0, second=0, microsecond=0
        )

        nav_map = NavigabilityMap(
            waterway_id=WaterwayID(waterway_id),
            month=month,
            year=year,
            predictions=predictions,
            total_segments=summary["total_segments"],
            navigable_count=summary["navigable_count"],
            conditional_count=summary["conditional_count"],
            non_navigable_count=summary["non_navigable_count"],
            navigable_length_km=summary["navigable_length_km"],
            conditional_length_km=summary["conditional_length_km"],
            non_navigable_length_km=summary["non_navigable_length_km"],
            mean_depth_m=summary["mean_depth_m"],
            mean_width_m=summary["mean_width_m"],
            mean_risk_score=summary["mean_risk_score"],
            overall_navigability_pct=summary["overall_navigability_pct"],
            generated_at=now,
            cache_expires_at=cache_expires,
        )

        await self._cache.set(cache_key, nav_map.model_dump(mode="json"), _TTL_MAP)
        return nav_map

    async def get_seasonal_calendar(
        self,
        waterway_id: str,
        year: int,
        force_refresh: bool = False,
    ) -> SeasonalCalendar:
        """
        Build a 12-month navigability calendar for every segment of a waterway.

        Parameters
        ----------
        waterway_id : str
            ``"NW-1"`` or ``"NW-2"``.
        year : int
            Calendar year for which to build the outlook.
        force_refresh : bool
            Bypass Redis cache and recompute all 12 months.

        Returns
        -------
        SeasonalCalendar
            Complete 12-month outlook per segment with waterway-level summaries.
        """
        cache_key = _cache_key("seasonal_cal", waterway_id, year)

        if not force_refresh:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit — seasonal_cal %s/%d.", waterway_id, year)
                return SeasonalCalendar(**cached)

        logger.info(
            "Computing 12-month seasonal calendar for %s %d …", waterway_id, year
        )

        # Fetch all 12 months concurrently
        month_tasks = [
            self._predict_all_segments(waterway_id, m, year, force_refresh)
            for m in range(1, 13)
        ]
        month_predictions: list[list[NavigabilityPrediction]] = list(
            await asyncio.gather(*month_tasks)
        )

        # Index predictions by segment_id for quick lookup
        seg_ids: list[str] = [p.segment_id for p in month_predictions[0]]

        # Build per-segment outlooks
        segment_outlooks: list[SegmentSeasonalOutlook] = []
        monthly_navigable_pct: dict[int, float] = {}

        for month_idx, (month_num, preds) in enumerate(
            zip(range(1, 13), month_predictions)
        ):
            n_total = len(preds)
            n_nav = sum(
                1 for p in preds if p.navigability_class == NavigabilityClass.NAVIGABLE
            )
            monthly_navigable_pct[month_num] = (
                round(100.0 * n_nav / n_total, 2) if n_total > 0 else 0.0
            )

        # Build a lookup: segment_id → {month: prediction}
        pred_by_seg_month: dict[str, dict[int, NavigabilityPrediction]] = {
            sid: {} for sid in seg_ids
        }
        for month_num, preds in zip(range(1, 13), month_predictions):
            for pred in preds:
                pred_by_seg_month.setdefault(pred.segment_id, {})[month_num] = pred

        for seg_id in seg_ids:
            monthly_preds = pred_by_seg_month.get(seg_id, {})
            monthly_outlooks: list[MonthlyOutlook] = []

            nav_months: list[int] = []
            cond_months: list[int] = []
            non_nav_months: list[int] = []

            for month_num in range(1, 13):
                pred = monthly_preds.get(month_num)
                if pred is None:
                    continue

                outlook = MonthlyOutlook(
                    month=month_num,
                    month_name=_MONTH_NAMES[month_num - 1],
                    season=_map_month_to_season(month_num),
                    predicted_depth_m=pred.predicted_depth_m,
                    depth_lower_ci=pred.depth_lower_ci,
                    depth_upper_ci=pred.depth_upper_ci,
                    width_m=pred.width_m,
                    navigability_class=pred.navigability_class,
                    navigability_probability=pred.navigability_probability,
                    risk_score=pred.risk_score,
                    is_historically_navigable=None,
                )
                monthly_outlooks.append(outlook)

                if pred.navigability_class == NavigabilityClass.NAVIGABLE:
                    nav_months.append(month_num)
                elif pred.navigability_class == NavigabilityClass.CONDITIONAL:
                    cond_months.append(month_num)
                else:
                    non_nav_months.append(month_num)

            # Peak/lowest month by depth
            if monthly_outlooks:
                depths = [(o.month, o.predicted_depth_m) for o in monthly_outlooks]
                peak_month = max(depths, key=lambda x: x[1])[0]
                lowest_month = min(depths, key=lambda x: x[1])[0]
            else:
                peak_month = None
                lowest_month = None

            annual_nav_pct = (
                round(100.0 * len(nav_months) / 12, 2) if monthly_outlooks else 0.0
            )

            segment_outlooks.append(
                SegmentSeasonalOutlook(
                    segment_id=seg_id,
                    waterway_id=WaterwayID(waterway_id),
                    year=year,
                    monthly_outlooks=monthly_outlooks,
                    navigable_months=nav_months,
                    conditional_months=cond_months,
                    non_navigable_months=non_nav_months,
                    annual_navigability_pct=annual_nav_pct,
                    peak_navigability_month=peak_month,
                    lowest_navigability_month=lowest_month,
                )
            )

        # Waterway-level best months (≥ 80% navigable)
        best_months = [
            m
            for m, pct in monthly_navigable_pct.items()
            if pct >= _BEST_MONTH_NAVIGABLE_PCT
        ]

        peak_start: Optional[int] = None
        peak_end: Optional[int] = None
        if best_months:
            peak_start = min(best_months)
            peak_end = max(best_months)

        seasonal_cal = SeasonalCalendar(
            waterway_id=WaterwayID(waterway_id),
            year=year,
            segment_outlooks=segment_outlooks,
            monthly_navigable_pct=monthly_navigable_pct,
            best_navigation_months=best_months,
            peak_season_start_month=peak_start,
            peak_season_end_month=peak_end,
            generated_at=datetime.now(timezone.utc),
        )

        await self._cache.set(
            cache_key, seasonal_cal.model_dump(mode="json"), _TTL_CALENDAR
        )
        return seasonal_cal

    async def get_risk_alerts(
        self,
        waterway_id: str,
        month: int,
        year: int,
        threshold: float = _DEFAULT_RISK_THRESHOLD,
        force_refresh: bool = False,
    ) -> list[RiskAlert]:
        """
        Identify river segments exceeding the risk threshold and build alerts.

        Parameters
        ----------
        waterway_id : str
            ``"NW-1"`` or ``"NW-2"``.
        month : int
            Calendar month (1–12).
        year : int
            Calendar year.
        threshold : float
            Risk-score threshold in [0, 1].  Segments with risk ≥ threshold
            generate an alert.
        force_refresh : bool
            Bypass Redis cache.

        Returns
        -------
        list[RiskAlert]
            Alerts sorted by descending risk score.
        """
        from app.services.alert_service import AlertService

        nav_map = await self.get_navigability_map(
            waterway_id, month, year, force_refresh
        )

        alert_svc = AlertService()
        alerts = await alert_svc.generate_risk_alerts(
            predictions=nav_map.predictions,
            threshold=threshold,
        )

        alerts.sort(key=lambda a: a.risk_score, reverse=True)
        return alerts

    async def get_depth_profile(
        self,
        waterway_id: str,
        month: int,
        year: int,
        force_refresh: bool = False,
    ) -> DepthProfile:
        """
        Build a longitudinal depth profile for a waterway at a given month/year.

        Parameters
        ----------
        waterway_id : str
            ``"NW-1"`` or ``"NW-2"``.
        month : int
            Calendar month (1–12).
        year : int
            Calendar year.
        force_refresh : bool
            Bypass Redis cache.

        Returns
        -------
        DepthProfile
            Ordered depth profile (upstream → downstream) with metadata.
        """
        cache_key = _cache_key("depth_profile", waterway_id, month, year)

        if not force_refresh:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                logger.debug(
                    "Cache hit — depth_profile %s %02d/%d.", waterway_id, month, year
                )
                return DepthProfile(**cached)

        nav_map = await self.get_navigability_map(
            waterway_id, month, year, force_refresh
        )

        predictions = nav_map.predictions
        if not predictions:
            raise ValueError(
                f"No predictions available for {waterway_id} {month:02d}/{year}."
            )

        seg_len_km = _WATERWAY_LENGTHS.get(waterway_id, 500.0) / max(
            len(predictions), 1
        )
        profile_points: list[DepthProfilePoint] = []
        critical_segments: list[str] = []

        depths = [p.predicted_depth_m for p in predictions]
        min_depth = float(min(depths))
        max_depth = float(max(depths))
        mean_depth = float(np.mean(depths))

        navigable_seg_count = 0
        chainage = 0.0

        for pred in predictions:
            # Midpoint chainage
            mid_chainage = chainage + seg_len_km / 2.0

            # GeoJSON point at segment midpoint (first coordinate as proxy)
            geom = pred.geometry
            if geom.get("type") == "LineString" and geom.get("coordinates"):
                coords = geom["coordinates"]
                mid_coord = coords[len(coords) // 2]
                point_geom: dict[str, Any] = {
                    "type": "Point",
                    "coordinates": mid_coord,
                }
            else:
                point_geom = pred.geometry

            profile_points.append(
                DepthProfilePoint(
                    segment_id=pred.segment_id,
                    chainage_km=round(mid_chainage, 3),
                    predicted_depth_m=pred.predicted_depth_m,
                    depth_lower_ci=pred.depth_lower_ci,
                    depth_upper_ci=pred.depth_upper_ci,
                    navigability_class=pred.navigability_class,
                    risk_score=pred.risk_score,
                    geometry=point_geom,
                )
            )

            if pred.predicted_depth_m < settings.DEPTH_NAVIGABLE_MIN:
                critical_segments.append(pred.segment_id)
            else:
                navigable_seg_count += 1

            chainage += seg_len_km

        navigable_stretch_km = navigable_seg_count * seg_len_km

        profile = DepthProfile(
            waterway_id=WaterwayID(waterway_id),
            month=month,
            year=year,
            profile_points=profile_points,
            total_length_km=round(chainage, 2),
            min_depth_m=round(min_depth, 3),
            max_depth_m=round(max_depth, 3),
            mean_depth_m=round(mean_depth, 3),
            critical_segments=critical_segments,
            navigable_stretch_km=round(navigable_stretch_km, 2),
            navigable_threshold_m=settings.DEPTH_NAVIGABLE_MIN,
            conditional_threshold_m=settings.DEPTH_CONDITIONAL_MIN,
            generated_at=datetime.now(timezone.utc),
        )

        await self._cache.set(cache_key, profile.model_dump(mode="json"), _TTL_DEPTH)
        return profile

    async def get_waterway_stats(
        self,
        waterway_id: str,
        year: int,
        force_refresh: bool = False,
    ) -> WaterwayStats:
        """
        Compute annual operational statistics for a National Waterway.

        Aggregates 12 monthly navigability maps into a full-year summary
        that includes monthly breakdowns, best/worst months, and trend
        comparison with the previous year.

        Parameters
        ----------
        waterway_id : str
            ``"NW-1"`` or ``"NW-2"``.
        year : int
            Target calendar year.
        force_refresh : bool
            Bypass Redis cache.

        Returns
        -------
        WaterwayStats
            Comprehensive annual statistics.
        """
        cache_key = _cache_key("waterway_stats", waterway_id, year)

        if not force_refresh:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit — waterway_stats %s/%d.", waterway_id, year)
                return WaterwayStats(**cached)

        logger.info("Computing annual stats for %s %d …", waterway_id, year)

        # Fetch all 12 monthly maps concurrently
        map_tasks = [
            self.get_navigability_map(waterway_id, m, year, force_refresh)
            for m in range(1, 13)
        ]
        monthly_maps: list[NavigabilityMap] = list(await asyncio.gather(*map_tasks))

        monthly_stats_list: list[MonthlyStats] = []
        total_alerts = 0
        critical_alerts = 0

        for nav_map in monthly_maps:
            n_segs = nav_map.total_segments or 1
            monthly_stats_list.append(
                MonthlyStats(
                    month=nav_map.month,
                    navigable_pct=nav_map.overall_navigability_pct,
                    conditional_pct=round(
                        100.0 * nav_map.conditional_count / n_segs, 2
                    ),
                    non_navigable_pct=round(
                        100.0 * nav_map.non_navigable_count / n_segs, 2
                    ),
                    mean_depth_m=nav_map.mean_depth_m or 0.0,
                    mean_width_m=nav_map.mean_width_m or 0.0,
                    mean_risk_score=nav_map.mean_risk_score or 0.0,
                    alert_count=nav_map.non_navigable_count,
                )
            )
            total_alerts += nav_map.non_navigable_count
            critical_alerts += sum(
                1
                for p in nav_map.predictions
                if p.risk_score >= settings.RISK_ALERT_THRESHOLD
            )

        # Annual averages
        nav_pcts = [ms.navigable_pct for ms in monthly_stats_list]
        depths = [ms.mean_depth_m for ms in monthly_stats_list]
        widths = [ms.mean_width_m for ms in monthly_stats_list]
        risks = [ms.mean_risk_score for ms in monthly_stats_list]

        annual_nav_pct = round(float(np.mean(nav_pcts)), 2)
        annual_depth = round(float(np.mean(depths)), 3)
        annual_width = round(float(np.mean(widths)), 2)
        annual_risk = round(float(np.mean(risks)), 4)

        best_month = int(np.argmax(nav_pcts)) + 1
        worst_month = int(np.argmin(nav_pcts)) + 1

        total_km = _WATERWAY_LENGTHS.get(waterway_id, 500.0)
        total_segs = _WATERWAY_SEGMENT_COUNTS.get(waterway_id, 100)

        # YoY comparison (try previous year — best effort)
        yoy_depth: Optional[float] = None
        yoy_nav_pct: Optional[float] = None
        try:
            prev_year_maps: list[NavigabilityMap] = list(
                await asyncio.gather(
                    *[
                        self.get_navigability_map(
                            waterway_id, m, year - 1, force_refresh=False
                        )
                        for m in range(1, 13)
                    ]
                )
            )
            prev_depths = [m.mean_depth_m or 0.0 for m in prev_year_maps]
            prev_nav_pcts = [m.overall_navigability_pct for m in prev_year_maps]
            yoy_depth = round(annual_depth - float(np.mean(prev_depths)), 3)
            yoy_nav_pct = round(annual_nav_pct - float(np.mean(prev_nav_pcts)), 2)
        except Exception as exc:
            logger.debug("YoY comparison failed (%s); skipping.", exc)

        # Identify deepest/shallowest segment across the best month's map
        best_map = monthly_maps[best_month - 1]
        deepest_seg: Optional[str] = None
        shallowest_seg: Optional[str] = None
        if best_map.predictions:
            sorted_by_depth = sorted(
                best_map.predictions, key=lambda p: p.predicted_depth_m
            )
            shallowest_seg = sorted_by_depth[0].segment_id
            deepest_seg = sorted_by_depth[-1].segment_id

        stats = WaterwayStats(
            waterway_id=WaterwayID(waterway_id),
            year=year,
            total_length_km=total_km,
            total_segments=total_segs,
            annual_navigable_pct=annual_nav_pct,
            annual_mean_depth_m=annual_depth,
            annual_mean_width_m=annual_width,
            annual_mean_risk_score=annual_risk,
            monthly_stats=monthly_stats_list,
            best_month=best_month,
            worst_month=worst_month,
            yoy_depth_change_m=yoy_depth,
            yoy_navigable_pct_change=yoy_nav_pct,
            total_alerts=total_alerts,
            critical_alerts=critical_alerts,
            deepest_segment_id=deepest_seg,
            shallowest_segment_id=shallowest_seg,
            generated_at=datetime.now(timezone.utc),
        )

        await self._cache.set(cache_key, stats.model_dump(mode="json"), _TTL_STATS)
        return stats

    async def compare_with_historical(
        self,
        waterway_id: str,
        month: int,
        year: int,
        base_years: int = 5,
        force_refresh: bool = False,
    ) -> HistoricalComparison:
        """
        Compare current month predictions against a multi-year historical baseline.

        Parameters
        ----------
        waterway_id : str
            ``"NW-1"`` or ``"NW-2"``.
        month : int
            Calendar month (1–12).
        year : int
            Current year.
        base_years : int
            Number of historical years to include in the baseline (default 5).
        force_refresh : bool
            Bypass Redis cache.

        Returns
        -------
        HistoricalComparison
            Anomaly statistics and trend direction.
        """
        cache_key = _cache_key("hist_compare", waterway_id, month, year, base_years)

        if not force_refresh:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                logger.debug(
                    "Cache hit — hist_compare %s %02d/%d.", waterway_id, month, year
                )
                return HistoricalComparison(**cached)

        logger.info(
            "Computing historical comparison for %s %s/%d (base %d yrs) …",
            waterway_id,
            _MONTH_NAMES[month - 1],
            year,
            base_years,
        )

        # Current year predictions
        current_map = await self.get_navigability_map(
            waterway_id, month, year, force_refresh
        )
        current_depth = current_map.mean_depth_m or 0.0
        current_nav_pct = current_map.overall_navigability_pct

        # Historical baselines
        hist_start = year - base_years
        hist_years = list(range(hist_start, year))

        hist_maps: list[NavigabilityMap] = []
        for hy in hist_years:
            try:
                hm = await self.get_navigability_map(
                    waterway_id, month, hy, force_refresh=False
                )
                hist_maps.append(hm)
            except Exception as exc:
                logger.debug(
                    "Historical map unavailable for %s %02d/%d: %s",
                    waterway_id,
                    month,
                    hy,
                    exc,
                )

        historical_series: list[HistoricalDataPoint] = []
        hist_depths: list[float] = []
        hist_nav_pcts: list[float] = []

        for hmap in hist_maps:
            if hmap.mean_depth_m is not None:
                hist_depths.append(hmap.mean_depth_m)
            hist_nav_pcts.append(hmap.overall_navigability_pct)

            # Representative point from first prediction in the map
            if hmap.predictions:
                rep = hmap.predictions[0]
                historical_series.append(
                    HistoricalDataPoint(
                        year=hmap.year,
                        month=hmap.month,
                        predicted_depth_m=hmap.mean_depth_m or 0.0,
                        width_m=hmap.mean_width_m or 0.0,
                        navigability_class=rep.navigability_class,
                        risk_score=rep.risk_score,
                    )
                )

        hist_mean_depth = float(np.mean(hist_depths)) if hist_depths else current_depth
        hist_mean_nav_pct = (
            float(np.mean(hist_nav_pcts)) if hist_nav_pcts else current_nav_pct
        )

        depth_anomaly = current_depth - hist_mean_depth
        depth_anomaly_pct = (
            100.0 * depth_anomaly / hist_mean_depth if hist_mean_depth > 0 else 0.0
        )

        # Trend detection: linear regression on historical depth series
        trend_dir = "stable"
        trend_sig = 0.0
        if len(hist_depths) >= 3:
            xs = np.arange(len(hist_depths), dtype=float)
            ys = np.array(hist_depths)
            coeffs = np.polyfit(xs, ys, 1)
            slope = float(coeffs[0])
            # Standardised slope as trend significance proxy
            std_depth = float(np.std(ys)) if np.std(ys) > 0 else 1.0
            trend_sig = float(np.clip(abs(slope) / std_depth, 0.0, 1.0))
            if slope > 0.05:
                trend_dir = "improving"
            elif slope < -0.05:
                trend_dir = "deteriorating"

        comparison = HistoricalComparison(
            waterway_id=WaterwayID(waterway_id),
            month=month,
            current_year=year,
            current_mean_depth_m=round(current_depth, 3),
            historical_mean_depth_m=round(hist_mean_depth, 3),
            depth_anomaly_m=round(depth_anomaly, 3),
            depth_anomaly_pct=round(depth_anomaly_pct, 2),
            current_navigable_pct=round(current_nav_pct, 2),
            historical_navigable_pct=round(hist_mean_nav_pct, 2),
            historical_series=historical_series,
            trend_direction=trend_dir,
            trend_significance=round(trend_sig, 4),
            generated_at=datetime.now(timezone.utc),
        )

        await self._cache.set(
            cache_key, comparison.model_dump(mode="json"), _TTL_HISTORICAL
        )
        return comparison

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    async def invalidate_cache(
        self,
        waterway_id: str,
        month: Optional[int] = None,
        year: Optional[int] = None,
    ) -> None:
        """Invalidate cached data for a waterway (or specific month/year).

        Parameters
        ----------
        waterway_id : str
            Target waterway.
        month : int | None
            If provided, only invalidate data for this month.
        year : int | None
            If provided, only invalidate data for this year.
        """
        # Simple targeted invalidation for known key patterns
        if month and year:
            keys_to_delete = [
                _cache_key("nav_map", waterway_id, month, year),
                _cache_key("depth_profile", waterway_id, month, year),
            ]
            for k in keys_to_delete:
                await self._cache.delete(k)
            logger.info("Cache invalidated for %s %02d/%d.", waterway_id, month, year)
        elif year:
            keys_to_delete = [
                _cache_key("seasonal_cal", waterway_id, year),
                _cache_key("waterway_stats", waterway_id, year),
            ]
            for k in keys_to_delete:
                await self._cache.delete(k)
            logger.info("Cache invalidated for %s year %d.", waterway_id, year)

    async def close(self) -> None:
        """Release Redis connection."""
        await self._cache.close()
