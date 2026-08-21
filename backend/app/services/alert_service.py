"""
AIDSTL Project — Alert Generation Service
==========================================
Generates, classifies, and manages risk alerts for inland waterway
navigability conditions on NW-1 (Ganga) and NW-2 (Brahmaputra).

Alert types
-----------
  DEPTH_CRITICAL       — depth has dropped below the non-navigable threshold (< 1.5 m)
  DEPTH_WARNING        — depth is below the navigable threshold but above conditional (1.5–3.0 m)
  WIDTH_RESTRICTION    — channel width below the navigable minimum (< 50 m)
  SEASONAL_TRANSITION  — navigability class has changed versus the previous month

Severity levels
---------------
  CRITICAL : immediate operational impact; route closure recommended
  HIGH     : significant degradation; reduced-draft vessel advisory
  MEDIUM   : conditions are marginal; monitoring required
  LOW      : pre-emptive warning; within acceptable operating limits

Caching
-------
  Generated alert lists are cached in Redis with a 30-minute TTL.
  Cache is invalidated whenever new predictions arrive for the same
  waterway / month / year combination.

Thread safety
-------------
  All public methods are async; heavy computation runs in a thread-pool
  executor so the FastAPI event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from calendar import month_name as _CAL_MONTH_NAME
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import numpy as np
import redis.asyncio as aioredis
from app.core.config import get_settings
from app.models.schemas.navigability import (
    AlertSeverity,
    AlertType,
    NavigabilityClass,
    NavigabilityPrediction,
    RiskAlert,
    WaterwayID,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Constants & thresholds
# ---------------------------------------------------------------------------

# Depth thresholds (metres) — derived from IWAI navigability standards
_DEPTH_NAVIGABLE: float = settings.DEPTH_NAVIGABLE_MIN  # 3.0 m
_DEPTH_CONDITIONAL: float = settings.DEPTH_CONDITIONAL_MIN  # 1.5 m

# Width thresholds (metres)
_WIDTH_NAVIGABLE: float = settings.WIDTH_NAVIGABLE_MIN  # 50 m
_WIDTH_CONDITIONAL: float = settings.WIDTH_CONDITIONAL_MIN  # 25 m

# Risk score threshold above which an alert is always emitted
_RISK_ALERT_THRESHOLD: float = settings.RISK_ALERT_THRESHOLD  # 0.7

# Redis cache TTL for alert lists (30 minutes)
_ALERT_CACHE_TTL: int = 1_800

# Alert expiry window — alerts are considered stale after 48 hours
_ALERT_EXPIRY_HOURS: int = 48

# Month labels (1-indexed) for readable descriptions
_MONTH_LABELS: dict[int, str] = {i: _CAL_MONTH_NAME[i] for i in range(1, 13)}

# ---------------------------------------------------------------------------
# Severity decision table
# depth_deficit  = max(0, depth_threshold - predicted_depth)
# width_deficit  = max(0, width_threshold - estimated_width)
# ---------------------------------------------------------------------------


def _depth_severity(depth_m: float) -> Optional[AlertSeverity]:
    """Map a depth reading to an alert severity, or None if no alert needed."""
    if depth_m < _DEPTH_CONDITIONAL:
        return AlertSeverity.CRITICAL
    if depth_m < _DEPTH_NAVIGABLE:
        return AlertSeverity.HIGH
    return None


def _width_severity(width_m: float) -> Optional[AlertSeverity]:
    """Map a width reading to an alert severity, or None if no alert needed."""
    if width_m < _WIDTH_CONDITIONAL:
        return AlertSeverity.CRITICAL
    if width_m < _WIDTH_NAVIGABLE:
        return AlertSeverity.MEDIUM
    return None


def _risk_score_severity(risk_score: float) -> Optional[AlertSeverity]:
    """Map a composite risk score to an alert severity."""
    if risk_score >= 0.90:
        return AlertSeverity.CRITICAL
    if risk_score >= 0.80:
        return AlertSeverity.HIGH
    if risk_score >= _RISK_ALERT_THRESHOLD:
        return AlertSeverity.MEDIUM
    return None


def _merge_severities(*severities: Optional[AlertSeverity]) -> AlertSeverity:
    """Return the most severe of a sequence of (possibly None) severities."""
    _order = {
        AlertSeverity.LOW: 1,
        AlertSeverity.MEDIUM: 2,
        AlertSeverity.HIGH: 3,
        AlertSeverity.CRITICAL: 4,
    }
    candidates = [s for s in severities if s is not None]
    if not candidates:
        return AlertSeverity.LOW
    return max(candidates, key=lambda s: _order[s])


# ---------------------------------------------------------------------------
# Alert text generators
# ---------------------------------------------------------------------------


def _depth_critical_description(
    segment_id: str,
    depth_m: float,
    month: int,
    year: int,
) -> tuple[str, str]:
    """Return (title, description) for a DEPTH_CRITICAL alert."""
    title = (
        f"Critical depth deficit on {segment_id} — "
        f"{depth_m:.2f} m ({_MONTH_LABELS[month]} {year})"
    )
    description = (
        f"Segment {segment_id} has a predicted water depth of {depth_m:.2f} m, "
        f"which is below the conditional navigability threshold of "
        f"{_DEPTH_CONDITIONAL:.1f} m. Vessel transit is not recommended. "
        f"Operators should divert traffic and contact the local harbour master. "
        f"Conditions are expected to persist until sufficient discharge is received "
        f"from upstream catchments."
    )
    return title, description


def _depth_warning_description(
    segment_id: str,
    depth_m: float,
    month: int,
    year: int,
) -> tuple[str, str]:
    """Return (title, description) for a DEPTH_WARNING alert."""
    title = (
        f"Depth warning on {segment_id} — "
        f"{depth_m:.2f} m ({_MONTH_LABELS[month]} {year})"
    )
    description = (
        f"Segment {segment_id} has a predicted water depth of {depth_m:.2f} m, "
        f"which is below the fully navigable threshold of {_DEPTH_NAVIGABLE:.1f} m. "
        f"Conditional navigation is possible for shallow-draft vessels (≤ 1.8 m). "
        f"Mariners are advised to proceed with caution and verify live gauge readings "
        f"before transit."
    )
    return title, description


def _width_restriction_description(
    segment_id: str,
    width_m: float,
    month: int,
    year: int,
) -> tuple[str, str]:
    """Return (title, description) for a WIDTH_RESTRICTION alert."""
    title = (
        f"Channel width restriction on {segment_id} — "
        f"{width_m:.0f} m ({_MONTH_LABELS[month]} {year})"
    )
    description = (
        f"Segment {segment_id} has an estimated channel width of {width_m:.0f} m, "
        f"which is below the navigable width threshold of {_WIDTH_NAVIGABLE:.0f} m. "
        f"Sandbar encroachment or bank erosion may be responsible. "
        f"Wide-beam vessels (beam > {width_m * 0.4:.0f} m) should not attempt transit. "
        f"A hydrographic survey is recommended to confirm current channel alignment."
    )
    return title, description


def _seasonal_transition_description(
    segment_id: str,
    previous_class: NavigabilityClass,
    current_class: NavigabilityClass,
    month: int,
    year: int,
) -> tuple[str, str]:
    """Return (title, description) for a SEASONAL_TRANSITION alert."""
    direction = (
        "deteriorated"
        if _class_rank(current_class) < _class_rank(previous_class)
        else "improved"
    )
    title = (
        f"Navigability transition on {segment_id}: "
        f"{previous_class.value} → {current_class.value} ({_MONTH_LABELS[month]} {year})"
    )
    description = (
        f"Conditions on segment {segment_id} have {direction} from "
        f"'{previous_class.value}' to '{current_class.value}' in {_MONTH_LABELS[month]} {year}. "
        f"This transition is consistent with the seasonal hydrological cycle. "
        f"Operators should update their route plans and review vessel suitability "
        f"for the new navigability classification."
    )
    return title, description


def _class_rank(nav_class: NavigabilityClass) -> int:
    """Rank navigability classes: navigable > conditional > non_navigable."""
    return {
        NavigabilityClass.NAVIGABLE: 2,
        NavigabilityClass.CONDITIONAL: 1,
        NavigabilityClass.NON_NAVIGABLE: 0,
    }[nav_class]


def _recommended_action(alert_type: AlertType, severity: AlertSeverity) -> str:
    """Return a recommended operational action string."""
    actions = {
        (AlertType.DEPTH_CRITICAL, AlertSeverity.CRITICAL): (
            "Suspend vessel operations immediately. Contact IWAI Regional Office for "
            "updated dredging schedule. Issue NOTAM (Notice to Mariners)."
        ),
        (AlertType.DEPTH_CRITICAL, AlertSeverity.HIGH): (
            "Restrict to shallow-draft vessels only (< 1.5 m draft). "
            "Implement reduced-speed zones and enhance radar watch."
        ),
        (AlertType.DEPTH_WARNING, AlertSeverity.HIGH): (
            "Issue advisory to fleet operators. Permit passage for light-draft vessels "
            "only. Monitor gauge stations daily."
        ),
        (AlertType.DEPTH_WARNING, AlertSeverity.MEDIUM): (
            "Monitor conditions closely. Advise operators to check live gauge data "
            "before departure. Maintain safety margins."
        ),
        (AlertType.WIDTH_RESTRICTION, AlertSeverity.CRITICAL): (
            "Route wide-beam vessels via alternative channels. Request emergency "
            "hydrographic survey. Coordinate with port authorities."
        ),
        (AlertType.WIDTH_RESTRICTION, AlertSeverity.MEDIUM): (
            "Single-lane traffic management recommended. Implement VHF radio "
            "communication between vessels for passing arrangements."
        ),
        (AlertType.SEASONAL_TRANSITION, AlertSeverity.HIGH): (
            "Update route plans to reflect changed navigability class. Notify all "
            "vessels in transit. Review insurance coverage for altered conditions."
        ),
        (AlertType.SEASONAL_TRANSITION, AlertSeverity.MEDIUM): (
            "Revise operational schedules. Distribute updated navigability charts to "
            "fleet operators. Increase monitoring frequency."
        ),
    }
    default = "Monitor conditions and consult local harbour master before transit."
    return actions.get((alert_type, severity), default)


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------


def _alert_cache_key(waterway_id: str, month: int, year: int) -> str:
    """Generate a deterministic Redis cache key for an alert list."""
    raw = f"aidstl:alerts:{waterway_id}:{month:02d}:{year}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _next_month_cache_key(segment_id: str) -> str:
    raw = f"aidstl:next_month_risk:{segment_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# AlertService
# ---------------------------------------------------------------------------


class AlertService:
    """
    Service for generating, classifying, and forecasting risk alerts.

    All alert generation is deterministic — given the same predictions,
    the same alerts are produced.  This means clients may safely re-request
    alerts without duplicates being stored downstream.

    Lifecycle
    ---------
    Instantiate once (singleton pattern via ``get_alert_service()``) and
    reuse across requests.  Redis is initialised lazily on first use.

    Usage
    -----
    ::

        svc = get_alert_service()
        alerts = await svc.generate_risk_alerts(predictions)
        next_risk = await svc.predict_next_month_risk(segment_id, history)
    """

    _instance: Optional["AlertService"] = None
    _instance_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._redis: Optional[aioredis.Redis] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    async def get_instance(cls) -> "AlertService":
        """Return (or create) the process-wide AlertService singleton."""
        if cls._instance is None:
            async with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    async def _get_redis(self) -> aioredis.Redis:  # type: ignore[type-arg]
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
        return self._redis

    async def _cache_get(self, key: str) -> Optional[list[dict[str, Any]]]:
        try:
            redis = await self._get_redis()
            raw = await redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Alert cache GET failed (key=%s): %s", key, exc)
        return None

    async def _cache_set(
        self,
        key: str,
        value: list[dict[str, Any]],
        ttl: int = _ALERT_CACHE_TTL,
    ) -> None:
        try:
            redis = await self._get_redis()
            await redis.set(key, json.dumps(value, default=str), ex=ttl)
        except Exception as exc:
            logger.warning("Alert cache SET failed (key=%s): %s", key, exc)

    async def invalidate_cache(self, waterway_id: str, month: int, year: int) -> None:
        """Remove the cached alert list for a waterway/month/year combination."""
        key = _alert_cache_key(waterway_id, month, year)
        try:
            redis = await self._get_redis()
            await redis.delete(key)
            logger.debug("Alert cache invalidated: %s", key)
        except Exception as exc:
            logger.warning("Alert cache invalidation failed: %s", exc)

    # ------------------------------------------------------------------
    # Core alert generation
    # ------------------------------------------------------------------

    async def generate_risk_alerts(
        self,
        predictions: list[NavigabilityPrediction],
        previous_predictions: Optional[list[NavigabilityPrediction]] = None,
        risk_threshold: float = _RISK_ALERT_THRESHOLD,
        use_cache: bool = True,
    ) -> list[RiskAlert]:
        """
        Generate risk alerts from a list of navigability predictions.

        Evaluates each prediction against four alert criteria:
        1. Depth critical   — depth < conditional threshold
        2. Depth warning    — depth < navigable threshold
        3. Width restriction — width < navigable threshold
        4. Seasonal transition — class differs from previous month

        Parameters
        ----------
        predictions : list[NavigabilityPrediction]
            Current-month predictions for a waterway (all segments).
        previous_predictions : list[NavigabilityPrediction], optional
            Previous-month predictions used to detect navigability transitions.
            Pass ``None`` to skip SEASONAL_TRANSITION alerts.
        risk_threshold : float, optional
            Minimum risk score to trigger an alert (default from settings).
        use_cache : bool, optional
            Whether to check and populate the Redis cache.

        Returns
        -------
        list[RiskAlert]
            Deduplicated, sorted list of risk alerts.
            Sorted by severity (CRITICAL first), then by segment chainage.
        """
        if not predictions:
            return []

        # Check cache
        if use_cache and predictions:
            p0 = predictions[0]
            cache_key = _alert_cache_key(p0.waterway_id, p0.month, p0.year)
            cached = await self._cache_get(cache_key)
            if cached is not None:
                logger.debug(
                    "Alert cache hit for %s %02d/%d — %d alerts.",
                    p0.waterway_id,
                    p0.month,
                    p0.year,
                    len(cached),
                )
                return [RiskAlert(**a) for a in cached]

        # Build a lookup for previous predictions by segment_id
        prev_by_segment: dict[str, NavigabilityPrediction] = {}
        if previous_predictions:
            prev_by_segment = {p.segment_id: p for p in previous_predictions}

        # Generate alerts per segment
        loop = asyncio.get_event_loop()
        alerts_nested = await loop.run_in_executor(
            None,
            self._generate_alerts_sync,
            predictions,
            prev_by_segment,
            risk_threshold,
        )
        alerts = _flatten(alerts_nested)

        # Sort: severity descending, then segment_id ascending
        _sev_order = {
            AlertSeverity.CRITICAL: 4,
            AlertSeverity.HIGH: 3,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 1,
        }
        alerts.sort(
            key=lambda a: (-_sev_order.get(AlertSeverity(a.severity), 0), a.segment_id)
        )

        logger.info(
            "Generated %d alerts for %s %02d/%d.",
            len(alerts),
            predictions[0].waterway_id if predictions else "unknown",
            predictions[0].month if predictions else 0,
            predictions[0].year if predictions else 0,
        )

        # Populate cache
        if use_cache and predictions:
            alert_dicts = [a.model_dump(mode="json") for a in alerts]
            await self._cache_set(cache_key, alert_dicts)

        return alerts

    def _generate_alerts_sync(
        self,
        predictions: list[NavigabilityPrediction],
        prev_by_segment: dict[str, NavigabilityPrediction],
        risk_threshold: float,
    ) -> list[list[RiskAlert]]:
        """
        Synchronous per-segment alert generation (runs in thread pool).

        Returns a nested list (one sub-list per segment) to allow parallel
        processing without requiring explicit locks.
        """
        result: list[list[RiskAlert]] = []
        expiry = datetime.now(timezone.utc) + timedelta(hours=_ALERT_EXPIRY_HOURS)

        for pred in predictions:
            seg_alerts: list[RiskAlert] = []

            # ── 1. Depth critical ─────────────────────────────────────────
            depth_sev = _depth_severity(pred.predicted_depth_m)
            if depth_sev in (AlertSeverity.CRITICAL, AlertSeverity.HIGH):
                alert_type = AlertType.DEPTH_CRITICAL
                if depth_sev == AlertSeverity.HIGH:
                    alert_type = AlertType.DEPTH_WARNING
                title, description = (
                    _depth_critical_description(
                        pred.segment_id,
                        pred.predicted_depth_m,
                        pred.month,
                        pred.year,
                    )
                    if depth_sev == AlertSeverity.CRITICAL
                    else _depth_warning_description(
                        pred.segment_id,
                        pred.predicted_depth_m,
                        pred.month,
                        pred.year,
                    )
                )
                seg_alerts.append(
                    RiskAlert(
                        alert_id=_make_alert_id(
                            pred.segment_id, alert_type, pred.month, pred.year
                        ),
                        waterway_id=pred.waterway_id,
                        segment_id=pred.segment_id,
                        alert_type=alert_type,
                        severity=depth_sev,
                        title=title,
                        description=description,
                        current_depth_m=pred.predicted_depth_m,
                        threshold_depth_m=(
                            _DEPTH_CONDITIONAL
                            if depth_sev == AlertSeverity.CRITICAL
                            else _DEPTH_NAVIGABLE
                        ),
                        current_width_m=pred.width_m,
                        risk_score=pred.risk_score,
                        risk_trend=_compute_risk_trend(
                            pred, prev_by_segment.get(pred.segment_id)
                        ),
                        affected_month=pred.month,
                        affected_year=pred.year,
                        previous_class=(
                            prev_by_segment[pred.segment_id].navigability_class
                            if pred.segment_id in prev_by_segment
                            else None
                        ),
                        current_class=pred.navigability_class,
                        recommended_action=_recommended_action(alert_type, depth_sev),
                        geometry=pred.geometry,
                        expires_at=expiry,
                    )
                )

            # ── 2. Depth warning (if not already captured as critical) ────
            elif depth_sev == AlertSeverity.MEDIUM or (
                _DEPTH_CONDITIONAL <= pred.predicted_depth_m < _DEPTH_NAVIGABLE
                and depth_sev is None
            ):
                if pred.predicted_depth_m < _DEPTH_NAVIGABLE:
                    alert_type = AlertType.DEPTH_WARNING
                    title, description = _depth_warning_description(
                        pred.segment_id, pred.predicted_depth_m, pred.month, pred.year
                    )
                    severity = AlertSeverity.MEDIUM
                    seg_alerts.append(
                        RiskAlert(
                            alert_id=_make_alert_id(
                                pred.segment_id, alert_type, pred.month, pred.year
                            ),
                            waterway_id=pred.waterway_id,
                            segment_id=pred.segment_id,
                            alert_type=alert_type,
                            severity=severity,
                            title=title,
                            description=description,
                            current_depth_m=pred.predicted_depth_m,
                            threshold_depth_m=_DEPTH_NAVIGABLE,
                            current_width_m=pred.width_m,
                            risk_score=pred.risk_score,
                            risk_trend=_compute_risk_trend(
                                pred, prev_by_segment.get(pred.segment_id)
                            ),
                            affected_month=pred.month,
                            affected_year=pred.year,
                            previous_class=(
                                prev_by_segment[pred.segment_id].navigability_class
                                if pred.segment_id in prev_by_segment
                                else None
                            ),
                            current_class=pred.navigability_class,
                            recommended_action=_recommended_action(
                                alert_type, severity
                            ),
                            geometry=pred.geometry,
                            expires_at=expiry,
                        )
                    )

            # ── 3. Width restriction ──────────────────────────────────────
            width_sev = _width_severity(pred.width_m)
            if width_sev is not None:
                alert_type = AlertType.WIDTH_RESTRICTION
                title, description = _width_restriction_description(
                    pred.segment_id, pred.width_m, pred.month, pred.year
                )
                seg_alerts.append(
                    RiskAlert(
                        alert_id=_make_alert_id(
                            pred.segment_id, alert_type, pred.month, pred.year
                        ),
                        waterway_id=pred.waterway_id,
                        segment_id=pred.segment_id,
                        alert_type=alert_type,
                        severity=width_sev,
                        title=title,
                        description=description,
                        current_depth_m=pred.predicted_depth_m,
                        current_width_m=pred.width_m,
                        threshold_width_m=_WIDTH_NAVIGABLE,
                        risk_score=pred.risk_score,
                        risk_trend=_compute_risk_trend(
                            pred, prev_by_segment.get(pred.segment_id)
                        ),
                        affected_month=pred.month,
                        affected_year=pred.year,
                        previous_class=(
                            prev_by_segment[pred.segment_id].navigability_class
                            if pred.segment_id in prev_by_segment
                            else None
                        ),
                        current_class=pred.navigability_class,
                        recommended_action=_recommended_action(alert_type, width_sev),
                        geometry=pred.geometry,
                        expires_at=expiry,
                    )
                )

            # ── 4. Seasonal / class transition ───────────────────────────
            prev = prev_by_segment.get(pred.segment_id)
            if prev is not None:
                prev_cls = NavigabilityClass(prev.navigability_class)
                curr_cls = NavigabilityClass(pred.navigability_class)

                if prev_cls != curr_cls:
                    # Only alert on deterioration or first-time improvement
                    rank_diff = _class_rank(curr_cls) - _class_rank(prev_cls)
                    severity = (
                        AlertSeverity.HIGH
                        if rank_diff < -1
                        else (
                            AlertSeverity.MEDIUM if rank_diff < 0 else AlertSeverity.LOW
                        )
                    )
                    alert_type = AlertType.SEASONAL_TRANSITION
                    title, description = _seasonal_transition_description(
                        pred.segment_id, prev_cls, curr_cls, pred.month, pred.year
                    )
                    seg_alerts.append(
                        RiskAlert(
                            alert_id=_make_alert_id(
                                pred.segment_id, alert_type, pred.month, pred.year
                            ),
                            waterway_id=pred.waterway_id,
                            segment_id=pred.segment_id,
                            alert_type=alert_type,
                            severity=severity,
                            title=title,
                            description=description,
                            current_depth_m=pred.predicted_depth_m,
                            current_width_m=pred.width_m,
                            risk_score=pred.risk_score,
                            risk_trend=_compute_risk_trend(pred, prev),
                            affected_month=pred.month,
                            affected_year=pred.year,
                            previous_class=prev_cls,
                            current_class=curr_cls,
                            recommended_action=_recommended_action(
                                alert_type, severity
                            ),
                            geometry=pred.geometry,
                            expires_at=expiry,
                        )
                    )

            # ── 5. High risk score (catch-all) ────────────────────────────
            # If no specific alert was generated but the risk score is high,
            # emit a generic DEPTH_WARNING to surface it.
            if not seg_alerts and pred.risk_score >= risk_threshold:
                rs_sev = _risk_score_severity(pred.risk_score)
                if rs_sev is not None:
                    alert_type = AlertType.DEPTH_WARNING
                    title = (
                        f"Elevated risk on {pred.segment_id} — "
                        f"risk score {pred.risk_score:.2f} ({_MONTH_LABELS[pred.month]} {pred.year})"
                    )
                    description = (
                        f"Segment {pred.segment_id} has a composite risk score of "
                        f"{pred.risk_score:.3f} (threshold: {risk_threshold:.2f}). "
                        f"Predicted depth: {pred.predicted_depth_m:.2f} m, "
                        f"width: {pred.width_m:.0f} m. "
                        f"Navigability class: {pred.navigability_class}. "
                        f"Enhanced monitoring is recommended."
                    )
                    seg_alerts.append(
                        RiskAlert(
                            alert_id=_make_alert_id(
                                pred.segment_id, alert_type, pred.month, pred.year
                            ),
                            waterway_id=pred.waterway_id,
                            segment_id=pred.segment_id,
                            alert_type=alert_type,
                            severity=rs_sev,
                            title=title,
                            description=description,
                            current_depth_m=pred.predicted_depth_m,
                            threshold_depth_m=_DEPTH_NAVIGABLE,
                            current_width_m=pred.width_m,
                            risk_score=pred.risk_score,
                            risk_trend=_compute_risk_trend(
                                pred, prev_by_segment.get(pred.segment_id)
                            ),
                            affected_month=pred.month,
                            affected_year=pred.year,
                            previous_class=(
                                prev_by_segment[pred.segment_id].navigability_class
                                if pred.segment_id in prev_by_segment
                                else None
                            ),
                            current_class=pred.navigability_class,
                            recommended_action=_recommended_action(alert_type, rs_sev),
                            geometry=pred.geometry,
                            expires_at=expiry,
                        )
                    )

            result.append(seg_alerts)

        return result

    # ------------------------------------------------------------------
    # Next-month risk forecasting
    # ------------------------------------------------------------------

    async def predict_next_month_risk(
        self,
        segment_id: str,
        history: list[dict[str, Any]],
        n_lag_months: int = 6,
    ) -> dict[str, Any]:
        """
        Forecast the risk score for the next calendar month using historical data.

        Uses a simple exponentially-weighted trend model on the historical risk
        score series.  In production this would be replaced by the TFT model's
        multi-step-ahead output.

        Parameters
        ----------
        segment_id : str
            The segment to forecast (e.g. ``"NW-1-042"``).
        history : list[dict[str, Any]]
            Ordered historical records (oldest first).  Each record must
            contain at minimum ``"risk_score"`` and optionally ``"month"``,
            ``"year"``, ``"predicted_depth_m"``, ``"width_m"``.
        n_lag_months : int, optional
            Number of most-recent months to use for trend estimation (default 6).

        Returns
        -------
        dict[str, Any]
            ``{
                "segment_id": str,
                "forecast_month": int,
                "forecast_year": int,
                "predicted_risk_score": float,
                "trend": "increasing" | "stable" | "decreasing",
                "confidence": float,
                "explanation": str,
                "alert_likely": bool,
            }``
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._forecast_sync,
            segment_id,
            history,
            n_lag_months,
        )
        return result

    def _forecast_sync(
        self,
        segment_id: str,
        history: list[dict[str, Any]],
        n_lag: int,
    ) -> dict[str, Any]:
        """Synchronous next-month risk forecasting (thread-pool)."""
        now = datetime.now(timezone.utc)
        forecast_month = now.month % 12 + 1
        forecast_year = now.year if forecast_month > 1 else now.year + 1

        if not history:
            return {
                "segment_id": segment_id,
                "forecast_month": forecast_month,
                "forecast_year": forecast_year,
                "predicted_risk_score": 0.5,
                "trend": "stable",
                "confidence": 0.0,
                "explanation": "No historical data available; returning neutral forecast.",
                "alert_likely": False,
            }

        # Extract risk scores from the most recent n_lag records
        recent = history[-n_lag:]
        risk_scores = np.array(
            [float(r.get("risk_score", 0.5)) for r in recent], dtype=np.float64
        )

        if len(risk_scores) == 0:
            risk_scores = np.array([0.5])

        # Exponential weighted mean — more recent observations carry higher weight
        weights = np.exp(np.linspace(-1.0, 0.0, len(risk_scores)))
        weights /= weights.sum()
        ewm_risk = float(np.dot(weights, risk_scores))

        # Trend detection via linear regression slope
        if len(risk_scores) >= 3:
            x = np.arange(len(risk_scores), dtype=np.float64)
            coeffs = np.polyfit(x, risk_scores, 1)
            slope = float(coeffs[0])
        else:
            slope = 0.0

        trend: str
        if slope > 0.02:
            trend = "increasing"
            projected = min(1.0, ewm_risk + slope * 1.5)
        elif slope < -0.02:
            trend = "decreasing"
            projected = max(0.0, ewm_risk + slope * 1.5)
        else:
            trend = "stable"
            projected = ewm_risk

        # Adjust for seasonal patterns (simple climatological nudge)
        projected = self._apply_seasonal_adjustment(projected, forecast_month)

        # Confidence: higher when more historical data is available
        confidence = min(1.0, len(risk_scores) / max(n_lag, 1))

        explanation = (
            f"Based on {len(risk_scores)} historical month(s), the exponentially "
            f"weighted mean risk score is {ewm_risk:.3f} with a {trend} trend "
            f"(slope={slope:.4f} per month). "
            f"Seasonal adjustment applied for month {forecast_month}. "
            f"Projected risk: {projected:.3f}."
        )

        return {
            "segment_id": segment_id,
            "forecast_month": forecast_month,
            "forecast_year": forecast_year,
            "predicted_risk_score": round(float(np.clip(projected, 0.0, 1.0)), 4),
            "trend": trend,
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "alert_likely": bool(projected >= _RISK_ALERT_THRESHOLD),
        }

    @staticmethod
    def _apply_seasonal_adjustment(risk_score: float, month: int) -> float:
        """Apply a simple climatological seasonal correction to the risk score.

        During the Indian summer monsoon (June–September), rivers typically
        have higher discharge and are MORE navigable (lower risk for depth).
        During the dry winter months (December–February), risk is higher.

        Parameters
        ----------
        risk_score : float
            Raw projected risk score [0, 1].
        month : int
            Target forecast month (1–12).

        Returns
        -------
        float
            Adjusted risk score [0, 1].
        """
        # Seasonal multipliers — empirically derived from NW-1 / NW-2 gauge data
        _seasonal_delta = {
            1: +0.05,  # January  — dry, risk elevated
            2: +0.08,  # February — driest month, maximum dry-season risk
            3: +0.06,  # March    — pre-monsoon
            4: +0.03,  # April    — early pre-monsoon
            5: +0.01,  # May      — late pre-monsoon
            6: -0.05,  # June     — monsoon onset, risk drops
            7: -0.10,  # July     — peak monsoon, lowest risk
            8: -0.10,  # August   — peak monsoon
            9: -0.07,  # September— late monsoon
            10: -0.02,  # October  — post-monsoon
            11: +0.02,  # November — transition
            12: +0.04,  # December — early dry season
        }
        delta = _seasonal_delta.get(month, 0.0)
        return float(np.clip(risk_score + delta, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Critical alert filter
    # ------------------------------------------------------------------

    async def get_critical_alerts(
        self,
        waterway_ids: Optional[list[str]] = None,
        month: Optional[int] = None,
        year: Optional[int] = None,
    ) -> list[RiskAlert]:
        """
        Retrieve all CRITICAL severity alerts across one or more waterways.

        Parameters
        ----------
        waterway_ids : list[str], optional
            Waterways to filter. Defaults to all supported waterways.
        month : int, optional
            Filter by month (1–12). Defaults to current month.
        year : int, optional
            Filter by year. Defaults to current year.

        Returns
        -------
        list[RiskAlert]
            All CRITICAL alerts, sorted by waterway then segment.
        """
        now = datetime.now(timezone.utc)
        month = month or now.month
        year = year or now.year
        waterway_ids = waterway_ids or settings.SUPPORTED_WATERWAYS

        all_alerts: list[RiskAlert] = []

        for wid in waterway_ids:
            key = _alert_cache_key(wid, month, year)
            cached = await self._cache_get(key)
            if cached:
                wid_alerts = [
                    RiskAlert(**a)
                    for a in cached
                    if a.get("severity") == AlertSeverity.CRITICAL
                ]
                all_alerts.extend(wid_alerts)

        all_alerts.sort(key=lambda a: (a.waterway_id, a.segment_id))
        logger.info(
            "Returning %d critical alerts across %s for %02d/%d.",
            len(all_alerts),
            waterway_ids,
            month,
            year,
        )
        return all_alerts

    # ------------------------------------------------------------------
    # Alert acknowledgement
    # ------------------------------------------------------------------

    async def acknowledge_alert(
        self, alert_id: str, waterway_id: str, month: int, year: int
    ) -> bool:
        """
        Mark a specific alert as acknowledged.

        Updates the cached alert list so the acknowledged flag persists
        until the cache entry expires.

        Parameters
        ----------
        alert_id : str
            The UUID of the alert to acknowledge.
        waterway_id : str
            Parent waterway (needed to locate the cache entry).
        month, year : int
            Temporal context of the alert.

        Returns
        -------
        bool
            ``True`` if the alert was found and updated, ``False`` otherwise.
        """
        key = _alert_cache_key(waterway_id, month, year)
        cached = await self._cache_get(key)
        if not cached:
            return False

        found = False
        now_iso = datetime.now(timezone.utc).isoformat()
        for alert_dict in cached:
            if alert_dict.get("alert_id") == alert_id:
                alert_dict["acknowledged"] = True
                alert_dict["acknowledged_at"] = now_iso
                found = True
                break

        if found:
            await self._cache_set(key, cached)
            logger.info("Alert %s acknowledged.", alert_id)

        return found

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summarise_alerts(self, alerts: list[RiskAlert]) -> dict[str, Any]:
        """
        Compute summary statistics over a list of alerts.

        Parameters
        ----------
        alerts : list[RiskAlert]
            The alerts to summarise.

        Returns
        -------
        dict[str, Any]
            Keys:
            ``total``, ``by_type``, ``by_severity``, ``acknowledged``,
            ``segments_affected``, ``mean_risk_score``.
        """
        if not alerts:
            return {
                "total": 0,
                "by_type": {},
                "by_severity": {},
                "acknowledged": 0,
                "segments_affected": 0,
                "mean_risk_score": 0.0,
            }

        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        acknowledged = 0
        segments: set[str] = set()
        risk_scores: list[float] = []

        for alert in alerts:
            atype = str(alert.alert_type)
            asev = str(alert.severity)
            by_type[atype] = by_type.get(atype, 0) + 1
            by_severity[asev] = by_severity.get(asev, 0) + 1
            if alert.acknowledged:
                acknowledged += 1
            segments.add(alert.segment_id)
            risk_scores.append(alert.risk_score)

        return {
            "total": len(alerts),
            "by_type": by_type,
            "by_severity": by_severity,
            "acknowledged": acknowledged,
            "unacknowledged": len(alerts) - acknowledged,
            "segments_affected": len(segments),
            "mean_risk_score": round(float(np.mean(risk_scores)), 4)
            if risk_scores
            else 0.0,
            "max_risk_score": round(float(np.max(risk_scores)), 4)
            if risk_scores
            else 0.0,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release Redis connection and reset singleton."""
        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception:
                pass
            self._redis = None
        AlertService._instance = None
        logger.info("AlertService shut down.")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_alert_id(
    segment_id: str,
    alert_type: AlertType,
    month: int,
    year: int,
) -> str:
    """Generate a deterministic UUID-format alert identifier.

    The ID is derived from the segment, alert type, and time period so
    that the same conditions always produce the same alert ID, enabling
    idempotent downstream processing.
    """
    raw = f"{segment_id}:{alert_type}:{month:02d}:{year}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    # Format as UUID v4 (RFC 4122 layout)
    return (
        f"{digest[0:8]}-{digest[8:12]}-4{digest[13:16]}-"
        f"{hex(int(digest[16:18], 16) & 0x3F | 0x80)[2:]}{digest[18:20]}-"
        f"{digest[20:32]}"
    )


def _flatten(nested: list[list[RiskAlert]]) -> list[RiskAlert]:
    """Flatten a list of alert lists into a single list."""
    return [alert for sub in nested for alert in sub]


def _compute_risk_trend(
    current: NavigabilityPrediction,
    previous: Optional[NavigabilityPrediction],
) -> str:
    """Determine whether the risk score trend is increasing, stable, or decreasing."""
    if previous is None:
        return "stable"
    delta = current.risk_score - previous.risk_score
    if delta > 0.05:
        return "increasing"
    if delta < -0.05:
        return "decreasing"
    return "stable"


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------


_service_instance: Optional[AlertService] = None
_service_lock = asyncio.Lock()


async def get_alert_service() -> AlertService:
    """
    Return the application-wide AlertService singleton.

    Safe to call from any async context; thread-safe singleton creation
    is enforced via an asyncio.Lock.

    Returns
    -------
    AlertService
        The singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        async with _service_lock:
            if _service_instance is None:
                _service_instance = AlertService()
    return _service_instance
