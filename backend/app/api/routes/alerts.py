"""
AIDSTL Project — Alerts API Routes
====================================
REST endpoints for navigability risk alert management.

Endpoints
---------
  GET  /api/v1/alerts/{waterway_id}         — alerts for a specific waterway
  GET  /api/v1/alerts/critical              — critical alerts across all waterways
  POST /api/v1/alerts/subscribe             — register a webhook subscription
  GET  /api/v1/alerts/{waterway_id}/summary — alert count summary
  POST /api/v1/alerts/{alert_id}/acknowledge — acknowledge a specific alert
  GET  /api/v1/alerts/next-month-risk/{segment_id} — next-month risk forecast

Design notes
------------
  - All alert generation is delegated to AlertService which caches results
    in Redis with a 30-minute TTL.
  - Alert IDs are deterministic (derived from segment + type + period) so
    clients can safely de-duplicate across repeated requests.
  - Webhook subscriptions are stored in Redis and validated on each call.
  - The /critical endpoint aggregates across all supported waterways.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

import redis.asyncio as aioredis
from app.core.config import get_settings
from app.models.schemas.navigability import (
    AlertSeverity,
    AlertSubscription,
    AlertType,
    NavigabilityPrediction,
    RiskAlert,
    WaterwayID,
)
from app.services.alert_service import AlertService, get_alert_service
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
from pydantic import BaseModel, Field, HttpUrl

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/alerts",
    tags=["Alerts"],
    responses={
        404: {"description": "Waterway or segment not found"},
        422: {"description": "Validation error — check request parameters"},
        503: {"description": "Service unavailable — Redis or model error"},
    },
)

# ---------------------------------------------------------------------------
# Dependency injection helpers
# ---------------------------------------------------------------------------

_nav_service: Optional[NavigabilityService] = None


def get_nav_service() -> NavigabilityService:
    """FastAPI dependency — returns the singleton NavigabilityService."""
    global _nav_service
    if _nav_service is None:
        _nav_service = NavigabilityService()
    return _nav_service


async def get_alert_svc() -> AlertService:
    """FastAPI dependency — returns the singleton AlertService."""
    return await get_alert_service()


# ---------------------------------------------------------------------------
# Path / query parameter type aliases
# ---------------------------------------------------------------------------

WaterwayPathParam = Annotated[
    str,
    Path(
        title="Waterway ID",
        description=(
            "National Waterway identifier. "
            "Use 'NW-1' for Ganga (Varanasi–Haldia) or "
            "'NW-2' for Brahmaputra (Dhubri–Sadiya)."
        ),
        pattern=r"^NW-[12]$",
        examples=["NW-1", "NW-2"],
    ),
]

MonthQuery = Annotated[
    Optional[int],
    Query(
        ge=1,
        le=12,
        title="Month",
        description="Calendar month (1–12). Defaults to current UTC month.",
    ),
]

YearQuery = Annotated[
    Optional[int],
    Query(
        ge=2015,
        le=2100,
        title="Year",
        description="Calendar year ≥ 2015. Defaults to current UTC year.",
    ),
]


# ---------------------------------------------------------------------------
# Local response models
# ---------------------------------------------------------------------------


class AlertSummaryResponse(BaseModel):
    """Aggregated alert statistics for a waterway/month/year."""

    waterway_id: str
    month: int
    year: int
    total: int
    by_type: dict[str, int]
    by_severity: dict[str, int]
    acknowledged: int
    unacknowledged: int
    segments_affected: int
    mean_risk_score: float
    max_risk_score: float
    generated_at: str


class AcknowledgeRequest(BaseModel):
    """Request body for alert acknowledgement."""

    waterway_id: str = Field(
        ...,
        description="Parent waterway of the alert (needed to locate the cache entry).",
        pattern=r"^NW-[12]$",
    )
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2015, le=2100)


class AcknowledgeResponse(BaseModel):
    """Response for alert acknowledgement."""

    alert_id: str
    acknowledged: bool
    acknowledged_at: Optional[str] = None
    message: str


class NextMonthRiskResponse(BaseModel):
    """Next-month risk forecast for a single segment."""

    segment_id: str
    forecast_month: int
    forecast_year: int
    predicted_risk_score: float
    trend: str
    confidence: float
    explanation: str
    alert_likely: bool
    generated_at: str


class WebhookSubscriptionResponse(BaseModel):
    """Confirmation of a webhook subscription."""

    subscription_id: str
    waterway_id: Optional[str]
    alert_types: list[str]
    min_severity: str
    webhook_url: str
    active: bool
    created_at: str
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_month_year() -> tuple[int, int]:
    """Return (month, year) for the current UTC date."""
    now = datetime.now(timezone.utc)
    return now.month, now.year


def _validate_waterway(waterway_id: str) -> str:
    """Raise HTTP 404 if the waterway ID is not supported."""
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
# GET /alerts/{waterway_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{waterway_id}",
    response_model=list[RiskAlert],
    summary="Get Risk Alerts for a Waterway",
    description=(
        "Return all risk alerts for the specified waterway and time period. "
        "Alerts are generated when a segment's risk score exceeds the "
        "configured threshold or a navigability class transition is detected.\n\n"
        "Alert types:\n"
        "- **DEPTH_CRITICAL** — depth below conditional threshold (< 1.5 m)\n"
        "- **DEPTH_WARNING** — depth below navigable threshold (1.5–3.0 m)\n"
        "- **WIDTH_RESTRICTION** — channel width below navigable minimum (< 50 m)\n"
        "- **SEASONAL_TRANSITION** — navigability class changed vs. previous month\n\n"
        "Results are cached in Redis with a 30-minute TTL."
    ),
    responses={
        200: {
            "description": "List of risk alerts (may be empty if no alerts found).",
        },
    },
)
async def get_alerts_for_waterway(
    waterway_id: WaterwayPathParam,
    month: MonthQuery = None,
    year: YearQuery = None,
    severity: Optional[str] = Query(
        None,
        description=(
            "Filter by severity level: 'low', 'medium', 'high', 'critical'. "
            "If omitted, all severities are returned."
        ),
        pattern=r"^(low|medium|high|critical)$",
    ),
    alert_type: Optional[str] = Query(
        None,
        description=(
            "Filter by alert type: 'DEPTH_CRITICAL', 'DEPTH_WARNING', "
            "'WIDTH_RESTRICTION', 'SEASONAL_TRANSITION'."
        ),
    ),
    threshold: float = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum risk score threshold for alert generation "
            f"(default: {settings.RISK_ALERT_THRESHOLD})."
        ),
    ),
    include_acknowledged: bool = Query(
        True,
        description="Include alerts that have already been acknowledged.",
    ),
    nav_service: NavigabilityService = Depends(get_nav_service),
    alert_svc: AlertService = Depends(get_alert_svc),
) -> list[RiskAlert]:
    """
    Retrieve risk alerts for a National Waterway.

    - **waterway_id**: `NW-1` (Ganga) or `NW-2` (Brahmaputra)
    - **month**: calendar month 1–12 (defaults to current month)
    - **year**: calendar year ≥ 2015 (defaults to current year)
    - **severity**: filter by alert severity level
    - **alert_type**: filter by alert type
    - **threshold**: override the default risk-score threshold
    - **include_acknowledged**: set `false` to hide acknowledged alerts
    """
    _validate_waterway(waterway_id)

    if month is None or year is None:
        cur_month, cur_year = _current_month_year()
        month = month or cur_month
        year = year or cur_year

    effective_threshold = (
        threshold if threshold is not None else settings.RISK_ALERT_THRESHOLD
    )

    try:
        alerts = await nav_service.get_risk_alerts(
            waterway_id=waterway_id,
            month=month,
            year=year,
            threshold=effective_threshold,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Error fetching alerts for %s %02d/%d: %s",
            waterway_id,
            month,
            year,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve alerts: {exc}",
        ) from exc

    # Apply optional client-side filters
    if severity is not None:
        sev_val = severity.lower()
        alerts = [a for a in alerts if str(a.severity).lower() == sev_val]

    if alert_type is not None:
        at_val = alert_type.upper()
        alerts = [a for a in alerts if str(a.alert_type).upper() == at_val]

    if not include_acknowledged:
        alerts = [a for a in alerts if not a.acknowledged]

    logger.info(
        "Returning %d alerts for %s %02d/%d (threshold=%.2f).",
        len(alerts),
        waterway_id,
        month,
        year,
        effective_threshold,
    )
    return alerts


# ---------------------------------------------------------------------------
# GET /alerts/critical
# ---------------------------------------------------------------------------


@router.get(
    "/critical",
    response_model=list[RiskAlert],
    summary="Get All Critical Alerts",
    description=(
        "Return all CRITICAL severity alerts across **all supported waterways** "
        "(NW-1 and NW-2) for the specified month and year. "
        "Useful for the operational control room dashboard.\n\n"
        "Critical alerts indicate segments where:\n"
        "- Depth < 1.5 m (below conditional navigation threshold)\n"
        "- Width < 25 m (severe channel restriction)\n"
        "- Risk score ≥ 0.90"
    ),
)
async def get_critical_alerts(
    month: MonthQuery = None,
    year: YearQuery = None,
    waterway_ids: Optional[str] = Query(
        None,
        description=(
            "Comma-separated list of waterway IDs to include. "
            "Defaults to all supported waterways (NW-1,NW-2)."
        ),
    ),
    alert_svc: AlertService = Depends(get_alert_svc),
    nav_service: NavigabilityService = Depends(get_nav_service),
) -> list[RiskAlert]:
    """
    Retrieve all CRITICAL severity alerts across all waterways.

    - **month**: calendar month 1–12 (defaults to current month)
    - **year**: calendar year ≥ 2015 (defaults to current year)
    - **waterway_ids**: optional comma-separated list (default: all waterways)
    """
    if month is None or year is None:
        cur_month, cur_year = _current_month_year()
        month = month or cur_month
        year = year or cur_year

    # Parse waterway IDs
    if waterway_ids:
        wids = [w.strip() for w in waterway_ids.split(",") if w.strip()]
        invalid = [w for w in wids if w not in settings.SUPPORTED_WATERWAYS]
        if invalid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Unsupported waterway ID(s): {invalid}. "
                    f"Supported: {settings.SUPPORTED_WATERWAYS}"
                ),
            )
    else:
        wids = list(settings.SUPPORTED_WATERWAYS)

    try:
        # First try to retrieve from cache
        critical_from_cache = await alert_svc.get_critical_alerts(
            waterway_ids=wids,
            month=month,
            year=year,
        )

        # If cache is empty, trigger alert generation for each waterway
        if not critical_from_cache:
            all_alerts: list[RiskAlert] = []
            for wid in wids:
                try:
                    wid_alerts = await nav_service.get_risk_alerts(
                        waterway_id=wid,
                        month=month,
                        year=year,
                        threshold=0.90,  # critical threshold only
                    )
                    all_alerts.extend(wid_alerts)
                except Exception as exc:
                    logger.warning(
                        "Failed to fetch critical alerts for %s %02d/%d: %s",
                        wid,
                        month,
                        year,
                        exc,
                    )

            critical_from_cache = [
                a
                for a in all_alerts
                if str(a.severity).lower() == AlertSeverity.CRITICAL.lower()
            ]

        # Sort: waterway first, then by descending risk score
        critical_from_cache.sort(key=lambda a: (a.waterway_id, -a.risk_score))

        logger.info(
            "Returning %d critical alerts across %s for %02d/%d.",
            len(critical_from_cache),
            wids,
            month,
            year,
        )
        return critical_from_cache

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching critical alerts: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve critical alerts: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# GET /alerts/{waterway_id}/summary
# ---------------------------------------------------------------------------


@router.get(
    "/{waterway_id}/summary",
    response_model=AlertSummaryResponse,
    summary="Get Alert Summary",
    description=(
        "Return aggregated alert statistics for a waterway at a given "
        "month/year — total count, breakdown by type and severity, "
        "number of affected segments, and mean/max risk scores."
    ),
)
async def get_alert_summary(
    waterway_id: WaterwayPathParam,
    month: MonthQuery = None,
    year: YearQuery = None,
    nav_service: NavigabilityService = Depends(get_nav_service),
    alert_svc: AlertService = Depends(get_alert_svc),
) -> AlertSummaryResponse:
    """
    Retrieve aggregated alert statistics for a waterway.

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
        alerts = await nav_service.get_risk_alerts(
            waterway_id=waterway_id,
            month=month,
            year=year,
        )
        summary = alert_svc.summarise_alerts(alerts)

        return AlertSummaryResponse(
            waterway_id=waterway_id,
            month=month,
            year=year,
            total=summary["total"],
            by_type=summary["by_type"],
            by_severity=summary["by_severity"],
            acknowledged=summary["acknowledged"],
            unacknowledged=summary["unacknowledged"],
            segments_affected=summary["segments_affected"],
            mean_risk_score=summary["mean_risk_score"],
            max_risk_score=summary["max_risk_score"],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Error computing alert summary for %s %02d/%d: %s",
            waterway_id,
            month,
            year,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to compute alert summary: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# POST /alerts/{alert_id}/acknowledge
# ---------------------------------------------------------------------------


@router.post(
    "/{alert_id}/acknowledge",
    response_model=AcknowledgeResponse,
    status_code=status.HTTP_200_OK,
    summary="Acknowledge an Alert",
    description=(
        "Mark a specific risk alert as acknowledged by an operator. "
        "Acknowledged alerts remain visible but are flagged to prevent "
        "repeated notifications. The acknowledgement is stored in the "
        "Redis cache and persists until the cache entry expires (30 min)."
    ),
)
async def acknowledge_alert(
    alert_id: str = Path(
        ...,
        title="Alert ID",
        description="Deterministic UUID of the alert to acknowledge.",
    ),
    body: AcknowledgeRequest = ...,
    alert_svc: AlertService = Depends(get_alert_svc),
) -> AcknowledgeResponse:
    """
    Acknowledge a risk alert.

    Marks the alert as acknowledged so control room operators and
    downstream subscribers are not repeatedly notified.

    - **alert_id**: the deterministic UUID of the alert
    - **waterway_id**: parent waterway (required to locate the cache entry)
    - **month** / **year**: temporal context of the alert
    """
    _validate_waterway(body.waterway_id)

    try:
        found = await alert_svc.acknowledge_alert(
            alert_id=alert_id,
            waterway_id=body.waterway_id,
            month=body.month,
            year=body.year,
        )

        if not found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Alert '{alert_id}' not found in the cache for "
                    f"{body.waterway_id} {body.month:02d}/{body.year}. "
                    "It may have expired or the ID is incorrect."
                ),
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        return AcknowledgeResponse(
            alert_id=alert_id,
            acknowledged=True,
            acknowledged_at=now_iso,
            message=(
                f"Alert {alert_id} has been acknowledged successfully. "
                f"Acknowledgement recorded at {now_iso}."
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error acknowledging alert %s: %s", alert_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to acknowledge alert: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# GET /alerts/next-month-risk/{segment_id}
# ---------------------------------------------------------------------------


@router.get(
    "/next-month-risk/{segment_id}",
    response_model=NextMonthRiskResponse,
    summary="Forecast Next-Month Risk",
    description=(
        "Forecast the risk score for the next calendar month for a specific "
        "river segment. Uses an exponentially-weighted trend model trained on "
        "historical risk scores. Returns the predicted risk score, trend "
        "direction, confidence interval, and a natural-language explanation.\n\n"
        "A `alert_likely` flag is set when the predicted score exceeds the "
        f"configured threshold ({settings.RISK_ALERT_THRESHOLD})."
    ),
)
async def get_next_month_risk(
    segment_id: str = Path(
        ...,
        title="Segment ID",
        description="Segment identifier, e.g. 'NW-1-042'.",
        pattern=r"^NW-[12]-\d{3,4}$",
        examples=["NW-1-042", "NW-2-107"],
    ),
    history_months: int = Query(
        6,
        ge=1,
        le=24,
        description=(
            "Number of historical months to use for trend estimation (1–24). "
            "Defaults to 6."
        ),
    ),
    nav_service: NavigabilityService = Depends(get_nav_service),
    alert_svc: AlertService = Depends(get_alert_svc),
) -> NextMonthRiskResponse:
    """
    Forecast the risk score for the next month for a river segment.

    Retrieves historical risk scores for the segment by replaying past
    navigability maps and feeds them into the exponential-weighted trend
    model in AlertService.

    - **segment_id**: e.g. `"NW-1-042"`
    - **history_months**: how many past months to consider (1–24)
    """
    # Parse waterway from segment_id
    parts = segment_id.rsplit("-", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid segment_id '{segment_id}'. "
                "Expected format: 'NW-X-NNN', e.g. 'NW-1-042'."
            ),
        )
    waterway_id = parts[0]
    _validate_waterway(waterway_id)

    try:
        # Build a lightweight historical record by pulling recent monthly maps
        now = datetime.now(timezone.utc)
        history: list[dict[str, Any]] = []

        for lag in range(history_months, 0, -1):
            # Walk back month by month
            total_months_back = lag
            target_year = now.year
            target_month = now.month - total_months_back
            while target_month <= 0:
                target_month += 12
                target_year -= 1

            if target_year < 2015:
                continue

            try:
                nav_map = await nav_service.get_navigability_map(
                    waterway_id=waterway_id,
                    month=target_month,
                    year=target_year,
                    force_refresh=False,
                )
                # Find the specific segment's prediction
                seg_pred = next(
                    (p for p in nav_map.predictions if p.segment_id == segment_id),
                    None,
                )
                if seg_pred is not None:
                    history.append(
                        {
                            "month": target_month,
                            "year": target_year,
                            "risk_score": seg_pred.risk_score,
                            "predicted_depth_m": seg_pred.predicted_depth_m,
                            "width_m": seg_pred.width_m,
                            "navigability_class": seg_pred.navigability_class,
                        }
                    )
                elif nav_map.mean_risk_score is not None:
                    # Fallback: use waterway mean if segment not found
                    history.append(
                        {
                            "month": target_month,
                            "year": target_year,
                            "risk_score": nav_map.mean_risk_score,
                        }
                    )
            except Exception as exc:
                logger.debug(
                    "Could not retrieve map for %s %02d/%d: %s",
                    waterway_id,
                    target_month,
                    target_year,
                    exc,
                )

        # Run forecast
        forecast = await alert_svc.predict_next_month_risk(
            segment_id=segment_id,
            history=history,
            n_lag_months=history_months,
        )

        return NextMonthRiskResponse(
            segment_id=forecast["segment_id"],
            forecast_month=forecast["forecast_month"],
            forecast_year=forecast["forecast_year"],
            predicted_risk_score=forecast["predicted_risk_score"],
            trend=forecast["trend"],
            confidence=forecast["confidence"],
            explanation=forecast["explanation"],
            alert_likely=forecast["alert_likely"],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Error forecasting next-month risk for %s: %s", segment_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to compute next-month risk forecast: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# POST /alerts/subscribe  — webhook subscription
# ---------------------------------------------------------------------------


# In-memory subscription store (replace with Redis or DB in production)
_subscriptions: dict[str, dict[str, Any]] = {}


@router.post(
    "/subscribe",
    response_model=WebhookSubscriptionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subscribe to Alert Webhooks",
    description=(
        "Register a webhook endpoint to receive real-time risk alert "
        "notifications. When an alert matching the subscription criteria "
        "is generated, the server will POST the `RiskAlert` payload to "
        "the registered `webhook_url`.\n\n"
        "**Security**: The optional `secret_header` value is sent in the "
        "`X-AIDSTL-Signature` HTTP header so subscribers can verify the "
        "payload origin.\n\n"
        "**Rate limiting**: Webhook calls are throttled to at most 1 "
        "delivery per minute per subscription to prevent flood conditions."
    ),
    responses={
        201: {"description": "Subscription created successfully."},
        400: {"description": "Invalid webhook URL or subscription parameters."},
    },
)
async def subscribe_to_alerts(
    subscription: AlertSubscription,
    background_tasks: BackgroundTasks,
) -> WebhookSubscriptionResponse:
    """
    Register a webhook subscription for risk alert notifications.

    **Request body fields:**
    - `waterway_id` — subscribe to a specific waterway (or `null` for all)
    - `segment_ids` — specific segments to watch (empty = all segments)
    - `alert_types` — alert types to receive (default: all types)
    - `min_severity` — minimum severity level (default: `medium`)
    - `webhook_url` — HTTPS endpoint to POST alert payloads to
    - `secret_header` — optional HMAC secret for payload verification
    """
    # Validate waterway_id if specified
    if subscription.waterway_id is not None:
        wid = str(subscription.waterway_id)
        if wid not in settings.SUPPORTED_WATERWAYS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Waterway '{wid}' is not supported. "
                    f"Supported: {settings.SUPPORTED_WATERWAYS}"
                ),
            )

    # Validate segment IDs if provided
    invalid_segments = []
    for seg_id in subscription.segment_ids:
        parts = seg_id.rsplit("-", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            invalid_segments.append(seg_id)
        elif parts[0] not in settings.SUPPORTED_WATERWAYS:
            invalid_segments.append(seg_id)

    if invalid_segments:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid segment ID(s): {invalid_segments}. "
                "Expected format: 'NW-X-NNN'."
            ),
        )

    # Validate webhook URL (must be HTTP/HTTPS)
    webhook_url_str = str(subscription.webhook_url)
    if not webhook_url_str.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="webhook_url must be a valid HTTP or HTTPS URL.",
        )

    # Generate unique subscription ID
    subscription_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    sub_record: dict[str, Any] = {
        "subscription_id": subscription_id,
        "waterway_id": (
            str(subscription.waterway_id) if subscription.waterway_id else None
        ),
        "segment_ids": subscription.segment_ids,
        "alert_types": [str(at) for at in subscription.alert_types],
        "min_severity": str(subscription.min_severity),
        "webhook_url": webhook_url_str,
        "secret_header": subscription.secret_header,
        "created_at": now_iso,
        "active": True,
        "delivery_count": 0,
        "last_delivered_at": None,
    }

    # Persist subscription (in-memory for demo; use Redis/DB in production)
    _subscriptions[subscription_id] = sub_record

    logger.info(
        "Webhook subscription created: id=%s, url=%s, waterway=%s, severity≥%s",
        subscription_id,
        webhook_url_str,
        sub_record["waterway_id"] or "all",
        sub_record["min_severity"],
    )

    # Schedule a test delivery ping in the background (optional)
    background_tasks.add_task(
        _send_subscription_confirmation_ping,
        subscription_id=subscription_id,
        webhook_url=webhook_url_str,
        secret_header=subscription.secret_header,
    )

    return WebhookSubscriptionResponse(
        subscription_id=subscription_id,
        waterway_id=sub_record["waterway_id"],
        alert_types=sub_record["alert_types"],
        min_severity=sub_record["min_severity"],
        webhook_url=webhook_url_str,
        active=True,
        created_at=now_iso,
        message=(
            f"Subscription '{subscription_id}' created successfully. "
            "A test ping has been scheduled to verify your webhook endpoint."
        ),
    )


# ---------------------------------------------------------------------------
# DELETE /alerts/subscribe/{subscription_id}  — cancel subscription
# ---------------------------------------------------------------------------


@router.delete(
    "/subscribe/{subscription_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel Alert Subscription",
    description=(
        "Deactivate and remove a webhook subscription. "
        "No further alert deliveries will be attempted after this call."
    ),
    responses={
        204: {"description": "Subscription cancelled successfully."},
        404: {"description": "Subscription not found."},
    },
)
async def cancel_subscription(
    subscription_id: str = Path(
        ...,
        title="Subscription ID",
        description="UUID of the subscription to cancel.",
    ),
) -> None:
    """
    Cancel a webhook alert subscription.

    - **subscription_id**: UUID returned when the subscription was created
    """
    if subscription_id not in _subscriptions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Subscription '{subscription_id}' not found. "
                "It may have already been cancelled or the ID is incorrect."
            ),
        )

    del _subscriptions[subscription_id]
    logger.info("Webhook subscription cancelled: %s", subscription_id)
    # Returns 204 No Content (no response body)


# ---------------------------------------------------------------------------
# GET /alerts/subscribe  — list active subscriptions
# ---------------------------------------------------------------------------


@router.get(
    "/subscribe",
    summary="List Active Subscriptions",
    description="Return all active webhook subscriptions (admin use).",
    response_model=list[dict[str, Any]],
)
async def list_subscriptions() -> list[dict[str, Any]]:
    """
    List all active webhook subscriptions.

    Returns the full subscription record for each active entry.
    In production this endpoint should be protected by an admin auth guard.
    """
    active = [
        {k: v for k, v in sub.items() if k != "secret_header"}
        for sub in _subscriptions.values()
        if sub.get("active", True)
    ]
    return active


# ---------------------------------------------------------------------------
# Background helper: subscription confirmation ping
# ---------------------------------------------------------------------------


async def _send_subscription_confirmation_ping(
    subscription_id: str,
    webhook_url: str,
    secret_header: Optional[str],
) -> None:
    """
    Send a lightweight confirmation ping to the subscriber's webhook URL.

    The ping payload contains the subscription ID and a ``"type": "ping"``
    field so the subscriber can identify it as a handshake message rather
    than a real alert.
    """
    import httpx

    payload = {
        "type": "ping",
        "subscription_id": subscription_id,
        "message": (
            "This is a confirmation ping from the AIDSTL Alert Service. "
            "Your webhook endpoint has been registered successfully."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": f"AIDSTL-AlertService/{settings.APP_VERSION}",
    }

    # Add HMAC signature header if secret was provided
    if secret_header:
        import hashlib as _hl
        import hmac

        body_bytes = json.dumps(payload).encode("utf-8")
        sig = hmac.new(
            secret_header.encode("utf-8"), body_bytes, _hl.sha256
        ).hexdigest()
        headers["X-AIDSTL-Signature"] = f"sha256={sig}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook_url,
                json=payload,
                headers=headers,
            )
            if response.status_code < 400:
                logger.info(
                    "Confirmation ping delivered to %s (status=%d).",
                    webhook_url,
                    response.status_code,
                )
                if subscription_id in _subscriptions:
                    _subscriptions[subscription_id]["last_delivered_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    _subscriptions[subscription_id]["delivery_count"] += 1
            else:
                logger.warning(
                    "Confirmation ping to %s failed (status=%d).",
                    webhook_url,
                    response.status_code,
                )
    except Exception as exc:
        logger.warning(
            "Confirmation ping to %s raised an exception: %s",
            webhook_url,
            exc,
        )


# ---------------------------------------------------------------------------
# Internal: deliver alert to all matching subscribers
# ---------------------------------------------------------------------------


async def deliver_alert_to_subscribers(alert: RiskAlert) -> None:
    """
    Deliver a RiskAlert to all matching active webhook subscribers.

    Called internally by the alert generation pipeline when a new alert
    is produced.  Runs concurrently for all matching subscriptions.

    Matching logic
    --------------
    A subscriber matches if ALL of the following are true:
    1. ``waterway_id`` matches (or subscriber has no waterway filter).
    2. ``segment_id`` is in the subscriber's list (or the list is empty).
    3. ``alert.alert_type`` is in the subscriber's ``alert_types``.
    4. ``alert.severity`` is ≥ the subscriber's ``min_severity``.

    Parameters
    ----------
    alert : RiskAlert
        The alert to deliver.
    """
    _sev_rank = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }
    alert_sev_rank = _sev_rank.get(str(alert.severity).lower(), 0)
    alert_type_str = str(alert.alert_type).upper()
    alert_wid = str(alert.waterway_id)

    matching_subs = [
        sub
        for sub in _subscriptions.values()
        if sub.get("active", True)
        and (sub["waterway_id"] is None or sub["waterway_id"] == alert_wid)
        and (not sub["segment_ids"] or alert.segment_id in sub["segment_ids"])
        and alert_type_str in [t.upper() for t in sub.get("alert_types", [])]
        and alert_sev_rank
        >= _sev_rank.get(str(sub.get("min_severity", "low")).lower(), 1)
    ]

    if not matching_subs:
        return

    import httpx

    payload = alert.model_dump(mode="json")

    async def _deliver(sub: dict[str, Any]) -> None:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": f"AIDSTL-AlertService/{settings.APP_VERSION}",
            "X-AIDSTL-Alert-Type": str(alert.alert_type),
            "X-AIDSTL-Severity": str(alert.severity),
        }
        secret = sub.get("secret_header")
        if secret:
            import hashlib as _hl
            import hmac

            body_bytes = json.dumps(payload, default=str).encode("utf-8")
            sig = hmac.new(secret.encode("utf-8"), body_bytes, _hl.sha256).hexdigest()
            headers["X-AIDSTL-Signature"] = f"sha256={sig}"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    sub["webhook_url"],
                    json=payload,
                    headers=headers,
                )
                if resp.status_code < 400:
                    sub["delivery_count"] = sub.get("delivery_count", 0) + 1
                    sub["last_delivered_at"] = datetime.now(timezone.utc).isoformat()
                    logger.debug(
                        "Alert %s delivered to %s (status=%d).",
                        alert.alert_id,
                        sub["webhook_url"],
                        resp.status_code,
                    )
                else:
                    logger.warning(
                        "Alert delivery to %s failed (status=%d): %s",
                        sub["webhook_url"],
                        resp.status_code,
                        resp.text[:200],
                    )
        except Exception as exc:
            logger.warning(
                "Alert delivery to %s raised exception: %s",
                sub["webhook_url"],
                exc,
            )

    await asyncio.gather(*[_deliver(sub) for sub in matching_subs])
