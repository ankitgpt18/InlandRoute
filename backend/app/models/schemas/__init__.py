"""
AIDSTL Project — Schemas Package
=================================
Centralised re-export of all Pydantic v2 data schemas used by the
navigability prediction API.

Import from here instead of the individual modules to keep consumer code
decoupled from the internal package layout.

Example
-------
    from app.models.schemas import (
        NavigabilityPrediction,
        NavigabilityMap,
        SeasonalCalendar,
        RiskAlert,
        PredictionRequest,
    )
"""

from app.models.schemas.navigability import (
    # --- Enumerations ---
    AlertSeverity,
    # --- Webhook / subscription ---
    AlertSubscription,
    AlertType,
    BatchPredictionRequest,
    DepthProfile,
    # --- Depth profile schemas ---
    DepthProfilePoint,
    HistoricalComparison,
    # --- Historical comparison ---
    HistoricalDataPoint,
    MonthlyOutlook,
    # --- Waterway statistics schemas ---
    MonthlyStats,
    NavigabilityClass,
    # --- Aggregated / map schemas ---
    NavigabilityMap,
    # --- Core prediction schema ---
    NavigabilityPrediction,
    PredictionRequest,
    # --- Risk alert schema ---
    RiskAlert,
    # --- River segment schemas ---
    RiverSegmentBase,
    RiverSegmentCreate,
    RiverSegmentResponse,
    Season,
    SeasonalCalendar,
    # --- Prediction request schemas ---
    SegmentFeatureInput,
    SegmentSeasonalOutlook,
    # --- Feature sub-schema ---
    SpectralFeatures,
    # --- Async task status ---
    TaskStatus,
    WaterwayID,
    WaterwayStats,
)

__all__: list[str] = [
    # Enumerations
    "AlertSeverity",
    "AlertType",
    "NavigabilityClass",
    "Season",
    "WaterwayID",
    # River segment schemas
    "RiverSegmentBase",
    "RiverSegmentCreate",
    "RiverSegmentResponse",
    # Feature sub-schema
    "SpectralFeatures",
    # Core prediction schema
    "NavigabilityPrediction",
    # Aggregated / map schemas
    "NavigabilityMap",
    "MonthlyOutlook",
    "SegmentSeasonalOutlook",
    "SeasonalCalendar",
    # Risk alert schema
    "RiskAlert",
    # Depth profile schemas
    "DepthProfilePoint",
    "DepthProfile",
    # Waterway statistics schemas
    "MonthlyStats",
    "WaterwayStats",
    # Prediction request schemas
    "SegmentFeatureInput",
    "PredictionRequest",
    "BatchPredictionRequest",
    # Webhook / subscription
    "AlertSubscription",
    # Historical comparison
    "HistoricalDataPoint",
    "HistoricalComparison",
    # Async task status
    "TaskStatus",
]
