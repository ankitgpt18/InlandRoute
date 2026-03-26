"""
AIDSTL Project — Navigability Pydantic Schemas
===============================================
Data contracts for the navigability prediction API.

Study areas
-----------
  NW-1 : Ganga       — Varanasi → Haldia   (~1,620 km)
  NW-2 : Brahmaputra — Dhubri  → Sadiya    (~891 km)

Navigability classes
--------------------
  navigable     : depth ≥ 3.0 m  AND  width ≥ 50 m
  conditional   : depth ≥ 1.5 m  AND  width ≥ 25 m
  non_navigable : below conditional thresholds

All schemas are Pydantic v2 compatible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NavigabilityClass(str, Enum):
    """Three-class navigability classification per IWAI standards."""

    NAVIGABLE = "navigable"
    CONDITIONAL = "conditional"
    NON_NAVIGABLE = "non_navigable"


class WaterwayID(str, Enum):
    """Supported National Waterways."""

    NW1 = "NW-1"  # Ganga: Varanasi → Haldia
    NW2 = "NW-2"  # Brahmaputra: Dhubri → Sadiya


class AlertType(str, Enum):
    """Risk alert categories."""

    DEPTH_CRITICAL = "DEPTH_CRITICAL"
    DEPTH_WARNING = "DEPTH_WARNING"
    WIDTH_RESTRICTION = "WIDTH_RESTRICTION"
    SEASONAL_TRANSITION = "SEASONAL_TRANSITION"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Season(str, Enum):
    """Indian hydrological seasons."""

    PRE_MONSOON = "pre_monsoon"  # March – May
    MONSOON = "monsoon"  # June – September
    POST_MONSOON = "post_monsoon"  # October – November
    WINTER = "winter"  # December – February


# ---------------------------------------------------------------------------
# Shared configuration mixin
# ---------------------------------------------------------------------------


class _BaseSchema(BaseModel):
    """Common Pydantic config shared across all schemas."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        json_schema_extra={},
    )


# ---------------------------------------------------------------------------
# River Segment schemas
# ---------------------------------------------------------------------------


class RiverSegmentBase(_BaseSchema):
    """Core attributes shared by all river-segment representations."""

    segment_id: Annotated[
        str,
        Field(
            ...,
            min_length=3,
            max_length=32,
            pattern=r"^NW-[12]-\d{3,4}$",
            description=(
                "Unique segment identifier following the pattern "
                "'NW-{waterway}-{sequence}', e.g. 'NW-1-042'."
            ),
            examples=["NW-1-042", "NW-2-107"],
        ),
    ]
    waterway_id: Annotated[
        WaterwayID,
        Field(..., description="Parent National Waterway identifier."),
    ]
    segment_index: Annotated[
        int,
        Field(..., ge=1, description="1-based sequential index of the segment."),
    ]
    chainage_start_km: Annotated[
        float,
        Field(
            ..., ge=0.0, description="Chainage at the upstream end of the segment (km)."
        ),
    ]
    chainage_end_km: Annotated[
        float,
        Field(
            ...,
            gt=0.0,
            description="Chainage at the downstream end of the segment (km).",
        ),
    ]
    length_km: Annotated[
        float,
        Field(
            ...,
            gt=0.0,
            le=10.0,
            description="Segment length in kilometres (target: 5 km).",
        ),
    ]
    geometry: Annotated[
        dict[str, Any],
        Field(
            ...,
            description="GeoJSON geometry (LineString) of the segment centreline.",
        ),
    ]

    @field_validator("geometry")
    @classmethod
    def validate_geojson_geometry(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure the geometry field is a valid GeoJSON geometry object."""
        allowed_types = {
            "Point",
            "LineString",
            "Polygon",
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
            "GeometryCollection",
        }
        if "type" not in v:
            raise ValueError("GeoJSON geometry must contain a 'type' field.")
        if v["type"] not in allowed_types:
            raise ValueError(
                f"GeoJSON geometry type '{v['type']}' is not recognised. "
                f"Expected one of: {sorted(allowed_types)}"
            )
        if "coordinates" not in v and v["type"] != "GeometryCollection":
            raise ValueError("GeoJSON geometry must contain a 'coordinates' field.")
        return v

    @model_validator(mode="after")
    def validate_chainage_order(self) -> "RiverSegmentBase":
        if self.chainage_end_km <= self.chainage_start_km:
            raise ValueError(
                "chainage_end_km must be strictly greater than chainage_start_km."
            )
        return self


class RiverSegmentCreate(RiverSegmentBase):
    """Schema used when inserting a new river segment into the database."""

    sinuosity: Annotated[
        float,
        Field(
            1.0,
            ge=1.0,
            description=(
                "Sinuosity index of the segment (arc-length / chord-length). "
                "Straight channels have sinuosity ≈ 1.0."
            ),
        ),
    ]
    bed_material: Annotated[
        str | None,
        Field(
            None, description="Dominant bed material (e.g. 'sand', 'gravel', 'silt')."
        ),
    ]
    gauge_station_id: Annotated[
        str | None,
        Field(None, description="ID of the nearest CWC gauge station, if any."),
    ]
    metadata: Annotated[
        dict[str, Any],
        Field(default_factory=dict, description="Arbitrary key-value metadata."),
    ]


class RiverSegmentResponse(RiverSegmentBase):
    """Full river segment representation returned by the API."""

    sinuosity: float = Field(1.0, ge=1.0)
    bed_material: str | None = None
    gauge_station_id: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of record creation.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of last update.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Spectral features sub-schema
# ---------------------------------------------------------------------------


class SpectralFeatures(_BaseSchema):
    """
    Sentinel-2 derived spectral indices and band statistics for a segment.

    All indices are dimensionless unless stated otherwise.
    """

    # Water indices
    mndwi: float | None = Field(
        None, ge=-1.0, le=1.0, description="Modified Normalised Difference Water Index."
    )
    ndwi: float | None = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Normalised Difference Water Index (McFeeters).",
    )
    awei_sh: float | None = Field(
        None, description="Automated Water Extraction Index (shadow-resistant)."
    )
    awei_ns: float | None = Field(
        None, description="Automated Water Extraction Index (no-shadow)."
    )

    # Depth proxy
    stumpf_ratio: float | None = Field(
        None, description="Stumpf log-ratio bathymetric index (blue/green)."
    )

    # Turbidity proxy
    turbidity_index: float | None = Field(
        None, ge=0.0, description="Turbidity index derived from red and green bands."
    )

    # Raw band medians (surface reflectance, scale factor 10000)
    b2_blue: float | None = Field(
        None, ge=0.0, description="Sentinel-2 Band 2 (Blue, 490 nm) median reflectance."
    )
    b3_green: float | None = Field(
        None,
        ge=0.0,
        description="Sentinel-2 Band 3 (Green, 560 nm) median reflectance.",
    )
    b4_red: float | None = Field(
        None, ge=0.0, description="Sentinel-2 Band 4 (Red, 665 nm) median reflectance."
    )
    b8_nir: float | None = Field(
        None, ge=0.0, description="Sentinel-2 Band 8 (NIR, 842 nm) median reflectance."
    )
    b11_swir1: float | None = Field(
        None,
        ge=0.0,
        description="Sentinel-2 Band 11 (SWIR-1, 1610 nm) median reflectance.",
    )
    b12_swir2: float | None = Field(
        None,
        ge=0.0,
        description="Sentinel-2 Band 12 (SWIR-2, 2190 nm) median reflectance.",
    )

    # Texture / statistical features
    ndvi: float | None = Field(
        None, ge=-1.0, le=1.0, description="Normalised Difference Vegetation Index."
    )
    evi: float | None = Field(None, description="Enhanced Vegetation Index.")
    water_pixel_fraction: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of pixels classified as water within the segment buffer.",
    )

    # Temporal features
    ndwi_trend_3m: float | None = Field(
        None, description="Linear trend in MNDWI over the preceding 3 months."
    )
    ndwi_anomaly: float | None = Field(
        None, description="MNDWI anomaly relative to the long-term monthly climatology."
    )

    # Additional raw features
    extra: dict[str, float] = Field(
        default_factory=dict,
        description="Additional computed features not covered by the named fields above.",
    )


# ---------------------------------------------------------------------------
# Core prediction schema
# ---------------------------------------------------------------------------


class NavigabilityPrediction(_BaseSchema):
    """
    Full navigability prediction for a single 5-km river segment.

    This is the primary output of the TFT + Swin Transformer ensemble model
    combined with the downstream navigability classifier.
    """

    # --- Identification ---
    segment_id: Annotated[
        str,
        Field(..., description="Unique segment identifier (e.g. 'NW-1-042')."),
    ]
    waterway_id: Annotated[
        WaterwayID,
        Field(..., description="Parent National Waterway ('NW-1' or 'NW-2')."),
    ]
    geometry: Annotated[
        dict[str, Any],
        Field(..., description="GeoJSON geometry of the segment centreline."),
    ]

    # --- Temporal context ---
    month: Annotated[
        int,
        Field(..., ge=1, le=12, description="Calendar month of the prediction (1–12)."),
    ]
    year: Annotated[
        int,
        Field(..., ge=2015, le=2100, description="Calendar year of the prediction."),
    ]

    # --- Depth prediction (TFT ensemble output) ---
    predicted_depth_m: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            le=50.0,
            description="Point estimate of water depth in metres.",
        ),
    ]
    depth_lower_ci: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            description="Lower bound of the 90% credible interval for depth (m).",
        ),
    ]
    depth_upper_ci: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            description="Upper bound of the 90% credible interval for depth (m).",
        ),
    ]

    # --- Width estimate (Swin Transformer water-extent output) ---
    width_m: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            le=20_000.0,
            description="Estimated channel width in metres derived from water-mask segmentation.",
        ),
    ]

    # --- Classification ---
    navigability_class: Annotated[
        NavigabilityClass,
        Field(
            ...,
            description="Navigability class: navigable | conditional | non_navigable.",
        ),
    ]
    navigability_probability: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            le=1.0,
            description="Probability assigned to the predicted navigability class.",
        ),
    ]

    # --- Uncertainty & risk ---
    confidence: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            le=1.0,
            description=(
                "Overall prediction confidence score (0–1). "
                "Accounts for model uncertainty, data availability, and cloud cover."
            ),
        ),
    ]
    risk_score: Annotated[
        float,
        Field(
            ...,
            ge=0.0,
            le=1.0,
            description=(
                "Composite risk score (0 = no risk, 1 = maximum risk). "
                "Higher scores indicate higher probability of non-navigability."
            ),
        ),
    ]

    # --- Class probability breakdown ---
    class_probabilities: Annotated[
        dict[str, float],
        Field(
            default_factory=lambda: {
                "navigable": 0.0,
                "conditional": 0.0,
                "non_navigable": 0.0,
            },
            description="Softmax probabilities for each navigability class.",
        ),
    ]

    # --- Feature inputs used for this prediction ---
    features: Annotated[
        SpectralFeatures,
        Field(
            default_factory=SpectralFeatures,
            description="Spectral and spatial features used as model inputs.",
        ),
    ]

    # --- Explainability ---
    shap_values: Annotated[
        dict[str, float] | None,
        Field(
            None,
            description="SHAP feature-contribution values (optional, may be None for batch calls).",
        ),
    ]

    # --- Metadata ---
    model_version: Annotated[
        str,
        Field(
            "1.0.0",
            description="Version of the ML model that produced this prediction.",
        ),
    ]
    data_date: Annotated[
        datetime | None,
        Field(None, description="Acquisition date of the Sentinel-2 composite used."),
    ]
    cloud_cover_pct: Annotated[
        float | None,
        Field(
            None,
            ge=0.0,
            le=100.0,
            description="Cloud cover percentage over the segment for the given month.",
        ),
    ]
    generated_at: Annotated[
        datetime,
        Field(
            default_factory=lambda: datetime.now(timezone.utc),
            description="UTC timestamp at which this prediction was generated.",
        ),
    ]

    # --- Derived / convenience properties ---
    @property
    def depth_uncertainty_m(self) -> float:
        """Width of the 90% credible interval (metres)."""
        return self.depth_upper_ci - self.depth_lower_ci

    @property
    def season(self) -> Season:
        """Map the calendar month to an Indian hydrological season."""
        if self.month in (3, 4, 5):
            return Season.PRE_MONSOON
        if self.month in (6, 7, 8, 9):
            return Season.MONSOON
        if self.month in (10, 11):
            return Season.POST_MONSOON
        return Season.WINTER  # 12, 1, 2

    @model_validator(mode="after")
    def validate_ci_ordering(self) -> "NavigabilityPrediction":
        """Ensure the credible interval is well-ordered."""
        if self.depth_lower_ci > self.predicted_depth_m:
            raise ValueError("depth_lower_ci must not exceed predicted_depth_m.")
        if self.depth_upper_ci < self.predicted_depth_m:
            raise ValueError("depth_upper_ci must not be less than predicted_depth_m.")
        return self

    @model_validator(mode="after")
    def validate_class_probabilities_sum(self) -> "NavigabilityPrediction":
        """Ensure class probabilities approximately sum to 1.0."""
        total = sum(self.class_probabilities.values())
        if self.class_probabilities and not (0.98 <= total <= 1.02):
            raise ValueError(f"class_probabilities must sum to 1.0 (got {total:.4f}).")
        return self


# ---------------------------------------------------------------------------
# Aggregated / collection schemas
# ---------------------------------------------------------------------------


class NavigabilityMap(_BaseSchema):
    """
    Complete navigability map for a waterway at a given month/year.

    Contains predictions for every 5-km segment along the waterway.
    """

    waterway_id: WaterwayID
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2015, le=2100)

    predictions: list[NavigabilityPrediction] = Field(
        ...,
        description="Ordered list of segment predictions (upstream → downstream).",
    )

    # Summary statistics computed from predictions
    total_segments: int = Field(
        ..., ge=0, description="Total number of segments in this waterway."
    )
    navigable_count: int = Field(0, ge=0)
    conditional_count: int = Field(0, ge=0)
    non_navigable_count: int = Field(0, ge=0)

    navigable_length_km: float = Field(
        0.0, ge=0.0, description="Total length of navigable segments (km)."
    )
    conditional_length_km: float = Field(0.0, ge=0.0)
    non_navigable_length_km: float = Field(0.0, ge=0.0)

    mean_depth_m: float | None = Field(
        None, description="Mean predicted depth across all segments (m)."
    )
    mean_width_m: float | None = Field(
        None, description="Mean estimated width across all segments (m)."
    )
    mean_risk_score: float | None = Field(None, ge=0.0, le=1.0)

    overall_navigability_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of waterway length classified as navigable.",
    )

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cache_expires_at: datetime | None = Field(
        None, description="UTC expiry time of the cached response."
    )

    model_config = ConfigDict(use_enum_values=True)


class MonthlyOutlook(_BaseSchema):
    """Navigability outlook for a single segment in a given month."""

    month: int = Field(..., ge=1, le=12)
    month_name: str = Field(..., description="Full month name, e.g. 'January'.")
    season: Season

    predicted_depth_m: float = Field(..., ge=0.0)
    depth_lower_ci: float = Field(..., ge=0.0)
    depth_upper_ci: float = Field(..., ge=0.0)
    width_m: float = Field(..., ge=0.0)

    navigability_class: NavigabilityClass
    navigability_probability: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)

    is_historically_navigable: bool | None = Field(
        None,
        description="Whether this month is typically navigable based on historical data.",
    )


class SegmentSeasonalOutlook(_BaseSchema):
    """12-month navigability outlook for a single river segment."""

    segment_id: str
    waterway_id: WaterwayID
    year: int = Field(..., ge=2015, le=2100)
    monthly_outlooks: list[MonthlyOutlook] = Field(
        ...,
        min_length=12,
        max_length=12,
        description="Ordered list of monthly outlooks (January → December).",
    )
    navigable_months: list[int] = Field(
        default_factory=list,
        description="List of month numbers (1–12) in which the segment is predicted navigable.",
    )
    conditional_months: list[int] = Field(default_factory=list)
    non_navigable_months: list[int] = Field(default_factory=list)

    annual_navigability_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Fraction of the year the segment is predicted navigable (%).",
    )
    peak_navigability_month: int | None = Field(
        None,
        ge=1,
        le=12,
        description="Month with the highest predicted depth.",
    )
    lowest_navigability_month: int | None = Field(
        None,
        ge=1,
        le=12,
        description="Month with the lowest predicted depth.",
    )


class SeasonalCalendar(_BaseSchema):
    """
    Annual 12-month navigability calendar for an entire waterway.

    Aggregates per-segment seasonal outlooks into a waterway-level view,
    enabling operational planning across the full navigation season.
    """

    waterway_id: WaterwayID
    year: int = Field(..., ge=2015, le=2100)

    segment_outlooks: list[SegmentSeasonalOutlook] = Field(
        ...,
        description="Seasonal outlook for each 5-km segment (upstream → downstream).",
    )

    # Waterway-level monthly summary
    monthly_navigable_pct: dict[int, float] = Field(
        default_factory=dict,
        description=(
            "Mapping of month number → percentage of waterway length that is navigable. "
            "Keys are 1–12."
        ),
    )

    best_navigation_months: list[int] = Field(
        default_factory=list,
        description="Month numbers (1–12) where ≥80% of waterway is navigable.",
    )
    peak_season_start_month: int | None = Field(None, ge=1, le=12)
    peak_season_end_month: int | None = Field(None, ge=1, le=12)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Risk alert schema
# ---------------------------------------------------------------------------


class RiskAlert(_BaseSchema):
    """
    A risk alert generated when a segment's risk score exceeds a threshold
    or a navigability class transition is detected.
    """

    alert_id: Annotated[
        str,
        Field(..., description="Unique alert identifier (UUID v4)."),
    ]
    waterway_id: WaterwayID
    segment_id: str
    alert_type: AlertType
    severity: AlertSeverity

    title: str = Field(
        ..., max_length=200, description="Short human-readable alert title."
    )
    description: str = Field(
        ..., max_length=2000, description="Detailed alert description."
    )

    current_depth_m: float | None = Field(None, ge=0.0)
    threshold_depth_m: float | None = Field(None, ge=0.0)
    current_width_m: float | None = Field(None, ge=0.0)
    threshold_width_m: float | None = Field(None, ge=0.0)

    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_trend: Literal["increasing", "stable", "decreasing"] = "stable"

    affected_month: int = Field(..., ge=1, le=12)
    affected_year: int = Field(..., ge=2015, le=2100)

    previous_class: NavigabilityClass | None = None
    current_class: NavigabilityClass

    recommended_action: str | None = Field(
        None,
        description="Operational recommendation for fleet operators or harbour masters.",
    )

    geometry: dict[str, Any] = Field(
        ..., description="GeoJSON point or line geometry of the affected segment."
    )

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = Field(
        None, description="UTC time after which this alert is considered stale."
    )
    acknowledged: bool = False
    acknowledged_at: datetime | None = None


# ---------------------------------------------------------------------------
# Depth profile schema
# ---------------------------------------------------------------------------


class DepthProfilePoint(_BaseSchema):
    """Single point on a longitudinal depth profile."""

    segment_id: str
    chainage_km: float = Field(
        ..., ge=0.0, description="Distance from waterway origin (km)."
    )
    predicted_depth_m: float = Field(..., ge=0.0)
    depth_lower_ci: float = Field(..., ge=0.0)
    depth_upper_ci: float = Field(..., ge=0.0)
    navigability_class: NavigabilityClass
    risk_score: float = Field(..., ge=0.0, le=1.0)
    geometry: dict[str, Any]  # GeoJSON Point at segment midpoint


class DepthProfile(_BaseSchema):
    """
    Longitudinal depth profile of a waterway for a given month/year.

    Provides a continuous view of predicted depths along the entire
    waterway from origin to terminus, useful for route planning.
    """

    waterway_id: WaterwayID
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2015, le=2100)

    profile_points: list[DepthProfilePoint] = Field(
        ..., description="Depth profile ordered by increasing chainage."
    )

    total_length_km: float = Field(..., ge=0.0)
    min_depth_m: float = Field(..., ge=0.0)
    max_depth_m: float = Field(..., ge=0.0)
    mean_depth_m: float = Field(..., ge=0.0)

    # Critical shoal / bottleneck information
    critical_segments: list[str] = Field(
        default_factory=list,
        description="Segment IDs where depth is below the navigable threshold (< 3.0 m).",
    )
    navigable_stretch_km: float = Field(0.0, ge=0.0)

    # Reference lines for visualisation
    navigable_threshold_m: float = 3.0
    conditional_threshold_m: float = 1.5

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Waterway statistics schema
# ---------------------------------------------------------------------------


class MonthlyStats(_BaseSchema):
    """Summary statistics for a waterway in a single calendar month."""

    month: int = Field(..., ge=1, le=12)
    navigable_pct: float = Field(..., ge=0.0, le=100.0)
    conditional_pct: float = Field(..., ge=0.0, le=100.0)
    non_navigable_pct: float = Field(..., ge=0.0, le=100.0)
    mean_depth_m: float
    mean_width_m: float
    mean_risk_score: float = Field(..., ge=0.0, le=1.0)
    alert_count: int = Field(0, ge=0)


class WaterwayStats(_BaseSchema):
    """
    Annual operational statistics for a National Waterway.

    Provides a comprehensive summary of navigability conditions,
    depth trends, and risk metrics over a full calendar year.
    """

    waterway_id: WaterwayID
    year: int = Field(..., ge=2015, le=2100)

    total_length_km: float = Field(..., gt=0.0)
    total_segments: int = Field(..., gt=0)

    # Annual averages
    annual_navigable_pct: float = Field(..., ge=0.0, le=100.0)
    annual_mean_depth_m: float
    annual_mean_width_m: float
    annual_mean_risk_score: float = Field(..., ge=0.0, le=1.0)

    # Monthly breakdown
    monthly_stats: list[MonthlyStats] = Field(..., min_length=12, max_length=12)

    # Best/worst months
    best_month: int = Field(
        ..., ge=1, le=12, description="Month with highest navigable percentage."
    )
    worst_month: int = Field(
        ..., ge=1, le=12, description="Month with lowest navigable percentage."
    )

    # Year-on-year change (populated when historical data is available)
    yoy_depth_change_m: float | None = Field(
        None,
        description="Mean depth change vs. previous year (positive = improvement).",
    )
    yoy_navigable_pct_change: float | None = Field(
        None, description="Change in annual navigable percentage vs. previous year."
    )

    # Alert counts
    total_alerts: int = Field(0, ge=0)
    critical_alerts: int = Field(0, ge=0)

    # Segment-level extremes
    deepest_segment_id: str | None = None
    shallowest_segment_id: str | None = None

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Prediction request schemas
# ---------------------------------------------------------------------------


class SegmentFeatureInput(_BaseSchema):
    """Input features for a single segment prediction request."""

    segment_id: str = Field(..., description="Segment to predict navigability for.")
    waterway_id: WaterwayID
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2015, le=2100)

    # Optional pre-computed spectral features (if None, fetched from GEE)
    features: SpectralFeatures | None = Field(
        None,
        description=(
            "Pre-computed spectral features. If omitted, the API will fetch "
            "them from Google Earth Engine for the requested month/year."
        ),
    )

    # Optional geometry override
    geometry: dict[str, Any] | None = Field(
        None,
        description="GeoJSON geometry override. If omitted, the stored geometry is used.",
    )

    # Hydrological ancillary inputs (optional — improve depth accuracy)
    gauge_discharge_m3s: float | None = Field(
        None,
        ge=0.0,
        description="River discharge at the nearest gauge station (m³/s).",
    )
    gauge_water_level_m: float | None = Field(
        None,
        ge=0.0,
        description="Water surface elevation at the nearest gauge station (m).",
    )
    precipitation_mm: float | None = Field(
        None,
        ge=0.0,
        description="Accumulated precipitation over the past 30 days (mm).",
    )


class PredictionRequest(_BaseSchema):
    """
    Request body for a single-segment navigability prediction.

    POST /api/v1/navigability/predict
    """

    segment: SegmentFeatureInput
    return_shap: bool = Field(
        False,
        description="If True, include SHAP feature-importance values in the response.",
    )
    return_features: bool = Field(
        True,
        description="If True, include the spectral feature vector in the response.",
    )
    force_refresh: bool = Field(
        False,
        description="If True, bypass the Redis cache and recompute the prediction.",
    )


class BatchPredictionRequest(_BaseSchema):
    """
    Request body for batched multi-segment navigability predictions.

    POST /api/v1/navigability/predict/batch

    Notes
    -----
    Large batches (> 50 segments) are automatically routed to a Celery
    background task. The response will contain a ``task_id`` that can
    be polled via GET /api/v1/tasks/{task_id}.
    """

    segments: Annotated[
        list[SegmentFeatureInput],
        Field(
            ...,
            min_length=1,
            max_length=500,
            description="List of segment feature inputs.",
        ),
    ]
    return_shap: bool = False
    return_features: bool = True
    force_refresh: bool = False
    async_threshold: int = Field(
        50,
        ge=1,
        description=(
            "Number of segments above which the batch is processed asynchronously "
            "via Celery. Synchronous responses are guaranteed for batches ≤ this value."
        ),
    )


# ---------------------------------------------------------------------------
# Webhook / subscription schemas
# ---------------------------------------------------------------------------


class AlertSubscription(_BaseSchema):
    """Webhook subscription for real-time risk alerts."""

    subscription_id: str | None = Field(
        None, description="Assigned by the server on creation."
    )
    waterway_id: WaterwayID | None = Field(
        None,
        description="Subscribe to alerts for a specific waterway. None = all waterways.",
    )
    segment_ids: list[str] = Field(
        default_factory=list,
        description="Specific segment IDs to subscribe to. Empty list = all segments.",
    )
    alert_types: list[AlertType] = Field(
        default_factory=lambda: list(AlertType),
        description="Alert types to receive. Defaults to all types.",
    )
    min_severity: AlertSeverity = Field(
        AlertSeverity.MEDIUM,
        description="Minimum severity level to trigger a webhook call.",
    )
    webhook_url: Annotated[
        str,
        Field(..., description="HTTPS endpoint to POST alert payloads to."),
    ]
    secret_header: str | None = Field(
        None,
        description=(
            "If provided, this value is sent in the X-AIDSTL-Signature header "
            "so the subscriber can verify the payload origin."
        ),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True


# ---------------------------------------------------------------------------
# Historical comparison schema
# ---------------------------------------------------------------------------


class HistoricalDataPoint(_BaseSchema):
    """A single data point in a historical time series."""

    year: int
    month: int
    predicted_depth_m: float
    width_m: float
    navigability_class: NavigabilityClass
    risk_score: float


class HistoricalComparison(_BaseSchema):
    """
    Comparison of current predictions against historical observations
    for the same waterway and month.
    """

    waterway_id: WaterwayID
    month: int = Field(..., ge=1, le=12)
    current_year: int

    current_mean_depth_m: float
    historical_mean_depth_m: float
    depth_anomaly_m: float = Field(
        ...,
        description="Current depth minus historical mean (positive = above average).",
    )
    depth_anomaly_pct: float = Field(
        ...,
        description="Depth anomaly expressed as a percentage of the historical mean.",
    )

    current_navigable_pct: float = Field(..., ge=0.0, le=100.0)
    historical_navigable_pct: float = Field(..., ge=0.0, le=100.0)

    historical_series: list[HistoricalDataPoint] = Field(
        default_factory=list,
        description="Year-by-year data points used to compute the historical baseline.",
    )

    trend_direction: Literal["improving", "stable", "deteriorating"] = "stable"
    trend_significance: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Statistical significance of the detected trend (p-value inverted: 1 = highly significant).",
    )

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Task / async job schema
# ---------------------------------------------------------------------------


class TaskStatus(_BaseSchema):
    """Status of a background Celery task (used for large batch predictions)."""

    task_id: str
    status: Literal["PENDING", "STARTED", "PROGRESS", "SUCCESS", "FAILURE"] = "PENDING"
    progress_pct: float = Field(0.0, ge=0.0, le=100.0)
    message: str = ""
    result_url: str | None = Field(
        None,
        description="URL to retrieve the task result once status is SUCCESS.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: str | None = None
