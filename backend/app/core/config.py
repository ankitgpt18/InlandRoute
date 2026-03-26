"""
AIDSTL API — Application Configuration
=======================================
Centralised settings management using pydantic-settings.
All values can be overridden via environment variables or a .env file.

Study areas
-----------
  NW-1 : Ganga       — Varanasi → Haldia   (~1,620 km)
  NW-2 : Brahmaputra — Dhubri  → Sadiya    (~891 km)
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AnyHttpUrl, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Project root helpers
# ---------------------------------------------------------------------------

# Resolve project root as two levels up from this file
# (backend/app/core/config.py  →  backend/)
_HERE = Path(__file__).resolve().parent  # backend/app/core
_APP_ROOT = _HERE.parent  # backend/app
_BACKEND_ROOT = _APP_ROOT.parent  # backend


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables / .env file.

    Precedence (highest → lowest):
        1. Actual environment variables
        2. Variables in the .env file
        3. Default values defined below
    """

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # silently discard unknown env vars
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_NAME: str = "AIDSTL API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = Field(
        "development", pattern="^(development|staging|production)$"
    )

    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_hex(64),
        description="JWT signing secret — MUST be set explicitly in production.",
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ------------------------------------------------------------------
    # Database  (PostgreSQL + PostGIS)
    # ------------------------------------------------------------------
    DATABASE_URL: str = Field(
        "postgresql+asyncpg://aidstl_user:aidstl_password@localhost:5432/aidstl_db",
        description="Async SQLAlchemy DSN (asyncpg driver).",
    )
    SYNC_DATABASE_URL: str = Field(
        "postgresql+psycopg2://aidstl_user:aidstl_password@localhost:5432/aidstl_db",
        description="Synchronous DSN used by Alembic migrations.",
    )

    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30  # seconds
    DB_POOL_RECYCLE: int = 1800  # seconds — avoids stale connections

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 21_600  # 6 hours — prediction cache lifetime

    # ------------------------------------------------------------------
    # Celery
    # ------------------------------------------------------------------
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_WORKER_CONCURRENCY: int = 4
    CELERY_MAX_TASKS_PER_CHILD: int = 100

    # ------------------------------------------------------------------
    # Google Earth Engine (GEE)
    # ------------------------------------------------------------------
    GEE_SERVICE_ACCOUNT: str = Field(
        "",
        description="Service-account email registered with Google Earth Engine.",
    )
    GEE_KEY_FILE: str = Field(
        "/app/secrets/gee_service_account_key.json",
        description="Path to GEE private-key JSON file.",
    )
    GEE_PROJECT_ID: str = ""
    GEE_S2_COLLECTION: str = "COPERNICUS/S2_SR_HARMONIZED"
    GEE_CLOUD_THRESHOLD: int = Field(20, ge=0, le=100)  # percent
    GEE_MAX_PIXELS: float = 1e9

    # ------------------------------------------------------------------
    # ML Model Paths
    # ------------------------------------------------------------------
    MODEL_DIR: str = "./ml/models/saved"
    ENSEMBLE_MODEL_PATH: str = "./ml/models/saved/ensemble/tft_swin_ensemble.pt"
    NAVIGABILITY_MODEL_PATH: str = (
        "./ml/models/saved/classifier/navigability_classifier.joblib"
    )
    FEATURE_SCALER_PATH: str = "./ml/models/saved/preprocessors/feature_scaler.joblib"
    SHAP_EXPLAINER_PATH: str = "./ml/models/saved/explainers/shap_explainer.joblib"

    # PyTorch inference device: "cpu" | "cuda" | "cuda:0" etc.
    TORCH_DEVICE: str = "cpu"
    INFERENCE_BATCH_SIZE: int = 32

    # ------------------------------------------------------------------
    # Navigability Thresholds
    # ------------------------------------------------------------------
    # Per IWAI (Inland Waterways Authority of India) standards
    DEPTH_NAVIGABLE_MIN: float = 3.0  # metres — minimum for "Navigable"
    DEPTH_CONDITIONAL_MIN: float = 1.5  # metres — minimum for "Conditional"
    WIDTH_NAVIGABLE_MIN: float = 50.0  # metres — minimum for "Navigable"
    WIDTH_CONDITIONAL_MIN: float = 25.0  # metres — minimum for "Conditional"
    RISK_ALERT_THRESHOLD: float = Field(0.7, ge=0.0, le=1.0)

    # ------------------------------------------------------------------
    # Study Area Configuration
    # ------------------------------------------------------------------
    SUPPORTED_WATERWAYS: list[str] = ["NW-1", "NW-2"]
    SEGMENT_LENGTH_KM: float = 5.0

    # Bounding boxes: [minLon, minLat, maxLon, maxLat]
    NW1_BBOX: list[float] = [83.0, 21.5, 88.5, 25.5]  # Ganga: Varanasi–Haldia
    NW2_BBOX: list[float] = [89.5, 26.5, 95.8, 28.2]  # Brahmaputra: Dhubri–Sadiya

    CRS_EPSG: int = 4326

    # ------------------------------------------------------------------
    # External APIs
    # ------------------------------------------------------------------
    MAPBOX_TOKEN: str = ""
    OPENWEATHER_API_KEY: str = ""
    CWC_API_ENDPOINT: str = "https://cwc.gov.in/api/v1"
    CWC_API_KEY: str = ""

    # ------------------------------------------------------------------
    # AWS S3 (optional artefact / data storage)
    # ------------------------------------------------------------------
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET_NAME: str = "aidstl-data"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    S3_MODEL_PREFIX: str = "models/saved"

    # ------------------------------------------------------------------
    # CORS & Security
    # ------------------------------------------------------------------
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8080"]
    ALLOW_CREDENTIALS: bool = True
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1"]

    # ------------------------------------------------------------------
    # Logging & Observability
    # ------------------------------------------------------------------
    LOG_LEVEL: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_FORMAT: str = Field("json", pattern="^(json|console)$")
    LOG_FILE: str = ""
    ENABLE_METRICS: bool = True
    SENTRY_DSN: str = ""

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = Field(8000, ge=1, le=65535)
    WORKERS: int = Field(4, ge=1)
    KEEP_ALIVE: int = 5
    REQUEST_TIMEOUT: int = 120

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure the async driver is used for SQLAlchemy."""
        if v and "postgresql://" in v and "asyncpg" not in v:
            # Transparently upgrade plain postgresql:// to asyncpg variant
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Any) -> list[str]:
        """Accept a comma-separated string OR a JSON list from the environment."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("SUPPORTED_WATERWAYS", mode="before")
    @classmethod
    def parse_waterways(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [w.strip() for w in v.split(",") if w.strip()]
        return v

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [h.strip() for h in v.split(",") if h.strip()]
        return v

    @field_validator("NW1_BBOX", "NW2_BBOX", mode="before")
    @classmethod
    def parse_bbox(cls, v: Any) -> list[float]:
        """Accept comma-separated string or list."""
        if isinstance(v, str):
            parts = [float(x.strip()) for x in v.split(",")]
            if len(parts) != 4:
                raise ValueError(
                    "Bounding box must have exactly 4 values: minLon,minLat,maxLon,maxLat"
                )
            return parts
        return v

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Enforce stricter constraints when running in production."""
        if self.ENVIRONMENT == "production":
            if not self.GEE_SERVICE_ACCOUNT:
                raise ValueError(
                    "GEE_SERVICE_ACCOUNT must be set in production environment."
                )
            if not self.MAPBOX_TOKEN:
                raise ValueError("MAPBOX_TOKEN must be set in production environment.")
            if self.SECRET_KEY and len(self.SECRET_KEY) < 64:
                raise ValueError(
                    "SECRET_KEY must be at least 64 characters long in production."
                )
        return self

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def waterway_bboxes(self) -> dict[str, list[float]]:
        """Map waterway IDs to their bounding boxes."""
        return {
            "NW-1": self.NW1_BBOX,
            "NW-2": self.NW2_BBOX,
        }

    @property
    def model_dir_path(self) -> Path:
        """Resolve MODEL_DIR to an absolute Path."""
        p = Path(self.MODEL_DIR)
        return p if p.is_absolute() else _BACKEND_ROOT / p

    @property
    def gee_key_file_path(self) -> Path:
        """Resolve GEE_KEY_FILE to a Path."""
        return Path(self.GEE_KEY_FILE)

    def get_bbox_for_waterway(self, waterway_id: str) -> list[float]:
        """
        Return the bounding box for a given waterway ID.

        Parameters
        ----------
        waterway_id : str
            One of "NW-1" or "NW-2".

        Returns
        -------
        list[float]
            [minLon, minLat, maxLon, maxLat]

        Raises
        ------
        ValueError
            If the waterway_id is not recognised.
        """
        bboxes = self.waterway_bboxes
        if waterway_id not in bboxes:
            raise ValueError(
                f"Unknown waterway '{waterway_id}'. "
                f"Supported waterways: {list(bboxes.keys())}"
            )
        return bboxes[waterway_id]

    def get_navigability_class(self, depth_m: float, width_m: float) -> str:
        """
        Determine the rule-based navigability class from depth and width.

        This mirrors the IWAI classification used during model training.

        Parameters
        ----------
        depth_m : float
            Predicted water depth in metres.
        width_m : float
            Channel width in metres.

        Returns
        -------
        str
            "navigable" | "conditional" | "non_navigable"
        """
        if depth_m >= self.DEPTH_NAVIGABLE_MIN and width_m >= self.WIDTH_NAVIGABLE_MIN:
            return "navigable"
        if (
            depth_m >= self.DEPTH_CONDITIONAL_MIN
            and width_m >= self.WIDTH_CONDITIONAL_MIN
        ):
            return "conditional"
        return "non_navigable"


# ---------------------------------------------------------------------------
# Module-level cached singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application Settings singleton.

    Using ``@lru_cache`` ensures the .env file is parsed exactly once per
    process lifetime, which is important for performance in async contexts.

    Usage
    -----
    In FastAPI dependency injection::

        from app.core.config import get_settings

        def some_endpoint(settings: Settings = Depends(get_settings)):
            ...

    Or directly::

        settings = get_settings()
    """
    return Settings()


# Convenience alias — import this in other modules
settings: Settings = get_settings()
