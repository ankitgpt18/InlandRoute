"""
AIDSTL Project — Google Earth Engine Integration Service
=========================================================
Provides authenticated access to Google Earth Engine (GEE) for retrieving
Sentinel-2 surface-reflectance composites, computing spectral indices, and
extracting per-segment feature vectors used as inputs to the navigability
prediction models.

Study areas
-----------
  NW-1 : Ganga       — Varanasi → Haldia   (~1,620 km)
  NW-2 : Brahmaputra — Dhubri  → Sadiya    (~891 km)

Architecture
------------
  GEEService is a singleton (instantiated once at application startup via
  the FastAPI lifespan hook).  All public methods are async-friendly: heavy
  GEE calls run in a thread-pool executor so they never block the event loop.

Dependencies
------------
  earthengine-api  (pip install earthengine-api)
  google-auth      (transitive dependency of earthengine-api)

Authentication
--------------
  Uses a GCP service account with the Earth Engine API enabled.
  The service-account JSON key path is read from settings.GEE_KEY_FILE.
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Optional GEE import — degrade gracefully if the SDK is not installed
# ---------------------------------------------------------------------------

try:
    import ee  # type: ignore[import]

    _GEE_AVAILABLE = True
except ImportError:
    ee = None  # type: ignore[assignment]
    _GEE_AVAILABLE = False
    logger.warning("earthengine-api not installed.  GEEService will run in MOCK mode.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel-2 Surface Reflectance (Harmonised) collection
_S2_COLLECTION = settings.GEE_S2_COLLECTION  # "COPERNICUS/S2_SR_HARMONIZED"

# Bands to select (raw 10 000-scale integers → divide by 1e4 for reflectance)
_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Friendly band-name mapping
_BAND_ALIAS: dict[str, str] = {
    "B2": "blue",
    "B3": "green",
    "B4": "red",
    "B5": "red_edge_1",
    "B6": "red_edge_2",
    "B7": "red_edge_3",
    "B8": "nir",
    "B8A": "nir_narrow",
    "B11": "swir1",
    "B12": "swir2",
}

# Scale factor for Sentinel-2 L2A / SR Harmonised
_S2_SCALE: float = 10_000.0

# Default cloud percentage threshold for compositing
_CLOUD_THRESHOLD: int = settings.GEE_CLOUD_THRESHOLD  # 20 %

# Spatial scale for zonal statistics (10 m Sentinel-2 native resolution)
_SAMPLE_SCALE_M: int = 10

# GEE reducer for compositing pixels over a segment AOI
_COMPOSITE_REDUCER = "median"

# Months labels
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

# Thread-pool executor for running blocking GEE calls
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gee_worker")


# ---------------------------------------------------------------------------
# Cloud-masking helpers
# ---------------------------------------------------------------------------


def _mask_s2_clouds_scl(image: "ee.Image") -> "ee.Image":  # type: ignore[name-defined]
    """Mask clouds using the Sentinel-2 Scene Classification Layer (SCL).

    SCL pixel values:
      3  — cloud shadows
      8  — medium probability clouds
      9  — high probability clouds
      10 — thin cirrus

    Parameters
    ----------
    image : ee.Image
        A single Sentinel-2 SR Harmonised image.

    Returns
    -------
    ee.Image
        Cloud-masked image with the same band structure.
    """
    scl = image.select("SCL")
    cloud_shadow = scl.eq(3)
    cloud_med = scl.eq(8)
    cloud_high = scl.eq(9)
    cirrus = scl.eq(10)
    cloud_mask = cloud_shadow.Or(cloud_med).Or(cloud_high).Or(cirrus).Not()
    return image.updateMask(cloud_mask)


def _mask_s2_clouds_qa60(image: "ee.Image") -> "ee.Image":  # type: ignore[name-defined]
    """Fallback cloud masking using the QA60 bitmask band.

    Bit 10 — opaque clouds
    Bit 11 — cirrus clouds

    Parameters
    ----------
    image : ee.Image
        A single Sentinel-2 SR image.

    Returns
    -------
    ee.Image
        Cloud-masked image.
    """
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)


# ---------------------------------------------------------------------------
# Spectral index computation within GEE (server-side)
# ---------------------------------------------------------------------------


def _add_mndwi(image: "ee.Image") -> "ee.Image":  # type: ignore[name-defined]
    mndwi = image.normalizedDifference(["green", "swir1"]).rename("mndwi")
    return image.addBands(mndwi)


def _add_ndwi(image: "ee.Image") -> "ee.Image":  # type: ignore[name-defined]
    ndwi = image.normalizedDifference(["green", "nir"]).rename("ndwi")
    return image.addBands(ndwi)


def _add_ndvi(image: "ee.Image") -> "ee.Image":  # type: ignore[name-defined]
    ndvi = image.normalizedDifference(["nir", "red"]).rename("ndvi")
    return image.addBands(ndvi)


def _add_awei_nsh(image: "ee.Image") -> "ee.Image":  # type: ignore[name-defined]
    """AWEInsh = 4*(Green-SWIR1) - (0.25*NIR + 2.75*SWIR2)."""
    g = image.select("green")
    n = image.select("nir")
    s1 = image.select("swir1")
    s2 = image.select("swir2")
    awei = (
        g.subtract(s1)
        .multiply(4)
        .subtract(n.multiply(0.25).add(s2.multiply(2.75)))
        .rename("awei_nsh")
    )
    return image.addBands(awei)


def _add_turbidity(image: "ee.Image") -> "ee.Image":  # type: ignore[name-defined]
    """NDTI = (Red - Green) / (Red + Green)."""
    ndti = image.normalizedDifference(["red", "green"]).rename("ndti")
    return image.addBands(ndti)


def _add_stumpf_bg(image: "ee.Image") -> "ee.Image":  # type: ignore[name-defined]
    """Stumpf log-ratio: ln(1000*Blue) / ln(1000*Green)."""
    n = 1000.0
    b = image.select("blue").multiply(n).log()
    g = image.select("green").multiply(n).log()
    ratio = b.divide(g).rename("stumpf_bg")
    return image.addBands(ratio)


# ---------------------------------------------------------------------------
# GEE Service
# ---------------------------------------------------------------------------


class GEEService:
    """
    Singleton service for Google Earth Engine data extraction.

    Responsibilities
    ----------------
    - Authenticate with GEE via a service account.
    - Build cloud-masked Sentinel-2 monthly median composites.
    - Extract per-segment spectral feature dictionaries.
    - Compute water-extent metrics (width, area, water fraction).
    - Retrieve multi-year historical time series for training / analytics.

    Thread Safety
    -------------
    The GEE Python API is not async-native; all blocking ``getInfo()`` /
    ``reduceRegion()`` calls are offloaded to ``_executor`` (a
    ``ThreadPoolExecutor``) so the FastAPI event loop is never blocked.

    Usage
    -----
    Access the singleton via ``get_gee_service()``.  The service must be
    initialised once via ``await gee_service.initialize()`` before use
    (handled in ``main.py`` lifespan).
    """

    _instance: Optional["GEEService"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "GEEService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
        return cls._instance

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Authenticate with GEE and verify connectivity.

        This method is idempotent — safe to call multiple times.  It
        authenticates using the service-account JSON key specified in
        settings.GEE_KEY_FILE.

        Raises
        ------
        RuntimeError
            If authentication fails or the GEE API is unreachable.
        """
        if self._initialised:
            logger.debug("GEEService already initialised; skipping.")
            return

        if not _GEE_AVAILABLE:
            logger.warning(
                "GEEService.initialize() called but earthengine-api is not "
                "installed.  Running in MOCK mode — all feature extractions "
                "will return synthetic data."
            )
            self._initialised = True
            self._mock_mode = True
            return

        self._mock_mode = False
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(_executor, self._authenticate_sync)
            logger.info("GEEService initialised successfully.")
            self._initialised = True
        except Exception as exc:
            logger.error("GEE authentication failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Failed to initialise GEE service: {exc}") from exc

    def _authenticate_sync(self) -> None:
        """Blocking GEE authentication (runs in thread pool)."""
        key_file = settings.gee_key_file_path

        if not key_file.exists():
            raise FileNotFoundError(
                f"GEE key file not found: {key_file}. "
                "Set GEE_KEY_FILE in your .env to a valid service-account JSON path."
            )

        credentials = ee.ServiceAccountCredentials(
            email=settings.GEE_SERVICE_ACCOUNT,
            key_file=str(key_file),
        )
        project = settings.GEE_PROJECT_ID or None
        ee.Initialize(
            credentials=credentials,
            project=project,
            opt_url="https://earthengine.googleapis.com",
        )
        logger.info("Authenticated with GEE as '%s'.", settings.GEE_SERVICE_ACCOUNT)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Verify GEE connectivity by running a trivial server-side computation.

        Returns
        -------
        dict[str, Any]
            ``{"status": "healthy"|"unhealthy", "mock_mode": bool, ...}``
        """
        if not self._initialised:
            return {"status": "uninitialised", "mock_mode": False}

        if getattr(self, "_mock_mode", False):
            return {"status": "healthy", "mock_mode": True, "message": "MOCK mode"}

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(_executor, self._health_check_sync)
            return {"status": "healthy", "mock_mode": False, **result}
        except Exception as exc:
            return {"status": "unhealthy", "mock_mode": False, "error": str(exc)}

    def _health_check_sync(self) -> dict[str, Any]:
        """Run a trivial GEE computation to confirm API connectivity."""
        image = ee.Image(
            "COPERNICUS/S2_SR_HARMONIZED/20230101T052649_20230101T053750_T44RLR"
        )
        info = image.getInfo()
        return {
            "example_image_id": info.get("id", "unknown"),
            "gee_sdk_version": ee.__version__,
        }

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _month_date_range(month: int, year: int) -> tuple[str, str]:
        """Return ISO 8601 start/end date strings for a calendar month.

        Parameters
        ----------
        month : int
            Calendar month (1–12).
        year : int
            Calendar year.

        Returns
        -------
        tuple[str, str]
            ``(start_date, end_date)`` as ``"YYYY-MM-DD"`` strings.
        """
        _, last_day = calendar.monthrange(year, month)
        start = f"{year:04d}-{month:02d}-01"
        end = f"{year:04d}-{month:02d}-{last_day:02d}"
        return start, end

    @staticmethod
    def _geometry_from_bbox(
        bbox: list[float],
    ) -> "ee.Geometry.Rectangle":  # type: ignore[name-defined]
        """Convert [minLon, minLat, maxLon, maxLat] to an ee.Geometry.Rectangle."""
        return ee.Geometry.Rectangle(bbox)

    @staticmethod
    def _geometry_from_geojson(geojson: dict[str, Any]) -> "ee.Geometry":  # type: ignore[name-defined]
        """Convert a GeoJSON geometry dict to an ee.Geometry."""
        geo_type = geojson.get("type", "")
        if geo_type == "LineString":
            return ee.Geometry.LineString(geojson["coordinates"])
        if geo_type == "Polygon":
            return ee.Geometry.Polygon(geojson["coordinates"])
        if geo_type == "MultiPolygon":
            return ee.Geometry.MultiPolygon(geojson["coordinates"])
        if geo_type == "MultiLineString":
            return ee.Geometry.MultiLineString(geojson["coordinates"])
        if geo_type == "Point":
            return ee.Geometry.Point(geojson["coordinates"])
        # Fallback: pass raw GeoJSON dict
        return ee.Geometry(geojson)

    # ------------------------------------------------------------------
    # Core composite builder
    # ------------------------------------------------------------------

    def _build_s2_composite_sync(
        self,
        aoi: "ee.Geometry",  # type: ignore[name-defined]
        start_date: str,
        end_date: str,
        cloud_threshold: int = _CLOUD_THRESHOLD,
    ) -> "ee.Image":  # type: ignore[name-defined]
        """Build a cloud-masked Sentinel-2 median composite over an AOI.

        Steps
        -----
        1. Filter the S2 collection by date, bounds, and cloud percentage.
        2. Mask clouds using SCL band (fallback to QA60).
        3. Select and rename bands.
        4. Scale reflectances to [0, 1].
        5. Compute a pixel-wise median composite.
        6. Add spectral indices as additional bands.

        Parameters
        ----------
        aoi : ee.Geometry
            Area of interest.
        start_date, end_date : str
            ISO date strings (``"YYYY-MM-DD"``).
        cloud_threshold : int
            Maximum CLOUDY_PIXEL_PERCENTAGE to include.

        Returns
        -------
        ee.Image
            Median composite image with all spectral bands and indices.
        """
        collection = (
            ee.ImageCollection(_S2_COLLECTION)
            .filterDate(start_date, end_date)
            .filterBounds(aoi)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
            .select(_S2_BANDS, list(_BAND_ALIAS.values()))
            .map(_mask_s2_clouds_scl)
            .map(lambda img: img.divide(_S2_SCALE))
        )

        # Fall back to a wider cloud threshold if insufficient images
        count = collection.size().getInfo()
        if count == 0:
            logger.warning(
                "No S2 images for [%s, %s] with cloud≤%d%%. Relaxing to 80%%.",
                start_date,
                end_date,
                cloud_threshold,
            )
            collection = (
                ee.ImageCollection(_S2_COLLECTION)
                .filterDate(start_date, end_date)
                .filterBounds(aoi)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
                .select(_S2_BANDS, list(_BAND_ALIAS.values()))
                .map(_mask_s2_clouds_qa60)
                .map(lambda img: img.divide(_S2_SCALE))
            )

        composite = collection.median()

        # Add spectral indices as server-side bands
        composite = (
            composite.pipe(_add_mndwi)
            .pipe(_add_ndwi)
            .pipe(_add_ndvi)
            .pipe(_add_awei_nsh)
            .pipe(_add_turbidity)
            .pipe(_add_stumpf_bg)
        )

        return composite

    # ------------------------------------------------------------------
    # Public API — Composites
    # ------------------------------------------------------------------

    async def get_current_composites(
        self,
        waterway_id: str,
        month: int,
        year: int,
        cloud_threshold: int = _CLOUD_THRESHOLD,
    ) -> dict[str, Any]:
        """Fetch the latest Sentinel-2 median composite for a waterway.

        Builds a cloud-masked monthly median composite over the full
        bounding box of the specified waterway.

        Parameters
        ----------
        waterway_id : str
            ``"NW-1"`` or ``"NW-2"``.
        month : int
            Calendar month (1–12).
        year : int
            Calendar year (e.g. 2024).
        cloud_threshold : int, optional
            Maximum cloud coverage percentage (default from settings).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``"composite_info"`` — GEE image metadata dict.
            - ``"start_date"`` / ``"end_date"`` — temporal coverage.
            - ``"waterway_id"``, ``"month"``, ``"year"``.
            - ``"cloud_threshold"`` — threshold used.
            - ``"image_count"`` — number of scenes composited.
        """
        if getattr(self, "_mock_mode", False):
            return self._mock_composite_response(waterway_id, month, year)

        from app.utils.spatial import get_waterway_meta

        meta = get_waterway_meta(waterway_id)
        bbox = meta["bbox"]

        start_date, end_date = self._month_date_range(month, year)
        loop = asyncio.get_event_loop()

        def _run() -> dict[str, Any]:
            aoi = self._geometry_from_bbox(bbox)
            collection_filtered = (
                ee.ImageCollection(_S2_COLLECTION)
                .filterDate(start_date, end_date)
                .filterBounds(aoi)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
            )
            image_count = int(collection_filtered.size().getInfo())
            composite = self._build_s2_composite_sync(
                aoi, start_date, end_date, cloud_threshold
            )
            info = composite.getInfo()
            return {
                "composite_info": info,
                "start_date": start_date,
                "end_date": end_date,
                "waterway_id": waterway_id,
                "month": month,
                "year": year,
                "cloud_threshold": cloud_threshold,
                "image_count": image_count,
            }

        try:
            result = await loop.run_in_executor(_executor, _run)
            logger.info(
                "Composite built for %s %s/%d: %d scenes.",
                waterway_id,
                _MONTH_NAMES[month - 1],
                year,
                result.get("image_count", 0),
            )
            return result
        except Exception as exc:
            logger.error(
                "Failed to build composite for %s %d/%d: %s",
                waterway_id,
                month,
                year,
                exc,
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Public API — Per-Segment Feature Extraction
    # ------------------------------------------------------------------

    async def extract_segment_features(
        self,
        segment_id: str,
        month: int,
        year: int,
        geometry: Optional[dict[str, Any]] = None,
        buffer_km: float = 2.0,
    ) -> dict[str, float | None]:
        """Extract spectral feature statistics for a single river segment.

        Performs a zonal-statistics reduction (median per band + index)
        over the segment's buffered AOI using a monthly S2 composite.

        Parameters
        ----------
        segment_id : str
            E.g. ``"NW-1-042"``.
        month : int
            Calendar month (1–12).
        year : int
            Calendar year.
        geometry : dict, optional
            GeoJSON geometry of the segment centreline.  If omitted, the
            segment centroid is used with a circular buffer.
        buffer_km : float, optional
            Buffer radius around the segment centreline in km (default 2 km).

        Returns
        -------
        dict[str, float | None]
            Feature dictionary keyed by band/index name.  None values
            indicate masked or unavailable pixels.
        """
        if getattr(self, "_mock_mode", False):
            return self._mock_segment_features(segment_id, month, year)

        from app.utils.spatial import buffer_segment, parse_segment_id

        waterway_id, _ = parse_segment_id(segment_id)

        start_date, end_date = self._month_date_range(month, year)
        loop = asyncio.get_event_loop()

        def _run() -> dict[str, float | None]:
            # Build AOI
            if geometry is not None:
                seg_geom = self._geometry_from_geojson(geometry)
                # Buffer by buffer_km in degrees (approximation)
                buffer_deg = buffer_km / 111.0  # 1° ≈ 111 km
                aoi = seg_geom.buffer(buffer_deg * 1000).bounds()
            else:
                from app.utils.spatial import get_waterway_meta

                meta = get_waterway_meta(waterway_id)
                aoi = self._geometry_from_bbox(meta["bbox"])

            composite = self._build_s2_composite_sync(aoi, start_date, end_date)

            bands_and_indices = list(_BAND_ALIAS.values()) + [
                "mndwi",
                "ndwi",
                "ndvi",
                "awei_nsh",
                "ndti",
                "stumpf_bg",
            ]

            reducer = ee.Reducer.median()
            stats = composite.select(bands_and_indices).reduceRegion(
                reducer=reducer,
                geometry=aoi,
                scale=_SAMPLE_SCALE_M,
                maxPixels=int(settings.GEE_MAX_PIXELS),
                bestEffort=True,
            )

            raw = stats.getInfo()  # dict[str, float | None]
            return {k: (float(v) if v is not None else None) for k, v in raw.items()}

        try:
            features = await loop.run_in_executor(_executor, _run)
            logger.debug(
                "Features extracted for %s %d/%d: %d bands/indices.",
                segment_id,
                month,
                year,
                len(features),
            )
            return features
        except Exception as exc:
            logger.error(
                "Feature extraction failed for %s %d/%d: %s",
                segment_id,
                month,
                year,
                exc,
                exc_info=True,
            )
            raise

    async def extract_batch_segment_features(
        self,
        segment_ids: list[str],
        month: int,
        year: int,
        geometries: Optional[dict[str, dict[str, Any]]] = None,
        buffer_km: float = 2.0,
    ) -> dict[str, dict[str, float | None]]:
        """Extract features for multiple segments concurrently.

        Parameters
        ----------
        segment_ids : list[str]
            List of segment IDs to process.
        month : int
            Calendar month (1–12).
        year : int
            Calendar year.
        geometries : dict[str, dict], optional
            Mapping of segment_id → GeoJSON geometry.
        buffer_km : float, optional
            Buffer radius in km.

        Returns
        -------
        dict[str, dict[str, float | None]]
            Mapping of ``segment_id → feature_dict``.
        """
        geometries = geometries or {}

        tasks = [
            self.extract_segment_features(
                segment_id=sid,
                month=month,
                year=year,
                geometry=geometries.get(sid),
                buffer_km=buffer_km,
            )
            for sid in segment_ids
        ]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, dict[str, float | None]] = {}
        for sid, result in zip(segment_ids, results_list):
            if isinstance(result, Exception):
                logger.warning("Feature extraction failed for %s: %s", sid, result)
                output[sid] = {}
            else:
                output[sid] = result  # type: ignore[assignment]

        return output

    # ------------------------------------------------------------------
    # Public API — Water Extent
    # ------------------------------------------------------------------

    async def compute_water_extent(
        self,
        segment_geometry: dict[str, Any],
        month: int,
        year: int,
        mndwi_threshold: float = 0.0,
        buffer_km: float = 2.0,
    ) -> dict[str, float]:
        """Compute water-body extent metrics over a river segment.

        Classifies pixels as water using an MNDWI threshold and computes:
        - Total water area (m²)
        - Estimated channel width (m)
        - Water pixel fraction

        Parameters
        ----------
        segment_geometry : dict
            GeoJSON geometry of the segment centreline.
        month : int
            Calendar month (1–12).
        year : int
            Calendar year.
        mndwi_threshold : float, optional
            MNDWI threshold for water classification (default 0.0).
        buffer_km : float, optional
            Buffer around the segment for AOI (default 2 km).

        Returns
        -------
        dict[str, float]
            Keys: ``"water_area_m2"``, ``"width_m"``, ``"water_fraction"``,
            ``"total_pixels"``, ``"water_pixels"``.
        """
        if getattr(self, "_mock_mode", False):
            return self._mock_water_extent(month)

        start_date, end_date = self._month_date_range(month, year)
        loop = asyncio.get_event_loop()

        def _run() -> dict[str, float]:
            seg_geom = self._geometry_from_geojson(segment_geometry)
            buffer_deg = buffer_km / 111.0
            aoi = seg_geom.buffer(buffer_deg * 1000).bounds()

            composite = self._build_s2_composite_sync(aoi, start_date, end_date)

            # Binary water mask from MNDWI
            water_mask = composite.select("mndwi").gt(mndwi_threshold).rename("water")
            composite_with_mask = composite.addBands(water_mask)

            # Count pixels
            pixel_counts = composite_with_mask.select(["water"]).reduceRegion(
                reducer=ee.Reducer.sum().combine(ee.Reducer.count(), sharedInputs=True),
                geometry=aoi,
                scale=_SAMPLE_SCALE_M,
                maxPixels=int(settings.GEE_MAX_PIXELS),
                bestEffort=True,
            )
            raw = pixel_counts.getInfo()

            water_pixels = float(raw.get("water_sum", 0) or 0)
            total_pixels = float(raw.get("water_count", 0) or 0)

            # Area in m² (each pixel = 10m × 10m = 100 m²)
            water_area_m2 = water_pixels * (_SAMPLE_SCALE_M**2)
            water_fraction = water_pixels / total_pixels if total_pixels > 0 else 0.0

            # Width estimate: area / segment length
            # Segment length computed from GEE geometry
            seg_length_m = float(seg_geom.length().getInfo())
            width_m = water_area_m2 / seg_length_m if seg_length_m > 0 else 0.0

            return {
                "water_area_m2": round(water_area_m2, 2),
                "width_m": round(width_m, 2),
                "water_fraction": round(water_fraction, 6),
                "total_pixels": total_pixels,
                "water_pixels": water_pixels,
            }

        try:
            result = await loop.run_in_executor(_executor, _run)
            logger.debug(
                "Water extent computed: width=%.1f m, area=%.0f m²",
                result["width_m"],
                result["water_area_m2"],
            )
            return result
        except Exception as exc:
            logger.error("Water extent computation failed: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Public API — Historical time series
    # ------------------------------------------------------------------

    async def get_historical_features(
        self,
        segment_id: str,
        geometry: Optional[dict[str, Any]],
        start_year: int,
        end_year: int,
        months: Optional[list[int]] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve multi-year monthly feature time series for a segment.

        For each requested month in each year from ``start_year`` to
        ``end_year`` (inclusive), extracts the Sentinel-2 spectral feature
        vector and water-extent metrics.

        Parameters
        ----------
        segment_id : str
            E.g. ``"NW-1-042"``.
        geometry : dict | None
            GeoJSON geometry of the segment centreline.
        start_year : int
            First year of the time series.
        end_year : int
            Last year of the time series (inclusive).
        months : list[int], optional
            Specific months to extract (1–12).  Defaults to all 12 months.

        Returns
        -------
        list[dict[str, Any]]
            Ordered list of feature records.  Each record is a dict with
            keys: ``"segment_id"``, ``"year"``, ``"month"``, ``"features"``.
        """
        if months is None:
            months = list(range(1, 13))

        records: list[dict[str, Any]] = []

        for year in range(start_year, end_year + 1):
            for month in months:
                # Skip future months
                now = datetime.now(timezone.utc)
                if year > now.year or (year == now.year and month > now.month):
                    continue

                try:
                    feats = await self.extract_segment_features(
                        segment_id=segment_id,
                        month=month,
                        year=year,
                        geometry=geometry,
                    )
                    records.append(
                        {
                            "segment_id": segment_id,
                            "year": year,
                            "month": month,
                            "features": feats,
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "Historical feature extraction failed for %s %d/%d: %s",
                        segment_id,
                        month,
                        year,
                        exc,
                    )
                    records.append(
                        {
                            "segment_id": segment_id,
                            "year": year,
                            "month": month,
                            "features": {},
                            "error": str(exc),
                        }
                    )

        logger.info(
            "Historical features extracted for %s: %d records (%d–%d).",
            segment_id,
            len(records),
            start_year,
            end_year,
        )
        return records

    async def get_monthly_climatology(
        self,
        waterway_id: str,
        base_period_start: int = 2016,
        base_period_end: int = 2023,
    ) -> dict[int, dict[str, float]]:
        """Compute multi-year monthly mean feature climatology for a waterway.

        Returns the long-term monthly average of each spectral feature over
        the specified baseline period.  Used to compute anomalies at
        inference time.

        Parameters
        ----------
        waterway_id : str
            ``"NW-1"`` or ``"NW-2"``.
        base_period_start : int
            First year of the baseline period.
        base_period_end : int
            Last year of the baseline period (inclusive).

        Returns
        -------
        dict[int, dict[str, float]]
            Mapping ``{month: {feature_name: mean_value}}``.
        """
        if getattr(self, "_mock_mode", False):
            return {
                m: self._mock_segment_features("clim", m, 2020) for m in range(1, 13)
            }

        from app.utils.spatial import get_waterway_meta

        meta = get_waterway_meta(waterway_id)
        bbox = meta["bbox"]

        loop = asyncio.get_event_loop()

        def _run_month(month: int) -> tuple[int, dict[str, float]]:
            aoi = self._geometry_from_bbox(bbox)
            all_features: list[dict[str, float]] = []

            for year in range(base_period_start, base_period_end + 1):
                start_date, end_date = self._month_date_range(month, year)
                composite = self._build_s2_composite_sync(aoi, start_date, end_date)

                bands = list(_BAND_ALIAS.values()) + ["mndwi", "ndwi", "ndvi"]
                stats = composite.select(bands).reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=aoi,
                    scale=_SAMPLE_SCALE_M * 10,  # coarser for full-waterway bbox
                    maxPixels=int(settings.GEE_MAX_PIXELS),
                    bestEffort=True,
                )
                raw = stats.getInfo()
                clean = {k: float(v) for k, v in raw.items() if v is not None}
                if clean:
                    all_features.append(clean)

            if not all_features:
                return month, {}

            # Compute feature means across years
            combined: dict[str, list[float]] = {}
            for feat in all_features:
                for k, v in feat.items():
                    combined.setdefault(k, []).append(v)

            return month, {k: float(np.mean(vals)) for k, vals in combined.items()}

        tasks_results = await asyncio.gather(
            *[loop.run_in_executor(_executor, _run_month, m) for m in range(1, 13)]
        )

        return dict(tasks_results)

    # ------------------------------------------------------------------
    # Image count / data availability
    # ------------------------------------------------------------------

    async def get_scene_count(
        self,
        waterway_id: str,
        month: int,
        year: int,
        cloud_threshold: int = _CLOUD_THRESHOLD,
    ) -> int:
        """Return the number of available Sentinel-2 scenes for a month.

        Parameters
        ----------
        waterway_id : str
            ``"NW-1"`` or ``"NW-2"``.
        month, year : int
            Target month and year.
        cloud_threshold : int
            Maximum cloud cover percentage.

        Returns
        -------
        int
            Number of available scenes.  Returns ``0`` if GEE is unavailable.
        """
        if getattr(self, "_mock_mode", False):
            return np.random.randint(3, 12)  # type: ignore[return-value]

        from app.utils.spatial import get_waterway_meta

        meta = get_waterway_meta(waterway_id)
        bbox = meta["bbox"]

        start_date, end_date = self._month_date_range(month, year)
        loop = asyncio.get_event_loop()

        def _run() -> int:
            aoi = self._geometry_from_bbox(bbox)
            collection = (
                ee.ImageCollection(_S2_COLLECTION)
                .filterDate(start_date, end_date)
                .filterBounds(aoi)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
            )
            return int(collection.size().getInfo())

        try:
            count = await loop.run_in_executor(_executor, _run)
            return count
        except Exception as exc:
            logger.warning(
                "Scene count failed for %s %d/%d: %s", waterway_id, month, year, exc
            )
            return 0

    # ------------------------------------------------------------------
    # Mock data generators (used when earthengine-api is unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_composite_response(
        waterway_id: str, month: int, year: int
    ) -> dict[str, Any]:
        """Return a plausible mock composite response for development."""
        return {
            "composite_info": {"type": "Image", "bands": list(_BAND_ALIAS.values())},
            "start_date": f"{year:04d}-{month:02d}-01",
            "end_date": f"{year:04d}-{month:02d}-28",
            "waterway_id": waterway_id,
            "month": month,
            "year": year,
            "cloud_threshold": _CLOUD_THRESHOLD,
            "image_count": 7,
            "mock": True,
        }

    @staticmethod
    def _mock_segment_features(
        segment_id: str, month: int, year: int
    ) -> dict[str, float | None]:
        """Return synthetic spectral features for development / testing.

        Values are seeded from the segment ID + date to ensure consistency
        across repeated calls.
        """
        seed_str = f"{segment_id}-{month}-{year}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(seed)

        # Simulate seasonal variation — higher MNDWI during monsoon (Jun–Sep)
        monsoon_boost = 0.2 if month in (6, 7, 8, 9) else 0.0

        return {
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
            "mndwi": float(np.clip(rng.uniform(0.1, 0.6) + monsoon_boost, -1, 1)),
            "ndwi": float(np.clip(rng.uniform(0.0, 0.5) + monsoon_boost, -1, 1)),
            "ndvi": float(rng.uniform(-0.1, 0.4)),
            "awei_nsh": float(rng.uniform(-0.2, 0.8) + monsoon_boost),
            "ndti": float(np.clip(rng.uniform(-0.1, 0.3), -1, 1)),
            "stumpf_bg": float(rng.uniform(0.9, 1.2)),
            "water_fraction": float(
                np.clip(rng.uniform(0.3, 0.9) + monsoon_boost * 0.3, 0, 1)
            ),
        }

    @staticmethod
    def _mock_water_extent(month: int) -> dict[str, float]:
        """Return synthetic water-extent metrics for development."""
        monsoon = month in (6, 7, 8, 9)
        base_width = 500.0 if monsoon else 200.0
        return {
            "water_area_m2": float(base_width * 5000 * np.random.uniform(0.8, 1.2)),
            "width_m": float(base_width * np.random.uniform(0.8, 1.2)),
            "water_fraction": float(
                np.random.uniform(0.4, 0.8) if monsoon else np.random.uniform(0.1, 0.4)
            ),
            "total_pixels": float(np.random.randint(50_000, 200_000)),
            "water_pixels": float(np.random.randint(10_000, 80_000)),
        }


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_gee_service() -> GEEService:
    """Return the application-wide GEEService singleton.

    The singleton is created on first call and reused thereafter.
    ``initialize()`` must be called separately (in the FastAPI lifespan).

    Returns
    -------
    GEEService
        The singleton instance.
    """
    return GEEService()
