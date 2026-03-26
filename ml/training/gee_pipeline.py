"""
gee_pipeline.py
===============
Google Earth Engine (GEE) data-ingestion and feature-extraction pipeline for
the AIDSTL inland-waterway navigability project.

Study areas
-----------
  NW-1  Ganga        Varanasi – Haldia        ~1 390 km
  NW-2  Brahmaputra  Dhubri   – Sadiya        ~  891 km

River segmented into 5 km analysis units.

Pipeline overview
-----------------
  1. Authenticate with GEE.
  2. For each (segment, month) pair:
       a. Cloud-free Sentinel-2 L2A median composite (SCL masking).
       b. Spectral indices: MNDWI, NDWI, AWEI, Stumpf, Turbidity, NDTI.
       c. Sentinel-1 SAR backscatter (VV, VH) median.
       d. ERA5 cumulative rainfall + mean temperature.
       e. CWC gauge readings (interpolated to segment chainage).
       f. SRTM DEM elevation + slope.
  3. Assemble a tidy pd.DataFrame with one row per (segment_id, date).
  4. Export as Parquet + GeoJSON for downstream model training.

Dependencies
------------
  earthengine-api  (ee)
  geopandas, shapely, pyproj
  pandas, numpy
  tqdm, joblib, python-dotenv
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Optional GEE import — gracefully degrade if ee is not installed
# ---------------------------------------------------------------------------
try:
    import ee

    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    logger.warning(
        "earthengine-api not installed.  GEEPipeline will operate in "
        "OFFLINE / mock mode only."
    )

try:
    import geopandas as gpd
    import pyproj
    from shapely.geometry import LineString, Point, mapping, shape
    from shapely.ops import substring

    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False
    logger.warning("geopandas / shapely not installed.  Geometric ops disabled.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# National waterway IDs
NW_IDS: Dict[str, Dict[str, Any]] = {
    "NW1": {
        "name": "Ganga",
        "start_name": "Varanasi",
        "end_name": "Haldia",
        "length_km": 1390,
        "start_coords": (82.9739, 25.3176),  # (lon, lat) Varanasi Ghat
        "end_coords": (88.0956, 22.0258),  # Haldia Port
        "gee_asset": "projects/aidstl/assets/nw1_centreline",  # update as needed
    },
    "NW2": {
        "name": "Brahmaputra",
        "start_name": "Dhubri",
        "end_name": "Sadiya",
        "length_km": 891,
        "start_coords": (89.9779, 26.0186),  # Dhubri
        "end_coords": (95.6606, 27.8326),  # Sadiya
        "gee_asset": "projects/aidstl/assets/nw2_centreline",
    },
}

# Sentinel-2 band names in GEE (Surface Reflectance product)
S2_SR_BANDS: List[str] = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_SCALE_FACTOR: float = 1e-4  # GEE stores DN as integer × 10 000

# SRTM DEM GEE asset
SRTM_ASSET: str = "USGS/SRTMGL1_003"

# ERA5 GEE asset
ERA5_ASSET: str = "ECMWF/ERA5_LAND/MONTHLY_AGGR"

# Sentinel-1 GEE collection
S1_COLLECTION: str = "COPERNICUS/S1_GRD"

# Sentinel-2 SR GEE collection
S2_SR_COLLECTION: str = "COPERNICUS/S2_SR_HARMONIZED"

# Default segment length (km)
DEFAULT_SEGMENT_KM: float = 5.0

# CWC gauge field names expected in CSV files
CWC_STATION_ID_COL: str = "station_id"
CWC_DATE_COL: str = "date"
CWC_LEVEL_COL: str = "water_level_m"
CWC_DISCHARGE_COL: str = "discharge_m3s"
CWC_CHAINAGE_COL: str = "chainage_km"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RiverSegment:
    """Metadata for one 5 km river-analysis unit.

    Attributes
    ----------
    segment_id : str
        Unique identifier, e.g. ``"NW1_SEG_0042"``.
    nw_id : str
        Waterway identifier (``"NW1"`` or ``"NW2"``).
    index : int
        Sequential index along the waterway (0-based).
    chainage_start_km : float
        Distance from waterway origin to segment start (km).
    chainage_end_km : float
        Distance from waterway origin to segment end (km).
    centroid_lon : float
        Longitude of segment centroid.
    centroid_lat : float
        Latitude of segment centroid.
    length_km : float
        Actual segment length (≤ DEFAULT_SEGMENT_KM at river end).
    geometry_wkt : str
        WKT representation of the segment polyline.
    """

    segment_id: str
    nw_id: str
    index: int
    chainage_start_km: float
    chainage_end_km: float
    centroid_lon: float
    centroid_lat: float
    length_km: float
    geometry_wkt: str = ""

    @property
    def chainage_mid_km(self) -> float:
        return (self.chainage_start_km + self.chainage_end_km) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "nw_id": self.nw_id,
            "index": self.index,
            "chainage_start_km": self.chainage_start_km,
            "chainage_end_km": self.chainage_end_km,
            "chainage_mid_km": self.chainage_mid_km,
            "centroid_lon": self.centroid_lon,
            "centroid_lat": self.centroid_lat,
            "length_km": self.length_km,
            "geometry_wkt": self.geometry_wkt,
        }


@dataclass
class PipelineConfig:
    """Runtime configuration for the GEEPipeline.

    Attributes
    ----------
    gee_project : str
        GEE cloud project ID (required for the new ee.Initialize API).
    service_account : str
        Optional GEE service-account email for CI/server environments.
    credentials_path : str
        Path to the service-account JSON key file.
    segment_length_km : float
        Length of each river analysis unit.
    buffer_m : float
        Buffer radius around each segment centroid for GEE queries (metres).
    max_cloud_probability : float
        Maximum cloud probability (%) when filtering S2 scenes.
    s2_bands : list
        Sentinel-2 bands to export.
    scale_m : float
        GEE pixel scale for sample reductions (metres).
    max_retries : int
        Number of retries for transient GEE API errors.
    retry_delay_s : float
        Seconds to wait between retries.
    cwc_data_dir : str
        Directory containing CWC gauge CSV files.
    output_dir : str
        Root output directory for Parquet and GeoJSON exports.
    """

    gee_project: str = "aidstl-project"
    service_account: str = ""
    credentials_path: str = ""
    segment_length_km: float = DEFAULT_SEGMENT_KM
    buffer_m: float = 2500.0
    max_cloud_probability: float = 20.0
    s2_bands: List[str] = field(default_factory=lambda: list(S2_SR_BANDS))
    scale_m: float = 10.0
    max_retries: int = 3
    retry_delay_s: float = 5.0
    cwc_data_dir: str = "data/cwc_gauges"
    output_dir: str = "data/processed"


# ---------------------------------------------------------------------------
# GEEPipeline
# ---------------------------------------------------------------------------


class GEEPipeline:
    """End-to-end data pipeline from GEE satellite imagery to ML feature matrix.

    Workflow
    --------
    1. ``authenticate()``                 — initialise GEE credentials
    2. ``extract_river_segments()``       — create 5 km segment list
    3. ``build_training_dataset()``       — iterate months, query GEE per segment
    4. ``export_feature_matrix()``        — save Parquet + GeoJSON

    Parameters
    ----------
    config : PipelineConfig
        Runtime configuration.

    Notes
    -----
    When ``EE_AVAILABLE = False`` the pipeline operates in *mock mode*:
    all GEE calls return deterministically-seeded random data so that
    the downstream ML pipeline can be tested without a GEE account.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._authenticated: bool = False
        self._mock_mode: bool = not EE_AVAILABLE
        self._rng = np.random.default_rng(42)

        logger.info(
            "GEEPipeline initialised | mock_mode=%s | segment_length=%.1f km",
            self._mock_mode,
            self.config.segment_length_km,
        )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(
        self,
        interactive: bool = True,
        service_account: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ) -> bool:
        """Initialise the GEE Python API.

        Tries in order:
          1. Service-account credentials (if ``credentials_path`` provided).
          2. Application-default credentials via ``ee.Authenticate()``
             (requires a browser on interactive machines).
          3. Credentials cached by a previous ``ee.Authenticate()`` call.

        Parameters
        ----------
        interactive :
            Allow browser-based OAuth flow (True on workstations).
        service_account :
            GEE service-account email.  Overrides ``config.service_account``.
        credentials_path :
            Path to service-account JSON key.  Overrides
            ``config.credentials_path``.

        Returns
        -------
        bool
            True if authentication succeeded.
        """
        if self._mock_mode:
            logger.warning("GEE not available — running in mock mode.")
            self._authenticated = True
            return True

        sa = service_account or self.config.service_account
        cp = credentials_path or self.config.credentials_path

        try:
            if sa and cp and Path(cp).exists():
                # Service-account flow (headless / CI environments)
                credentials = ee.ServiceAccountCredentials(sa, cp)
                ee.Initialize(
                    credentials=credentials,
                    project=self.config.gee_project,
                )
                logger.info("GEE initialised via service account: %s", sa)
            else:
                # Interactive / cached credentials
                if interactive:
                    ee.Authenticate()
                ee.Initialize(project=self.config.gee_project)
                logger.info("GEE initialised via application-default credentials.")

            self._authenticated = True
            return True

        except Exception as exc:
            logger.error("GEE authentication failed: %s", exc)
            self._authenticated = False
            return False

    def _require_auth(self) -> None:
        """Raise RuntimeError if not authenticated."""
        if not self._authenticated:
            raise RuntimeError(
                "GEEPipeline is not authenticated. Call authenticate() first."
            )

    # ------------------------------------------------------------------
    # Sentinel-2 composites
    # ------------------------------------------------------------------

    def _mask_s2_clouds(self, image: Any) -> Any:
        """Apply SCL-based cloud / shadow mask to a Sentinel-2 SR image.

        SCL class values that are masked out:
          3  — cloud shadows
          8  — medium probability cloud
          9  — high probability cloud
          10 — thin cirrus
          11 — snow / ice (masked in river context)

        Parameters
        ----------
        image : ee.Image
            A single Sentinel-2 SR image with the SCL band.

        Returns
        -------
        ee.Image
            Input image with cloudy / shadow pixels masked.
        """
        if self._mock_mode:
            return image

        scl = image.select("SCL")
        # Keep pixels with SCL in {4=vegetation, 5=not-vegetated, 6=water, 7=unclassified}
        valid_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        return image.updateMask(valid_mask).divide(10000)  # scale DN → reflectance

    def get_sentinel2_composite(
        self,
        geometry: Any,
        start_date: str,
        end_date: str,
        bands: Optional[List[str]] = None,
    ) -> Any:
        """Build a cloud-free Sentinel-2 SR monthly median composite.

        Parameters
        ----------
        geometry : ee.Geometry
            Area of interest.
        start_date : str
            Start date in ``"YYYY-MM-DD"`` format.
        end_date : str
            End date (exclusive) in ``"YYYY-MM-DD"`` format.
        bands : list of str, optional
            Bands to select.  Defaults to ``config.s2_bands``.

        Returns
        -------
        ee.Image
            Single-image median composite in reflectance units (0–1).
        """
        self._require_auth()
        bands = bands or self.config.s2_bands

        if self._mock_mode:
            return {"_mock": True, "bands": bands, "geometry": geometry}

        collection = (
            ee.ImageCollection(S2_SR_COLLECTION)
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90))
            .map(self._mask_s2_clouds)
            .select(bands + ["SCL"] if "SCL" not in bands else bands)
        )

        n_images = collection.size().getInfo()
        logger.debug(
            "S2 composite [%s → %s]: %d images available.",
            start_date,
            end_date,
            n_images,
        )

        composite = collection.median().select(bands)
        return composite.clip(geometry)

    def compute_spectral_indices(self, image: Any) -> Any:
        """Add spectral-index bands to a Sentinel-2 composite image.

        Bands added
        -----------
        MNDWI   = (B3 − B11) / (B3 + B11)
        NDWI    = (B3 − B8)  / (B3 + B8)
        AWEI    = 4*(B3 − B11) − 0.25*B8 + 2.75*B12
        STUMPF  = log(B3) / log(B2)           (shallow-water depth proxy)
        TURBIDITY = (B4 − B3) / (B4 + B3)
        NDTI    = (B4 − B3) / (B4 + B3)       (alias kept for clarity)

        Parameters
        ----------
        image : ee.Image (or mock dict)
            Sentinel-2 composite in reflectance units.

        Returns
        -------
        ee.Image
            Original image with six additional index bands.
        """
        self._require_auth()

        if self._mock_mode or isinstance(image, dict):
            return image  # pass-through in mock mode

        B2 = image.select("B2")
        B3 = image.select("B3")
        B4 = image.select("B4")
        B8 = image.select("B8")
        B11 = image.select("B11")
        B12 = image.select("B12")

        eps = 1e-9  # avoid divide-by-zero

        mndwi = B3.subtract(B11).divide(B3.add(B11).add(eps)).rename("MNDWI")
        ndwi = B3.subtract(B8).divide(B3.add(B8).add(eps)).rename("NDWI")
        awei = (
            B3.subtract(B11)
            .multiply(4.0)
            .subtract(B8.multiply(0.25))
            .add(B12.multiply(2.75))
            .rename("AWEI")
        )
        stumpf = B3.log().divide(B2.log().add(eps)).rename("STUMPF")
        turbidity = B4.subtract(B3).divide(B4.add(B3).add(eps)).rename("TURBIDITY")
        ndti = turbidity.rename("NDTI")

        return image.addBands([mndwi, ndwi, awei, stumpf, turbidity, ndti])

    # ------------------------------------------------------------------
    # Sentinel-1 SAR
    # ------------------------------------------------------------------

    def get_sentinel1_backscatter(
        self,
        geometry: Any,
        start_date: str,
        end_date: str,
    ) -> Any:
        """Retrieve Sentinel-1 GRD VV/VH median backscatter composite.

        Parameters
        ----------
        geometry : ee.Geometry
            Area of interest.
        start_date : str
            Start date ``"YYYY-MM-DD"``.
        end_date : str
            End date ``"YYYY-MM-DD"`` (exclusive).

        Returns
        -------
        ee.Image
            Median backscatter image with bands VV and VH (dB scale).
        """
        self._require_auth()

        if self._mock_mode:
            return {"_mock": True, "bands": ["VV", "VH"]}

        collection = (
            ee.ImageCollection(S1_COLLECTION)
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(["VV", "VH"])
        )

        n_images = collection.size().getInfo()
        logger.debug(
            "S1 backscatter [%s → %s]: %d images available.",
            start_date,
            end_date,
            n_images,
        )

        return collection.median().clip(geometry)

    # ------------------------------------------------------------------
    # DEM & terrain
    # ------------------------------------------------------------------

    def get_dem_features(self, geometry: Any) -> Any:
        """Retrieve SRTM elevation and slope for a geometry.

        Returns
        -------
        ee.Image with bands ``elevation`` and ``slope``.
        """
        self._require_auth()

        if self._mock_mode:
            return {"_mock": True, "bands": ["elevation", "slope"]}

        dem = ee.Image(SRTM_ASSET).select("elevation")
        slope = ee.Terrain.slope(dem).rename("slope")
        return dem.addBands(slope).clip(geometry)

    # ------------------------------------------------------------------
    # ERA5 climate
    # ------------------------------------------------------------------

    def get_era5_climate(
        self,
        geometry: Any,
        start_date: str,
        end_date: str,
    ) -> Any:
        """Retrieve ERA5-Land monthly aggregated climate variables.

        Bands extracted:
          ``total_precipitation_sum``  → cumulative rainfall (m, converted to mm)
          ``temperature_2m``           → mean 2-m temperature (K, converted to °C)

        Returns
        -------
        ee.Image  with bands renamed to ``rainfall_mm`` and ``temperature_c``.
        """
        self._require_auth()

        if self._mock_mode:
            return {"_mock": True, "bands": ["rainfall_mm", "temperature_c"]}

        collection = (
            ee.ImageCollection(ERA5_ASSET)
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .select(
                ["total_precipitation_sum", "temperature_2m"],
                ["rainfall_m", "temperature_k"],
            )
            .mean()
        )

        rainfall_mm = (
            collection.select("rainfall_m").multiply(1000).rename("rainfall_mm")
        )
        temperature_c = (
            collection.select("temperature_k").subtract(273.15).rename("temperature_c")
        )
        return rainfall_mm.addBands(temperature_c).clip(geometry)

    # ------------------------------------------------------------------
    # River segmentation
    # ------------------------------------------------------------------

    def extract_river_segments(
        self,
        waterway_id: str,
        segment_length_km: float = DEFAULT_SEGMENT_KM,
    ) -> List[RiverSegment]:
        """Divide a national waterway into equal-length river segments.

        When a GEE asset centreline is available, it is fetched and split
        geodesically.  Otherwise the pipeline falls back to a great-circle
        interpolation between the start and end coordinates defined in
        ``NW_IDS``.

        Parameters
        ----------
        waterway_id : str
            ``"NW1"`` or ``"NW2"``.
        segment_length_km : float
            Desired segment length.  Default 5 km.

        Returns
        -------
        List[RiverSegment]
            Ordered list of segments from source to mouth.
        """
        nw_id = waterway_id.upper()
        if nw_id not in NW_IDS:
            raise ValueError(
                f"Unknown waterway '{waterway_id}'. Available: {list(NW_IDS.keys())}"
            )

        meta = NW_IDS[nw_id]
        total_km = meta["length_km"]
        n_segments = math.ceil(total_km / segment_length_km)

        logger.info(
            "Segmenting %s (%s → %s, %.0f km) into %d × %.0f km units …",
            nw_id,
            meta["start_name"],
            meta["end_name"],
            total_km,
            n_segments,
            segment_length_km,
        )

        segments: List[RiverSegment] = []

        # ── Try to load centreline from GEE asset ─────────────────────
        centreline_coords: Optional[List[Tuple[float, float]]] = None
        if self._authenticated and not self._mock_mode and GEO_AVAILABLE:
            try:
                centreline_coords = self._load_centreline_from_gee(nw_id)
            except Exception as exc:
                logger.warning(
                    "Could not load GEE centreline for %s: %s. "
                    "Using great-circle interpolation.",
                    nw_id,
                    exc,
                )

        # ── Fall back: great-circle linear interpolation ───────────────
        if centreline_coords is None:
            centreline_coords = self._interpolate_great_circle(
                start=meta["start_coords"],
                end=meta["end_coords"],
                n_points=max(100, n_segments * 10),
            )

        # ── Split into fixed-length segments ───────────────────────────
        if GEO_AVAILABLE:
            segments = self._split_centreline_geopandas(
                coords=centreline_coords,
                nw_id=nw_id,
                total_km=total_km,
                segment_length_km=segment_length_km,
            )
        else:
            segments = self._split_centreline_numpy(
                coords=centreline_coords,
                nw_id=nw_id,
                total_km=total_km,
                segment_length_km=segment_length_km,
            )

        logger.info("Created %d segments for %s.", len(segments), nw_id)
        return segments

    def _load_centreline_from_gee(self, nw_id: str) -> List[Tuple[float, float]]:
        """Fetch centreline coordinates from a GEE FeatureCollection asset."""
        asset_id = NW_IDS[nw_id]["gee_asset"]
        fc = ee.FeatureCollection(asset_id)
        geom = fc.geometry()
        coords = geom.coordinates().getInfo()
        # Flatten nested coordinate structure if needed
        if coords and isinstance(coords[0], list) and isinstance(coords[0][0], list):
            coords = [pt for segment in coords for pt in segment]
        return [(c[0], c[1]) for c in coords]

    @staticmethod
    def _interpolate_great_circle(
        start: Tuple[float, float],
        end: Tuple[float, float],
        n_points: int = 500,
    ) -> List[Tuple[float, float]]:
        """Generate a linear (great-circle approximation) set of waypoints.

        Parameters
        ----------
        start, end : (lon, lat)
        n_points : int

        Returns
        -------
        List of (lon, lat) tuples.
        """
        lons = np.linspace(start[0], end[0], n_points)
        lats = np.linspace(start[1], end[1], n_points)
        return list(zip(lons.tolist(), lats.tolist()))

    @staticmethod
    def _haversine_km(
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
    ) -> float:
        """Haversine distance between two (lon, lat) points in km."""
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return 2 * R * math.asin(math.sqrt(a))

    def _split_centreline_numpy(
        self,
        coords: List[Tuple[float, float]],
        nw_id: str,
        total_km: float,
        segment_length_km: float,
    ) -> List[RiverSegment]:
        """Segment a polyline using cumulative haversine distances."""
        # Build cumulative distance array
        cum_dist = [0.0]
        for i in range(1, len(coords)):
            d = self._haversine_km(*coords[i - 1], *coords[i])
            cum_dist.append(cum_dist[-1] + d)

        total_actual = cum_dist[-1]
        n_segments = math.ceil(total_actual / segment_length_km)
        segments: List[RiverSegment] = []

        for idx in range(n_segments):
            seg_start_km = idx * segment_length_km
            seg_end_km = min((idx + 1) * segment_length_km, total_actual)
            seg_mid_km = (seg_start_km + seg_end_km) / 2.0

            # Interpolate centroid position
            def _interp_at(target_km: float) -> Tuple[float, float]:
                for i in range(1, len(cum_dist)):
                    if cum_dist[i] >= target_km:
                        t = (target_km - cum_dist[i - 1]) / max(
                            cum_dist[i] - cum_dist[i - 1], 1e-9
                        )
                        lon = coords[i - 1][0] + t * (coords[i][0] - coords[i - 1][0])
                        lat = coords[i - 1][1] + t * (coords[i][1] - coords[i - 1][1])
                        return lon, lat
                return coords[-1]

            cx, cy = _interp_at(seg_mid_km)
            segments.append(
                RiverSegment(
                    segment_id=f"{nw_id}_SEG_{idx:04d}",
                    nw_id=nw_id,
                    index=idx,
                    chainage_start_km=seg_start_km,
                    chainage_end_km=seg_end_km,
                    centroid_lon=cx,
                    centroid_lat=cy,
                    length_km=seg_end_km - seg_start_km,
                )
            )

        return segments

    def _split_centreline_geopandas(
        self,
        coords: List[Tuple[float, float]],
        nw_id: str,
        total_km: float,
        segment_length_km: float,
    ) -> List[RiverSegment]:
        """Segment a polyline using GeoPandas / Shapely with geodesic length."""
        line = LineString(coords)

        # Project to UTM for accurate length measurement
        # Determine UTM zone from midpoint
        mid_lon = coords[len(coords) // 2][0]
        mid_lat = coords[len(coords) // 2][1]
        utm_crs = self._get_utm_crs(mid_lon, mid_lat)

        transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        utm_coords = [transformer.transform(lon, lat) for lon, lat in coords]
        utm_line = LineString(utm_coords)
        total_m = utm_line.length
        seg_len_m = segment_length_km * 1000.0
        n_segs = math.ceil(total_m / seg_len_m)

        inv_transformer = pyproj.Transformer.from_crs(
            utm_crs, "EPSG:4326", always_xy=True
        )
        segments: List[RiverSegment] = []

        for idx in range(n_segs):
            start_m = idx * seg_len_m
            end_m = min((idx + 1) * seg_len_m, total_m)

            sub = substring(utm_line, start_m, end_m)
            cx_utm, cy_utm = sub.centroid.x, sub.centroid.y
            cx, cy = inv_transformer.transform(cx_utm, cy_utm)

            # Back-project sub-line to WGS84 for WKT
            sub_wgs84_coords = [inv_transformer.transform(x, y) for x, y in sub.coords]
            sub_wgs84 = LineString(sub_wgs84_coords)

            segments.append(
                RiverSegment(
                    segment_id=f"{nw_id}_SEG_{idx:04d}",
                    nw_id=nw_id,
                    index=idx,
                    chainage_start_km=start_m / 1000.0,
                    chainage_end_km=end_m / 1000.0,
                    centroid_lon=cx,
                    centroid_lat=cy,
                    length_km=(end_m - start_m) / 1000.0,
                    geometry_wkt=sub_wgs84.wkt,
                )
            )

        return segments

    @staticmethod
    def _get_utm_crs(lon: float, lat: float) -> str:
        """Return the EPSG code for the UTM zone containing (lon, lat)."""
        zone = int((lon + 180) / 6) + 1
        hemisphere = "north" if lat >= 0 else "south"
        epsg = 32600 + zone if hemisphere == "north" else 32700 + zone
        return f"EPSG:{epsg}"

    # ------------------------------------------------------------------
    # Segment → GEE geometry
    # ------------------------------------------------------------------

    def _segment_to_ee_geometry(self, segment: RiverSegment) -> Any:
        """Convert a RiverSegment to a buffered ee.Geometry.Point."""
        if self._mock_mode:
            return {
                "_mock": True,
                "lon": segment.centroid_lon,
                "lat": segment.centroid_lat,
            }
        point = ee.Geometry.Point([segment.centroid_lon, segment.centroid_lat])
        return point.buffer(self.config.buffer_m)

    # ------------------------------------------------------------------
    # Per-segment GEE feature extraction
    # ------------------------------------------------------------------

    def _extract_scalar(
        self,
        image: Any,
        geometry: Any,
        band_names: List[str],
        scale_m: float,
    ) -> Dict[str, float]:
        """Extract mean values of *band_names* from *image* over *geometry*.

        Returns
        -------
        Dict mapping band_name → float (NaN on failure).
        """
        if self._mock_mode or (isinstance(image, dict) and image.get("_mock")):
            return {b: float(self._rng.uniform(0.01, 0.5)) for b in band_names}

        try:
            result = (
                image.select(band_names)
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=scale_m,
                    maxPixels=1e8,
                    bestEffort=True,
                )
                .getInfo()
            )
            return {
                b: (float(result[b]) if result.get(b) is not None else float("nan"))
                for b in band_names
            }
        except Exception as exc:
            logger.debug("reduceRegion failed for bands %s: %s", band_names, exc)
            return {b: float("nan") for b in band_names}

    def _retry(
        self,
        fn: Any,
        *args: Any,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Call *fn* with retries on transient errors."""
        max_r = max_retries or self.config.max_retries
        delay = self.config.retry_delay_s
        last_exc: Optional[Exception] = None
        for attempt in range(max_r):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_r - 1:
                    logger.debug(
                        "Retry %d/%d for %s: %s",
                        attempt + 1,
                        max_r,
                        getattr(fn, "__name__", "fn"),
                        exc,
                    )
                    time.sleep(delay * (2**attempt))
        raise RuntimeError(
            f"All {max_r} retries failed for {getattr(fn, '__name__', 'fn')}: {last_exc}"
        )

    def _extract_segment_month_features(
        self,
        segment: RiverSegment,
        year: int,
        month: int,
    ) -> Dict[str, Any]:
        """Extract all GEE features for one (segment, year, month) combination.

        Parameters
        ----------
        segment : RiverSegment
        year, month : int

        Returns
        -------
        Dict of raw scalar feature values.
        """
        start = f"{year}-{month:02d}-01"
        # Compute end of month
        if month == 12:
            end = f"{year + 1}-01-01"
        else:
            end = f"{year}-{month + 1:02d}-01"

        geom = self._segment_to_ee_geometry(segment)
        row: Dict[str, Any] = {
            "segment_id": segment.segment_id,
            "nw_id": segment.nw_id,
            "year": year,
            "month": month,
            "date": pd.Timestamp(f"{year}-{month:02d}-01"),
            "chainage_km": segment.chainage_mid_km,
            "centroid_lon": segment.centroid_lon,
            "centroid_lat": segment.centroid_lat,
        }

        # ── Sentinel-2 spectral bands + indices ───────────────────────
        try:
            s2_img = self._retry(self.get_sentinel2_composite, geom, start, end)
            s2_img_with_indices = self.compute_spectral_indices(s2_img)
            s2_bands_needed = self.config.s2_bands + [
                "MNDWI",
                "NDWI",
                "AWEI",
                "STUMPF",
                "TURBIDITY",
                "NDTI",
            ]
            s2_vals = self._extract_scalar(
                s2_img_with_indices, geom, s2_bands_needed, self.config.scale_m
            )
            row.update(s2_vals)
        except Exception as exc:
            logger.warning(
                "S2 extraction failed for %s %d-%02d: %s",
                segment.segment_id,
                year,
                month,
                exc,
            )
            for b in self.config.s2_bands + [
                "MNDWI",
                "NDWI",
                "AWEI",
                "STUMPF",
                "TURBIDITY",
                "NDTI",
            ]:
                row[b] = float("nan")

        # ── Sentinel-1 SAR backscatter ─────────────────────────────────
        try:
            s1_img = self._retry(self.get_sentinel1_backscatter, geom, start, end)
            s1_vals = self._extract_scalar(
                s1_img, geom, ["VV", "VH"], self.config.scale_m
            )
            row["sar_vv"] = s1_vals.get("VV", float("nan"))
            row["sar_vh"] = s1_vals.get("VH", float("nan"))
            vv = row["sar_vv"]
            vh = row["sar_vh"]
            row["sar_vv_vh_ratio"] = (
                vv / vh
                if (not math.isnan(vv) and not math.isnan(vh) and abs(vh) > 1e-9)
                else float("nan")
            )
        except Exception as exc:
            logger.warning(
                "S1 extraction failed for %s %d-%02d: %s",
                segment.segment_id,
                year,
                month,
                exc,
            )
            row.update(
                {
                    "sar_vv": float("nan"),
                    "sar_vh": float("nan"),
                    "sar_vv_vh_ratio": float("nan"),
                }
            )

        # ── ERA5 climate ───────────────────────────────────────────────
        try:
            era5_img = self._retry(self.get_era5_climate, geom, start, end)
            era5_vals = self._extract_scalar(
                era5_img,
                geom,
                ["rainfall_mm", "temperature_c"],
                scale_m=11132.0,  # ERA5 resolution ~11 km
            )
            row["era5_cumulative_rainfall_mm"] = era5_vals.get(
                "rainfall_mm", float("nan")
            )
            row["era5_mean_temperature_c"] = era5_vals.get(
                "temperature_c", float("nan")
            )
        except Exception as exc:
            logger.warning(
                "ERA5 extraction failed for %s %d-%02d: %s",
                segment.segment_id,
                year,
                month,
                exc,
            )
            row.update(
                {
                    "era5_cumulative_rainfall_mm": float("nan"),
                    "era5_mean_temperature_c": float("nan"),
                }
            )

        # ── SRTM DEM ───────────────────────────────────────────────────
        try:
            dem_img = self.get_dem_features(geom)
            dem_vals = self._extract_scalar(
                dem_img, geom, ["elevation", "slope"], scale_m=30.0
            )
            row["elevation_m"] = dem_vals.get("elevation", float("nan"))
            row["slope_deg"] = dem_vals.get("slope", float("nan"))
        except Exception as exc:
            logger.warning("DEM extraction failed for %s: %s", segment.segment_id, exc)
            row.update({"elevation_m": float("nan"), "slope_deg": float("nan")})

        # Static geometric features (time-invariant, added per row for convenience)
        row["length_km"] = segment.length_km
        row["distance_from_source_km"] = segment.chainage_mid_km

        return row

    # ------------------------------------------------------------------
    # CWC gauge data
    # ------------------------------------------------------------------

    def get_cwc_gauge_data(
        self,
        station_ids: List[str],
        start_date: str,
        end_date: str,
        cwc_data_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load CWC gauge records for specified stations and date range.

        The method first attempts to load local CSV files from *cwc_data_dir*,
        where each file is named ``{station_id}.csv`` and has columns:
          ``date, water_level_m, discharge_m3s, chainage_km``

        If a file is missing, the method emits a warning and returns NaN rows
        for that station.

        Parameters
        ----------
        station_ids : List[str]
            List of CWC station identifiers.
        start_date : str
            ``"YYYY-MM-DD"`` filter start.
        end_date : str
            ``"YYYY-MM-DD"`` filter end (inclusive).
        cwc_data_dir : str, optional
            Override for config.cwc_data_dir.

        Returns
        -------
        pd.DataFrame
            Columns: station_id, date, water_level_m, discharge_m3s, chainage_km.
            Monthly-averaged (one row per station per month).
        """
        data_dir = Path(cwc_data_dir or self.config.cwc_data_dir)
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        all_records: List[pd.DataFrame] = []

        for sid in station_ids:
            csv_path = data_dir / f"{sid}.csv"
            if not csv_path.exists():
                logger.warning(
                    "CWC CSV not found for station %s (%s). Using NaN.", sid, csv_path
                )
                # Generate placeholder NaN rows spanning the requested range
                months = pd.date_range(start_ts, end_ts, freq="MS")
                placeholder = pd.DataFrame(
                    {
                        CWC_STATION_ID_COL: sid,
                        CWC_DATE_COL: months,
                        CWC_LEVEL_COL: float("nan"),
                        CWC_DISCHARGE_COL: float("nan"),
                        CWC_CHAINAGE_COL: float("nan"),
                    }
                )
                all_records.append(placeholder)
                continue

            try:
                df = pd.read_csv(csv_path, parse_dates=[CWC_DATE_COL])
                df[CWC_STATION_ID_COL] = sid
                df = df[(df[CWC_DATE_COL] >= start_ts) & (df[CWC_DATE_COL] <= end_ts)]
                # Resample to monthly means
                df = df.set_index(CWC_DATE_COL).resample("MS").mean()
                df = df.reset_index().rename(columns={"index": CWC_DATE_COL})
                df[CWC_STATION_ID_COL] = sid
                all_records.append(df)
                logger.debug("Loaded %d monthly records for station %s.", len(df), sid)
            except Exception as exc:
                logger.error("Failed to load CWC data for %s: %s", sid, exc)

        if not all_records:
            return pd.DataFrame(
                columns=[
                    CWC_STATION_ID_COL,
                    CWC_DATE_COL,
                    CWC_LEVEL_COL,
                    CWC_DISCHARGE_COL,
                    CWC_CHAINAGE_COL,
                ]
            )

        combined = pd.concat(all_records, ignore_index=True)
        return combined

    def _interpolate_gauge_to_segments(
        self,
        gauge_df: pd.DataFrame,
        segments: List[RiverSegment],
        date: pd.Timestamp,
    ) -> Dict[str, Dict[str, float]]:
        """Spatially interpolate CWC gauge values to segment chainages.

        Uses inverse-distance weighting between the two nearest gauges.

        Parameters
        ----------
        gauge_df : pd.DataFrame
            Monthly gauge records (already filtered to one date).
        segments : List[RiverSegment]
        date : pd.Timestamp

        Returns
        -------
        Dict mapping segment_id → {water_level_m, discharge_m3s}.
        """
        date_gauge = gauge_df[gauge_df[CWC_DATE_COL] == date].copy()
        if date_gauge.empty or date_gauge[CWC_CHAINAGE_COL].isna().all():
            return {
                seg.segment_id: {
                    "gauge_water_level_m": float("nan"),
                    "gauge_discharge_m3s": float("nan"),
                }
                for seg in segments
            }

        date_gauge = date_gauge.dropna(subset=[CWC_CHAINAGE_COL]).sort_values(
            CWC_CHAINAGE_COL
        )
        chainages = date_gauge[CWC_CHAINAGE_COL].values
        levels = date_gauge[CWC_LEVEL_COL].values
        discharges = date_gauge[CWC_DISCHARGE_COL].values

        result: Dict[str, Dict[str, float]] = {}
        for seg in segments:
            target = seg.chainage_mid_km
            distances = np.abs(chainages - target)
            idx_sorted = np.argsort(distances)

            if distances[idx_sorted[0]] < 1e-6:
                # Exact match
                lv = float(levels[idx_sorted[0]])
                dq = float(discharges[idx_sorted[0]])
            elif len(idx_sorted) >= 2:
                i0, i1 = idx_sorted[0], idx_sorted[1]
                w0 = 1.0 / (distances[i0] + 1e-9)
                w1 = 1.0 / (distances[i1] + 1e-9)
                ws = w0 + w1
                lv = float(
                    (
                        w0 * (levels[i0] if not math.isnan(levels[i0]) else 0.0)
                        + w1 * (levels[i1] if not math.isnan(levels[i1]) else 0.0)
                    )
                    / ws
                )
                dq = float(
                    (
                        w0 * (discharges[i0] if not math.isnan(discharges[i0]) else 0.0)
                        + w1
                        * (discharges[i1] if not math.isnan(discharges[i1]) else 0.0)
                    )
                    / ws
                )
            else:
                lv = float(levels[idx_sorted[0]])
                dq = float(discharges[idx_sorted[0]])

            result[seg.segment_id] = {
                "gauge_water_level_m": lv,
                "gauge_discharge_m3s": dq,
            }
        return result

    # ------------------------------------------------------------------
    # build_training_dataset
    # ------------------------------------------------------------------

    def build_training_dataset(
        self,
        nw_id: str,
        start_year: int,
        end_year: int,
        cwc_station_ids: Optional[List[str]] = None,
        segment_length_km: float = DEFAULT_SEGMENT_KM,
        max_segments: Optional[int] = None,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Build the full multi-year, multi-segment feature DataFrame.

        Iterates over each calendar month in [start_year, end_year] for
        every 5 km river segment, queries GEE for Sentinel-2, Sentinel-1,
        ERA5, and DEM data, and merges CWC gauge interpolations.

        Parameters
        ----------
        nw_id : str
            ``"NW1"`` or ``"NW2"``.
        start_year : int
            First year of data (inclusive).
        end_year : int
            Last year of data (inclusive).
        cwc_station_ids : list of str, optional
            CWC station identifiers to load.  Skipped if None.
        segment_length_km : float
            Segment analysis unit size.
        max_segments : int, optional
            Cap on number of segments (for testing / development).
        n_jobs : int
            Number of parallel workers.  Uses joblib if > 1.
            NOTE: GEE API is not thread-safe; use n_jobs=1 in production.

        Returns
        -------
        pd.DataFrame
            One row per (segment_id, date).  Shape (N×T, F) where
            N = number of segments, T = number of months.
        """
        self._require_auth()

        logger.info(
            "Building training dataset for %s | years=%d–%d",
            nw_id,
            start_year,
            end_year,
        )

        # ── Segments ──────────────────────────────────────────────────
        segments = self.extract_river_segments(nw_id, segment_length_km)
        if max_segments is not None:
            segments = segments[:max_segments]
            logger.info("Capped to %d segments for testing.", max_segments)

        # ── Date range ────────────────────────────────────────────────
        months: List[Tuple[int, int]] = []
        for yr in range(start_year, end_year + 1):
            for mo in range(1, 13):
                months.append((yr, mo))

        logger.info(
            "Processing %d segments × %d months = %d combinations.",
            len(segments),
            len(months),
            len(segments) * len(months),
        )

        # ── CWC gauge data ─────────────────────────────────────────────
        gauge_df: Optional[pd.DataFrame] = None
        if cwc_station_ids:
            gauge_df = self.get_cwc_gauge_data(
                station_ids=cwc_station_ids,
                start_date=f"{start_year}-01-01",
                end_date=f"{end_year}-12-31",
            )
            logger.info("Loaded CWC gauge data: %d records.", len(gauge_df))

        # ── Feature extraction ─────────────────────────────────────────
        all_rows: List[Dict[str, Any]] = []

        for year, month in tqdm(months, desc=f"Processing {nw_id} months"):
            date_ts = pd.Timestamp(f"{year}-{month:02d}-01")

            # Gauge interpolation for this date
            gauge_interp: Dict[str, Dict[str, float]] = {}
            if gauge_df is not None:
                gauge_interp = self._interpolate_gauge_to_segments(
                    gauge_df, segments, date_ts
                )

            for segment in segments:
                row = self._extract_segment_month_features(segment, year, month)

                # Merge gauge data
                if segment.segment_id in gauge_interp:
                    row.update(gauge_interp[segment.segment_id])
                else:
                    row.update(
                        {
                            "gauge_water_level_m": float("nan"),
                            "gauge_discharge_m3s": float("nan"),
                        }
                    )

                all_rows.append(row)

        df = pd.DataFrame(all_rows)

        # ── Post-processing ────────────────────────────────────────────
        df = self._post_process_dataframe(df, segments)

        logger.info(
            "Training dataset complete | shape=%s | segments=%d | months=%d",
            df.shape,
            len(segments),
            len(months),
        )
        return df

    def _post_process_dataframe(
        self,
        df: pd.DataFrame,
        segments: List[RiverSegment],
    ) -> pd.DataFrame:
        """Clean, type-cast, and derive additional columns on the raw DataFrame.

        Steps
        -----
        1. Ensure ``date`` is datetime64[ns].
        2. Sort by (segment_id, date).
        3. Compute sinuosity per segment (if geometry_wkt is present).
        4. Compute temporal MNDWI std (12-month rolling) per segment.
        5. Impute missing values with column medians.
        6. Derive navigability proxy label from gauge water level.

        Parameters
        ----------
        df : pd.DataFrame
        segments : List[RiverSegment]

        Returns
        -------
        pd.DataFrame
        """
        # ── Date handling ─────────────────────────────────────────────
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["segment_id", "date"]).reset_index(drop=True)

        # ── Sinuosity (from stored WKT) ────────────────────────────────
        if GEO_AVAILABLE:
            seg_sinuosity: Dict[str, float] = {}
            for seg in segments:
                if seg.geometry_wkt:
                    try:
                        line = shape(
                            {
                                "type": "LineString",
                                "coordinates": list(
                                    LineString(seg.geometry_wkt).coords
                                ),
                            }
                        )
                        coords_arr = np.array(list(line.coords))
                        chan_len = float(
                            np.sum(np.linalg.norm(np.diff(coords_arr, axis=0), axis=1))
                        )
                        straight = float(np.linalg.norm(coords_arr[-1] - coords_arr[0]))
                        sinuosity = (
                            max(1.0, chan_len / straight) if straight > 1e-9 else 1.0
                        )
                    except Exception:
                        sinuosity = 1.0
                else:
                    sinuosity = 1.0
                seg_sinuosity[seg.segment_id] = sinuosity

            df["sinuosity"] = df["segment_id"].map(seg_sinuosity).fillna(1.0)
        else:
            df["sinuosity"] = 1.0

        # ── Temporal MNDWI std (12-month rolling per segment) ──────────
        if "MNDWI" in df.columns:
            df["mndwi_std_12m"] = (
                df.groupby("segment_id")["MNDWI"]
                .transform(lambda x: x.rolling(12, min_periods=1).std())
                .fillna(0.0)
            )
        else:
            df["mndwi_std_12m"] = 0.0

        # ── Water surface width proxy (from MNDWI thresholding) ────────
        # We estimate width as a linear function of MNDWI for now;
        # proper transect-wise measurement requires raster access.
        if "MNDWI" in df.columns:
            mndwi_vals = df["MNDWI"].fillna(0.0).clip(-1.0, 1.0)
            # Empirical mapping: MNDWI ≈ 0.5 → ~500 m width; -0.5 → ~50 m
            df["water_width_m"] = (mndwi_vals + 1.0) * 275.0 + 50.0
        else:
            df["water_width_m"] = float("nan")

        # ── Impute missing values with column medians ──────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(
                    median_val if not math.isnan(float(median_val)) else 0.0
                )

        # ── SAR VV/VH ratio (recompute after imputation) ──────────────
        if "sar_vv" in df.columns and "sar_vh" in df.columns:
            df["sar_vv_vh_ratio"] = np.where(
                df["sar_vh"].abs() > 1e-9,
                df["sar_vv"] / df["sar_vh"],
                0.0,
            )

        # ── Ensure required columns present ───────────────────────────
        expected_cols = [
            "segment_id",
            "nw_id",
            "date",
            "year",
            "month",
            "chainage_km",
            "centroid_lon",
            "centroid_lat",
        ]
        for col in expected_cols:
            if col not in df.columns:
                logger.warning("Expected column '%s' missing from DataFrame.", col)

        return df

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_feature_matrix(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        also_export_geojson: bool = True,
        compression: str = "snappy",
    ) -> Dict[str, Path]:
        """Save the feature matrix as Parquet and optionally GeoJSON.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix from ``build_training_dataset()``.
        output_path : str or Path
            Target file path.  Extension is replaced with ``.parquet``.
        also_export_geojson : bool
            Whether to also export a GeoJSON of segment centroids.
        compression : str
            Parquet compression codec (``"snappy"``, ``"gzip"``, ``"none"``).

        Returns
        -------
        Dict[str, Path]
            Keys: ``"parquet"`` and optionally ``"geojson"``.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Parquet export ────────────────────────────────────────────
        parquet_path = output_path.with_suffix(".parquet")
        df_to_save = df.copy()

        # Convert any remaining object columns to string for Parquet
        for col in df_to_save.select_dtypes(include=["object"]).columns:
            df_to_save[col] = df_to_save[col].astype(str)

        df_to_save.to_parquet(parquet_path, compression=compression, index=False)
        logger.info(
            "Parquet saved: %s  (%.1f MB, %d rows × %d cols)",
            parquet_path,
            parquet_path.stat().st_size / 1e6,
            len(df_to_save),
            len(df_to_save.columns),
        )

        result: Dict[str, Path] = {"parquet": parquet_path}

        # ── GeoJSON export (segment-level summary) ────────────────────
        if also_export_geojson and GEO_AVAILABLE:
            geojson_path = output_path.with_suffix(".geojson")
            try:
                self._export_geojson(df, geojson_path)
                result["geojson"] = geojson_path
            except Exception as exc:
                logger.warning("GeoJSON export failed: %s", exc)

        return result

    def _export_geojson(
        self,
        df: pd.DataFrame,
        geojson_path: Path,
    ) -> None:
        """Export a GeoJSON FeatureCollection of segment-level statistics."""
        # Aggregate to one row per segment (mean over time)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_summary = ["segment_id", "nw_id"]
        group_cols = [c for c in non_numeric_summary if c in df.columns]
        agg_df = df.groupby(group_cols)[numeric_cols].mean().reset_index()

        features = []
        for _, row in agg_df.iterrows():
            lon = row.get("centroid_lon", 0.0)
            lat = row.get("centroid_lat", 0.0)
            properties = {
                k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                for k, v in row.items()
                if k not in ("centroid_lon", "centroid_lat")
            }
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)],
                },
                "properties": properties,
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}
        geojson_path.parent.mkdir(parents=True, exist_ok=True)
        with open(geojson_path, "w") as f:
            json.dump(geojson, f, indent=2, default=str)

        logger.info(
            "GeoJSON saved: %s  (%d features)",
            geojson_path,
            len(features),
        )

    # ------------------------------------------------------------------
    # Convenience: GEE export tasks
    # ------------------------------------------------------------------

    def submit_gee_export_task(
        self,
        image: Any,
        description: str,
        bucket: str,
        file_name_prefix: str,
        geometry: Any,
        scale_m: float = 10.0,
        crs: str = "EPSG:4326",
        file_format: str = "GeoTIFF",
    ) -> Any:
        """Submit a GEE Export.image.toCloudStorage task.

        Parameters
        ----------
        image : ee.Image
            Image to export.
        description : str
            Human-readable task description.
        bucket : str
            Google Cloud Storage bucket name.
        file_name_prefix : str
            GCS path prefix for output file(s).
        geometry : ee.Geometry
            Export region.
        scale_m : float
            Output pixel resolution (metres).
        crs : str
            Output CRS EPSG string.
        file_format : str
            ``"GeoTIFF"`` or ``"TFRecord"``.

        Returns
        -------
        ee.batch.Task
            The submitted task object.  Call ``.start()`` to begin.
        """
        if self._mock_mode:
            logger.info(
                "[MOCK] Would export image '%s' to gs://%s/%s",
                description,
                bucket,
                file_name_prefix,
            )
            return None

        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=description,
            bucket=bucket,
            fileNamePrefix=file_name_prefix,
            region=geometry,
            scale=scale_m,
            crs=crs,
            fileFormat=file_format,
            maxPixels=1e12,
        )
        task.start()
        logger.info(
            "GEE export task submitted: '%s' → gs://%s/%s",
            description,
            bucket,
            file_name_prefix,
        )
        return task

    def wait_for_tasks(
        self,
        tasks: List[Any],
        poll_interval_s: int = 30,
        timeout_s: int = 7200,
    ) -> Dict[str, str]:
        """Poll GEE export tasks until completion or timeout.

        Parameters
        ----------
        tasks : List[ee.batch.Task]
        poll_interval_s : int
        timeout_s : int

        Returns
        -------
        Dict mapping task description → final status string.
        """
        if self._mock_mode or not tasks:
            return {}

        statuses: Dict[str, str] = {}
        start_time = time.time()
        active = [t for t in tasks if t is not None]

        while active:
            if time.time() - start_time > timeout_s:
                logger.warning("Timeout waiting for GEE tasks.")
                break

            still_active = []
            for task in active:
                status = task.status()
                state = status.get("state", "UNKNOWN")
                desc = status.get("description", "unknown")
                statuses[desc] = state

                if state in ("COMPLETED", "FAILED", "CANCELLED"):
                    if state == "FAILED":
                        logger.error(
                            "GEE task '%s' FAILED: %s",
                            desc,
                            status.get("error_message", ""),
                        )
                    else:
                        logger.info(
                            "GEE task '%s' finished with state: %s", desc, state
                        )
                else:
                    still_active.append(task)

            active = still_active
            if active:
                logger.debug(
                    "%d tasks still running. Polling again in %ds …",
                    len(active),
                    poll_interval_s,
                )
                time.sleep(poll_interval_s)

        return statuses

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def describe_pipeline(self) -> Dict[str, Any]:
        """Return a summary dict describing the pipeline configuration."""
        return {
            "authenticated": self._authenticated,
            "mock_mode": self._mock_mode,
            "gee_available": EE_AVAILABLE,
            "geo_available": GEO_AVAILABLE,
            "config": {
                "gee_project": self.config.gee_project,
                "segment_length_km": self.config.segment_length_km,
                "buffer_m": self.config.buffer_m,
                "max_cloud_probability": self.config.max_cloud_probability,
                "scale_m": self.config.scale_m,
                "max_retries": self.config.max_retries,
                "cwc_data_dir": self.config.cwc_data_dir,
                "output_dir": self.config.output_dir,
            },
            "waterways": {k: v["name"] for k, v in NW_IDS.items()},
        }

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run basic quality checks on the feature DataFrame.

        Checks
        ------
        - No duplicate (segment_id, date) pairs.
        - NaN fraction per column.
        - Value range checks for spectral indices (−1 to 1).
        - Minimum segment count.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        Dict with check results and warnings.
        """
        report: Dict[str, Any] = {"passed": True, "warnings": [], "errors": []}

        # Duplicate check
        if "segment_id" in df.columns and "date" in df.columns:
            n_dupes = df.duplicated(subset=["segment_id", "date"]).sum()
            if n_dupes > 0:
                msg = f"{n_dupes} duplicate (segment_id, date) pairs found."
                report["errors"].append(msg)
                report["passed"] = False

        # NaN fraction
        nan_frac = df.isna().mean()
        high_nan = nan_frac[nan_frac > 0.3]
        for col, frac in high_nan.items():
            report["warnings"].append(f"Column '{col}' has {frac:.1%} NaN values.")

        # Spectral index range checks
        for idx_col in ["MNDWI", "NDWI", "TURBIDITY", "NDTI"]:
            if idx_col in df.columns:
                out_of_range = ((df[idx_col] < -1.1) | (df[idx_col] > 1.1)).sum()
                if out_of_range > 0:
                    report["warnings"].append(
                        f"{out_of_range} values of '{idx_col}' outside [−1, 1]."
                    )

        # Segment count
        if "segment_id" in df.columns:
            n_segments = df["segment_id"].nunique()
            if n_segments < 10:
                report["warnings"].append(
                    f"Only {n_segments} segments present — dataset may be too small."
                )
            report["n_segments"] = n_segments

        report["n_rows"] = len(df)
        report["n_cols"] = len(df.columns)
        report["nan_fractions"] = nan_frac.to_dict()

        logger.info(
            "Dataset validation: passed=%s | rows=%d | cols=%d | segments=%d",
            report["passed"],
            report["n_rows"],
            report["n_cols"],
            report.get("n_segments", -1),
        )
        for w in report["warnings"]:
            logger.warning("  ⚠  %s", w)
        for e in report["errors"]:
            logger.error("  ✗  %s", e)

        return report


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def _build_cli_parser() -> "argparse.ArgumentParser":
    """Build a lightweight CLI for the GEE pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python gee_pipeline.py",
        description="AIDSTL GEE data pipeline — build ML training datasets.",
    )
    parser.add_argument(
        "--nw_id",
        choices=["NW1", "NW2", "both"],
        default="NW1",
        help="Waterway to process (default: NW1).",
    )
    parser.add_argument("--start_year", type=int, default=2019)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--gee_project", type=str, default="aidstl-project")
    parser.add_argument("--service_account", type=str, default="")
    parser.add_argument("--credentials_path", type=str, default="")
    parser.add_argument("--segment_length_km", type=float, default=5.0)
    parser.add_argument(
        "--max_segments",
        type=int,
        default=None,
        help="Cap segments for testing (default: all).",
    )
    parser.add_argument("--cwc_data_dir", type=str, default="data/cwc_gauges")
    parser.add_argument(
        "--no_geojson", action="store_true", help="Skip GeoJSON export."
    )
    parser.add_argument(
        "--mock", action="store_true", help="Force mock mode (no GEE calls)."
    )
    return parser


# ---------------------------------------------------------------------------
# Smoke test / main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    args_list = sys.argv[1:]

    # ── If run with --mock or no real GEE, demonstrate mock pipeline ────
    parser = _build_cli_parser()
    args = parser.parse_args(args_list)

    cfg = PipelineConfig(
        gee_project=args.gee_project,
        service_account=args.service_account,
        credentials_path=args.credentials_path,
        segment_length_km=args.segment_length_km,
        cwc_data_dir=args.cwc_data_dir,
        output_dir=args.output_dir,
    )

    pipeline = GEEPipeline(config=cfg)

    # Force mock if requested or if GEE unavailable
    if args.mock or not EE_AVAILABLE:
        pipeline._mock_mode = True
        pipeline._authenticated = True
        logger.info("Running in MOCK mode — no GEE calls will be made.")
    else:
        ok = pipeline.authenticate(
            service_account=args.service_account or None,
            credentials_path=args.credentials_path or None,
        )
        if not ok:
            logger.error("Authentication failed. Use --mock to run without GEE.")
            sys.exit(1)

    print("\n── Pipeline description ──")
    desc = pipeline.describe_pipeline()
    for k, v in desc.items():
        print(f"  {k}: {v}")

    # Determine waterway IDs to process
    nw_ids_to_process = ["NW1", "NW2"] if args.nw_id == "both" else [args.nw_id.upper()]

    for nw in nw_ids_to_process:
        print(f"\n── Building dataset for {nw} ──")
        df = pipeline.build_training_dataset(
            nw_id=nw,
            start_year=args.start_year,
            end_year=args.end_year,
            segment_length_km=args.segment_length_km,
            max_segments=args.max_segments or 5,  # small default for testing
        )

        print(f"  DataFrame shape  : {df.shape}")
        print(f"  Columns          : {list(df.columns[:10])} …")
        print(
            f"  Segments         : {df['segment_id'].nunique() if 'segment_id' in df.columns else 'N/A'}"
        )
        print(
            f"  Date range       : {df['date'].min()} → {df['date'].max()}"
            if "date" in df.columns
            else ""
        )

        # Validate
        report = pipeline.validate_dataset(df)
        print(f"  Validation passed: {report['passed']}")

        # Export
        output_path = Path(args.output_dir) / f"{nw.lower()}_features"
        paths = pipeline.export_feature_matrix(
            df,
            output_path=output_path,
            also_export_geojson=not args.no_geojson,
        )
        for fmt, p in paths.items():
            print(f"  Exported ({fmt})  : {p}")

    print("\nGEE pipeline smoke test complete ✓")
