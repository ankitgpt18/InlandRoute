"""
AIDSTL Project — Spatial Utility Functions
==========================================
River segmentation, geometry operations, and spatial analysis helpers
for the Inland Waterway Navigability prediction system.

Study areas
-----------
  NW-1 : Ganga       — Varanasi → Haldia   (~1,620 km)
  NW-2 : Brahmaputra — Dhubri  → Sadiya    (~891 km)

All geometries are stored and processed in EPSG:4326 (WGS 84) unless
explicitly projected for distance/area calculations, in which case
EPSG:32644 (UTM Zone 44N) is used — appropriate for both study areas.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, split, transform, unary_union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coordinate Reference Systems
# ---------------------------------------------------------------------------

WGS84_EPSG: int = 4326
UTM_EPSG: int = 32644  # UTM Zone 44N — covers both NW-1 and NW-2

CRS_WGS84 = CRS.from_epsg(WGS84_EPSG)
CRS_UTM = CRS.from_epsg(UTM_EPSG)

# Reusable transformers (always_xy=True ensures lon/lat ordering)
_wgs84_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)
_utm_to_wgs84 = Transformer.from_crs(CRS_UTM, CRS_WGS84, always_xy=True)


def _to_utm(geom: BaseGeometry) -> BaseGeometry:
    """Project a WGS-84 geometry to UTM Zone 44N (metres)."""
    return transform(_wgs84_to_utm.transform, geom)


def _to_wgs84(geom: BaseGeometry) -> BaseGeometry:
    """Reproject a UTM Zone 44N geometry back to WGS-84."""
    return transform(_utm_to_wgs84.transform, geom)


# ---------------------------------------------------------------------------
# Waterway metadata
# ---------------------------------------------------------------------------

WATERWAY_META: dict[str, dict[str, Any]] = {
    "NW-1": {
        "name": "National Waterway 1 — Ganga",
        "from": "Varanasi",
        "to": "Haldia",
        "length_km": 1620,
        "bbox": [83.0, 21.5, 88.5, 25.5],  # [minLon, minLat, maxLon, maxLat]
    },
    "NW-2": {
        "name": "National Waterway 2 — Brahmaputra",
        "from": "Dhubri",
        "to": "Sadiya",
        "length_km": 891,
        "bbox": [89.5, 26.5, 95.8, 28.2],
    },
}


def get_waterway_meta(waterway_id: str) -> dict[str, Any]:
    """Return metadata dictionary for a supported waterway.

    Parameters
    ----------
    waterway_id : str
        One of "NW-1" or "NW-2".

    Returns
    -------
    dict[str, Any]
        Metadata including name, terminal points, length and bounding box.

    Raises
    ------
    ValueError
        If *waterway_id* is not in ``WATERWAY_META``.
    """
    if waterway_id not in WATERWAY_META:
        raise ValueError(
            f"Unknown waterway '{waterway_id}'. Supported: {list(WATERWAY_META.keys())}"
        )
    return WATERWAY_META[waterway_id]


# ---------------------------------------------------------------------------
# Segment ID generation
# ---------------------------------------------------------------------------


def make_segment_id(waterway_id: str, index: int) -> str:
    """Generate a deterministic, zero-padded segment identifier.

    Parameters
    ----------
    waterway_id : str
        E.g. "NW-1".
    index : int
        Zero-based segment index along the river centreline.

    Returns
    -------
    str
        E.g. "NW-1-042".

    Examples
    --------
    >>> make_segment_id("NW-1", 0)
    'NW-1-000'
    >>> make_segment_id("NW-2", 178)
    'NW-2-178'
    """
    return f"{waterway_id}-{index:03d}"


def parse_segment_id(segment_id: str) -> tuple[str, int]:
    """Split a segment identifier back into its components.

    Parameters
    ----------
    segment_id : str
        E.g. "NW-1-042".

    Returns
    -------
    tuple[str, int]
        *(waterway_id, index)* — e.g. ``("NW-1", 42)``.

    Raises
    ------
    ValueError
        If the string does not match the expected pattern.
    """
    parts = segment_id.rsplit("-", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(
            f"Invalid segment ID '{segment_id}'. Expected format: 'NW-X-NNN'."
        )
    waterway_id = parts[0]  # e.g. "NW-1"
    index = int(parts[1])
    return waterway_id, index


# ---------------------------------------------------------------------------
# River segmentation
# ---------------------------------------------------------------------------


def _ensure_single_linestring(geom: BaseGeometry) -> LineString:
    """Merge a MultiLineString into a single LineString, or pass through."""
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = unary_union(geom)
        if isinstance(merged, LineString):
            return merged
        # Merge by concatenating coordinate sequences in order
        coords: list[tuple[float, float]] = []
        for line in merged.geoms:
            coords.extend(line.coords)
        return LineString(coords)
    raise TypeError(
        f"Expected LineString or MultiLineString, got {type(geom).__name__}"
    )


def _split_line_at_distance(
    line: LineString, segment_length_m: float
) -> list[LineString]:
    """Cut *line* (in metres) into sub-lines of at most *segment_length_m*.

    The final segment may be shorter than *segment_length_m*.

    Parameters
    ----------
    line : LineString
        A LineString in a **projected** CRS (metres).
    segment_length_m : float
        Target segment length in metres.

    Returns
    -------
    list[LineString]
        Ordered list of LineString segments.
    """
    total_length = line.length
    segments: list[LineString] = []
    distance = 0.0

    while distance < total_length:
        start_pt = line.interpolate(distance)
        end_dist = min(distance + segment_length_m, total_length)
        end_pt = line.interpolate(end_dist)

        # Extract the sub-string between the two distances
        # Shapely ≥ 2.0: use substring from shapely.ops
        from shapely.ops import substring  # local import to be explicit

        seg = substring(line, distance, end_dist)
        if seg and not seg.is_empty and isinstance(seg, LineString):
            segments.append(seg)
        distance = end_dist

    return segments


def segment_river(
    geojson_path: Union[str, Path],
    segment_length_km: float = 5.0,
    waterway_id: Optional[str] = None,
    crs_out: int = WGS84_EPSG,
) -> gpd.GeoDataFrame:
    """Segment a river centreline GeoJSON into fixed-length analysis units.

    The centreline is first projected to UTM Zone 44N for accurate distance
    calculations, divided into *segment_length_km* km sub-lines, and then
    reprojected back to WGS-84 for storage.

    Parameters
    ----------
    geojson_path : str | Path
        Path to a GeoJSON file containing the river centreline.  Must have
        at least one ``LineString`` or ``MultiLineString`` feature.
    segment_length_km : float, optional
        Target length of each segment in kilometres (default: 5 km).
    waterway_id : str, optional
        Waterway identifier ("NW-1" | "NW-2") used when generating segment IDs.
        Inferred from the GeoJSON ``waterway_id`` property if not provided.
    crs_out : int, optional
        EPSG code for the output GeoDataFrame (default: 4326).

    Returns
    -------
    gpd.GeoDataFrame
        One row per segment with columns:
        ``segment_id``, ``waterway_id``, ``segment_index``,
        ``length_km``, ``centroid_lon``, ``centroid_lat``,
        ``chainage_start_km``, ``chainage_end_km``, ``geometry``.

    Raises
    ------
    FileNotFoundError
        If *geojson_path* does not exist.
    ValueError
        If no valid centreline geometry is found in the file.
    """
    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"No features found in {path}.")

    # Dissolve all features into a single centreline
    line_features = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    if line_features.empty:
        raise ValueError(
            "GeoJSON does not contain any LineString or MultiLineString features."
        )

    merged_geom = unary_union(line_features.geometry.values)
    centreline = _ensure_single_linestring(merged_geom)

    # Determine waterway_id
    if waterway_id is None:
        if "waterway_id" in gdf.columns and not gdf["waterway_id"].isnull().all():
            waterway_id = str(gdf["waterway_id"].iloc[0])
        else:
            # Attempt to derive from filename
            stem = path.stem.upper()
            if "NW-1" in stem or "NW1" in stem:
                waterway_id = "NW-1"
            elif "NW-2" in stem or "NW2" in stem:
                waterway_id = "NW-2"
            else:
                waterway_id = "UNKNOWN"
        logger.info("Inferred waterway_id='%s' from file.", waterway_id)

    # Project to UTM for metric calculations
    centreline_utm = _to_utm(centreline)
    segment_length_m = segment_length_km * 1000.0

    logger.info(
        "Segmenting %.1f km centreline into %.1f km chunks …",
        centreline_utm.length / 1000.0,
        segment_length_km,
    )

    segments_utm = _split_line_at_distance(centreline_utm, segment_length_m)

    records: list[dict[str, Any]] = []
    chainage = 0.0

    for idx, seg_utm in enumerate(segments_utm):
        seg_len_km = seg_utm.length / 1000.0
        seg_wgs84 = _to_wgs84(seg_utm)
        centroid = seg_wgs84.centroid

        records.append(
            {
                "segment_id": make_segment_id(waterway_id, idx),
                "waterway_id": waterway_id,
                "segment_index": idx,
                "length_km": round(seg_len_km, 4),
                "centroid_lon": round(centroid.x, 6),
                "centroid_lat": round(centroid.y, 6),
                "chainage_start_km": round(chainage, 3),
                "chainage_end_km": round(chainage + seg_len_km, 3),
                "geometry": seg_wgs84,
            }
        )
        chainage += seg_len_km

    result = gpd.GeoDataFrame(records, crs=f"EPSG:{WGS84_EPSG}")

    if crs_out != WGS84_EPSG:
        result = result.to_crs(epsg=crs_out)

    logger.info(
        "Segmentation complete: %d segments for waterway '%s'.",
        len(result),
        waterway_id,
    )
    return result


# ---------------------------------------------------------------------------
# Gauge-to-segment interpolation
# ---------------------------------------------------------------------------


def interpolate_gauge_to_segment(
    segment: Union[gpd.GeoSeries, dict[str, Any]],
    gauges: gpd.GeoDataFrame,
    value_col: str = "water_level_m",
    max_distance_km: float = 50.0,
    method: str = "idw",
    idw_power: float = 2.0,
) -> float:
    """Interpolate gauge measurements to a river segment centroid.

    Uses Inverse Distance Weighting (IDW) by default with a configurable
    fallback to nearest-neighbour interpolation.

    Parameters
    ----------
    segment : GeoSeries | dict
        Segment row containing at minimum ``centroid_lon`` and ``centroid_lat``
        *or* a Shapely geometry under key ``"geometry"``.
    gauges : gpd.GeoDataFrame
        Point GeoDataFrame of gauge stations.  Must contain *value_col* and
        a valid point geometry column.
    value_col : str
        Column in *gauges* holding the measured quantity (e.g. water level).
    max_distance_km : float
        Gauges beyond this distance are excluded from the interpolation.
    method : str
        ``"idw"`` — inverse distance weighting (default).
        ``"nearest"`` — value from closest gauge only.
    idw_power : float
        Power parameter *p* for IDW (higher = more local influence).

    Returns
    -------
    float
        Interpolated value at the segment centroid, or ``float("nan")``
        if no gauges are within *max_distance_km*.

    Raises
    ------
    ValueError
        If *value_col* is not present in *gauges*.
    """
    if value_col not in gauges.columns:
        raise ValueError(
            f"Column '{value_col}' not found in gauges GeoDataFrame. "
            f"Available: {list(gauges.columns)}"
        )

    # Resolve segment centroid
    if isinstance(segment, dict):
        if "centroid_lon" in segment and "centroid_lat" in segment:
            centroid = Point(segment["centroid_lon"], segment["centroid_lat"])
        elif "geometry" in segment:
            centroid = shape(segment["geometry"]).centroid
        else:
            raise ValueError("segment must contain centroid coordinates or geometry.")
    else:
        # GeoSeries row
        if hasattr(segment, "centroid_lon"):
            centroid = Point(segment["centroid_lon"], segment["centroid_lat"])
        else:
            centroid = segment.geometry.centroid

    # Project both to UTM for metric distances
    centroid_utm = _to_utm(centroid)
    gauges_utm = gauges.to_crs(epsg=UTM_EPSG)

    max_dist_m = max_distance_km * 1000.0
    valid_gauges = gauges_utm[~gauges_utm[value_col].isna()].copy()

    if valid_gauges.empty:
        logger.warning("No valid gauge readings; returning NaN.")
        return float("nan")

    # Compute distances from segment centroid to all gauges
    distances = valid_gauges.geometry.apply(
        lambda g: centroid_utm.distance(g)
    ).values.astype(float)

    within_mask = distances <= max_dist_m
    if not within_mask.any():
        logger.debug(
            "No gauges within %.1f km of segment centroid; returning NaN.",
            max_distance_km,
        )
        return float("nan")

    d = distances[within_mask]
    v = valid_gauges[value_col].values[within_mask].astype(float)

    if method == "nearest":
        return float(v[np.argmin(d)])

    if method == "idw":
        # Handle the case where the centroid coincides with a gauge
        zero_mask = d == 0.0
        if zero_mask.any():
            return float(v[zero_mask][0])

        weights = 1.0 / (d**idw_power)
        return float(np.sum(weights * v) / np.sum(weights))

    raise ValueError(
        f"Unknown interpolation method '{method}'. Use 'idw' or 'nearest'."
    )


# ---------------------------------------------------------------------------
# Channel width computation
# ---------------------------------------------------------------------------


def compute_channel_width(
    water_mask: Union[np.ndarray, gpd.GeoDataFrame, Polygon, MultiPolygon],
    segment: Union[gpd.GeoSeries, dict[str, Any], LineString],
    num_transects: int = 10,
    transect_length_m: float = 1000.0,
    pixel_size_m: Optional[float] = None,
) -> float:
    """Estimate the average channel width at a river segment.

    Supports two input modes:

    **Vector mode**: *water_mask* is a Shapely Polygon / MultiPolygon or a
    GeoDataFrame of water-body polygons.  Perpendicular transects are cast
    from the centreline and their intersection widths are averaged.

    **Raster mode**: *water_mask* is a 2-D boolean NumPy array where ``True``
    indicates water.  Uses a simple area-based estimate: ``width = area / length``.

    Parameters
    ----------
    water_mask : np.ndarray | GeoDataFrame | Polygon | MultiPolygon
        Water extent — see description above.
    segment : GeoSeries | dict | LineString
        The river segment centreline.
    num_transects : int
        Number of equally spaced transects to cast across the centreline
        (used in vector mode only).
    transect_length_m : float
        Half-length (each side) of each transect in metres.
    pixel_size_m : float | None
        Pixel size in metres — required for raster mode.

    Returns
    -------
    float
        Estimated mean channel width in metres.  Returns ``0.0`` if the
        water mask and segment do not overlap.
    """
    # ---------- Resolve centreline ----------
    if isinstance(segment, LineString):
        centreline = segment
    elif isinstance(segment, dict):
        centreline = shape(segment["geometry"])
    else:
        centreline = segment.geometry if hasattr(segment, "geometry") else segment

    centreline = _ensure_single_linestring(centreline)

    # ---------- Raster mode ----------
    if isinstance(water_mask, np.ndarray):
        if pixel_size_m is None:
            raise ValueError(
                "pixel_size_m must be provided when water_mask is a NumPy array."
            )
        water_pixels = int(np.sum(water_mask))
        water_area_m2 = water_pixels * (pixel_size_m**2)
        seg_utm = _to_utm(centreline)
        length_m = seg_utm.length
        if length_m < 1e-6:
            return 0.0
        return float(water_area_m2 / length_m)

    # ---------- Vector mode ----------
    if isinstance(water_mask, gpd.GeoDataFrame):
        water_poly = unary_union(water_mask.to_crs(epsg=UTM_EPSG).geometry)
    elif isinstance(water_mask, (Polygon, MultiPolygon)):
        water_poly = water_mask
        # Detect CRS: if coordinates look like lon/lat, project to UTM
        if centreline.bounds[0] < 180:
            water_poly = _to_utm(water_poly)
    else:
        raise TypeError(
            f"Unsupported water_mask type: {type(water_mask).__name__}. "
            "Expected np.ndarray, GeoDataFrame, Polygon, or MultiPolygon."
        )

    centreline_utm = _to_utm(centreline)
    total_length = centreline_utm.length

    if total_length < 1e-6 or water_poly.is_empty:
        return 0.0

    widths: list[float] = []
    spacing = total_length / (num_transects + 1)

    for i in range(1, num_transects + 1):
        dist = i * spacing
        pt = centreline_utm.interpolate(dist)
        # Tangent direction at this point
        eps = min(spacing * 0.1, 1.0)
        pt_ahead = centreline_utm.interpolate(min(dist + eps, total_length))
        dx = pt_ahead.x - pt.x
        dy = pt_ahead.y - pt.y
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            continue
        # Perpendicular unit vector
        px, py = -dy / seg_len, dx / seg_len
        # Build transect
        transect = LineString(
            [
                (pt.x - px * transect_length_m, pt.y - py * transect_length_m),
                (pt.x + px * transect_length_m, pt.y + py * transect_length_m),
            ]
        )
        intersection = transect.intersection(water_poly)
        if intersection.is_empty:
            widths.append(0.0)
        else:
            widths.append(intersection.length)

    if not widths:
        return 0.0
    return float(np.mean(widths))


# ---------------------------------------------------------------------------
# Sinuosity
# ---------------------------------------------------------------------------


def compute_sinuosity(segment_geom: Union[LineString, dict[str, Any]]) -> float:
    """Compute the sinuosity of a river segment.

    Sinuosity is defined as the ratio of the *thalweg length* (actual path
    along the channel) to the *straight-line distance* between endpoints.

    A perfectly straight channel has sinuosity = 1.0; meandering reaches
    typically exceed 1.5.

    Parameters
    ----------
    segment_geom : LineString | dict
        Shapely LineString or a GeoJSON geometry dict.

    Returns
    -------
    float
        Sinuosity ratio ≥ 1.0.  Returns ``1.0`` for degenerate segments
        where start and end coincide.

    References
    ----------
    Leopold & Wolman (1957) — *River Channel Patterns: Braided, Meandering
    and Straight*.
    """
    if isinstance(segment_geom, dict):
        segment_geom = shape(segment_geom)

    if not isinstance(segment_geom, LineString):
        segment_geom = _ensure_single_linestring(segment_geom)

    # Project to UTM for accurate length and distance
    seg_utm = _to_utm(segment_geom)
    thalweg_length = seg_utm.length

    coords = list(seg_utm.coords)
    if len(coords) < 2:
        return 1.0

    start = Point(coords[0])
    end = Point(coords[-1])
    straight_distance = start.distance(end)

    if straight_distance < 1e-6:
        logger.debug("Degenerate segment (start == end); sinuosity = 1.0.")
        return 1.0

    sinuosity = thalweg_length / straight_distance
    return round(float(sinuosity), 4)


# ---------------------------------------------------------------------------
# Segment buffering & spatial queries
# ---------------------------------------------------------------------------


def buffer_segment(
    segment_geom: Union[LineString, dict[str, Any]],
    buffer_km: float = 2.0,
) -> Polygon:
    """Return a buffered polygon around a segment centreline.

    Useful for defining the area-of-interest (AOI) passed to GEE when
    extracting Sentinel-2 composites.

    Parameters
    ----------
    segment_geom : LineString | dict
        Segment centreline geometry.
    buffer_km : float
        Buffer radius in kilometres.

    Returns
    -------
    Polygon
        Buffered polygon in WGS-84.
    """
    if isinstance(segment_geom, dict):
        segment_geom = shape(segment_geom)
    seg_utm = _to_utm(segment_geom)
    buffered_utm = seg_utm.buffer(buffer_km * 1000.0)
    return _to_wgs84(buffered_utm)


def segment_bounding_box(
    segment_geom: Union[LineString, dict[str, Any]],
    padding_km: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return the bounding box of a segment with optional padding.

    Parameters
    ----------
    segment_geom : LineString | dict
        Segment centreline geometry.
    padding_km : float
        Padding to add on all sides, in kilometres.

    Returns
    -------
    tuple[float, float, float, float]
        ``(minLon, minLat, maxLon, maxLat)`` in WGS-84 degrees.
    """
    if isinstance(segment_geom, dict):
        segment_geom = shape(segment_geom)

    seg_utm = _to_utm(segment_geom)
    pad = padding_km * 1000.0
    minx, miny, maxx, maxy = seg_utm.bounds
    padded = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    # Convert corners back to WGS-84
    min_lon, min_lat = _utm_to_wgs84.transform(padded[0], padded[1])
    max_lon, max_lat = _utm_to_wgs84.transform(padded[2], padded[3])

    return (
        round(min_lon, 6),
        round(min_lat, 6),
        round(max_lon, 6),
        round(max_lat, 6),
    )


# ---------------------------------------------------------------------------
# GeoJSON conversion helpers
# ---------------------------------------------------------------------------


def geojson_to_response(
    gdf: gpd.GeoDataFrame,
    properties_cols: Optional[list[str]] = None,
    crs_out: int = WGS84_EPSG,
) -> dict[str, Any]:
    """Serialise a GeoDataFrame to a GeoJSON-compatible response dictionary.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to serialise.
    properties_cols : list[str] | None
        Subset of columns to include in feature properties.  Defaults to
        all non-geometry columns.
    crs_out : int
        EPSG code for output coordinates (default: 4326).

    Returns
    -------
    dict[str, Any]
        A ``FeatureCollection`` dictionary compatible with ``geojson.dumps``.
    """
    if gdf.crs and gdf.crs.to_epsg() != crs_out:
        gdf = gdf.to_crs(epsg=crs_out)

    if properties_cols is None:
        properties_cols = [c for c in gdf.columns if c != gdf.geometry.name]

    features: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        geom = row[gdf.geometry.name]
        props: dict[str, Any] = {}
        for col in properties_cols:
            val = row.get(col)
            # Convert numpy scalars to native Python types for JSON serialisation
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, (np.bool_,)):
                val = bool(val)
            elif isinstance(val, (np.ndarray,)):
                val = val.tolist()
            elif (
                pd.isna(val)
                if not isinstance(val, (list, dict, BaseGeometry))
                else False
            ):
                val = None
            props[col] = val

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom) if geom is not None else None,
                "properties": props,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {"name": f"EPSG:{crs_out}"},
        },
    }


def geojson_to_geodataframe(
    geojson: Union[dict[str, Any], str, Path],
    crs: int = WGS84_EPSG,
) -> gpd.GeoDataFrame:
    """Parse a GeoJSON dict, string or file path into a GeoDataFrame.

    Parameters
    ----------
    geojson : dict | str | Path
        GeoJSON content.  Can be:
        - A parsed dictionary (``FeatureCollection`` or ``Feature``).
        - A raw JSON string.
        - A file path to a ``.geojson`` / ``.json`` file.
    crs : int
        EPSG code to assign to the resulting GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        Parsed GeoDataFrame.
    """
    if isinstance(geojson, Path) or (
        isinstance(geojson, str) and Path(geojson).exists()
    ):
        return gpd.read_file(str(geojson))

    if isinstance(geojson, str):
        geojson = json.loads(geojson)

    if geojson.get("type") == "Feature":
        geojson = {"type": "FeatureCollection", "features": [geojson]}

    return gpd.GeoDataFrame.from_features(geojson["features"], crs=f"EPSG:{crs}")


# ---------------------------------------------------------------------------
# Nearest-gauge assignment
# ---------------------------------------------------------------------------


def assign_nearest_gauge(
    segments: gpd.GeoDataFrame,
    gauges: gpd.GeoDataFrame,
    gauge_id_col: str = "gauge_id",
    max_distance_km: float = 100.0,
) -> gpd.GeoDataFrame:
    """Spatially join each segment centroid to the nearest gauge station.

    Parameters
    ----------
    segments : gpd.GeoDataFrame
        River segments with centreline geometries.
    gauges : gpd.GeoDataFrame
        Point GeoDataFrame of gauge stations.
    gauge_id_col : str
        Column in *gauges* holding the station identifier.
    max_distance_km : float
        Segments further than this distance from all gauges receive
        ``NaN`` for ``nearest_gauge_id`` and ``nearest_gauge_dist_km``.

    Returns
    -------
    gpd.GeoDataFrame
        Input *segments* with two new columns:
        ``nearest_gauge_id`` and ``nearest_gauge_dist_km``.
    """
    # Project both to UTM
    segs_utm = segments.copy().to_crs(epsg=UTM_EPSG)
    segs_utm["_centroid_utm"] = segs_utm.geometry.centroid

    gauges_utm = gauges.to_crs(epsg=UTM_EPSG)

    nearest_ids: list[Optional[str]] = []
    nearest_dists: list[Optional[float]] = []

    for _, seg_row in segs_utm.iterrows():
        centroid = seg_row["_centroid_utm"]
        dists = gauges_utm.geometry.apply(lambda g: centroid.distance(g)).values
        min_idx = int(np.argmin(dists))
        min_dist_km = float(dists[min_idx]) / 1000.0

        if min_dist_km <= max_distance_km:
            nearest_ids.append(str(gauges.iloc[min_idx][gauge_id_col]))
            nearest_dists.append(round(min_dist_km, 3))
        else:
            nearest_ids.append(None)
            nearest_dists.append(None)

    result = segments.copy()
    result["nearest_gauge_id"] = nearest_ids
    result["nearest_gauge_dist_km"] = nearest_dists
    return result


# ---------------------------------------------------------------------------
# Hydraulic geometry helpers
# ---------------------------------------------------------------------------


def estimate_cross_sectional_area(
    width_m: float,
    depth_m: float,
    channel_shape: str = "trapezoidal",
    bank_slope: float = 1.5,
) -> float:
    """Estimate cross-sectional area from width and depth.

    Parameters
    ----------
    width_m : float
        Channel top-width in metres.
    depth_m : float
        Mean water depth in metres.
    channel_shape : str
        ``"rectangular"`` or ``"trapezoidal"`` (default).
    bank_slope : float
        Side slope (H:V) for trapezoidal channels.

    Returns
    -------
    float
        Cross-sectional area in m².
    """
    if channel_shape == "rectangular":
        return width_m * depth_m
    if channel_shape == "trapezoidal":
        bottom_width = max(0.0, width_m - 2 * bank_slope * depth_m)
        return (bottom_width + bank_slope * depth_m) * depth_m
    raise ValueError(
        f"Unknown channel_shape '{channel_shape}'. Use 'rectangular' or 'trapezoidal'."
    )


def estimate_hydraulic_radius(
    width_m: float,
    depth_m: float,
    channel_shape: str = "trapezoidal",
    bank_slope: float = 1.5,
) -> float:
    """Estimate the hydraulic radius (A / P) for Manning's equation.

    Parameters
    ----------
    width_m, depth_m, channel_shape, bank_slope
        See :func:`estimate_cross_sectional_area`.

    Returns
    -------
    float
        Hydraulic radius in metres.
    """
    area = estimate_cross_sectional_area(width_m, depth_m, channel_shape, bank_slope)
    if channel_shape == "rectangular":
        wetted_perimeter = width_m + 2 * depth_m
    elif channel_shape == "trapezoidal":
        bottom_width = max(0.0, width_m - 2 * bank_slope * depth_m)
        wetted_perimeter = bottom_width + 2 * depth_m * math.sqrt(1 + bank_slope**2)
    else:
        raise ValueError(f"Unknown channel_shape '{channel_shape}'.")

    return area / wetted_perimeter if wetted_perimeter > 0 else 0.0


def manning_velocity(
    hydraulic_radius_m: float,
    slope: float,
    mannings_n: float = 0.030,
) -> float:
    """Compute mean flow velocity using Manning's equation.

    Parameters
    ----------
    hydraulic_radius_m : float
        Hydraulic radius in metres.
    slope : float
        Channel bed slope (dimensionless, e.g. 0.0001).
    mannings_n : float
        Manning's roughness coefficient.  Typical value for a natural
        river is 0.025–0.040.

    Returns
    -------
    float
        Mean flow velocity in m/s.
    """
    if mannings_n <= 0 or hydraulic_radius_m <= 0 or slope <= 0:
        return 0.0
    return (1.0 / mannings_n) * (hydraulic_radius_m ** (2 / 3)) * math.sqrt(slope)


# ---------------------------------------------------------------------------
# Geometry hashing
# ---------------------------------------------------------------------------


def geometry_hash(geom: BaseGeometry, precision: int = 6) -> str:
    """Compute a stable MD5 hash of a geometry for cache keying.

    Parameters
    ----------
    geom : BaseGeometry
        Any Shapely geometry.
    precision : int
        Number of decimal places to round coordinates before hashing.

    Returns
    -------
    str
        8-character hex digest string.
    """
    # Round and serialise coordinates to ensure float-equality stability
    rounded = json.dumps(
        mapping(geom), sort_keys=True, default=lambda x: round(x, precision)
    )
    return hashlib.md5(rounded.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Terrain / elevation helpers
# ---------------------------------------------------------------------------


def compute_along_channel_slope(
    elevations_m: np.ndarray,
    distances_m: np.ndarray,
) -> float:
    """Estimate the average bed slope from along-channel elevation samples.

    Uses a least-squares linear fit rather than a simple rise/run ratio to
    be robust against noisy DEM data.

    Parameters
    ----------
    elevations_m : np.ndarray
        Elevation values (metres) sampled along the thalweg.
    distances_m : np.ndarray
        Cumulative chainage (metres) corresponding to each elevation sample.

    Returns
    -------
    float
        Estimated bed slope (dimensionless, always non-negative).
    """
    if len(elevations_m) < 2 or len(elevations_m) != len(distances_m):
        raise ValueError("elevations_m and distances_m must have the same length ≥ 2.")
    # Linear regression: elevation ~ a + b * distance  →  slope = |b|
    coeffs = np.polyfit(distances_m, elevations_m, 1)
    return abs(float(coeffs[0]))


# ---------------------------------------------------------------------------
# Clip utilities
# ---------------------------------------------------------------------------


def clip_to_waterway_bbox(
    gdf: gpd.GeoDataFrame,
    waterway_id: str,
    padding_deg: float = 0.1,
) -> gpd.GeoDataFrame:
    """Clip a GeoDataFrame to the bounding box of a waterway with padding.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Features to clip (any geometry type).
    waterway_id : str
        ``"NW-1"`` or ``"NW-2"``.
    padding_deg : float
        Extra padding in decimal degrees added on every side.

    Returns
    -------
    gpd.GeoDataFrame
        Subset of *gdf* whose geometries intersect the waterway bounding box.
    """
    meta = get_waterway_meta(waterway_id)
    minx, miny, maxx, maxy = meta["bbox"]
    minx -= padding_deg
    miny -= padding_deg
    maxx += padding_deg
    maxy += padding_deg

    bbox_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

    if gdf.crs and gdf.crs.to_epsg() != WGS84_EPSG:
        gdf = gdf.to_crs(epsg=WGS84_EPSG)

    return gdf[gdf.geometry.intersects(bbox_poly)].copy()
