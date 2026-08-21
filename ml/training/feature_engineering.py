"""
feature_engineering.py
======================
Full feature engineering pipeline for the AIDSTL project.

Covers:
  - SpectralFeatureExtractor  : Sentinel-2 / SAR / ancillary indices
  - TemporalSequenceBuilder   : builds (T=12, F) tensors per river segment
  - RiverSegmentDataset       : torch Dataset for model training / evaluation

Study areas
-----------
  NW-1  Ganga        Varanasi – Haldia        ~1 390 km
  NW-2  Brahmaputra  Dhubri   – Sadiya        ~  891 km

River segmented into 5 km analysis units.
Sentinel-2 bands (10 m, 12 bands):  B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 (+ derived)
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTINEL2_BANDS: List[str] = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]
"""All 10 Sentinel-2 MultiSpectral Instrument bands used in the project."""

SENTINEL1_BANDS: List[str] = ["VV", "VH"]
"""Sentinel-1 SAR bands."""

SPECTRAL_INDICES: List[str] = [
    "MNDWI",
    "NDWI",
    "AWEI",
    "STUMPF",
    "TURBIDITY",
    "NDTI",
]

ANCILLARY_FEATURES: List[str] = [
    "water_width_m",
    "sinuosity",
    "mndwi_std_12m",
    "gauge_water_level_m",
    "gauge_discharge_m3s",
    "era5_cumulative_rainfall_mm",
    "era5_mean_temperature_c",
    "sar_vv",
    "sar_vh",
    "sar_vv_vh_ratio",
]

ALL_FEATURES: List[str] = SENTINEL2_BANDS + SPECTRAL_INDICES + ANCILLARY_FEATURES

# Default navigability thresholds (metres)
DEPTH_NAVIGABLE: float = 3.0
DEPTH_CONDITIONAL: float = 2.0

# Navigability class labels
NAV_CLASSES: Dict[int, str] = {
    0: "Non-Navigable",
    1: "Conditional",
    2: "Navigable",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Element-wise division, replacing NaN / Inf with *fill_value*."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.abs(denominator) < 1e-9,
            fill_value,
            numerator / denominator,
        )
    return result.astype(np.float32)


def _safe_log(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Natural log with a small epsilon floor to avoid log(0)."""
    return np.log(np.maximum(arr, eps)).astype(np.float32)


def _clip_reflectance(arr: np.ndarray) -> np.ndarray:
    """Clip surface reflectance to [0, 1]."""
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# SpectralFeatureExtractor
# ---------------------------------------------------------------------------


class SpectralFeatureExtractor:
    """Compute spectral, SAR, and ancillary features for a river segment.

    All Sentinel-2 band values are expected to be surface reflectance
    (0 – 1 float32).  Band naming follows the ESA convention:
    B2 (blue), B3 (green), B4 (red), B5–B7 (red-edge),
    B8 (NIR broad), B8A (NIR narrow), B11 / B12 (SWIR).

    Parameters
    ----------
    fill_value:
        Value used to replace NaN / Inf after index computation.
    scale_factor:
        If Sentinel-2 digital numbers are in [0, 10 000] range, pass
        ``scale_factor=1e-4`` to normalise to [0, 1].  Default 1.0 means
        inputs are already reflectance.
    """

    def __init__(
        self,
        fill_value: float = 0.0,
        scale_factor: float = 1.0,
    ) -> None:
        self.fill_value = fill_value
        self.scale_factor = scale_factor

    # ------------------------------------------------------------------
    # Individual index methods
    # ------------------------------------------------------------------

    def compute_mndwi(self, B3: np.ndarray, B11: np.ndarray) -> np.ndarray:
        """Modified Normalised Difference Water Index.

        MNDWI = (B3 − B11) / (B3 + B11)

        Positive values indicate open water.
        """
        return _safe_divide(B3 - B11, B3 + B11, self.fill_value)

    def compute_ndwi(self, B3: np.ndarray, B8: np.ndarray) -> np.ndarray:
        """Normalised Difference Water Index (McFeeters 1996).

        NDWI = (B3 − B8) / (B3 + B8)
        """
        return _safe_divide(B3 - B8, B3 + B8, self.fill_value)

    def compute_awei(
        self,
        B3: np.ndarray,
        B8: np.ndarray,
        B11: np.ndarray,
        B12: np.ndarray,
    ) -> np.ndarray:
        """Automated Water Extraction Index (Feyisa et al. 2014).

        AWEI_sh = 4*(B3 − B11) − 0.25*B8 + 2.75*B12

        Designed for shadow removal in complex terrain.
        """
        return (4.0 * (B3 - B11) - 0.25 * B8 + 2.75 * B12).astype(np.float32)

    def compute_stumpf_ratio(self, B3: np.ndarray, B2: np.ndarray) -> np.ndarray:
        """Stumpf log-ratio bathymetric depth proxy.

        ratio = log(B3) / log(B2)

        Originally applied to multispectral satellite imagery for
        shallow-water depth estimation (Stumpf et al. 2003).
        """
        return _safe_divide(_safe_log(B3), _safe_log(B2), self.fill_value)

    def compute_turbidity(self, B4: np.ndarray, B3: np.ndarray) -> np.ndarray:
        """Simple turbidity index using red vs green reflectance.

        Turbidity = (B4 − B3) / (B4 + B3)

        Positive values correspond to higher turbidity / sediment load.
        """
        return _safe_divide(B4 - B3, B4 + B3, self.fill_value)

    def compute_ndti(self, B4: np.ndarray, B3: np.ndarray) -> np.ndarray:
        """Normalised Difference Turbidity Index.

        NDTI = (B4 − B3) / (B4 + B3)  — alias kept for explicit naming.

        Note: same formula as compute_turbidity; kept separate for
        downstream pipeline clarity.
        """
        return self.compute_turbidity(B4, B3)

    # ------------------------------------------------------------------
    # Geometric / ancillary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_water_width(
        water_mask: np.ndarray,
        pixel_size_m: float = 10.0,
        n_transects: int = 10,
    ) -> float:
        """Estimate water-surface width (m) from a binary water mask.

        Samples *n_transects* horizontal scan-lines uniformly spaced
        across the mask rows and returns the **median** width.

        Parameters
        ----------
        water_mask:
            2-D boolean array (H × W), True where water is present.
        pixel_size_m:
            Ground sampling distance in metres.
        n_transects:
            Number of horizontal transects to sample.

        Returns
        -------
        float
            Median water width in metres.  Returns 0.0 for empty mask.
        """
        if water_mask.ndim != 2:
            raise ValueError("water_mask must be 2-D (H × W).")
        H, W = water_mask.shape
        if H == 0 or W == 0:
            return 0.0

        rows = np.linspace(0, H - 1, min(n_transects, H), dtype=int)
        widths: List[float] = []
        for r in rows:
            row_pixels = int(np.sum(water_mask[r, :]))
            widths.append(row_pixels * pixel_size_m)

        return float(np.median(widths)) if widths else 0.0

    @staticmethod
    def compute_sinuosity(
        centreline_coords: np.ndarray,
    ) -> float:
        """Compute channel sinuosity from centreline coordinates.

        sinuosity = channel_length / straight_line_distance

        Parameters
        ----------
        centreline_coords:
            Array of shape (N, 2) with (lon, lat) or (x, y) pairs.

        Returns
        -------
        float
            Sinuosity ratio (≥ 1.0).  Returns 1.0 for degenerate input.
        """
        if centreline_coords.ndim != 2 or centreline_coords.shape[1] != 2:
            raise ValueError("centreline_coords must be shape (N, 2).")
        if len(centreline_coords) < 2:
            return 1.0

        # Channel length: sum of Euclidean segment lengths
        diffs = np.diff(centreline_coords, axis=0)
        channel_len = float(np.sum(np.linalg.norm(diffs, axis=1)))

        # Straight-line (end-to-end) distance
        straight = float(np.linalg.norm(centreline_coords[-1] - centreline_coords[0]))
        if straight < 1e-9:
            return 1.0

        return max(1.0, channel_len / straight)

    @staticmethod
    def compute_temporal_variability(
        mndwi_series: np.ndarray,
    ) -> float:
        """Return standard deviation of MNDWI values over T time steps.

        Parameters
        ----------
        mndwi_series:
            1-D array of MNDWI values (length = number of months).

        Returns
        -------
        float
            Temporal standard deviation (0 if only one observation).
        """
        if len(mndwi_series) < 2:
            return 0.0
        return float(np.nanstd(mndwi_series))

    # ------------------------------------------------------------------
    # Master extraction method
    # ------------------------------------------------------------------

    def extract(
        self,
        bands: Dict[str, np.ndarray],
        ancillary: Optional[Dict[str, float]] = None,
        water_mask: Optional[np.ndarray] = None,
        centreline_coords: Optional[np.ndarray] = None,
        mndwi_series: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Extract all features for a single image observation.

        Parameters
        ----------
        bands:
            Dict mapping band name → reflectance array (scalar or 2-D).
            Keys should match SENTINEL2_BANDS and SENTINEL1_BANDS.
        ancillary:
            Dict of scalar ancillary values (gauge, ERA5, etc.).
        water_mask:
            Optional 2-D water mask for width estimation.
        centreline_coords:
            Optional (N, 2) array for sinuosity estimation.
        mndwi_series:
            Optional 1-D array of historical MNDWI for temporal std.

        Returns
        -------
        Dict[str, float]
            Feature dictionary with one value per feature.
        """

        # ── Resolve band scalars (mean over pixels if array) ───────────
        def _to_scalar(arr: Union[np.ndarray, float]) -> np.ndarray:
            a = np.asarray(arr, dtype=np.float32) * self.scale_factor
            return a.mean() if a.ndim > 0 else a

        B2 = _to_scalar(bands.get("B2", 0.0))
        B3 = _to_scalar(bands.get("B3", 0.0))
        B4 = _to_scalar(bands.get("B4", 0.0))
        B5 = _to_scalar(bands.get("B5", 0.0))
        B6 = _to_scalar(bands.get("B6", 0.0))
        B7 = _to_scalar(bands.get("B7", 0.0))
        B8 = _to_scalar(bands.get("B8", 0.0))
        B8A = _to_scalar(bands.get("B8A", 0.0))
        B11 = _to_scalar(bands.get("B11", 0.0))
        B12 = _to_scalar(bands.get("B12", 0.0))
        VV = _to_scalar(bands.get("VV", 0.0))
        VH = _to_scalar(bands.get("VH", 0.0))

        features: Dict[str, float] = {}

        # ── Sentinel-2 band medians ────────────────────────────────────
        for name, val in [
            ("B2", B2),
            ("B3", B3),
            ("B4", B4),
            ("B5", B5),
            ("B6", B6),
            ("B7", B7),
            ("B8", B8),
            ("B8A", B8A),
            ("B11", B11),
            ("B12", B12),
        ]:
            features[name] = float(val)

        # ── Spectral indices ──────────────────────────────────────────
        features["MNDWI"] = float(self.compute_mndwi(B3, B11))
        features["NDWI"] = float(self.compute_ndwi(B3, B8))
        features["AWEI"] = float(self.compute_awei(B3, B8, B11, B12))
        features["STUMPF"] = float(self.compute_stumpf_ratio(B3, B2))
        features["TURBIDITY"] = float(self.compute_turbidity(B4, B3))
        features["NDTI"] = float(self.compute_ndti(B4, B3))

        # ── Geometric features ────────────────────────────────────────
        if water_mask is not None:
            features["water_width_m"] = self.compute_water_width(water_mask)
        else:
            features["water_width_m"] = float(
                ancillary.get("water_width_m", 0.0) if ancillary else 0.0
            )

        if centreline_coords is not None:
            features["sinuosity"] = self.compute_sinuosity(centreline_coords)
        else:
            features["sinuosity"] = float(
                ancillary.get("sinuosity", 1.0) if ancillary else 1.0
            )

        # ── Temporal variability ──────────────────────────────────────
        if mndwi_series is not None:
            features["mndwi_std_12m"] = self.compute_temporal_variability(mndwi_series)
        else:
            features["mndwi_std_12m"] = float(
                ancillary.get("mndwi_std_12m", 0.0) if ancillary else 0.0
            )

        # ── CWC gauge data ────────────────────────────────────────────
        features["gauge_water_level_m"] = float(
            ancillary.get("gauge_water_level_m", 0.0) if ancillary else 0.0
        )
        features["gauge_discharge_m3s"] = float(
            ancillary.get("gauge_discharge_m3s", 0.0) if ancillary else 0.0
        )

        # ── ERA5 climate ──────────────────────────────────────────────
        features["era5_cumulative_rainfall_mm"] = float(
            ancillary.get("era5_cumulative_rainfall_mm", 0.0) if ancillary else 0.0
        )
        features["era5_mean_temperature_c"] = float(
            ancillary.get("era5_mean_temperature_c", 25.0) if ancillary else 25.0
        )

        # ── SAR backscatter ───────────────────────────────────────────
        features["sar_vv"] = float(VV)
        features["sar_vh"] = float(VH)
        features["sar_vv_vh_ratio"] = float(
            _safe_divide(np.array([VV]), np.array([VH]), self.fill_value)[0]
        )

        return features

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        ancillary_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Batch-extract features from a DataFrame of band values.

        Parameters
        ----------
        df:
            DataFrame where each row is one observation and columns include
            Sentinel-2 band names and optional ancillary columns.
        ancillary_cols:
            Columns in *df* that should be forwarded as ancillary values.

        Returns
        -------
        pd.DataFrame
            New DataFrame with all extracted feature columns appended.
        """
        records: List[Dict[str, float]] = []
        ancillary_cols = ancillary_cols or []

        for _, row in df.iterrows():
            band_dict = {
                b: row[b] for b in SENTINEL2_BANDS + SENTINEL1_BANDS if b in row
            }
            anc_dict = {c: row[c] for c in ancillary_cols if c in row}
            record = self.extract(bands=band_dict, ancillary=anc_dict)
            records.append(record)

        feat_df = pd.DataFrame(records, index=df.index)
        return pd.concat([df, feat_df], axis=1)


# ---------------------------------------------------------------------------
# TemporalSequenceBuilder
# ---------------------------------------------------------------------------


@dataclass
class SequenceConfig:
    """Configuration for temporal sequence construction.

    Attributes
    ----------
    sequence_length:
        Number of time steps T (default = 12 months).
    feature_columns:
        Ordered list of feature names to include (determines F).
    target_column:
        Name of the depth column (regression target).
    segment_id_column:
        Column that identifies river segment.
    date_column:
        Column with datetime / period information.
    pad_value:
        Value used for missing time steps.
    normalise:
        Whether to z-score normalise features before returning.
    """

    sequence_length: int = 12
    feature_columns: List[str] = field(default_factory=lambda: list(ALL_FEATURES))
    target_column: str = "depth_m"
    segment_id_column: str = "segment_id"
    date_column: str = "date"
    pad_value: float = 0.0
    normalise: bool = True


class TemporalSequenceBuilder:
    """Build fixed-length temporal sequences from a flat feature DataFrame.

    Each river segment gets one sequence tensor of shape **(T, F)** where:
      T = ``config.sequence_length``  (default 12 months)
      F = ``len(config.feature_columns)``

    Missing time steps are padded with ``config.pad_value``.

    Parameters
    ----------
    config:
        A :class:`SequenceConfig` instance.
    """

    def __init__(self, config: Optional[SequenceConfig] = None) -> None:
        self.config = config or SequenceConfig()
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def fit_normalisation(self, df: pd.DataFrame) -> "TemporalSequenceBuilder":
        """Fit z-score statistics on the full feature DataFrame.

        Parameters
        ----------
        df:
            DataFrame containing all feature_columns.

        Returns
        -------
        self
        """
        cols = [c for c in self.config.feature_columns if c in df.columns]
        arr = df[cols].values.astype(np.float32)
        self._mean = np.nanmean(arr, axis=0)
        self._mean = np.nan_to_num(self._mean, nan=0.0)
        self._std = np.nanstd(arr, axis=0)
        self._std = np.nan_to_num(self._std, nan=1.0)
        self._std[self._std < 1e-8] = 1.0  # avoid divide-by-zero
        logger.info("Fitted normalisation on %d samples × %d features.", *arr.shape)
        return self

    def normalise_array(self, arr: np.ndarray) -> np.ndarray:
        """Apply fitted z-score normalisation to a (T, F) array."""
        if self._mean is None or self._std is None:
            raise RuntimeError("Call fit_normalisation() before normalise_array().")
        return (arr - self._mean) / self._std

    def save_stats(self, path: Union[str, Path]) -> None:
        """Persist normalisation statistics to a NumPy .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self._mean, std=self._std)
        logger.info("Saved normalisation stats to %s", path)

    def load_stats(self, path: Union[str, Path]) -> "TemporalSequenceBuilder":
        """Load normalisation statistics from a .npz file."""
        data = np.load(path)
        self._mean = data["mean"]
        self._std = data["std"]
        logger.info("Loaded normalisation stats from %s", path)
        return self

    # ------------------------------------------------------------------
    # Sequence construction
    # ------------------------------------------------------------------

    def _build_single_sequence(
        self,
        segment_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Build a (T, F) feature array and scalar target for one segment.

        Parameters
        ----------
        segment_df:
            Sub-DataFrame for a single segment, sorted by date,
            length ≤ T.

        Returns
        -------
        seq : np.ndarray  shape (T, F)
        target : float or None
        """
        cfg = self.config
        T = cfg.sequence_length
        cols = [c for c in cfg.feature_columns if c in segment_df.columns]
        F = len(cols)

        seq = np.full((T, F), cfg.pad_value, dtype=np.float32)

        # Sort by date
        tmp = segment_df.sort_values(cfg.date_column)
        n = min(len(tmp), T)
        seq[:n] = tmp[cols].values[:n].astype(np.float32)

        # Replace any remaining NaN with pad value
        np.nan_to_num(
            seq,
            nan=cfg.pad_value,
            posinf=cfg.pad_value,
            neginf=cfg.pad_value,
            copy=False,
        )

        # Target: mean depth over available time steps
        target: Optional[float] = None
        if cfg.target_column in tmp.columns:
            target = float(tmp[cfg.target_column].mean())

        return seq, target

    def build(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build sequence tensors for all segments in *df*.

        Parameters
        ----------
        df:
            DataFrame with columns for ``segment_id``, ``date``,
            all ``feature_columns``, and ``target_column``.

        Returns
        -------
        X : np.ndarray  shape (N, T, F)
        y : np.ndarray  shape (N,)      — depth targets (NaN if missing)
        segment_ids : np.ndarray shape (N,)
        """
        cfg = self.config
        group = df.groupby(cfg.segment_id_column)
        seqs, targets, ids = [], [], []

        for seg_id, grp in group:
            seq, tgt = self._build_single_sequence(grp)
            seqs.append(seq)
            targets.append(tgt if tgt is not None else float("nan"))
            ids.append(seg_id)

        X = np.stack(seqs, axis=0)  # (N, T, F)
        y = np.array(targets, dtype=np.float32)
        segment_ids = np.array(ids)

        if cfg.normalise:
            if self._mean is None:
                logger.warning(
                    "normalise=True but stats not fitted — fitting now on this data."
                )
                cols = [c for c in cfg.feature_columns if c in df.columns]
                arr = df[cols].values.astype(np.float32)
                self._mean = np.nanmean(arr, axis=0)
                self._std = np.nanstd(arr, axis=0)
                self._std[self._std < 1e-8] = 1.0
            # Broadcast normalisation over T dimension
            X = (X - self._mean[np.newaxis, np.newaxis, :]) / self._std[
                np.newaxis, np.newaxis, :
            ]
            X = np.nan_to_num(
                X, nan=cfg.pad_value, posinf=cfg.pad_value, neginf=cfg.pad_value
            )

        logger.info(
            "Built sequences: X=%s, y=%s, segments=%d",
            X.shape,
            y.shape,
            len(ids),
        )
        return X, y, segment_ids

    def build_from_parquet(
        self,
        parquet_path: Union[str, Path],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience wrapper: load a Parquet file then call build()."""
        df = pd.read_parquet(parquet_path)
        return self.build(df)


# ---------------------------------------------------------------------------
# Static feature helpers
# ---------------------------------------------------------------------------


def compute_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time-invariant static features per segment.

    These are appended as extra columns to the segment-level summary.
    Static features are used as the *x_static* input to the TFT model.

    Features derived:
      - mean / std of all spectral indices across all time steps
      - elevation_m (from SRTM DEM; expected as a column)
      - slope_deg   (from SRTM DEM; expected as a column)
      - distance_from_source_km

    Parameters
    ----------
    df:
        DataFrame with one row per (segment_id, date).

    Returns
    -------
    pd.DataFrame
        One row per segment_id with static feature columns.
    """
    index_cols = SPECTRAL_INDICES + ["water_width_m", "gauge_discharge_m3s"]
    agg_dict: Dict[str, list] = {}
    for c in index_cols:
        if c in df.columns:
            agg_dict[c] = ["mean", "std", "min", "max"]

    static_passthrough = ["elevation_m", "slope_deg", "distance_from_source_km"]
    for c in static_passthrough:
        if c in df.columns:
            agg_dict[c] = ["first"]

    if not agg_dict:
        raise ValueError(
            "DataFrame is missing expected feature columns.  "
            f"Available: {list(df.columns)}"
        )

    grouped = df.groupby("segment_id").agg(agg_dict)
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
    grouped.reset_index(inplace=True)
    return grouped


# ---------------------------------------------------------------------------
# RiverSegmentDataset
# ---------------------------------------------------------------------------


class RiverSegmentDataset(Dataset):
    """PyTorch Dataset for river-segment depth prediction.

    Each item yields:
      - ``x_temporal``  : FloatTensor (T, F_temporal)
      - ``x_static``    : FloatTensor (F_static,)
      - ``x_patch``     : FloatTensor (C, H, W) or zeros if not available
      - ``y``           : FloatTensor scalar — depth in metres
      - ``nav_label``   : LongTensor scalar — 0/1/2 navigability class
      - ``segment_id``  : str

    Parameters
    ----------
    sequences : np.ndarray  shape (N, T, F_temporal)
        Temporal feature sequences.
    targets : np.ndarray  shape (N,)
        Depth targets in metres.
    segment_ids : np.ndarray  shape (N,)
        Segment identifier strings / integers.
    static_features : np.ndarray or None  shape (N, F_static)
        Time-invariant static features.  If None, zeros are used.
    patches : np.ndarray or None  shape (N, C, H, W)
        Multi-band satellite image patches.  If None, zeros are used.
    patch_size : int
        Spatial size of patches (H = W = patch_size).  Default 64.
    n_patch_bands : int
        Number of image bands in patches.  Default 12.
    augment : bool
        Whether to apply random data augmentation to patches.
    depth_navigable : float
        Depth threshold for "Navigable" class (default 3.0 m).
    depth_conditional : float
        Lower depth threshold for "Conditional" class (default 2.0 m).
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        segment_ids: np.ndarray,
        static_features: Optional[np.ndarray] = None,
        patches: Optional[np.ndarray] = None,
        patch_size: int = 64,
        n_patch_bands: int = 12,
        augment: bool = False,
        depth_navigable: float = DEPTH_NAVIGABLE,
        depth_conditional: float = DEPTH_CONDITIONAL,
    ) -> None:
        if len(sequences) != len(targets):
            raise ValueError(
                f"sequences ({len(sequences)}) and targets ({len(targets)}) "
                "must have the same length."
            )
        self.sequences = torch.from_numpy(sequences.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.segment_ids = segment_ids
        self.augment = augment
        self.patch_size = patch_size
        self.n_patch_bands = n_patch_bands
        self.depth_navigable = depth_navigable
        self.depth_conditional = depth_conditional

        N = len(sequences)
        T, F = sequences.shape[1], sequences.shape[2]

        # Static features
        if static_features is not None:
            if static_features.shape[0] != N:
                raise ValueError("static_features first dimension must equal N.")
            self.static_features = torch.from_numpy(static_features.astype(np.float32))
        else:
            self.static_features = torch.zeros(N, 16, dtype=torch.float32)

        # Image patches
        if patches is not None:
            if patches.shape[0] != N:
                raise ValueError("patches first dimension must equal N.")
            self.patches = torch.from_numpy(patches.astype(np.float32))
        else:
            self.patches = torch.zeros(
                N, n_patch_bands, patch_size, patch_size, dtype=torch.float32
            )

        # Pre-compute navigability labels from depth targets
        self.nav_labels = self._compute_nav_labels(targets)

        logger.info(
            "RiverSegmentDataset: N=%d, T=%d, F_temporal=%d, "
            "F_static=%d, patch_shape=(%d,%d,%d)",
            N,
            T,
            F,
            self.static_features.shape[-1],
            n_patch_bands,
            patch_size,
            patch_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_nav_labels(self, depths: np.ndarray) -> torch.Tensor:
        """Map continuous depth values to navigability class indices.

        0 → Non-Navigable  (depth < depth_conditional)
        1 → Conditional    (depth_conditional ≤ depth < depth_navigable)
        2 → Navigable      (depth ≥ depth_navigable)
        """
        labels = np.zeros(len(depths), dtype=np.int64)
        labels[depths >= self.depth_conditional] = 1
        labels[depths >= self.depth_navigable] = 2
        # NaN depths → Non-Navigable
        labels[np.isnan(depths)] = 0
        return torch.from_numpy(labels)

    def _augment_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """Apply simple random augmentations to a satellite patch.

        Augmentations:
          - Random horizontal flip
          - Random vertical flip
          - Random 90° rotation
          - Gaussian noise (σ = 0.01)
        """
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[2])  # horizontal
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[1])  # vertical
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            patch = torch.rot90(patch, k=int(k), dims=[1, 2])
        noise = torch.randn_like(patch) * 0.01
        patch = patch + noise
        return patch

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        x_temporal = self.sequences[idx]  # (T, F)
        x_static = self.static_features[idx]  # (F_static,)
        x_patch = self.patches[idx]  # (C, H, W)
        y = self.targets[idx]  # scalar
        nav_label = self.nav_labels[idx]  # scalar int
        seg_id = str(self.segment_ids[idx])

        if self.augment:
            x_patch = self._augment_patch(x_patch)

        return {
            "x_temporal": x_temporal,
            "x_static": x_static,
            "x_patch": x_patch,
            "y": y,
            "nav_label": nav_label,
            "segment_id": seg_id,
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(
        cls,
        parquet_path: Union[str, Path],
        patch_dir: Optional[Union[str, Path]] = None,
        seq_config: Optional[SequenceConfig] = None,
        norm_stats_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> "RiverSegmentDataset":
        """Build a dataset directly from a Parquet feature file.

        Parameters
        ----------
        parquet_path:
            Path to the feature matrix Parquet file.
        patch_dir:
            Directory containing per-segment .npy patch arrays named
            ``{segment_id}.npy``.  If None, patches will be zeros.
        seq_config:
            SequenceConfig; defaults if None.
        norm_stats_path:
            Path to a .npz file with pre-fitted normalisation stats.
            If None, stats are fitted on-the-fly.
        **kwargs:
            Forwarded to ``RiverSegmentDataset.__init__``.

        Returns
        -------
        RiverSegmentDataset
        """
        df = pd.read_parquet(parquet_path)
        seq_cfg = seq_config or SequenceConfig()
        builder = TemporalSequenceBuilder(seq_cfg)

        if norm_stats_path and Path(norm_stats_path).exists():
            builder.load_stats(norm_stats_path)
        else:
            builder.fit_normalisation(df)

        X, y, seg_ids = builder.build(df)

        # Static features
        try:
            static_df = compute_static_features(df)
            static_df = static_df.set_index("segment_id")
            static_arr = static_df.loc[seg_ids].values.astype(np.float32)
            np.nan_to_num(static_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        except Exception as exc:
            logger.warning("Could not compute static features: %s", exc)
            static_arr = None

        # Patches
        patches = None
        if patch_dir is not None:
            patch_dir = Path(patch_dir)
            patch_list = []
            for sid in seg_ids:
                p = patch_dir / f"{sid}.npy"
                if p.exists():
                    patch_list.append(np.load(p))
                else:
                    n_bands = kwargs.get("n_patch_bands", 12)
                    ps = kwargs.get("patch_size", 64)
                    patch_list.append(np.zeros((n_bands, ps, ps), dtype=np.float32))
            patches = np.stack(patch_list, axis=0)

        return cls(
            sequences=X,
            targets=y,
            segment_ids=seg_ids,
            static_features=static_arr,
            patches=patches,
            **kwargs,
        )

    @classmethod
    def train_val_split(
        cls,
        dataset: "RiverSegmentDataset",
        val_fraction: float = 0.2,
        spatial_block: bool = True,
        random_seed: int = 42,
    ) -> Tuple["RiverSegmentDataset", "RiverSegmentDataset"]:
        """Split a dataset into train and validation subsets.

        Parameters
        ----------
        dataset:
            The full dataset to split.
        val_fraction:
            Fraction of segments to hold out for validation.
        spatial_block:
            If True, splits by contiguous segment blocks to avoid
            spatial autocorrelation leakage.  If False, random split.
        random_seed:
            RNG seed for reproducibility.

        Returns
        -------
        (train_dataset, val_dataset)
        """
        N = len(dataset)
        rng = np.random.default_rng(random_seed)

        if spatial_block:
            # Hold out the last val_fraction of sorted segment IDs
            n_val = max(1, int(N * val_fraction))
            indices = np.arange(N)
            val_idx = indices[-n_val:]
            train_idx = indices[:-n_val]
        else:
            indices = rng.permutation(N)
            n_val = max(1, int(N * val_fraction))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

        def _subset(idx: np.ndarray) -> "RiverSegmentDataset":
            return cls(
                sequences=dataset.sequences[idx].numpy(),
                targets=dataset.targets[idx].numpy(),
                segment_ids=dataset.segment_ids[idx],
                static_features=dataset.static_features[idx].numpy(),
                patches=dataset.patches[idx].numpy(),
                patch_size=dataset.patch_size,
                n_patch_bands=dataset.n_patch_bands,
                augment=dataset.augment,
                depth_navigable=dataset.depth_navigable,
                depth_conditional=dataset.depth_conditional,
            )

        return _subset(train_idx), _subset(val_idx)


# ---------------------------------------------------------------------------
# Gauge interpolation utilities
# ---------------------------------------------------------------------------


def interpolate_gauge_to_segments(
    gauge_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    gauge_col: str = "water_level_m",
    method: str = "linear",
) -> pd.DataFrame:
    """Spatially interpolate CWC gauge readings to 5 km river segments.

    Uses inverse-distance weighting (IDW) between the two nearest gauges.

    Parameters
    ----------
    gauge_df:
        DataFrame with columns: ``station_id``, ``chainage_km``,
        ``date``, and *gauge_col*.
    segment_df:
        DataFrame with columns: ``segment_id``, ``chainage_km``, ``date``.
    gauge_col:
        Name of the gauge measurement column to interpolate.
    method:
        Interpolation method — ``"linear"`` (IDW) or ``"nearest"``.

    Returns
    -------
    pd.DataFrame
        *segment_df* with a new column ``{gauge_col}_interp``.
    """
    result_rows: List[Dict] = []

    for _, seg_row in segment_df.iterrows():
        seg_chainage = seg_row["chainage_km"]
        seg_date = seg_row["date"]

        # Filter gauges for same date
        date_gauges = gauge_df[gauge_df["date"] == seg_date].copy()
        if date_gauges.empty:
            interp_val = float("nan")
        else:
            date_gauges = date_gauges.sort_values("chainage_km")
            chainages = date_gauges["chainage_km"].values
            values = date_gauges[gauge_col].values

            if method == "nearest":
                closest_idx = np.argmin(np.abs(chainages - seg_chainage))
                interp_val = float(values[closest_idx])
            else:
                # Linear IDW between two bracketing gauges
                distances = np.abs(chainages - seg_chainage)
                sorted_dist_idx = np.argsort(distances)

                if distances[sorted_dist_idx[0]] < 1e-6:
                    interp_val = float(values[sorted_dist_idx[0]])
                elif len(sorted_dist_idx) >= 2:
                    i0, i1 = sorted_dist_idx[0], sorted_dist_idx[1]
                    d0, d1 = distances[i0], distances[i1]
                    w0, w1 = 1.0 / (d0 + 1e-9), 1.0 / (d1 + 1e-9)
                    interp_val = float((w0 * values[i0] + w1 * values[i1]) / (w0 + w1))
                else:
                    interp_val = float(values[sorted_dist_idx[0]])

        row_dict = dict(seg_row)
        row_dict[f"{gauge_col}_interp"] = interp_val
        result_rows.append(row_dict)

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Navigability label utilities
# ---------------------------------------------------------------------------


def depth_to_nav_label(
    depth_m: float,
    width_m: float = float("inf"),
    depth_navigable: float = DEPTH_NAVIGABLE,
    depth_conditional: float = DEPTH_CONDITIONAL,
    width_navigable: float = 50.0,
) -> int:
    """Convert continuous depth + width to a navigability class index.

    Parameters
    ----------
    depth_m:
        Estimated or measured water depth in metres.
    width_m:
        Water-surface width in metres.  Defaults to inf (not considered).
    depth_navigable:
        Minimum depth for "Navigable" class (default 3.0 m).
    depth_conditional:
        Minimum depth for "Conditional" class (default 2.0 m).
    width_navigable:
        Minimum width for "Navigable" class (default 50 m).

    Returns
    -------
    int
        0 = Non-Navigable, 1 = Conditional, 2 = Navigable.
    """
    if math.isnan(depth_m):
        return 0
    if depth_m >= depth_navigable and width_m >= width_navigable:
        return 2
    if depth_m >= depth_conditional:
        return 1
    return 0


def add_nav_labels(
    df: pd.DataFrame,
    depth_col: str = "depth_m",
    width_col: Optional[str] = "water_width_m",
) -> pd.DataFrame:
    """Add a ``nav_label`` column to a segment DataFrame.

    Parameters
    ----------
    df:
        DataFrame with at least a depth column.
    depth_col:
        Name of the depth column.
    width_col:
        Name of the width column (optional; if absent, width is ignored).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional ``nav_label`` and ``nav_class`` columns.
    """
    df = df.copy()
    widths = (
        df[width_col].values
        if (width_col and width_col in df.columns)
        else (np.full(len(df), float("inf")))
    )
    labels = [
        depth_to_nav_label(float(d), float(w))
        for d, w in zip(df[depth_col].values, widths)
    ]
    df["nav_label"] = labels
    df["nav_class"] = [NAV_CLASSES[lbl] for lbl in labels]
    return df


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # ── Minimal smoke test ──────────────────────────────────────────────
    rng = np.random.default_rng(0)

    extractor = SpectralFeatureExtractor(scale_factor=1e-4)

    # Simulate 10 000-pixel DN values for each band
    fake_bands = {
        b: rng.integers(200, 3000, size=100).astype(float) for b in SENTINEL2_BANDS
    }
    fake_bands["VV"] = rng.uniform(-15, -5, size=100)
    fake_bands["VH"] = rng.uniform(-20, -10, size=100)
    fake_ancillary = {
        "gauge_water_level_m": 4.5,
        "gauge_discharge_m3s": 1200.0,
        "era5_cumulative_rainfall_mm": 85.0,
        "era5_mean_temperature_c": 28.3,
    }
    fake_water_mask = rng.random((64, 64)) > 0.6
    fake_centreline = rng.uniform(0, 100, size=(50, 2))
    fake_mndwi_series = rng.uniform(-0.2, 0.8, size=12)

    feats = extractor.extract(
        bands=fake_bands,
        ancillary=fake_ancillary,
        water_mask=fake_water_mask,
        centreline_coords=fake_centreline,
        mndwi_series=fake_mndwi_series,
    )

    print("\n── SpectralFeatureExtractor output ──")
    for k, v in feats.items():
        print(f"  {k:35s}: {v:.6f}")

    # ── TemporalSequenceBuilder smoke test ─────────────────────────────
    N_segs, T_steps = 20, 12
    seg_ids = [f"NW1_SEG_{i:04d}" for i in range(N_segs)]
    dates = pd.date_range("2022-01-01", periods=T_steps, freq="MS")

    rows = []
    for sid in seg_ids:
        for dt in dates:
            row = {"segment_id": sid, "date": dt, "depth_m": rng.uniform(1, 6)}
            row.update({k: rng.uniform(0, 1) for k in ALL_FEATURES[:10]})
            row.update({k: 0.0 for k in ALL_FEATURES[10:]})
            rows.append(row)

    df_fake = pd.DataFrame(rows)
    builder = TemporalSequenceBuilder(SequenceConfig(normalise=True))
    X, y, ids = builder.build(df_fake)
    print(f"\n── TemporalSequenceBuilder output ──")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  ids     : {ids[:3]} …")

    # ── RiverSegmentDataset smoke test ─────────────────────────────────
    ds = RiverSegmentDataset(X, y, ids)
    sample = ds[0]
    print(f"\n── RiverSegmentDataset[0] ──")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:15s}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k:15s}: {v}")

    print("\nAll smoke tests passed.")
