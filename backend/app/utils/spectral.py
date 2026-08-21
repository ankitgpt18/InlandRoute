"""
AIDSTL Project — Spectral Index Computation Utilities
======================================================
Functions for computing water-body spectral indices from Sentinel-2 band
reflectance values.  All inputs are expected as floating-point arrays with
values in the range [0, 1] (i.e. after dividing raw DN by 10 000 for
Sentinel-2 L2A products).

Indices implemented
-------------------
  MNDWI  — Modified Normalised Difference Water Index (Xu 2006)
  NDWI   — Normalised Difference Water Index (McFeeters 1996)
  AWEI   — Automated Water Extraction Index (Feyisa et al. 2014)
             both shadow-robust (AWEIsh) and no-shadow (AWEInsh) variants
  Stumpf — Log-ratio bathymetric index (Stumpf et al. 2003)
  NDTI   — Normalised Difference Turbidity Index (Lacaux et al. 2007)
  NDSI   — Normalised Difference Sediment Index
  EVI    — Enhanced Vegetation Index (to mask riparian vegetation)
  NDVI   — Normalised Difference Vegetation Index

Feature engineering
-------------------
  build_feature_vector  — assemble all indices + raw bands into a 1-D array
  normalize_features    — standardise with (optional) fitted sklearn scaler

References
----------
  McFeeters S.K. (1996) Int. J. Remote Sens., 17(7), 1425–1432.
  Xu H. (2006) Int. J. Remote Sens., 27(14), 3025–3033.
  Feyisa G.L. et al. (2014) Remote Sens. Environ., 140, 23–35.
  Stumpf R.P. et al. (2003) Limnol. Oceanogr., 48(1 part 2), 547–556.
  Lacaux J.P. et al. (2007) Remote Sens. Environ., 107(1–2), 141–149.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FloatArray = NDArray[np.float64]
BandInput = Union[float, ArrayLike]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Small epsilon to avoid division-by-zero in ratio calculations
_EPS: float = 1e-9

# Log base used by Stumpf bathymetric index
_STUMPF_BASE: float = 1000.0

# Expected band names when building feature vectors
SENTINEL2_BANDS: tuple[str, ...] = (
    "blue",  # B2  — 490 nm
    "green",  # B3  — 560 nm
    "red",  # B4  — 665 nm
    "red_edge_1",  # B5  — 705 nm
    "red_edge_2",  # B6  — 740 nm
    "red_edge_3",  # B7  — 783 nm
    "nir",  # B8  — 842 nm
    "nir_narrow",  # B8A — 865 nm
    "swir1",  # B11 — 1610 nm
    "swir2",  # B12 — 2190 nm
)

# Ordered list of feature names produced by build_feature_vector()
FEATURE_NAMES: tuple[str, ...] = (
    # Raw bands
    "blue",
    "green",
    "red",
    "red_edge_1",
    "red_edge_2",
    "red_edge_3",
    "nir",
    "nir_narrow",
    "swir1",
    "swir2",
    # Water indices
    "mndwi",
    "ndwi",
    "awei_nsh",
    "awei_sh",
    # Bathymetric
    "stumpf_bg",
    "stumpf_brg",
    # Turbidity / sediment
    "ndti",
    "ndsi",
    # Vegetation (for masking / context)
    "ndvi",
    "evi",
    # Texture / ratio features
    "nir_swir1_ratio",
    "blue_red_ratio",
    "green_nir_ratio",
    # Band statistics (band sum, variance proxy)
    "vis_sum",
    "nir_swir_sum",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float64(arr: BandInput) -> FloatArray:
    """Convert any array-like or scalar to a float64 NumPy array."""
    return np.asarray(arr, dtype=np.float64)


def _safe_normalised_diff(a: FloatArray, b: FloatArray) -> FloatArray:
    """Compute (a - b) / (a + b) with safe division.

    Returns 0.0 where the denominator is effectively zero.
    """
    numerator = a - b
    denominator = a + b
    return np.where(np.abs(denominator) > _EPS, numerator / denominator, 0.0)


def _clip_index(arr: FloatArray, low: float = -1.0, high: float = 1.0) -> FloatArray:
    """Clip an index to its valid theoretical range."""
    return np.clip(arr, low, high)


# ---------------------------------------------------------------------------
# Water extraction indices
# ---------------------------------------------------------------------------


def compute_mndwi(green: BandInput, swir: BandInput) -> FloatArray:
    """Compute the Modified Normalised Difference Water Index (MNDWI).

    Formula:
        MNDWI = (Green - SWIR1) / (Green + SWIR1)

    Water bodies typically have MNDWI > 0, while built-up areas and dry
    land have MNDWI < 0.  Xu (2006) showed this outperforms NDWI in urban
    and semi-arid contexts.

    Parameters
    ----------
    green : array-like
        Sentinel-2 Band 3 (Green, ~560 nm) reflectance in [0, 1].
    swir : array-like
        Sentinel-2 Band 11 (SWIR1, ~1610 nm) reflectance in [0, 1].

    Returns
    -------
    FloatArray
        MNDWI values clipped to [-1, 1].

    Examples
    --------
    >>> compute_mndwi(0.15, 0.05)
    array(0.5)
    """
    g = _to_float64(green)
    s = _to_float64(swir)
    result = _safe_normalised_diff(g, s)
    return _clip_index(result)


def compute_ndwi(green: BandInput, nir: BandInput) -> FloatArray:
    """Compute the Normalised Difference Water Index (NDWI).

    Formula:
        NDWI = (Green - NIR) / (Green + NIR)

    The original McFeeters (1996) index.  Generally positive over open
    water; negative over vegetation and soil.  In riverine contexts MNDWI
    is preferred, but NDWI remains a useful secondary feature.

    Parameters
    ----------
    green : array-like
        Sentinel-2 Band 3 (Green, ~560 nm) reflectance in [0, 1].
    nir : array-like
        Sentinel-2 Band 8 (NIR, ~842 nm) reflectance in [0, 1].

    Returns
    -------
    FloatArray
        NDWI values clipped to [-1, 1].
    """
    g = _to_float64(green)
    n = _to_float64(nir)
    result = _safe_normalised_diff(g, n)
    return _clip_index(result)


def compute_awei(
    blue: BandInput,
    green: BandInput,
    nir: BandInput,
    swir1: BandInput,
    swir2: BandInput,
    shadow_robust: bool = True,
) -> FloatArray:
    """Compute the Automated Water Extraction Index (AWEI).

    Two variants are provided (Feyisa et al. 2014):

    AWEInsh (no-shadow):
        4 * (Green - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)

    AWEIsh (shadow-robust):
        Blue + 2.5 * Green - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2

    AWEIsh is generally recommended for riverine environments with mixed
    shadowed riparian cover.

    Parameters
    ----------
    blue : array-like
        Sentinel-2 Band 2 (Blue, ~490 nm) reflectance in [0, 1].
    green : array-like
        Sentinel-2 Band 3 (Green, ~560 nm) reflectance in [0, 1].
    nir : array-like
        Sentinel-2 Band 8 (NIR, ~842 nm) reflectance in [0, 1].
    swir1 : array-like
        Sentinel-2 Band 11 (SWIR1, ~1610 nm) reflectance in [0, 1].
    swir2 : array-like
        Sentinel-2 Band 12 (SWIR2, ~2190 nm) reflectance in [0, 1].
    shadow_robust : bool, optional
        If ``True`` (default), return AWEIsh; otherwise return AWEInsh.

    Returns
    -------
    FloatArray
        AWEI values.  Positive values indicate water presence.
    """
    b = _to_float64(blue)
    g = _to_float64(green)
    n = _to_float64(nir)
    s1 = _to_float64(swir1)
    s2 = _to_float64(swir2)

    if shadow_robust:
        # AWEIsh
        result = b + 2.5 * g - 1.5 * (n + s1) - 0.25 * s2
    else:
        # AWEInsh
        result = 4.0 * (g - s1) - (0.25 * n + 2.75 * s2)

    return result.astype(np.float64)


def compute_awei_nsh(
    green: BandInput,
    nir: BandInput,
    swir1: BandInput,
    swir2: BandInput,
) -> FloatArray:
    """Convenience wrapper — AWEInsh (no-shadow variant).

    Parameters
    ----------
    green, nir, swir1, swir2 : array-like
        Band reflectances in [0, 1].

    Returns
    -------
    FloatArray
        AWEInsh values; positive indicates water.
    """
    return compute_awei(
        blue=np.zeros_like(_to_float64(green)),
        green=green,
        nir=nir,
        swir1=swir1,
        swir2=swir2,
        shadow_robust=False,
    )


def compute_awei_sh(
    blue: BandInput,
    green: BandInput,
    nir: BandInput,
    swir1: BandInput,
    swir2: BandInput,
) -> FloatArray:
    """Convenience wrapper — AWEIsh (shadow-robust variant).

    Parameters
    ----------
    blue, green, nir, swir1, swir2 : array-like
        Band reflectances in [0, 1].

    Returns
    -------
    FloatArray
        AWEIsh values; positive indicates water.
    """
    return compute_awei(
        blue=blue,
        green=green,
        nir=nir,
        swir1=swir1,
        swir2=swir2,
        shadow_robust=True,
    )


# ---------------------------------------------------------------------------
# Bathymetric index
# ---------------------------------------------------------------------------


def compute_stumpf_ratio(
    blue: BandInput,
    green: BandInput,
    red: Optional[BandInput] = None,
    n: float = 1.0,
) -> FloatArray:
    """Compute the Stumpf log-ratio bathymetric index.

    For two-band (Blue / Green) version (Stumpf et al. 2003):
        Depth ∝ m * [ln(n * Blue) / ln(n * Green)] - b

    This function returns the dimensionless log-ratio; the caller is
    responsible for applying gain ``m`` and offset ``b`` calibrated against
    in-situ gauge observations.

    When ``red`` is supplied, an extended three-band log-ratio is also
    returned as the second element of the output tuple.

    Parameters
    ----------
    blue : array-like
        Sentinel-2 Band 2 (Blue, ~490 nm) reflectance in [0, 1].
    green : array-like
        Sentinel-2 Band 3 (Green, ~560 nm) reflectance in [0, 1].
    red : array-like, optional
        Sentinel-2 Band 4 (Red, ~665 nm) reflectance in [0, 1].
        When supplied, compute ``ln(n*Blue) / ln(n*Red)`` as well.
    n : float, optional
        Scaling constant (default 1000, matching Stumpf et al. 2003).

    Returns
    -------
    FloatArray
        Log-ratio values (Blue/Green).  Larger values typically correspond
        to shallower water because blue penetrates less than green.

    Notes
    -----
    Reflectance values are clipped to [1e-6, 1] before log transformation
    to avoid log(0) singularities in fully saturated or masked pixels.
    """
    n_val = float(n) if n != 1.0 else _STUMPF_BASE

    b = np.clip(_to_float64(blue), 1e-6, 1.0)
    g = np.clip(_to_float64(green), 1e-6, 1.0)

    log_b = np.log(n_val * b)
    log_g = np.log(n_val * g)

    ratio_bg = np.where(np.abs(log_g) > _EPS, log_b / log_g, 0.0)

    return ratio_bg.astype(np.float64)


def compute_stumpf_ratio_brg(
    blue: BandInput,
    red: BandInput,
    green: BandInput,
    n: float = 1.0,
) -> FloatArray:
    """Blue/Red log-ratio variant of the Stumpf bathymetric index.

    Useful as an additional depth proxy since red light attenuates faster
    than green in turbid inland waters.

    Parameters
    ----------
    blue, red, green : array-like
        Band reflectances in [0, 1].
    n : float, optional
        Scaling constant (default 1000).

    Returns
    -------
    FloatArray
        Log-ratio Blue/Red values.
    """
    n_val = float(n) if n != 1.0 else _STUMPF_BASE

    b = np.clip(_to_float64(blue), 1e-6, 1.0)
    r = np.clip(_to_float64(red), 1e-6, 1.0)

    log_b = np.log(n_val * b)
    log_r = np.log(n_val * r)

    ratio_br = np.where(np.abs(log_r) > _EPS, log_b / log_r, 0.0)

    return ratio_br.astype(np.float64)


# ---------------------------------------------------------------------------
# Turbidity / sediment indices
# ---------------------------------------------------------------------------


def compute_turbidity(red: BandInput, green: BandInput) -> FloatArray:
    """Compute the Normalised Difference Turbidity Index (NDTI).

    Formula:
        NDTI = (Red - Green) / (Red + Green)

    Higher NDTI values indicate greater suspended sediment load (higher
    turbidity), which is common during monsoon season in Ganga/Brahmaputra.

    Parameters
    ----------
    red : array-like
        Sentinel-2 Band 4 (Red, ~665 nm) reflectance in [0, 1].
    green : array-like
        Sentinel-2 Band 3 (Green, ~560 nm) reflectance in [0, 1].

    Returns
    -------
    FloatArray
        NDTI values clipped to [-1, 1].

    References
    ----------
    Lacaux J.P. et al. (2007) Remote Sens. Environ., 107(1–2), 141–149.
    """
    r = _to_float64(red)
    g = _to_float64(green)
    result = _safe_normalised_diff(r, g)
    return _clip_index(result)


def compute_ndsi(green: BandInput, swir1: BandInput) -> FloatArray:
    """Compute the Normalised Difference Sediment Index (NDSI).

    Note: not to be confused with Snow Index (which uses the same bands
    but a different threshold direction).  In riverine contexts this serves
    as a suspended sediment proxy.

    Formula:
        NDSI = (Green - SWIR1) / (Green + SWIR1)
        (same as MNDWI but interpreted as a sediment signal over water masks)

    Parameters
    ----------
    green : array-like
        Sentinel-2 Band 3 (Green, ~560 nm) reflectance in [0, 1].
    swir1 : array-like
        Sentinel-2 Band 11 (SWIR1, ~1610 nm) reflectance in [0, 1].

    Returns
    -------
    FloatArray
        NDSI values clipped to [-1, 1].
    """
    g = _to_float64(green)
    s = _to_float64(swir1)
    result = _safe_normalised_diff(g, s)
    return _clip_index(result)


# ---------------------------------------------------------------------------
# Vegetation indices (used as auxiliary / masking features)
# ---------------------------------------------------------------------------


def compute_ndvi(nir: BandInput, red: BandInput) -> FloatArray:
    """Compute the Normalised Difference Vegetation Index (NDVI).

    Formula:
        NDVI = (NIR - Red) / (NIR + Red)

    High NDVI within a river-segment polygon indicates riparian vegetation
    or sand-bar encroachment — an important context feature for navigability.

    Parameters
    ----------
    nir : array-like
        Sentinel-2 Band 8 (NIR, ~842 nm) reflectance in [0, 1].
    red : array-like
        Sentinel-2 Band 4 (Red, ~665 nm) reflectance in [0, 1].

    Returns
    -------
    FloatArray
        NDVI values clipped to [-1, 1].
    """
    n = _to_float64(nir)
    r = _to_float64(red)
    result = _safe_normalised_diff(n, r)
    return _clip_index(result)


def compute_evi(
    nir: BandInput,
    red: BandInput,
    blue: BandInput,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    l: float = 1.0,
) -> FloatArray:
    """Compute the Enhanced Vegetation Index (EVI).

    Formula:
        EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

    EVI reduces atmospheric and soil-background noise compared with NDVI,
    making it more reliable for monitoring riverbank vegetation.

    Parameters
    ----------
    nir : array-like
        Sentinel-2 Band 8 (NIR, ~842 nm) reflectance in [0, 1].
    red : array-like
        Sentinel-2 Band 4 (Red, ~665 nm) reflectance in [0, 1].
    blue : array-like
        Sentinel-2 Band 2 (Blue, ~490 nm) reflectance in [0, 1].
    g : float, optional
        Gain factor (default 2.5).
    c1, c2 : float, optional
        Aerosol resistance coefficients (defaults 6.0, 7.5).
    l : float, optional
        Canopy background adjustment factor (default 1.0).

    Returns
    -------
    FloatArray
        EVI values clipped to [-1, 1].
    """
    n = _to_float64(nir)
    r = _to_float64(red)
    b = _to_float64(blue)

    numerator = g * (n - r)
    denominator = n + c1 * r - c2 * b + l
    result = np.where(np.abs(denominator) > _EPS, numerator / denominator, 0.0)
    return _clip_index(result.astype(np.float64))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_feature_vector(
    bands: dict[str, float | ArrayLike],
    include_indices: bool = True,
    extra_features: Optional[dict[str, float | ArrayLike]] = None,
) -> FloatArray:
    """Assemble a flat feature vector from raw Sentinel-2 band reflectances.

    The function computes all spectral indices listed in ``FEATURE_NAMES``
    and concatenates them with the raw bands to produce a single 1-D
    (or 2-D if arrays are passed) feature matrix ready for model inference.

    Required keys in ``bands``
    -------------------------
    ``blue``, ``green``, ``red``, ``red_edge_1``, ``red_edge_2``,
    ``red_edge_3``, ``nir``, ``nir_narrow``, ``swir1``, ``swir2``

    All values should be reflectances in [0, 1].  Missing keys raise a
    ``KeyError``; unexpected extra keys are silently ignored.

    Parameters
    ----------
    bands : dict[str, float | array-like]
        Dictionary mapping Sentinel-2 band names (see ``SENTINEL2_BANDS``)
        to reflectance values.  Values may be scalars or arrays of the same
        shape (e.g. one value per pixel in a segment).
    include_indices : bool, optional
        Whether to compute and append spectral indices.  When ``False``,
        only the 10 raw band values are returned (useful for ablation
        studies).  Default ``True``.
    extra_features : dict, optional
        Additional numeric features to append verbatim (e.g. gauge height,
        precipitation, morphometric properties).

    Returns
    -------
    FloatArray
        Shape ``(n_features,)`` if scalar inputs, or ``(n_pixels, n_features)``
        if array inputs.

    Raises
    ------
    KeyError
        If a required band key is missing from ``bands``.
    ValueError
        If band arrays have inconsistent shapes.

    Examples
    --------
    >>> b = {"blue": 0.05, "green": 0.12, "red": 0.08, "red_edge_1": 0.10,
    ...      "red_edge_2": 0.14, "red_edge_3": 0.15, "nir": 0.18,
    ...      "nir_narrow": 0.17, "swir1": 0.06, "swir2": 0.03}
    >>> fv = build_feature_vector(b)
    >>> fv.shape
    (25,)
    """
    # ---- validate required bands ------------------------------------------
    missing = [k for k in SENTINEL2_BANDS if k not in bands]
    if missing:
        raise KeyError(
            f"Missing required band(s) for feature vector: {missing}. "
            f"Expected keys: {SENTINEL2_BANDS}"
        )

    # ---- extract bands as float64 arrays ----------------------------------
    blue = _to_float64(bands["blue"])
    green = _to_float64(bands["green"])
    red = _to_float64(bands["red"])
    re1 = _to_float64(bands["red_edge_1"])
    re2 = _to_float64(bands["red_edge_2"])
    re3 = _to_float64(bands["red_edge_3"])
    nir = _to_float64(bands["nir"])
    nir_n = _to_float64(bands["nir_narrow"])
    swir1 = _to_float64(bands["swir1"])
    swir2 = _to_float64(bands["swir2"])

    # Determine whether we are dealing with scalars or arrays
    scalar_mode = all(arr.ndim == 0 for arr in [blue, green, red, nir, swir1])

    # Raw band list
    raw_bands: list[FloatArray] = [
        blue,
        green,
        red,
        re1,
        re2,
        re3,
        nir,
        nir_n,
        swir1,
        swir2,
    ]

    if not include_indices:
        feature_parts = raw_bands
    else:
        # ---- compute water / depth indices --------------------------------
        mndwi = compute_mndwi(green, swir1)
        ndwi = compute_ndwi(green, nir)
        awei_nsh = compute_awei_nsh(green, nir, swir1, swir2)
        awei_sh = compute_awei_sh(blue, green, nir, swir1, swir2)

        # ---- compute bathymetric indices ----------------------------------
        stumpf_bg = compute_stumpf_ratio(blue, green)
        stumpf_brg = compute_stumpf_ratio_brg(blue, red, green)

        # ---- compute turbidity / sediment ---------------------------------
        ndti = compute_turbidity(red, green)
        ndsi = compute_ndsi(green, swir1)

        # ---- compute vegetation indices -----------------------------------
        ndvi_val = compute_ndvi(nir, red)
        evi_val = compute_evi(nir, red, blue)

        # ---- ratio features -----------------------------------------------
        nir_swir1_ratio = np.where(
            np.abs(swir1) > _EPS, nir / (swir1 + _EPS), 0.0
        ).astype(np.float64)
        blue_red_ratio = np.where(np.abs(red) > _EPS, blue / (red + _EPS), 0.0).astype(
            np.float64
        )
        green_nir_ratio = np.where(
            np.abs(nir) > _EPS, green / (nir + _EPS), 0.0
        ).astype(np.float64)

        # ---- band-sum features --------------------------------------------
        vis_sum = (blue + green + red).astype(np.float64)
        nir_swir_sum = (nir + nir_n + swir1 + swir2).astype(np.float64)

        feature_parts = (
            raw_bands
            + [mndwi, ndwi, awei_nsh, awei_sh]
            + [stumpf_bg, stumpf_brg]
            + [ndti, ndsi]
            + [ndvi_val, evi_val]
            + [nir_swir1_ratio, blue_red_ratio, green_nir_ratio]
            + [vis_sum, nir_swir_sum]
        )

    # ---- append any extra features ----------------------------------------
    if extra_features:
        for key, val in extra_features.items():
            feature_parts.append(_to_float64(val))

    # ---- stack into array -------------------------------------------------
    if scalar_mode:
        feature_vector = np.array([float(f) for f in feature_parts], dtype=np.float64)
    else:
        # Stack along the last axis: (n_pixels, n_features)
        expanded = [
            (f if f.ndim > 0 else np.full_like(feature_parts[0], float(f)))
            for f in feature_parts
        ]
        feature_vector = np.stack(expanded, axis=-1).astype(np.float64)

    logger.debug(
        "Feature vector built: shape=%s, include_indices=%s",
        feature_vector.shape,
        include_indices,
    )
    return feature_vector


def normalize_features(
    X: ArrayLike,
    scaler: Optional[StandardScaler] = None,
    fit: bool = False,
) -> tuple[FloatArray, StandardScaler]:
    """Standardise feature array to zero-mean unit-variance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or (n_features,)
        Feature matrix or vector to standardise.
    scaler : sklearn.preprocessing.StandardScaler, optional
        A pre-fitted scaler instance.  When provided, ``fit`` is ignored and
        this scaler is used for transformation.  Pass the scaler saved
        alongside the trained model to ensure consistent standardisation
        between training and inference.
    fit : bool, optional
        When ``True`` (and ``scaler`` is ``None``), fit a new
        :class:`StandardScaler` on ``X`` and return it.  When ``False``
        and ``scaler`` is ``None``, the function returns ``X`` unchanged
        together with a freshly created (unfitted) scaler object.

    Returns
    -------
    X_norm : FloatArray
        Standardised feature array with the same shape as the input.
    scaler : StandardScaler
        The fitted scaler (useful for persisting alongside model artefacts).

    Raises
    ------
    ValueError
        If a pre-fitted ``scaler`` is provided but has not been fitted yet.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(100, 25)
    >>> X_norm, fitted_scaler = normalize_features(X, fit=True)
    >>> # Later, at inference time:
    >>> X_inf, _ = normalize_features(X_new, scaler=fitted_scaler)
    """
    X_arr = np.asarray(X, dtype=np.float64)
    was_1d = X_arr.ndim == 1
    if was_1d:
        X_arr = X_arr.reshape(1, -1)

    if scaler is not None:
        # Use the provided fitted scaler
        if not hasattr(scaler, "mean_") or scaler.mean_ is None:
            raise ValueError(
                "The provided scaler has not been fitted. "
                "Call normalize_features(X, fit=True) first, or pass a fitted scaler."
            )
        X_norm = scaler.transform(X_arr)
    elif fit:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_arr)
        logger.info(
            "StandardScaler fitted on %d samples with %d features.",
            X_arr.shape[0],
            X_arr.shape[1],
        )
    else:
        # No scaler provided and fit=False — return as-is
        scaler = StandardScaler()
        X_norm = X_arr.copy()
        logger.warning(
            "normalize_features called without scaler or fit=True. "
            "Features are returned unstandardised."
        )

    X_norm = X_norm.astype(np.float64)
    if was_1d:
        X_norm = X_norm.reshape(-1)

    return X_norm, scaler  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Bulk / pixel-level helpers
# ---------------------------------------------------------------------------


def compute_all_indices(bands: dict[str, ArrayLike]) -> dict[str, FloatArray]:
    """Compute all spectral indices from a band dictionary.

    Convenience function that returns a named dictionary of all indices
    instead of a flat feature vector.  Useful for exploratory analysis,
    GEE export scripts, and diagnostic visualisations.

    Parameters
    ----------
    bands : dict[str, array-like]
        Same format as accepted by :func:`build_feature_vector`.

    Returns
    -------
    dict[str, FloatArray]
        Keys are index names (``mndwi``, ``ndwi``, etc.);
        values are float64 arrays.
    """
    missing = [k for k in SENTINEL2_BANDS if k not in bands]
    if missing:
        raise KeyError(f"Missing required band(s): {missing}")

    b = _to_float64(bands["blue"])
    g = _to_float64(bands["green"])
    r = _to_float64(bands["red"])
    n = _to_float64(bands["nir"])
    s1 = _to_float64(bands["swir1"])
    s2 = _to_float64(bands["swir2"])

    return {
        "mndwi": compute_mndwi(g, s1),
        "ndwi": compute_ndwi(g, n),
        "awei_nsh": compute_awei_nsh(g, n, s1, s2),
        "awei_sh": compute_awei_sh(b, g, n, s1, s2),
        "stumpf_bg": compute_stumpf_ratio(b, g),
        "stumpf_brg": compute_stumpf_ratio_brg(b, r, g),
        "ndti": compute_turbidity(r, g),
        "ndsi": compute_ndsi(g, s1),
        "ndvi": compute_ndvi(n, r),
        "evi": compute_evi(n, r, b),
    }


def water_mask_from_mndwi(
    green: BandInput,
    swir1: BandInput,
    threshold: float = 0.0,
) -> NDArray[np.bool_]:
    """Generate a binary water mask using an MNDWI threshold.

    Parameters
    ----------
    green : array-like
        Sentinel-2 Band 3 reflectance in [0, 1].
    swir1 : array-like
        Sentinel-2 Band 11 reflectance in [0, 1].
    threshold : float, optional
        MNDWI threshold above which a pixel is classified as water.
        Default 0.0 (the conventional threshold from Xu 2006).

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask; ``True`` = water pixel.
    """
    mndwi = compute_mndwi(green, swir1)
    return mndwi > threshold


def estimate_depth_from_stumpf(
    blue: BandInput,
    green: BandInput,
    m1: float,
    m0: float,
    n: float = 1000.0,
) -> FloatArray:
    """Estimate water depth (metres) using the calibrated Stumpf model.

    Full empirical model (Stumpf et al. 2003):
        Depth = m1 * [ln(n * Blue) / ln(n * Green)] - m0

    Parameters
    ----------
    blue : array-like
        Sentinel-2 Band 2 reflectance in [0, 1].
    green : array-like
        Sentinel-2 Band 3 reflectance in [0, 1].
    m1 : float
        Gain coefficient calibrated against in-situ gauge data.
    m0 : float
        Offset coefficient calibrated against in-situ gauge data.
    n : float, optional
        Scaling constant (default 1000).

    Returns
    -------
    FloatArray
        Estimated depth in metres, clipped to [0, ∞).
    """
    ratio = compute_stumpf_ratio(blue, green, n=n)
    depth = m1 * ratio - m0
    return np.maximum(depth, 0.0).astype(np.float64)


def aggregate_segment_features(
    pixel_features: FloatArray,
    aggregations: Sequence[str] = ("mean", "median", "std", "min", "max", "p25", "p75"),
) -> FloatArray:
    """Aggregate pixel-level feature vectors over a river segment.

    The model operates on 5 km analysis units; within each unit many
    pixels contribute reflectance values.  This function reduces the
    pixel-level matrix to a single representative feature vector using
    multiple statistical aggregations.

    Parameters
    ----------
    pixel_features : FloatArray of shape (n_pixels, n_features)
        Feature matrix where each row is one pixel's feature vector.
    aggregations : sequence of str, optional
        Statistical aggregations to apply per feature column.
        Supported: ``"mean"``, ``"median"``, ``"std"``, ``"min"``,
        ``"max"``, ``"p25"`` (25th percentile), ``"p75"`` (75th percentile).

    Returns
    -------
    FloatArray of shape (n_features * n_aggregations,)
        Concatenated aggregated feature vector.

    Raises
    ------
    ValueError
        If ``pixel_features`` is not 2-D or ``aggregations`` contains an
        unknown aggregation name.
    """
    arr = np.asarray(pixel_features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"pixel_features must be 2-D (n_pixels, n_features), got shape {arr.shape}"
        )

    _agg_map: dict[str, callable] = {  # type: ignore[type-arg]
        "mean": lambda a: np.mean(a, axis=0),
        "median": lambda a: np.median(a, axis=0),
        "std": lambda a: (
            np.std(a, axis=0, ddof=1) if a.shape[0] > 1 else np.zeros(a.shape[1])
        ),
        "min": lambda a: np.min(a, axis=0),
        "max": lambda a: np.max(a, axis=0),
        "p25": lambda a: np.percentile(a, 25, axis=0),
        "p75": lambda a: np.percentile(a, 75, axis=0),
    }

    unknown = [agg for agg in aggregations if agg not in _agg_map]
    if unknown:
        raise ValueError(
            f"Unknown aggregation(s): {unknown}. Supported: {list(_agg_map.keys())}"
        )

    parts = [_agg_map[agg](arr) for agg in aggregations]
    aggregated = np.concatenate(parts, axis=0)

    logger.debug(
        "Segment features aggregated: %d pixels → %d features (%s)",
        arr.shape[0],
        aggregated.shape[0],
        ", ".join(aggregations),
    )
    return aggregated
