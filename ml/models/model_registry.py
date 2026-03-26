"""
model_registry.py
=================
Model artefact registry for the AIDSTL inland-waterway navigability project.

Responsibilities
----------------
  ModelRegistry
    • Register new model versions with full metadata and metrics.
    • Save / load model artefacts (PyTorch checkpoints, joblib pickles,
      LightGBM / XGBoost native formats).
    • Promote a version to "production" or "staging" deployment stage.
    • Generate machine-readable JSON model cards (Model Card Toolkit style).
    • Query the registry for the best version by a chosen metric.
    • Deprecate / delete old versions with audit trail.

Storage layout
--------------
  <registry_root>/
    registry.json                  ← master index (all versions)
    <model_name>/
      <version>/
        artefacts/                 ← model weights, serialised objects
        model_card.json            ← auto-generated model card
        metrics.json               ← evaluation metrics snapshot
        config.json                ← training hyperparameters
        metadata.json              ← timestamps, tags, notes

Usage example
-------------
  registry = ModelRegistry("ml/models/registry")
  version = registry.register(
      model_name="HydroFormer",
      model_object=trained_model,
      metrics={"r2": 0.923, "rmse": 1.12},
      config=config.to_dict(),
      tags=["sentinel2", "nw1"],
  )
  registry.promote(model_name="HydroFormer", version=version, stage="production")
  best = registry.best_version("HydroFormer", metric="r2", higher_is_better=True)
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
import shutil
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Optional heavy imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGISTRY_INDEX_FILE: str = "registry.json"
MODEL_CARD_FILE: str = "model_card.json"
METRICS_FILE: str = "metrics.json"
CONFIG_FILE: str = "config.json"
METADATA_FILE: str = "metadata.json"
ARTEFACTS_DIR: str = "artefacts"

# Deployment stages ordered by maturity
STAGES: List[str] = ["experimental", "staging", "production", "archived", "deprecated"]

# Artefact type identifiers
ARTEFACT_TYPES: Dict[str, str] = {
    "pytorch": ".pt",
    "lightgbm": ".txt",
    "xgboost": ".json",
    "sklearn": ".pkl",
    "joblib": ".pkl",
    "numpy": ".npz",
    "json": ".json",
}

# Project metadata (embedded in every model card)
PROJECT_META: Dict[str, str] = {
    "project": "AIDSTL",
    "title": "Predicting Inland Waterway Navigability Using Satellite Remote Sensing and Deep Learning",
    "study_areas": "NW-1 (Ganga: Varanasi-Haldia, ~1390 km), NW-2 (Brahmaputra: Dhubri-Sadiya, ~891 km)",
    "data_sources": "Sentinel-2 (10 m, 5-day), CWC gauge readings, ERA5 climate, SRTM DEM",
    "task_a": "Depth Estimation (Regression) — target R2 > 0.90, RMSE < 1.5 m",
    "task_b": "Navigability Classification — 3 classes: Navigable, Conditional, Non-Navigable",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VersionRecord:
    """A single entry in the model registry index.

    Attributes
    ----------
    model_name : str
        Logical model family name, e.g. ``"HydroFormer"``.
    version : str
        Semantic version string, e.g. ``"1.0.0"``.
    run_id : str
        Unique run identifier (UUID4).
    stage : str
        Deployment stage: experimental / staging / production / archived / deprecated.
    created_at : str
        ISO-8601 UTC timestamp of registration.
    updated_at : str
        ISO-8601 UTC timestamp of last metadata update.
    author : str
        Name or identifier of the person / CI job that created this version.
    description : str
        Free-text description of the model version.
    tags : List[str]
        Searchable tags, e.g. ``["nw1", "sentinel2", "tft"]``.
    metrics : Dict[str, float]
        Evaluation metrics snapshot at registration time.
    artefact_paths : Dict[str, str]
        Mapping from artefact role to relative file path within the version dir.
    git_commit : str
        Git SHA at time of training (empty if not in a git repo).
    python_version : str
        Python version used for training.
    framework_versions : Dict[str, str]
        Key framework versions (torch, lightgbm, xgboost, sklearn, ...).
    notes : str
        Additional free-text notes or change-log entry.
    is_deleted : bool
        Soft-delete flag.
    """

    model_name: str
    version: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage: str = "experimental"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    author: str = "unknown"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    artefact_paths: Dict[str, str] = field(default_factory=dict)
    git_commit: str = ""
    python_version: str = platform.python_version()
    framework_versions: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_version(v: str) -> Tuple[int, int, int]:
    """Parse ``"MAJOR.MINOR.PATCH"`` into a sortable tuple."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", v.strip())
    if not match:
        raise ValueError(f"Version '{v}' does not match MAJOR.MINOR.PATCH format.")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _bump_version(current: str, part: str = "patch") -> str:
    """Increment the version string by the specified part.

    Parameters
    ----------
    current : str
        Current version, e.g. ``"1.2.3"``.
    part : str
        One of ``"major"``, ``"minor"``, ``"patch"``.

    Returns
    -------
    str
        Bumped version string.
    """
    major, minor, patch = _parse_version(current)
    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    else:
        return f"{major}.{minor}.{patch + 1}"


def _sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _collect_framework_versions() -> Dict[str, str]:
    """Collect installed framework version strings."""
    versions: Dict[str, str] = {"python": platform.python_version()}
    for pkg_name in (
        "torch",
        "lightgbm",
        "xgboost",
        "sklearn",
        "numpy",
        "pandas",
        "timm",
    ):
        try:
            import importlib

            mod = importlib.import_module(pkg_name)
            versions[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def _get_git_commit() -> str:
    """Return the current HEAD git SHA (empty string on failure)."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Artefact serialisers / deserialisers
# ---------------------------------------------------------------------------


class ArtefactIO:
    """Handles saving and loading of diverse model artefact types.

    Supported types
    ---------------
    ``"pytorch"``    — ``torch.nn.Module`` or state-dict
    ``"lightgbm"``   — ``lgb.Booster``
    ``"xgboost"``    — ``xgb.Booster``
    ``"sklearn"``    — scikit-learn estimator (joblib)
    ``"joblib"``     — any joblib-serialisable Python object
    ``"numpy"``      — ``np.ndarray`` or dict of arrays (.npz)
    ``"json"``       — any JSON-serialisable Python object
    """

    @staticmethod
    def save(
        obj: Any,
        path: Path,
        artefact_type: str = "joblib",
    ) -> Path:
        """Serialise *obj* to *path* and return the actual path written.

        The file extension is automatically set to match *artefact_type*.
        """
        path = path.with_suffix(ARTEFACT_TYPES.get(artefact_type, ".pkl"))
        path.parent.mkdir(parents=True, exist_ok=True)

        if artefact_type == "pytorch":
            if not TORCH_AVAILABLE:
                raise ImportError("torch is required to save pytorch artefacts.")
            import torch

            if isinstance(obj, torch.nn.Module):
                torch.save(obj.state_dict(), path)
            else:
                torch.save(obj, path)

        elif artefact_type == "lightgbm":
            if not LGB_AVAILABLE:
                raise ImportError("lightgbm is required to save LightGBM artefacts.")
            obj.save_model(str(path))

        elif artefact_type == "xgboost":
            if not XGB_AVAILABLE:
                raise ImportError("xgboost is required to save XGBoost artefacts.")
            obj.save_model(str(path))

        elif artefact_type == "numpy":
            if isinstance(obj, np.ndarray):
                np.savez_compressed(path.with_suffix(""), data=obj)
                path = path.with_suffix(".npz")
            elif isinstance(obj, dict):
                np.savez_compressed(path.with_suffix(""), **obj)
                path = path.with_suffix(".npz")
            else:
                raise TypeError(
                    f"numpy artefact_type expects np.ndarray or dict, got {type(obj)}."
                )

        elif artefact_type == "json":
            path = path.with_suffix(".json")
            with open(path, "w") as f:
                json.dump(obj, f, indent=2, default=str)

        else:
            # Fallback: joblib / sklearn
            joblib.dump(obj, path, compress=3)

        logger.debug("Artefact saved: %s (type=%s)", path, artefact_type)
        return path

    @staticmethod
    def load(
        path: Path,
        artefact_type: str = "joblib",
        model_class: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> Any:
        """Load an artefact from *path*.

        Parameters
        ----------
        path : Path
            File path to load from.
        artefact_type : str
            Must match the type used when saving.
        model_class : type, optional
            For ``"pytorch"`` type: the ``nn.Module`` subclass to instantiate
            before loading the state dict.  If None, returns raw state dict.
        device : torch.device, optional
            Device to map tensors to when loading pytorch artefacts.
        """
        if not path.exists():
            raise FileNotFoundError(f"Artefact not found: {path}")

        if artefact_type == "pytorch":
            if not TORCH_AVAILABLE:
                raise ImportError("torch is required to load pytorch artefacts.")
            import torch

            map_loc = device or torch.device("cpu")
            state = torch.load(path, map_location=map_loc)
            if model_class is not None:
                instance = model_class()
                instance.load_state_dict(state)
                return instance
            return state

        elif artefact_type == "lightgbm":
            if not LGB_AVAILABLE:
                raise ImportError("lightgbm is required to load LightGBM artefacts.")
            return lgb.Booster(model_file=str(path))

        elif artefact_type == "xgboost":
            if not XGB_AVAILABLE:
                raise ImportError("xgboost is required to load XGBoost artefacts.")
            booster = xgb.Booster()
            booster.load_model(str(path))
            return booster

        elif artefact_type == "numpy":
            data = np.load(path)
            if list(data.keys()) == ["data"]:
                return data["data"]
            return dict(data)

        elif artefact_type == "json":
            with open(path) as f:
                return json.load(f)

        else:
            return joblib.load(path)


# ---------------------------------------------------------------------------
# ModelCard generator
# ---------------------------------------------------------------------------


class ModelCardGenerator:
    """Generates a structured JSON model card for a registered model version.

    The card follows the spirit of Google's Model Card Toolkit specification,
    adapted for the AIDSTL project.

    Sections
    --------
    model_details          — identity, version, intended use
    model_parameters       — architecture, hyperparameters
    training_data          — data sources, splits, preprocessing
    evaluation_data        — test-set description
    quantitative_analysis  — metrics table
    considerations         — limitations, ethical considerations
    provenance             — git, timestamps, author
    """

    # Model-type specific descriptions
    MODEL_DESCRIPTIONS: Dict[str, str] = {
        "HydroFormer": (
            "End-to-end multi-modal deep learning model combining a "
            "Swin Transformer spatial encoder (12-channel Sentinel-2 patches) "
            "with a Temporal Fusion Transformer (TFT) for time-series depth "
            "forecasting.  Outputs a depth estimate with calibrated 10th–90th "
            "percentile uncertainty bounds."
        ),
        "EnsembleDepthEstimator": (
            "Level-2 stacking ensemble.  Base learners: HydroFormer (optional), "
            "LightGBM, XGBoost.  Meta-learner: RidgeCV trained on out-of-fold "
            "predictions via 5-fold spatial block cross-validation.  "
            "Conformal prediction intervals provided by MAPIE."
        ),
        "NavigabilityClassifier": (
            "Calibrated LightGBM multiclass classifier (3 classes: Navigable, "
            "Conditional, Non-Navigable).  Probability calibration via isotonic "
            "regression.  Conformal prediction sets via MAPIE RAPS.  "
            "SHAP TreeExplainer for feature attribution."
        ),
        "SwinSpectralEncoder": (
            "Swin-Tiny Transformer backbone (timm) adapted for 12-channel "
            "Sentinel-2 input with a 64-dim projection head.  Used as the "
            "spatial feature extractor inside HydroFormer."
        ),
        "HydroForecastTFT": (
            "Temporal Fusion Transformer with Variable Selection Networks, "
            "LSTM encoder (hidden=128, layers=2), 8-head self-attention, "
            "and a quantile output head (q10, q50, q90) for uncertainty "
            "quantification."
        ),
    }

    INTENDED_USES: Dict[str, str] = {
        "HydroFormer": (
            "Predict water depth (metres) at 5 km river-segment resolution "
            "from monthly Sentinel-2 spectral sequences and static ancillary "
            "features.  Intended for inland waterway navigability assessment "
            "on NW-1 (Ganga) and NW-2 (Brahmaputra)."
        ),
        "EnsembleDepthEstimator": (
            "Operational depth estimation with quantified uncertainty for "
            "waterway managers and port authorities.  Combines deep-learning "
            "and gradient-boosting signals for improved robustness."
        ),
        "NavigabilityClassifier": (
            "Classify river segments into Navigable (depth >= 3.0 m, "
            "width >= 50 m), Conditional (2.0–3.0 m), or Non-Navigable "
            "(< 2.0 m).  Provides calibrated class probabilities and "
            "conformal prediction sets for decision support."
        ),
    }

    LIMITATIONS: Dict[str, List[str]] = {
        "HydroFormer": [
            "Trained on Indian inland waterways; performance on other rivers is unknown.",
            "Cloud cover > 80% in a given month leads to imputed Sentinel-2 values.",
            "Depth estimates below 0.5 m may be unreliable (sensor saturation).",
            "Model does not account for sudden flood events within a monthly window.",
        ],
        "EnsembleDepthEstimator": [
            "Meta-learner trained on OOF predictions; requires re-calibration if "
            "base-learner mix changes.",
            "Conformal coverage guarantee holds marginally, not conditionally.",
        ],
        "NavigabilityClassifier": [
            "Class boundaries (3.0 m, 2.0 m) are regulatory thresholds; "
            "ecological navigability may differ.",
            "Calibration quality degrades on out-of-distribution seasonal regimes.",
        ],
    }

    def generate(
        self,
        record: VersionRecord,
        config: Optional[Dict[str, Any]] = None,
        training_data_description: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a complete model card dict.

        Parameters
        ----------
        record : VersionRecord
            Registry entry for the model version.
        config : dict, optional
            Training hyperparameter dict.
        training_data_description : dict, optional
            Dataset summary statistics to embed in the card.

        Returns
        -------
        Dict[str, Any]
            Full model card as a nested Python dict (JSON-serialisable).
        """
        name = record.model_name
        description = self.MODEL_DESCRIPTIONS.get(
            name,
            f"AIDSTL model '{name}' version {record.version}.",
        )
        intended_use = self.INTENDED_USES.get(
            name,
            "Inland waterway navigability assessment.",
        )
        limitations = self.LIMITATIONS.get(name, [])

        card: Dict[str, Any] = {
            "schema_version": "1.0.0",
            "generated_at": _now_iso(),
            "project": PROJECT_META,
            # ── Model details ──────────────────────────────────────────────
            "model_details": {
                "name": name,
                "version": record.version,
                "run_id": record.run_id,
                "stage": record.stage,
                "type": self._infer_model_type(name),
                "description": description,
                "intended_use": intended_use,
                "out_of_scope_use": (
                    "Global rivers without re-training; "
                    "real-time (sub-daily) navigability alerts; "
                    "navigation safety certification without human review."
                ),
                "license": "Apache-2.0",
                "contact": "AIDSTL Project Team",
            },
            # ── Model parameters ───────────────────────────────────────────
            "model_parameters": {
                "architecture": self._describe_architecture(name, config),
                "hyperparameters": config or {},
                "input_features": self._describe_inputs(name),
                "output": self._describe_outputs(name),
            },
            # ── Training data ──────────────────────────────────────────────
            "training_data": training_data_description
            or {
                "sources": [
                    "Sentinel-2 Level-2A Surface Reflectance (10 m, 5-day revisit)",
                    "Sentinel-1 GRD SAR backscatter (VV, VH)",
                    "ERA5-Land monthly climate aggregates",
                    "SRTM Digital Elevation Model (30 m)",
                    "CWC (Central Water Commission) gauge readings",
                ],
                "temporal_coverage": f"{record.created_at[:4]}: historical (specify year range)",
                "spatial_coverage": "NW-1: Varanasi to Haldia (~1 390 km); NW-2: Dhubri to Sadiya (~891 km)",
                "segment_resolution": "5 km analysis units",
                "preprocessing": [
                    "SCL-based cloud masking for Sentinel-2",
                    "Monthly median compositing",
                    "Z-score normalisation of all features",
                    "Inverse-distance weighting for gauge interpolation",
                    "Median imputation for residual NaN values",
                ],
                "train_val_test_split": "Spatial block CV: 75% train / 15% val / 10% test",
                "class_distribution": {
                    "Navigable (depth>=3.0m, width>=50m)": "~35%",
                    "Conditional (2.0-3.0m)": "~30%",
                    "Non-Navigable (<2.0m)": "~35%",
                },
            },
            # ── Evaluation ─────────────────────────────────────────────────
            "evaluation_data": {
                "description": (
                    "Spatially held-out test segments (last 10% of ordered "
                    "river chainage) not seen during training or meta-learner fitting."
                ),
                "evaluation_approach": "Spatial block hold-out to prevent autocorrelation leakage.",
            },
            # ── Quantitative analysis ──────────────────────────────────────
            "quantitative_analysis": {
                "metrics": record.metrics,
                "performance_targets": {
                    "task_a_r2": "> 0.90",
                    "task_a_rmse_m": "< 1.5 m",
                    "task_b_f1_macro": "> 0.85",
                    "pi_coverage_90": "> 0.88",
                },
                "disaggregated_metrics": {
                    "note": (
                        "Per-waterway and per-season breakdowns should be "
                        "computed at evaluation time and added here."
                    )
                },
            },
            # ── Considerations ─────────────────────────────────────────────
            "considerations": {
                "limitations": limitations,
                "ethical_considerations": [
                    "Depth predictions should supplement, not replace, "
                    "physical survey and professional pilot judgement.",
                    "Model uncertainty bounds must be communicated to "
                    "end-users to avoid over-reliance.",
                    "Fairness auditing across seasons (monsoon vs. dry) "
                    "recommended before operational deployment.",
                ],
                "caveats_and_recommendations": [
                    "Re-train annually when new multi-year CWC gauge data is available.",
                    "Validate against independent bathymetric surveys before "
                    "safety-critical decisions.",
                    "Monitor concept drift via rolling RMSE on new gauge readings.",
                ],
            },
            # ── Provenance ─────────────────────────────────────────────────
            "provenance": {
                "author": record.author,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "git_commit": record.git_commit,
                "tags": record.tags,
                "notes": record.notes,
                "python_version": record.python_version,
                "framework_versions": record.framework_versions,
                "artefact_checksums": {},  # populated by registry after saving
            },
        }
        return card

    @staticmethod
    def _infer_model_type(name: str) -> str:
        """Infer a broad model type label from the model name."""
        lower = name.lower()
        if "tft" in lower or "transformer" in lower or "hydroformer" in lower:
            return "Deep Learning (Transformer)"
        if "swin" in lower:
            return "Deep Learning (Vision Transformer)"
        if "ensemble" in lower:
            return "Stacking Ensemble"
        if "lgb" in lower or "lightgbm" in lower:
            return "Gradient Boosting (LightGBM)"
        if "xgb" in lower or "xgboost" in lower:
            return "Gradient Boosting (XGBoost)"
        if "classifier" in lower:
            return "Classifier"
        return "Machine Learning"

    @staticmethod
    def _describe_architecture(
        name: str, config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build an architecture summary dict from the model name and config."""
        arch: Dict[str, Any] = {}
        if "HydroFormer" in name:
            arch = {
                "backbone": "Swin-Tiny Transformer (12-channel Sentinel-2 input)",
                "temporal_model": "Temporal Fusion Transformer",
                "d_model": (config or {}).get("d_model", 128),
                "n_heads": (config or {}).get("n_heads", 8),
                "lstm_hidden": (config or {}).get("lstm_hidden", 128),
                "lstm_layers": (config or {}).get("lstm_layers", 2),
                "swin_embed_dim": (config or {}).get("swin_embed_dim", 64),
                "quantile_output": "[0.10, 0.50, 0.90]",
            }
        elif "Ensemble" in name:
            arch = {
                "level_0_learners": ["HydroFormer", "LightGBM", "XGBoost"],
                "level_1_meta_learner": "RidgeCV",
                "cv_strategy": "5-fold spatial block CV",
                "conformal_method": "MAPIE (method=plus)",
            }
        elif "Classifier" in name or "Navigator" in name:
            arch = {
                "base_model": "LightGBM (multiclass)",
                "calibration": "CalibratedClassifierCV (isotonic)",
                "conformal": "MAPIE MapieClassifier (RAPS)",
                "n_classes": 3,
                "class_labels": ["Non-Navigable", "Conditional", "Navigable"],
            }
        return arch

    @staticmethod
    def _describe_inputs(name: str) -> Dict[str, Any]:
        """Describe model input feature sets."""
        if "HydroFormer" in name:
            return {
                "x_temporal": "(B, T=12, F_t) — monthly spectral + ancillary sequence",
                "x_static": "(B, F_s) — time-invariant segment features",
                "x_patch": "(B, 12, H, W) — Sentinel-2 image patch (optional)",
                "temporal_features": [
                    "B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12",
                    "MNDWI, NDWI, AWEI, STUMPF, TURBIDITY, NDTI",
                    "water_width_m, sinuosity, mndwi_std_12m",
                    "gauge_water_level_m, gauge_discharge_m3s",
                    "era5_cumulative_rainfall_mm, era5_mean_temperature_c",
                    "sar_vv, sar_vh, sar_vv_vh_ratio",
                ],
                "static_features": [
                    "elevation_m, slope_deg",
                    "distance_from_source_km",
                    "temporal aggregates (mean/std/min/max) of spectral indices",
                ],
            }
        elif "Classifier" in name:
            return {
                "features": [
                    "depth_pred (m) — ensemble depth estimate",
                    "depth_uncertainty (m) — PI width",
                    "water_width_m — water-surface width",
                    "gauge_discharge_m3s — river discharge",
                    "sinuosity — channel sinuosity ratio",
                ]
            }
        return {"features": "See training configuration."}

    @staticmethod
    def _describe_outputs(name: str) -> Dict[str, Any]:
        """Describe model outputs."""
        if "HydroFormer" in name:
            return {
                "depth_pred": "Scalar depth estimate (metres)",
                "lower_ci": "10th-percentile lower bound (metres)",
                "upper_ci": "90th-percentile upper bound (metres)",
            }
        elif "Ensemble" in name:
            return {
                "mean_pred": "Ensemble depth estimate (metres)",
                "lower_ci": "Conformal lower bound (metres)",
                "upper_ci": "Conformal upper bound (metres)",
            }
        elif "Classifier" in name:
            return {
                "class_label": "0=Non-Navigable, 1=Conditional, 2=Navigable",
                "confidence": "Calibrated class probability",
                "conformal_set": "Set of plausible classes at chosen alpha",
            }
        return {}


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Versioned model artefact registry for the AIDSTL project.

    Stores model weights, metrics, configs, and model cards in a structured
    directory tree.  A master JSON index (``registry.json``) tracks all
    registered versions and their metadata.

    Parameters
    ----------
    registry_root : str or Path
        Root directory for the registry.  Created if it does not exist.
    auto_bump : str or None
        If set to ``"major"``, ``"minor"``, or ``"patch"``, automatically
        bumps the version when registering a new version of an existing model.
        Default ``"patch"``.

    Directory layout
    ----------------
    registry_root/
        registry.json
        HydroFormer/
            1.0.0/
                artefacts/
                    model.pt
                model_card.json
                metrics.json
                config.json
                metadata.json
        EnsembleDepthEstimator/
            1.0.0/
                artefacts/
                    lgb_full.txt
                    xgb_full.json
                    meta_learner.pkl
                    mapie_regressor.pkl
                ...
    """

    def __init__(
        self,
        registry_root: Union[str, Path] = "ml/models/registry",
        auto_bump: str = "patch",
    ) -> None:
        self.root = Path(registry_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.auto_bump = auto_bump
        self._index_path = self.root / REGISTRY_INDEX_FILE
        self._card_gen = ModelCardGenerator()
        self._io = ArtefactIO()

        # Load or initialise the registry index
        self._index: Dict[str, List[Dict[str, Any]]] = self._load_index()

        logger.info(
            "ModelRegistry initialised | root=%s | models registered: %d",
            self.root,
            len(self._index),
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load the registry index JSON, or return an empty index."""
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    data = json.load(f)
                logger.debug("Loaded registry index with %d model(s).", len(data))
                return data
            except json.JSONDecodeError as exc:
                logger.error("Registry index corrupt: %s. Starting fresh.", exc)
                return {}
        return {}

    def _save_index(self) -> None:
        """Persist the in-memory index to disk atomically."""
        tmp_path = self._index_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)
        tmp_path.replace(self._index_path)
        logger.debug("Registry index saved (%d model(s)).", len(self._index))

    def _get_records(self, model_name: str) -> List[VersionRecord]:
        """Return all VersionRecords for a model name (active only)."""
        entries = self._index.get(model_name, [])
        return [
            VersionRecord.from_dict(e)
            for e in entries
            if not e.get("is_deleted", False)
        ]

    def _version_dir(self, model_name: str, version: str) -> Path:
        """Return the directory path for a specific model version."""
        return self.root / model_name / version

    # ------------------------------------------------------------------
    # Version helpers
    # ------------------------------------------------------------------

    def _next_version(self, model_name: str) -> str:
        """Auto-compute the next version for *model_name*."""
        records = self._get_records(model_name)
        if not records:
            return "1.0.0"
        # Find the highest existing version
        versions = []
        for rec in records:
            try:
                versions.append(_parse_version(rec.version))
            except ValueError:
                pass
        if not versions:
            return "1.0.0"
        latest = max(versions)
        current_str = ".".join(str(x) for x in latest)
        return _bump_version(current_str, self.auto_bump or "patch")

    def list_versions(
        self,
        model_name: str,
        include_deleted: bool = False,
    ) -> List[VersionRecord]:
        """List all registered versions for a model.

        Parameters
        ----------
        model_name : str
            Model family name.
        include_deleted : bool
            If True, include soft-deleted versions.

        Returns
        -------
        List[VersionRecord]  sorted by version (ascending).
        """
        entries = self._index.get(model_name, [])
        records = [VersionRecord.from_dict(e) for e in entries]
        if not include_deleted:
            records = [r for r in records if not r.is_deleted]
        records.sort(key=lambda r: _parse_version(r.version))
        return records

    def list_models(self) -> List[str]:
        """Return the list of all model family names in the registry."""
        return sorted(self._index.keys())

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model_name: str,
        model_object: Any,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        artefact_type: str = "joblib",
        version: Optional[str] = None,
        author: str = "unknown",
        description: str = "",
        tags: Optional[List[str]] = None,
        notes: str = "",
        extra_artefacts: Optional[Dict[str, Tuple[Any, str]]] = None,
        training_data_description: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a new model version.

        Parameters
        ----------
        model_name : str
            Logical model name (e.g. ``"HydroFormer"``).
        model_object : Any
            The model object to serialise (nn.Module, lgb.Booster, …).
        metrics : dict, optional
            Evaluation metric dict, e.g. ``{"r2": 0.92, "rmse": 1.1}``.
        config : dict, optional
            Training hyperparameter dict (TrainingConfig.to_dict()).
        artefact_type : str
            Serialisation format for *model_object*.
        version : str, optional
            Explicit version string.  Auto-incremented if None.
        author : str
            Creator identifier.
        description : str
            Free-text description.
        tags : list of str, optional
            Searchable tags.
        notes : str
            Change-log notes.
        extra_artefacts : dict, optional
            Additional artefacts to save.  Format:
            ``{role: (object, artefact_type)}``, e.g.
            ``{"norm_stats": (stats_dict, "numpy")}``.
        training_data_description : dict, optional
            Dataset summary to embed in model card.

        Returns
        -------
        str
            The version string that was registered.
        """
        # ── Determine version ────────────────────────────────────────────
        if version is None:
            version = self._next_version(model_name)
        else:
            # Validate format
            _parse_version(version)

        # Check for collision
        existing_versions = [r.version for r in self._get_records(model_name)]
        if version in existing_versions:
            raise ValueError(
                f"Version '{version}' already exists for model '{model_name}'. "
                "Use a different version string or delete the existing entry first."
            )

        ver_dir = self._version_dir(model_name, version)
        art_dir = ver_dir / ARTEFACTS_DIR
        art_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Registering %s v%s | stage=experimental | artefact_type=%s",
            model_name,
            version,
            artefact_type,
        )

        # ── Save primary artefact ────────────────────────────────────────
        primary_path = art_dir / "model"
        primary_path = self._io.save(model_object, primary_path, artefact_type)

        artefact_paths: Dict[str, str] = {
            "model": str(primary_path.relative_to(ver_dir))
        }

        # ── Save extra artefacts ─────────────────────────────────────────
        if extra_artefacts:
            for role, (obj, a_type) in extra_artefacts.items():
                extra_path = art_dir / role
                saved = self._io.save(obj, extra_path, a_type)
                artefact_paths[role] = str(saved.relative_to(ver_dir))
                logger.debug("Extra artefact '%s' saved: %s", role, saved)

        # ── Compute file checksums ───────────────────────────────────────
        checksums: Dict[str, str] = {}
        for role, rel_path in artefact_paths.items():
            abs_path = ver_dir / rel_path
            if abs_path.exists():
                checksums[role] = _sha256_file(abs_path)

        # ── Build VersionRecord ──────────────────────────────────────────
        record = VersionRecord(
            model_name=model_name,
            version=version,
            author=author,
            description=description,
            tags=tags or [],
            metrics=metrics or {},
            artefact_paths=artefact_paths,
            git_commit=_get_git_commit(),
            framework_versions=_collect_framework_versions(),
            notes=notes,
        )

        # ── Save config.json ─────────────────────────────────────────────
        if config:
            config_path = ver_dir / CONFIG_FILE
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

        # ── Save metrics.json ────────────────────────────────────────────
        metrics_path = ver_dir / METRICS_FILE
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "version": version,
                    "recorded_at": _now_iso(),
                    "metrics": metrics or {},
                },
                f,
                indent=2,
            )

        # ── Save metadata.json ───────────────────────────────────────────
        meta_path = ver_dir / METADATA_FILE
        meta = record.to_dict()
        meta["artefact_checksums"] = checksums
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # ── Generate and save model card ─────────────────────────────────
        card = self._card_gen.generate(record, config, training_data_description)
        card["provenance"]["artefact_checksums"] = checksums
        card_path = ver_dir / MODEL_CARD_FILE
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2, default=str)

        # ── Update registry index ────────────────────────────────────────
        if model_name not in self._index:
            self._index[model_name] = []
        self._index[model_name].append(meta)
        self._save_index()

        logger.info(
            "Registered %s v%s (run_id=%s) → %s",
            model_name,
            version,
            record.run_id,
            ver_dir,
        )
        return version

    # ------------------------------------------------------------------
    # Loading artefacts
    # ------------------------------------------------------------------

    def load(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
        artefact_role: str = "model",
        artefact_type: str = "joblib",
        model_class: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> Any:
        """Load a model artefact from the registry.

        Parameters
        ----------
        model_name : str
        version : str, optional
            Specific version to load.  If None, uses *stage* or latest.
        stage : str, optional
            Load the version currently in this deployment stage.
            Ignored if *version* is given.
        artefact_role : str
            Which artefact to load (default ``"model"``).
        artefact_type : str
            Deserialisation format.
        model_class : type, optional
            For pytorch artefacts: the nn.Module class to instantiate.
        device : torch.device, optional
            For pytorch artefacts: target device.

        Returns
        -------
        Any
            Deserialised artefact object.
        """
        record = self._resolve_version(model_name, version, stage)
        ver_dir = self._version_dir(model_name, record.version)
        rel_path = record.artefact_paths.get(artefact_role)
        if rel_path is None:
            raise KeyError(
                f"Artefact role '{artefact_role}' not found for "
                f"{model_name} v{record.version}.  "
                f"Available: {list(record.artefact_paths.keys())}"
            )
        abs_path = ver_dir / rel_path
        logger.info(
            "Loading %s v%s artefact '%s' from %s",
            model_name,
            record.version,
            artefact_role,
            abs_path,
        )
        return self._io.load(
            abs_path,
            artefact_type=artefact_type,
            model_class=model_class,
            device=device,
        )

    def load_model_card(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load and return the JSON model card for a version."""
        record = self._resolve_version(model_name, version, stage)
        card_path = self._version_dir(model_name, record.version) / MODEL_CARD_FILE
        if not card_path.exists():
            raise FileNotFoundError(f"Model card not found: {card_path}")
        with open(card_path) as f:
            return json.load(f)

    def load_config(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load training config for a version."""
        record = self._resolve_version(model_name, version, stage)
        cfg_path = self._version_dir(model_name, record.version) / CONFIG_FILE
        if not cfg_path.exists():
            return {}
        with open(cfg_path) as f:
            return json.load(f)

    def load_metrics(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load the metrics snapshot for a version."""
        record = self._resolve_version(model_name, version, stage)
        met_path = self._version_dir(model_name, record.version) / METRICS_FILE
        if not met_path.exists():
            return {}
        with open(met_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def promote(
        self,
        model_name: str,
        version: str,
        stage: str,
        notes: str = "",
    ) -> None:
        """Promote a version to a deployment stage.

        Transitions a version to *stage*.  If another version is currently
        in the target stage, it is automatically moved to ``"archived"``.

        Parameters
        ----------
        model_name : str
        version : str
        stage : str
            Target stage (experimental / staging / production / archived / deprecated).
        notes : str
            Optional note to append to the record.
        """
        if stage not in STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {STAGES}.")

        # Find the target record in the index
        target_idx = self._find_record_index(model_name, version)
        if target_idx is None:
            raise KeyError(f"Version '{version}' not found for model '{model_name}'.")

        # Archive any existing version in the target stage
        if stage in ("production", "staging"):
            for i, entry in enumerate(self._index.get(model_name, [])):
                if (
                    entry.get("stage") == stage
                    and entry.get("version") != version
                    and not entry.get("is_deleted", False)
                ):
                    self._index[model_name][i]["stage"] = "archived"
                    self._index[model_name][i]["updated_at"] = _now_iso()
                    logger.info(
                        "Archived %s v%s (superseded by v%s → %s)",
                        model_name,
                        entry["version"],
                        version,
                        stage,
                    )

        # Promote the target version
        old_stage = self._index[model_name][target_idx].get("stage", "experimental")
        self._index[model_name][target_idx]["stage"] = stage
        self._index[model_name][target_idx]["updated_at"] = _now_iso()
        if notes:
            existing = self._index[model_name][target_idx].get("notes", "")
            self._index[model_name][target_idx]["notes"] = (
                f"{existing}\n[{_now_iso()}] Stage: {old_stage} → {stage}. {notes}"
            ).strip()

        # Update on-disk metadata
        self._update_metadata_on_disk(model_name, version)
        self._save_index()

        logger.info(
            "Promoted %s v%s: %s → %s",
            model_name,
            version,
            old_stage,
            stage,
        )

    def deprecate(self, model_name: str, version: str, reason: str = "") -> None:
        """Mark a version as deprecated."""
        self.promote(model_name, version, "deprecated", notes=f"Deprecated: {reason}")

    def delete(
        self,
        model_name: str,
        version: str,
        hard_delete: bool = False,
    ) -> None:
        """Soft-delete (or hard-delete) a version.

        Parameters
        ----------
        hard_delete : bool
            If True, permanently removes files from disk.  Default False
            (soft-delete: sets ``is_deleted=True`` in index).
        """
        target_idx = self._find_record_index(model_name, version)
        if target_idx is None:
            raise KeyError(f"Version '{version}' not found for model '{model_name}'.")

        if hard_delete:
            ver_dir = self._version_dir(model_name, version)
            if ver_dir.exists():
                shutil.rmtree(ver_dir)
                logger.warning(
                    "Hard-deleted %s v%s from disk: %s", model_name, version, ver_dir
                )
            self._index[model_name].pop(target_idx)
        else:
            self._index[model_name][target_idx]["is_deleted"] = True
            self._index[model_name][target_idx]["updated_at"] = _now_iso()
            logger.info("Soft-deleted %s v%s.", model_name, version)

        self._save_index()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def best_version(
        self,
        model_name: str,
        metric: str,
        higher_is_better: bool = True,
        stage: Optional[str] = None,
    ) -> Optional[VersionRecord]:
        """Return the version record with the best value of *metric*.

        Parameters
        ----------
        model_name : str
        metric : str
            Metric key, e.g. ``"r2"`` or ``"rmse"``.
        higher_is_better : bool
            True for metrics like R²; False for RMSE, MAE.
        stage : str, optional
            If given, restricts search to versions in this stage.

        Returns
        -------
        VersionRecord or None
        """
        records = self._get_records(model_name)
        if stage:
            records = [r for r in records if r.stage == stage]
        records_with_metric = [r for r in records if metric in r.metrics]
        if not records_with_metric:
            logger.warning("No versions of '%s' have metric '%s'.", model_name, metric)
            return None
        return (
            max(records_with_metric, key=lambda r: r.metrics[metric])
            if higher_is_better
            else min(records_with_metric, key=lambda r: r.metrics[metric])
        )

    def production_version(self, model_name: str) -> Optional[VersionRecord]:
        """Return the current production version, if any."""
        records = [r for r in self._get_records(model_name) if r.stage == "production"]
        if not records:
            return None
        records.sort(key=lambda r: _parse_version(r.version))
        return records[-1]

    def latest_version(
        self,
        model_name: str,
        stage: Optional[str] = None,
    ) -> Optional[VersionRecord]:
        """Return the highest-version record, optionally filtered by stage."""
        records = self._get_records(model_name)
        if stage:
            records = [r for r in records if r.stage == stage]
        if not records:
            return None
        records.sort(key=lambda r: _parse_version(r.version))
        return records[-1]

    def search(
        self,
        model_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        stage: Optional[str] = None,
        min_metric: Optional[Dict[str, float]] = None,
        max_metric: Optional[Dict[str, float]] = None,
    ) -> List[VersionRecord]:
        """Search the registry with filter criteria.

        Parameters
        ----------
        model_name : str, optional
            Filter to a specific model family.
        tags : list of str, optional
            All specified tags must be present on the record.
        stage : str, optional
            Filter by deployment stage.
        min_metric : dict, optional
            Lower bounds on metrics, e.g. ``{"r2": 0.90}``.
        max_metric : dict, optional
            Upper bounds on metrics, e.g. ``{"rmse": 1.5}``.

        Returns
        -------
        List[VersionRecord]  sorted by (model_name, version).
        """
        model_names = [model_name] if model_name else list(self._index.keys())
        results: List[VersionRecord] = []

        for mname in model_names:
            for record in self._get_records(mname):
                if stage and record.stage != stage:
                    continue
                if tags and not all(t in record.tags for t in tags):
                    continue
                if min_metric:
                    if not all(
                        record.metrics.get(k, float("-inf")) >= v
                        for k, v in min_metric.items()
                    ):
                        continue
                if max_metric:
                    if not all(
                        record.metrics.get(k, float("inf")) <= v
                        for k, v in max_metric.items()
                    ):
                        continue
                results.append(record)

        results.sort(key=lambda r: (r.model_name, _parse_version(r.version)))
        return results

    # ------------------------------------------------------------------
    # Metrics update
    # ------------------------------------------------------------------

    def update_metrics(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        overwrite: bool = False,
    ) -> None:
        """Update / append metrics for an existing version.

        Parameters
        ----------
        overwrite : bool
            If True, replace all metrics.  If False, merge (new keys win).
        """
        target_idx = self._find_record_index(model_name, version)
        if target_idx is None:
            raise KeyError(f"Version '{version}' not found for model '{model_name}'.")

        existing = self._index[model_name][target_idx].get("metrics", {})
        updated = metrics if overwrite else {**existing, **metrics}
        self._index[model_name][target_idx]["metrics"] = updated
        self._index[model_name][target_idx]["updated_at"] = _now_iso()

        # Update on-disk metrics.json
        ver_dir = self._version_dir(model_name, version)
        met_path = ver_dir / METRICS_FILE
        with open(met_path, "w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "version": version,
                    "recorded_at": _now_iso(),
                    "metrics": updated,
                },
                f,
                indent=2,
            )

        self._save_index()
        logger.info("Updated metrics for %s v%s: %s", model_name, version, updated)

    # ------------------------------------------------------------------
    # Summary and reporting
    # ------------------------------------------------------------------

    def summary(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Return a summary DataFrame of registered versions.

        Parameters
        ----------
        model_name : str, optional
            Restrict to a specific model family.

        Returns
        -------
        pd.DataFrame with columns: model_name, version, stage, created_at,
            author, tags, and one column per distinct metric.
        """
        import pandas as pd

        model_names = [model_name] if model_name else list(self._index.keys())
        rows: List[Dict[str, Any]] = []

        for mname in model_names:
            for record in self._get_records(mname):
                row: Dict[str, Any] = {
                    "model_name": record.model_name,
                    "version": record.version,
                    "stage": record.stage,
                    "run_id": record.run_id[:8],  # abbreviated
                    "created_at": record.created_at[:19],
                    "author": record.author,
                    "tags": ", ".join(record.tags),
                    "description": record.description[:60] + "…"
                    if len(record.description) > 60
                    else record.description,
                }
                row.update(record.metrics)
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values(["model_name", "version"])
        return df.reset_index(drop=True)

    def print_summary(self, model_name: Optional[str] = None) -> None:
        """Print a human-readable registry summary to stdout."""
        df = self.summary(model_name)
        if df.empty:
            print("No models registered.")
            return

        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title="AIDSTL Model Registry",
                show_header=True,
                header_style="bold cyan",
            )
            for col in df.columns:
                table.add_column(str(col), overflow="fold")
            for _, row in df.iterrows():
                table.add_row(*[str(v) for v in row.values])
            console.print(table)
        except ImportError:
            # Fallback: plain pandas repr
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(df.to_string(index=False))

    def export_registry_json(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Export the full registry index as a formatted JSON file.

        Parameters
        ----------
        path : str or Path, optional
            Output path.  Defaults to ``registry_root/registry_export.json``.

        Returns
        -------
        Path
            Path to the written file.
        """
        export_path = Path(path) if path else self.root / "registry_export.json"
        with open(export_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)
        logger.info("Registry exported to %s", export_path)
        return export_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_record_index(
        self,
        model_name: str,
        version: str,
    ) -> Optional[int]:
        """Return the list-index of a version entry in the index, or None."""
        for i, entry in enumerate(self._index.get(model_name, [])):
            if entry.get("version") == version and not entry.get("is_deleted", False):
                return i
        return None

    def _resolve_version(
        self,
        model_name: str,
        version: Optional[str],
        stage: Optional[str],
    ) -> VersionRecord:
        """Resolve a version identifier to a VersionRecord.

        Priority
        --------
        1. Explicit *version* string.
        2. Version currently in *stage*.
        3. Latest version (highest version number).
        """
        if version is not None:
            idx = self._find_record_index(model_name, version)
            if idx is None:
                raise KeyError(
                    f"Version '{version}' not found for model '{model_name}'."
                )
            return VersionRecord.from_dict(self._index[model_name][idx])

        if stage is not None:
            records = [r for r in self._get_records(model_name) if r.stage == stage]
            if not records:
                raise KeyError(
                    f"No version of '{model_name}' is currently in stage '{stage}'."
                )
            return max(records, key=lambda r: _parse_version(r.version))

        latest = self.latest_version(model_name)
        if latest is None:
            raise KeyError(f"No versions registered for model '{model_name}'.")
        return latest

    def _update_metadata_on_disk(self, model_name: str, version: str) -> None:
        """Rewrite metadata.json for a version from the in-memory index."""
        idx = self._find_record_index(model_name, version)
        if idx is None:
            return
        ver_dir = self._version_dir(model_name, version)
        meta_path = ver_dir / METADATA_FILE
        if meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(self._index[model_name][idx], f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def get_default_registry() -> ModelRegistry:
    """Return a ModelRegistry rooted at the default project path.

    Looks for the ``AIDSTL_REGISTRY_ROOT`` environment variable.
    Falls back to ``ml/models/registry`` relative to cwd.
    """
    root = os.environ.get("AIDSTL_REGISTRY_ROOT", "ml/models/registry")
    return ModelRegistry(registry_root=root)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    import numpy as np

    # We need pandas for summary()
    try:
        import pandas as pd
    except ImportError:
        pd = None

    logging.basicConfig(level=logging.DEBUG)
    print("\n── ModelRegistry smoke test ──\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(registry_root=tmpdir, auto_bump="patch")

        # ── Register a mock model ─────────────────────────────────────────
        mock_model = {"weights": np.random.randn(10, 10), "bias": np.zeros(10)}
        version_1 = registry.register(
            model_name="HydroFormer",
            model_object=mock_model,
            metrics={"r2": 0.887, "rmse": 1.43, "mae": 1.01, "coverage_90": 0.891},
            config={
                "d_model": 128,
                "n_heads": 8,
                "lstm_hidden": 128,
                "max_epochs": 100,
                "learning_rate": 3e-4,
            },
            artefact_type="joblib",
            author="test_user",
            description="Baseline HydroFormer on NW-1 data.",
            tags=["nw1", "sentinel2", "baseline"],
            notes="Initial training run.",
        )
        print(f"  Registered HydroFormer v{version_1}")

        # ── Register a second version ─────────────────────────────────────
        mock_model_v2 = {"weights": np.random.randn(10, 10) * 0.9, "bias": np.zeros(10)}
        version_2 = registry.register(
            model_name="HydroFormer",
            model_object=mock_model_v2,
            metrics={"r2": 0.921, "rmse": 1.09, "mae": 0.88, "coverage_90": 0.906},
            config={
                "d_model": 128,
                "n_heads": 8,
                "lstm_hidden": 128,
                "max_epochs": 150,
                "learning_rate": 1e-4,
            },
            artefact_type="joblib",
            author="test_user",
            description="Improved HydroFormer — longer training + lower LR.",
            tags=["nw1", "sentinel2", "improved"],
        )
        print(f"  Registered HydroFormer v{version_2}")

        # ── Register NavigabilityClassifier ──────────────────────────────
        mock_clf = {"classes": [0, 1, 2], "threshold": 0.5}
        clf_version = registry.register(
            model_name="NavigabilityClassifier",
            model_object=mock_clf,
            metrics={
                "accuracy": 0.883,
                "f1_macro": 0.876,
                "precision_macro": 0.891,
                "recall_macro": 0.862,
            },
            artefact_type="joblib",
            author="test_user",
            description="Calibrated LightGBM navigability classifier.",
            tags=["classifier", "lgbm", "calibrated"],
        )
        print(f"  Registered NavigabilityClassifier v{clf_version}")

        # ── Promote v2 to production ──────────────────────────────────────
        registry.promote(
            "HydroFormer",
            version_2,
            "production",
            notes="Meets R2 > 0.90 target.",
        )
        print(f"  Promoted HydroFormer v{version_2} → production")

        # ── Query best version ────────────────────────────────────────────
        best = registry.best_version("HydroFormer", "r2", higher_is_better=True)
        print(
            f"  Best HydroFormer by R2: v{best.version} (R2={best.metrics['r2']:.4f})"
        )
        assert best.version == version_2, "Expected v2 to be best by R2."

        # ── Load artefact ─────────────────────────────────────────────────
        loaded = registry.load(
            "HydroFormer",
            version=version_2,
            artefact_type="joblib",
        )
        assert "weights" in loaded, "Loaded artefact missing 'weights' key."
        print(f"  Loaded HydroFormer v{version_2} artefact successfully.")

        # ── Model card ────────────────────────────────────────────────────
        card = registry.load_model_card("HydroFormer", version=version_2)
        print(f"  Model card sections: {list(card.keys())}")
        assert "quantitative_analysis" in card

        # ── Update metrics ────────────────────────────────────────────────
        registry.update_metrics(
            "HydroFormer",
            version_2,
            {"nw2_r2": 0.893, "nw2_rmse": 1.28},
        )
        updated_rec = registry.latest_version("HydroFormer")
        assert "nw2_r2" in updated_rec.metrics
        print(
            f"  Updated metrics for v{version_2}: nw2_r2={updated_rec.metrics['nw2_r2']}"
        )

        # ── Search ────────────────────────────────────────────────────────
        results = registry.search(
            tags=["sentinel2"],
            min_metric={"r2": 0.88},
        )
        print(
            f"  Search (sentinel2, r2>=0.88): {[r.version for r in results]} versions"
        )

        # ── Summary ───────────────────────────────────────────────────────
        if pd is not None:
            df = registry.summary()
            print(f"\n  Registry summary ({len(df)} versions):")
            print(
                df[["model_name", "version", "stage", "r2", "rmse"]].to_string(
                    index=False
                )
            )

        # ── Soft-delete v1 ────────────────────────────────────────────────
        registry.delete("HydroFormer", version_1, hard_delete=False)
        remaining = registry.list_versions("HydroFormer")
        assert all(r.version != version_1 for r in remaining)
        print(
            f"\n  Soft-deleted v{version_1}. Remaining: {[r.version for r in remaining]}"
        )

        # ── Export ────────────────────────────────────────────────────────
        export_path = registry.export_registry_json()
        print(f"  Registry exported to: {export_path}")

    print("\nAll smoke tests passed ✓")
    sys.exit(0)
