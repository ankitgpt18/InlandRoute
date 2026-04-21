"""
ensemble.py
===========
Stacking ensemble for inland-waterway depth estimation and navigability
classification.

EnsembleDepthEstimator
    Level-0 base learners:
        • HydroFormer  (Swin-T spatial encoder + TFT temporal encoder)
        • LightGBM     (tabular, engineered features)
        • XGBoost      (tabular, engineered features)
    Level-1 meta-learner:
        • RidgeCV      (trained on out-of-fold level-0 predictions)
    Cross-validation:  k=5 spatial block CV to prevent leakage.
    Uncertainty:       Conformal prediction intervals on the stacked output.
    Explainability:    SHAP TreeExplainer on LightGBM/XGBoost.

NavigabilityClassifier
    Base model: LightGBM classifier
    Calibration: CalibratedClassifierCV (isotonic regression)
    Conformal intervals: MAPIE (MapieClassifier)
    Explainability: SHAP TreeExplainer
    Output: class label, confidence, (lower, upper) conformal bound
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
import xgboost as xgb
from mapie.classification import CrossConformalClassifier
from mapie.regression.regression import CrossConformalRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Navigability class constants
# ---------------------------------------------------------------------------

NAV_CLASS_NAMES: List[str] = ["Non-Navigable", "Conditional", "Navigable"]
NAV_CLASS_MAP: Dict[int, str] = {i: n for i, n in enumerate(NAV_CLASS_NAMES)}

DEPTH_NAVIGABLE: float = 3.0
DEPTH_CONDITIONAL: float = 2.0


# ---------------------------------------------------------------------------
# Spatial block cross-validation
# ---------------------------------------------------------------------------


def spatial_block_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    segment_ids: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate spatial-block cross-validation splits.

    Instead of shuffling randomly, divides samples into *n_splits* contiguous
    blocks along the river (ordered by segment_id if provided, else by index).
    Each fold uses one block as the validation set, preserving spatial
    autocorrelation structure.

    Parameters
    ----------
    n_samples:
        Total number of samples.
    n_splits:
        Number of CV folds.
    segment_ids:
        Optional array of segment identifiers.  If provided, samples are
        sorted by segment_id before blocking.

    Returns
    -------
    List of (train_idx, val_idx) tuples.
    """
    if segment_ids is not None:
        order = np.argsort(segment_ids)
    else:
        order = np.arange(n_samples)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    block_size = n_samples // n_splits

    for fold in range(n_splits):
        val_start = fold * block_size
        val_end = val_start + block_size if fold < n_splits - 1 else n_samples
        val_idx = order[val_start:val_end]
        train_idx = np.concatenate([order[:val_start], order[val_end:]])
        splits.append((train_idx, val_idx))

    logger.info(
        "Spatial block CV: %d folds, block_size≈%d, n=%d",
        n_splits,
        block_size,
        n_samples,
    )
    return splits


# ---------------------------------------------------------------------------
# LightGBM / XGBoost helpers
# ---------------------------------------------------------------------------


def _default_lgb_params() -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }


def _default_xgb_params() -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "alpha": 0.1,
        "lambda": 0.1,
        "tree_method": "hist",
        "seed": 42,
        "nthread": -1,
    }


def _default_lgb_clf_params() -> Dict[str, Any]:
    return {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# HydroFormer inference wrapper for ensemble integration
# ---------------------------------------------------------------------------


class HydroFormerInferenceWrapper:
    """Thin wrapper around a trained HydroFormer for sklearn-style predict().

    Used as the neural-network base learner inside EnsembleDepthEstimator.
    """

    def __init__(
        self,
        model: Any,  # HydroFormer instance
        device: Optional[torch.device] = None,
        batch_size: int = 64,
        n_static_features: int = 16,
        sequence_length: int = 12,
    ) -> None:
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.n_static_features = n_static_features
        self.sequence_length = sequence_length
        self.model.to(self.device)

    def predict(
        self,
        X: np.ndarray,
        X_static: Optional[np.ndarray] = None,
        X_temporal: Optional[np.ndarray] = None,
        X_patches: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference and return (depth_pred, lower_ci, upper_ci).

        Parameters
        ----------
        X:
            Flat feature matrix (N, F) — used to reconstruct temporal /
            static arrays if explicit splits not given.
        X_static:
            Pre-split static features (N, F_s).
        X_temporal:
            Pre-split temporal features (N, T, F_t).
        X_patches:
            Image patches (N, 12, H, W) or None.

        Returns
        -------
        depth_pred, lower_ci, upper_ci — each shape (N,)
        """
        self.model.eval()
        N = X.shape[0]

        # Fall back to reshaping X if explicit arrays not provided
        if X_temporal is None:
            # Assume last n_static_features cols are static
            fs = self.n_static_features
            ft_total = X.shape[1] - fs
            ft = ft_total // self.sequence_length
            X_temporal = X[:, :ft_total].reshape(N, self.sequence_length, ft)
            X_static = X[:, ft_total:]

        ds = TensorDataset(
            torch.from_numpy(X_static.astype(np.float32)),
            torch.from_numpy(X_temporal.astype(np.float32)),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        preds, lowers, uppers = [], [], []
        patch_idx = 0

        with torch.no_grad():
            for batch in loader:
                xs, xt = [b.to(self.device) for b in batch]
                bs = xs.size(0)

                xp = None
                if X_patches is not None:
                    xp = torch.from_numpy(
                        X_patches[patch_idx : patch_idx + bs].astype(np.float32)
                    ).to(self.device)
                    patch_idx += bs

                depth, lower, upper = self.model(xs, xt, xp)
                preds.append(depth.cpu().numpy())
                lowers.append(lower.cpu().numpy())
                uppers.append(upper.cpu().numpy())

        return (
            np.concatenate(preds),
            np.concatenate(lowers),
            np.concatenate(uppers),
        )


# ---------------------------------------------------------------------------
# EnsembleDepthEstimator
# ---------------------------------------------------------------------------


@dataclass
class EnsembleConfig:
    """Configuration for the stacking ensemble.

    Attributes
    ----------
    n_splits:
        Number of spatial CV folds for OOF training.
    lgb_params:
        LightGBM hyperparameter dict.
    xgb_params:
        XGBoost hyperparameter dict.
    lgb_n_estimators:
        Number of boosting rounds for LightGBM.
    xgb_n_estimators:
        Number of boosting rounds for XGBoost.
    ridge_alphas:
        Alpha grid for RidgeCV meta-learner.
    use_hydroformer:
        Whether to include HydroFormer as a base learner.
    hydroformer_batch_size:
        Batch size for HydroFormer inference.
    conformal_alpha:
        Miscoverage level for conformal intervals (default 0.1 → 90% CI).
    random_seed:
        Random state for reproducibility.
    feature_names:
        Optional list of feature column names for SHAP plots.
    n_static_features:
        Number of static features (used for inferring shapes from flat input).
    """

    n_splits: int = 5
    lgb_params: Dict[str, Any] = field(default_factory=_default_lgb_params)
    xgb_params: Dict[str, Any] = field(default_factory=_default_xgb_params)
    lgb_n_estimators: int = 1000
    xgb_n_estimators: int = 1000
    ridge_alphas: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0]
    )
    use_hydroformer: bool = False  # set True when a trained HydroFormer is available
    hydroformer_batch_size: int = 64
    conformal_alpha: float = 0.1
    random_seed: int = 42
    feature_names: Optional[List[str]] = None
    n_static_features: int = 16


class EnsembleDepthEstimator:
    """Level-2 stacking ensemble for river-depth estimation.

    Base learners (Level-0):
        • HydroFormer (optional — requires a pre-trained model)
        • LightGBM
        • XGBoost
    Meta-learner (Level-1):
        • RidgeCV (trained on out-of-fold predictions)

    The meta-learner input is the concatenation of all base-learner OOF
    predictions, optionally augmented with the original features (passthrough).

    Parameters
    ----------
    config:
        :class:`EnsembleConfig` instance.
    hydroformer_model:
        Optional pre-trained HydroFormer instance.  Required if
        ``config.use_hydroformer = True``.
    device:
        Torch device for HydroFormer inference.
    """

    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        hydroformer_model: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or EnsembleConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Base learners ────────────────────────────────────────────────
        self.lgb_models: List[lgb.Booster] = []
        self.xgb_models: List[xgb.Booster] = []
        self.hf_wrapper: Optional[HydroFormerInferenceWrapper] = None

        if self.config.use_hydroformer and hydroformer_model is not None:
            self.hf_wrapper = HydroFormerInferenceWrapper(
                model=hydroformer_model,
                device=self.device,
                batch_size=self.config.hydroformer_batch_size,
                n_static_features=self.config.n_static_features,
            )

        # ── Meta-learner ─────────────────────────────────────────────────
        self.meta_learner = RidgeCV(alphas=self.config.ridge_alphas)

        # ── Conformal wrapper ────────────────────────────────────────────
        self.mapie_regressor: Optional[CrossConformalRegressor] = None

        # ── State flags ──────────────────────────────────────────────────
        self._is_fitted: bool = False
        self._n_base_learners: int = 0
        self._feature_names: Optional[List[str]] = self.config.feature_names

        # ── SHAP explainers ──────────────────────────────────────────────
        self.lgb_shap_explainer: Optional[shap.TreeExplainer] = None
        self.xgb_shap_explainer: Optional[shap.TreeExplainer] = None

        logger.info(
            "EnsembleDepthEstimator | use_hydroformer=%s, n_splits=%d",
            self.config.use_hydroformer,
            self.config.n_splits,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lgb_train_fold(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[lgb.Booster, np.ndarray]:
        """Train one LightGBM fold and return (model, val_predictions)."""
        dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
        params = dict(self.config.lgb_params)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=self.config.lgb_n_estimators,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        return model, val_preds

    def _xgb_train_fold(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[xgb.Booster, np.ndarray]:
        """Train one XGBoost fold and return (model, val_predictions)."""
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=self._feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self._feature_names)
        params = dict(self.config.xgb_params)
        evals_result: Dict = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.xgb_n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=False,
        )
        val_preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        return model, val_preds

    def _hf_predict_fold(
        self,
        X: np.ndarray,
        X_patches: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run HydroFormer inference and return depth predictions."""
        if self.hf_wrapper is None:
            raise RuntimeError("HydroFormer wrapper not initialised.")
        preds, _, _ = self.hf_wrapper.predict(X, X_patches=X_patches)
        return preds

    def _build_oof_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_patches: Optional[np.ndarray],
        splits: List[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Build the out-of-fold prediction matrix for meta-learner training.

        Returns
        -------
        oof_matrix : (N, n_base_learners)
        """
        N = len(y)
        n_learners = 2 + (1 if self.config.use_hydroformer else 0)
        self._n_base_learners = n_learners
        oof = np.zeros((N, n_learners), dtype=np.float32)

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            logger.info("OOF fold %d/%d …", fold_idx + 1, len(splits))
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            col = 0

            # LightGBM
            lgb_model, lgb_preds = self._lgb_train_fold(X_tr, y_tr, X_val, y_val)
            self.lgb_models.append(lgb_model)
            oof[val_idx, col] = lgb_preds
            col += 1

            # XGBoost
            xgb_model, xgb_preds = self._xgb_train_fold(X_tr, y_tr, X_val, y_val)
            self.xgb_models.append(xgb_model)
            oof[val_idx, col] = xgb_preds
            col += 1

            # HydroFormer (optional)
            if self.config.use_hydroformer and self.hf_wrapper is not None:
                patches_val = X_patches[val_idx] if X_patches is not None else None
                hf_preds = self._hf_predict_fold(X_val, patches_val)
                oof[val_idx, col] = hf_preds

        return oof

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_patches: Optional[np.ndarray] = None,
        segment_ids: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "EnsembleDepthEstimator":
        """Fit the full stacking ensemble.

        Parameters
        ----------
        X:
            Feature matrix (N, F).
        y:
            Depth targets in metres (N,).
        X_patches:
            Optional satellite image patches (N, 12, H, W).
        segment_ids:
            Optional segment identifiers for spatial block CV sorting.
        feature_names:
            Column names for SHAP plots.

        Returns
        -------
        self
        """
        if feature_names is not None:
            self._feature_names = feature_names

        splits = spatial_block_cv_splits(
            n_samples=len(y),
            n_splits=self.config.n_splits,
            segment_ids=segment_ids,
        )

        # ── Build OOF matrix ─────────────────────────────────────────────
        logger.info("Building out-of-fold predictions …")
        oof_matrix = self._build_oof_matrix(X, y, X_patches, splits)

        # ── Train meta-learner on OOF predictions ────────────────────────
        logger.info("Training Ridge meta-learner …")
        self.meta_learner.fit(oof_matrix, y)
        logger.info("Ridge best alpha: %.4f", self.meta_learner.alpha_)

        # ── Retrain base learners on full data ───────────────────────────
        logger.info("Retraining base learners on full dataset …")
        self.lgb_full_model = self._train_lgb_full(X, y)
        self.xgb_full_model = self._train_xgb_full(X, y)

        # ── Build SHAP explainers ────────────────────────────────────────
        self.lgb_shap_explainer = shap.TreeExplainer(self.lgb_full_model)
        self.xgb_shap_explainer = shap.TreeExplainer(self.xgb_full_model)

        # ── Conformal wrapper on meta-learner ────────────────────────────
        base_ridge = RidgeCV(alphas=self.config.ridge_alphas)
        self.mapie_regressor = CrossConformalRegressor(
            estimator=base_ridge,
            confidence_level=1.0 - self.config.conformal_alpha,
            method="plus",
            cv=5,
            random_state=self.config.random_seed,
        )
        self.mapie_regressor.fit_conformalize(oof_matrix, y)

        self._is_fitted = True
        logger.info("EnsembleDepthEstimator fit complete.")
        return self

    def _train_lgb_full(self, X: np.ndarray, y: np.ndarray) -> lgb.Booster:
        """Retrain LightGBM on the full dataset (no early stopping)."""
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        best_iters = [m.best_iteration for m in self.lgb_models if m.best_iteration > 0]
        n_rounds = (
            int(np.mean(best_iters)) if best_iters else self.config.lgb_n_estimators
        )
        params = dict(self.config.lgb_params)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            callbacks=[lgb.log_evaluation(-1)],
        )
        return model

    def _train_xgb_full(self, X: np.ndarray, y: np.ndarray) -> xgb.Booster:
        """Retrain XGBoost on the full dataset (no early stopping)."""
        dtrain = xgb.DMatrix(X, label=y, feature_names=self._feature_names)
        best_iters = [m.best_iteration for m in self.xgb_models if m.best_iteration > 0]
        n_rounds = (
            int(np.mean(best_iters)) if best_iters else self.config.xgb_n_estimators
        )
        params = dict(self.config.xgb_params)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            verbose_eval=False,
        )
        return model

    def _build_meta_input(
        self,
        X: np.ndarray,
        X_patches: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build the level-1 meta-feature matrix from base-learner predictions."""
        n_learners = 2 + (1 if self.config.use_hydroformer else 0)
        N = X.shape[0]
        meta = np.zeros((N, n_learners), dtype=np.float32)

        # LightGBM (ensemble average across all fold models)
        lgb_preds = np.mean(
            np.stack(
                [m.predict(X, num_iteration=m.best_iteration) for m in self.lgb_models],
                axis=0,
            ),
            axis=0,
        )
        meta[:, 0] = lgb_preds

        # XGBoost (ensemble average)
        dtest = xgb.DMatrix(X, feature_names=self._feature_names)
        xgb_preds = np.mean(
            np.stack(
                [
                    m.predict(dtest, iteration_range=(0, m.best_iteration + 1))
                    for m in self.xgb_models
                ],
                axis=0,
            ),
            axis=0,
        )
        meta[:, 1] = xgb_preds

        # HydroFormer (optional)
        if self.config.use_hydroformer and self.hf_wrapper is not None:
            hf_preds, _, _ = self.hf_wrapper.predict(X, X_patches=X_patches)
            meta[:, 2] = hf_preds

        return meta

    def predict(
        self,
        X: np.ndarray,
        X_patches: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict depth with conformal prediction intervals.

        Parameters
        ----------
        X:
            Feature matrix (N, F).
        X_patches:
            Optional satellite patches (N, 12, H, W).

        Returns
        -------
        mean_pred  : (N,)  ensemble depth estimate
        lower_ci   : (N,)  lower conformal bound
        upper_ci   : (N,)  upper conformal bound
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        meta = self._build_meta_input(X, X_patches)

        # Point prediction from meta-learner
        mean_pred = self.meta_learner.predict(meta)

        # Conformal intervals from MAPIE
        _, intervals = self.mapie_regressor.predict_interval(
            meta
        )
        lower_ci = intervals[:, 0, 0]
        upper_ci = intervals[:, 1, 0]

        return (
            mean_pred.astype(np.float32),
            lower_ci.astype(np.float32),
            upper_ci.astype(np.float32),
        )

    def predict_base_learners(
        self,
        X: np.ndarray,
        X_patches: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Return individual base-learner predictions for diagnostics."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_base_learners().")

        meta = self._build_meta_input(X, X_patches)
        result: Dict[str, np.ndarray] = {
            "lgb": meta[:, 0],
            "xgb": meta[:, 1],
        }
        if self.config.use_hydroformer:
            result["hydroformer"] = meta[:, 2]
        result["ensemble"] = self.meta_learner.predict(meta).astype(np.float32)
        return result

    # ------------------------------------------------------------------
    # SHAP explanations
    # ------------------------------------------------------------------

    def compute_shap_values(
        self,
        X: np.ndarray,
        model: str = "lgb",
        max_samples: int = 500,
    ) -> np.ndarray:
        """Compute SHAP values for feature importance analysis.

        Parameters
        ----------
        X:
            Feature matrix (N, F).
        model:
            Which model to explain: ``"lgb"`` or ``"xgb"``.
        max_samples:
            Cap on number of samples for efficiency.

        Returns
        -------
        shap_values : (N_capped, F) array of SHAP values.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before compute_shap_values().")

        X_sub = X[:max_samples]
        if model == "lgb":
            if self.lgb_shap_explainer is None:
                raise RuntimeError("LightGBM SHAP explainer not available.")
            return self.lgb_shap_explainer.shap_values(X_sub)
        elif model == "xgb":
            if self.xgb_shap_explainer is None:
                raise RuntimeError("XGBoost SHAP explainer not available.")
            return self.xgb_shap_explainer.shap_values(
                xgb.DMatrix(X_sub, feature_names=self._feature_names)
            )
        else:
            raise ValueError(f"Unknown model '{model}'. Choose 'lgb' or 'xgb'.")

    def feature_importance_df(self, model: str = "lgb") -> pd.DataFrame:
        """Return a sorted feature-importance DataFrame.

        Parameters
        ----------
        model:
            ``"lgb"`` or ``"xgb"``.

        Returns
        -------
        pd.DataFrame with columns ['feature', 'importance'].
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before feature_importance_df().")

        if model == "lgb":
            imp = self.lgb_full_model.feature_importance(importance_type="gain")
            names = self.lgb_full_model.feature_name()
        elif model == "xgb":
            imp_dict = self.xgb_full_model.get_score(importance_type="gain")
            names = list(imp_dict.keys())
            imp = np.array([imp_dict[k] for k in names])
        else:
            raise ValueError(f"Unknown model '{model}'.")

        df = pd.DataFrame({"feature": names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_patches: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute regression metrics on a held-out test set.

        Returns
        -------
        Dict with keys: r2, rmse, mae, coverage_90, interval_width.
        """
        mean_pred, lower_ci, upper_ci = self.predict(X, X_patches)
        rmse = float(np.sqrt(mean_squared_error(y, mean_pred)))
        mae = float(mean_absolute_error(y, mean_pred))
        r2 = float(r2_score(y, mean_pred))
        coverage = float(np.mean((y >= lower_ci) & (y <= upper_ci)))
        interval_width = float(np.mean(upper_ci - lower_ci))

        metrics = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "coverage_90": coverage,
            "interval_width": interval_width,
        }
        logger.info(
            "Evaluation | R²=%.4f  RMSE=%.4f m  MAE=%.4f m  "
            "Coverage=%.3f  Width=%.4f m",
            r2,
            rmse,
            mae,
            coverage,
            interval_width,
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> None:
        """Save all ensemble artefacts to *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save LightGBM fold models
        for i, m in enumerate(self.lgb_models):
            m.save_model(str(directory / f"lgb_fold_{i}.txt"))
        if hasattr(self, "lgb_full_model"):
            self.lgb_full_model.save_model(str(directory / "lgb_full.txt"))

        # Save XGBoost fold models
        for i, m in enumerate(self.xgb_models):
            m.save_model(str(directory / f"xgb_fold_{i}.json"))
        if hasattr(self, "xgb_full_model"):
            self.xgb_full_model.save_model(str(directory / "xgb_full.json"))

        # Save meta-learner and MAPIE regressor
        joblib.dump(self.meta_learner, directory / "meta_learner.pkl")
        if self.mapie_regressor is not None:
            joblib.dump(self.mapie_regressor, directory / "mapie_regressor.pkl")

        logger.info("Ensemble saved to %s", directory)

    @classmethod
    def load(
        cls,
        directory: Union[str, Path],
        config: Optional[EnsembleConfig] = None,
    ) -> "EnsembleDepthEstimator":
        """Load a previously saved ensemble from *directory*."""
        directory = Path(directory)
        estimator = cls(config=config)

        # LightGBM fold models
        fold_files = sorted(directory.glob("lgb_fold_*.txt"))
        for p in fold_files:
            m = lgb.Booster(model_file=str(p))
            estimator.lgb_models.append(m)

        # LightGBM full model
        lgb_full_path = directory / "lgb_full.txt"
        if lgb_full_path.exists():
            estimator.lgb_full_model = lgb.Booster(model_file=str(lgb_full_path))
            estimator.lgb_shap_explainer = shap.TreeExplainer(estimator.lgb_full_model)

        # XGBoost fold models
        xgb_fold_files = sorted(directory.glob("xgb_fold_*.json"))
        for p in xgb_fold_files:
            m = xgb.Booster()
            m.load_model(str(p))
            estimator.xgb_models.append(m)

        # XGBoost full model
        xgb_full_path = directory / "xgb_full.json"
        if xgb_full_path.exists():
            estimator.xgb_full_model = xgb.Booster()
            estimator.xgb_full_model.load_model(str(xgb_full_path))
            estimator.xgb_shap_explainer = shap.TreeExplainer(estimator.xgb_full_model)

        # Meta-learner
        meta_path = directory / "meta_learner.pkl"
        if meta_path.exists():
            estimator.meta_learner = joblib.load(meta_path)

        # MAPIE regressor
        mapie_path = directory / "mapie_regressor.pkl"
        if mapie_path.exists():
            estimator.mapie_regressor = joblib.load(mapie_path)

        estimator._is_fitted = bool(estimator.lgb_models) and bool(estimator.xgb_models)
        logger.info(
            "Ensemble loaded from %s (fitted=%s)", directory, estimator._is_fitted
        )
        return estimator


# ---------------------------------------------------------------------------
# NavigabilityClassifier
# ---------------------------------------------------------------------------


@dataclass
class NavigabilityConfig:
    """Configuration for the NavigabilityClassifier.

    Attributes
    ----------
    lgb_params:
        LightGBM multiclass parameters.
    n_estimators:
        LightGBM boosting rounds.
    calibration_method:
        Scikit-learn calibration method (``"isotonic"`` or ``"sigmoid"``).
    conformal_alpha:
        Miscoverage rate for MAPIE conformal sets (default 0.1 → 90%).
    cv:
        CV folds for CalibratedClassifierCV.
    random_seed:
        RNG seed.
    feature_names:
        Input feature names for SHAP.
    """

    lgb_params: Dict[str, Any] = field(default_factory=_default_lgb_clf_params)
    n_estimators: int = 500
    calibration_method: str = "isotonic"
    conformal_alpha: float = 0.1
    cv: int = 5
    random_seed: int = 42
    feature_names: Optional[List[str]] = None


class NavigabilityClassifier:
    """Calibrated LightGBM classifier for river-segment navigability.

    Input features (minimum required):
        depth_pred          – predicted depth (m)
        depth_uncertainty   – prediction interval width (m)
        water_width_m       – water-surface width (m)
        gauge_discharge_m3s – river discharge (m³/s)
        sinuosity           – channel sinuosity ratio

    Output:
        class label   – 0/1/2 (Non-Navigable / Conditional / Navigable)
        confidence    – calibrated probability of predicted class
        conformal_set – set of plausible class indices at chosen alpha

    Architecture:
        1. LightGBM multiclass → raw probabilities
        2. CalibratedClassifierCV (isotonic) → calibrated probabilities
        3. MAPIE MapieClassifier (RAPS) → conformal prediction sets

    Parameters
    ----------
    config:
        :class:`NavigabilityConfig` instance.
    """

    # Required input feature names
    REQUIRED_FEATURES: List[str] = [
        "depth_pred",
        "depth_uncertainty",
        "water_width_m",
        "gauge_discharge_m3s",
        "sinuosity",
    ]

    def __init__(self, config: Optional[NavigabilityConfig] = None) -> None:
        self.config = config or NavigabilityConfig()
        self._is_fitted: bool = False

        # ── LightGBM base classifier (sklearn API) ───────────────────────
        self.lgb_base = lgb.LGBMClassifier(
            **{
                k: v
                for k, v in self.config.lgb_params.items()
                if k != "metric"  # LGBMClassifier ignores 'metric' kwarg
            },
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_seed,
        )

        # ── Calibrated classifier ────────────────────────────────────────
        self.calibrated_clf = CalibratedClassifierCV(
            estimator=self.lgb_base,
            method=self.config.calibration_method,
            cv=self.config.cv,
        )

        # ── MAPIE conformal classifier ───────────────────────────────────
        self.mapie_clf: Optional[CrossConformalClassifier] = None

        # ── SHAP explainer ───────────────────────────────────────────────
        self.shap_explainer: Optional[shap.TreeExplainer] = None

        # ── Label encoder ────────────────────────────────────────────────
        self.label_encoder = LabelEncoder()
        self._feature_names: Optional[List[str]] = self.config.feature_names

        logger.info(
            "NavigabilityClassifier | calibration=%s, conformal_alpha=%.2f",
            self.config.calibration_method,
            self.config.conformal_alpha,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_features(
        self, X: np.ndarray, feature_names: Optional[List[str]]
    ) -> None:
        """Warn if required features appear to be absent."""
        if feature_names is not None:
            missing = [f for f in self.REQUIRED_FEATURES if f not in feature_names]
            if missing:
                logger.warning(
                    "Potentially missing required features: %s. "
                    "Make sure columns are in the correct order.",
                    missing,
                )

    def _depth_to_nav_label(self, depth: float, width: float = float("inf")) -> int:
        """Convert scalar depth + width to navigability class index."""
        if depth >= DEPTH_NAVIGABLE and width >= 50.0:
            return 2
        if depth >= DEPTH_CONDITIONAL:
            return 1
        return 0

    def build_nav_labels_from_depth(
        self,
        depth: np.ndarray,
        width: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorised depth → navigability label conversion.

        Parameters
        ----------
        depth:
            Array of depth values in metres.
        width:
            Optional array of water-surface widths in metres.

        Returns
        -------
        np.ndarray of int labels (0/1/2).
        """
        if width is None:
            width = np.full_like(depth, fill_value=float("inf"))
        labels = np.zeros(len(depth), dtype=np.int64)
        labels[depth >= DEPTH_CONDITIONAL] = 1
        labels[(depth >= DEPTH_NAVIGABLE) & (width >= 50.0)] = 2
        return labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "NavigabilityClassifier":
        """Fit the calibrated LightGBM classifier and MAPIE conformal model.

        Parameters
        ----------
        X:
            Feature matrix (N, F).
        y:
            Integer navigability labels 0/1/2 (N,).
        feature_names:
            Optional list of F feature names.

        Returns
        -------
        self
        """
        if feature_names is not None:
            self._feature_names = feature_names
        self._validate_features(X, self._feature_names)

        y_enc = y.astype(np.int64)
        unique_classes = np.unique(y_enc)
        logger.info(
            "Fitting NavigabilityClassifier | N=%d, classes=%s",
            len(y),
            {int(c): NAV_CLASS_MAP.get(int(c), str(c)) for c in unique_classes},
        )

        # ── Calibrated classifier ────────────────────────────────────────
        self.calibrated_clf.fit(X, y_enc)
        logger.info("CalibratedClassifierCV fitted.")

        # ── MAPIE conformal classifier ───────────────────────────────────
        # Use a fresh LightGBM inside MAPIE for conformal calibration
        lgb_for_mapie = lgb.LGBMClassifier(
            **{k: v for k, v in self.config.lgb_params.items() if k != "metric"},
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_seed,
        )
        self.mapie_clf = CrossConformalClassifier(
            estimator=lgb_for_mapie,
            confidence_level=1.0 - self.config.conformal_alpha,
            cv=5,  # use CV internally
            random_state=self.config.random_seed,
        )
        self.mapie_clf.fit_conformalize(X, y_enc)
        logger.info("MAPIE MapieClassifier fitted.")

        # ── SHAP explainer (on underlying LightGBM) ─────────────────────
        # Access the underlying booster from one calibrated estimator
        try:
            # CalibratedClassifierCV stores estimators in .calibrated_classifiers_
            base_lgb = self.calibrated_clf.calibrated_classifiers_[0].estimator
            self.shap_explainer = shap.TreeExplainer(base_lgb)
            logger.info("SHAP TreeExplainer ready.")
        except Exception as exc:
            logger.warning("Could not build SHAP explainer: %s", exc)

        self._is_fitted = True
        logger.info("NavigabilityClassifier fit complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated class probabilities (N, 3)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.calibrated_clf.predict_proba(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (N,)."""
        return self.calibrated_clf.predict(X).astype(np.int64)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Predict class + calibrated confidence + conformal prediction set.

        Parameters
        ----------
        X:
            Feature matrix (N, F).

        Returns
        -------
        List of N dicts, each containing:
            'class'          : int label (0/1/2)
            'class_name'     : str label
            'confidence'     : float calibrated probability
            'probabilities'  : dict {class_name: prob}
            'conformal_set'  : list of plausible class indices
            'conformal_names': list of plausible class names
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_with_uncertainty().")
        if self.mapie_clf is None:
            raise RuntimeError("MAPIE classifier not fitted.")

        proba = self.predict_proba(X)  # (N, 3)
        labels = self.predict(X)  # (N,)

        # MAPIE conformal prediction sets
        _, conf_sets = self.mapie_clf.predict_set(X)
        # conf_sets shape: (N, n_classes, n_alphas)

        results: List[Dict[str, Any]] = []
        for i in range(len(X)):
            cls = int(labels[i])
            conf = float(proba[i, cls])
            prob_dict = {NAV_CLASS_MAP[j]: float(proba[i, j]) for j in range(3)}

            # Conformal set: classes where conf_sets[i, j, 0] == True
            conf_set_arr = conf_sets[i, :, 0]  # (3,)
            conf_set = [j for j in range(3) if conf_set_arr[j]]
            if not conf_set:
                conf_set = [cls]  # fallback: singleton

            results.append(
                {
                    "class": cls,
                    "class_name": NAV_CLASS_MAP[cls],
                    "confidence": conf,
                    "probabilities": prob_dict,
                    "conformal_set": conf_set,
                    "conformal_names": [NAV_CLASS_MAP[j] for j in conf_set],
                }
            )

        return results

    def compute_shap_values(
        self,
        X: np.ndarray,
        max_samples: int = 500,
    ) -> Optional[np.ndarray]:
        """Compute SHAP values for the classifier.

        Parameters
        ----------
        X:
            Feature matrix (N, F).
        max_samples:
            Cap on samples used for SHAP computation.

        Returns
        -------
        shap_values : list of (N, F) arrays, one per class.
            Returns None if SHAP explainer is not available.
        """
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not available.")
            return None
        X_sub = X[:max_samples]
        return self.shap_explainer.shap_values(X_sub)

    def shap_summary_df(
        self,
        X: np.ndarray,
        class_idx: int = 2,
        max_samples: int = 500,
    ) -> Optional[pd.DataFrame]:
        """Return a sorted SHAP importance DataFrame for a given class.

        Parameters
        ----------
        X:
            Feature matrix.
        class_idx:
            Which navigability class to explain (default 2 = Navigable).
        max_samples:
            Sample cap.

        Returns
        -------
        pd.DataFrame with columns ['feature', 'mean_abs_shap'] sorted desc.
        """
        sv = self.compute_shap_values(X, max_samples=max_samples)
        if sv is None:
            return None

        if isinstance(sv, list):
            class_sv = sv[class_idx]
        else:
            class_sv = sv

        mean_abs = np.abs(class_sv).mean(axis=0)
        names = (
            self._feature_names
            if self._feature_names and len(self._feature_names) == mean_abs.shape[0]
            else [f"f{i}" for i in range(mean_abs.shape[0])]
        )
        df = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate classifier on a held-out test set.

        Returns
        -------
        Dict with accuracy, macro-F1, per-class precision/recall/F1.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before evaluate().")

        y_pred = self.predict(X)
        acc = float(accuracy_score(y, y_pred))
        f1_macro = float(f1_score(y, y_pred, average="macro", zero_division=0))
        prec_macro = float(precision_score(y, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y, y_pred, average="macro", zero_division=0))

        metrics: Dict[str, float] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
        }

        # Per-class metrics
        f1_per = f1_score(y, y_pred, average=None, zero_division=0)
        prec_per = precision_score(y, y_pred, average=None, zero_division=0)
        rec_per = recall_score(y, y_pred, average=None, zero_division=0)
        for i, name in NAV_CLASS_MAP.items():
            if i < len(f1_per):
                safe_name = name.lower().replace("-", "_")
                metrics[f"f1_{safe_name}"] = float(f1_per[i])
                metrics[f"precision_{safe_name}"] = float(prec_per[i])
                metrics[f"recall_{safe_name}"] = float(rec_per[i])

        logger.info(
            "NavigabilityClassifier eval | Acc=%.4f  F1=%.4f  Prec=%.4f  Recall=%.4f",
            acc,
            f1_macro,
            prec_macro,
            rec_macro,
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the classifier to a single joblib file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("NavigabilityClassifier saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NavigabilityClassifier":
        """Load a serialised NavigabilityClassifier."""
        obj = joblib.load(path)
        logger.info("NavigabilityClassifier loaded from %s", path)
        return obj


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    rng = np.random.default_rng(42)
    print("\n── EnsembleDepthEstimator smoke test ──")

    N, F = 200, 26
    X = rng.standard_normal((N, F)).astype(np.float32)
    y = (rng.uniform(0, 6, size=N)).astype(np.float32)
    seg_ids = np.array([f"SEG_{i:04d}" for i in range(N)])
    feat_names = [f"feat_{i}" for i in range(F)]

    cfg = EnsembleConfig(
        n_splits=3,
        lgb_n_estimators=50,
        xgb_n_estimators=50,
        use_hydroformer=False,
    )
    ens = EnsembleDepthEstimator(config=cfg)
    ens.fit(X, y, segment_ids=seg_ids, feature_names=feat_names)

    mean_p, lower, upper = ens.predict(X[:20])
    print(f"  depth_pred[:5] : {mean_p[:5]}")
    print(f"  lower_ci[:5]   : {lower[:5]}")
    print(f"  upper_ci[:5]   : {upper[:5]}")

    metrics = ens.evaluate(X[:20], y[:20])
    print(f"  metrics        : {metrics}")

    shap_df = ens.feature_importance_df("lgb")
    print(f"  top-5 features :\n{shap_df.head()}")

    print("\n── NavigabilityClassifier smoke test ──")

    X_clf = rng.standard_normal((N, 5)).astype(np.float32)
    y_clf = np.clip((rng.uniform(0, 6, size=N) / 2).astype(np.int64), 0, 2)
    nav_feat_names = NavigabilityClassifier.REQUIRED_FEATURES

    nav_cfg = NavigabilityConfig(n_estimators=50)
    clf = NavigabilityClassifier(config=nav_cfg)
    clf.fit(X_clf, y_clf, feature_names=nav_feat_names)

    preds = clf.predict_with_uncertainty(X_clf[:5])
    for p in preds:
        print(
            f"  class={p['class_name']:14s}  "
            f"conf={p['confidence']:.3f}  "
            f"conformal_set={p['conformal_names']}"
        )

    eval_metrics = clf.evaluate(X_clf[100:], y_clf[100:])
    print(f"  eval: {eval_metrics}")

    print("\nAll smoke tests passed ✓")
    sys.exit(0)
