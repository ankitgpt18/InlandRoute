"""
train.py
========
Training orchestration for the AIDSTL inland-waterway navigability project.

Covers
------
TrainingConfig          Pydantic-validated hyperparameter container.
train_hydroformer()     Trains the HydroFormer (Swin-T + TFT) with:
                          • cosine-annealing LR schedule with linear warmup
                          • mixed-precision (torch.cuda.amp)
                          • pinball / quantile loss
                          • WandB experiment tracking
                          • early stopping (patience=15)
                          • spatial block cross-validation
train_full_ensemble()   Trains the complete stacking ensemble
                        (HydroFormer + LightGBM + XGBoost + Ridge).
evaluate_model()        Computes regression + classification metrics on
                        a held-out test split.
CLI                     argparse entry point supporting
                        `train hydroformer | ensemble | evaluate`.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ensemble import (
    EnsembleConfig,
    EnsembleDepthEstimator,
    NavigabilityClassifier,
    NavigabilityConfig,
    spatial_block_cv_splits,
)

# ── Project imports ──────────────────────────────────────────────────────────
from feature_engineering import (
    ALL_FEATURES,
    NAV_CLASSES,
    RiverSegmentDataset,
    SequenceConfig,
    TemporalSequenceBuilder,
    add_nav_labels,
)
from hydroformer import (
    HydroFormer,
    HydroFormerLoss,
    build_hydroformer,
    count_parameters,
)
from pydantic import BaseModel, Field, field_validator, model_validator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Optional WandB import (graceful degradation if not installed / logged in)
# ---------------------------------------------------------------------------
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed — experiment tracking disabled.")


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    """Pydantic-validated hyperparameter container for all training runs.

    All fields have sensible defaults matching the project specification.
    Override any field via the CLI ``--config`` JSON argument or by
    instantiating the class programmatically.

    Attributes
    ----------
    # ── Paths ──
    data_path:
        Path to the Parquet feature matrix produced by GEEPipeline.
    patch_dir:
        Optional directory of per-segment .npy satellite patches.
    norm_stats_path:
        Path to pre-fitted normalisation stats (.npz).  Auto-fitted if absent.
    output_dir:
        Root directory for saving checkpoints, logs, and artefacts.

    # ── Data ──
    sequence_length:
        Number of time steps T (default 12 months).
    n_temporal_features:
        Number of time-varying features F_t.
    n_static_features:
        Number of static features F_s.
    patch_size:
        Spatial size of satellite image patches (H = W).
    n_patch_bands:
        Number of image channels in patches (default 12 for Sentinel-2).
    val_fraction:
        Fraction of data to hold out for validation.
    test_fraction:
        Fraction of data to hold out for testing.
    spatial_block_cv:
        If True use spatial block CV; if False use random split.

    # ── Model ──
    d_model:
        TFT / HydroFormer core hidden dimension.
    n_heads:
        Number of multi-head attention heads.
    lstm_hidden:
        LSTM hidden state size.
    lstm_layers:
        Number of stacked LSTM layers.
    swin_embed_dim:
        Output dimension of SwinSpectralEncoder.
    use_swin:
        Whether to include the Swin-T spatial encoder.
    dropout:
        Dropout rate throughout the model.
    quantiles:
        Output quantile levels.

    # ── Training ──
    batch_size:
        Training batch size.
    learning_rate:
        Peak learning rate after warmup.
    weight_decay:
        AdamW weight decay.
    max_epochs:
        Maximum number of training epochs.
    warmup_epochs:
        Number of linear-warmup epochs before cosine annealing begins.
    eta_min:
        Minimum LR at end of cosine annealing schedule.
    early_stopping_patience:
        Epochs without validation improvement before stopping.
    gradient_clip_norm:
        Max gradient norm for clipping (0 = disabled).
    mixed_precision:
        Enable torch.cuda.amp automatic mixed precision.
    accumulation_steps:
        Gradient accumulation steps (effective_batch = batch_size × steps).

    # ── Loss ──
    lambda_mse:
        Weight on MSE term of HydroFormerLoss.
    lambda_quantile:
        Weight on quantile (pinball) loss term.
    lambda_coverage:
        Weight on PI coverage penalty term.

    # ── Ensemble ──
    ensemble_n_splits:
        Number of spatial CV folds for ensemble OOF training.
    lgb_n_estimators:
        LightGBM boosting rounds.
    xgb_n_estimators:
        XGBoost boosting rounds.
    conformal_alpha:
        Miscoverage rate for conformal prediction intervals.

    # ── Misc ──
    seed:
        Global random seed.
    n_workers:
        DataLoader worker processes.
    use_wandb:
        Enable WandB logging.
    wandb_project:
        WandB project name.
    wandb_run_name:
        Optional WandB run name.
    log_every_n_steps:
        Log metrics every N optimiser steps.
    save_top_k:
        Number of best checkpoints to keep.
    """

    # ── Paths ─────────────────────────────────────────────────────────────
    data_path: str = "data/features.parquet"
    patch_dir: Optional[str] = None
    norm_stats_path: Optional[str] = None
    output_dir: str = "outputs"

    # ── Data ──────────────────────────────────────────────────────────────
    sequence_length: int = Field(default=12, ge=1, le=120)
    n_temporal_features: int = Field(default=26, ge=1)
    n_static_features: int = Field(default=16, ge=1)
    patch_size: int = Field(default=224, ge=32)
    n_patch_bands: int = Field(default=12, ge=1)
    val_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.10, gt=0.0, lt=1.0)
    spatial_block_cv: bool = True

    # ── Model ─────────────────────────────────────────────────────────────
    d_model: int = Field(default=128, ge=32)
    n_heads: int = Field(default=8, ge=1)
    lstm_hidden: int = Field(default=128, ge=32)
    lstm_layers: int = Field(default=2, ge=1, le=4)
    swin_embed_dim: int = Field(default=64, ge=16)
    use_swin: bool = True
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    quantiles: List[float] = Field(default=[0.10, 0.50, 0.90])

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    max_epochs: int = Field(default=100, ge=1)
    warmup_epochs: int = Field(default=5, ge=0)
    eta_min: float = Field(default=1e-6, ge=0.0)
    early_stopping_patience: int = Field(default=15, ge=1)
    gradient_clip_norm: float = Field(default=1.0, ge=0.0)
    mixed_precision: bool = True
    accumulation_steps: int = Field(default=1, ge=1)

    # ── Loss ──────────────────────────────────────────────────────────────
    lambda_mse: float = Field(default=1.0, ge=0.0)
    lambda_quantile: float = Field(default=0.5, ge=0.0)
    lambda_coverage: float = Field(default=0.1, ge=0.0)

    # ── Ensemble ──────────────────────────────────────────────────────────
    ensemble_n_splits: int = Field(default=5, ge=2)
    lgb_n_estimators: int = Field(default=1000, ge=10)
    xgb_n_estimators: int = Field(default=1000, ge=10)
    conformal_alpha: float = Field(default=0.1, gt=0.0, lt=1.0)

    # ── Misc ──────────────────────────────────────────────────────────────
    seed: int = 42
    n_workers: int = Field(default=4, ge=0)
    use_wandb: bool = False
    wandb_project: str = "AIDSTL"
    wandb_run_name: Optional[str] = None
    log_every_n_steps: int = Field(default=10, ge=1)
    save_top_k: int = Field(default=3, ge=1)

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, v: List[float]) -> List[float]:
        if not all(0.0 < q < 1.0 for q in v):
            raise ValueError("All quantiles must be in (0, 1).")
        if sorted(v) != v:
            raise ValueError("Quantiles must be in ascending order.")
        return v

    @model_validator(mode="after")
    def validate_fractions(self) -> "TrainingConfig":
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"val_fraction ({self.val_fraction}) + test_fraction "
                f"({self.test_fraction}) must be < 1.0."
            )
        return self

    @model_validator(mode="after")
    def validate_d_model_heads(self) -> "TrainingConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        return self

    # ── Helpers ───────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return all fields as a plain Python dict."""
        return self.model_dump()

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load a TrainingConfig from a JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save_json(self, path: Union[str, Path]) -> None:
        """Persist config to a JSON file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Config saved to %s", path)


# ---------------------------------------------------------------------------
# Reproducibility & device helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Fix all RNG seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available device."""
    if prefer_gpu and torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(
            "Using GPU: %s (VRAM %.1f GiB)",
            torch.cuda.get_device_name(dev),
            torch.cuda.get_device_properties(dev).total_memory / 1e9,
        )
    else:
        dev = torch.device("cpu")
        logger.info("Using CPU.")
    return dev


# ---------------------------------------------------------------------------
# WandB context manager
# ---------------------------------------------------------------------------


@contextmanager
def wandb_run(
    config: TrainingConfig,
    job_type: str = "train",
) -> Generator[Any, None, None]:
    """Context manager that starts / stops a WandB run if enabled."""
    if config.use_wandb and WANDB_AVAILABLE:
        run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.to_dict(),
            job_type=job_type,
            reinit=True,
        )
        try:
            yield run
        finally:
            run.finish()
    else:
        yield None


def log_metrics(
    run: Any,
    metrics: Dict[str, float],
    step: Optional[int] = None,
) -> None:
    """Log metrics to WandB if a run is active; always log to logger."""
    logger.info("Metrics @ step %s: %s", step, metrics)
    if run is not None:
        run.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Keeps the top-k model checkpoints by validation loss.

    Parameters
    ----------
    save_dir:
        Directory where checkpoints are stored.
    top_k:
        Maximum number of checkpoints to retain.
    monitor:
        Metric to monitor (lower is better).
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        top_k: int = 3,
        monitor: str = "val_loss",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.monitor = monitor
        self._checkpoints: List[Tuple[float, Path]] = []  # (score, path)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
        extra: Optional[Dict] = None,
    ) -> Path:
        """Save a checkpoint, pruning the worst if > top_k."""
        fname = self.save_dir / f"ckpt_epoch{epoch:04d}_score{score:.4f}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            self.monitor: score,
        }
        if extra:
            state.update(extra)
        torch.save(state, fname)
        self._checkpoints.append((score, fname))

        # Keep only top_k (lowest score = best)
        self._checkpoints.sort(key=lambda x: x[0])
        while len(self._checkpoints) > self.top_k:
            _, worst_path = self._checkpoints.pop()
            if worst_path.exists():
                worst_path.unlink()
                logger.debug("Pruned checkpoint: %s", worst_path)

        logger.info("Saved checkpoint: %s (score=%.4f)", fname.name, score)
        return fname

    def best_checkpoint(self) -> Optional[Path]:
        """Return path to the best (lowest score) checkpoint."""
        if not self._checkpoints:
            return None
        return self._checkpoints[0][1]

    def load_best(self, model: nn.Module, device: torch.device) -> int:
        """Load the best checkpoint into *model* and return its epoch."""
        best = self.best_checkpoint()
        if best is None:
            raise RuntimeError("No checkpoints saved yet.")
        state = torch.load(best, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        epoch = state.get("epoch", -1)
        logger.info("Loaded best checkpoint: %s (epoch=%d)", best.name, epoch)
        return epoch


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Monitor validation loss and trigger early stopping.

    Parameters
    ----------
    patience:
        Epochs without improvement before stopping.
    min_delta:
        Minimum absolute improvement to count as a new best.
    mode:
        ``"min"`` or ``"max"`` depending on the monitored metric.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, score: float) -> bool:
        """Update state.  Returns True if training should stop."""
        improved = (
            score < self.best - self.min_delta
            if self.mode == "min"
            else score > self.best + self.min_delta
        )
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered (patience=%d, best=%.4f).",
                    self.patience,
                    self.best,
                )
        return self.should_stop


# ---------------------------------------------------------------------------
# LR scheduler factory
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a cosine-annealing schedule with optional linear warmup."""
    if config.warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.max_epochs - config.warmup_epochs),
            eta_min=config.eta_min,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=config.eta_min,
        )
    return scheduler


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(config: TrainingConfig) -> RiverSegmentDataset:
    """Load and prepare the full RiverSegmentDataset from config paths.

    Parameters
    ----------
    config:
        TrainingConfig with data_path, patch_dir, norm_stats_path.

    Returns
    -------
    RiverSegmentDataset
    """
    seq_cfg = SequenceConfig(
        sequence_length=config.sequence_length,
        feature_columns=list(ALL_FEATURES),
        normalise=True,
    )
    dataset = RiverSegmentDataset.from_parquet(
        parquet_path=config.data_path,
        patch_dir=config.patch_dir,
        seq_config=seq_cfg,
        norm_stats_path=config.norm_stats_path,
        patch_size=config.patch_size,
        n_patch_bands=config.n_patch_bands,
        augment=False,
    )
    return dataset


def split_dataset(
    dataset: RiverSegmentDataset,
    config: TrainingConfig,
) -> Tuple[RiverSegmentDataset, RiverSegmentDataset, RiverSegmentDataset]:
    """Split into train / val / test subsets.

    Returns
    -------
    (train_ds, val_ds, test_ds)
    """
    N = len(dataset)
    n_test = max(1, int(N * config.test_fraction))
    n_val = max(1, int(N * config.val_fraction))
    n_train = N - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Dataset too small ({N}) for the requested val/test fractions."
        )

    if config.spatial_block_cv:
        # Spatial ordering: contiguous blocks
        all_idx = np.arange(N)
        test_idx = all_idx[-n_test:]
        val_idx = all_idx[-(n_test + n_val) : -n_test]
        train_idx = all_idx[: -(n_test + n_val)]
    else:
        rng = np.random.default_rng(config.seed)
        perm = rng.permutation(N)
        test_idx = perm[:n_test]
        val_idx = perm[n_test : n_test + n_val]
        train_idx = perm[n_test + n_val :]

    def _subset(idx: np.ndarray, augment: bool) -> RiverSegmentDataset:
        return RiverSegmentDataset(
            sequences=dataset.sequences[idx].numpy(),
            targets=dataset.targets[idx].numpy(),
            segment_ids=dataset.segment_ids[idx],
            static_features=dataset.static_features[idx].numpy(),
            patches=dataset.patches[idx].numpy(),
            patch_size=dataset.patch_size,
            n_patch_bands=dataset.n_patch_bands,
            augment=augment,
        )

    train_ds = _subset(train_idx, augment=True)
    val_ds = _subset(val_idx, augment=False)
    test_ds = _subset(test_idx, augment=False)

    logger.info(
        "Dataset split | train=%d  val=%d  test=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------


def _train_epoch(
    model: HydroFormer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: HydroFormerLoss,
    scaler: GradScaler,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
    run: Any,
) -> Tuple[float, int]:
    """Run one training epoch.

    Returns
    -------
    (mean_train_loss, updated_global_step)
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for step, batch in enumerate(pbar):
        x_temporal = batch["x_temporal"].to(device)
        x_static = batch["x_static"].to(device)
        x_patch = batch["x_patch"].to(device) if config.use_swin else None
        y = batch["y"].to(device)

        use_amp = config.mixed_precision and device.type == "cuda"
        with autocast(enabled=use_amp):
            depth_pred, lower_ci, upper_ci = model(x_static, x_temporal, x_patch)
            # Re-derive q10 / q90 from HydroFormer CI bounds
            q10 = lower_ci
            q90 = upper_ci
            loss_dict = criterion(depth_pred, q10, q90, y)
            loss = loss_dict["total"] / config.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.accumulation_steps == 0:
            if config.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            if global_step % config.log_every_n_steps == 0:
                step_metrics = {
                    "train/loss": loss_dict["total"].item(),
                    "train/mse": loss_dict["mse"].item(),
                    "train/quantile": loss_dict["quantile"].item(),
                    "train/coverage_penalty": loss_dict["coverage"].item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                log_metrics(run, step_metrics, step=global_step)

        total_loss += loss_dict["total"].item()
        pbar.set_postfix({"loss": f"{loss_dict['total'].item():.4f}"})

    return total_loss / max(len(loader), 1), global_step


@torch.no_grad()
def _val_epoch(
    model: HydroFormer,
    loader: DataLoader,
    criterion: HydroFormerLoss,
    device: torch.device,
    config: TrainingConfig,
) -> Dict[str, float]:
    """Run one validation epoch and return aggregated metrics."""
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_lower: List[np.ndarray] = []
    all_upper: List[np.ndarray] = []

    use_amp = config.mixed_precision and device.type == "cuda"

    for batch in tqdm(loader, desc="  [val]", leave=False):
        x_temporal = batch["x_temporal"].to(device)
        x_static = batch["x_static"].to(device)
        x_patch = batch["x_patch"].to(device) if config.use_swin else None
        y = batch["y"].to(device)

        with autocast(enabled=use_amp):
            depth_pred, lower_ci, upper_ci = model(x_static, x_temporal, x_patch)
            q10, q90 = lower_ci, upper_ci
            loss_dict = criterion(depth_pred, q10, q90, y)

        total_loss += loss_dict["total"].item()
        all_preds.append(depth_pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        all_lower.append(lower_ci.cpu().numpy())
        all_upper.append(upper_ci.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    lower = np.concatenate(all_lower)
    upper = np.concatenate(all_upper)

    # Regression metrics
    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    mae = float(mean_absolute_error(targets, preds))
    r2 = float(r2_score(targets, preds))
    coverage = float(np.mean((targets >= lower) & (targets <= upper)))

    return {
        "val/loss": total_loss / max(len(loader), 1),
        "val/rmse": rmse,
        "val/mae": mae,
        "val/r2": r2,
        "val/coverage_90": coverage,
    }


# ---------------------------------------------------------------------------
# train_hydroformer
# ---------------------------------------------------------------------------


def train_hydroformer(
    config: TrainingConfig,
    dataset: Optional[RiverSegmentDataset] = None,
) -> HydroFormer:
    """Train the HydroFormer (Swin-T + TFT) model.

    Parameters
    ----------
    config:
        :class:`TrainingConfig` with all hyperparameters.
    dataset:
        Optional pre-loaded dataset.  If None, loaded from config.data_path.

    Returns
    -------
    HydroFormer
        The best model (loaded from best checkpoint) in eval mode.
    """
    set_seed(config.seed)
    device = get_device()
    output_dir = Path(config.output_dir) / "hydroformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Persist config ───────────────────────────────────────────────────
    config.save_json(output_dir / "config.json")

    # ── Dataset ──────────────────────────────────────────────────────────
    if dataset is None:
        logger.info("Loading dataset from %s …", config.data_path)
        dataset = load_dataset(config)

    train_ds, val_ds, _ = split_dataset(dataset, config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.n_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.n_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = build_hydroformer(
        n_temporal_features=config.n_temporal_features,
        n_static_features=config.n_static_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers,
        swin_embed_dim=config.swin_embed_dim,
        dropout=config.dropout,
        patch_size=config.patch_size,
        quantiles=config.quantiles,
        use_swin=config.use_swin,
    ).to(device)

    param_counts = count_parameters(model)
    logger.info(
        "HydroFormer trainable parameters: %d",
        param_counts["_overall"]["trainable"],
    )

    # ── Optimiser, Scheduler, Loss ───────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config)
    criterion = HydroFormerLoss(
        lambda_mse=config.lambda_mse,
        lambda_quantile=config.lambda_quantile,
        lambda_coverage=config.lambda_coverage,
        quantiles=config.quantiles,
    )
    scaler = GradScaler(enabled=config.mixed_precision and device.type == "cuda")

    ckpt_manager = CheckpointManager(
        save_dir=output_dir / "checkpoints",
        top_k=config.save_top_k,
        monitor="val/loss",
    )
    early_stopper = EarlyStopping(
        patience=config.early_stopping_patience,
        mode="min",
    )

    # ── Training loop ────────────────────────────────────────────────────
    global_step = 0
    t0 = time.time()

    with wandb_run(config, job_type="train_hydroformer") as run:
        if run is not None:
            run.watch(model, log="gradients", log_freq=50)

        for epoch in range(1, config.max_epochs + 1):
            # Train
            train_loss, global_step = _train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                device,
                config,
                epoch,
                global_step,
                run,
            )

            # Validate
            val_metrics = _val_epoch(model, val_loader, criterion, device, config)
            val_loss = val_metrics["val/loss"]

            # LR step
            scheduler.step()

            # Log
            epoch_metrics = {
                "train/loss_epoch": train_loss,
                "train/lr_epoch": optimizer.param_groups[0]["lr"],
                **val_metrics,
            }
            log_metrics(run, epoch_metrics, step=epoch)

            logger.info(
                "Epoch %03d | train_loss=%.4f | val_loss=%.4f | "
                "val_rmse=%.4f m | val_r2=%.4f | val_cov=%.3f | "
                "lr=%.2e | elapsed=%.0fs",
                epoch,
                train_loss,
                val_loss,
                val_metrics["val/rmse"],
                val_metrics["val/r2"],
                val_metrics["val/coverage_90"],
                optimizer.param_groups[0]["lr"],
                time.time() - t0,
            )

            # Checkpoint
            ckpt_manager.save(
                model,
                optimizer,
                epoch,
                val_loss,
                extra={"val_metrics": val_metrics},
            )

            # Early stopping
            if early_stopper.step(val_loss):
                logger.info("Early stopping at epoch %d.", epoch)
                break

        # ── Load best model ───────────────────────────────────────────────
        best_epoch = ckpt_manager.load_best(model, device)
        logger.info(
            "Training complete. Best epoch=%d | val_loss=%.4f",
            best_epoch,
            early_stopper.best,
        )

        if run is not None:
            run.summary["best_val_loss"] = early_stopper.best
            run.summary["best_epoch"] = best_epoch

    model.eval()
    return model


# ---------------------------------------------------------------------------
# train_full_ensemble
# ---------------------------------------------------------------------------


def train_full_ensemble(
    config: TrainingConfig,
    dataset: Optional[RiverSegmentDataset] = None,
    pretrained_hydroformer: Optional[HydroFormer] = None,
) -> Tuple[EnsembleDepthEstimator, NavigabilityClassifier]:
    """Train the complete stacking ensemble and navigability classifier.

    Steps
    -----
    1. (Optionally) train HydroFormer or use a pre-trained one.
    2. Extract tabular feature matrix from the dataset.
    3. Fit :class:`EnsembleDepthEstimator` (LightGBM + XGBoost + Ridge).
    4. Derive navigability labels from ground-truth depth.
    5. Fit :class:`NavigabilityClassifier`.
    6. Save all artefacts to ``config.output_dir/ensemble``.

    Parameters
    ----------
    config:
        :class:`TrainingConfig`.
    dataset:
        Optional pre-loaded dataset.
    pretrained_hydroformer:
        Optional pre-trained HydroFormer.  If supplied and
        ``config.use_swin`` is True, it is used as the neural base learner.

    Returns
    -------
    (EnsembleDepthEstimator, NavigabilityClassifier)
    """
    set_seed(config.seed)
    device = get_device()
    output_dir = Path(config.output_dir) / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    config.save_json(output_dir / "config.json")

    # ── Dataset ──────────────────────────────────────────────────────────
    if dataset is None:
        logger.info("Loading dataset from %s …", config.data_path)
        dataset = load_dataset(config)

    train_ds, val_ds, test_ds = split_dataset(dataset, config)

    # Extract flat numpy arrays for tabular learners
    def _extract_tabular(
        ds: RiverSegmentDataset,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_flat, y, seg_ids) from a dataset."""
        X_temporal = ds.sequences.numpy()  # (N, T, F_t)
        X_static = ds.static_features.numpy()  # (N, F_s)
        N, T, F_t = X_temporal.shape
        X_flat = np.concatenate([X_temporal.reshape(N, T * F_t), X_static], axis=1)
        return X_flat, ds.targets.numpy(), ds.segment_ids

    X_train, y_train, seg_ids_train = _extract_tabular(train_ds)
    X_val, y_val, _ = _extract_tabular(val_ds)
    X_test, y_test, _ = _extract_tabular(test_ds)

    # Combine train + val for ensemble fitting
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)
    seg_ids_fit = np.concatenate([seg_ids_train, val_ds.segment_ids], axis=0)

    # Patches for HydroFormer (optional)
    def _get_patches(ds: RiverSegmentDataset) -> Optional[np.ndarray]:
        p = ds.patches.numpy()
        if np.all(p == 0):
            return None
        return p

    patches_fit: Optional[np.ndarray] = None
    if pretrained_hydroformer is not None:
        p1 = _get_patches(train_ds)
        p2 = _get_patches(val_ds)
        if p1 is not None and p2 is not None:
            patches_fit = np.concatenate([p1, p2], axis=0)

    # ── EnsembleDepthEstimator ────────────────────────────────────────────
    logger.info("Fitting EnsembleDepthEstimator …")
    ens_config = EnsembleConfig(
        n_splits=config.ensemble_n_splits,
        lgb_n_estimators=config.lgb_n_estimators,
        xgb_n_estimators=config.xgb_n_estimators,
        conformal_alpha=config.conformal_alpha,
        use_hydroformer=(pretrained_hydroformer is not None),
        random_seed=config.seed,
    )

    ensemble = EnsembleDepthEstimator(
        config=ens_config,
        hydroformer_model=pretrained_hydroformer,
        device=device,
    )
    ensemble.fit(X_fit, y_fit, X_patches=patches_fit, segment_ids=seg_ids_fit)

    # ── Evaluate ensemble on test set ─────────────────────────────────────
    with wandb_run(config, job_type="train_ensemble") as run:
        test_patches = _get_patches(test_ds)
        ens_metrics = ensemble.evaluate(X_test, y_test, X_patches=test_patches)
        log_metrics(run, {f"ensemble_test/{k}": v for k, v in ens_metrics.items()})
        logger.info("Ensemble test metrics: %s", ens_metrics)

    # ── NavigabilityClassifier ────────────────────────────────────────────
    logger.info("Fitting NavigabilityClassifier …")

    # Build navigability feature matrix from ensemble predictions + extras
    def _build_nav_features(
        X: np.ndarray,
        depths: np.ndarray,
        ds: RiverSegmentDataset,
        X_patches: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build classifier input: [depth_pred, uncertainty, width, Q, sin]."""
        pred, lower, upper = ensemble.predict(X, X_patches=X_patches)
        uncertainty = upper - lower
        # Extract width and discharge from last few columns of X if available
        # We use the raw dataset widths from the feature matrix
        N = len(depths)
        # Attempt to extract key columns from static features
        # Columns: depth_pred, uncertainty, water_width_m, discharge, sinuosity
        width = (
            ds.static_features[:, 0].numpy()
            if ds.static_features.shape[1] > 0
            else np.zeros(N)
        )
        discharge = (
            ds.static_features[:, 1].numpy()
            if ds.static_features.shape[1] > 1
            else np.zeros(N)
        )
        sinuosity = (
            ds.static_features[:, 2].numpy()
            if ds.static_features.shape[1] > 2
            else np.ones(N)
        )

        return np.stack(
            [pred, uncertainty, width, discharge, sinuosity], axis=1
        ).astype(np.float32)

    X_nav_fit = _build_nav_features(X_fit, y_fit, train_ds, patches_fit)

    # Ground-truth navigability labels
    y_nav_fit = np.where(y_fit >= 3.0, 2, np.where(y_fit >= 2.0, 1, 0)).astype(np.int64)

    nav_config = NavigabilityConfig(
        n_estimators=500,
        calibration_method="isotonic",
        conformal_alpha=config.conformal_alpha,
        random_seed=config.seed,
    )
    nav_clf = NavigabilityClassifier(config=nav_config)
    nav_clf.fit(
        X_nav_fit,
        y_nav_fit,
        feature_names=NavigabilityClassifier.REQUIRED_FEATURES,
    )

    # Evaluate classifier
    X_nav_test = _build_nav_features(X_test, y_test, test_ds, test_patches)
    y_nav_test = np.where(y_test >= 3.0, 2, np.where(y_test >= 2.0, 1, 0)).astype(
        np.int64
    )

    clf_metrics = nav_clf.evaluate(X_nav_test, y_nav_test)
    logger.info("NavigabilityClassifier test metrics: %s", clf_metrics)

    with wandb_run(config, job_type="train_ensemble") as run:
        log_metrics(run, {f"nav_test/{k}": v for k, v in clf_metrics.items()})

    # ── Save artefacts ────────────────────────────────────────────────────
    ensemble.save(output_dir / "depth_ensemble")
    nav_clf.save(output_dir / "nav_classifier.pkl")

    # Save SHAP feature importance
    try:
        fi_df = ensemble.feature_importance_df("lgb")
        fi_df.to_csv(output_dir / "lgb_feature_importance.csv", index=False)

        nav_shap_df = nav_clf.shap_summary_df(X_nav_fit[:200], class_idx=2)
        if nav_shap_df is not None:
            nav_shap_df.to_csv(output_dir / "nav_shap_importance.csv", index=False)
    except Exception as exc:
        logger.warning("Could not save feature importance: %s", exc)

    logger.info("Full ensemble training complete. Artefacts saved to %s", output_dir)
    return ensemble, nav_clf


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


def evaluate_model(
    model: Union[HydroFormer, EnsembleDepthEstimator],
    test_data: RiverSegmentDataset,
    nav_clf: Optional[NavigabilityClassifier] = None,
    device: Optional[torch.device] = None,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, float]:
    """Compute a comprehensive evaluation metric suite.

    Computes:
        - Regression: R², RMSE, MAE
        - Uncertainty: PI coverage (90%), mean interval width
        - Classification (if nav_clf provided): accuracy, macro-F1,
          per-class precision / recall / F1

    Parameters
    ----------
    model:
        A trained :class:`HydroFormer` or :class:`EnsembleDepthEstimator`.
    test_data:
        A :class:`RiverSegmentDataset` test split.
    nav_clf:
        Optional :class:`NavigabilityClassifier` for classification metrics.
    device:
        Torch device (auto-selected if None).
    config:
        Optional :class:`TrainingConfig` for batch size / workers.

    Returns
    -------
    Dict[str, float]
        All computed metrics as a flat dict.
    """
    device = device or get_device()
    cfg = config or TrainingConfig()

    # ── Extract ground truth ──────────────────────────────────────────────
    y_true = test_data.targets.numpy()
    nav_true = test_data.nav_labels.numpy()
    N = len(y_true)

    # ── Run inference ─────────────────────────────────────────────────────
    if isinstance(model, HydroFormer):
        loader = DataLoader(
            test_data,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            num_workers=cfg.n_workers,
        )
        preds_list, lower_list, upper_list = [], [], []
        model.eval()
        model.to(device)

        with torch.no_grad():
            for batch in loader:
                xs = batch["x_static"].to(device)
                xt = batch["x_temporal"].to(device)
                xp = batch["x_patch"].to(device) if cfg.use_swin else None
                depth, lower, upper = model(xs, xt, xp)
                preds_list.append(depth.cpu().numpy())
                lower_list.append(lower.cpu().numpy())
                upper_list.append(upper.cpu().numpy())

        y_pred = np.concatenate(preds_list)
        lower_ci = np.concatenate(lower_list)
        upper_ci = np.concatenate(upper_list)

    elif isinstance(model, EnsembleDepthEstimator):
        X_temporal = test_data.sequences.numpy()
        X_static = test_data.static_features.numpy()
        N_, T, Ft = X_temporal.shape
        X_flat = np.concatenate([X_temporal.reshape(N_, T * Ft), X_static], axis=1)
        patches = test_data.patches.numpy()
        if np.all(patches == 0):
            patches = None
        y_pred, lower_ci, upper_ci = model.predict(X_flat, X_patches=patches)

    else:
        raise TypeError(
            f"Unsupported model type: {type(model).__name__}. "
            "Expected HydroFormer or EnsembleDepthEstimator."
        )

    # ── Regression metrics ────────────────────────────────────────────────
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    coverage = float(np.mean((y_true >= lower_ci) & (y_true <= upper_ci)))
    interval_width = float(np.mean(upper_ci - lower_ci))
    bias = float(np.mean(y_pred - y_true))

    metrics: Dict[str, float] = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "coverage_90": coverage,
        "mean_interval_width": interval_width,
    }

    logger.info(
        "Regression | R²=%.4f  RMSE=%.4f m  MAE=%.4f m  "
        "Bias=%.4f m  Coverage=%.3f  Width=%.4f m",
        r2,
        rmse,
        mae,
        bias,
        coverage,
        interval_width,
    )

    # ── Classification metrics ─────────────────────────────────────────────
    if nav_clf is not None:
        # Build classifier features from predictions
        static = test_data.static_features.numpy()
        width = static[:, 0] if static.shape[1] > 0 else np.zeros(N)
        discharge = static[:, 1] if static.shape[1] > 1 else np.zeros(N)
        sinuosity = static[:, 2] if static.shape[1] > 2 else np.ones(N)
        uncertainty = upper_ci - lower_ci

        X_nav = np.stack(
            [y_pred, uncertainty, width, discharge, sinuosity], axis=1
        ).astype(np.float32)

        nav_pred = nav_clf.predict(X_nav)

        acc = float(accuracy_score(nav_true, nav_pred))
        f1_mac = float(f1_score(nav_true, nav_pred, average="macro", zero_division=0))
        prec_mac = float(
            precision_score(nav_true, nav_pred, average="macro", zero_division=0)
        )
        rec_mac = float(
            recall_score(nav_true, nav_pred, average="macro", zero_division=0)
        )
        f1_per = f1_score(nav_true, nav_pred, average=None, zero_division=0)
        prec_per = precision_score(nav_true, nav_pred, average=None, zero_division=0)
        rec_per = recall_score(nav_true, nav_pred, average=None, zero_division=0)

        metrics.update(
            {
                "nav_accuracy": acc,
                "nav_f1_macro": f1_mac,
                "nav_precision_macro": prec_mac,
                "nav_recall_macro": rec_mac,
            }
        )

        class_names = ["non_navigable", "conditional", "navigable"]
        for i, cname in enumerate(class_names):
            if i < len(f1_per):
                metrics[f"nav_f1_{cname}"] = float(f1_per[i])
                metrics[f"nav_prec_{cname}"] = float(prec_per[i])
                metrics[f"nav_rec_{cname}"] = float(rec_per[i])

        logger.info(
            "Navigation | Acc=%.4f  F1=%.4f  Prec=%.4f  Recall=%.4f",
            acc,
            f1_mac,
            prec_mac,
            rec_mac,
        )

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python train.py",
        description="AIDSTL ML training CLI — inland waterway navigability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Train HydroFormer only
python train.py hydroformer --data_path data/features.parquet \\
    --output_dir outputs/ --max_epochs 100 --use_wandb

# Train full ensemble (HydroFormer → Ensemble → Classifier)
python train.py ensemble --data_path data/features.parquet \\
    --output_dir outputs/ --lgb_n_estimators 500

# Evaluate a saved model
python train.py evaluate \\
    --model_path outputs/hydroformer/checkpoints/best.pt \\
    --data_path data/features.parquet

# Load config from JSON
python train.py hydroformer --config configs/my_config.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Shared arguments ──────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON TrainingConfig file. CLI flags override JSON values.",
    )
    shared.add_argument("--data_path", type=str)
    shared.add_argument("--patch_dir", type=str)
    shared.add_argument("--norm_stats_path", type=str)
    shared.add_argument("--output_dir", type=str)
    shared.add_argument("--seed", type=int)
    shared.add_argument("--n_workers", type=int)
    shared.add_argument("--use_wandb", action="store_true")
    shared.add_argument("--wandb_project", type=str)
    shared.add_argument("--wandb_run_name", type=str)
    shared.add_argument("--no_mixed_precision", action="store_true")
    shared.add_argument(
        "--no_swin", action="store_true", help="Disable the Swin-T encoder."
    )

    # ── hydroformer sub-command ───────────────────────────────────────────
    hf_parser = subparsers.add_parser(
        "hydroformer",
        parents=[shared],
        help="Train the HydroFormer (Swin-T + TFT).",
    )
    hf_parser.add_argument("--batch_size", type=int)
    hf_parser.add_argument("--learning_rate", type=float)
    hf_parser.add_argument("--max_epochs", type=int)
    hf_parser.add_argument("--warmup_epochs", type=int)
    hf_parser.add_argument("--early_stopping_patience", type=int)
    hf_parser.add_argument("--d_model", type=int)
    hf_parser.add_argument("--n_heads", type=int)
    hf_parser.add_argument("--lstm_hidden", type=int)
    hf_parser.add_argument("--lstm_layers", type=int)
    hf_parser.add_argument("--dropout", type=float)
    hf_parser.add_argument("--gradient_clip_norm", type=float)
    hf_parser.add_argument("--accumulation_steps", type=int)
    hf_parser.add_argument("--save_top_k", type=int)

    # ── ensemble sub-command ─────────────────────────────────────────────
    ens_parser = subparsers.add_parser(
        "ensemble",
        parents=[shared],
        help="Train the full stacking ensemble.",
    )
    ens_parser.add_argument("--ensemble_n_splits", type=int)
    ens_parser.add_argument("--lgb_n_estimators", type=int)
    ens_parser.add_argument("--xgb_n_estimators", type=int)
    ens_parser.add_argument("--conformal_alpha", type=float)
    ens_parser.add_argument(
        "--hydroformer_checkpoint",
        type=str,
        default=None,
        help="Path to a pre-trained HydroFormer .pt file. "
        "If not provided, HydroFormer will be trained first.",
    )
    ens_parser.add_argument(
        "--skip_hydroformer_training",
        action="store_true",
        help="Skip HydroFormer training (requires --hydroformer_checkpoint).",
    )

    # ── evaluate sub-command ─────────────────────────────────────────────
    ev_parser = subparsers.add_parser(
        "evaluate",
        parents=[shared],
        help="Evaluate a trained model on the test split.",
    )
    ev_parser.add_argument(
        "--model_type", choices=["hydroformer", "ensemble"], default="ensemble"
    )
    ev_parser.add_argument(
        "--model_path", type=str, help="Path to HydroFormer .pt checkpoint."
    )
    ev_parser.add_argument(
        "--ensemble_dir", type=str, help="Directory of saved EnsembleDepthEstimator."
    )
    ev_parser.add_argument(
        "--nav_clf_path", type=str, help="Path to saved NavigabilityClassifier .pkl."
    )
    ev_parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Save metrics dict to this JSON file.",
    )

    return parser


def _override_config(
    config: TrainingConfig,
    args: argparse.Namespace,
) -> TrainingConfig:
    """Override TrainingConfig fields with non-None CLI args."""
    overrides: Dict[str, Any] = {}
    # Map argparse fields to config fields
    field_map = {
        "data_path": "data_path",
        "patch_dir": "patch_dir",
        "norm_stats_path": "norm_stats_path",
        "output_dir": "output_dir",
        "seed": "seed",
        "n_workers": "n_workers",
        "use_wandb": "use_wandb",
        "wandb_project": "wandb_project",
        "wandb_run_name": "wandb_run_name",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "max_epochs": "max_epochs",
        "warmup_epochs": "warmup_epochs",
        "early_stopping_patience": "early_stopping_patience",
        "d_model": "d_model",
        "n_heads": "n_heads",
        "lstm_hidden": "lstm_hidden",
        "lstm_layers": "lstm_layers",
        "dropout": "dropout",
        "gradient_clip_norm": "gradient_clip_norm",
        "accumulation_steps": "accumulation_steps",
        "save_top_k": "save_top_k",
        "ensemble_n_splits": "ensemble_n_splits",
        "lgb_n_estimators": "lgb_n_estimators",
        "xgb_n_estimators": "xgb_n_estimators",
        "conformal_alpha": "conformal_alpha",
    }
    for arg_key, cfg_key in field_map.items():
        val = getattr(args, arg_key, None)
        if val is not None:
            overrides[cfg_key] = val

    # Boolean flags
    if getattr(args, "no_mixed_precision", False):
        overrides["mixed_precision"] = False
    if getattr(args, "no_swin", False):
        overrides["use_swin"] = False

    if overrides:
        cfg_dict = config.to_dict()
        cfg_dict.update(overrides)
        config = TrainingConfig(**cfg_dict)
    return config


def _load_hydroformer_from_checkpoint(
    checkpoint_path: str,
    config: TrainingConfig,
    device: torch.device,
) -> HydroFormer:
    """Load HydroFormer weights from a .pt checkpoint file."""
    model = build_hydroformer(
        n_temporal_features=config.n_temporal_features,
        n_static_features=config.n_static_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers,
        swin_embed_dim=config.swin_embed_dim,
        dropout=config.dropout,
        patch_size=config.patch_size,
        quantiles=config.quantiles,
        use_swin=config.use_swin,
    )
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Loaded HydroFormer from %s", checkpoint_path)
    return model


def main() -> None:
    """CLI entry point."""
    import json
    import sys

    parser = _build_arg_parser()
    args = parser.parse_args()

    # ── Load base config (from JSON if provided) ─────────────────────────
    if args.config:
        config = TrainingConfig.from_json(args.config)
        logger.info("Loaded config from %s", args.config)
    else:
        config = TrainingConfig()

    # ── Apply CLI overrides ───────────────────────────────────────────────
    config = _override_config(config, args)

    logger.info("Running command: %s | output_dir: %s", args.command, config.output_dir)

    device = get_device()

    # ── hydroformer ───────────────────────────────────────────────────────
    if args.command == "hydroformer":
        model = train_hydroformer(config)
        # Quick evaluation on test set
        dataset = load_dataset(config)
        _, _, test_ds = split_dataset(dataset, config)
        metrics = evaluate_model(model, test_ds, device=device, config=config)
        logger.info("Test metrics: %s", metrics)

    # ── ensemble ──────────────────────────────────────────────────────────
    elif args.command == "ensemble":
        pretrained_hf: Optional[HydroFormer] = None

        if getattr(args, "hydroformer_checkpoint", None):
            pretrained_hf = _load_hydroformer_from_checkpoint(
                args.hydroformer_checkpoint, config, device
            )
        elif not getattr(args, "skip_hydroformer_training", False):
            logger.info("No HydroFormer checkpoint provided — training now …")
            pretrained_hf = train_hydroformer(config)

        ensemble, nav_clf = train_full_ensemble(
            config, pretrained_hydroformer=pretrained_hf
        )

        # Final evaluation
        dataset = load_dataset(config)
        _, _, test_ds = split_dataset(dataset, config)
        metrics = evaluate_model(
            ensemble, test_ds, nav_clf=nav_clf, device=device, config=config
        )
        logger.info("Final test metrics: %s", metrics)

        # Save metrics
        metrics_path = Path(config.output_dir) / "ensemble" / "test_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", metrics_path)

    # ── evaluate ──────────────────────────────────────────────────────────
    elif args.command == "evaluate":
        dataset = load_dataset(config)
        _, _, test_ds = split_dataset(dataset, config)

        nav_clf: Optional[NavigabilityClassifier] = None
        if getattr(args, "nav_clf_path", None):
            nav_clf = NavigabilityClassifier.load(args.nav_clf_path)

        if args.model_type == "hydroformer":
            if not args.model_path:
                parser.error("--model_path is required for model_type=hydroformer")
            model = _load_hydroformer_from_checkpoint(args.model_path, config, device)
            metrics = evaluate_model(
                model, test_ds, nav_clf=nav_clf, device=device, config=config
            )
        else:
            if not getattr(args, "ensemble_dir", None):
                parser.error("--ensemble_dir is required for model_type=ensemble")
            ensemble = EnsembleDepthEstimator.load(args.ensemble_dir)
            metrics = evaluate_model(
                ensemble, test_ds, nav_clf=nav_clf, device=device, config=config
            )

        logger.info("Evaluation metrics:\n%s", json.dumps(metrics, indent=2))

        if getattr(args, "output_json", None):
            with open(args.output_json, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info("Metrics saved to %s", args.output_json)

    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
