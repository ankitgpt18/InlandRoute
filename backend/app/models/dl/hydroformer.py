"""
hydroformer.py
==============
Deep-learning backbone for the AIDSTL inland-waterway navigability project.

Architecture overview
---------------------
SwinSpectralEncoder
    Swin-Tiny Transformer backbone (timm) adapted for 12-channel Sentinel-2
    input.  Outputs a 64-dim spatial embedding per river-segment patch.

HydroForecastTFT
    Temporal Fusion Transformer for multi-horizon depth forecasting.
    Inputs:
      x_static   (B, F_s)     – time-invariant segment features
      x_temporal (B, T, F_t)  – time-series features (T = 12 months)
    Outputs:
      depth_pred (B,)          – median (50th-percentile) depth estimate
      q10        (B,)          – 10th-percentile lower bound
      q90        (B,)          – 90th-percentile upper bound

HydroFormer
    End-to-end model that fuses SwinSpectralEncoder spatial embeddings with
    HydroForecastTFT temporal outputs via cross-attention, then produces
    a final depth estimate with calibrated uncertainty bounds.

Loss functions
--------------
QuantileLoss   – Pinball / check loss for the three output quantiles.
HydroFormerLoss – Combined quantile + MSE loss with configurable weights.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

D_MODEL: int = 128  # core hidden dimension for TFT
N_HEADS: int = 8  # multi-head attention heads
LSTM_HIDDEN: int = 128  # LSTM hidden state size
LSTM_LAYERS: int = 2  # stacked LSTM layers
SWIN_EMBED_DIM: int = 64  # output dimension of SwinSpectralEncoder
QUANTILES: List[float] = [0.10, 0.50, 0.90]  # q10, q50, q90


# ---------------------------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------------------------


class GLU(nn.Module):
    """Gated Linear Unit.

    Splits the last dimension in half; the second half gates the first.
    output = sigmoid(x_gate) ⊙ x_signal
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc(x)
        signal, gate = out.chunk(2, dim=-1)
        return torch.sigmoid(gate) * signal


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) block from the TFT paper.

    GRN(a, c) = LayerNorm(a + GLU(η₁))
      η₁ = W₁·η₂ + b₁
      η₂ = ELU(W₂·a + W₃·c + b₂)   (c is optional context vector)

    Parameters
    ----------
    input_dim:
        Dimension of primary input *a*.
    hidden_dim:
        Dimension of the intermediate dense layer.
    output_dim:
        Dimension of the output.  If different from *input_dim* a linear
        projection is used for the residual.
    dropout:
        Dropout rate applied to η₂ before the GLU.
    context_dim:
        Dimension of optional context input *c*.  0 means no context.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(
            input_dim + (context_dim if context_dim > 0 else 0), hidden_dim
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GLU(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Residual projection when dims differ
        self.residual_proj: Optional[nn.Linear] = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim
            else None
        )

    def forward(self, a: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        a : (*, input_dim)
        c : (*, context_dim) optional context

        Returns
        -------
        Tensor (*, output_dim)
        """
        if c is not None:
            inp = torch.cat([a, c], dim=-1)
        else:
            inp = a

        eta2 = F.elu(self.fc1(inp))
        eta2 = self.dropout(eta2)
        eta1 = self.fc2(eta2)
        gated = self.glu(eta1)

        residual = a if self.residual_proj is None else self.residual_proj(a)
        return self.layer_norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) from the TFT paper.

    Learns soft variable weights via a GRN + softmax gate, then produces
    a weighted sum of per-variable GRN embeddings.

    Parameters
    ----------
    n_features:
        Number of input variables (F).
    d_model:
        Target embedding dimension for each variable.
    dropout:
        Dropout rate for internal GRNs.
    context_dim:
        Dimension of optional static context.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        dropout: float = 0.1,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Per-variable single-variable GRNs
        self.var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(1, d_model, d_model, dropout=dropout)
                for _ in range(n_features)
            ]
        )

        # Flat GRN to produce variable weights
        self.flat_grn = GatedResidualNetwork(
            input_dim=n_features,
            hidden_dim=d_model,
            output_dim=n_features,
            dropout=dropout,
            context_dim=context_dim,
        )

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : (B, F) or (B, T, F)
        context : (B, context_dim) optional

        Returns
        -------
        combined : (B, d_model) or (B, T, d_model)
        weights  : (B, F) or (B, T, F)
        """
        # Handle both 2-D (static) and 3-D (temporal) inputs
        if x.dim() == 2:
            return self._forward_2d(x, context)
        return self._forward_3d(x, context)

    def _forward_2d(
        self, x: Tensor, context: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        B, F = x.shape
        # Per-variable embeddings: (B, F, d_model)
        var_embs = torch.stack(
            [self.var_grns[i](x[:, i : i + 1]) for i in range(F)], dim=1
        )
        # Selection weights
        flat_weights = self.flat_grn(x, context)  # (B, F)
        weights = torch.softmax(flat_weights, dim=-1)  # (B, F)
        # Weighted sum: (B, d_model)
        combined = torch.einsum("bf,bfd->bd", weights, var_embs)
        return combined, weights

    def _forward_3d(
        self, x: Tensor, context: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        B, T, F = x.shape
        # Reshape to (B*T, F) for processing
        x_flat = x.reshape(B * T, F)
        ctx = (
            context.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            if context is not None
            else None
        )
        combined_flat, weights_flat = self._forward_2d(x_flat, ctx)
        combined = combined_flat.reshape(B, T, self.d_model)
        weights = weights_flat.reshape(B, T, F)
        return combined, weights


# ---------------------------------------------------------------------------
# SwinSpectralEncoder
# ---------------------------------------------------------------------------


class SwinSpectralEncoder(nn.Module):
    """Swin Transformer spatial encoder for 12-channel Sentinel-2 patches.

    Uses ``swin_tiny_patch4_window7_224`` backbone from *timm* with
    ``in_chans=12`` (no pre-trained weights).  A projection MLP maps the
    backbone's 768-dim feature vector down to *embed_dim* (default 64).

    Parameters
    ----------
    embed_dim:
        Output embedding dimension.  Default 64.
    dropout:
        Dropout applied after the projection head.  Default 0.1.
    pretrained:
        Whether to load ImageNet pre-trained weights.  Must be False for
        12-channel input (not compatible with standard ImageNet weights).
    img_size:
        Spatial size of the input patch (H = W).  Default 224.
    """

    def __init__(
        self,
        embed_dim: int = SWIN_EMBED_DIM,
        dropout: float = 0.1,
        pretrained: bool = False,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Build Swin-Tiny backbone for 12-channel input
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            in_chans=12,
            num_classes=0,  # remove classification head → returns feature map
            img_size=img_size,
        )
        backbone_out_dim: int = self.backbone.num_features  # 768 for swin_tiny

        # Projection head: 768 → 256 → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        logger.info(
            "SwinSpectralEncoder | backbone_out=%d → embed_dim=%d",
            backbone_out_dim,
            embed_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, 12, H, W)  float32 surface-reflectance patches

        Returns
        -------
        Tensor (B, embed_dim)
        """
        if x.shape[1] != 12:
            raise ValueError(f"Expected 12 input channels, got {x.shape[1]}.")
        features = self.backbone(x)  # (B, 768)
        return self.projection(features)  # (B, embed_dim)


# ---------------------------------------------------------------------------
# HydroForecastTFT
# ---------------------------------------------------------------------------


class HydroForecastTFT(nn.Module):
    """Temporal Fusion Transformer for river-depth forecasting.

    Architecture
    ------------
    1. Static Variable Selection Network (VSN)  →  static_context (B, d_model)
    2. Temporal Variable Selection Network (VSN) → selected_temporal (B, T, d_model)
    3. LSTM Encoder   (bidirectional, 2 layers)  →  lstm_out (B, T, d_model)
    4. Post-LSTM GRN with static context
    5. Multi-Head Self-Attention (8 heads)       →  attn_out (B, T, d_model)
    6. GRN → feed-forward
    7. Temporal pooling (mean over T)
    8. Quantile head  →  (q10, q50, q90)

    Parameters
    ----------
    n_temporal_features:
        Number of time-varying input features (F_t).
    n_static_features:
        Number of static input features (F_s).
    d_model:
        Core hidden dimension.  Default 128.
    n_heads:
        Number of attention heads.  Default 8.
    lstm_hidden:
        LSTM hidden-state dimension.  Default 128.
    lstm_layers:
        Number of stacked LSTM layers.  Default 2.
    dropout:
        Dropout rate.  Default 0.1.
    quantiles:
        Output quantile levels.  Default [0.10, 0.50, 0.90].
    """

    def __init__(
        self,
        n_temporal_features: int,
        n_static_features: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        dropout: float = 0.1,
        quantiles: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.quantiles = quantiles or QUANTILES
        self.n_quantiles = len(self.quantiles)

        # ── Variable Selection Networks ─────────────────────────────────
        self.static_vsn = VariableSelectionNetwork(
            n_features=n_static_features,
            d_model=d_model,
            dropout=dropout,
        )
        self.temporal_vsn = VariableSelectionNetwork(
            n_features=n_temporal_features,
            d_model=d_model,
            dropout=dropout,
            context_dim=d_model,  # conditioned on static context
        )

        # ── Static context encoders ─────────────────────────────────────
        # Four context vectors: enrichment, initial cell/hidden state, query
        self.static_enrichment_grn = GatedResidualNetwork(
            d_model, d_model, d_model, dropout=dropout
        )
        self.static_h_grn = GatedResidualNetwork(
            d_model, d_model, lstm_hidden, dropout=dropout
        )
        self.static_c_grn = GatedResidualNetwork(
            d_model, d_model, lstm_hidden, dropout=dropout
        )

        # ── LSTM Encoder ────────────────────────────────────────────────
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.lstm_proj = nn.Linear(lstm_hidden, d_model)

        # ── Post-LSTM gate ──────────────────────────────────────────────
        self.post_lstm_gate = GLU(d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)
        self.post_lstm_grn = GatedResidualNetwork(
            d_model, d_model, d_model, dropout=dropout, context_dim=d_model
        )

        # ── Multi-Head Self-Attention ───────────────────────────────────
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_gate = GLU(d_model)
        self.attn_norm = nn.LayerNorm(d_model)

        # ── Position-wise Feed-Forward GRN ──────────────────────────────
        self.ff_grn = GatedResidualNetwork(
            d_model, d_model * 4, d_model, dropout=dropout
        )

        # ── Output normalisation & quantile head ────────────────────────
        self.output_norm = nn.LayerNorm(d_model)
        self.quantile_head = nn.Linear(d_model, self.n_quantiles)

        self.dropout = nn.Dropout(dropout)

        logger.info(
            "HydroForecastTFT | n_temporal=%d, n_static=%d, "
            "d_model=%d, heads=%d, lstm=%dx%d, quantiles=%s",
            n_temporal_features,
            n_static_features,
            d_model,
            n_heads,
            lstm_layers,
            lstm_hidden,
            self.quantiles,
        )

    # ------------------------------------------------------------------

    def _init_lstm_state(
        self, static_context: Tensor, batch_size: int
    ) -> Tuple[Tensor, Tensor]:
        """Derive LSTM initial (h_0, c_0) from static context."""
        # static_context: (B, d_model)
        h0 = self.static_h_grn(static_context)  # (B, lstm_hidden)
        c0 = self.static_c_grn(static_context)  # (B, lstm_hidden)
        # LSTM expects (num_layers, B, hidden)
        h0 = h0.unsqueeze(0).expand(self.lstm_encoder.num_layers, -1, -1).contiguous()
        c0 = c0.unsqueeze(0).expand(self.lstm_encoder.num_layers, -1, -1).contiguous()
        return h0, c0

    # ------------------------------------------------------------------

    def forward(
        self,
        x_static: Tensor,
        x_temporal: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x_static   : (B, F_s)    time-invariant segment features
        x_temporal : (B, T, F_t) time-series features

        Returns
        -------
        depth_pred : (B,)   median depth estimate
        q10        : (B,)   10th-percentile lower bound
        q90        : (B,)   90th-percentile upper bound
        """
        B, T, _ = x_temporal.shape

        # ── 1. Static VSN ───────────────────────────────────────────────
        static_emb, static_weights = self.static_vsn(x_static)  # (B, d_model)
        static_context = self.static_enrichment_grn(static_emb)  # (B, d_model)

        # ── 2. Temporal VSN conditioned on static context ───────────────
        temporal_emb, temporal_weights = self.temporal_vsn(
            x_temporal, context=static_context
        )  # (B, T, d_model)

        # ── 3. LSTM encoder ─────────────────────────────────────────────
        h0, c0 = self._init_lstm_state(static_context, B)
        lstm_out, _ = self.lstm_encoder(temporal_emb, (h0, c0))  # (B, T, lstm_hidden)
        lstm_out = self.lstm_proj(lstm_out)  # (B, T, d_model)

        # Post-LSTM gated skip connection
        gated = self.post_lstm_gate(lstm_out)
        lstm_out = self.post_lstm_norm(temporal_emb + gated)

        # Enrich with static context (broadcast over T)
        ctx_expanded = static_context.unsqueeze(1).expand(-1, T, -1)
        lstm_out = self.post_lstm_grn(lstm_out, ctx_expanded)  # (B, T, d_model)

        # ── 4. Multi-Head Self-Attention ────────────────────────────────
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)  # (B, T, d_model)
        gated_attn = self.attn_gate(attn_out)
        attn_out = self.attn_norm(lstm_out + gated_attn)  # (B, T, d_model)

        # ── 5. Position-wise feed-forward ───────────────────────────────
        ff_out = self.ff_grn(attn_out)  # (B, T, d_model)
        ff_out = self.output_norm(attn_out + ff_out)  # residual

        # ── 6. Temporal pooling → quantile head ─────────────────────────
        pooled = ff_out.mean(dim=1)  # (B, d_model)
        quantiles = self.quantile_head(pooled)  # (B, n_quantiles)

        # Enforce monotonicity: q10 ≤ q50 ≤ q90 via cumsum on sorted deltas
        # We use a soft ordering trick
        q_sorted = quantiles + torch.relu(
            torch.cumsum(
                torch.cat(
                    [
                        torch.zeros(B, 1, device=quantiles.device),
                        quantiles.diff(dim=-1),
                    ],
                    dim=-1,
                ),
                dim=-1,
            )
            - quantiles.diff(dim=-1, prepend=quantiles[:, :1]).clamp(min=0)
        )
        # Simple: just sort each row to guarantee order
        q_sorted, _ = quantiles.sort(dim=-1)  # (B, 3) sorted ascending

        q10 = q_sorted[:, 0]
        q50 = q_sorted[:, 1]
        q90 = q_sorted[:, 2]

        return q50, q10, q90

    # ------------------------------------------------------------------

    def predict_with_uncertainty(
        self,
        x_static: Tensor,
        x_temporal: Tensor,
    ) -> Dict[str, Tensor]:
        """Convenience wrapper returning a labelled dict."""
        q50, q10, q90 = self.forward(x_static, x_temporal)
        return {
            "depth_pred": q50,
            "q10": q10,
            "q90": q90,
            "uncertainty_range": q90 - q10,
        }


# ---------------------------------------------------------------------------
# Cross-modal fusion attention
# ---------------------------------------------------------------------------


class CrossModalAttentionFusion(nn.Module):
    """Fuse TFT temporal output with Swin-T spatial embedding via cross-attention.

    The TFT output acts as the *query* and the spatial embedding (tiled to
    match T time-steps) acts as the *key* and *value*.

    Parameters
    ----------
    d_model:
        Hidden dimension of both modalities (after projection if needed).
    n_heads:
        Number of attention heads.
    spatial_dim:
        Dimension of the Swin-T embedding (will be projected to d_model).
    dropout:
        Attention dropout.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        spatial_dim: int = SWIN_EMBED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_grn = GatedResidualNetwork(
            d_model, d_model * 2, d_model, dropout=dropout
        )

    def forward(self, tft_feat: Tensor, swin_feat: Tensor) -> Tensor:
        """
        Parameters
        ----------
        tft_feat  : (B, d_model)  – pooled TFT representation
        swin_feat : (B, spatial_dim) – Swin-T embedding

        Returns
        -------
        Tensor (B, d_model)
        """
        B = tft_feat.size(0)
        # Project spatial to d_model, add sequence dim → (B, 1, d_model)
        swin_proj = self.spatial_proj(swin_feat).unsqueeze(1)
        query = tft_feat.unsqueeze(1)  # (B, 1, d_model)

        # Cross-attention: query=TFT, key/value=Swin
        attn_out, _ = self.cross_attn(query, swin_proj, swin_proj)  # (B, 1, d_model)
        attn_out = attn_out.squeeze(1)  # (B, d_model)

        # Residual + GRN
        fused = self.norm(tft_feat + attn_out)
        return self.output_grn(fused)


# ---------------------------------------------------------------------------
# HydroFormer
# ---------------------------------------------------------------------------


class HydroFormer(nn.Module):
    """Full multi-modal ensemble model for river depth estimation.

    Takes satellite image patches (Sentinel-2, 12 bands) **and** temporal
    feature sequences as input, fuses them via cross-modal attention, and
    produces a depth estimate with calibrated uncertainty bounds.

    Architecture
    ------------
    SwinSpectralEncoder:
        (B, 12, H, W) → (B, 64)  spatial embedding

    HydroForecastTFT:
        (B, F_s), (B, T, F_t) → (q50, q10, q90) each (B,)

    CrossModalAttentionFusion:
        (B, D_tft) × (B, 64) → (B, D_tft) fused representation

    Final head:
        (B, D_tft + 1) → depth_mean, depth_lower, depth_upper

    Parameters
    ----------
    n_temporal_features:
        Time-varying feature count for TFT.
    n_static_features:
        Static feature count for TFT.
    d_model:
        Core hidden dimension.
    n_heads:
        Attention heads.
    lstm_hidden:
        LSTM hidden size.
    lstm_layers:
        Number of LSTM layers.
    swin_embed_dim:
        Output dim of SwinSpectralEncoder.
    dropout:
        Dropout rate throughout.
    patch_size:
        Spatial size of satellite patches fed to Swin-T.
    quantiles:
        Quantile levels for output uncertainty.
    use_swin:
        If False, skips the Swin encoder (useful when patches are unavailable).
    """

    def __init__(
        self,
        n_temporal_features: int,
        n_static_features: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        swin_embed_dim: int = SWIN_EMBED_DIM,
        dropout: float = 0.1,
        patch_size: int = 224,
        quantiles: Optional[List[float]] = None,
        use_swin: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.swin_embed_dim = swin_embed_dim
        self.quantiles = quantiles or QUANTILES
        self.use_swin = use_swin

        # ── Sub-modules ─────────────────────────────────────────────────
        if use_swin:
            self.swin_encoder = SwinSpectralEncoder(
                embed_dim=swin_embed_dim,
                dropout=dropout,
                pretrained=False,
                img_size=patch_size,
            )
            self.fusion = CrossModalAttentionFusion(
                d_model=d_model,
                n_heads=n_heads,
                spatial_dim=swin_embed_dim,
                dropout=dropout,
            )

        self.tft = HydroForecastTFT(
            n_temporal_features=n_temporal_features,
            n_static_features=n_static_features,
            d_model=d_model,
            n_heads=n_heads,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            quantiles=self.quantiles,
        )

        # ── Depth uncertainty head ───────────────────────────────────────
        # Takes pooled fused representation + TFT median estimate
        self.depth_head = nn.Sequential(
            nn.Linear(d_model + 1, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # → (depth_mean, delta_lower, delta_upper)
        )

        # ── Internal TFT pooling for fusion ─────────────────────────────
        # We need the full TFT hidden repr. for fusion, not just scalars.
        # We hook into TFT by re-exposing its pooled feature.
        self._tft_pooled: Optional[Tensor] = None

        logger.info(
            "HydroFormer | use_swin=%s, d_model=%d, swin_embed=%d",
            use_swin,
            d_model,
            swin_embed_dim,
        )

    # ------------------------------------------------------------------

    def _get_tft_representation(
        self,
        x_static: Tensor,
        x_temporal: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the TFT and also return the internal pooled representation.

        We re-implement the pooling step here to get the intermediate
        feature vector for cross-modal fusion.

        Returns
        -------
        q50, q10, q90 : (B,) depth quantiles
        pooled        : (B, d_model) internal TFT representation
        """
        tft = self.tft
        B, T, _ = x_temporal.shape

        # ── VSN ──────────────────────────────────────────────────────────
        static_emb, _ = tft.static_vsn(x_static)
        static_context = tft.static_enrichment_grn(static_emb)
        temporal_emb, _ = tft.temporal_vsn(x_temporal, context=static_context)

        # ── LSTM ─────────────────────────────────────────────────────────
        h0, c0 = tft._init_lstm_state(static_context, B)
        lstm_out, _ = tft.lstm_encoder(temporal_emb, (h0, c0))
        lstm_out = tft.lstm_proj(lstm_out)
        gated = tft.post_lstm_gate(lstm_out)
        lstm_out = tft.post_lstm_norm(temporal_emb + gated)
        ctx_expanded = static_context.unsqueeze(1).expand(-1, T, -1)
        lstm_out = tft.post_lstm_grn(lstm_out, ctx_expanded)

        # ── Attention ────────────────────────────────────────────────────
        attn_out, _ = tft.self_attn(lstm_out, lstm_out, lstm_out)
        gated_attn = tft.attn_gate(attn_out)
        attn_out = tft.attn_norm(lstm_out + gated_attn)
        ff_out = tft.ff_grn(attn_out)
        ff_out = tft.output_norm(attn_out + ff_out)

        # ── Pooling ──────────────────────────────────────────────────────
        pooled = ff_out.mean(dim=1)  # (B, d_model)

        # ── Quantile head ────────────────────────────────────────────────
        quantiles = tft.quantile_head(pooled)
        q_sorted, _ = quantiles.sort(dim=-1)
        q10 = q_sorted[:, 0]
        q50 = q_sorted[:, 1]
        q90 = q_sorted[:, 2]

        return q50, q10, q90, pooled

    # ------------------------------------------------------------------

    def forward(
        self,
        x_static: Tensor,
        x_temporal: Tensor,
        x_patch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x_static   : (B, F_s)       static segment features
        x_temporal : (B, T, F_t)    temporal feature sequences
        x_patch    : (B, 12, H, W)  Sentinel-2 image patches (optional)

        Returns
        -------
        depth_pred : (B,)  final depth estimate
        lower_ci   : (B,)  lower 90% prediction bound
        upper_ci   : (B,)  upper 90% prediction bound
        """
        # ── TFT stream ───────────────────────────────────────────────────
        q50, q10, q90, tft_pooled = self._get_tft_representation(x_static, x_temporal)

        # ── Swin stream + fusion ─────────────────────────────────────────
        if self.use_swin and x_patch is not None:
            swin_emb = self.swin_encoder(x_patch)  # (B, swin_embed_dim)
            fused = self.fusion(tft_pooled, swin_emb)  # (B, d_model)
        else:
            fused = tft_pooled  # fallback: TFT only

        # ── Final depth head ─────────────────────────────────────────────
        # Concatenate fused repr + TFT median estimate
        inp = torch.cat([fused, q50.unsqueeze(-1)], dim=-1)  # (B, d_model+1)
        head_out = self.depth_head(inp)  # (B, 3)

        depth_mean = head_out[:, 0] + q50  # residual on q50
        # Positive offsets for CI bounds
        delta_lower = F.softplus(head_out[:, 1])
        delta_upper = F.softplus(head_out[:, 2])

        lower_ci = depth_mean - delta_lower
        upper_ci = depth_mean + delta_upper

        return depth_mean, lower_ci, upper_ci

    # ------------------------------------------------------------------

    def predict(
        self,
        x_static: Tensor,
        x_temporal: Tensor,
        x_patch: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Inference wrapper returning a labelled dict."""
        self.eval()
        with torch.no_grad():
            depth, lower, upper = self.forward(x_static, x_temporal, x_patch)
        return {
            "depth_pred": depth,
            "lower_ci": lower,
            "upper_ci": upper,
            "uncertainty": upper - lower,
        }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class QuantileLoss(nn.Module):
    """Pinball (quantile / check) loss for multiple quantile levels.

    L_q(y, ŷ) = q·max(y − ŷ, 0) + (1 − q)·max(ŷ − y, 0)

    Parameters
    ----------
    quantiles:
        List of quantile levels, e.g. [0.10, 0.50, 0.90].
    """

    def __init__(self, quantiles: Optional[List[float]] = None) -> None:
        super().__init__()
        self.quantiles = quantiles or QUANTILES

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        preds  : (B, Q)  predicted quantile values
        target : (B,)    ground-truth depth

        Returns
        -------
        Tensor scalar — mean pinball loss over all samples and quantiles.
        """
        if target.dim() == 1:
            target = target.unsqueeze(-1).expand_as(preds)  # (B, Q)

        losses = []
        for i, q in enumerate(self.quantiles):
            err = target[:, i] - preds[:, i]
            loss_q = torch.where(err >= 0, q * err, (q - 1.0) * err)
            losses.append(loss_q)

        return torch.stack(losses, dim=-1).mean()


class HydroFormerLoss(nn.Module):
    """Combined training loss for HydroFormer.

    L = λ_mse · MSE(depth_pred, y)
      + λ_quantile · QuantileLoss([q10, q50, q90], y)
      + λ_coverage · CoveragePenalty(q10, q90, y)

    The coverage penalty encourages the PI [q10, q90] to contain y
    roughly 80% of the time.

    Parameters
    ----------
    lambda_mse:
        Weight on the MSE term.
    lambda_quantile:
        Weight on the pinball-loss term.
    lambda_coverage:
        Weight on the coverage penalty.
    quantiles:
        Quantile levels (must match model output order).
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_quantile: float = 0.5,
        lambda_coverage: float = 0.1,
        quantiles: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_quantile = lambda_quantile
        self.lambda_coverage = lambda_coverage
        self.quantile_loss = QuantileLoss(quantiles or QUANTILES)

    def forward(
        self,
        depth_pred: Tensor,
        q10: Tensor,
        q90: Tensor,
        target: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        depth_pred : (B,)  median depth estimate
        q10        : (B,)  lower quantile
        q90        : (B,)  upper quantile
        target     : (B,)  ground-truth depth

        Returns
        -------
        Dict with keys: 'total', 'mse', 'quantile', 'coverage'
        """
        # MSE on median estimate
        mse = F.mse_loss(depth_pred, target)

        # Pinball loss on all three quantiles stacked
        preds_q = torch.stack([q10, depth_pred, q90], dim=-1)  # (B, 3)
        q_loss = self.quantile_loss(preds_q, target)

        # Coverage penalty: penalise when target falls outside [q10, q90]
        below = F.relu(q10 - target)  # positive when q10 > target
        above = F.relu(target - q90)  # positive when target > q90
        coverage_penalty = (below + above).mean()

        total = (
            self.lambda_mse * mse
            + self.lambda_quantile * q_loss
            + self.lambda_coverage * coverage_penalty
        )

        return {
            "total": total,
            "mse": mse,
            "quantile": q_loss,
            "coverage": coverage_penalty,
        }


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def init_weights(module: nn.Module) -> None:
    """Apply sensible default initialisations to Linear and LSTM layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                # Set forget-gate bias to 1.0 for better gradient flow
                n = param.size(0)
                param.data.fill_(0.0)
                param.data[n // 4 : n // 2].fill_(1.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Model factory helpers
# ---------------------------------------------------------------------------


def build_hydroformer(
    n_temporal_features: int,
    n_static_features: int,
    use_swin: bool = True,
    **kwargs,
) -> HydroFormer:
    """Instantiate a HydroFormer and apply weight initialisation.

    Parameters
    ----------
    n_temporal_features:
        Number of time-varying features.
    n_static_features:
        Number of static features.
    use_swin:
        Whether to include the Swin-T spatial encoder.
    **kwargs:
        Forwarded to HydroFormer.__init__.

    Returns
    -------
    HydroFormer (weight-initialised, in train mode)
    """
    model = HydroFormer(
        n_temporal_features=n_temporal_features,
        n_static_features=n_static_features,
        use_swin=use_swin,
        **kwargs,
    )
    model.apply(init_weights)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("HydroFormer built | trainable parameters: %d", n_params)
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return trainable / total parameter counts per sub-module."""
    counts: Dict[str, int] = {}
    for name, module in model.named_children():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        counts[name] = {"trainable": trainable, "total": total}
    counts["_overall"] = {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total": sum(p.numel() for p in model.parameters()),
    }
    return counts


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    torch.manual_seed(42)

    B, T, F_t, F_s = 4, 12, 26, 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning smoke tests on device: {device}\n")

    # ── SwinSpectralEncoder ─────────────────────────────────────────────
    print("── SwinSpectralEncoder ──")
    swin = SwinSpectralEncoder(embed_dim=64, pretrained=False, img_size=224).to(device)
    x_patch = torch.randn(B, 12, 224, 224, device=device)
    swin_out = swin(x_patch)
    print(f"  Input : {tuple(x_patch.shape)}")
    print(f"  Output: {tuple(swin_out.shape)}")
    assert swin_out.shape == (B, 64), f"Expected (B, 64), got {swin_out.shape}"

    # ── HydroForecastTFT ────────────────────────────────────────────────
    print("\n── HydroForecastTFT ──")
    tft = HydroForecastTFT(
        n_temporal_features=F_t,
        n_static_features=F_s,
        d_model=128,
        n_heads=8,
        lstm_hidden=128,
        lstm_layers=2,
    ).to(device)
    x_static = torch.randn(B, F_s, device=device)
    x_temporal = torch.randn(B, T, F_t, device=device)
    q50, q10, q90 = tft(x_static, x_temporal)
    print(f"  x_static  : {tuple(x_static.shape)}")
    print(f"  x_temporal: {tuple(x_temporal.shape)}")
    print(f"  q50       : {tuple(q50.shape)}")
    print(f"  q10       : {tuple(q10.shape)}")
    print(f"  q90       : {tuple(q90.shape)}")
    assert (q10 <= q50).all() and (q50 <= q90).all(), "Quantile ordering violated!"

    # ── HydroFormer ─────────────────────────────────────────────────────
    print("\n── HydroFormer ──")
    hf = build_hydroformer(
        n_temporal_features=F_t,
        n_static_features=F_s,
        use_swin=True,
        d_model=128,
        patch_size=224,
    ).to(device)
    depth, lower, upper = hf(x_static, x_temporal, x_patch)
    print(f"  depth_pred: {tuple(depth.shape)}")
    print(f"  lower_ci  : {tuple(lower.shape)}")
    print(f"  upper_ci  : {tuple(upper.shape)}")
    assert depth.shape == (B,)

    # ── Loss ─────────────────────────────────────────────────────────────
    print("\n── HydroFormerLoss ──")
    criterion = HydroFormerLoss()
    y = torch.rand(B, device=device) * 5.0
    _, q10_v, q90_v = tft(x_static, x_temporal)
    losses = criterion(depth, q10_v, q90_v, y)
    print(f"  total   : {losses['total'].item():.4f}")
    print(f"  mse     : {losses['mse'].item():.4f}")
    print(f"  quantile: {losses['quantile'].item():.4f}")
    print(f"  coverage: {losses['coverage'].item():.4f}")

    params = count_parameters(hf)
    print(f"\nTotal trainable parameters: {params['_overall']['trainable']:,}")

    print("\nAll smoke tests passed ✓")
    sys.exit(0)
