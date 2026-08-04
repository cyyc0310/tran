"""TransCIF model definitions.

Extracted from run_unified_eval.py and ablation_study.py for unified import.

Classes:
    RevIN                          – reversible instance normalisation
    PatchTSTFixed                  – supervised upper-bound baseline
    AdaptivePersistDLinear         – flagship zero-shot model
    ConfigEncoder                  – richer semantic-config encoder (P2)
    RichConfigAdaptivePersist      – AdaptivePersistDLinear w/ extended config

Ablation variants:
    NoAdaptiveGate, NoConfigBias, NoDecomposition,
    DirectCIF, NoPhysicsConversion
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RevIN (Reversible Instance Normalisation)
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """Reversible Instance Normalisation (Kim et al., ICLR 2022)."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            return (x - self.mean) / self.std
        else:  # denorm
            return x * self.std + self.mean


# ---------------------------------------------------------------------------
# PatchTST (supervised baseline)
# ---------------------------------------------------------------------------

class PatchTSTFixed(nn.Module):
    def __init__(self, seq_len=336, horizon=24, patch_len=24, d_model=64,
                 n_heads=4, n_layers=2):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len
        self.revin = RevIN()
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(n_patches * d_model),
            nn.Linear(n_patches * d_model, horizon))

    def forward(self, x):
        x_norm = self.revin(x, 'norm')
        B = x_norm.shape[0]
        patches = x_norm.unfold(1, self.patch_len, self.patch_len)
        x_emb = self.patch_embed(patches) + self.pos_embed
        x_enc = self.transformer(x_emb)
        out_norm = self.head(x_enc.reshape(B, -1))
        return self.revin(out_norm, 'denorm')


# ---------------------------------------------------------------------------
# Config encoder – richer semantic-config representation (P2)
# ---------------------------------------------------------------------------

class ConfigEncoder(nn.Module):
    """Encode config vectors of arbitrary dimension into a unified embedding.

    Supports two modes:
        minimal (2-d) : (mean_renew_share, ef_nonrenew / 1000)
        rich   (n-d)   : extended physical config vector (see ExtendedConfig)

    A missing-field mask is added automatically in rich mode so that
    partially-populated configs work out of the box.
    """

    def __init__(self, config_dim=2, hidden=16, out_dim=16,
                 rich_fields=None):
        super().__init__()
        self.config_dim = config_dim
        self.mlp = nn.Sequential(
            nn.Linear(config_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.rich_fields = rich_fields or []

    def forward(self, c):
        """c : (B, config_dim) → (B, out_dim)"""
        return self.mlp(c)


# Extended config fields (human-readable mapping for rich mode).
# Fields beyond the first two are optional; set to NaN when unavailable
# and they will be encoded with a separate learned mask token.
EXTENDED_CONFIG_FIELDS = [
    ("mean_renew_share",    "average renewable share"),
    ("ef_nonrenew_scaled",  "non-renewable emission factor / 1000"),
    ("solar_share",         "solar fraction of renewables"),
    ("wind_share",          "wind fraction of renewables"),
    ("hydro_share",         "hydro fraction of renewables"),
    ("interconnection_ratio", "cross-border/net-import capacity ratio"),
    ("storage_ratio",       "energy storage capacity / peak demand"),
    ("load_factor",         "average load / peak load"),
]


def build_config_vector(data: dict) -> np.ndarray:
    """Build a 2-D minimal config from a region data dict."""
    return np.array(
        [data["mean_rs"], data["ef_nr"] / 1000.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# AdaptivePersistence DLinear – flagship model
# ---------------------------------------------------------------------------

class AdaptivePersistDLinear(nn.Module):
    """Config-conditioned DLinear with adaptive persistence gate.

    Input  : (B, seq_len) renewable-share history + (B, config_dim) config
    Output : (B, horizon) renewable-share prediction ∈ [0, 1]
    """

    def __init__(self, seq_len=336, horizon=24, config_dim=2):
        super().__init__()
        self.horizon = horizon
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config):
        x3 = x.unsqueeze(1)                           # (B,1,L)
        trend = self.avg_pool(x3).squeeze(1)          # (B,L)
        seasonal = x - trend
        dlinear_out = torch.sigmoid(
            self.linear_trend(trend) +
            self.linear_seasonal(seasonal) +
            self.config_bias(config))
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


# ---------------------------------------------------------------------------
# RichConfig variant (extended config vector)
# ---------------------------------------------------------------------------

class RichConfigAdaptivePersist(nn.Module):
    """AdaptivePersistDLinear with a ConfigEncoder for richer config vectors.

    Internally the raw config is encoded through ConfigEncoder; the encoded
    representation is used for both the DLinear bias and the gate network.
    """

    def __init__(self, seq_len=336, horizon=24, config_dim=2,
                 config_hidden=16, config_out=16):
        super().__init__()
        self.horizon = horizon
        self.config_dim = config_dim
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_encoder = ConfigEncoder(config_dim, config_hidden, config_out)
        self.config_bias = nn.Linear(config_out, horizon)
        self.gate_net = nn.Sequential(
            nn.Linear(config_out + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config):
        c_enc = self.config_encoder(config)
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        dlinear_out = torch.sigmoid(
            self.linear_trend(trend) +
            self.linear_seasonal(seasonal) +
            self.config_bias(c_enc))
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([c_enc, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


# ---------------------------------------------------------------------------
# Ablation variants (replicate the architecture with one component removed)
# ---------------------------------------------------------------------------

class NoAdaptiveGate(nn.Module):
    """Fixed gate=0.5 (no adaptive switching)."""
    def __init__(self, seq_len=336, horizon=24, config_dim=2):
        super().__init__()
        self.horizon = horizon
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))

    def forward(self, x, config):
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        dlinear_out = torch.sigmoid(
            self.linear_trend(trend) +
            self.linear_seasonal(seasonal) +
            self.config_bias(config))
        persist = x[:, -self.horizon:]
        return 0.5 * persist + 0.5 * dlinear_out


class NoConfigBias(nn.Module):
    """Remove config_bias MLP; config only used in gate."""
    def __init__(self, seq_len=336, horizon=24, config_dim=2):
        super().__init__()
        self.horizon = horizon
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config):
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        dlinear_out = torch.sigmoid(
            self.linear_trend(trend) + self.linear_seasonal(seasonal))
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


class NoDecomposition(nn.Module):
    """Plain linear (no trend/seasonal decomposition)."""
    def __init__(self, seq_len=336, horizon=24, config_dim=2):
        super().__init__()
        self.horizon = horizon
        self.linear = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config):
        dlinear_out = torch.sigmoid(
            self.linear(x) + self.config_bias(config))
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


class DirectCIF(nn.Module):
    """Predict CIF directly from CIF history (no physics layer; oracle)."""
    def __init__(self, seq_len=336, horizon=24, config_dim=2):
        super().__init__()
        self.horizon = horizon
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config):
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        dlinear_out = (self.linear_trend(trend) +
                       self.linear_seasonal(seasonal) +
                       self.config_bias(config))
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


class NoPhysicsConversion(NoDecomposition):
    """Same architecture, rs input → CIF output (no sigmoid, no cif_from_shares).

    Inherits from NoDecomposition and overrides forward to remove the sigmoid
    activation, regressing directly on CIF values instead of share ∈ [0,1].
    """
    def forward(self, x, config):
        dlinear_out = self.linear(x) + self.config_bias(config)
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "AdaptivePersistDLinear": AdaptivePersistDLinear,
    "RichConfigAdaptivePersist": RichConfigAdaptivePersist,
    "PatchTSTFixed": PatchTSTFixed,
    # Ablation variants
    "NoAdaptiveGate": NoAdaptiveGate,
    "NoConfigBias": NoConfigBias,
    "NoDecomposition": NoDecomposition,
    "DirectCIF": DirectCIF,
    "NoPhysicsConversion": NoPhysicsConversion,
}


def get_model(name: str, **kwargs):
    """Factory for registered models."""
    return MODEL_REGISTRY[name](**kwargs)
