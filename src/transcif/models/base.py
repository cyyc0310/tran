"""TransCIF model definitions.

Classes:
    RevIN                          - reversible instance normalisation
    PatchTSTFixed                 - supervised upper-bound baseline
    AdaptivePersistDLinear        - flagship zero-shot model
    ConfigEncoder                 - richer semantic-config encoder (P2)
    RichConfigAdaptivePersist     - AdaptivePersistDLinear w/ extended config

Ablation variants:
    NoAdaptiveGate, NoConfigBias, NoDecomposition, DirectCIF, NoPhysicsConversion
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
# Config encoder - richer semantic-config representation (P2)
# ---------------------------------------------------------------------------

class ConfigEncoder(nn.Module):
    """Encode config vectors of arbitrary dimension into a unified embedding."""

    def __init__(self, config_dim=2, hidden=16, out_dim=16, rich_fields=None):
        super().__init__()
        self.config_dim = config_dim
        self.mlp = nn.Sequential(
            nn.Linear(config_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.rich_fields = rich_fields or []

    def forward(self, c):
        """c : (B, config_dim) -> (B, out_dim)"""
        return self.mlp(c)


# Extended config fields (human-readable mapping for rich mode).
EXTENDED_CONFIG_FIELDS = [
    ("mean_renew_share", "average renewable share"),
    ("ef_nonrenew_scaled", "non-renewable emission factor / 1000"),
    ("solar_share", "solar fraction of renewables"),
    ("wind_share", "wind fraction of renewables"),
    ("hydro_share", "hydro fraction of renewables"),
    ("interconnection_ratio", "cross-border/net-import capacity ratio"),
    ("storage_ratio", "energy storage capacity / peak demand"),
    ("load_factor", "average load / peak load"),
]


def build_config_vector(data: dict) -> np.ndarray:
    """Build a 2-D minimal config from a region data dict."""
    return np.array([data["mean_rs"], data["ef_nr"] / 1000.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# AdaptivePersistence DLinear - flagship model
# ---------------------------------------------------------------------------

class AdaptivePersistDLinear(nn.Module):
    """Config-conditioned DLinear with adaptive persistence gate."""

    def __init__(self, seq_len=336, horizon=24, config_dim=2):
        super().__init__()
        self.config_dim = config_dim
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
# Ablation variants
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
    """Same architecture, rs input -> CIF output (no sigmoid, no cif_from_shares)."""

    def forward(self, x, config):
        dlinear_out = self.linear(x) + self.config_bias(config)
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


# ---------------------------------------------------------------------------
# Stage D: RevIN-wrapped variant
# ---------------------------------------------------------------------------
# RevIN (Kim et al., ICLR 2022) normalises the DLinear branch per-instance so
# the model learns temporal *patterns* rather than absolute CIF levels.  This
# directly targets the temporal-OOD problem (seasonal distribution shift between
# train/test splits).  The persist branch and gate statistics use raw x so the
# physical semantics of renew_share are preserved.

class RevINAdaptivePersistDLinear(nn.Module):
    """AdaptivePersistDLinear with RevIN on the DLinear branch only."""

    def __init__(self, seq_len=336, horizon=24, config_dim=2):
        super().__init__()
        self.config_dim = config_dim
        self.horizon = horizon
        self.revin = RevIN()
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config):
        x_norm = self.revin(x, 'norm')
        x3 = x_norm.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x_norm - trend
        dlinear_raw = (self.linear_trend(trend)
                       + self.linear_seasonal(seasonal)
                       + self.config_bias(config))
        dlinear_out = torch.sigmoid(self.revin(dlinear_raw, 'denorm'))
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


# ---------------------------------------------------------------------------
# Stage C: Regime Mixture-of-Experts
# ---------------------------------------------------------------------------
# K parallel trend/seasonal experts + a softmax router conditioned on the
# (fuel-augmented) config.  The router learns to send solar-heavy regions to
# one expert, coal-heavy to another, etc., reducing the "false-neighbour"
# negative transfer that a single linear head suffers.  A +1 persist fallback
# expert guarantees a safe mode.  All config input is already fuel-augmented
# (Stage A), so the router exploits fuel structure for free.

class RegimeMoEAdaptivePersist(nn.Module):
    """K-expert MoE with config-conditioned softmax routing + persist fallback."""

    def __init__(self, seq_len=336, horizon=24, config_dim=2, num_experts=3):
        super().__init__()
        self.config_dim = config_dim
        self.horizon = horizon
        self.num_experts = num_experts
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.experts_trend = nn.ModuleList(
            [nn.Linear(seq_len, horizon) for _ in range(num_experts)])
        self.experts_seasonal = nn.ModuleList(
            [nn.Linear(seq_len, horizon) for _ in range(num_experts)])
        # config_bias must be Sequential[Linear,...] so evaluate_target can
        # probe config_dim via config_bias[0].in_features.
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))
        # router: outputs num_experts (dlinear) + 1 (persist) weights.
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 32), nn.ReLU(),
            nn.Linear(32, num_experts + 1))

    def forward(self, x, config):
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        cb = self.config_bias(config)
        expert_outs = [torch.sigmoid(self.experts_trend[i](trend)
                                     + self.experts_seasonal[i](seasonal) + cb)
                       for i in range(self.num_experts)]
        persist = x[:, -self.horizon:]
        all_outs = torch.stack(expert_outs + [persist], dim=1)  # (B, K+1, H)
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gates = torch.softmax(self.gate_net(gate_input), dim=1)  # (B, K+1)
        return (gates.unsqueeze(-1) * all_outs).sum(dim=1)


# ---------------------------------------------------------------------------
# Stage B: Weather-augmented variant
# ---------------------------------------------------------------------------
# Adds an optional weather channel (temperature, solar radiation, wind speed)
# as a second input stream.  The weather encoder compresses the (T, 3) weather
# sequence into a horizon-length bias that is added to the DLinear output,
# letting the model exploit the physical upstream of renewable generation.
# Falls back to the base model when weather is not provided.

class WeatherAdaptivePersistDLinear(nn.Module):
    """AdaptivePersistDLinear with an optional weather side channel."""

    def __init__(self, seq_len=336, horizon=24, config_dim=2, n_weather=3):
        super().__init__()
        self.config_dim = config_dim
        self.horizon = horizon
        self.n_weather = n_weather
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))
        # Weather encoder: compresses (B, seq_len, n_weather) -> (B, horizon).
        # Uses a patch-average + linear to keep parameter count modest.
        self.weather_pool = nn.AvgPool1d(kernel_size=24, stride=24)
        self.weather_head = nn.Linear(
            (seq_len // 24) * n_weather, horizon)
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config, weather=None):
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        out = (self.linear_trend(trend)
               + self.linear_seasonal(seasonal)
               + self.config_bias(config))
        if weather is not None:
            # weather: (B, seq_len, n_weather) — pool to (B, seq_len//24, n_weather)
            B = weather.shape[0]
            wp = self.weather_pool(weather.permute(0, 2, 1)).permute(0, 2, 1)
            out = out + self.weather_head(wp.reshape(B, -1))
        dlinear_out = torch.sigmoid(out)
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
    "RevINAdaptivePersistDLinear": RevINAdaptivePersistDLinear,
    "RegimeMoEAdaptivePersist": RegimeMoEAdaptivePersist,
    "WeatherAdaptivePersistDLinear": WeatherAdaptivePersistDLinear,
    "RichConfigAdaptivePersist": RichConfigAdaptivePersist,
    "PatchTSTFixed": PatchTSTFixed,
    "NoAdaptiveGate": NoAdaptiveGate,
    "NoConfigBias": NoConfigBias,
    "NoDecomposition": NoDecomposition,
    "DirectCIF": DirectCIF,
    "NoPhysicsConversion": NoPhysicsConversion,
}


def get_model(name: str, **kwargs):
    """Factory for registered models."""
    return MODEL_REGISTRY[name](**kwargs)
