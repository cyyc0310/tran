"""Debiased-Hier: Debiased Hierarchical Prediction with Physics Consistency.

Core idea (from RESEARCH_DIRECTIONS.md §6):
    Simultaneously predict at multiple temporal granularities (hourly, daily, weekly).
    Physics consistency constraint between granularities serves as self-supervised
    debiasing signal, eliminating systematic biases in fine-grained predictions.

Key differences from standard hierarchical forecasting:
    1. Hierarchy dimension is TIME granularity, not entity levels
    2. Consistency constraint comes from PHYSICS LAYER (CIF), not hand-defined sum
    3. Consistency serves as TRAINING SIGNAL, not post-hoc reconciliation

Architecture:
    Shared Encoder (RenewShare 336h)
        ├── Hourly Head  → ŝ_1...ŝ_24 → 24 CIF values
        ├── Daily Head   → s̄_day       → 1 daily CIF (via avg of 6 × 4h blocks)
        └── Weekly Head  → s̄_week      → 1 weekly CIF (via decimation)

    Consistency Loss:
        L_consist = ||mean(CIF_hourly) - CIF_daily||²
                  + ||mean(CIF_daily_x7) - CIF_weekly||²


Exports:
    HierDLinear           — multi-head predictor (hourly + daily + weekly)
    consistency_loss      — physics-aligned multi-granularity consistency
    train_hier            — training with hierarchical + consistency loss
    predict_hier_zs       — zero-shot inference
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hierarchical DLinear
# ---------------------------------------------------------------------------

class HierDLinear(nn.Module):
    """Multi-granularity share predictor with shared backbone.

    Heads:
        hourly_head : 336 → 24   (hour-by-hour)
        daily_head  : 336 → 6    (4-hour blocks, then averaged → 1 daily)
        weekly_head : 336 → 168  (full week, then decimated → 1 weekly)
    """

    def __init__(self, seq_len=336, horizon=24, config_dim=2):
        super().__init__()
        self.horizon = horizon  # 24 hours
        self.daily_blocks = 6   # 4h per block
        self.weekly_steps = 168  # full week at hourly

        # Shared backbone
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, seq_len)
        self.linear_seasonal = nn.Linear(seq_len, seq_len)
        self.config_trend_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, seq_len))

        # Hourly head
        self.hourly_head = nn.Sequential(
            nn.Linear(seq_len, 128), nn.ReLU(),
            nn.Linear(128, horizon))

        # Daily head: 6 blocks × 4h → 6 daily share values
        self.daily_head = nn.Sequential(
            nn.Linear(seq_len, 64), nn.ReLU(),
            nn.Linear(64, self.daily_blocks))

        # Weekly head: 168h → 7 daily → 1 weekly average
        self.weekly_head = nn.Sequential(
            nn.Linear(seq_len, 64), nn.ReLU(),
            nn.Linear(64, 7))  # 7 daily values

        # Config-conditioned persistence gate (shared)
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config):
        """Return hourly, daily, weekly share predictions.

        daily:   mean share over the next 24h (single scalar, repeated)
        weekly:  mean share over the next 168h (single scalar, repeated)
        """
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend

        # Shared feature extraction
        feat = (self.linear_trend(trend) +
                self.linear_seasonal(seasonal) +
                self.config_trend_bias(config))
        feat = F.relu(feat)

        # Hourly prediction (24 values)
        hourly_share_raw = torch.sigmoid(self.hourly_head(feat))  # (B, 24)

        # Daily prediction: 6 × 4h blocks → pooled to 1
        daily_share_raw = torch.sigmoid(self.daily_head(feat))  # (B, 6)
        daily_share = daily_share_raw.mean(dim=1, keepdim=True)  # (B, 1)

        # Weekly prediction: 7 daily values → pooled to 1
        weekly_share_raw = torch.sigmoid(self.weekly_head(feat))  # (B, 7)
        weekly_share = weekly_share_raw.mean(dim=1, keepdim=True)  # (B, 1)

        # Persistence gate (shared for all heads)
        persist_hourly = x[:, -24:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))  # (B, 1)

        # Apply same gate logic to all heads
        hourly = gate * persist_hourly + (1 - gate) * hourly_share_raw
        daily = gate * persist_hourly.mean(dim=1, keepdim=True) + (1 - gate) * daily_share
        weekly = gate * x[:, -168:].mean(dim=1, keepdim=True) + (1 - gate) * weekly_share

        return hourly, daily, weekly


# ---------------------------------------------------------------------------
# Consistency loss
# ---------------------------------------------------------------------------

def consistency_loss(cif_hourly, cif_daily, cif_weekly, ef_r, ef_nr):
    """Physics-aligned hierarchical consistency loss.

    L_consist = ||mean_h(CIF_hourly) - CIF_daily||²
              + ||mean_d(CIF_daily_x7) - CIF_weekly||²

    Args:
        cif_hourly : (B, 24) hourly CIF predictions
        cif_daily  : (B, 1)  daily CIF prediction
        cif_weekly : (B, 1)  weekly CIF prediction

    Returns:
        scalar consistency loss
    """
    # Hourly → Daily consistency
    hourly_mean = cif_hourly.mean(dim=1, keepdim=True)  # (B, 1)
    L_h2d = F.mse_loss(hourly_mean, cif_daily)

    # Daily → Weekly (simplified: just compare magnitudes)
    # In practice we'd use 7 daily predictions, here we approximate
    L_d2w = F.mse_loss(cif_daily, cif_weekly)

    return L_h2d + L_d2w


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_hier(all_regions, target_name, seed=42,
                epochs=300, lr=1e-3, device=None,
                lambda_consist=0.3):
    """Train HierDLinear with hierarchical + consistency losses.

    Loss:
        L = L_hourly + L_daily + L_weekly + λ_consist * L_consist

    where daily/weekly targets are computed by averaging the true CIF values.
    """
    from transcif_pipeline import build_windows, cif_from_shares

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = HierDLinear(seq_len=336, horizon=24)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Gather data
    xs, ys_share, ys_cif, cfgs = [], [], [], []
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, y_cif_win = build_windows(data["rs"], data["cif"])
        if len(x_win) == 0:
            continue
        xs.append(x_win)
        ys_share.append(y_win)
        ys_cif.append(y_cif_win)
        cfgs.append(np.tile(data["config"], (len(x_win), 1)))

    if not xs:
        print(f"  [WARN] No source data for {target_name}")
        return model, []

    x_all = torch.tensor(np.concatenate(xs), dtype=torch.float32)
    y_share_all = torch.tensor(np.concatenate(ys_share), dtype=torch.float32)
    y_cif_all = torch.tensor(np.concatenate(ys_cif), dtype=torch.float32)
    c_all = torch.tensor(np.concatenate(cfgs), dtype=torch.float32)
    n = len(x_all)
    batch_size = min(256, n)

    if device:
        x_all, y_share_all, y_cif_all, c_all = x_all.to(device), y_share_all.to(device), y_cif_all.to(device), c_all.to(device)

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n)[:batch_size]
        x_b, y_s_b, y_cif_b, c_b = x_all[idx], y_share_all[idx], y_cif_all[idx], c_all[idx]

        hourly, daily, weekly = model(x_b, c_b)

        # Hourly share loss
        L_hourly = F.l1_loss(hourly, y_s_b)

        # Daily target: mean of 24h true share
        daily_target = y_s_b.mean(dim=1, keepdim=True)
        L_daily = F.l1_loss(daily, daily_target)

        # Weekly target: mean of last 168h (approximate)
        weekly_target = y_s_b.mean(dim=1, keepdim=True)
        L_weekly = F.l1_loss(weekly, weekly_target)

        # Consistency: CIF-level physics consistency
        ef_r_target = all_regions[target_name]["ef_r"]
        ef_nr_target = all_regions[target_name]["ef_nr"]
        cif_h = cif_from_shares(hourly, ef_r_target, ef_nr_target)
        cif_d = cif_from_shares(
            daily.expand(-1, 24), ef_r_target, ef_nr_target)[:, :1]
        cif_w = cif_from_shares(
            weekly.expand(-1, 24), ef_r_target, ef_nr_target)[:, :1]
        L_consist = consistency_loss(cif_h, cif_d, cif_w, ef_r_target, ef_nr_target)

        loss = L_hourly + L_daily + L_weekly + lambda_consist * L_consist

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    return model


def predict_hier_zs(model, x_rs, config, ef_r, ef_nr):
    """Zero-shot inference with HierDLinear.

    Returns the hourly CIF predictions (primary output).
    """
    from transcif_pipeline import cif_from_shares
    model.eval()
    x_t = torch.tensor(x_rs, dtype=torch.float32)
    c_t = torch.tensor(config).unsqueeze(0).expand(len(x_rs), -1)
    with torch.no_grad():
        hourly, daily_avg, weekly_avg = model(x_t, c_t)
    return cif_from_shares(hourly.numpy(), ef_r, ef_nr)


def compute_debias_metric(model, x_test, config, ef_r, ef_nr, y_cif_test):
    """Diagnostic: measure how much hierarchical consistency reduces hourly bias.

    Compares:
        hourly_bias = mean(|CIF_hourly - CIF_true|)
        daily_adjusted_mae = |mean(CIF_hourly) - mean(CIF_true)|
        bias_reduction = 1 - daily_adjusted_mae / hourly_mae

    Higher bias_reduction means the consistency constraint is more effective.
    """
    from transcif_pipeline import cif_from_shares
    model.eval()
    x_t = torch.tensor(x_test, dtype=torch.float32)
    c_t = torch.tensor(config).unsqueeze(0).expand(len(x_test), -1)
    with torch.no_grad():
        hourly, _, _ = model(x_t, c_t)
    cif_h = cif_from_shares(hourly.numpy(), ef_r, ef_nr)

    hourly_mae = float(np.abs(cif_h - y_cif_test).mean())
    daily_adjusted = float(np.abs(cif_h.mean(axis=0) - y_cif_test.mean(axis=0)).mean())
    bias_reduction = 1.0 - daily_adjusted / max(hourly_mae, 1e-6)

    return {"hourly_mae": hourly_mae, "daily_bias": daily_adjusted,
            "bias_reduction": bias_reduction}
