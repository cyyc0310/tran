"""Phys-IRM: Physics-Informed Invariant Risk Minimization for Zero-Shot CIF Forecasting.

Core idea (from Theorem 1):
    CIF_error = |ef_nr - ef_r| * |s_hat - s| + residual
              = L_T * share_error + residual

where L_T = |ef_nr - ef_r| is the region-specific physics amplification coefficient.

Problem: Standard ERM minimizes sum of CIF errors across regions, which over-weights
high-L_T regions and under-weights low-L_T regions. This hurts share-prediction
quality on low-L_T regions, which in turn hurts cross-domain generalization.

Solution: Phys-IRM uses (1) 1/L_T-weighted share loss to neutralize the amplification
bias, and (2) an IRM penalty to enforce that the share predictor is locally optimal
(and thus invariant) across all source regions.

Loss:
    L = L_share + λ_cif * L_cif + γ * L_irm
    where:
        L_share = mean( (1/L_T^e) * |s_hat - s| )  [per-region]
        L_cif   = mean( |CIF_pred - CIF_true| )     [per-region]
        L_irm   = ||grad_w L_share||^2              [IRM penalty]


Exports:
    PhysIRMDLinear    — share predictor with IRM-compatible interface
    IrmPenalty         — IRM gradient-penalty computation
    train_phys_irm    — Phys-IRM training loop
    predict_phys_irm  — zero-shot inference
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transcif_model import AdaptivePersistDLinear
from transcif_data import ramp_aware_loss


# ---------------------------------------------------------------------------
# Phys-IRM Share Predictor
# ---------------------------------------------------------------------------

class PhysIRMDLinear(nn.Module):
    """Share predictor that returns intermediate features for IRM penalty.

    Unlike AdaptivePersistDLinear, this model exposes:
    - The raw DLinear logits (before sigmoid) so IRM can compute a gradient
      penalty on the dummy-invariant representation.
    - The share prediction and final fused output.

    Input:  (B, seq_len) rs history + (B, config_dim) config vector
    Output: (share, feat) — share ∈ [0,1]^H, feat = invariant representation
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
        # IRM dummy classifier: linear scalar → invariant representation
        self.irm_head = nn.Linear(horizon, 1, bias=True)

    def forward(self, x, config):
        """Return (share_prediction, invariant_feat)."""
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        dlinear_logits = (
            self.linear_trend(trend) +
            self.linear_seasonal(seasonal) +
            self.config_bias(config))
        feat = dlinear_logits  # intermediate representation for IRM
        dlinear_out = torch.sigmoid(dlinear_logits)
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        share = gate * persist + (1 - gate) * dlinear_out
        return share, feat

    def predict_share(self, x, config):
        """Convenience: return share only (for inference)."""
        share, _ = self.forward(x, config)
        return share


# ---------------------------------------------------------------------------
# IRM Gradient Penalty
# ---------------------------------------------------------------------------

def irm_penalty(feat, targets, reg_weight=0.001):
    """Compute IRM gradient penalty on the invariant representation.

    IRM original formulation (Arjovsky et al., 2019):
        L_irm = ||grad_w E[w * feat - target]||^2

    For numerical stability, we use a stochastic approximation:
        L_irm = (grad_w L).norm()^2

    Args:
        feat      : (B, horizon) invariant representation
        targets   : (B, horizon) share targets
        reg_weight: stability constant (should be small)

    Returns:
        scalar irm_penalty
    """
    # Detach target from the computation graph so gradients are w.r.t. w only
    dummy_w = torch.ones(1, requires_grad=True, device=feat.device)
    # Linear dummy classifier: pred = w * feat
    # The loss is MSE of this dummy head
    pred = dummy_w * feat  # (B, horizon)
    loss = F.mse_loss(pred, targets.detach(), reduction='mean')
    # Add L2 regularisation on w for stability
    loss = loss + reg_weight * (dummy_w ** 2).sum()

    grad = torch.autograd.grad(loss, dummy_w, create_graph=True)[0]
    penalty = (grad ** 2).sum()
    return penalty


def irm_penalty_batched(feat_list, target_list, reg_weight=0.001):
    """IRM penalty summed over multiple environments (regions).

    Args:
        feat_list   : list of (B_e, horizon) tensors, one per environment
        target_list : list of (B_e, horizon) tensors, one per environment
        reg_weight  : stability constant

    Returns:
        scalar penalty (mean over environments)
    """
    penalties = []
    for f, t in zip(feat_list, target_list):
        if f.shape[0] == 0:
            continue
        p = irm_penalty(f, t, reg_weight=reg_weight)
        penalties.append(p)
    if not penalties:
        return torch.tensor(0.0, device=feat_list[0].device)
    return torch.stack(penalties).mean()


# ---------------------------------------------------------------------------
# Phys-IRM Training
# ---------------------------------------------------------------------------

def train_phys_irm(all_regions, target_name, seed=42,
                   epochs=300, lr=1e-3, device=None,
                   lambda_cif=0.5, gamma_irm=0.1,
                   use_ramp_loss=False):
    """Train a Phys-IRM share predictor on all source regions for one LORO target.

    Algorithm per epoch:
        For each source region e:
            1. L_share = mean( (1/L_T^e) * |s_hat - s| )  ← physics-debiased
            2. L_cif   = mean( |CIF(s_hat) - CIF_true| )  ← CIF quality
            3. L_irm   = ||grad + reg||^2                  ← invariance

    Args:
        all_regions : dict  {name: {"rs":..., "cif":..., "config":..., "ef_r":..., "ef_nr":...}}
        target_name : region to leave out
        lambda_cif   : weight on CIF-level supervision term
        gamma_irm   : weight on IRM gradient penalty
    """
    import random as _random_module
    torch.manual_seed(seed)
    _random_module.seed(seed)
    np.random.seed(seed)

    model = PhysIRMDLinear(seq_len=336, horizon=24)
    if device:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Gather per-region data with L_T
    region_data = []
    for name, data in all_regions.items():
        if name == target_name:
            continue
        from transcif_pipeline import build_windows
        x_win, y_win, y_cif_win = build_windows(data["rs"], data["cif"])
        if len(x_win) == 0:
            continue
        L_T = abs(data["ef_nr"] - data["ef_r"])
        region_data.append({
            "name": name,
            "x": torch.tensor(x_win, dtype=torch.float32),
            "y_share": torch.tensor(y_win, dtype=torch.float32),
            "y_cif": torch.tensor(y_cif_win, dtype=torch.float32),
            "config": torch.tensor(
                np.tile(data["config"], (len(x_win), 1)), dtype=torch.float32),
            "L_T": max(L_T, 1.0),  # prevent division by zero
            "ef_r": data["ef_r"],
            "ef_nr": data["ef_nr"],
        })

    if not region_data:
        print(f"  [WARN] No source data for {target_name}")
        return model, []

    model.train()
    log = []

    for epoch in range(epochs):
        total_loss = 0.0
        share_losses, cif_losses, irm_penalties = [], [], []

        for rd in region_data:
            n = rd["x"].shape[0]
            batch_size = min(BATCH_SIZE, n)
            idx = torch.randperm(n)[:batch_size]

            x_b = rd["x"][idx]
            y_share_b = rd["y_share"][idx]
            y_cif_b = rd["y_cif"][idx]
            c_b = rd["config"][idx]

            if device:
                x_b = x_b.to(device)
                y_share_b = y_share_b.to(device)
                y_cif_b = y_cif_b.to(device)
                c_b = c_b.to(device)

            # Forward: get share prediction + invariant features
            share_pred, feat = model(x_b, c_b)

            # 1. Share loss with 1/L_T weighting (physics-debiased)
            share_err = torch.abs(share_pred - y_share_b)
            L_share = share_err.mean() / rd["L_T"]
            share_losses.append(L_share.item())

            # 2. CIF-level supervision
            from transcif_pipeline import cif_from_shares
            cif_pred = cif_from_shares(share_pred, rd["ef_r"], rd["ef_nr"])
            L_cif = F.l1_loss(cif_pred, y_cif_b)
            cif_losses.append(L_cif.item())

            # 3. IRM penalty on this environment's invariant representation
            L_irm_env = irm_penalty(feat, y_share_b, reg_weight=0.001)
            irm_penalties.append(L_irm_env.item())

            loss = L_share + lambda_cif * L_cif + gamma_irm * L_irm_env
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Log epoch summary
        if (epoch + 1) % 50 == 0 or epoch == 0:
            n_regions = len(region_data)
            log.append({
                "epoch": epoch + 1,
                "L_share": np.mean(share_losses) if share_losses else 0,
                "L_cif": np.mean(cif_losses) if cif_losses else 0,
                "L_irm": np.mean(irm_penalties) if irm_penalties else 0,
                "total": total_loss / max(n_regions, 1),
            })

    model.eval()
    return model, log


def predict_phys_irm(model, x_rs, config, ef_r, ef_nr):
    """Zero-shot inference with PhysIRM model.

    Args:
        model  : PhysIRMDLinear
        x_rs   : (N, seq_len) numpy array — RenewShare windows
        config : (config_dim,) numpy array — target region config
        ef_r, ef_nr : emission factors

    Returns:
        cif_pred : (N, horizon) numpy array — CIF predictions
    """
    from transcif_pipeline import cif_from_shares
    model.eval()
    with torch.no_grad():
        cfg_t = torch.tensor(config).unsqueeze(0).expand(len(x_rs), -1)
        share = model.predict_share(
            torch.tensor(x_rs, dtype=torch.float32), cfg_t).numpy()
    return cif_from_shares(share, ef_r, ef_nr)


# ---------------------------------------------------------------------------
# Ablation: Phys-IRM without IRM penalty (only 1/L_T weighting)
# ---------------------------------------------------------------------------

def train_phys_weighted_only(all_regions, target_name, seed=42,
                              epochs=300, lr=1e-3, device=None,
                              lambda_cif=0.5, use_ramp_loss=False):
    """Ablation: only 1/L_T weighting, no IRM penalty.

    This isolates the contribution of the IRM invariance term vs pure
    physics-aware reweighting.
    """
    return train_phys_irm(all_regions, target_name, seed=seed,
                          epochs=epochs, lr=lr, device=device,
                          lambda_cif=lambda_cif, gamma_irm=0.0,
                          use_ramp_loss=use_ramp_loss)


# ---------------------------------------------------------------------------
# Analysis: compute L_T statistics for interpretability
# ---------------------------------------------------------------------------

def compute_L_T(all_regions):
    """Compute L_T = |ef_nr - ef_r| for every region.

    Returns:
        dict {name: L_T} sorted by L_T descending
    """
    lt = {}
    for name, data in all_regions.items():
        lt[name] = abs(data["ef_nr"] - data["ef_r"])
    return dict(sorted(lt.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Helper: BATCH_SIZE imported indirectly through pipeline constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 512  # same as transcif_pipeline default
