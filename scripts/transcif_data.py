"""TransCIF data handling with missing-value awareness and robust losses.

P1 items from IMPROVEMENT_PLAN.md:
  - Missing-mask generation & augmentation
  - Huber / ramp-aware loss
  - Data quality scoring

Usage:
    from transcif_data import (
        build_windows_with_mask, huber_loss, ramp_aware_loss,
        compute_data_quality, MissingMaskAugmentor,
    )
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Window builder with missing mask
# ---------------------------------------------------------------------------

def build_windows_with_mask(rs, cif, seq_len, horizon, stride):
    """Build sliding windows with a missing-validity mask.

    Returns
    -------
    x_rs   : (N, seq_len)
    y_rs   : (N, horizon)
    y_cif  : (N, horizon)
    x_mask : (N, seq_len)  1=valid 0=missing
    y_mask : (N, horizon)  1=valid 0=missing
    """
    window = seq_len + horizon
    x_rs, y_rs, y_cif = [], [], []
    x_mask, y_mask = [], []
    for start in range(0, len(rs) - window + 1, stride):
        x_rs.append(rs[start:start + seq_len])
        y_rs.append(rs[start + seq_len:start + window])
        y_cif.append(cif[start + seq_len:start + window])
        # Heuristic mask: NaN / negative cif / share clipped to [0,1]
        x_mask.append(np.isfinite(rs[start:start + seq_len]).astype(np.float32))
        y_mask.append(np.isfinite(rs[start + seq_len:start + window]).astype(np.float32))
    if not x_rs:
        return (np.empty((0, seq_len)), np.empty((0, horizon)),
                np.empty((0, horizon)), np.empty((0, seq_len)),
                np.empty((0, horizon)))
    return (np.stack(x_rs), np.stack(y_rs), np.stack(y_cif),
            np.stack(x_mask), np.stack(y_mask))


# ---------------------------------------------------------------------------
# Robust losses
# ---------------------------------------------------------------------------

def huber_loss(pred, target, delta=1.0, mask=None):
    """Huber loss (smooth L1).

    Args:
        pred   : (B, H) tensor
        target : (B, H) tensor
        delta  : transition point between L1 and L2
        mask   : (B, H) optional valid-pixel mask (1=valid)
    """
    loss = F.smooth_l1_loss(pred, target, beta=delta, reduction='none')
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


def ramp_aware_loss(pred, target, ramp_thresh=0.05, ramp_weight=2.0,
                    mask=None, reduction='mean'):
    """L1 loss with higher weight on ramp (large-change) time steps.

    Ramp weights are computed from the target sequence's absolute differences
    so that periods with large CIF shifts receive higher penalty.

    Args:
        pred, target : (B, H) tensors
        ramp_thresh  : threshold for "large change"
        ramp_weight  : multiplier for ramp time steps
        mask         : (B, H) optional valid mask
        reduction    : 'mean' | 'none' — if 'none', returns (B, H) per-element
    """
    base = torch.abs(pred - target)  # (B, H)
    # ramp factor: 1.0 for flat, ramp_weight for large changes
    with torch.no_grad():
        diffs = torch.abs(target[:, 1:] - target[:, :-1])  # (B, H-1)
        ramp_factor = torch.ones_like(target)
        ramp_factor[:, 1:] += (ramp_weight - 1.0) * (diffs > ramp_thresh).float()
    weighted = base * ramp_factor
    if mask is not None:
        weighted = weighted * mask
    if reduction == 'none':
        return weighted
    if reduction == 'mean':
        denom = mask.sum() + 1e-8 if mask is not None else weighted.numel()
        return weighted.sum() / denom


# ---------------------------------------------------------------------------
# Data quality scoring
# ---------------------------------------------------------------------------

def compute_data_quality(rs, cif):
    """Return a quality score ∈ [0, 1] for a raw timeseries.

    Checks:  - missing fraction
             - physical range violations (share ∈ [0,1], cif ≥ 0)
             - consecutive-identical blocks (stuck sensor)
             - variance / stdev sufficiency
    """
    rs = np.asarray(rs, dtype=np.float32)
    cif = np.asarray(cif, dtype=np.float32)
    n = len(rs)
    if n < 24:
        return 0.0

    # Completeness
    missing_f = np.mean(~np.isfinite(rs)) + np.mean(~np.isfinite(cif))
    completeness = max(0.0, 1.0 - missing_f / 0.3)  # 30% missing → 0

    # Physical validity
    valid = (rs >= 0) & (rs <= 1) & (cif >= 0)
    validity = np.mean(valid.astype(float))

    # Stuck-sensor penalty
    n_unique_rs = len(set(np.round(rs[::24], 4)))  # daily samples
    stuck_penalty = min(1.0, n_unique_rs / max(1, n // 24 + 1))

    # Variance sufficiency
    rs_std = np.nanstd(rs)
    suf = min(1.0, rs_std / 0.02)  # 0.02 ≈ 2% min std

    # Weighted average
    quality = (0.25 * completeness + 0.3 * validity +
               0.25 * stuck_penalty + 0.2 * suf)
    return float(np.clip(quality, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Missing-pattern augmentation
# ---------------------------------------------------------------------------

class MissingMaskAugmentor:
    """Randomly drop out time steps during training to improve robustness.

    Supports two modes:
        point:  independent Bernoulli per time step
        block:  contiguous block missing (simulates sensor outage)
    """

    def __init__(self, prob=0.05, mode='point', min_block=1, max_block=12):
        self.prob = prob
        self.mode = mode
        self.min_block = min_block
        self.max_block = max_block

    def __call__(self, x):
        """x : (B, L) numpy or torch tensor → augmented x, mask"""
        is_np = isinstance(x, np.ndarray)
        if is_np:
            x = torch.from_numpy(x).float()
        B, L = x.shape
        mask = torch.ones(B, L)
        if self.mode == 'point':
            mask = torch.bernoulli(
                torch.full((B, L), 1.0 - self.prob))
        elif self.mode == 'block':
            for i in range(B):
                j = 0
                while j < L:
                    if torch.rand(1).item() < self.prob:
                        blen = np.random.randint(self.min_block, self.max_block + 1)
                        mask[i, j:j + blen] = 0.0
                        j += blen
                    else:
                        j += 1
        x_aug = x * mask
        if is_np:
            return x_aug.numpy(), mask.numpy()
        return x_aug, mask
