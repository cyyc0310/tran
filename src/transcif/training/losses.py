"""Loss functions for TransCIF training (robust / ramp-aware)."""

import torch
import torch.nn.functional as F


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

    Args:
        pred, target : (B, H) tensors
        ramp_thresh  : threshold for "large change"
        ramp_weight  : multiplier for ramp time steps
        mask         : (B, H) optional valid mask
        reduction    : 'mean' | 'none' - if 'none', returns (B, H) per-element
    """
    base = torch.abs(pred - target)
    with torch.no_grad():
        diffs = torch.abs(target[:, 1:] - target[:, :-1])
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
