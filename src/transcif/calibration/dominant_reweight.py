"""Innovation 4: reuse CV-DWCC's dominant-variable identification (originally an internal
prediction-time mechanism) as a cross-domain calibration handle. Deploying to a new
region, we run dominant-variable identification once on the small calibration set to
decide whether the region is "renewable-driven" or "load/temperature-driven", then only
reweight the already-trained wavelet fusion coefficients accordingly — no other network
weights are touched."""

import torch

from transcif.models.encoder import DomainInvariantEncoder

RENEW_SHARE_CHANNEL_IDX = 0


def recompute_dominant_variable(encoder: DomainInvariantEncoder, calibration_x: torch.Tensor) -> int:
    """Run CV-DWCC's dominant-predictor identification on the calibration set and tally
    votes for the true global channel that wins most often.

    `encoder.cv_dwcc.forward` computes, for each target variable, a local weighted
    regression against all *other* variables and reports `dominant_idx` as an index into
    that reduced (num_variables - 1) predictor list, not a global channel index. Which
    global channel a given local index refers to depends on which variable is currently
    the target, so each target's local indices must be remapped back to global channel
    ids before being pooled into a single vote count.
    """
    with torch.no_grad():
        _, dominant_idx = encoder.cv_dwcc(calibration_x)
    num_variables = encoder.cv_dwcc.num_variables
    _, _, num_targets, _ = dominant_idx.shape
    counts = torch.zeros(num_variables, dtype=torch.long)
    for target_idx in range(num_targets):
        predictor_global_ids = torch.tensor(
            [channel for channel in range(num_variables) if channel != target_idx]
        )
        local_indices = dominant_idx[:, :, target_idx, :].reshape(-1)
        counts += torch.bincount(predictor_global_ids[local_indices], minlength=num_variables)
    return int(counts.argmax().item())


def reweight_lt_mwkc_alpha(
    encoder: DomainInvariantEncoder,
    dominant_variable_idx: int,
    boost: float = 2.0,
) -> None:
    """Boost the short-kernel (high-frequency) branch when RenewShare dominates (regions
    with volatile renewable output need fine-grained temporal resolution); otherwise boost
    the long-kernel (low-frequency) branch (load/temperature-driven regions are smoother)."""
    if dominant_variable_idx == RENEW_SHARE_CHANNEL_IDX:
        branch = encoder.lt_mwkc.branches[0]
    else:
        branch = encoder.lt_mwkc.branches[-1]
    with torch.no_grad():
        branch.alpha.mul_(boost)
