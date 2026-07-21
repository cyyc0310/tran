"""Innovation 3: synthetic perturbation consistency regularization. With only one real
source domain, we simulate other regions' renewable-penetration and load regimes by
re-targeting each measured channel to a sampled (mean, std) pair drawn from the real
cross-region range observed across AEMO 2023 regions, and penalize CV-DWCC's cross-variable
feature for changing under these physically-plausible perturbations — pushing the encoder
toward relational, not region-specific, structure. Defaults to perturbing every channel in
`x`, so this keeps working unmodified whether the encoder is fed 2 channels (RenewShare,
LoadNorm) or 4 (also RenewOutNorm, NonRenewOutNorm)."""

import torch


def synthetic_perturb(
    x: torch.Tensor,
    channel_indices: tuple = None,
    target_mean_range: tuple = (0.05, 0.90),
    target_std_range: tuple = (0.08, 0.30),
    eps: float = 1e-5,
) -> torch.Tensor:
    """Re-targets each window's channel to a sampled (mean, std) pair instead of rescaling
    it by a fraction of its OWN current level. A purely multiplicative rescale (the
    original scale_range=(0.7, 1.3) design) can only ever reach values close to the
    window's own level -- e.g. a QLD1-like window (mean ~0.18) rescaled by 1.3x tops out
    around 0.23, nowhere near SA1's real ~0.69 mean -- so it never actually exercised the
    cross-region gap the consistency regularizer is meant to guard against. Re-targeting to
    an explicit (mean, std) pair, sampled from the real range spanned by AEMO 2023's four
    regions (mean 0.18-0.69, std 0.11-0.24, widened here for margin), forces the perturbed
    window to plausibly resemble an entirely different region's regime regardless of the
    source window's own level. `channel_indices` defaults to every channel present in `x`."""
    batch, _, num_channels = x.shape
    channels = channel_indices if channel_indices is not None else tuple(range(num_channels))
    perturbed = x.clone()
    for channel_idx in channels:
        channel = x[..., channel_idx]
        current_mean = channel.mean(dim=1, keepdim=True)
        current_std = channel.std(dim=1, keepdim=True).clamp_min(eps)
        target_mean = torch.empty(batch, 1).uniform_(*target_mean_range)
        target_std = torch.empty(batch, 1).uniform_(*target_std_range)
        retargeted = (channel - current_mean) / current_std * target_std + target_mean
        perturbed[..., channel_idx] = retargeted.clamp(0.0, 1.0)
    return perturbed


def consistency_loss(encoder, x: torch.Tensor) -> torch.Tensor:
    """L_consist = || CV-DWCC(x) - CV-DWCC(perturb(x)) ||^2."""
    perturbed = synthetic_perturb(x)
    features_original, _ = encoder.cv_dwcc(x)
    features_perturbed, _ = encoder.cv_dwcc(perturbed)
    return ((features_original - features_perturbed) ** 2).mean()
