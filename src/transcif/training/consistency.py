"""Innovation 3: synthetic perturbation consistency regularization. With only one real
source domain, we simulate other regions' renewable-penetration and load regimes by
independently rescaling the RenewShare and LoadNorm channels, and penalize CV-DWCC's
cross-variable feature for changing under these physically-plausible perturbations —
pushing the encoder toward relational, not region-specific, structure."""

import torch


def synthetic_perturb(
    x: torch.Tensor,
    renew_share_idx: int = 0,
    load_norm_idx: int = 1,
    scale_range: tuple = (0.7, 1.3),
) -> torch.Tensor:
    renew_scale = torch.empty(x.shape[0], 1).uniform_(*scale_range)
    load_scale = torch.empty(x.shape[0], 1).uniform_(*scale_range)
    perturbed = x.clone()
    perturbed[..., renew_share_idx] = (x[..., renew_share_idx] * renew_scale).clamp(0.0, 1.0)
    perturbed[..., load_norm_idx] = (x[..., load_norm_idx] * load_scale).clamp(0.0, 1.0)
    return perturbed


def consistency_loss(encoder, x: torch.Tensor) -> torch.Tensor:
    """L_consist = || CV-DWCC(x) - CV-DWCC(perturb(x)) ||^2."""
    perturbed = synthetic_perturb(x)
    features_original, _ = encoder.cv_dwcc(x)
    features_perturbed, _ = encoder.cv_dwcc(perturbed)
    return ((features_original - features_perturbed) ** 2).mean()
