"""Naive-transfer baseline (the "lower bound" from the design doc's experiment plan):
apply the source-trained encoder directly to target-domain data, reconstruct CI with the
SOURCE region's emission factor table, with no residual correction and no dominant-
variable reweighting."""

import numpy as np
import torch

from transcif.models.encoder import DomainInvariantEncoder
from transcif.physics.cif import cif_from_shares, get_emission_factors
from transcif.training.train_source import train_source_domain
from transcif.physics.residual import ResidualCorrectionHead, fit_residual_head
from transcif.calibration.dominant_reweight import recompute_dominant_variable, reweight_lt_mwkc_alpha
from transcif.calibration.conformal import (
    compute_nonconformity_scores,
    conformal_interval_halfwidth,
    predict_with_interval,
    empirical_coverage,
)
from transcif.evaluation.metrics import mae, cross_domain_degradation_rate


def naive_transfer_predict(
    encoder: DomainInvariantEncoder,
    x_target: torch.Tensor,
    source_region_code: str,
) -> np.ndarray:
    with torch.no_grad():
        renew_share_pred, _ = encoder(x_target)
    renew_factor, nonrenew_factor = get_emission_factors(source_region_code)
    return cif_from_shares(renew_share_pred.numpy(), renew_factor, nonrenew_factor)


def run_end_to_end_smoke_test(
    encoder: DomainInvariantEncoder,
    x_source: torch.Tensor,
    y_source: torch.Tensor,
    x_calib: torch.Tensor,
    y_calib_share: torch.Tensor,
    target_region_code: str,
    train_epochs: int = 25,
    train_lr: float = 5e-3,
    consistency_weight: float = 0.05,
    residual_hidden_dim: int = 8,
    residual_epochs: int = 100,
    residual_lr: float = 1e-2,
    reweight_boost: float = 1.5,
    conformal_coverage: float = 0.9,
) -> dict:
    """Runs the full Stage 1-3 pipeline (source-domain training, physics reconstruction,
    dominant-variable reweight, residual correction, conformal interval) on
    already-constructed source/calibration batches against `target_region_code`'s
    emission factors. Returns a dict of every intermediate and final output so callers
    (this module's pytest smoke test, and future ablation-experiment scripts) can
    inspect or assert on any stage without re-deriving this orchestration."""
    losses = train_source_domain(
        encoder,
        x_source,
        y_source,
        epochs=train_epochs,
        lr=train_lr,
        consistency_weight=consistency_weight,
    )

    with torch.no_grad():
        renew_share_pred, _ = encoder(x_calib)

    renew_factor, nonrenew_factor = get_emission_factors(target_region_code)
    ci_pred_physics_only = cif_from_shares(renew_share_pred.numpy(), renew_factor, nonrenew_factor)
    ci_true = cif_from_shares(y_calib_share.numpy(), renew_factor, nonrenew_factor)

    dominant_idx = recompute_dominant_variable(encoder, x_calib)
    reweight_lt_mwkc_alpha(encoder, dominant_idx, boost=reweight_boost)

    residual_head = ResidualCorrectionHead(input_dim=1, hidden_dim=residual_hidden_dim)
    calib_features = torch.tensor(renew_share_pred.numpy().reshape(-1, 1), dtype=torch.float32)
    calib_targets = torch.tensor((ci_true - ci_pred_physics_only).reshape(-1), dtype=torch.float32)
    fit_residual_head(residual_head, calib_features, calib_targets, epochs=residual_epochs, lr=residual_lr)

    with torch.no_grad():
        delta = residual_head(calib_features).numpy().reshape(ci_pred_physics_only.shape)
    ci_pred_corrected = ci_pred_physics_only + delta

    nonconformity = compute_nonconformity_scores(ci_true.reshape(-1), ci_pred_corrected.reshape(-1))
    halfwidth = conformal_interval_halfwidth(nonconformity, coverage=conformal_coverage)
    lower, upper = predict_with_interval(ci_pred_corrected.reshape(-1), halfwidth)
    coverage = empirical_coverage(ci_true.reshape(-1), lower, upper)

    corrected_mae = mae(ci_true.reshape(-1), ci_pred_corrected.reshape(-1))
    physics_only_mae = mae(ci_true.reshape(-1), ci_pred_physics_only.reshape(-1))
    degradation = cross_domain_degradation_rate(
        in_domain_metric=max(physics_only_mae, 1e-6), cross_domain_metric=corrected_mae
    )

    return {
        "losses": losses,
        "dominant_idx": dominant_idx,
        "ci_true": ci_true,
        "ci_pred_physics_only": ci_pred_physics_only,
        "ci_pred_corrected": ci_pred_corrected,
        "halfwidth": halfwidth,
        "lower": lower,
        "upper": upper,
        "coverage": coverage,
        "corrected_mae": corrected_mae,
        "physics_only_mae": physics_only_mae,
        "degradation": degradation,
    }
