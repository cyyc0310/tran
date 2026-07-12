import math

import numpy as np
import torch

from transcif.models.encoder import DomainInvariantEncoder
from transcif.training.train_source import train_source_domain
from transcif.physics.cif import cif_from_shares, get_emission_factors
from transcif.physics.residual import ResidualCorrectionHead, fit_residual_head
from transcif.calibration.dominant_reweight import recompute_dominant_variable, reweight_lt_mwkc_alpha
from transcif.calibration.conformal import (
    compute_nonconformity_scores,
    conformal_interval_halfwidth,
    predict_with_interval,
    empirical_coverage,
)
from transcif.evaluation.metrics import mae, cross_domain_degradation_rate
from transcif.evaluation.baselines import naive_transfer_predict


SEQ_LEN = 48
HORIZON = 12


def _synthetic_region_batch(num_samples: int, renew_baseline: float, seed: int):
    """Builds a synthetic (RenewShare, LoadNorm, TempAnomaly) batch whose RenewShare
    trajectory oscillates around `renew_baseline`, standing in for a real region's data
    with the exact schema `DomainInvariantEncoder` expects."""
    generator = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 4 * math.pi, SEQ_LEN + HORIZON)

    x_list, y_list = [], []
    for _ in range(num_samples):
        phase_shift = torch.empty(1).uniform_(0, 2 * math.pi, generator=generator).item()
        amplitude = min(renew_baseline, 1 - renew_baseline)
        wave = renew_baseline + amplitude * torch.sin(t + phase_shift)
        wave = wave.clamp(0.0, 1.0)
        renew_share = wave[:SEQ_LEN]
        target = wave[SEQ_LEN : SEQ_LEN + HORIZON]

        load_norm = 0.5 + 0.05 * torch.randn(SEQ_LEN, generator=generator)
        temp_anomaly = 0.05 * torch.randn(SEQ_LEN, generator=generator)
        sample = torch.stack([renew_share, load_norm, temp_anomaly], dim=-1)
        x_list.append(sample)
        y_list.append(target)

    return torch.stack(x_list), torch.stack(y_list)


def test_naive_transfer_predict_uses_source_region_factors_only():
    encoder = DomainInvariantEncoder(num_variables=3, horizon=HORIZON, lt_feature_dim=8, cv_feature_dim=4)
    x_target, _ = _synthetic_region_batch(num_samples=4, renew_baseline=0.7, seed=1)

    ci_pred = naive_transfer_predict(encoder, x_target, source_region_code="AU_NSW")

    renew_factor, nonrenew_factor = get_emission_factors("AU_NSW")
    assert ci_pred.shape == (4, HORIZON)
    assert np.all(ci_pred >= min(renew_factor, nonrenew_factor) - 1e-6)
    assert np.all(ci_pred <= max(renew_factor, nonrenew_factor) + 1e-6)


def test_end_to_end_pipeline_on_synthetic_source_and_target_domains():
    """Full smoke test: train on a fossil-dominant synthetic source domain, calibrate on
    a small renewable-dominant synthetic target domain sample, and verify every stage
    (training, physics reconstruction, residual correction, dominant-variable reweight,
    conformal interval) runs end-to-end and produces sane outputs."""
    torch.manual_seed(42)

    x_source, y_source = _synthetic_region_batch(num_samples=32, renew_baseline=0.2, seed=10)
    encoder = DomainInvariantEncoder(num_variables=3, horizon=HORIZON, lt_feature_dim=16, cv_feature_dim=8)
    losses = train_source_domain(encoder, x_source, y_source, epochs=25, lr=5e-3, consistency_weight=0.05)
    assert losses[-1] < losses[0]

    x_calib, y_calib_share = _synthetic_region_batch(num_samples=20, renew_baseline=0.75, seed=20)
    with torch.no_grad():
        renew_share_pred, _ = encoder(x_calib)

    renew_factor, nonrenew_factor = get_emission_factors("AU_SA")
    ci_pred_physics_only = cif_from_shares(renew_share_pred.numpy(), renew_factor, nonrenew_factor)
    ci_true = cif_from_shares(y_calib_share.numpy(), renew_factor, nonrenew_factor)

    dominant_idx = recompute_dominant_variable(encoder, x_calib)
    reweight_lt_mwkc_alpha(encoder, dominant_idx, boost=1.5)

    residual_head = ResidualCorrectionHead(input_dim=1, hidden_dim=8)
    calib_features = torch.tensor(renew_share_pred.numpy().reshape(-1, 1), dtype=torch.float32)
    calib_targets = torch.tensor((ci_true - ci_pred_physics_only).reshape(-1), dtype=torch.float32)
    fit_residual_head(residual_head, calib_features, calib_targets, epochs=100, lr=1e-2)

    with torch.no_grad():
        delta = residual_head(calib_features).numpy().reshape(ci_pred_physics_only.shape)
    ci_pred_corrected = ci_pred_physics_only + delta

    nonconformity = compute_nonconformity_scores(ci_true.reshape(-1), ci_pred_corrected.reshape(-1))
    halfwidth = conformal_interval_halfwidth(nonconformity, coverage=0.9)
    lower, upper = predict_with_interval(ci_pred_corrected.reshape(-1), halfwidth)
    coverage = empirical_coverage(ci_true.reshape(-1), lower, upper)

    assert halfwidth >= 0.0
    assert 0.0 <= coverage <= 1.0

    corrected_mae = mae(ci_true.reshape(-1), ci_pred_corrected.reshape(-1))
    physics_only_mae = mae(ci_true.reshape(-1), ci_pred_physics_only.reshape(-1))
    degradation = cross_domain_degradation_rate(
        in_domain_metric=max(physics_only_mae, 1e-6), cross_domain_metric=corrected_mae
    )
    assert isinstance(degradation, float)
