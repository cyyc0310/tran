import math

import numpy as np
import torch

from transcif.models.encoder import DomainInvariantEncoder
from transcif.physics.cif import get_emission_factors
from transcif.evaluation.baselines import naive_transfer_predict, run_end_to_end_smoke_test


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
    x_calib, y_calib_share = _synthetic_region_batch(num_samples=20, renew_baseline=0.75, seed=20)

    result = run_end_to_end_smoke_test(
        encoder, x_source, y_source, x_calib, y_calib_share, target_region_code="AU_SA",
    )

    assert result["losses"][-1] < result["losses"][0]
    assert result["halfwidth"] >= 0.0
    assert 0.0 <= result["coverage"] <= 1.0
    assert isinstance(result["degradation"], float)
