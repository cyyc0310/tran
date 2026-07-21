import math
import random

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call
from transcif.models.encoder import DomainInvariantEncoder
from transcif.training.consistency import consistency_loss
from transcif.training.train_multi_source import compute_domain_weight, train_multi_source_mldg


def _make_synthetic_renew_share_dataset(num_samples: int, seq_len: int, horizon: int, seed: int):
    """Same diurnal-sine generator as test_train_source.py, parameterized by seed so each
    fake "region" gets its own phase/noise draw while sharing the same learnable
    sine-shift relationship a working encoder should pick up."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 4 * math.pi, seq_len + horizon)

    x_list, y_list = [], []
    for _ in range(num_samples):
        phase_shift = torch.empty(1).uniform_(0, 2 * math.pi).item()
        shifted = (torch.sin(t + phase_shift) + 1) / 2
        renew_share = shifted[:seq_len]
        target = shifted[seq_len : seq_len + horizon]

        load_norm = 0.5 + 0.1 * torch.randn(seq_len)
        temp_anomaly = 0.1 * torch.randn(seq_len)
        sample = torch.stack([renew_share, load_norm, temp_anomaly], dim=-1)
        x_list.append(sample)
        y_list.append(target)

    return torch.stack(x_list), torch.stack(y_list)


def _make_source_windows(num_regions: int, num_samples: int = 12, seq_len: int = 48, horizon: int = 12):
    return {
        f"region_{i}": _make_synthetic_renew_share_dataset(num_samples, seq_len, horizon, seed=i)
        for i in range(num_regions)
    }


def test_train_multi_source_mldg_raises_with_fewer_than_two_regions():
    encoder = DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    source_windows = _make_source_windows(num_regions=1)

    with pytest.raises(ValueError):
        train_multi_source_mldg(encoder, source_windows, epochs=5)


def test_train_multi_source_mldg_returns_losses_of_correct_length():
    encoder = DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    source_windows = _make_source_windows(num_regions=3)

    losses = train_multi_source_mldg(encoder, source_windows, epochs=5)

    assert len(losses) == 5


def test_train_multi_source_mldg_reduces_loss_over_training():
    encoder = DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    source_windows = _make_source_windows(num_regions=3)

    losses = train_multi_source_mldg(encoder, source_windows, epochs=40, outer_lr=5e-3, inner_lr=1e-2)

    assert losses[-1] < losses[0] * 0.7


def test_train_multi_source_mldg_updates_encoder_parameters():
    torch.manual_seed(3)
    encoder = DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    before = {name: param.detach().clone() for name, param in encoder.named_parameters()}
    source_windows = _make_source_windows(num_regions=3)

    train_multi_source_mldg(encoder, source_windows, epochs=5)

    changed = any(
        not torch.allclose(before[name], param.detach()) for name, param in encoder.named_parameters()
    )
    assert changed


def test_compute_domain_weight_is_std_plus_abs_skew_for_a_skewed_target():
    torch.manual_seed(1)
    y = torch.rand(200, 12).pow(3)

    weight = compute_domain_weight(y)

    flat = y.reshape(-1)
    mean, std = flat.mean(), flat.std(unbiased=False)
    expected_skew = ((flat - mean) / std).pow(3).mean().abs()
    assert weight.item() == pytest.approx((std + expected_skew).item(), rel=1e-4)


def test_compute_domain_weight_is_higher_for_a_more_volatile_target():
    torch.manual_seed(2)
    low_volatility_y = 0.5 + 0.01 * torch.randn(200, 12)
    high_volatility_y = torch.rand(200, 12)

    low_weight = compute_domain_weight(low_volatility_y)
    high_weight = compute_domain_weight(high_volatility_y)

    assert high_weight.item() > low_weight.item()


def test_train_multi_source_mldg_meta_train_loss_matches_domain_weighted_average(monkeypatch):
    """The pooled meta-train loss must be a domain-weighted average of each meta-train
    region's own MSE (weighted by compute_domain_weight), not a flat concatenated-batch
    MSE -- otherwise a region contributing more/easier samples numerically drowns out a
    harder, more-volatile region's transfer-relevant signal regardless of sample count."""
    source_windows = _make_source_windows(num_regions=3)
    meta_train_regions = ["region_1", "region_2"]

    torch.manual_seed(7)
    encoder = DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)

    params = dict(encoder.named_parameters())
    buffers = dict(encoder.named_buffers())
    domain_weights = {r: compute_domain_weight(source_windows[r][1]) for r in meta_train_regions}
    weight_sum = sum(domain_weights.values())

    mse_loss = nn.MSELoss()
    expected_meta_train_loss = torch.zeros(())
    for region in meta_train_regions:
        x_r, y_r = source_windows[region]
        pred_r, _ = functional_call(encoder, (params, buffers), (x_r,))
        region_loss = mse_loss(pred_r, y_r)
        expected_meta_train_loss = expected_meta_train_loss + (domain_weights[region] / weight_sum) * region_loss
    x_meta_train = torch.cat([source_windows[r][0] for r in meta_train_regions], dim=0)
    expected_meta_train_loss = expected_meta_train_loss + 0.05 * consistency_loss(encoder, x_meta_train)

    x_meta_test, y_meta_test = source_windows["region_0"]
    grads = torch.autograd.grad(expected_meta_train_loss, list(params.values()), create_graph=True, allow_unused=True)
    updated_params = {
        name: p if g is None else p - 1e-2 * g for (name, p), g in zip(params.items(), grads)
    }
    pred_meta_test, _ = functional_call(encoder, (updated_params, buffers), (x_meta_test,))
    expected_meta_test_loss = mse_loss(pred_meta_test, y_meta_test)
    expected_total_loss = expected_meta_train_loss + expected_meta_test_loss

    monkeypatch.setattr(random, "choice", lambda regions: "region_0")
    torch.manual_seed(7)
    encoder_for_training = DomainInvariantEncoder(
        num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8
    )
    losses = train_multi_source_mldg(encoder_for_training, source_windows, epochs=1)

    assert losses[0] == pytest.approx(expected_total_loss.item(), rel=1e-4)
