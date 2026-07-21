import math

import pytest
import torch
from transcif.models.encoder import DomainInvariantEncoder, PersistenceSkipEncoder
from transcif.training.domain_adaptation import (
    coral_loss,
    fine_tune_on_calibration,
    train_multi_source_mldg_coral,
)


def _make_synthetic_renew_share_dataset(num_samples: int, seq_len: int, horizon: int, seed: int):
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


def test_coral_loss_is_zero_for_identical_feature_distributions():
    torch.manual_seed(0)
    features = torch.randn(50, 10)

    loss = coral_loss(features, features.clone())

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_coral_loss_is_positive_for_differing_covariance():
    torch.manual_seed(1)
    source_features = torch.randn(50, 10)
    target_features = torch.randn(50, 10) * 5.0 + 2.0

    loss = coral_loss(source_features, target_features)

    assert loss.item() > 0.0


def test_coral_loss_is_symmetric():
    torch.manual_seed(2)
    source_features = torch.randn(30, 6)
    target_features = torch.randn(40, 6) * 2.0

    forward_loss = coral_loss(source_features, target_features)
    backward_loss = coral_loss(target_features, source_features)

    assert forward_loss.item() == pytest.approx(backward_loss.item(), rel=1e-5)


def test_train_multi_source_mldg_coral_raises_with_fewer_than_two_regions():
    encoder = PersistenceSkipEncoder(
        DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    )
    source_windows = _make_source_windows(num_regions=1)
    x_target, _ = _make_synthetic_renew_share_dataset(8, 48, 12, seed=99)

    with pytest.raises(ValueError):
        train_multi_source_mldg_coral(encoder, source_windows, x_target, epochs=2)


def test_train_multi_source_mldg_coral_returns_losses_of_correct_length():
    encoder = PersistenceSkipEncoder(
        DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    )
    source_windows = _make_source_windows(num_regions=3)
    x_target, _ = _make_synthetic_renew_share_dataset(8, 48, 12, seed=99)

    losses = train_multi_source_mldg_coral(encoder, source_windows, x_target, epochs=5)

    assert len(losses) == 5


def test_train_multi_source_mldg_coral_updates_encoder_parameters():
    torch.manual_seed(3)
    encoder = PersistenceSkipEncoder(
        DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    )
    before = {name: param.detach().clone() for name, param in encoder.named_parameters()}
    source_windows = _make_source_windows(num_regions=3)
    x_target, _ = _make_synthetic_renew_share_dataset(8, 48, 12, seed=99)

    train_multi_source_mldg_coral(encoder, source_windows, x_target, epochs=5)

    changed = any(
        not torch.allclose(before[name], param.detach()) for name, param in encoder.named_parameters()
    )
    assert changed


def test_fine_tune_on_calibration_reduces_loss_over_training():
    torch.manual_seed(4)
    encoder = PersistenceSkipEncoder(
        DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    )
    x_calib, y_calib = _make_synthetic_renew_share_dataset(40, 48, 12, seed=5)

    losses = fine_tune_on_calibration(encoder, x_calib, y_calib, epochs_per_stage=15, lr=1e-2)

    assert len(losses) == 45
    assert losses[-1] < losses[0] * 0.9


def test_fine_tune_on_calibration_respects_gradual_unfreezing_order():
    """During stage 0 (only gate_logit/volatility_gain_raw/predict unfrozen), the deepest
    group (base_encoder.lt_mwkc) must not move -- gradual unfreezing is meaningless if
    every parameter is trainable from the first stage."""
    torch.manual_seed(6)
    encoder = PersistenceSkipEncoder(
        DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    )
    lt_mwkc_before = {
        name: param.detach().clone()
        for name, param in encoder.named_parameters()
        if name.startswith("base_encoder.lt_mwkc")
    }
    x_calib, y_calib = _make_synthetic_renew_share_dataset(20, 48, 12, seed=7)

    fine_tune_on_calibration(
        encoder,
        x_calib,
        y_calib,
        epochs_per_stage=5,
        lr=1e-2,
        unfreeze_groups=(("gate_logit", "volatility_gain_raw", "base_encoder.predict"),),
    )

    for name, param in encoder.named_parameters():
        if name.startswith("base_encoder.lt_mwkc"):
            torch.testing.assert_close(param.detach(), lt_mwkc_before[name])


def test_fine_tune_on_calibration_eventually_unfreezes_deepest_group():
    torch.manual_seed(8)
    encoder = PersistenceSkipEncoder(
        DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    )
    lt_mwkc_before = {
        name: param.detach().clone()
        for name, param in encoder.named_parameters()
        if name.startswith("base_encoder.lt_mwkc")
    }
    x_calib, y_calib = _make_synthetic_renew_share_dataset(20, 48, 12, seed=9)

    fine_tune_on_calibration(encoder, x_calib, y_calib, epochs_per_stage=10, lr=1e-2)

    changed = any(
        not torch.allclose(lt_mwkc_before[name], param.detach())
        for name, param in encoder.named_parameters()
        if name.startswith("base_encoder.lt_mwkc")
    )
    assert changed
