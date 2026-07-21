import pytest
import torch
from transcif.models.encoder import DomainInvariantEncoder, PersistenceSkipEncoder, instance_normalize


def test_instance_normalize_produces_zero_mean_unit_std_per_instance_per_channel():
    x = torch.rand(4, 96, 3) * 5 + 10
    x_norm, mean, std = instance_normalize(x)

    assert x_norm.shape == x.shape
    assert mean.shape == (4, 1, 3)
    assert std.shape == (4, 1, 3)
    torch.testing.assert_close(x_norm.mean(dim=1), torch.zeros(4, 3), atol=1e-5, rtol=0)
    torch.testing.assert_close(x_norm.std(dim=1), torch.ones(4, 3), atol=1e-4, rtol=0)


def test_instance_normalize_is_invariant_to_per_channel_affine_rescaling():
    """A region with a different absolute RenewShare level (e.g. SA1's ~0.69 mean vs
    QLD1's ~0.18 mean) but the same relative temporal shape should normalize to the same
    values -- this is the property that lets LT-MWKC's wavelet kernels learn relative
    dynamics instead of memorizing one region's absolute level."""
    torch.manual_seed(0)
    x1 = torch.rand(4, 50, 2)
    scale = torch.tensor([2.5, 0.4])
    shift = torch.tensor([0.3, -1.1])
    x2 = x1 * scale + shift

    x1_norm, _, _ = instance_normalize(x1)
    x2_norm, _, _ = instance_normalize(x2)

    torch.testing.assert_close(x1_norm, x2_norm, atol=1e-4, rtol=1e-4)


def test_encoder_lt_mwkc_features_are_scale_invariant_across_absolute_level():
    """The internal LT-MWKC path should no longer depend on the input's absolute level,
    confirming the instance-normalization fix actually reaches the scale-sensitive wavelet
    convolutions (unlike CV-DWCC's R^2 features, LT-MWKC previously ran on raw values)."""
    torch.manual_seed(0)
    model = DomainInvariantEncoder(num_variables=2, horizon=12, lt_feature_dim=8, cv_feature_dim=4)
    model.eval()

    x1 = torch.rand(3, 60, 2) * 0.3 + 0.1
    x2 = x1 * 2.0 + 0.4

    with torch.no_grad():
        x1_norm, _, _ = instance_normalize(x1, eps=model.norm_eps)
        x2_norm, _, _ = instance_normalize(x2, eps=model.norm_eps)
        lt_features_1 = model.lt_mwkc(x1_norm.permute(0, 2, 1)).mean(dim=-1)
        lt_features_2 = model.lt_mwkc(x2_norm.permute(0, 2, 1)).mean(dim=-1)

    torch.testing.assert_close(lt_features_1, lt_features_2, atol=1e-4, rtol=1e-4)


def test_encoder_output_shape_and_range():
    model = DomainInvariantEncoder(num_variables=3, horizon=24, lt_feature_dim=16, cv_feature_dim=8)
    x = torch.rand(4, 96, 3)
    renew_share_pred, dominant_idx = model(x)

    assert renew_share_pred.shape == (4, 24)
    assert torch.all(renew_share_pred >= 0.0) and torch.all(renew_share_pred <= 1.0)
    assert dominant_idx.dim() == 4


def test_encoder_forward_features_matches_manual_fused_computation():
    """forward_features must expose exactly the fused vector forward() feeds into
    self.predict -- domain-adaptation code (Deep CORAL) taps this method directly, so it
    has to agree with forward()'s own internal computation bit-for-bit."""
    torch.manual_seed(0)
    model = DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    model.eval()
    x = torch.rand(4, 60, 3)

    with torch.no_grad():
        fused, dominant_idx = model.forward_features(x)
        renew_share_pred = model.predict(fused)
        expected_pred, expected_dominant_idx = model(x)

    torch.testing.assert_close(renew_share_pred, expected_pred)
    torch.testing.assert_close(dominant_idx, expected_dominant_idx)


def test_encoder_backward_pass_updates_parameters():
    model = DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    x = torch.rand(2, 60, 3)
    target = torch.rand(2, 12)

    renew_share_pred, _ = model(x)
    loss = torch.nn.functional.mse_loss(renew_share_pred, target)
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert any(norm > 0 for norm in grad_norms)


def test_encoder_exposes_submodules_for_calibration():
    model = DomainInvariantEncoder(num_variables=3, horizon=12)
    assert hasattr(model, "lt_mwkc")
    assert hasattr(model, "cv_dwcc")
    assert model.cv_dwcc.num_variables == 3


def test_persistence_skip_encoder_output_shape_and_range():
    base = DomainInvariantEncoder(num_variables=2, horizon=12, lt_feature_dim=16, cv_feature_dim=8)
    model = PersistenceSkipEncoder(base)
    x = torch.rand(4, 60, 2)

    renew_share_pred, dominant_idx = model(x)

    assert renew_share_pred.shape == (4, 12)
    assert torch.all(renew_share_pred >= 0.0) and torch.all(renew_share_pred <= 1.0)
    assert dominant_idx.dim() == 4


def test_persistence_skip_encoder_exposes_same_submodules_as_base_encoder():
    base = DomainInvariantEncoder(num_variables=2, horizon=12)
    model = PersistenceSkipEncoder(base)

    assert model.cv_dwcc is base.cv_dwcc
    assert model.lt_mwkc is base.lt_mwkc


def test_persistence_skip_encoder_gate_initializes_favoring_persistence():
    base = DomainInvariantEncoder(num_variables=2, horizon=12)
    model = PersistenceSkipEncoder(base)

    gate = torch.sigmoid(model.gate_logit)
    assert gate.item() == pytest.approx(0.8175744762, abs=1e-4)
    assert gate.item() > 0.5


def test_persistence_skip_encoder_output_matches_gated_blend_of_last_observed_and_network():
    base = DomainInvariantEncoder(num_variables=2, horizon=12)
    model = PersistenceSkipEncoder(base)
    x = torch.rand(3, 60, 2)

    with torch.no_grad():
        network_pred, _ = base(x)
        blended_pred, _ = model(x)
        volatility_gain = torch.nn.functional.softplus(model.volatility_gain_raw)
        effective_gate_logit = model.gate_logit - volatility_gain * model.recent_volatility(x)
        gate = torch.sigmoid(effective_gate_logit)
        last_observed = x[:, -1, 0:1].expand(-1, network_pred.shape[-1])
        expected = gate * last_observed + (1 - gate) * network_pred

    assert blended_pred.numpy() == pytest.approx(expected.numpy(), abs=1e-6)


def test_persistence_skip_encoder_gate_at_init_is_nearly_volatility_blind():
    """volatility_gain_raw initializes to -6.0 (softplus ~= 0.0025), so training starts at
    essentially the original volatility-blind gate rather than an untrained, arbitrary
    conditioning -- confirmed by comparing against the pre-existing plain-gate_logit blend."""
    base = DomainInvariantEncoder(num_variables=2, horizon=12)
    model = PersistenceSkipEncoder(base)
    x = torch.rand(3, 60, 2)

    with torch.no_grad():
        network_pred, _ = base(x)
        blended_pred, _ = model(x)
        plain_gate = torch.sigmoid(model.gate_logit)
        last_observed = x[:, -1, 0:1].expand(-1, network_pred.shape[-1])
        plain_expected = plain_gate * last_observed + (1 - plain_gate) * network_pred

    assert blended_pred.numpy() == pytest.approx(plain_expected.numpy(), abs=1e-2)


def test_persistence_skip_encoder_higher_volatility_lowers_effective_gate():
    """A high-volatility SA1-like window should trust the network correction more (lower
    gate) than a near-constant low-volatility window, once volatility_gain_raw is trained
    away from its near-zero init -- otherwise conditioning the gate on volatility has no
    effect regardless of how strongly a later training loss would want it to."""
    base = DomainInvariantEncoder(num_variables=2, horizon=12)
    model = PersistenceSkipEncoder(base)
    model.volatility_gain_raw = torch.nn.Parameter(torch.tensor(3.0))

    torch.manual_seed(0)
    low_vol_x = torch.full((2, 60, 2), 0.5) + torch.rand(2, 60, 2) * 0.001
    high_vol_x = torch.rand(2, 60, 2)

    with torch.no_grad():
        low_vol_gate = torch.sigmoid(
            model.gate_logit
            - torch.nn.functional.softplus(model.volatility_gain_raw) * model.recent_volatility(low_vol_x)
        )
        high_vol_gate = torch.sigmoid(
            model.gate_logit
            - torch.nn.functional.softplus(model.volatility_gain_raw) * model.recent_volatility(high_vol_x)
        )

    assert torch.all(high_vol_gate < low_vol_gate)


def test_persistence_skip_encoder_recent_volatility_matches_std_of_last_window():
    base = DomainInvariantEncoder(num_variables=2, horizon=12)
    model = PersistenceSkipEncoder(base, volatility_window=24)
    x = torch.rand(2, 60, 2)

    volatility = model.recent_volatility(x)

    expected = x[:, -24:, 0].std(dim=1, unbiased=False, keepdim=True)
    assert volatility.numpy() == pytest.approx(expected.numpy(), abs=1e-6)


def test_persistence_skip_encoder_backward_pass_updates_gate_and_base_parameters():
    base = DomainInvariantEncoder(num_variables=2, horizon=12)
    model = PersistenceSkipEncoder(base)
    x = torch.rand(2, 60, 2)
    target = torch.rand(2, 12)

    renew_share_pred, _ = model(x)
    loss = torch.nn.functional.mse_loss(renew_share_pred, target)
    loss.backward()

    assert model.gate_logit.grad is not None
    assert model.volatility_gain_raw.grad is not None
    base_grad_norms = [p.grad.norm().item() for p in base.parameters() if p.grad is not None]
    assert len(base_grad_norms) > 0
    assert any(norm > 0 for norm in base_grad_norms)
