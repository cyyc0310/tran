import torch
from transcif.models.encoder import DomainInvariantEncoder
from transcif.training.consistency import synthetic_perturb, consistency_loss


def test_synthetic_perturb_keeps_renew_share_and_load_norm_in_unit_interval():
    x = torch.rand(4, 30, 3)
    perturbed = synthetic_perturb(x, channel_indices=(0, 1))
    assert perturbed.shape == x.shape
    assert torch.all(perturbed[..., 0] >= 0.0) and torch.all(perturbed[..., 0] <= 1.0)
    assert torch.all(perturbed[..., 1] >= 0.0) and torch.all(perturbed[..., 1] <= 1.0)


def test_synthetic_perturb_leaves_untargeted_channel_untouched():
    x = torch.rand(2, 20, 3)
    perturbed = synthetic_perturb(x, channel_indices=(0, 1))
    torch.testing.assert_close(perturbed[..., 2:], x[..., 2:])


def test_synthetic_perturb_defaults_to_perturbing_every_channel():
    x = torch.rand(2, 20, 4)
    perturbed = synthetic_perturb(x)
    for channel_idx in range(4):
        assert not torch.allclose(perturbed[..., channel_idx], x[..., channel_idx])


def test_synthetic_perturb_can_retarget_a_low_mean_window_to_a_high_mean_region():
    """The old multiplicative-only perturbation could never move a QLD1-like window
    (mean ~0.18) anywhere near SA1's real ~0.69 mean. Retargeting to an explicit (mean,
    std) pair must be able to reach that range."""
    torch.manual_seed(3)
    x = torch.rand(200, 30, 2) * 0.1 + 0.13  # mimics a QLD1-like low-mean window
    perturbed = synthetic_perturb(
        x, channel_indices=(0, 1),
        target_mean_range=(0.65, 0.70), target_std_range=(0.2, 0.25),
    )
    assert perturbed[..., 0].mean().item() > 0.5


def test_consistency_loss_is_nonnegative_and_differentiable():
    encoder = DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=8, cv_feature_dim=4)
    x = torch.rand(2, 50, 3, requires_grad=False)
    loss = consistency_loss(encoder, x)
    assert loss.item() >= 0.0
    loss.backward()


def test_consistency_loss_near_zero_for_identity_perturbation():
    torch.manual_seed(1)
    encoder = DomainInvariantEncoder(num_variables=3, horizon=12, lt_feature_dim=8, cv_feature_dim=4)
    x = torch.rand(2, 50, 3)
    from transcif.training import consistency as consistency_module

    original_perturb = consistency_module.synthetic_perturb
    consistency_module.synthetic_perturb = lambda inp, **kwargs: inp.clone()
    try:
        loss = consistency_module.consistency_loss(encoder, x)
    finally:
        consistency_module.synthetic_perturb = original_perturb
    assert loss.item() < 1e-6
