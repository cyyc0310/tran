import torch
from transcif.models.encoder import DomainInvariantEncoder
from transcif.training.consistency import synthetic_perturb, consistency_loss


def test_synthetic_perturb_keeps_renew_share_in_unit_interval():
    x = torch.rand(4, 30, 3)
    perturbed = synthetic_perturb(x, renew_share_idx=0, scale_range=(0.7, 1.3))
    assert perturbed.shape == x.shape
    assert torch.all(perturbed[..., 0] >= 0.0) and torch.all(perturbed[..., 0] <= 1.0)


def test_synthetic_perturb_leaves_other_channels_untouched():
    x = torch.rand(2, 20, 3)
    perturbed = synthetic_perturb(x, renew_share_idx=0, scale_range=(0.5, 1.5))
    torch.testing.assert_close(perturbed[..., 1:], x[..., 1:])


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
