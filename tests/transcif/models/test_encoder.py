import torch
from transcif.models.encoder import DomainInvariantEncoder


def test_encoder_output_shape_and_range():
    model = DomainInvariantEncoder(num_variables=3, horizon=24, lt_feature_dim=16, cv_feature_dim=8)
    x = torch.rand(4, 96, 3)
    renew_share_pred, dominant_idx = model(x)

    assert renew_share_pred.shape == (4, 24)
    assert torch.all(renew_share_pred >= 0.0) and torch.all(renew_share_pred <= 1.0)
    assert dominant_idx.dim() == 4


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
