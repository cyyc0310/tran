"""TDD tests for adversarial-persistence loss (Task 8.2).

The loss explicitly rewards beating the persistence baseline by a relative
margin. Tests define the contract before implementation:

    L = ReLU(pred_mae - persistence_mae * (1 - margin)).mean()

Run with:
    .venv/bin/python -m pytest tests/test_adversarial_loss.py -v
"""

import torch

from transcif.training.adversarial_loss import adversarial_persistence_loss


def test_pred_equals_persistence():
    """When pred_mae == persistence_mae per window, loss = mean(margin * persistence_mae)."""
    pred = torch.tensor([50.0, 100.0])
    persistence = torch.tensor([50.0, 100.0])
    margin = 0.10
    # Per window: ReLU(50 - 50*0.9) = ReLU(5) = 5; ReLU(100 - 90) = 10
    # mean = 7.5
    loss = adversarial_persistence_loss(pred, persistence, margin=margin)
    expected = torch.tensor(7.5)
    assert torch.allclose(loss, expected, atol=1e-5), (
        f"Expected {expected}, got {loss}"
    )


def test_pred_beats_persistence_by_margin():
    """When pred_mae == persistence_mae * (1 - margin), loss = 0 (ReLU saturated)."""
    persistence = torch.tensor([50.0, 100.0, 200.0])
    margin = 0.10
    pred = persistence * (1 - margin)  # exactly at the margin boundary
    loss = adversarial_persistence_loss(pred, persistence, margin=margin)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), (
        f"Expected 0.0 at margin boundary, got {loss}"
    )


def test_pred_worse_than_persistence():
    """When pred_mae > persistence_mae * (1 + margin), loss grows linearly."""
    persistence = torch.tensor([100.0])
    margin = 0.10
    pred = torch.tensor([150.0])  # way worse
    # threshold = 100 * 0.9 = 90; ReLU(150 - 90) = 60
    loss = adversarial_persistence_loss(pred, persistence, margin=margin)
    expected = torch.tensor(60.0)
    assert torch.allclose(loss, expected, atol=1e-5), (
        f"Expected {expected}, got {loss}"
    )


def test_gradient_active_region_nonzero():
    """In the active region (ReLU > 0), gradient on pred is +1 per window."""
    pred = torch.tensor([80.0, 120.0], requires_grad=True)
    persistence = torch.tensor([50.0, 100.0])
    margin = 0.10
    # Window 0: ReLU(80 - 45) = 35 (active)
    # Window 1: ReLU(120 - 90) = 30 (active)
    # mean = 32.5
    # d(mean)/d(pred[0]) = 0.5 (1/N × 1)
    # d(mean)/d(pred[1]) = 0.5
    loss = adversarial_persistence_loss(pred, persistence, margin=margin)
    loss.backward()
    grad = pred.grad
    assert grad is not None
    assert (grad > 0).all(), f"Expected nonzero positive gradients, got {grad}"
    expected_grad = torch.tensor([0.5, 0.5])
    assert torch.allclose(grad, expected_grad, atol=1e-5), (
        f"Expected {expected_grad}, got {grad}"
    )


def test_shape_handling():
    """Function accepts (N,) and (N, H). For (N, H), per-window MAE is along axis=1."""
    # 1-D case
    pred_1d = torch.tensor([60.0, 110.0])
    persistence_1d = torch.tensor([50.0, 100.0])
    margin = 0.10
    loss_1d = adversarial_persistence_loss(pred_1d, persistence_1d, margin=margin)
    # Window 0: ReLU(60 - 45) = 15
    # Window 1: ReLU(110 - 90) = 20
    # mean = 17.5
    assert torch.allclose(loss_1d, torch.tensor(17.5), atol=1e-5)

    # 2-D case: per-window MAE first, then ReLU margin loss
    # Window 0 has 3 elements with mean abs = 60; window 1 has mean abs = 110
    pred_2d = torch.tensor([[60.0, 60.0, 60.0], [110.0, 110.0, 110.0]])
    persistence_2d = torch.tensor([[50.0, 50.0, 50.0], [100.0, 100.0, 100.0]])
    loss_2d = adversarial_persistence_loss(pred_2d, persistence_2d, margin=margin)
    # Should match the 1-D case because per-window means are identical
    assert torch.allclose(loss_2d, loss_1d, atol=1e-5), (
        f"2-D loss {loss_2d} should match 1-D loss {loss_1d}"
    )


def test_margin_parameter():
    """Different margin values shift the threshold."""
    pred = torch.tensor([95.0])
    persistence = torch.tensor([100.0])

    # margin=0.0: threshold = 100; ReLU(95 - 100) = 0
    loss_0 = adversarial_persistence_loss(pred, persistence, margin=0.0)
    assert torch.allclose(loss_0, torch.tensor(0.0), atol=1e-6)

    # margin=0.10: threshold = 90; ReLU(95 - 90) = 5
    loss_10 = adversarial_persistence_loss(pred, persistence, margin=0.10)
    assert torch.allclose(loss_10, torch.tensor(5.0), atol=1e-5)

    # margin=0.20: threshold = 80; ReLU(95 - 80) = 15
    loss_20 = adversarial_persistence_loss(pred, persistence, margin=0.20)
    assert torch.allclose(loss_20, torch.tensor(15.0), atol=1e-5)
