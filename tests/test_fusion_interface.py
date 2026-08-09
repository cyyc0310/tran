"""Interface tests for the 5-direction FusionModel (Task 1.1).

These tests pin the public contract that Phase 1.2/1.3 and the ZS+ calibration
pipeline will rely on. Real training behavior (LOO-CV, diversity reg) is
exercised in later tasks; here we only assert shapes, ranges, and the
configuration flow that ``zs_plus_predict(..., share_fn=...)`` requires.
"""

import numpy as np
import pytest
import torch

from transcif.config import HORIZON, SEQ_LEN
from transcif.models.zeroshot.fusion import (
    BasisMixFusion,
    DIRECTION_ORDER,
    EqualWeightFusion,
    FusionHead,
    FusionModel,
    MedianFusion,
    basis_mix_loss,
    train_fusion,
)


def _stub_predictor(value: float):
    """Return a single-window predictor that ignores its inputs and emits a
    constant CIF array. Good enough for shape/range assertions."""

    def _pred(x_window, config, ef_r, ef_nr):
        assert x_window.shape == (SEQ_LEN,)
        return np.full(HORIZON, float(value), dtype=np.float32)

    return _pred


def _stub_predictor_set(values=(500.0, 510.0, 490.0, 505.0, 495.0)):
    return {name: _stub_predictor(values[i])
            for i, name in enumerate(DIRECTION_ORDER)}


# ---------------------------------------------------------------------------
# FusionHead
# ---------------------------------------------------------------------------

def test_fusion_head_forward_shape():
    head = FusionHead()
    stack = np.random.randn(8, 5, HORIZON).astype(np.float32)
    out = head(stack)
    assert out.shape == (8, HORIZON)


def test_fusion_head_weights_sum_to_one():
    head = FusionHead()
    weights = head.weights()
    assert weights.shape == (5,)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= 0).all()


# ---------------------------------------------------------------------------
# train_fusion
# ---------------------------------------------------------------------------

def test_train_fusion_returns_model_with_predictors_attached():
    rng = np.random.default_rng(0)
    src_stacks = [rng.normal(size=(10, 5, HORIZON)).astype(np.float32)
                  for _ in range(3)]
    src_true = [rng.normal(size=(10, HORIZON)).astype(np.float32)
                for _ in range(3)]
    predictors = _stub_predictor_set()

    model = train_fusion(src_stacks, src_true, predictors=predictors,
                         epochs=2, seed=0)

    assert isinstance(model, FusionModel)
    assert model.head is not None
    assert set(model.predictors.keys()) == set(predictors.keys())
    for name in predictors:
        assert model.predictors[name] is predictors[name]


def test_train_fusion_accepts_empty_predictors():
    """Predictors are optional at training time so the head can be trained
    independently and the predictors wired up at evaluation time."""
    rng = np.random.default_rng(1)
    src_stacks = [rng.normal(size=(4, 5, HORIZON)).astype(np.float32)]
    src_true = [rng.normal(size=(4, HORIZON)).astype(np.float32)]

    model = train_fusion(src_stacks, src_true, predictors=None,
                         epochs=1, seed=0)
    assert isinstance(model, FusionModel)


# ---------------------------------------------------------------------------
# FusionModel.predict_cif_from_stack — pure combiner (no predictors needed)
# ---------------------------------------------------------------------------

def test_predict_cif_from_stack_shape():
    head = FusionHead()
    model = FusionModel(head, predictors=None)
    stack = np.random.randn(6, 5, HORIZON).astype(np.float32)
    out = model.predict_cif_from_stack(stack)
    assert out.shape == (6, HORIZON)


def test_predict_cif_from_stack_matches_head_forward():
    head = FusionHead()
    model = FusionModel(head, predictors=None)
    stack = np.random.randn(4, 5, HORIZON).astype(np.float32)
    with torch.no_grad():
        expected = head(stack).cpu().numpy()
    out = model.predict_cif_from_stack(stack)
    assert np.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# FusionModel.predict_cif — end-to-end (requires predictors)
# ---------------------------------------------------------------------------

def test_predict_cif_end_to_end_shape():
    predictors = _stub_predictor_set()
    head = FusionHead()
    model = FusionModel(head, predictors=predictors)

    x_rs = np.random.randn(4, SEQ_LEN).astype(np.float32)
    config = np.array([0.4, 100.0], dtype=np.float32)
    out = model.predict_cif(x_rs, config, ef_r=600.0, ef_nr=400.0)
    assert out.shape == (4, HORIZON)


def test_predict_cif_requires_predictors():
    head = FusionHead()
    model = FusionModel(head, predictors=None)
    x_rs = np.random.randn(2, SEQ_LEN).astype(np.float32)
    config = np.array([0.4, 100.0], dtype=np.float32)
    with pytest.raises(RuntimeError, match="predictors"):
        model.predict_cif(x_rs, config, ef_r=600.0, ef_nr=400.0)


# ---------------------------------------------------------------------------
# FusionModel.share_fn — for zs_plus_predict integration
# ---------------------------------------------------------------------------

def test_share_fn_requires_target_config():
    predictors = _stub_predictor_set()
    head = FusionHead()
    model = FusionModel(head, predictors=predictors)
    x_win = np.random.randn(SEQ_LEN).astype(np.float32)
    with pytest.raises(RuntimeError, match="configure_for_target"):
        model.share_fn(x_win)


def test_share_fn_shape_and_range():
    predictors = _stub_predictor_set(values=(600.0, 550.0, 500.0,
                                             450.0, 400.0))
    head = FusionHead()
    model = FusionModel(head, predictors=predictors)
    model.configure_for_target(config=np.array([0.4, 100.0], dtype=np.float32),
                               ef_r=600.0, ef_nr=400.0)

    x_win = np.random.randn(SEQ_LEN).astype(np.float32)
    share = model.share_fn(x_win)

    assert share.shape == (HORIZON,)
    assert np.isfinite(share).all()
    assert (share >= 0.0).all() and (share <= 1.0).all()


def test_share_fn_clips_when_predictor_exceeds_ef_range():
    """A predictor returning CIF > ef_r must clip to share ≤ 1, and a
    predictor returning CIF < ef_nr must clip to share ≥ 0. This guards the
    physical-inversion step used downstream by ZS+."""
    predictors = _stub_predictor_set(values=(900.0, 50.0, 500.0,
                                             500.0, 500.0))
    head = FusionHead()
    model = FusionModel(head, predictors=predictors)
    model.configure_for_target(config=np.array([0.4, 100.0], dtype=np.float32),
                               ef_r=600.0, ef_nr=400.0)

    x_win = np.random.randn(SEQ_LEN).astype(np.float32)
    share = model.share_fn(x_win)
    assert (share >= 0.0).all() and (share <= 1.0).all()


# ---------------------------------------------------------------------------
# EqualWeightFusion (Task 2.1)
# ---------------------------------------------------------------------------

def test_equal_weight_forward_shape_and_value():
    """EqualWeightFusion should output the mean of all 5 directions."""
    head = EqualWeightFusion()
    # Stack where first direction is clearly different
    stack = np.zeros((4, 5, HORIZON), dtype=np.float32)
    stack[:, 0, :] = 600.0  # First direction always 600
    stack[:, 1, :] = 500.0
    stack[:, 2, :] = 500.0
    stack[:, 3, :] = 500.0
    stack[:, 4, :] = 500.0

    out = head(stack)
    assert out.shape == (4, HORIZON)
    # Mean of (600, 500, 500, 500, 500) = 520
    expected = np.full((4, HORIZON), 520.0)
    assert np.allclose(out.cpu().numpy(), expected, atol=1e-6)


def test_equal_weight_weights():
    """EqualWeightFusion weights() should return uniform (0.2, 0.2, 0.2, 0.2, 0.2)."""
    head = EqualWeightFusion()
    weights = head.weights()
    assert weights.shape == (5,)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(weights, torch.ones(5) / 5)


# ---------------------------------------------------------------------------
# MedianFusion (Task 2.1)
# ---------------------------------------------------------------------------

def test_median_ignores_outlier_direction():
    """MedianFusion should ignore outlier values like Hier's failures."""
    head = MedianFusion()
    # Stack with 4 good directions (~500) and 1 outlier (9999, like Hier failure)
    stack = np.zeros((4, 5, HORIZON), dtype=np.float32)
    stack[:, 0, :] = 500.0
    stack[:, 1, :] = 510.0
    stack[:, 2, :] = 490.0
    stack[:, 3, :] = 505.0
    stack[:, 4, :] = 9999.0  # Outlier

    out = head(stack)
    assert out.shape == (4, HORIZON)
    # Median of (500, 510, 490, 505, 9999) = 505 (ignores 9999)
    expected = np.full((4, HORIZON), 505.0)
    assert np.allclose(out.cpu().numpy(), expected, atol=1e-6)


def test_median_weights():
    """MedianFusion weights() should return uniform weights for interface consistency."""
    head = MedianFusion()
    weights = head.weights()
    assert weights.shape == (5,)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(weights, torch.ones(5) / 5)


# ---------------------------------------------------------------------------
# BasisMixFusion (Task 3.1)
# ---------------------------------------------------------------------------

def test_basis_mix_forward_shape():
    """BasisMixFusion should have the same shape contract as FusionHead."""
    head = BasisMixFusion()
    stack = np.random.randn(8, 5, HORIZON).astype(np.float32)
    out = head(stack)
    assert out.shape == (8, HORIZON)


def test_basis_mix_entropy_floor_penalizes_collapse():
    """Entropy floor should penalize one-hot weight collapse."""
    head = BasisMixFusion()

    # Set logit to extreme values -> near one-hot weights
    with torch.no_grad():
        head.logit.data = torch.tensor([10.0, -10.0, -10.0, -10.0, -10.0])

    loss = head.entropy_floor_loss()
    assert loss.item() > 0, "Entropy floor should penalize one-hot weights"

    # Set logit to zeros -> uniform weights
    with torch.no_grad():
        head.logit.data = torch.zeros(5)

    loss_uniform = head.entropy_floor_loss()
    assert loss_uniform.item() == pytest.approx(0.0, abs=1e-6), \
        "Entropy floor should be zero for uniform weights"


def test_basis_mix_diversity_loss_zero_for_orthogonal():
    """Diversity loss should be ~0 for orthogonal predictions."""
    head = BasisMixFusion()

    # Create 5 orthogonal predictions using an identity-like structure
    n = 10
    stack = torch.zeros(n, 5, HORIZON)
    for d in range(5):
        # Each direction has energy in a different slice of the horizon
        start = d * (HORIZON // 5)
        end = start + (HORIZON // 5)
        stack[:, d, start:end] = torch.randn(n, end - start)

    loss = head.diversity_loss(stack)
    # Low cosine similarity for orthogonal predictions
    assert loss.item() < 0.5, f"Diversity loss should be low for orthogonal predictions, got {loss.item()}"


def test_basis_mix_diversity_loss_high_for_identical():
    """Diversity loss should be > 0 when all directions are identical."""
    head = BasisMixFusion()

    # All 5 directions produce identical predictions
    n = 10
    base_pred = torch.randn(n, HORIZON)
    stack = base_pred.unsqueeze(1).expand(n, 5, HORIZON).clone()

    loss = head.diversity_loss(stack)
    assert loss.item() > 0.5, f"Diversity loss should be high for identical predictions, got {loss.item()}"


def test_basis_mix_loss_combines_terms():
    """basis_mix_loss should combine MAE + L2 + entropy + diversity terms."""
    head = BasisMixFusion()
    cif_stack = torch.randn(4, 5, HORIZON)
    y_true = torch.randn(4, HORIZON)

    loss = basis_mix_loss(
        head, cif_stack, y_true,
        lambda_l2=1e-3,
        lambda_entropy=1e-2,
        lambda_diversity=1e-2
    )

    assert loss.item() > 0
    assert torch.isfinite(loss).all()

    # Verify differentiability
    loss.backward()
    assert head.logit.grad is not None
    assert torch.isfinite(head.logit.grad).all()


def test_basis_mix_weights_sum_to_one():
    """BasisMixFusion weights should sum to 1 (softmax property)."""
    head = BasisMixFusion()
    weights = head.weights()
    assert weights.shape == (5,)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= 0).all(), "Softmax weights should be non-negative"
