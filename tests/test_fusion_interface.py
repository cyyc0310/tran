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
    DIRECTION_ORDER,
    FusionHead,
    FusionModel,
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
