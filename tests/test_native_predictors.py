"""Tests for torch-native zero-shot direction wrappers + LearnedFusion (Phase 9).

TDD: these validate the differentiable predictor contract before the joint
training pipeline consumes it. The key properties:
  - Native{Phys,Causal,Hier}.forward_cif preserves the gradient graph back to
    the underlying model parameters.
  - FrozenConstant.forward_cif returns a graph-detached constant (RAG/ICL).
  - LearnedFusion produces per-window softmax weights summing to 1 and is
    differentiable.
"""
import numpy as np
import pytest
import torch

from transcif.config import HORIZON, SEQ_LEN
from transcif.models.base import AdaptivePersistDLinear
from transcif.models.zeroshot.causal import CausalDomainVAE
from transcif.models.zeroshot.hier import HierDLinear
from transcif.models.zeroshot.native import (
    FrozenConstant,
    LearnedFusion,
    NativeCausal,
    NativeHier,
    NativeICL,
    NativePhys,
    NativeRAG,
    TorchNativePredictor,
    pad_config_t,
)

EF_R, EF_NR = 0.0, 600.0
CONFIG_DIM = 2


def _x_config(batch=2, seq_len=SEQ_LEN, config_dim=CONFIG_DIM):
    torch.manual_seed(0)
    x = torch.rand(batch, seq_len, dtype=torch.float32)
    config = torch.rand(batch, config_dim, dtype=torch.float32)
    return x, config


# ---------------------------------------------------------------------------
# NativePhys
# ---------------------------------------------------------------------------

def test_native_phys_forward_cif_shape_and_grad():
    model = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=CONFIG_DIM)
    native = NativePhys(model)
    x, config = _x_config()
    cif = native.forward_cif(x, config, EF_R, EF_NR)
    assert cif.shape == (x.shape[0], HORIZON)
    loss = cif.sum()
    loss.backward()
    # gradient must reach the underlying model parameters
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradient reached NativePhys model params"
    assert all(g.abs().sum() > 0 for g in grads)


def test_native_phys_matches_manual_physics():
    """forward_cif should equal share*ef_r + (1-share)*ef_nr with share=model(x,c)."""
    model = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=CONFIG_DIM).eval()
    native = NativePhys(model)
    x, config = _x_config()
    with torch.no_grad():
        share = model(x, config)
        expected = share * EF_R + (1 - share) * EF_NR
    out = native.forward_cif(x, config, EF_R, EF_NR)
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# NativeHier
# ---------------------------------------------------------------------------

def test_native_hier_forward_cif_grad():
    model = HierDLinear(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=CONFIG_DIM)
    native = NativeHier(model)
    x, config = _x_config()
    cif = native.forward_cif(x, config, EF_R, EF_NR)
    assert cif.shape == (x.shape[0], HORIZON)
    cif.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())


# ---------------------------------------------------------------------------
# NativeCausal
# ---------------------------------------------------------------------------

def test_native_causal_forward_cif_grad():
    model = CausalDomainVAE(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=CONFIG_DIM)
    native = NativeCausal(model)
    x, config = _x_config()
    cif = native.forward_cif(x, config, EF_R, EF_NR)
    assert cif.shape == (x.shape[0], HORIZON)
    cif.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())


# ---------------------------------------------------------------------------
# NativeRAG
# ---------------------------------------------------------------------------

def test_native_rag_forward_cif_grad():
    """Retrieval is differentiable; model params (rag_proj/gate) receive grad."""
    from transcif.models.zeroshot.rag import RagDLinear
    model = RagDLinear(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=CONFIG_DIM)
    # dummy bank: 20 random contexts + share targets
    N = 20
    bank_X = torch.rand(N, SEQ_LEN, dtype=torch.float32)
    bank_Y = torch.rand(N, HORIZON, dtype=torch.float32)
    native = NativeRAG(model, bank_X, bank_Y, k=5)
    x, config = _x_config()
    cif = native.forward_cif(x, config, EF_R, EF_NR)
    assert cif.shape == (x.shape[0], HORIZON)
    cif.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(grads) > 0, "no gradient reached NativeRAG model params"
    # bank buffers must NOT be in the optimizer param set (they are buffers)
    assert native.bank_X.requires_grad is False


# ---------------------------------------------------------------------------
# NativeICL
# ---------------------------------------------------------------------------

def test_native_icl_forward_cif_grad():
    """Context retrieval is no-grad preprocessing; transformer params get grad."""
    from transcif.models.zeroshot.icl import ICTransformer
    model = ICTransformer(horizon=HORIZON, n_examples=3)
    # minimal source regions with enough data to build windows
    rng = np.random.default_rng(0)
    T = SEQ_LEN + HORIZON + 50
    all_regions = {
        "TARGET": {"rs": rng.random(T).astype(np.float32),
                   "cif": rng.random(T).astype(np.float32) * 500,
                   "config": np.array([0.4, 0.5], dtype=np.float32)},
        "SRC1": {"rs": rng.random(T).astype(np.float32),
                 "cif": rng.random(T).astype(np.float32) * 500,
                 "config": np.array([0.4, 0.5], dtype=np.float32)},
        "SRC2": {"rs": rng.random(T).astype(np.float32),
                 "cif": rng.random(T).astype(np.float32) * 500,
                 "config": np.array([0.45, 0.5], dtype=np.float32)},
    }
    native = NativeICL(model, all_regions, "TARGET", n_examples=3, horizon=HORIZON)
    x, _config = _x_config(batch=1)
    cif = native.forward_cif(x, _config, EF_R, EF_NR)
    assert cif.shape == (1, HORIZON)
    cif.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(grads) > 0, "no gradient reached ICTransformer params"


# ---------------------------------------------------------------------------
# FrozenConstant
# ---------------------------------------------------------------------------

def test_frozen_constant_blocks_gradient():
    """A learnable scalar should NOT receive gradient through FrozenConstant."""
    elsewhere = torch.nn.Linear(3, 3)  # a dummy module with params
    # predict_fn ignores its args and returns a fixed value
    def predict_fn(x_rs, config, ef_r, ef_nr):
        return np.ones((x_rs.shape[0], HORIZON), dtype=np.float32) * 5.0

    frozen = FrozenConstant(predict_fn)
    x, config = _x_config()
    out = frozen.forward_cif(x, config, EF_R, EF_NR)
    assert out.requires_grad is False
    # composing with a learnable term and backprop must not error / must be None
    other = elsewhere(torch.rand(1, 3)).sum()
    loss = out.sum() + other
    loss.backward()  # should not raise
    assert elsewhere.weight.grad is not None  # the live branch got grad


# ---------------------------------------------------------------------------
# LearnedFusion
# ---------------------------------------------------------------------------

def test_learned_fusion_shape_and_weight_normalization():
    fusion = LearnedFusion(n_directions=5, config_dim=CONFIG_DIM, horizon=HORIZON)
    batch = 4
    stack = torch.randn(batch, 5, HORIZON, requires_grad=True)
    config = torch.rand(batch, CONFIG_DIM)
    out, weights = fusion(stack, config, return_weights=True)
    assert out.shape == (batch, HORIZON)
    assert weights.shape == (batch, 5)
    # each window's weights sum to 1
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch), atol=1e-5)


def test_learned_fusion_is_differentiable_and_per_window():
    """Weights must vary across windows (not a global softmax) and carry grad."""
    fusion = LearnedFusion(n_directions=5, config_dim=CONFIG_DIM, horizon=HORIZON)
    batch = 3
    stack = torch.randn(batch, 5, HORIZON)
    config = torch.rand(batch, CONFIG_DIM)
    out, weights = fusion(stack, config, return_weights=True)
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in fusion.parameters())
    # weights differ across the 3 windows (per-window conditioning)
    assert not torch.allclose(weights[0], weights[1])


# ---------------------------------------------------------------------------
# pad_config_t
# ---------------------------------------------------------------------------

def test_pad_config_t_matches_numpy():
    from transcif.physics.bounds import pad_config
    cfg = np.array([0.5, 0.3], dtype=np.float32)
    np_padded = pad_config(cfg, 5)
    t_padded = pad_config_t(torch.tensor(cfg), 5)
    assert t_padded.shape == (5,)
    assert torch.allclose(t_padded, torch.tensor(np_padded, dtype=torch.float32))


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------

def test_torch_native_predictor_is_abstract():
    """TorchNativePredictor.forward_cif must be overridden."""
    class Bad(TorchNativePredictor):
        pass
    with pytest.raises(TypeError):
        Bad()
