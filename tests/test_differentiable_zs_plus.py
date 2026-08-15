"""TDD tests for DifferentiableZSPlus (Task 8.1).

These tests define the contract for the differentiable ZS+ module before
implementation. Run with:
    .venv/bin/python -m pytest tests/test_differentiable_zs_plus.py -v
"""

import numpy as np
import pytest
import torch

from transcif.calibration.differentiable_zs_plus import DifferentiableZSPlus
from transcif.config import HORIZON, SEQ_LEN


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Small synthetic rs/cif series for fast unit tests."""
    torch.manual_seed(0)
    np.random.seed(0)
    T = 800  # enough for SEQ_LEN + a few origins
    rs = np.cumsum(np.random.randn(T) * 0.02).clip(0, 1).astype(np.float32)
    ef_r, ef_nr = 50.0, 500.0  # gCO2/kWh
    cif = (rs * ef_r + (1 - rs) * ef_nr).astype(np.float32)
    return rs, cif, ef_r, ef_nr


@pytest.fixture
def origins(synthetic_data):
    """Test origins aligned with build_windows TEST stride."""
    from transcif.config import TEST_STRIDE, TRAIN_FRACTION
    rs, cif, _, _ = synthetic_data
    split = int(len(rs) * TRAIN_FRACTION)
    return [split + st for st in range(0, len(cif) - split - HORIZON + 1, TEST_STRIDE)][:5]


class ConstantShareFn(torch.nn.Module):
    """A dummy share_fn that returns a learnable constant + tanh(affine(x))
    so gradient can flow back to its parameters.

    This isn't a real direction model; it just gives the differentiable ZS+
    something with parameters to backprop through.
    """

    def __init__(self, horizon=HORIZON):
        super().__init__()
        self.lin = torch.nn.Linear(SEQ_LEN, horizon)
        self.bias = torch.nn.Parameter(torch.full((horizon,), 0.5))

    def forward(self, x_window):
        # x_window: (SEQ_LEN,) torch tensor
        # Output: (horizon,) torch tensor in [0, 1]
        return torch.sigmoid(self.lin(x_window) + self.bias - 0.5)


# ---------------------------------------------------------------------------
# Test (a): forward shape
# ---------------------------------------------------------------------------

def test_forward_shape(synthetic_data, origins):
    """forward returns (n_origins, HORIZON) torch tensor."""
    rs_np, cif_np, ef_r, ef_nr = synthetic_data
    rs = torch.as_tensor(rs_np, dtype=torch.float32)
    cif = torch.as_tensor(cif_np, dtype=torch.float32)

    model = DifferentiableZSPlus()
    share_model = ConstantShareFn()
    share_fn = lambda x: share_model(x)

    out = model(rs, cif, ef_r, ef_nr, origins, share_fn)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (len(origins), HORIZON)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test (b): gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow(synthetic_data, origins):
    """Gradient flows from output back to share_fn parameters."""
    rs_np, cif_np, ef_r, ef_nr = synthetic_data
    rs = torch.as_tensor(rs_np, dtype=torch.float32)
    cif = torch.as_tensor(cif_np, dtype=torch.float32)

    model = DifferentiableZSPlus()
    share_model = ConstantShareFn()
    share_fn = lambda x: share_model(x)

    out = model(rs, cif, ef_r, ef_nr, origins, share_fn)
    # Construct a dummy target so we have a loss
    target = torch.zeros_like(out)
    loss = (out - target).pow(2).mean()
    loss.backward()

    # share_model parameters MUST have non-zero gradient
    assert share_model.lin.weight.grad is not None
    grad_norm = share_model.lin.weight.grad.norm().item()
    assert grad_norm > 0, (
        f"Gradient did not flow to share_fn parameters (grad norm = {grad_norm})"
    )


# ---------------------------------------------------------------------------
# Test (c): branches 1-4 match definitions (persistence branches)
# ---------------------------------------------------------------------------

def test_persistence_branches_match_definitions(synthetic_data, origins):
    """Branches 1, 2, 3, 4 (pure persistence / lag) match their definitions.

    These don't depend on the model, so we can check them deterministically
    by exposing the internal branch-computation method.
    """
    rs_np, cif_np, ef_r, ef_nr = synthetic_data
    rs = torch.as_tensor(rs_np, dtype=torch.float32)
    cif = torch.as_tensor(cif_np, dtype=torch.float32)

    model = DifferentiableZSPlus()
    share_model = ConstantShareFn()
    share_fn = lambda x: share_model(x)

    # Pick a representative origin
    t0 = origins[2]
    branches = model.compute_branches(t0, rs, cif, ef_r, ef_nr, share_fn)
    # Expected shape: (6, HORIZON)
    assert branches.shape == (6, HORIZON)

    # Branch 1: cif[t0-24 : t0-24+HORIZON]  (daily lag)
    expected_b1 = cif[t0 - 24 : t0 - 24 + HORIZON]
    assert torch.allclose(branches[1], expected_b1, atol=1e-5), (
        f"Branch 1 mismatch: got {branches[1, :5]}, expected {expected_b1[:5]}"
    )

    # Branch 2: cif[t0-168 : t0-168+HORIZON]  (weekly lag)
    expected_b2 = cif[t0 - 168 : t0 - 168 + HORIZON]
    assert torch.allclose(branches[2], expected_b2, atol=1e-5), (
        f"Branch 2 mismatch: got {branches[2, :5]}, expected {expected_b2[:5]}"
    )

    # Branch 3: mean of cif[t0-j*24 : t0-j*24+HORIZON] for j in 1..7
    expected_b3 = torch.stack([
        cif[t0 - j * 24 : t0 - j * 24 + HORIZON]
        for j in range(1, 8)
    ]).mean(dim=0)
    assert torch.allclose(branches[3], expected_b3, atol=1e-4), (
        f"Branch 3 mismatch: got {branches[3, :5]}, expected {expected_b3[:5]}"
    )

    # Branch 4: mean of weekly lags (j=1..4 if t0 - j*168 >= 0)
    weekly_lags = [j * 168 for j in range(1, 5) if t0 - j * 168 >= 0]
    expected_b4 = torch.stack([
        cif[t0 - lag : t0 - lag + HORIZON]
        for lag in weekly_lags
    ]).mean(dim=0)
    assert torch.allclose(branches[4], expected_b4, atol=1e-4), (
        f"Branch 4 mismatch: got {branches[4, :5]}, expected {expected_b4[:5]}"
    )


# ---------------------------------------------------------------------------
# Test (d): sane MAE on a real QLD1 setup (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_sane_mae_on_qlD1():
    """On QLD1 seed 0 with a causal direction model, DifferentiableZSPlus
    produces MAE in the sane range [10, 80].

    Non-differentiable DEFAULT ZS+ produces MAE ~27 on this setup (from
    results/fused_five_full.json QLD1 row). We don't require exact parity
    (architectures differ), but the output must be in the same order of
    magnitude — not e.g. 500 (broken) or 1 (trivially copying cif).
    """
    pytest.importorskip("transcif.data.loaders")
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from transcif.data.loaders import load_region_data, all_region_configs
    from transcif.data.windows import build_windows
    from transcif.config import TRAIN_FRACTION, TEST_STRIDE
    from transcif.models.zeroshot.causal import (
        train_causal_zero_shot, predict_causal_zs,
    )

    np.random.seed(0)
    torch.manual_seed(0)

    all_configs = all_region_configs()
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}

    target = "QLD1"
    src_names = [n for n in all_regions if n != target][:3]
    small_regions = {target: all_regions[target]}
    for n in src_names:
        small_regions[n] = all_regions[n]

    data = all_regions[target]
    config = data["config"].astype(np.float32)
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]
    rs_np, cif_np = data["rs"], data["cif"]
    rs = torch.as_tensor(rs_np, dtype=torch.float32)
    cif = torch.as_tensor(cif_np, dtype=torch.float32)

    # Train causal direction
    causal_model, _ = train_causal_zero_shot(
        small_regions, target, seed=0, device=None,
    )

    # Wrap predict as a torch share_fn
    def share_fn(x_window):
        # x_window: (SEQ_LEN,) torch tensor
        x_np = x_window.detach().cpu().numpy().astype(np.float32)[None, :]
        # predict_causal_zs expects config as 1-D (config_dim,); it unsqueezes internally
        pred_cif = predict_causal_zs(
            causal_model, x_np, config.astype(np.float32), ef_r, ef_nr,
        )[0]
        # Convert CIF back to share
        share = (pred_cif - ef_nr) / (ef_r - ef_nr + 1e-8)
        share = np.clip(share, 0.0, 1.0)
        # Return as torch tensor WITHOUT detaching — we want grad to flow
        # through some learnable transform. For this integration test, we
        # wrap in a constant (no grad expected); the test just checks MAE.
        return torch.as_tensor(share, dtype=torch.float32)

    split = int(len(rs_np) * TRAIN_FRACTION)
    origins = [
        split + st
        for st in range(0, len(cif_np) - split - HORIZON + 1, TEST_STRIDE)
    ]

    model = DifferentiableZSPlus()
    pred = model(rs, cif, ef_r, ef_nr, origins, share_fn)
    pred_np = pred.detach().cpu().numpy()

    _, _, y_true = build_windows(
        rs_np[split - SEQ_LEN:], cif_np[split - SEQ_LEN:],
        seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE,
    )
    # Align lengths
    n = min(len(pred_np), len(y_true))
    mae = float(np.abs(pred_np[:n] - y_true[:n]).mean())

    assert 10.0 <= mae <= 80.0, (
        f"DiffZS+ QLD1 MAE {mae:.2f} outside sane range [10, 80]. "
        f"Non-diff DEFAULT ZS+ typically produces ~27 on this setup."
    )


# ---------------------------------------------------------------------------
# Test (e): attention weights are valid probability distribution
# ---------------------------------------------------------------------------

def test_attention_weights_sum_to_one(synthetic_data, origins):
    """Attention weights over branches sum to 1 for each origin."""
    rs_np, cif_np, ef_r, ef_nr = synthetic_data
    rs = torch.as_tensor(rs_np, dtype=torch.float32)
    cif = torch.as_tensor(cif_np, dtype=torch.float32)

    model = DifferentiableZSPlus()
    share_model = ConstantShareFn()
    share_fn = lambda x: share_model(x)

    t0 = origins[1]
    weights = model.compute_attention_weights(t0, rs, cif, ef_r, ef_nr, share_fn)
    # Should be (6,) and sum to 1
    assert weights.shape == (6,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5), (
        f"Weights sum to {weights.sum().item()}, expected 1.0"
    )
    assert (weights >= 0).all() and (weights <= 1).all()
