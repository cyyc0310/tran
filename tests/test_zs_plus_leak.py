"""Leak guards for the ZS+ calibration protocol.

The ZS+ information set (docs/PROTOCOL.md §2.2) allows target history
strictly BEFORE the forecast origin ``t0``.  These tests pin that contract:

1. **No future information**: mutating ``rs``/``cif`` at or after ``t0``
   must not change the prediction at ``t0`` — for both the numpy
   ``zs_plus_predict`` and the differentiable ``DifferentiableZSPlus``.
   A single wrong slice direction (e.g. ``cif[t0 + 24]`` in a backtest)
   would flip these assertions red.
2. **Origin alignment**: the origin lists used by the orchestrators
   (``scripts.experiments._shared.zs_plus_origins``) must line up exactly
   with ``build_windows`` TEST output, so metrics compare predictions
   against the right ground truth rows.
"""

import numpy as np
import torch

from transcif.calibration.differentiable_zs_plus import DifferentiableZSPlus
from transcif.calibration.zs_plus import zs_plus_predict
from transcif.config import (
    HORIZON, SEQ_LEN, TEST_STRIDE, TRAIN_FRACTION,
)
from transcif.data.windows import build_windows
from scripts.experiments._shared import zs_plus_origins

EF_R, EF_NR = 0.0, 600.0


def _smooth_series(n_hours=1400, seed=0):
    """Smooth synthetic rs/cif so the calibration selector has signal.

    cif follows the physics identity (plus small observation noise) exactly
    like the real dataset.
    """
    rng = np.random.default_rng(seed)
    rs = np.cumsum(rng.normal(0, 0.02, n_hours)).clip(0.05, 0.95).astype(np.float32)
    noise = rng.normal(0, 2.0, n_hours).astype(np.float32)
    cif = (rs * EF_R + (1 - rs) * EF_NR + noise).astype(np.float32)
    return rs, cif


def _test_origins(rs, n=4):
    split = int(len(rs) * TRAIN_FRACTION)
    return [split + st for st in range(0, n * TEST_STRIDE, TEST_STRIDE)]


class _FixedShareModel(torch.nn.Module):
    """Tiny stand-in satisfying the ``model(x, config) -> share`` contract."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(SEQ_LEN, HORIZON)

    def forward(self, x, config):
        return torch.sigmoid(self.lin(x))


class _ConstantShareFn(torch.nn.Module):
    """Learnable share map for the differentiable ZS+ path."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(SEQ_LEN, HORIZON)

    def forward(self, x_window):
        return torch.sigmoid(self.lin(x_window))


# ---------------------------------------------------------------------------
# 1. Origin alignment with build_windows TEST output
# ---------------------------------------------------------------------------

def test_zs_plus_origins_align_with_build_windows():
    """zs_plus_origins must enumerate exactly the build_windows TEST rows:
    same count, and origin t0 corresponds to ground truth cif[t0:t0+H]."""
    rs, cif = _smooth_series()
    split = int(len(rs) * TRAIN_FRACTION)

    _, _, y_cif_test = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
        seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE,
    )
    origins = zs_plus_origins(rs, cif)

    assert len(origins) == len(y_cif_test), (
        f"origin count {len(origins)} != TEST window count {len(y_cif_test)}: "
        f"metrics would be computed against misaligned ground truth"
    )
    for i, t0 in enumerate(origins):
        assert np.array_equal(y_cif_test[i], cif[t0:t0 + HORIZON]), (
            f"origin {i} (t0={t0}) does not match TEST window {i}: "
            f"evaluation is comparing against the wrong hours"
        )


# ---------------------------------------------------------------------------
# 2. No future information (numpy zs_plus_predict)
# ---------------------------------------------------------------------------

def test_numpy_zs_plus_no_future_leak():
    """Mutating the series at or after t0 must leave pred(t0) untouched.

    The ZS+ branches, anchor, residual and backtest windows may only read
    indices < t0.  The sentinel rewrite below would corrupt every one of
    those reads if any slice reached into the future.
    """
    rs, cif = _smooth_series()
    origins = _test_origins(rs)
    t0 = origins[1]
    model = _FixedShareModel()
    config = np.array([0.4, EF_NR / 1000.0], dtype=np.float32)

    pred_clean = zs_plus_predict(model, config, rs, cif, EF_R, EF_NR, [t0])

    rs_dirty = rs.copy()
    cif_dirty = cif.copy()
    rs_dirty[t0:] = 0.0          # extreme sentinel share from t0 onward
    cif_dirty[t0:] = 99999.0     # extreme sentinel CIF from t0 onward
    pred_dirty = zs_plus_predict(
        model, config, rs_dirty, cif_dirty, EF_R, EF_NR, [t0])

    assert np.allclose(pred_clean, pred_dirty, atol=1e-9), (
        f"zs_plus_predict read data at/after t0={t0}: prediction changed "
        f"from {pred_clean[0, :3]} to {pred_dirty[0, :3]} when only future "
        f"values were mutated"
    )


def test_numpy_zs_plus_no_leak_across_origins():
    """Batch prediction at several origins: dirtying the LATER part of the
    series must not change predictions at EARLIER origins."""
    rs, cif = _smooth_series(seed=1)
    origins = _test_origins(rs)
    model = _FixedShareModel()
    config = np.array([0.4, EF_NR / 1000.0], dtype=np.float32)

    pred_clean = zs_plus_predict(model, config, rs, cif, EF_R, EF_NR, origins)

    t_cut = origins[1]           # dirty everything from the 2nd origin on
    rs2, cif2 = rs.copy(), cif.copy()
    rs2[t_cut:] = 0.0
    cif2[t_cut:] = 99999.0
    pred_dirty = zs_plus_predict(model, config, rs2, cif2, EF_R, EF_NR, origins)

    assert np.allclose(pred_clean[0], pred_dirty[0], atol=1e-9), (
        "prediction at origin 0 changed when only data after origin 1 "
        "was mutated — cross-origin future leak"
    )


# ---------------------------------------------------------------------------
# 3. No future information (differentiable ZS+)
# ---------------------------------------------------------------------------

def test_differentiable_zs_plus_no_future_leak():
    """Same sentinel contract for the torch-native DifferentiableZSPlus
    consumed by the Phase-9 joint-training pipeline."""
    rs, cif = _smooth_series(seed=2)
    origins = _test_origins(rs)
    t0 = origins[1]

    zs = DifferentiableZSPlus()
    share = _ConstantShareFn()

    rs_t = torch.as_tensor(rs, dtype=torch.float32)
    cif_t = torch.as_tensor(cif, dtype=torch.float32)

    pred_clean = zs(rs_t, cif_t, EF_R, EF_NR, [t0],
                    lambda x: share(x)).detach()

    rs_d = rs_t.clone()
    cif_d = cif_t.clone()
    rs_d[t0:] = 0.0
    cif_d[t0:] = 99999.0
    pred_dirty = zs(rs_d, cif_d, EF_R, EF_NR, [t0],
                    lambda x: share(x)).detach()

    assert torch.allclose(pred_clean, pred_dirty, atol=1e-9), (
        f"DifferentiableZSPlus read data at/after t0={t0}: prediction "
        f"changed from {pred_clean[0, :3]} to {pred_dirty[0, :3]} when only "
        f"future values were mutated"
    )
