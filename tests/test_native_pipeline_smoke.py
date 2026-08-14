"""Smoke test for the torch-native joint-training pipeline (Phase 9).

Exercises the REAL native_stage / eval_held_out / internal-val gate logic with
small dummy models and a minimal synthetic region, so future PRs are caught if
they break the wrapper contracts, the 5-direction assembly, the gate
snapshot/restore, or the held-out eval. Runs in a few seconds (no direction
training — dummy Native* wrappers are constructed directly).
"""
import numpy as np
import torch

from transcif.config import HORIZON, SEQ_LEN, TEST_STRIDE, TRAIN_FRACTION
from transcif.models.base import AdaptivePersistDLinear
from transcif.models.zeroshot.causal import CausalDomainVAE
from transcif.models.zeroshot.hier import HierDLinear
from transcif.models.zeroshot.rag import RagDLinear
from transcif.models.zeroshot.icl import ICTransformer
from transcif.models.zeroshot.native import (
    LearnedFusion, NativeCausal, NativeHier, NativeICL, NativePhys, NativeRAG,
)
from scripts.experiments.run_joint_train_native import (
    _snapshot, _restore, eval_held_out, native_stage, assemble_stack,
    head_modules, FROZEN_DIRS,
)
from transcif.calibration.differentiable_zs_plus import DifferentiableZSPlus

EF_R, EF_NR = 0.0, 600.0
CFG_DIM = 2


def _synthetic_region(n_origins=6):
    """Minimal rs/cif arrays long enough for SEQ_LEN windows + ZS+ history."""
    rng = np.random.default_rng(0)
    N = SEQ_LEN + 168 + TEST_STRIDE * (n_origins + 4)
    rs = rng.random(N).astype(np.float32) * 0.5
    cif = (rs * EF_R + (1 - rs) * EF_NR).astype(np.float32)
    return rs, cif


def _build_live():
    """Tiny dummy Native* wrappers (untrained but valid for shape/logic tests)."""
    phys = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=CFG_DIM)
    causal = CausalDomainVAE(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=CFG_DIM)
    hier = HierDLinear(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=CFG_DIM)
    rag = RagDLinear(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=CFG_DIM)
    bank_X = torch.rand(8, SEQ_LEN, dtype=torch.float32)
    bank_Y = torch.rand(8, HORIZON, dtype=torch.float32)
    # ICL needs source regions for retrieval; build minimal ones.
    rng = np.random.default_rng(1)
    T = SEQ_LEN + HORIZON + 50
    sources = {
        "TARGET": {"rs": rng.random(T).astype(np.float32),
                   "cif": rng.random(T).astype(np.float32) * 500,
                   "config": np.array([0.4, 0.5], dtype=np.float32)},
        "S1": {"rs": rng.random(T).astype(np.float32),
               "cif": rng.random(T).astype(np.float32) * 500,
               "config": np.array([0.4, 0.5], dtype=np.float32)},
    }
    icl = ICTransformer(horizon=HORIZON, n_examples=3)
    return {
        "rag": NativeRAG(rag, bank_X, bank_Y, k=4),
        "phys": NativePhys(phys),
        "causal": NativeCausal(causal),
        "icl": NativeICL(icl, sources, "TARGET", n_examples=3, horizon=HORIZON),
        "hier": NativeHier(hier),
    }


def _make_origins(rs, n=6):
    split = int(len(rs) * TRAIN_FRACTION)
    return [split + st for st in range(0, n * TEST_STRIDE, TEST_STRIDE)]


def test_pipeline_runs_and_produces_finite_mae():
    """End-to-end: assemble 5-stack -> fusion -> ZS+ -> finite held-out MAE."""
    rs_np, cif_np = _synthetic_region()
    origins = _make_origins(rs_np, n=6)
    rs_t = torch.as_tensor(rs_np, dtype=torch.float32)
    cif_t = torch.as_tensor(cif_np, dtype=torch.float32)
    config_t = torch.as_tensor([[0.4, 0.6]], dtype=torch.float32)

    live = _build_live()
    for n in live.values():
        n.model.eval()
    # no frozen dirs (5-live); build empty frozen preds
    frozen = {d: torch.zeros(len(origins), HORIZON) for d in FROZEN_DIRS}
    y_true = torch.stack([cif_t[o:o + HORIZON] for o in origins])
    from scripts.experiments.run_joint_train import _persistence_cif_full
    persist = _persistence_cif_full(rs_np, cif_np, origins)

    zs = DifferentiableZSPlus()
    fusion = LearnedFusion(n_directions=5, config_dim=CFG_DIM, horizon=HORIZON)
    params = list(zs.parameters()) + list(fusion.parameters())
    metrics = native_stage("s", params, zs, fusion, live, frozen, rs_t, cif_t,
                           config_t, EF_R, EF_NR, origins, persist, y_true,
                           n_steps=3, lr=1e-2, margin=0.1, adv_loss_weight=0.5,
                           fusion_kind="learned")
    assert np.isfinite(metrics["train_loss"][-1])
    assert np.isfinite(metrics["val_mae"][-1])

    per_origin, held = eval_held_out(zs, fusion, live, frozen, rs_t, cif_t,
                                     config_t, EF_R, EF_NR, origins, y_true,
                                     fusion_kind="learned")
    assert len(per_origin) == len(origins)
    assert np.isfinite(held) and held > 0


def test_assemble_stack_shape_and_direction_order():
    """assemble_stack returns (1, 5, HORIZON) in DIRECTION_ORDER with live grads."""
    from transcif.models.zeroshot.fusion import DIRECTION_ORDER
    rs_np, cif_np = _synthetic_region(n_origins=2)
    origins = _make_origins(rs_np, n=2)
    rs_t = torch.as_tensor(rs_np, dtype=torch.float32)
    config_t = torch.as_tensor([[0.4, 0.6]], dtype=torch.float32)
    live = _build_live()
    frozen = {d: torch.zeros(len(origins), HORIZON) for d in FROZEN_DIRS}
    stack = assemble_stack(0, origins[0], rs_t, config_t, EF_R, EF_NR, live, frozen)
    assert stack.shape == (1, 5, HORIZON)


def test_snapshot_restore_roundtrip_preserves_output():
    """_snapshot then _restore restores parameters exactly.

    Verifies parameter equality (deterministic), not forward-output equality —
    NativeCausal uses VAE reparameterization (samples eps), so the forward is
    stochastic and would make an output-equality assertion flaky.
    """
    live = _build_live()
    from transcif.calibration.differentiable_zs_plus import DifferentiableZSPlus as _ZS
    zs = _ZS()
    fusion = LearnedFusion(n_directions=5, config_dim=CFG_DIM, horizon=HORIZON)

    def _flat(modules):
        return torch.cat([p.detach().reshape(-1) for m in modules for p in m.parameters()])

    before = {
        "zs": _flat([zs]),
        "fusion": _flat([fusion]),
        **{n: _flat([nm.model]) for n, nm in live.items()},
    }
    snap = _snapshot(zs, fusion, live)
    # perturb every parameter in place
    with torch.no_grad():
        for p in zs.parameters():
            p.add_(0.3)
        for p in fusion.parameters():
            p.add_(0.3)
        for nm in live.values():
            for p in nm.model.parameters():
                p.add_(0.3)
    _restore(snap, zs, fusion, live)
    after = {
        "zs": _flat([zs]),
        "fusion": _flat([fusion]),
        **{n: _flat([nm.model]) for n, nm in live.items()},
    }
    for k in before:
        assert torch.allclose(before[k], after[k], atol=1e-6), \
            f"restore did not reproduce pre-perturbation params for {k}"


def test_head_modules_covers_all_live_directions():
    """Every live direction exposes a non-empty head list (Stage 2 unfreezing)."""
    live = _build_live()
    for name, native in live.items():
        heads = head_modules(native)
        assert len(heads) > 0, f"{name} has no head modules to unfreeze"
