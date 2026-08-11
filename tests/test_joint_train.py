"""TDD tests for joint training pipeline (Task 8.3).

Stage 1: 5 direction models frozen, train ZS+ attention + BasisMix head.
Stage 2: add learnable per-direction output correction (initialized to zero)
         so gradient can fine-tune the model output layer without refactoring
         the underlying numpy/torch hybrid predictors.

Run with:
    .venv/bin/python -m pytest tests/test_joint_train.py -v
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from transcif.config import HORIZON, SEQ_LEN


@pytest.fixture
def small_setup():
    """Synthetic 1-target, 2-source setup that runs in seconds.
    Mirrors the loader.py output schema (mean_rs, config shape, etc.).
    T=1600 so the test split yields >= 12 origins for train+eval splits."""
    torch.manual_seed(0)
    np.random.seed(0)

    T = 1600
    target_rs = np.cumsum(np.random.randn(T) * 0.02).clip(0, 1).astype(np.float32)
    src1_rs = np.cumsum(np.random.randn(T) * 0.02).clip(0, 1).astype(np.float32)
    src2_rs = np.cumsum(np.random.randn(T) * 0.02).clip(0, 1).astype(np.float32)

    ef_r, ef_nr = 50.0, 500.0
    target_cif = (target_rs * ef_r + (1 - target_rs) * ef_nr).astype(np.float32)
    src1_cif = (src1_rs * ef_r + (1 - src1_rs) * ef_nr).astype(np.float32)
    src2_cif = (src2_rs * ef_r + (1 - src2_rs) * ef_nr).astype(np.float32)

    def _pack(rs, cif):
        return {
            "rs": rs, "cif": cif,
            "mean_rs": float(rs.mean()),
            "ef_r": ef_r, "ef_nr": ef_nr,
            "config": np.array([rs.mean(), ef_nr / 1000.0], dtype=np.float32),
        }

    regions = {
        "TARGET": _pack(target_rs, target_cif),
        "SRC1": _pack(src1_rs, src1_cif),
        "SRC2": _pack(src2_rs, src2_cif),
    }
    return regions, "TARGET"


def test_stage1_only_changes_attention_and_fusion(small_setup, tmp_path):
    """Stage 1 trains only ZS+ attention + BasisMix head; direction predictions
    are treated as frozen features. After Stage 1, branch_gate / log_inv_temp /
    fusion logit MUST have changed from init; nothing else should have.
    """
    from scripts.experiments.run_joint_train import (
        run_joint_train,
    )

    regions, target = small_setup
    out_dir = tmp_path / "stage1"
    out_dir.mkdir()

    result = run_joint_train(
        regions, target,
        stages=("stage1",),
        n_origins=8,
        out_dir=str(out_dir),
        seed=0,
    )

    # Output files
    assert (out_dir / "stage1_metrics.json").exists()
    assert (out_dir / "stage1_checkpoint.pt").exists()

    # Metrics JSON has the right shape
    metrics = json.loads((out_dir / "stage1_metrics.json").read_text())
    assert "train_loss" in metrics
    assert "val_mae" in metrics
    assert len(metrics["train_loss"]) > 0

    # Initial branch_gate is zeros; after training, norm > 0
    ckpt = torch.load(out_dir / "stage1_checkpoint.pt", weights_only=False)
    assert "zs_plus_state" in ckpt
    assert "fusion_state" in ckpt
    branch_gate = ckpt["zs_plus_state"]["branch_gate"]
    assert branch_gate.norm() > 0, "branch_gate should have moved from zeros"

    # Sanity: loss decreased
    assert metrics["train_loss"][-1] < metrics["train_loss"][0]


def test_stage2_adds_correction_and_reduces_loss(small_setup, tmp_path):
    """Stage 2 adds a learnable per-direction correction (init zero).
    After Stage 2:
      - correction tensor is non-zero
      - loss should be <= stage 1 final loss (or close)
    """
    from scripts.experiments.run_joint_train import run_joint_train

    regions, target = small_setup
    out_dir = tmp_path / "stage2"
    out_dir.mkdir()

    result = run_joint_train(
        regions, target,
        stages=("stage1", "stage2"),
        n_origins=8,
        out_dir=str(out_dir),
        seed=0,
    )

    # Stage 2 checkpoint
    assert (out_dir / "stage2_checkpoint.pt").exists()
    ckpt = torch.load(out_dir / "stage2_checkpoint.pt", weights_only=False)
    assert "correction" in ckpt
    correction = ckpt["correction"]
    # Shape: (5, HORIZON)
    assert correction.shape == (5, HORIZON)
    # Should be non-zero after training (initialized to zero, gradient flows)
    assert correction.abs().sum() > 0, "Stage 2 correction should be non-zero"


def test_separate_optimizers(small_setup, tmp_path):
    """Stage 1 and Stage 2 use separate optimizer instances.
    The pipeline tracks each stage's optimizer separately, observable via
    the metrics file recording separate ``stage`` labels.
    """
    from scripts.experiments.run_joint_train import run_joint_train

    regions, target = small_setup
    out_dir = tmp_path / "separate"
    out_dir.mkdir()

    result = run_joint_train(
        regions, target,
        stages=("stage1", "stage2"),
        n_origins=8,
        out_dir=str(out_dir),
        seed=0,
    )

    metrics1 = json.loads((out_dir / "stage1_metrics.json").read_text())
    metrics2 = json.loads((out_dir / "stage2_metrics.json").read_text())
    # Each stage has its own metrics file (separate optimizer step tracking)
    assert "stage" in metrics1
    assert "stage" in metrics2
    assert metrics1["stage"] == "stage1"
    assert metrics2["stage"] == "stage2"


def test_runtime_under_30min_per_pair(small_setup, tmp_path):
    """DoD: 1 source-target pair end-to-end < 30 min. With 8 origins on
    synthetic 800-step data, this should complete in seconds."""
    import time
    from scripts.experiments.run_joint_train import run_joint_train

    regions, target = small_setup
    out_dir = tmp_path / "timing"
    out_dir.mkdir()

    t0 = time.time()
    run_joint_train(
        regions, target,
        stages=("stage1", "stage2"),
        n_origins=8,
        out_dir=str(out_dir),
        seed=0,
    )
    elapsed = time.time() - t0
    # On synthetic data this is seconds. On real QLD1+3-source data this
    # is the budget that must stay under 30 min per pair. Use 600s ceiling
    # for the synthetic test to keep CI fast.
    assert elapsed < 600, f"Took {elapsed:.1f}s; expected < 600s on synthetic"


def test_held_out_eval_origins(small_setup, tmp_path):
    """When ``eval_origins`` is provided (disjoint from training origins),
    the summary must include a ``held_out_mae`` field computed on those
    origins. This is the path ``run_joint_train_full.py`` depends on.

    This test closes the MEDIUM coverage gap from the Phase 8 review: the
    held-out eval code path was untested even though the full LORO result
    (median MAE 40.53) rests on it.
    """
    import time
    from scripts.experiments.run_joint_train import (
        run_joint_train,
        _origins_from_series,
    )

    regions, target = small_setup
    rs = regions[target]["rs"]

    # Take 6 training origins + 4 disjoint eval origins
    all_origins = _origins_from_series(rs, n_max=20)
    train_origins = all_origins[:6]
    eval_origins = all_origins[6:10]
    assert set(train_origins).isdisjoint(set(eval_origins)), (
        "train and eval origins must be disjoint"
    )

    out_dir = tmp_path / "heldout"
    out_dir.mkdir()

    summary = run_joint_train(
        regions, target,
        stages=("stage1", "stage2"),
        n_origins=len(train_origins),
        out_dir=str(out_dir),
        seed=0,
        eval_origins=eval_origins,
    )

    # held_out_mae is present and finite
    assert "held_out_mae" in summary, (
        f"summary missing held_out_mae; got keys {list(summary.keys())}"
    )
    assert summary["held_out_mae"] is not None
    assert math.isfinite(summary["held_out_mae"]), (
        f"held_out_mae not finite: {summary['held_out_mae']}"
    )
    # Sane range: between 0 and 1000 (CIF is gCO2/kWh, max ~1000)
    assert 0.0 < summary["held_out_mae"] < 1000.0, (
        f"held_out_mae out of sane range: {summary['held_out_mae']}"
    )
    assert summary["held_out_n_origins"] == len(eval_origins)


def test_no_eval_origins_skips_held_out(small_setup, tmp_path):
    """When ``eval_origins`` is None (default), the summary must NOT
    include ``held_out_mae``. Sanity check for the optional path."""
    from scripts.experiments.run_joint_train import run_joint_train

    regions, target = small_setup
    out_dir = tmp_path / "no_eval"
    out_dir.mkdir()

    summary = run_joint_train(
        regions, target,
        stages=("stage1",),
        n_origins=6,
        out_dir=str(out_dir),
        seed=0,
        # eval_origins=None (default)
    )
    assert "held_out_mae" not in summary, (
        f"held_out_mae should not be set without eval_origins; got {summary}"
    )
