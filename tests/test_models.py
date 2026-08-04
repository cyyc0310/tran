"""Tests for transcif.models base zoo and zero-shot training (tiny)."""

import numpy as np
import torch

from transcif.models.base import (
    AdaptivePersistDLinear, PatchTSTFixed, get_model, MODEL_REGISTRY,
)
from transcif.models.zeroshot.base_zs import train_zero_shot, evaluate_target
from transcif.config import SEQ_LEN, HORIZON


def test_model_registry():
    for name in ["AdaptivePersistDLinear", "PatchTSTFixed"]:
        m = get_model(name, seq_len=SEQ_LEN, horizon=HORIZON)
        assert isinstance(m, torch.nn.Module)


def test_adaptive_persist_forward_shape():
    m = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON)
    x = torch.rand(4, SEQ_LEN)
    c = torch.rand(4, 2)
    out = m(x, c)
    assert out.shape == (4, HORIZON)


def test_train_zero_shot_runs():
    # Build two tiny synthetic regions and train a leave-one-out model.
    rng = np.random.default_rng(0)
    regions = {}
    for name in ["A", "B"]:
        rs = rng.random(2000).astype(np.float32)
        cif = (rs * 100 + 300).astype(np.float32)
        regions[name] = {
            "rs": rs, "cif": cif, "ef_r": 0.0, "ef_nr": 400.0,
            "mean_rs": float(rs.mean()),
            "config": np.array([rs.mean(), 0.4], dtype=np.float32),
        }
    model = train_zero_shot(regions, "B", seed=0, epochs=2)
    assert isinstance(model, torch.nn.Module)


def test_evaluate_target_default_methods():
    rng = np.random.default_rng(1)
    regions = {}
    for name in ["A", "B"]:
        rs = rng.random(2000).astype(np.float32)
        cif = (rs * 100 + 300).astype(np.float32)
        regions[name] = {
            "rs": rs, "cif": cif, "ef_r": 0.0, "ef_nr": 400.0,
            "mean_rs": float(rs.mean()),
            "config": np.array([rs.mean(), 0.4], dtype=np.float32),
        }
    res = evaluate_target("B", regions, seed=0)
    for key in ["persistence", "patchtst_sup", "transcif_zs", "transcif_zs_plus"]:
        assert key in res
        assert res[key]["mae"] >= 0
