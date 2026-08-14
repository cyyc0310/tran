"""Tests for transcif.models base zoo and zero-shot training (tiny)."""

import numpy as np
import torch

from transcif.models.base import AdaptivePersistDLinear, PatchTSTFixed
from transcif.models.zeroshot.base_zs import train_zero_shot, evaluate_target
from transcif.config import SEQ_LEN, HORIZON


def test_flagship_models_instantiate():
    for cls in [AdaptivePersistDLinear, PatchTSTFixed]:
        m = cls(seq_len=SEQ_LEN, horizon=HORIZON)
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


def test_mixed_dim_config_pool_trains_and_evaluates():
    """A.5: a pool mixing 2-D (legacy) and N-D (multi-fuel) config vectors
    must train and evaluate without shape errors — shorter configs are
    right-padded with zeros to the pool's max width."""
    rng = np.random.default_rng(2)
    regions = {}
    # Source A: legacy 2-D config.
    rs_a = rng.random(2000).astype(np.float32)
    regions["A"] = {
        "rs": rs_a, "cif": (rs_a * 100 + 300).astype(np.float32),
        "ef_r": 0.0, "ef_nr": 400.0, "mean_rs": float(rs_a.mean()),
        "config": np.array([rs_a.mean(), 0.4], dtype=np.float32),
    }
    # Source B: multi-fuel 5-D config (mean_rs, ef_nr/1000, coal, gas, wind).
    rs_b = rng.random(2000).astype(np.float32)
    regions["B"] = {
        "rs": rs_b, "cif": (rs_b * 100 + 300).astype(np.float32),
        "ef_r": 0.0, "ef_nr": 400.0, "mean_rs": float(rs_b.mean()),
        "config": np.array([rs_b.mean(), 0.4, 0.3, 0.5, 0.2], dtype=np.float32),
    }
    # Target C: legacy 2-D config (will be padded at inference).
    rs_c = rng.random(2000).astype(np.float32)
    regions["C"] = {
        "rs": rs_c, "cif": (rs_c * 100 + 300).astype(np.float32),
        "ef_r": 0.0, "ef_nr": 400.0, "mean_rs": float(rs_c.mean()),
        "config": np.array([rs_c.mean(), 0.4], dtype=np.float32),
    }
    model = train_zero_shot(regions, "C", seed=0, epochs=2)
    assert isinstance(model, torch.nn.Module)
    # The model must have widened its config input to the pool's max (5).
    assert model.config_bias[0].in_features == 5
    # Full evaluate_target must succeed on the mixed pool.
    res = evaluate_target("C", regions, seed=0)
    assert res["transcif_zs"]["mae"] >= 0
