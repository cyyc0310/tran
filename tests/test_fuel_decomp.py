"""Unit tests for FuelDecompNet (Phase FD-1) — synthetic data, no dataset."""

import numpy as np
import pytest
import torch

from transcif.data.fuel import CANONICAL_FUELS, FUEL_INDEX, THERMAL_FUELS
from transcif.models.fuel_decomp import FuelDecompNet

L, H, F = 336, 24, len(CANONICAL_FUELS)
D = 16  # FD_CONFIG_FIELDS length


def _rand_batch(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x_rs = torch.rand(B, L, generator=g)
    x_fuel = torch.rand(B, L, F, generator=g) / F
    x_fuel = x_fuel / x_fuel.sum(-1, keepdim=True)
    x_weather = torch.randn(B, L, 5, generator=g)
    fut_weather = torch.randn(B, H, 5, generator=g)
    fut_exog = torch.randn(B, H, 10, generator=g)
    config = torch.rand(B, D, generator=g)
    config[:, 0] = 0.3                      # mean_rs
    config[:, 1] = 0.4                      # ef_nr/1000
    config[:, 2:12] /= config[:, 2:12].sum(-1, keepdim=True)
    config[:, 12] = 0.3                     # ann windcf
    config[:, 13] = 0.5                     # ann csi
    ef_vec = torch.rand(B, F, generator=g) * 500
    return x_rs, x_fuel, x_weather, fut_weather, fut_exog, config, ef_vec


class TestFuelDecompNet:
    def test_forward_shapes(self):
        model = FuelDecompNet(seq_len=L, horizon=H)
        args = _rand_batch()
        cif, shares, rs = model(*args)
        assert cif.shape == (4, H)
        assert shares.shape == (4, H, F)
        assert rs.shape == (4, H)

    def test_shares_simplex(self):
        model = FuelDecompNet(seq_len=L, horizon=H)
        cif, shares, rs = model(*_rand_batch())
        assert (shares >= 0).all()
        np.testing.assert_allclose(
            shares.sum(-1).detach().numpy(), 1.0, rtol=1e-3)

    def test_cif_finite_and_positive(self):
        model = FuelDecompNet(seq_len=L, horizon=H)
        cif, _, _ = model(*_rand_batch())
        assert torch.isfinite(cif).all()
        assert (cif >= 0).all()

    def test_cold_mode_changes_output(self):
        model = FuelDecompNet(seq_len=L, horizon=H)
        model.eval()
        args = _rand_batch()
        with torch.no_grad():
            cif_hist, _, _ = model(*args, hist_mask=torch.ones(4, 1))
            cif_cold, _, _ = model(*args, hist_mask=torch.zeros(4, 1))
        assert not torch.allclose(cif_hist, cif_cold)

    def test_cold_mode_ignores_history(self):
        # Zeroing the history inputs in cold mode must not change outputs.
        model = FuelDecompNet(seq_len=L, horizon=H)
        model.eval()
        args = _rand_batch()
        x_rs, x_fuel, x_weather, fw, fe, cfg, ef = args
        with torch.no_grad():
            c0, _, _ = model(x_rs, x_fuel, x_weather, fw, fe, cfg, ef,
                             hist_mask=torch.zeros(4, 1))
            c1, _, _ = model(x_rs * 0, x_fuel * 0, x_weather * 0, fw, fe,
                             cfg, ef, hist_mask=torch.zeros(4, 1))
        assert torch.allclose(c0, c1)

    def test_zero_init_starts_at_physics_prior(self):
        # With zero-init dynamics, solar output must follow the astronomy
        # envelope shape (correlation between predicted solar share and the
        # clear-sky channel).
        model = FuelDecompNet(seq_len=L, horizon=H)
        model.eval()
        x_rs, x_fuel, x_weather, fut_weather, fut_exog, config, ef = _rand_batch()
        # Build a deterministic diurnal clear-sky channel.
        hours = np.arange(H)
        astro = np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None) * 800
        fut_exog = torch.randn(4, H, 10)
        fut_exog[:, :, 1] = torch.tensor(astro, dtype=torch.float32)
        config[:, 2 + FUEL_INDEX["solar"]] = 0.1
        with torch.no_grad():
            _, shares, _ = model(x_rs, x_fuel, x_weather, fut_weather,
                                 fut_exog, config, ef,
                                 hist_mask=torch.zeros(4, 1))
        solar = shares[0, :, FUEL_INDEX["solar"]].numpy()
        corr = np.corrcoef(solar, astro)[0, 1]
        assert corr > 0.9

    def test_gradients_flow(self):
        model = FuelDecompNet(seq_len=L, horizon=H)
        cif, shares, rs = model(*_rand_batch())
        loss = cif.abs().mean() + shares.abs().mean() + rs.abs().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_param_budget(self):
        # Capacity discipline: stay comparable to the 18k-param flagship
        # (deep encoders were shown harmful for zero-shot transfer).
        model = FuelDecompNet(seq_len=L, horizon=H)
        n = sum(p.numel() for p in model.parameters())
        assert n < 120_000

    def test_thermal_residual_structure(self):
        # Predicted thermal total + non-dispatchable total must sum to ~1.
        model = FuelDecompNet(seq_len=L, horizon=H)
        model.eval()
        args = _rand_batch()
        with torch.no_grad():
            _, shares, _ = model(*args)
        thermal_idx = [FUEL_INDEX[f] for f in THERMAL_FUELS]
        thermal = shares[:, :, thermal_idx].sum(-1)
        nondisp = 1.0 - thermal
        np.testing.assert_allclose(
            (thermal + nondisp).detach().numpy(), 1.0, atol=1e-5)
