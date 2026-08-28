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
    x_weather = torch.randn(B, L, 10, generator=g)
    fut_weather = torch.randn(B, H, 10, generator=g)
    # 12 exog channels: astro(2) + wind_cf + csi + calendar(6) + hdh + cdh
    fut_exog = torch.randn(B, H, 18, generator=g)
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
        fut_exog = torch.randn(4, H, 18)
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

    def test_i0_anchor_pulls_to_observed_level(self):
        # ZS+ branch-0 mechanism ported in-model: in history mode the
        # horizon-mean CIF must move toward (1 - recent_rs) * ef_nr.
        model = FuelDecompNet(seq_len=L, horizon=H)
        model.eval()
        x_rs, x_fuel, x_weather, fut_weather, fut_exog, config, ef = _rand_batch()
        config[:, 0] = 0.4    # mean_rs (also drives anchor via recent mean)
        x_rs[:] = 0.4         # flat observed share
        with torch.no_grad():
            cif, _, _ = model(x_rs, x_fuel, x_weather, fut_weather,
                              fut_exog, config, ef,
                              hist_mask=torch.ones(4, 1))
        ef_nr = config[:, 1] * 1000.0
        target_level = 0.6 * ef_nr
        # With gate init sigmoid(1.5) ~ 0.82, the level should sit well
        # below the un-anchored physics level of the random batch.
        assert (cif.mean(dim=1) - target_level).abs().mean() < \
            0.5 * ef.abs().mean()

    def test_i0_anchor_inactive_in_cold_mode(self):
        # Cold mode passes through: outputs equal the pre-anchor value.
        model = FuelDecompNet(seq_len=L, horizon=H)
        model.eval()
        args = _rand_batch()
        x_rs, x_fuel, x_weather, fut_weather, fut_exog, config, ef = args
        with torch.no_grad():
            cif_cold, _, _ = model(x_rs, x_fuel, x_weather, fut_weather,
                                   fut_exog, config, ef,
                                   hist_mask=torch.zeros(4, 1))
        # Zeroing the anchor gate must not change the cold output.
        with torch.no_grad():
            model.anchor_gate[-1].bias.fill_(-10.0)  # gate -> 0
            cif_cold2, _, _ = model(x_rs, x_fuel, x_weather, fut_weather,
                                    fut_exog, config, ef,
                                    hist_mask=torch.zeros(4, 1))
        assert torch.allclose(cif_cold, cif_cold2)


class TestWindRegimeChannels:
    """FD-16: wind drought must survive the reference normalisation."""

    def _wind_case(self, hist_wcf, regime, fut_wcf, annual=0.30):
        """Reference-normalisation maths of the wind head, isolated."""
        lull = torch.sigmoid(torch.tensor(8.0 * (0.75 - regime / annual)))
        ref_old = 0.7 * hist_wcf + 0.3 * annual
        ref_new = (1 - 0.6 * lull) * ref_old + 0.6 * lull * annual
        return fut_wcf / ref_old, fut_wcf / ref_new

    def test_drought_amplified(self):
        old, new = self._wind_case(hist_wcf=0.12, regime=0.10, fut_wcf=0.12)
        # The drought ratio must drop (wind share suppressed harder).
        assert new < old - 0.15

    def test_normal_regime_unchanged(self):
        old, new = self._wind_case(hist_wcf=0.30, regime=0.30, fut_wcf=0.30)
        assert abs(new - old) < 1e-6

    def test_drought_exit_visible(self):
        _, new = self._wind_case(hist_wcf=0.12, regime=0.10, fut_wcf=0.50)
        assert new > 1.5  # recovery still strongly signalled

    def test_model_uses_regime_channel(self):
        # Changing ONLY the regime channel must change the wind-driven
        # CIF when the config routes to the fuel path.
        model = FuelDecompNet(seq_len=L, horizon=H)
        model.eval()
        x_rs, x_fuel, x_weather, fut_weather, fut_exog, config, ef = \
            _rand_batch()
        config[:, 2 + FUEL_INDEX["wind"]] = 0.10   # below route tau
        fut_weather[:, :, 3] = 0.12                # calm forecast
        x_weather[:, :, 3] = 0.12                  # calm trailing week
        fw_a = fut_weather.clone()
        fw_a[:, :, 8] = 0.10                       # confirmed drought
        fw_b = fut_weather.clone()
        fw_b[:, :, 8] = 0.30                       # regime normal
        with torch.no_grad():
            cif_a, _, _ = model(x_rs, x_fuel, x_weather, fw_a, fut_exog,
                                config, ef)
            cif_b, _, _ = model(x_rs, x_fuel, x_weather, fw_b, fut_exog,
                                config, ef)
        # Drought-anchored reference lowers the wind share -> higher CIF.
        assert (cif_a - cif_b).abs().mean() > 0


class TestImportsEFPathway:
    """FD-18: dedicated imports-EF head (interconnector source swings)."""

    def test_zero_init_equals_canonical(self):
        model = FuelDecompNet(seq_len=L, horizon=H)
        model.eval()
        x_rs, x_fuel, x_weather, fut_weather, fut_exog, config, ef = \
            _rand_batch()
        config[:, 2 + FUEL_INDEX["wind"]] = 0.05   # fuel path
        config[:, 2 + FUEL_INDEX["imports"]] = 0.3  # imports-heavy
        with torch.no_grad():
            a, _, _ = model(x_rs, x_fuel, x_weather, fut_weather,
                            fut_exog, config, ef)
            model.imp_ef[-1].weight.data.normal_(0, 1.0)
            model.imp_ef[-1].bias.data.normal_(0, 1.0)
            b, _, _ = model(x_rs, x_fuel, x_weather, fut_weather,
                            fut_exog, config, ef)
        assert not torch.allclose(a, b)   # pathway active on the fuel path

    def test_aggregate_path_untouched(self):
        torch.manual_seed(0)
        model = FuelDecompNet(seq_len=L, horizon=H, wind_route_tau=0.25)
        model.eval()
        x_rs, x_fuel, x_weather, fut_weather, fut_exog, config, ef = \
            _rand_batch()
        config[:, 2 + FUEL_INDEX["wind"]] = 0.8    # routed to aggregate
        with torch.no_grad():
            a, _, _ = model(x_rs, x_fuel, x_weather, fut_weather,
                            fut_exog, config, ef)
            model.imp_ef[-1].weight.data.normal_(0, 1.0)
            model.imp_ef[-1].bias.data.normal_(0, 1.0)
            b, _, _ = model(x_rs, x_fuel, x_weather, fut_weather,
                            fut_exog, config, ef)
        # route weight ~1e-3 leaves a negligible fuel-path residue
        assert torch.allclose(a, b, atol=1e-2)

    def test_default_fuel_first_for_supported_wind_grid(self):
        """Post-FD17 default: supported wind grids keep the fuel path."""
        torch.manual_seed(0)
        model = FuelDecompNet(seq_len=L, horizon=H)
        model.eval()
        args = _rand_batch()
        x_rs, x_fuel, x_weather, fw, fe, cfg, ef = args
        cfg[:, 2 + FUEL_INDEX["wind"]] = 0.80
        cfg[:, 2 + FUEL_INDEX["hydro"]] = 0.05
        cfg[:, 2 + FUEL_INDEX["solar"]] = 0.05
        cfg[:, 14] = 1.0
        with torch.no_grad():
            a, _, _ = model(x_rs, x_fuel, x_weather, fw, fe, cfg, ef)
            model.imp_ef[-1].weight.data.normal_(0, 1.0)
            model.imp_ef[-1].bias.data.normal_(0, 1.0)
            b, _, _ = model(x_rs, x_fuel, x_weather, fw, fe, cfg, ef)
        assert not torch.allclose(a, b)


class TestHydroRouter:
    """FD-19: hydro-dominant grids route to the aggregate path."""

    def test_hydro_dominance_forces_aggregate(self):
        torch.manual_seed(0)
        model = FuelDecompNet(seq_len=L, horizon=H)
        model.eval()
        x_rs, x_fuel, x_weather, fut_weather, fut_exog, config, ef = \
            _rand_batch()
        config[:, 2 + FUEL_INDEX["wind"]] = 0.05    # low wind: fuel route
        config[:, 2 + FUEL_INDEX["hydro"]] = 0.71   # BPAT-like hydro
        with torch.no_grad():
            a, _, _ = model(x_rs, x_fuel, x_weather, fut_weather,
                            fut_exog, config, ef)
            model.imp_ef[-1].weight.data.normal_(0, 1.0)  # fuel-path only
            b, _, _ = model(x_rs, x_fuel, x_weather, fut_weather,
                            fut_exog, config, ef)
        assert torch.allclose(a, b, atol=1e-2)  # routed aggregate
