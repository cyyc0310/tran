"""Tests for the FD-2 modules: ConfigHyperNet and GridRecombiner."""

import numpy as np
import pytest
import torch

from transcif.models.hypernet import (
    ConfigHyperNet, GENERATED_HEADS, TOTAL_PARAMS, apply_generated_head,
)
from transcif.training.synthetic import GridRecombiner


class TestConfigHyperNet:
    def test_generates_all_heads(self):
        hn = ConfigHyperNet(config_dim=16)
        gen = hn(torch.rand(4, 16))
        assert set(gen.keys()) == set(GENERATED_HEADS.keys())
        for name, (n_out, n_in) in GENERATED_HEADS.items():
            w, b = gen[name]
            assert w.shape == (4, n_out, n_in)
            assert b.shape == (4, n_out)

    def test_zero_init_output(self):
        # Fresh hypernet generates all-zero weights: the FD-1 warm start.
        hn = ConfigHyperNet(config_dim=16)
        gen = hn(torch.rand(8, 16))
        for name, _ in GENERATED_HEADS.items():
            w, b = gen[name]
            assert torch.count_nonzero(w) == 0
            assert torch.count_nonzero(b) == 0

    def test_param_count_reasonable(self):
        hn = ConfigHyperNet(config_dim=16)
        n = sum(p.numel() for p in hn.parameters())
        assert n < 25_000  # lightweight next to the 20k base model

    def test_apply_generated_head_shapes(self):
        hn = ConfigHyperNet(config_dim=16)
        gen = hn(torch.rand(3, 16))
        x2 = torch.rand(3, 17)
        x3 = torch.rand(3, 24, 17)
        assert apply_generated_head(x2, "base_delta", gen).shape == (3, 5)
        assert apply_generated_head(x3, "therm_dyn", gen).shape == (3, 24, 3)

    def test_per_sample_weights_differ(self):
        hn = ConfigHyperNet(config_dim=16)
        hn.mlp[-1].weight.data.normal_(0, 0.1)  # break the zero init
        cfg = torch.tensor([[0.1] * 16, [0.9] * 16])
        gen = hn(cfg)
        w, _ = gen["base_delta"]
        assert not torch.allclose(w[0], w[1])


class TestFuelConfigWeight:
    def test_fuel_distance_boosts_similar(self):
        from transcif.models.zeroshot.fuel import fuel_config_weight
        tgt = {"mean_rs": 0.3, "has_fuel": True,
               "fd_config": np.array([0.3, 0.4] + [0.3, 0.4, 0.0, 0.1, 0.05,
                                                   0.05, 0.05, 0.0, 0.0, 0.05]
                                                  + [0.3, 0.5, 1.0, 0.6])}
        near = dict(tgt)
        far = {"mean_rs": 0.3, "has_fuel": True,
               "fd_config": np.array([0.3, 0.4] + [0.0, 0.7, 0.15, 0.0, 0.0,
                                                   0.0, 0.0, 0.1, 0.0, 0.05]
                                                  + [0.3, 0.5, 1.0, 0.6])}
        w_near = fuel_config_weight(near, tgt)
        w_far = fuel_config_weight(far, tgt)
        assert w_near > w_far

    def test_fuelless_falls_back_to_legacy(self):
        from transcif.models.zeroshot.fuel import fuel_config_weight
        from transcif.physics.bounds import config_weight
        src = {"mean_rs": 0.4, "has_fuel": False}
        tgt = {"mean_rs": 0.3, "has_fuel": True}
        assert fuel_config_weight(src, tgt) == config_weight(0.4, 0.3)

    def test_ef_corr_bound(self):
        from transcif.models.fuel_decomp import FuelDecompNet
        m = FuelDecompNet(ef_corr_bound=0.15)
        assert m.ef_corr_bound == 0.15
        m2 = FuelDecompNet()
        assert m2.ef_corr_bound == 0.35


class TestMonthlyConfig:
    @pytest.fixture()
    def fd_data(self):
        from transcif.models.zeroshot.fuel import prepare_fd_region
        from transcif.data.loaders import all_region_configs
        cfgs = all_region_configs()
        if "US_CISO" not in cfgs:
            pytest.skip("data_2023 not present")
        return prepare_fd_region("US_CISO", cfgs)

    def test_monthly_table_shape_and_shares(self, fd_data):
        from transcif.data.fuel import (FD_CONFIG_FIELDS, CANONICAL_FUELS,
                                        build_monthly_config_table,
                                        monthly_config_at)
        table = fd_data["monthly_table"]
        assert table is not None
        assert table.shape == (12, len(FD_CONFIG_FIELDS))
        # Monthly fuel shares stay on the simplex.
        np.testing.assert_allclose(
            table[:, 2:12].sum(axis=1), 1.0, rtol=5e-3)
        # Monthly mean_rs derived from shares, in [0, 1].
        assert (table[:, 0] >= 0).all() and (table[:, 0] <= 1).all()

    def test_monthly_config_lag_lookup(self, fd_data):
        from transcif.data.fuel import monthly_config_at
        import pandas as pd
        table = fd_data["monthly_table"]
        hours = pd.DatetimeIndex(["2023-07-15 06:00", "2023-01-15 06:00"])
        cfg = monthly_config_at(table, hours, lag_months=1)
        # July window -> June row (index 5); January -> December (index 11)
        np.testing.assert_allclose(cfg[0], table[5])
        np.testing.assert_allclose(cfg[1], table[11])

    def test_au_monthly_table_exists(self):
        # AU fuel telemetry (FD-14) enables the monthly table everywhere.
        from transcif.models.zeroshot.fuel import prepare_fd_region
        from transcif.data.loaders import all_region_configs
        cfgs = all_region_configs()
        if "QLD1" not in cfgs:
            pytest.skip("data_2023 not present")
        data = prepare_fd_region("QLD1", cfgs)
        assert data["monthly_table"] is not None
        assert data["monthly_table"].shape[0] == 12

    def test_windows_carry_monthly_config(self, fd_data):
        from transcif.config import SEQ_LEN, HORIZON, TEST_STRIDE
        from transcif.data.fuel import build_fd_windows
        table = fd_data["monthly_table"]
        w = build_fd_windows(fd_data, seq_len=SEQ_LEN, horizon=HORIZON,
                             stride=TEST_STRIDE, monthly_table=table)
        assert "config" in w
        assert w["config"].shape == (len(w["x_rs"]), 16)
        # Annual fallback: no table -> no config key
        w2 = build_fd_windows(fd_data, seq_len=SEQ_LEN, horizon=HORIZON,
                              stride=TEST_STRIDE)
        assert "config" not in w2


class TestGridRecombiner:
    @pytest.fixture()
    def fd_regions(self):
        from transcif.models.zeroshot.fuel import prepare_fd_region
        from transcif.data.loaders import all_region_configs
        cfgs = all_region_configs()
        names = [n for n in cfgs
                 if n in ("US_CISO", "UK_13_London", "US_ISNE", "QLD1")]
        return {n: prepare_fd_region(n, cfgs) for n in names}

    def test_sample_shapes_and_simplex(self, fd_regions):
        if not fd_regions:
            pytest.skip("data_2023 not present")
        rec = GridRecombiner(fd_regions, "US_CISO", seed=0,
                             max_windows_per_region=30)
        b = rec.sample(16)
        assert b["x_rs"].shape == (16, 336)
        assert b["x_fuel"].shape[1:] == (336, 10)
        sums = b["y_fuel"].sum(-1)
        # Mixed share targets stay on the simplex (both sources are;
        # float32 mixing admits ~1e-3 relative slack).
        np.testing.assert_allclose(sums.numpy(), 1.0, rtol=2e-3)
        assert b["config"].shape == (16, 16)
        assert b["ef_vec"].shape == (16, 10)

    def test_mixed_cif_is_exact_physics(self, fd_regions):
        if not fd_regions:
            pytest.skip("data_2023 not present")
        rec = GridRecombiner(fd_regions, "US_CISO", seed=0,
                             max_windows_per_region=30)
        b = rec.sample(32)
        cif = torch.einsum("bhf,bf->bh", b["y_fuel"], b["ef_vec"])
        # Physics identity must hold exactly for mixed pseudo-grids.
        np.testing.assert_allclose(
            cif.numpy(), b["y_cif"].numpy(), rtol=1e-3, atol=2.0)

    def test_neighbourhood_config_close(self, fd_regions):
        if not fd_regions:
            pytest.skip("data_2023 not present")
        rec = GridRecombiner(fd_regions, "US_CISO", seed=0,
                             max_windows_per_region=30)
        target_cfg = fd_regions["US_ISNE"]["fd_config"]
        target_ef = fd_regions["US_ISNE"]["ef_vec"].astype(np.float32)
        b = rec.neighbourhood_batch(target_cfg, target_ef, 32)
        # Distance from the neighbourhood to the target must beat the mean
        # distance of the pure source configs (alpha re-biasing works).
        d_near = np.linalg.norm(
            b["config"].numpy() - target_cfg[None, :], axis=1).mean()
        src = np.stack([fd_regions[n]["fd_config"] for n in fd_regions
                        if fd_regions[n].get("has_fuel")])
        d_far = np.linalg.norm(src - target_cfg[None, :], axis=1).mean()
        assert d_near < d_far
