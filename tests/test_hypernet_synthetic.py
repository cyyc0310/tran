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
        x2 = torch.rand(3, 10)
        x3 = torch.rand(3, 24, 10)
        assert apply_generated_head(x2, "base_delta", gen).shape == (3, 5)
        assert apply_generated_head(x3, "therm_dyn", gen).shape == (3, 24, 3)

    def test_per_sample_weights_differ(self):
        hn = ConfigHyperNet(config_dim=16)
        hn.mlp[-1].weight.data.normal_(0, 0.1)  # break the zero init
        cfg = torch.tensor([[0.1] * 16, [0.9] * 16])
        gen = hn(cfg)
        w, _ = gen["base_delta"]
        assert not torch.allclose(w[0], w[1])


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
