"""Synthetic grid recombination for TransCIF-FD (Phase FD-2).

Physics-guided mixup over real regions: a pseudo-grid is the convex
combination of two real grids over the SAME calendar period,

    grid_M = a * grid_A + (1 - a) * grid_B,   a ~ Uniform(0.2, 0.8)

with every channel (per-fuel shares, CIF, weather-exog, config, EF vector)
mixed with the same ``a``.  Because the physics layer is linear in the
shares, the mixed CIF label is EXACT — zero label noise — while the config
space fills in between the 29 real grids ("30% Scotland + 70% Texas").

Every mixed channel remains deployment-consistent: the mixed weather is the
weather the pseudo-grid would observe, the mixed config the statistics it
would publish.  This attacks the 29-domain small-sample problem and — for
a new telemetry-free region — generates neighbourhood pseudo-grids around
its published config for adaptation (see ``neighbourhood_batch``).

Mixing is restricted to fuel-telemetry regions so share targets stay on
the simplex.
"""

import numpy as np
import torch

from transcif.config import SEQ_LEN, HORIZON, TRAIN_STRIDE
from transcif.data.fuel import build_fd_windows


class GridRecombiner:
    """Prebuilds same-origin window stacks for fuel regions and samples
    mixed pseudo-grid batches.

    Usage (inside a training loop)::

        rec = GridRecombiner(fd_regions, target_name, seed)
        batch = rec.sample(64)          # dict of torch tensors
    """

    def __init__(self, fd_regions, target_name, seed=0,
                 seq_len=SEQ_LEN, horizon=HORIZON, stride=TRAIN_STRIDE,
                 max_windows_per_region=400, alpha_range=(0.2, 0.8)):
        rng = np.random.default_rng(seed)
        self.alpha_range = alpha_range
        self.rng = rng
        self.keys = ("x_rs", "x_fuel", "y_fuel", "y_rs", "y_cif",
                     "x_weather", "fut_weather", "fut_exog")
        self.stacks = []      # list of per-region window dicts (fuel regions)
        self.cfgs = []        # per-region fd_config (D,)
        self.efs = []         # per-region ef_vec (F,)
        for name, data in fd_regions.items():
            if name == target_name or not data.get("has_fuel"):
                continue
            w = build_fd_windows(data, seq_len=seq_len, horizon=horizon,
                                 stride=stride,
                                 max_windows=max_windows_per_region, rng=rng)
            if len(w["x_rs"]) == 0:
                continue
            self.stacks.append(w)
            self.cfgs.append(data["fd_config"])
            self.efs.append(data["ef_vec"].astype(np.float32))

    def sample(self, n, device=None):
        """One batch of ``n`` mixed pseudo-grid windows (numpy -> torch).

        The mixed CIF label is RE-COMPUTED from the mixed shares via the
        physics layer (``Σ_f s_f · ef_f``) rather than mixing the two
        reported CIF series — each source's reported CIF carries its own
        methodology noise, so only the recomputed label is exact.
        """
        assert self.stacks, "no fuel-telemetry source regions available"
        lo, hi = self.alpha_range
        out = {k: [] for k in self.keys}
        cfgs, efs = [], []
        for _ in range(n):
            i, j = self.rng.integers(0, len(self.stacks), size=2)
            a = float(self.rng.uniform(lo, hi))
            wa, wb = self.stacks[i], self.stacks[j]
            ia = self.rng.integers(0, len(wa["x_rs"]))
            ib = self.rng.integers(0, len(wb["x_rs"]))
            ef_mix = a * self.efs[i] + (1 - a) * self.efs[j]
            shares_mix = a * wa["y_fuel"][ia] + (1 - a) * wb["y_fuel"][ib]
            for k in self.keys:
                if k == "y_cif":
                    # Exact physics label for the pseudo-grid.
                    out[k].append(np.einsum(
                        "hf,f->h", shares_mix, ef_mix).astype(np.float32))
                else:
                    out[k].append(a * wa[k][ia] + (1 - a) * wb[k][ib])
            cfgs.append(a * self.cfgs[i] + (1 - a) * self.cfgs[j])
            efs.append(ef_mix)
        batch = {k: torch.tensor(np.stack(v)) for k, v in out.items()}
        batch["config"] = torch.tensor(np.stack(cfgs))
        batch["ef_vec"] = torch.tensor(np.stack(efs))
        if device:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    def neighbourhood_batch(self, target_config, target_ef, n, device=None):
        """Pseudo-grids anchored near a target config (no target telemetry).

        Mixes random source pairs and re-biases alpha per pair so that the
        resulting config is pulled toward ``target_config`` — a synthetic
        neighbourhood of the deployment region for cold-start adaptation.
        """
        assert self.stacks, "no fuel-telemetry source regions available"
        D = len(target_config)
        out = {k: [] for k in self.keys}
        cfgs, efs = [], []
        tgt_cfg = np.asarray(target_config, dtype=np.float32)
        tgt_ef = np.asarray(target_ef, dtype=np.float32)
        for _ in range(n):
            i, j = self.rng.integers(0, len(self.stacks), size=2)
            wa, wb = self.stacks[i], self.stacks[j]
            ia = self.rng.integers(0, len(wa["x_rs"]))
            ib = self.rng.integers(0, len(wb["x_rs"]))
            # Choose alpha that minimises ||a*c_i + (1-a)*c_j - target|| over
            # the scalar a in [lo, hi] (closed form on the config distance).
            ci, cj = self.cfgs[i].astype(np.float64), self.cfgs[j].astype(np.float64)
            diff = cj - ci
            denom = float(diff @ diff)
            if denom < 1e-9:
                a = float(self.rng.uniform(*self.alpha_range))
            else:
                a = float(np.clip((tgt_cfg.astype(np.float64) - cj) @ diff
                                  / denom, *self.alpha_range))
            for k in self.keys:
                out[k].append(a * wa[k][ia] + (1 - a) * wb[k][ib])
            cfgs.append(a * self.cfgs[i] + (1 - a) * self.cfgs[j])
            efs.append(a * self.efs[i] + (1 - a) * self.efs[j])
        batch = {k: torch.tensor(np.stack(v)) for k, v in out.items()}
        batch["config"] = torch.tensor(np.stack(cfgs))
        batch["ef_vec"] = torch.tensor(np.stack(efs))
        if device:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch
