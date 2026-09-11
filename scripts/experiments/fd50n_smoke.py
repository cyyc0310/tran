"""Smoke test for FD-50n multiyear pipeline on a single target (UK_02).

Verifies:
1. dk.DUMP_DIR monkey-patch works (npz lands in results/dunkelflaute_my,
   official dump untouched).
2. Target prepared with multi_year=False (2023 holdout preserved).
3. Windows count / origin_hours match the official dump (same test split).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

RESULTS = Path("results")
TGT = "UK_02_South_Scotland"


def main():
    spec = importlib.util.spec_from_file_location(
        "dk", "scripts/experiments/dunkelflaute_buckets.py")
    dk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dk)

    import torch
    from transcif.models.zeroshot.fuel import prepare_fd_region

    cfgs = dk.load_all_configs() if hasattr(dk, "load_all_configs") else None
    if cfgs is None:
        from transcif.data.loaders import all_region_configs
        cfgs = all_region_configs()

    DUMP_MY = RESULTS / "dunkelflaute_my"
    DUMP_MY.mkdir(exist_ok=True)
    dk.DUMP_DIR = DUMP_MY

    # prepare a small source pool: target + a few diverse sources
    pool = [TGT, "NSW1", "US_CISO", "US_ERCO", "UK_18_GB", "QLD1", "US_MISO"]
    fd_regions = {}
    for name in pool:
        fd_regions[name] = prepare_fd_region(name, cfgs, multi_year=True)
    # target must be 2023-only
    fd_regions[TGT] = prepare_fd_region(TGT, cfgs, multi_year=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    res = dk.dump_region(TGT, fd_regions, device, seed=0, epochs=60)

    p_new = DUMP_MY / f"{TGT}_seed0.npz"
    p_old = RESULTS / "dunkelflaute" / f"{TGT}_seed0.npz"
    print("new dump exists:", p_new.exists())
    print("old dump intact:", p_old.exists())
    if p_new.exists() and p_old.exists():
        zn = np.load(p_new)
        zo = np.load(p_old)
        on = np.asarray(zn["origin_hours"])
        oo = np.asarray(zo["origin_hours"])
        print("windows new/old:", len(on), len(oo))
        print("origin_hours equal:",
              np.array_equal(np.sort(on), np.sort(oo)))
        y_n = zn["y_true"].astype(float)
        y_o = zo["y_true"].astype(float)
        print("y_true equal:", np.allclose(np.sort(y_n), np.sort(y_o)))
    print("res:", res)


if __name__ == "__main__":
    main()
