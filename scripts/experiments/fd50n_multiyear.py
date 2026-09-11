"""FD-50n: multi-year channel evaluation (2022+2023+2024 sources, 2023 test).

Protocol: identical to dunkelflaute dump (FD-41 official: 28-source LORO,
seed 0, ep900, p_cold 0.3, monthly lag 1, tau ROUTE_TABLE) but every SOURCE
region's data now concatenates 2022+2023+2024 hourly truth files
(multi_year=True).  Target region test windows remain 2023-holdout (the
target's own 2023 tail), same as official protocol — extra years only enrich
SOURCE training data (more seasons of cross-domain fuel-weather dynamics).

Honesty protocol (two guarantees):
1. Target region is prepared with multi_year=False — its 2022/2024 truth is
   NOT loaded, so target test windows stay on the official 2023 holdout and
   remain zero-telemetry (no leakage of the target's extra-year data).
2. dump_region in dunkelflaute_buckets.py writes to its module-level
   DUMP_DIR (results/dunkelflaute).  We monkey-patch dk.DUMP_DIR to
   results/dunkelflaute_my BEFORE calling, so base dumps are never touched.

Arms:
  base  : official single-year (2023-only) I_cfg  — from existing dumps
  my    : multi-year-source I_cfg re-trained (seed 0)
  +J5   : J5 stack on both (identical recipe, see fd50n_analysis.py)

Usage:
  PYTHONPATH=src .venv-nemed/bin/python scripts/experiments/fd50n_multiyear.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path("results")


def mae(p, y):
    return float(np.abs(p - y).mean())


def main():
    import importlib.util

    import torch  # noqa: F401

    spec = importlib.util.spec_from_file_location(
        "dk", "scripts/experiments/dunkelflaute_buckets.py")
    dk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dk)

    from transcif.models.zeroshot.fuel import prepare_fd_region

    DUMP_MY = RESULTS / "dunkelflaute_my"
    DUMP_MY.mkdir(exist_ok=True)

    # Guarantee 2: redirect dump output away from the official dump dir.
    dk.DUMP_DIR = DUMP_MY

    targets = sorted(p.name[: -len("_seed0.npz")]
                     for p in (RESULTS / "dunkelflaute").glob("*_seed0.npz"))

    fd_regions = {}
    cfgs = dk.load_all_configs() if hasattr(dk, "load_all_configs") else None
    # fall back to the same discovery used by the dump script
    if cfgs is None:
        from transcif.data.loaders import all_region_configs
        cfgs = all_region_configs()

    # Prepare multi-year data for every region (needed as SOURCE pool), but
    # remember which regions actually have extra-year files on disk.
    has_multi = {}
    for name in list(cfgs):
        if name == "CN_5MIN":
            continue
        try:
            has_multi[name] = _detect_multiyear_files(name, cfgs)
            fd_regions[name] = prepare_fd_region(
                name, cfgs, multi_year=True)
        except Exception as e:
            print(f"[prep][WARN] {name}: {e}")

    # Guarantee 1: re-prepare every TARGET with 2023-only data (no extra
    # years) so the test split is the official 2023 holdout.
    for tgt in targets:
        fd_regions[tgt] = prepare_fd_region(tgt, cfgs, multi_year=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_rows = {}
    for tgt in targets:
        try:
            res = dk.dump_region(tgt, fd_regions, device, seed=0, epochs=900)
        except Exception as e:
            print(f"[dump][ERR] {tgt}: {e}")
            continue
        if res is None:
            continue
        path = DUMP_MY / f"{tgt}_seed0.npz"
        if res and Path(res).exists() and Path(res) != path:
            import shutil
            shutil.move(str(Path(res)), path)
        if not path.exists():
            print(f"[dump][WARN] {tgt}: no npz at {path}")
            continue
        z_new = np.load(path)
        y = z_new["y_true"].astype(float)
        p = z_new["pred_i_cfg"].astype(float)
        z_old = np.load(RESULTS / "dunkelflaute" / f"{tgt}_seed0.npz")
        y0 = z_old["y_true"].astype(float)
        p0 = z_old["pred_i_cfg"].astype(float)
        n = min(len(y), len(y0))
        l = slice(n // 2, n)
        out_rows[tgt] = {
            "n": int(n),
            "base_late": round(mae(p0[l], y0[l]), 3),
            "my_late": round(mae(p[l], y[l]), 3),
            "base_full": round(mae(p0, y0), 3),
            "my_full": round(mae(p, y), 3),
        }
        print(tgt, out_rows[tgt])
    (RESULTS / "fd50n_multiyear.json").write_text(
        json.dumps(out_rows, indent=1, ensure_ascii=False))
    print("done", len(out_rows))


def _detect_multiyear_files(name, cfgs):
    """Report which extra-year truth files exist for a region (any year)."""
    from transcif.data.loaders import DATA_DIR
    info = cfgs[name]
    stem = info["file"].replace("_2023_hourly.csv", "")
    extra = sorted(p.name for p in DATA_DIR.glob(f"{stem}_*_hourly.csv")
                   if not p.name.endswith("_2023_hourly.csv"))
    return extra


if __name__ == "__main__":
    main()
