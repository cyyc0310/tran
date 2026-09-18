#!/usr/bin/env python
"""FD-48 A/B: monthly weather climatology in the target config table.

Attribution (2026-09-18, on the fd47_nwp era5 dumps): in cold mode the
wind head normalises future wind CF by the ANNUAL climatology slot
(config[12]); the test season (Oct-Dec) drifts away from the annual
mean, which shows up as a slow level drift coupled to the seasonal wind
cycle — 16/29 regions have |corr(per-window bias, horizon-mean wind CF)|
> 0.5 and the late-vs-early-third bias shift reaches +-84 gCO2/kWh
(UK_09 +83.5, UK_08 +78.1, UK_11 +62.9, NSW1 -63.5).

Fix under test: replace the annual weather slots (12 ann_windcf,
13 ann_csi) of the TARGET's evaluation-time monthly config table with
monthly climatology of the SAME public reanalysis product the future
weather channels already come from (no new information class), shrunk
0.5 toward the annual value (MONTHLY_SHRINK convention).  Training is
untouched — identical seed/hyperparameters/routes — so this is an
inference-time interface upgrade evaluated as a paired within-run A/B:
the baseline arm is re-predicted by the same model in the same process,
so device FP differences cannot contaminate the comparison.

AU regions have no config table (tier definition) and are skipped.

Usage:
    env PYTHONPATH=src .venv-nemed/bin/python \
        scripts/experiments/fd47_mwx_ab.py \
        --regions UK_09_East_Midlands UK_08_West_Midlands
    env PYTHONPATH=src .venv-nemed/bin/python \
        scripts/experiments/fd47_mwx_ab.py --analyze
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transcif.config import HORIZON, SEQ_LEN, TEST_STRIDE, TRAIN_FRACTION
from transcif.data.loaders import all_region_configs
from transcif.data.fuel import build_fd_windows, monthly_config_at
from transcif.models.zeroshot.fuel import predict_fuel_windows, train_fuel_zero_shot

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
OUT_DIR = RESULTS / "fd47_mwx_ab"
ROUTE_TABLE = {"UK_01_North_Scotland": 1.1, "UK_02_South_Scotland": 1.1,
               "UK_16_Scotland": 1.1}


def monthly_wx_cfg_table(data, shrink=0.5):
    """Monthly config table with weather slots (12/13) made monthly.

    Slots 12/13 of ``data['monthly_table']`` hold the annual wind-CF /
    daytime clear-sky-index means.  This rebuilds them as monthly means of
    the same reanalysis-derived channels, shrunk toward the annual value.
    Fuel-share slots (0-11) are already monthly and stay untouched.
    """
    mt = data["monthly_table"]
    ex = data["exog"]
    hours = pd.DatetimeIndex(data["hours"])
    wcf = ex["wind_cf"]
    csi = ex["clearsky_index"]
    day = ex["astro"][:, 0] > 0
    mcf = np.array([wcf[hours.month == m].mean() for m in range(1, 13)],
                   dtype=np.float32)
    mcsi = np.array(
        [csi[(hours.month == m) & day].mean() if ((hours.month == m) & day).any()
         else float(mt[0, 13]) for m in range(1, 13)], dtype=np.float32)
    out = mt.copy()
    out[:, 12] = ((1.0 - shrink) * float(mt[0, 12]) + shrink * mcf).astype(np.float32)
    out[:, 13] = ((1.0 - shrink) * float(mt[0, 13]) + shrink * mcsi).astype(np.float32)
    return out


def _predict_safe(model, windows, fd_config, ef_vec, cold, device):
    """MPS-safe wrapper around predict_fuel_windows.

    The upstream helper tiles ``ef_vec`` (float64) after moving the first
    tensor to ``device``; on MPS the float64 conversion then fails.  Build
    the ef tile as float32 on CPU first so the device never sees a float64
    tensor.  On CPU the behaviour is identical to the upstream function."""
    import torch
    n = len(windows["x_rs"])
    hist = 0.0 if cold else 1.0
    model.eval()
    outs = []
    dev = device or next(model.parameters()).device
    ef_t = torch.tensor(np.tile(np.asarray(ef_vec, dtype=np.float32),
                                (1, 1)))
    with torch.no_grad():
        for s in range(0, n, 512):
            e = min(s + 512, n)
            args = [torch.tensor(np.asarray(windows[k][s:e],
                                            dtype=np.float32)).to(dev)
                    for k in ("x_rs", "x_fuel", "x_weather",
                              "fut_weather", "fut_exog")]
            cfg_key = "config" if "config" in windows else None
            if cfg_key is not None:
                cfg = torch.tensor(np.asarray(windows[cfg_key][s:e],
                                              dtype=np.float32)).to(dev)
            else:
                cfg = torch.tensor(np.tile(
                    np.asarray(fd_config, dtype=np.float32),
                    (e - s, 1))).to(dev)
            ef = ef_t.expand(e - s, -1).to(dev)
            hm = torch.full((e - s, 1), hist, device=dev)
            cif, _, _ = model(*args, cfg, ef, hist_mask=hm)
            outs.append(cif.cpu().numpy().astype(np.float32))
    return np.concatenate(outs)


def run_region(target, fd_regions, cfgs, device, seed=0, epochs=900,
               shrink=0.5):
    t0 = time.time()
    data = fd_regions[target]
    if data.get("monthly_table") is None:
        print(f"[fd48] {target}: no monthly config table (AU tier), skipped")
        return None
    model = train_fuel_zero_shot(
        fd_regions, target, seed=seed, epochs=epochs, device=device,
        p_cold=0.3, use_monthly=True, lag_months=1, wind_route_tau=1.1)
    model.wind_route_tau = float(ROUTE_TABLE.get(target, 1.1))

    split = int(len(data["rs"]) * TRAIN_FRACTION)
    sl = slice(split - SEQ_LEN, None)
    sliced = {**data,
              "rs": data["rs"][sl], "cif": data["cif"][sl],
              "fuel_shares": data["fuel_shares"][sl],
              "hours": data["hours"][sl],
              "exog": {k: v[sl] for k, v in data["exog"].items()}}
    w = build_fd_windows(
        sliced, seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE,
        monthly_table=(data.get("monthly_table_target",
                                data.get("monthly_table"))),
        lag_months=1)

    def predict(windows, tier):
        return _predict_safe(
            model, windows, data["fd_config"], data["ef_vec"],
            cold=(tier == "icfg"), device=device)

    base = {t: predict(w, t) for t in ("icfg", "i0")}
    mt_wx = monthly_wx_cfg_table(data, shrink=shrink)
    w_mwx = dict(w)
    w_mwx["config"] = monthly_config_at(mt_wx, w["origin_hours"], lag_months=1)
    model.cold_wx_hist = False
    base_off = {t: predict(w, t) for t in ("icfg",)}
    model.cold_wx_hist = True
    mwx_cwx = {t: predict(w_mwx, t) for t in ("icfg", "i0")}
    cwx = predict(w, "icfg")
    model.cold_wx_hist = False

    oh = w["origin_hours"]
    out = {"origin_hours": oh.astype("datetime64[s]").astype(np.int64) // 3600,
           "y_true": w["y_cif"].astype(np.float32),
           "pred_era5_icfg": base_off["icfg"], "pred_era5_i0": base["i0"],
           "pred_mwx_icfg": mwx_cwx["icfg"], "pred_mwx_i0": mwx_cwx["i0"],
           "pred_cwx_icfg": cwx,
           "shrink": np.float32(shrink)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{target}_seed{seed}.npz"
    np.savez_compressed(path, **out)

    y = out["y_true"].astype(float)
    for tag, p in (("base", base_off["icfg"].astype(float)),
                   ("mwx ", mwx_cwx["icfg"].astype(float)),
                   ("cwx ", cwx.astype(float))):
        e = p - y
        k = max(1, len(e) // 3)
        print(f"[fd48] {target} {tag}: MAE {np.abs(e).mean():6.2f}  "
              f"bias {e.mean():+7.2f}  lateDrift "
              f"{e[-k:].mean() - e[:k].mean():+7.2f}")
    print(f"[fd48] {target}: done ({time.time() - t0:.0f}s) -> {path.name}")
    return path


def analyze():
    rows = []
    for path in sorted(OUT_DIR.glob("*_seed0.npz")):
        t = path.name.replace("_seed0.npz", "")
        z = np.load(path)
        y = z["y_true"].astype(float)
        row = {"target": t}
        for arm in ("era5", "mwx"):
            e = z[f"pred_{arm}_icfg"].astype(float) - y
            k = max(1, len(e) // 3)
            row[f"mae_{arm}"] = round(float(np.abs(e).mean()), 2)
            row[f"bias_{arm}"] = round(float(e.mean()), 2)
            row[f"drift_{arm}"] = round(float(e[-k:].mean() - e[:k].mean()), 2)
        row["delta"] = round(row["mae_mwx"] - row["mae_era5"], 2)
        rows.append(row)
    if not rows:
        print("[fd48] no A/B dumps found")
        return
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    n = len(rows)
    better = sum(1 for r in rows if r["delta"] < 0)
    pooled_b = float(np.mean([r["mae_era5"] for r in rows]))
    pooled_m = float(np.mean([r["mae_mwx"] for r in rows]))
    print(f"\n[fd48] n={n}  mwx better in {better}/{n}  "
          f"pooled MAE {pooled_b:.2f} -> {pooled_m:.2f} "
          f"({(pooled_m - pooled_b) / pooled_b * 100:+.1f}%)")
    out = RESULTS / "fd48_mwx_ab.json"
    with open(out, "w") as f:
        json.dump({"rows": rows,
                   "pooled": {"era5": pooled_b, "mwx": pooled_m}}, f, indent=1)
    print(f"[fd48] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze()
        return

    import torch
    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[fd48] device={device} shrink={args.shrink} epochs={args.epochs}")

    cfgs = all_region_configs()
    targets = args.regions or []
    print(f"[fd48] preparing {len(cfgs)} FD regions ...")
    t0 = time.time()
    fd_regions = {}
    for name in cfgs:
        from scripts.experiments.fd47_nwp_fut_weather import (
            _safe_prepare_fd_region, DATA_DIR)
        fd_regions[name] = _safe_prepare_fd_region(name, cfgs, DATA_DIR)
    print(f"[fd48] {len(fd_regions)} regions ready ({time.time() - t0:.0f}s)")

    for tgt in targets:
        for seed in args.seeds:
            try:
                run_region(tgt, fd_regions, cfgs, device, seed=seed,
                           epochs=args.epochs, shrink=args.shrink)
            except Exception as e:  # noqa: BLE001
                print(f"[fd48][ERR] {tgt} seed {seed}: {e}")


if __name__ == "__main__":
    main()
