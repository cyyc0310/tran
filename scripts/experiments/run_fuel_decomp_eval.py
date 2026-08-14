#!/usr/bin/env python
"""FuelDecompNet LORO evaluation (TransCIF-FD, Phase FD-1).

Evaluates the fuel-decomposed physics-structured model on two information
tiers simultaneously (one set of weights, cold-mode dropout at training):

    I_0    config + live share telemetry        (paper-comparable)
    I_cfg  config + weather + calendar only     (China deployment tier)

Baselines:
    persistence        lag-24 CIF (forecast reference; needs CIF history)
    config-constant    mean_rs*ef_r + (1-mean_rs)*ef_nr — the "official
                       annual factor" analog available to any region
    monthly-constant   per-month true mean (oracle level anchor)

Metrics: MAE/RMSE + shape metrics (diurnal MAE, monthly-shape MAE,
Spearman hourly ranking) — the shape/ranking metrics are what carbon-aware
scheduling in telemetry-free regions actually consumes.

Usage:
    python scripts/experiments/run_fuel_decomp_eval.py                 # 8 regions x seeds 0-1
    python scripts/experiments/run_fuel_decomp_eval.py --full          # 29 regions x seeds 0-4
    python scripts/experiments/run_fuel_decomp_eval.py --regions US_CISO --seeds 0 --epochs 60
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import torch

from transcif.config import (
    SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION, SEEDS_FULL, RESULTS_DIR,
)
from transcif.data.loaders import all_region_configs
from transcif.data.fuel import build_fd_windows
from transcif.models.zeroshot.fuel import (
    prepare_fd_region, train_fuel_zero_shot, predict_fuel_windows,
    shape_metrics_with_months,
)
from transcif.evaluation.metrics import compute_metrics

QUICK_REGIONS = [
    "QLD1", "NSW1", "VIC1", "SA1",
    "US_BPAT", "US_PJM",
    "UK_02_South_Scotland", "UK_08_West_Midlands",
]


def build_target_test_windows(data):
    """Test-split windows (last 20%, stride 24) matching the paper protocol."""
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    sl = slice(split - SEQ_LEN, None)
    sliced = {
        **data,
        "rs": data["rs"][sl], "cif": data["cif"][sl],
        "fuel_shares": data["fuel_shares"][sl], "hours": data["hours"][sl],
        "exog": {k: v[sl] for k, v in data["exog"].items()},
    }
    return build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                            stride=TEST_STRIDE)


def evaluate_one(target, fd_regions, seed, epochs, device, p_cold=0.3,
                 p_mix=0.0, use_hypernet=False):
    t0 = time.time()
    model = train_fuel_zero_shot(
        fd_regions, target, seed=seed, epochs=epochs, device=device,
        p_cold=p_cold, p_mix=p_mix, use_hypernet=use_hypernet)
    data = fd_regions[target]
    w = build_target_test_windows(data)
    n = len(w["x_rs"])
    if n == 0:
        return None
    y = w["y_cif"]
    ef_vec = data["ef_vec"].astype(np.float32)
    fd_cfg = data["fd_config"]

    res = {"target": target, "seed": seed, "n_test": n,
           "mean_rs": data["mean_rs"], "ef_nr": data["ef_nr"],
           "has_fuel": bool(data["has_fuel"]),
           "train_s": round(time.time() - t0, 1)}

    # --- FuelDecompNet, both tiers
    for tier, cold in (("i0", False), ("i_cfg", True)):
        cif, _, _ = predict_fuel_windows(model, w, fd_cfg, ef_vec,
                                         cold=cold, device=device)
        res[f"fuel_{tier}"] = shape_metrics_with_months(cif, y, w["origin_hours"])
        res[f"fuel_{tier}"]["std_metrics"] = compute_metrics(cif, y)

    # --- persistence (lag-24 CIF; forecast reference, needs CIF history)
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    cif_off = data["cif"][split - SEQ_LEN:]
    persist = np.stack([
        cif_off[s + SEQ_LEN - HORIZON:s + SEQ_LEN]
        for s in range(0, len(cif_off) - SEQ_LEN - HORIZON + 1, TEST_STRIDE)
    ])[:n]
    res["persistence"] = shape_metrics_with_months(persist, y, w["origin_hours"])

    # --- config-constant (official-annual-factor analog; deployment-legal)
    cfg_const = np.full_like(y, data["mean_rs"] * data["ef_r"]
                             + (1 - data["mean_rs"]) * data["ef_nr"])
    res["config_constant"] = shape_metrics_with_months(cfg_const, y, w["origin_hours"])

    # --- monthly-constant (oracle level anchor)
    months = w["origin_hours"].month.values
    monthly = np.zeros_like(y, dtype=np.float64)
    for m in np.unique(months):
        monthly[months == m] = y[months == m].mean()
    res["monthly_constant"] = shape_metrics_with_months(monthly, y, w["origin_hours"])

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=QUICK_REGIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--max-windows", type=int, default=700)
    ap.add_argument("--p-cold", type=float, default=0.3)
    ap.add_argument("--p-mix", type=float, default=0.0,
                    help="fraction of steps on synthetic mixed pseudo-grids (FD-2)")
    ap.add_argument("--use-hypernet", action="store_true",
                    help="config hypernet generates dynamic head weights (FD-2)")
    ap.add_argument("--full", action="store_true",
                    help="29-region protocol, seeds 0-4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.full:
        args.regions = None
        args.seeds = SEEDS_FULL

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[fuel-eval] device={device} epochs={args.epochs} "
          f"p_cold={args.p_cold} p_mix={args.p_mix} "
          f"max_windows={args.max_windows}")

    cfgs = all_region_configs()
    # UK_18_GB is the national aggregate — excluded from the LORO protocol.
    pool_names = [r for r in cfgs if r != "UK_18_GB"]
    targets = args.regions or pool_names

    print("[fuel-eval] preparing FD data for all regions ...")
    fd_regions = {}
    for name in pool_names:
        fd_regions[name] = prepare_fd_region(name, cfgs)
    print(f"[fuel-eval] {len(fd_regions)} regions ready")

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / ("fuel_decomp_eval_full.json" if args.full
                       else "fuel_decomp_eval_quick.json"))
    # Resume support: skip (target, seed) pairs already recorded.
    rows = []
    if out_path.exists():
        with open(out_path) as f:
            doc = json.load(f)
        rows = doc.get("rows", [])
        done = {(r["target"], r["seed"]) for r in rows}
        print(f"[fuel-eval] resuming: {len(done)} pairs already done")
    else:
        done = set()

    for target in targets:
        for seed in args.seeds:
            if (target, seed) in done:
                continue
            t0 = time.time()
            try:
                row = evaluate_one(target, fd_regions, seed, args.epochs,
                                   device, p_cold=args.p_cold,
                                   p_mix=args.p_mix,
                                   use_hypernet=args.use_hypernet)
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] {target} seed {seed} failed: {e}")
                continue
            if row is None:
                continue
            rows.append(row)
            r0, rc = row["fuel_i0"]["mae"], row["fuel_i_cfg"]["mae"]
            print(f"  {target:28s} seed {seed}: I_0 MAE {r0:6.1f} | "
                  f"I_cfg MAE {rc:6.1f} (persist {row['persistence']['mae']:6.1f}, "
                  f"cfg-const {row['config_constant']['mae']:6.1f}) "
                  f"[{time.time() - t0:.0f}s]")
            with open(out_path, "w") as f:
                json.dump({"rows": rows, "meta": {
                    "epochs": args.epochs, "p_cold": args.p_cold,
                    "p_mix": args.p_mix,
                    "use_hypernet": bool(args.use_hypernet),
                    "max_windows": args.max_windows}}, f, indent=1)

    # --- summary
    if rows:
        for col in ("fuel_i0", "fuel_i_cfg", "persistence",
                    "config_constant", "monthly_constant"):
            maes = [r[col]["mae"] for r in rows if r.get(col)]
            print(f"{col:18s} median MAE {np.median(maes):7.2f}  "
                  f"mean {np.mean(maes):7.2f}  n={len(maes)}")
        for col in ("fuel_i0", "fuel_i_cfg"):
            sp = [r[col]["spearman"] for r in rows if r.get(col)]
            dm = [r[col]["diurnal_mae"] for r in rows if r.get(col)]
            print(f"{col:18s} median Spearman {np.median(sp):.3f}  "
                  f"diurnal MAE {np.median(dm):.2f}")
        print(f"[fuel-eval] wrote {out_path} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
