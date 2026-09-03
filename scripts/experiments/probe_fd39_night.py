#!/usr/bin/env python
"""FD-39 night probe: anchor-trust gate + dynres bound + cold TTA.

Seven sentinel regions covering every mechanism family:
  anchor winners  VIC1 NSW1  (must keep their FD-34 gains)
  anchor victims  UK_05_Yorkshire US_BPAT US_NYIS (FD-34 regressions)
  dynres victim   UK_16_Scotland (I_0 21.7 -> 33.3)
  dynres winner   UK_09_East_Midlands (must keep the FD-35 gain)

Configs (seed 0, deployment v3 base = monthly x cold anchor x dynres):
  A  ANCHOR_TRUST=0, bound 220  (FD-35 replica; baseline)
  B  ANCHOR_TRUST=1, bound 220  (trust gate only)
Each run additionally reports a stride-3 multi-origin cold TTA metric for
I_cfg (information-tier clean: the cold path consumes no telemetry, so
denser origins add no extra information — pure variance reduction).

Usage:
    ANCHOR_TRUST=0|1 .venv/bin/python scripts/experiments/probe_fd39_night.py \
        --tag A --dynres-bound 220
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transcif.config import (SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION,
                             RESULTS_DIR)
from transcif.data.loaders import all_region_configs
from transcif.data.fuel import build_fd_windows
from transcif.models.zeroshot.fuel import (
    prepare_fd_region, train_fuel_zero_shot,
)
from transcif.models.fuel_decomp import FuelDecompNet

REGIONS = ["VIC1", "NSW1", "UK_05_Yorkshire", "US_BPAT", "US_NYIS",
           "UK_16_Scotland", "UK_09_East_Midlands"]

ROUTE_TAU = {"UK_16_Scotland": 1.1}


def predict(model, w, fd_cfg, ef_vec, device, cold, batch=256):
    preds = []
    with torch.no_grad():
        for i in range(0, len(w["x_rs"]), batch):
            wb = {k: (v[i:i + batch] if isinstance(v, np.ndarray) else v)
                  for k, v in w.items()}
            x_rs = torch.tensor(wb["x_rs"], device=device)
            x_fuel = torch.tensor(wb["x_fuel"], device=device)
            x_w = torch.tensor(wb["x_weather"], device=device)
            f_w = torch.tensor(wb["fut_weather"], device=device)
            f_e = torch.tensor(wb["fut_exog"], device=device)
            c = (torch.tensor(wb["config"], device=device)
                 if "config" in wb else
                 torch.tensor(np.tile(fd_cfg, (len(wb["x_rs"]), 1)),
                              device=device).float())
            e = torch.tensor(np.tile(ef_vec, (len(wb["x_rs"]), 1)),
                             device=device)
            hm = torch.zeros(len(x_rs), 1, device=device) if cold \
                else torch.ones(len(x_rs), 1, device=device)
            p, _, _ = model(x_rs, x_fuel, x_w, f_w, f_e, c, e, hist_mask=hm)
            preds.append(p.cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dynres-bound", type=float, default=220.0)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--tta-stride", type=int, default=3)
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfgs = all_region_configs()
    fd_regions = {n: prepare_fd_region(n, cfgs) for n in cfgs}

    out_path = RESULTS_DIR / f"fd39_night_{args.tag}.json"
    results = []
    for target in args.regions:
        t0 = time.time()
        model = train_fuel_zero_shot(
            fd_regions, target, seed=0, epochs=args.epochs, device=device,
            use_monthly=True, dynamic_residual=True,
            dynamic_residual_bound=args.dynres_bound,
            wind_route_tau=ROUTE_TAU.get(target, 0.45))
        data = fd_regions[target]
        split = int(len(data["rs"]) * TRAIN_FRACTION)
        sl = slice(split - SEQ_LEN, None)
        sliced = {**data, "rs": data["rs"][sl], "cif": data["cif"][sl],
                  "fuel_shares": data["fuel_shares"][sl],
                  "hours": data["hours"][sl],
                  "exog": {k: v[sl] for k, v in data["exog"].items()}}
        w24 = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                               stride=TEST_STRIDE,
                               monthly_table=data.get("monthly_table"),
                               lag_months=1)
        y = w24["y_cif"]
        ef_vec = data["ef_vec"].astype(np.float32)
        cfg_i = predict(model, w24, data["fd_config"], ef_vec, device,
                        cold=True)
        i0 = predict(model, w24, data["fd_config"], ef_vec, device,
                     cold=False)
        row = {"tag": args.tag, "target": target, "seed": 0,
               "anchor_trust": round(float(data["anchor_trust"]), 3),
               "bound": args.dynres_bound,
               "train_s": round(time.time() - t0, 1),
               "mae_cfg": float(np.abs(cfg_i - y).mean()),
               "mae_i0": float(np.abs(i0 - y).mean())}

        # --- stride-k multi-origin cold TTA for I_cfg.
        wtta = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                                stride=args.tta_stride,
                                monthly_table=data.get("monthly_table"),
                                lag_months=1)
        if len(wtta["x_rs"]) > len(w24["x_rs"]):
            p3 = predict(model, wtta, data["fd_config"], ef_vec, device,
                         cold=True, batch=192)
            acc = np.zeros(len(data["rs"][sl]), np.float64)
            cnt = np.zeros(len(data["rs"][sl]), np.float64)
            oh = wtta["origin_hours"]
            base = data["hours"][sl]
            idx0 = {t: i for i, t in enumerate(base)}
            for j in range(len(oh)):
                s = idx0[oh[j]]
                acc[s:s + HORIZON] += p3[j]
                cnt[s:s + HORIZON] += 1
            tta = []
            for j in range(len(w24["origin_hours"])):
                s = idx0[w24["origin_hours"][j]]
                tta.append((acc[s:s + HORIZON] /
                            np.maximum(cnt[s:s + HORIZON], 1)))
            tta = np.stack(tta)
            row["mae_cfg_tta"] = float(np.abs(tta - y).mean())
        results.append(row)
        print(f"[{args.tag}] {target:24s} trust={row['anchor_trust']:.2f} "
              f"cfg={row['mae_cfg']:.1f} tta={row.get('mae_cfg_tta', -1):.1f} "
              f"i0={row['mae_i0']:.1f} ({row['train_s']}s)", flush=True)
        out_path.write_text(json.dumps(results, indent=1))
    print(f"[fd39] wrote {out_path}")


if __name__ == "__main__":
    main()
