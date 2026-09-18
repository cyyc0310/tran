#!/usr/bin/env python
"""Cache PatchTST-sup per-origin day-ahead predictions (29 regions, seed 0).

Reproduces the official gap4 protocol exactly (same windows, same model,
same training recipe: PatchTST+RevIN, 300 epochs, Adam 1e-3, cosine warm
restarts, smooth-L1, seed 0) but additionally stores the per-origin
prediction series so figures can plot supervised curves without retraining.

MAE of the cached series must equal gap4_supervised_vs_fd41.json
patchtst_mae_s0 (trace check, asserts below).

Usage:
    PYTHONPATH=src .venv-nemed/bin/python scripts/experiments/gap4_cache_patchtst.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from transcif.config import (
    SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION, TRAIN_STRIDE, RESULTS_DIR,
)
from transcif.data.loaders import all_region_configs
from transcif.models.zeroshot.fuel import prepare_fd_region
from gap4_supervised_baselines import PatchTST, train_torch  # same module dir

DUMP = Path("results/gap4_sup_cache")
SEED = 0
EPOCHS = 300


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    DUMP.mkdir(exist_ok=True)
    gap4 = {r["target"]: r for r in json.load(
        open(RESULTS_DIR / "gap4_supervised_vs_fd41.json"))["rows"]}
    cfgs = all_region_configs()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[cache] device={device}, regions={len(gap4)}, epochs={EPOCHS}, seed={SEED}")

    for target in sorted(gap4):
        f_out = DUMP / f"{target}_s{SEED}.npz"
        if f_out.exists():
            print(f"[cache] {target}: exists, skip")
            continue
        t0 = time.time()
        data = prepare_fd_region(target, cfgs)
        rs, cif = data["rs"], data["cif"]
        split = int(len(rs) * TRAIN_FRACTION)

        sl = slice(split - SEQ_LEN, None)
        sliced = {**data, "rs": rs[sl], "cif": cif[sl],
                  "fuel_shares": data["fuel_shares"][sl],
                  "hours": data["hours"][sl],
                  "exog": {k: v[sl] for k, v in data["exog"].items()}}
        from transcif.data.fuel import build_fd_windows
        w = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                             stride=TEST_STRIDE)
        y = w["y_cif"]
        n = len(w["x_rs"])

        x_cif_train, y_cif_train = [], []
        for start in range(0, split - SEQ_LEN - HORIZON + 1, TRAIN_STRIDE):
            x_cif_train.append(cif[start:start + SEQ_LEN])
            y_cif_train.append(cif[start + SEQ_LEN:start + SEQ_LEN + HORIZON])
        x_cif_train = np.stack(x_cif_train)
        y_cif_train = np.stack(y_cif_train)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        ptst = train_torch(PatchTST(), x_cif_train, y_cif_train,
                           EPOCHS, 1e-3, device)
        x_cif_test = np.stack([
            cif[(split - SEQ_LEN) + s * TEST_STRIDE:
                (split - SEQ_LEN) + s * TEST_STRIDE + SEQ_LEN]
            for s in range(n)])
        with torch.no_grad():
            pred = ptst(torch.tensor(x_cif_test, dtype=torch.float32,
                                     device=device)).cpu().numpy()

        m = float(np.abs(pred - y).mean())
        ref = gap4[target][f"patchtst_mae_s{SEED}"]
        print(f"[cache] {target:28s} mae {m:6.2f} vs gap4 s0 {ref:6.2f} "
              f"({'OK' if abs(m-ref) < 0.5 else 'MISMATCH'}) ({time.time()-t0:.0f}s)")
        np.savez_compressed(f_out, y_true=y, pred=pred,
                            mae=m, gap4_mae=ref)
        if abs(m - ref) >= 0.5:
            raise SystemExit(f"trace check failed for {target}")

    print("[cache] all done ->", DUMP)


if __name__ == "__main__":
    main()
