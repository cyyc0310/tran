#!/usr/bin/env python
"""FuelDecompNet joint fine-tuning — the I_J tier (Phase FD-9).

Mirrors the legacy joint protocol (Phase 8/9): the target supplies its
FIRST 12 test origins (288 h of CIF labels) for calibration; evaluation
happens on the NEXT 12 disjoint origins.  Fine-tuning follows the Stage-2
discipline that historically broke the equalizer:

    * frozen:    config encoder, EF correction, anchor gate (level pathways)
    * unfrozen:  per-hour dynamic heads (solar/wind modulation, baseload
                 delta, thermal split, aggregate DLinear heads)
    * loss:      MAE + 0.5 * adversarial-persistence (10% relative margin)
    * guard:     internal-val gate — revert if the fine-tuned model is
                 clearly worse (> eps) on 3 disjoint inner-validation
                 origins carved after the eval split

Usage:
    python scripts/experiments/run_fuel_joint.py                       # 8 regions x seeds 0-1
    python scripts/experiments/run_fuel_joint.py --full                # 29 x 5
    python scripts/experiments/run_fuel_joint.py --regions QLD1 --seeds 0
"""

import argparse
import copy
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
    shape_metrics_with_months, make_zs_plus_share_fn,
)
from transcif.training.adversarial_loss import adversarial_persistence_loss
from transcif.evaluation.metrics import compute_metrics
from transcif.calibration.zs_plus import zs_plus_predict

QUICK_REGIONS = [
    "QLD1", "NSW1", "VIC1", "SA1",
    "US_BPAT", "US_PJM",
    "UK_02_South_Scotland", "UK_08_West_Midlands",
]
LEVEL_MODULES = ("cfg_mlp", "ef_corr", "anchor_gate", "therm_cfg",
                 "base_prior", "rs_cfg_bias", "rs_gate")


def _snapshot(model):
    return copy.deepcopy(model.state_dict())


def _restore(model, state):
    model.load_state_dict(state)


def _target_windows_at(data, origins):
    """Build FD windows whose FORECAST origins match the given absolute
    hour positions (origin = start + SEQ_LEN)."""
    hours = data["hours"]
    pos = {h: i for i, h in enumerate(hours)}
    starts = [pos[h] - SEQ_LEN for h in origins if h in pos]
    starts = [s for s in starts if s >= 0 and s + SEQ_LEN + HORIZON <= len(hours)]
    sliced = data
    return build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                            stride=1, starts=starts)


def _persistence_cif(data, w):
    """Lag-24 CIF persistence aligned to the windows' forecast horizons."""
    out = np.zeros((len(w["x_rs"]), HORIZON), dtype=np.float32)
    hours = data["hours"]
    pos = {h: i for i, h in enumerate(hours)}
    for i, oh in enumerate(w["origin_hours"]):
        t0 = pos[oh]
        out[i] = data["cif"][t0 - 24:t0 - 24 + HORIZON]
    return out


def finetune_on_origins(model, data, calib_w, device, steps=40, lr=1e-3,
                        adv_weight=0.5, margin=0.10):
    """Stage-2 style fine-tune: freeze level pathways, unfreeze dynamic
    heads, optimise MAE + adversarial-persistence on 288 h of labels."""
    for name in LEVEL_MODULES:
        for p in getattr(model, name).parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return model
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    ef = torch.tensor(np.tile(data["ef_vec"].astype(np.float32),
                              (len(calib_w["x_rs"]), 1))).to(device)
    cfg = torch.tensor(calib_w.get(
        "config", np.tile(data["fd_config"],
                          (len(calib_w["x_rs"]), 1)))).to(device)
    tensors = {k: torch.tensor(calib_w[k]).to(device)
               for k in ("x_rs", "x_fuel", "x_weather", "fut_weather",
                         "fut_exog", "y_cif")}
    persist = torch.tensor(_persistence_cif(data, calib_w)).to(device)
    model.train()
    n = len(tensors["x_rs"])
    for step in range(steps):
        idx = torch.randperm(n)[:min(64, n)]
        b = {k: v[idx] for k, v in tensors.items()}
        cif_hat, _, _ = model(b["x_rs"], b["x_fuel"], b["x_weather"],
                              b["fut_weather"], b["fut_exog"],
                              cfg[idx], ef[idx],
                              hist_mask=torch.ones(len(idx), 1, device=device))
        loss = torch.abs(cif_hat - b["y_cif"]).mean()
        if adv_weight > 0:
            loss = loss + adv_weight * adversarial_persistence_loss(
                cif_hat, persist[idx], margin=margin)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.eval()
    return model


def evaluate_origins(model, data, w, device, zs_plus=True):
    """Predict the given windows; optionally wrap with ZS+ calibration.

    Returns keys i0 / i_plus / persistence (callers add tier prefixes).
    """
    out = {}
    cif, _, _ = predict_fuel_windows(model, w, data["fd_config"],
                                     data["ef_vec"].astype(np.float32),
                                     cold=False, device=device)
    y = w["y_cif"]
    out["i0"] = shape_metrics_with_months(cif, y, w["origin_hours"])
    if zs_plus:
        share_fn = make_zs_plus_share_fn(model, data, device=device)
        origins = []
        pos = {h: i for i, h in enumerate(data["hours"])}
        for oh in w["origin_hours"]:
            origins.append(pos[oh])
        zsp = zs_plus_predict(model, data["fd_config"], data["rs"], data["cif"],
                              data["ef_r"], data["ef_nr"], origins,
                              share_fn=share_fn)
        out["i_plus"] = shape_metrics_with_months(zsp, y, w["origin_hours"])
    out["persistence"] = shape_metrics_with_months(
        _persistence_cif(data, w), y, w["origin_hours"])
    return out, cif


def run_one(target, fd_regions, seed, epochs, device, steps=40,
            gate_eps=2.0):
    t0 = time.time()
    model = train_fuel_zero_shot(fd_regions, target, seed=seed,
                                 epochs=epochs, device=device)
    data = fd_regions[target]

    # Test origins (same geometry as the standard eval), split 12/12/3.
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    offset_hours = data["hours"][split - SEQ_LEN:]
    origin_hours = [offset_hours[s + SEQ_LEN]
                    for s in range(0, len(offset_hours) - SEQ_LEN - HORIZON + 1,
                                   TEST_STRIDE)]
    if len(origin_hours) < 27:
        return None
    calib_o = origin_hours[:12]
    eval_o = origin_hours[12:24]
    inner_o = origin_hours[24:27]  # internal-val guard, disjoint from eval

    calib_w = _target_windows_at(data, calib_o)
    eval_w = _target_windows_at(data, eval_o)
    inner_w = _target_windows_at(data, inner_o)
    if min(len(calib_w["x_rs"]), len(eval_w["x_rs"])) < 6:
        return None

    # Baseline (no fine-tune) on the same eval origins — paired reference.
    base_res, _ = evaluate_origins(model, data, eval_w, device)

    # Internal-val reference on the guard origins BEFORE fine-tuning.
    gate_fired = False
    base_inner_mae = None
    if len(inner_w["x_rs"]) >= 3:
        cif_b, _, _ = predict_fuel_windows(
            model, inner_w, data["fd_config"],
            data["ef_vec"].astype(np.float32), cold=False, device=device)
        base_inner_mae = float(np.abs(cif_b - inner_w["y_cif"]).mean())

    # Joint fine-tune + internal-val gate.
    snap = _snapshot(model)
    model = finetune_on_origins(model, data, calib_w, device, steps=steps)
    if base_inner_mae is not None:
        cif_ft, _, _ = predict_fuel_windows(
            model, inner_w, data["fd_config"],
            data["ef_vec"].astype(np.float32), cold=False, device=device)
        ft_inner_mae = float(np.abs(cif_ft - inner_w["y_cif"]).mean())
        if ft_inner_mae > base_inner_mae + gate_eps:
            _restore(model, snap)
            gate_fired = True
    ft_res, _ = evaluate_origins(model, data, eval_w, device)

    res = {"target": target, "seed": seed, "gate_fired": gate_fired,
           "elapsed_s": round(time.time() - t0, 1)}
    for k, v in base_res.items():
        res[f"base_{k}"] = v
    for k, v in ft_res.items():
        res[f"joint_{k}"] = v
    return res


def _restore_or_new(model, snap):
    """Deprecated helper kept for import compatibility."""
    import copy as _copy  # noqa: PLC0415
    m2 = _copy.deepcopy(model)
    m2.load_state_dict(snap)
    return m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=QUICK_REGIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.full:
        args.regions = None
        args.seeds = SEEDS_FULL

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    cfgs = all_region_configs()
    pool = list(cfgs)
    targets = args.regions or pool

    print(f"[fuel-joint] device={device} epochs={args.epochs} steps={args.steps}")
    fd_regions = {n: prepare_fd_region(n, cfgs) for n in pool}
    print(f"[fuel-joint] {len(fd_regions)} regions ready")

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / ("fuel_joint_full.json" if args.full
                       else "fuel_joint_quick.json"))
    rows = []
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            doc = json.load(f)
        rows = doc.get("rows", [])
        done = {(r["target"], r["seed"]) for r in rows}
        print(f"[fuel-joint] resuming: {len(done)} done")

    for target in targets:
        for seed in args.seeds:
            if (target, seed) in done:
                continue
            try:
                row = run_one(target, fd_regions, seed, args.epochs, device,
                              steps=args.steps)
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] {target} seed {seed}: {e}")
                continue
            if row is None:
                continue
            rows.append(row)
            jb = row["joint_i_plus"]["mae"]
            bb = row["base_i_plus"]["mae"]
            print(f"  {target:28s} seed {seed}: I_J {jb:6.1f} "
                  f"(base {bb:6.1f}, persist {row['joint_persistence']['mae']:6.1f}) "
                  f"{'[gate]' if row['gate_fired'] else ''} "
                  f"[{row['elapsed_s']:.0f}s]")
            with open(out_path, "w") as f:
                json.dump({"rows": rows, "meta": {
                    "epochs": args.epochs, "steps": args.steps}}, f, indent=1)

    if rows:
        for a, b, name in (("joint_i_plus", "base_i_plus", "I_J vs base I_+"),
                           ("joint_i_plus", "joint_persistence", "I_J vs persistence")):
            from scipy.stats import wilcoxon  # noqa: PLC0415
            x = np.array([r[a]["mae"] for r in rows if r.get(a)])
            y = np.array([r[b]["mae"] for r in rows if r.get(b)])
            _, p = wilcoxon(x, y)
            print(f"{name:22s}: {np.median(y):6.2f} -> {np.median(x):6.2f} "
                  f"| win {(x < y).mean() * 100:4.0f}% | p={p:.1e} | n={len(x)}")
        gates = sum(r.get("gate_fired", False) for r in rows)
        print(f"[fuel-joint] gate fired {gates}/{len(rows)} | wrote {out_path}")


if __name__ == "__main__":
    main()
