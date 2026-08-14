"""Dump per-origin prediction residuals for DM tests + error decomposition.

For each (target, seed) pair, train the 5 direction models ONCE, then evaluate
every method on the SAME held-out eval origins (the disjoint second half of the
test split, matching ``run_joint_train_full.split_origins``). This gives
aligned per-origin ``(n_eval, HORIZON)`` predictions for proper paired
Diebold-Mariano tests and bias/variance decomposition.

Outputs ``results/residuals/{target}_seed{seed}.npz`` containing:
    y_true               : (n_eval, HORIZON) ground-truth CIF
    eval_origins         : (n_eval,) origin indices
    pred_{method}        : (n_eval, HORIZON) predictions, per method
    err_{method}         : (n_eval, HORIZON) abs(pred - y_true)

Methods: rag, phys, causal, icl, hier, equal, basismix,
         rag_plus, phys_plus, causal_plus, icl_plus, hier_plus,
         equal_plus, basismix_plus, persistence, persistence_cif, joint_trained.

Reuses (does not duplicate) direction training: ``_train_directions`` trains the
5 models on the target once; those frozen predictions feed both the zero-shot
methods AND the joint-training Stage 1/2 forward pass.

Usage:
    .venv/bin/python scripts/experiments/run_residual_dump.py
    .venv/bin/python scripts/experiments/run_residual_dump.py --regions QLD1 NSW1
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transcif.calibration.differentiable_zs_plus import DifferentiableZSPlus
from transcif.calibration.zs_plus import zs_plus_predict
from transcif.config import HORIZON, SEQ_LEN
from transcif.data.loaders import all_region_configs, load_region_data
from transcif.models.zeroshot.fusion import (
    DIRECTION_ORDER,
    BasisMixFusion,
    EqualWeightFusion,
    FusionModel,
)

# Reuse helpers from the existing pipelines (no behaviour duplication)
from scripts.experiments.run_fused_five_full import _train_basismix
from scripts.experiments.run_joint_train import (
    _build_share_fn,
    _cif_to_share,
    _persistence_cif_full,
    _stage,
    _train_directions,
)
from scripts.experiments._shared import split_origins

DEFAULT_REGIONS = [
    "QLD1", "NSW1", "VIC1", "SA1",
    "US_BPAT", "US_PJM",
    "UK_02_South_Scotland", "UK_08_West_Midlands",
]

PERSIST_METHODS = [
    "rag", "phys", "causal", "icl", "hier",
    "equal", "basismix",
    "rag_plus", "phys_plus", "causal_plus", "icl_plus", "hier_plus",
    "equal_plus", "basismix_plus",
    "persistence", "persistence_cif", "joint_trained",
]


# ---------------------------------------------------------------------------
# ZS+ share_fn builders (mirror run_fused_five_full._eval_* but origin-agnostic)
# ---------------------------------------------------------------------------

class _SingleDirShare:
    """share_fn for a single direction: window -> renewable share."""

    def __init__(self, pred_fn, config, ef_r, ef_nr):
        self.pred_fn = pred_fn
        self.config = config
        self.ef_r = ef_r
        self.ef_nr = ef_nr

    def __call__(self, x_window_np):
        cif_pred = self.pred_fn(
            x_window_np[None, :].astype(np.float32),
            self.config, self.ef_r, self.ef_nr,
        ).reshape(-1)
        share = (cif_pred - self.ef_nr) / (self.ef_r - self.ef_nr + 1e-8)
        return np.clip(share, 0.0, 1.0)


def _zs_plus_origins_for(rs, eval_origins):
    """Identity helper: we already have explicit eval origins."""
    return list(eval_origins)


# ---------------------------------------------------------------------------
# Per-pair dump
# ---------------------------------------------------------------------------

def dump_pair(target, all_regions, seed, out_path, n_train=12, n_eval=12):
    """Train + evaluate all methods on one (target, seed), write .npz."""
    src_names = [n for n in all_regions if n != target][:3]
    # Force the legacy 2-D config ([mean_rs, ef_nr/1000]) to match the
    # committed fused_five_full.json protocol (2-D-config era). The working-copy
    # loaders.py builds a richer 12-D multi-fuel config for US/UK ("Stage A",
    # in progress), but the causal/physics direction models were not yet
    # updated to accept it — using it here raises a shape mismatch. Slicing to
    # config[:2] reproduces the committed-results protocol and keeps the dump
    # apples-to-apples with results/fused_five_significance.json. The original
    # all_regions entries are not mutated (shallow-copied).
    small_regions = {}
    for n in [target] + src_names:
        rd = dict(all_regions[n])
        rd["config"] = np.asarray(rd["config"], dtype=np.float32)[:2]
        small_regions[n] = rd

    data = all_regions[target]
    rs_np = data["rs"].astype(np.float32)
    cif_np = data["cif"].astype(np.float32)
    # Use the 2-D config from small_regions (sliced above) so predictors —
    # trained on the 2-D config — receive a matching config at inference.
    config = small_regions[target]["config"].astype(np.float32)
    ef_r = float(data["ef_r"])
    ef_nr = float(data["ef_nr"])
    rs_t = torch.as_tensor(rs_np, dtype=torch.float32)
    cif_t = torch.as_tensor(cif_np, dtype=torch.float32)

    train_origins, eval_origins = split_origins(rs_np, n_train=n_train, n_eval=n_eval)
    n_eval_actual = len(eval_origins)

    # 1. Train 5 direction models on target (the expensive step, once).
    t0 = time.time()
    predictors = _train_directions(small_regions, target, seed=seed)
    print(f"  [dirs] trained 5 directions in {time.time()-t0:.1f}s", flush=True)

    # 2. Source stacks + BasisMix head (for basismix fusion).
    from transcif.models.zeroshot.collector import collect_source_stacks
    t0 = time.time()
    src_stacks, src_true, _ = collect_source_stacks(
        small_regions, target, seed=seed, device=None,
        source_names=src_names, progress=False,
    )
    bm_model = _train_basismix(src_stacks, src_true, predictors, seed)
    print(f"  [basismix] head trained in {time.time()-t0:.1f}s", flush=True)

    # 3. Frozen direction preds on eval origins: (5, n_eval, HORIZON)
    frozen_eval = np.zeros((5, n_eval_actual, HORIZON), dtype=np.float32)
    for o_idx, t0_orig in enumerate(eval_origins):
        x_win = rs_np[t0_orig - SEQ_LEN:t0_orig][None, :].astype(np.float32)
        for d_idx, name in enumerate(DIRECTION_ORDER):
            try:
                pred = predictors[name](x_win, config, ef_r, ef_nr)[0]
            except Exception:
                last = rs_np[t0_orig - HORIZON:t0_orig]
                pred = last * ef_r + (1 - last) * ef_nr
            frozen_eval[d_idx, o_idx, :] = pred

    y_true = np.stack([cif_np[o:o + HORIZON] for o in eval_origins])  # (n_eval, H)

    preds = {}

    # --- Base (no ZS+) single directions + fusions ---
    for d_idx, name in enumerate(DIRECTION_ORDER):
        preds[name] = frozen_eval[d_idx]
    preds["equal"] = frozen_eval.mean(axis=0)
    # basismix: head over (n_eval, 5, H)
    stack_eval = np.transpose(frozen_eval, (1, 0, 2))  # (n_eval, 5, H)
    preds["basismix"] = bm_model.predict_cif_from_stack(stack_eval)

    # --- Persistence baselines ---
    # rs-based (matches fused_five_full._eval_persistence)
    last_rs = np.stack([rs_np[o - HORIZON:o] for o in eval_origins])
    preds["persistence"] = last_rs * ef_r + (1 - last_rs) * ef_nr
    # CIF-lag (matches run_joint_train._persistence_cif_full)
    preds["persistence_cif"] = np.stack([cif_np[o - HORIZON:o] for o in eval_origins])

    # --- ZS+ calibrated versions ---
    origins_zs = _zs_plus_origins_for(rs_np, eval_origins)

    def _run_zsplus(share_fn_obj, label):
        try:
            cf = zs_plus_predict(
                model=None, config=config, rs=rs_np, cif=cif_np,
                ef_r=ef_r, ef_nr=ef_nr, origins=origins_zs,
                share_fn=share_fn_obj,
            )
            preds[label] = np.asarray(cf, dtype=np.float32)
        except Exception as e:
            print(f"    [WARN] ZS+ {label} failed: {e}", flush=True)

    # single-direction ZS+
    for name in DIRECTION_ORDER:
        share_obj = _SingleDirShare(predictors[name], config, ef_r, ef_nr)
        _run_zsplus(share_obj, f"{name}_plus")
    # fusion ZS+
    equal_model = FusionModel(EqualWeightFusion(), predictors=predictors)
    equal_model.configure_for_target(config, ef_r, ef_nr)
    _run_zsplus(equal_model.share_fn, "equal_plus")
    bm_model.configure_for_target(config, ef_r, ef_nr)
    _run_zsplus(bm_model.share_fn, "basismix_plus")

    # --- Joint-trained (Stage 1 + Stage 2 on train_origins, eval on eval_origins) ---
    try:
        joint_pred = _train_and_eval_joint(
            predictors, small_regions, target, rs_np, cif_np, rs_t, cif_t,
            config, ef_r, ef_nr, train_origins, eval_origins, seed,
        )
        preds["joint_trained"] = joint_pred
        print(f"  [joint] stage1+2 done", flush=True)
    except Exception as e:
        print(f"  [WARN] joint_trained failed: {e}", flush=True)
        traceback.print_exc()

    # 4. Compute abs errors + save
    out = {
        "y_true": y_true,
        "eval_origins": np.asarray(eval_origins, dtype=np.int64),
    }
    for m in PERSIST_METHODS:
        if m in preds:
            p = preds[m]
            out[f"pred_{m}"] = p
            out[f"err_{m}"] = np.abs(p - y_true)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    # quick MAE summary
    summary = {m: float(np.abs(preds[m] - y_true).mean()) for m in PERSIST_METHODS if m in preds}
    return summary


def _train_and_eval_joint(predictors, small_regions, target,
                           rs_np, cif_np, rs_t, cif_t,
                           config, ef_r, ef_nr,
                           train_origins, eval_origins, seed,
                           n_steps_s1=30, n_steps_s2=30,
                           lr_s1=5e-2, lr_s2=1e-2, margin=0.10):
    """Replicate run_joint_train Stage1+2 + held-out eval, return (n_eval, H) preds.

    Reuses the imported ``_stage`` so the trained module is identical to what
    ``run_joint_train_full`` produced; only the per-origin eval output differs.
    """
    # frozen preds on train origins: (5, n_train, H)
    n_train = len(train_origins)
    frozen_train = np.zeros((5, n_train, HORIZON), dtype=np.float32)
    for o_idx, t0_orig in enumerate(train_origins):
        x_win = rs_np[t0_orig - SEQ_LEN:t0_orig][None, :].astype(np.float32)
        for d_idx, name in enumerate(DIRECTION_ORDER):
            try:
                pred = predictors[name](x_win, config, ef_r, ef_nr)[0]
            except Exception:
                last = rs_np[t0_orig - HORIZON:t0_orig]
                pred = last * ef_r + (1 - last) * ef_nr
            frozen_train[d_idx, o_idx, :] = pred
    frozen_train_t = torch.as_tensor(frozen_train, dtype=torch.float32)

    y_true_train = torch.stack([cif_t[o:o + HORIZON] for o in train_origins])
    persistence_train = _persistence_cif_full(rs_np, cif_np, train_origins)

    zs_plus = DifferentiableZSPlus()
    fusion = BasisMixFusion()
    correction = nn.Parameter(torch.zeros(5, HORIZON))

    params1 = list(zs_plus.parameters()) + list(fusion.parameters())
    _stage("stage1", params1, zs_plus, fusion, rs_t, cif_t, frozen_train_t,
           train_origins, ef_r, ef_nr, persistence_train, y_true_train,
           n_steps=n_steps_s1, lr=lr_s1, margin=margin, correction=None)

    params2 = list(zs_plus.parameters()) + list(fusion.parameters()) + [correction]
    _stage("stage2", params2, zs_plus, fusion, rs_t, cif_t, frozen_train_t,
           train_origins, ef_r, ef_nr, persistence_train, y_true_train,
           n_steps=n_steps_s2, lr=lr_s2, margin=margin, correction=correction)

    # frozen preds on eval origins: (5, n_eval, H)
    n_eval = len(eval_origins)
    frozen_eval = np.zeros((5, n_eval, HORIZON), dtype=np.float32)
    for o_idx, t0_orig in enumerate(eval_origins):
        x_win = rs_np[t0_orig - SEQ_LEN:t0_orig][None, :].astype(np.float32)
        for d_idx, name in enumerate(DIRECTION_ORDER):
            try:
                pred = predictors[name](x_win, config, ef_r, ef_nr)[0]
            except Exception:
                last = rs_np[t0_orig - HORIZON:t0_orig]
                pred = last * ef_r + (1 - last) * ef_nr
            frozen_eval[d_idx, o_idx, :] = pred
    frozen_eval_t = torch.as_tensor(frozen_eval, dtype=torch.float32)

    joint_pred = np.zeros((n_eval, HORIZON), dtype=np.float32)
    with torch.no_grad():
        for o_idx in range(n_eval):
            preds_o = frozen_eval_t[:, o_idx, :] + correction
            fused = fusion(preds_o.unsqueeze(0)).squeeze(0)
            share = _cif_to_share(fused, ef_r, ef_nr).clamp(0.0, 1.0)
            sfn = _build_share_fn(share)
            pred = zs_plus(rs_t, cif_t, ef_r, ef_nr,
                           [eval_origins[o_idx]], sfn)
            joint_pred[o_idx] = pred.squeeze(0).cpu().numpy()
    return joint_pred


def main():
    ap = argparse.ArgumentParser(description="Per-origin residual dump (Phase 5.2 DM + decomposition)")
    ap.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--out-dir", default="results/residuals")
    args = ap.parse_args()

    print("[LOAD] regions...", flush=True)
    all_configs = all_region_configs()
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}
    print(f"[LOAD] {len(all_regions)} regions", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in args.regions:
        for seed in args.seeds:
            out_path = out_dir / f"{target}_seed{seed}.npz"
            if out_path.exists():
                print(f"[SKIP] {target} seed{seed} exists", flush=True)
                continue
            if target not in all_regions:
                print(f"[SKIP] {target} not in regions", flush=True)
                continue
            print(f"\n=== {target} seed{seed} ===", flush=True)
            t0 = time.time()
            try:
                summary = dump_pair(
                    target, all_regions, seed, out_path,
                    n_train=args.n_train, n_eval=args.n_eval,
                )
                jt = summary.get("joint_trained", float("nan"))
                bm = summary.get("basismix_plus", float("nan"))
                pers = summary.get("persistence", float("nan"))
                print(f"  MAE: joint={jt:.2f}  basismix+={bm:.2f}  persistence={pers:.2f}",
                      flush=True)
                print(f"  [DONE] {time.time()-t0:.1f}s -> {out_path}", flush=True)
            except Exception as e:
                print(f"  [ERROR] {target}/{seed}: {e}", flush=True)
                traceback.print_exc()

    print("\n[FINISHED]", flush=True)


if __name__ == "__main__":
    main()
