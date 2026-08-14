"""Torch-native joint training pipeline (Phase 9, Step 3).

Replaces the frozen-prediction + ``(5, HORIZON)`` correction proxy in
``run_joint_train.py`` with true end-to-end finetuning of three "easy"
directions (causal / phys_irm / hier), while RAG and ICL stay frozen constants.

Stages:
  Stage 1: all direction params frozen; train only ``LearnedFusion`` +
           ``DifferentiableZSPlus`` (establishes a per-window fusion base).
  Stage 2: additionally unfreeze the **prediction heads** of the 3 live
           directions (DLinear heads / VAE predictor / hourly head) + fusion +
           ZS+. Main backbone stays frozen to avoid catastrophic forgetting of
           the zero-shot transfer.

Loss = MAE + adv_loss_weight * adversarial_persistence_loss (reused).

Usage (validation):
    .venv/bin/python scripts/experiments/run_joint_train_native.py
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transcif.calibration.differentiable_zs_plus import DifferentiableZSPlus
from transcif.config import HORIZON, SEQ_LEN, TEST_STRIDE, TRAIN_FRACTION
from transcif.models.zeroshot.fusion import DIRECTION_ORDER, BasisMixFusion
from transcif.models.zeroshot.native import (
    LearnedFusion,
    NativeCausal,
    NativeHier,
    NativeICL,
    NativePhys,
    NativeRAG,
    pad_config_t,
)
from transcif.training.adversarial_loss import adversarial_persistence_loss

# Reuse existing direction training + helpers (no behaviour duplication)
from scripts.experiments.run_joint_train import (
    _build_share_fn,
    _cif_to_share,
    _persistence_cif_full,
    _train_directions,
)
from scripts.experiments._shared import split_origins

DEVICE = None  # CPU/MPS; direction training is device-agnostic here

# Direction roles in the 5-stack (DIRECTION_ORDER = rag, phys, causal, icl, hier)
# Phase 9.6: ALL FIVE directions are now torch-native. ICL's per-query context
# retrieval is no-grad preprocessing (discrete neighbour choice), but the
# transformer forward carries gradient end-to-end.
LIVE_DIRS = ["rag", "phys", "causal", "icl", "hier"]   # all torch-native
FROZEN_DIRS = []                                        # none


# ---------------------------------------------------------------------------
# Predictor construction
# ---------------------------------------------------------------------------

def build_native_predictors(small_regions, target, seed):
    """Train the 5 directions; wrap 4 as live Native* (rag/phys/causal/hier)
    and keep ICL as a numpy predict_fn (frozen constant).

    Returns:
        live: dict direction -> TorchNativePredictor (with the trained model)
        frozen_fns: dict direction -> numpy predict_fn(x_rs, config, ef_r, ef_nr)
        config_dim: int (the unified config width the directions were trained with)
    """
    from transcif.physics.bounds import unify_config_dim
    predictors = _train_directions(small_regions, target, seed=seed)
    config_dim = unify_config_dim(small_regions)
    models = _train_direction_with_models(small_regions, target, seed, config_dim)

    live = {
        "rag": NativeRAG(
            models["rag"]["model"],
            models["rag"]["bank_X"], models["rag"]["bank_Y"], k=5),
        "phys": NativePhys(models["phys"]),
        "causal": NativeCausal(models["causal"]),
        "icl": NativeICL(models["icl"], small_regions, target,
                         n_examples=3, horizon=HORIZON),
        "hier": NativeHier(models["hier"]),
    }
    frozen_fns = {d: predictors[d] for d in FROZEN_DIRS}
    return live, frozen_fns, config_dim


def _train_direction_with_models(regions, target, seed, config_dim):
    """Train all 5 live directions and return their underlying objects.

    Returns dict with nn.Modules for phys/causal/icl/hier and, for rag, a
    sub-dict ``{model, bank_X, bank_Y}`` extracted from the trained bank.
    """
    from transcif.models.zeroshot.rag import train_rag_zero_shot
    from transcif.models.zeroshot.phys_irm import train_phys_irm
    from transcif.models.zeroshot.causal import train_causal_zero_shot
    from transcif.models.zeroshot.icl import train_icl
    from transcif.models.zeroshot.hier import train_hier

    models = {}
    rag_m, bank = train_rag_zero_shot(regions, target, seed=seed, device=DEVICE)
    models["rag"] = {
        "model": rag_m,
        "bank_X": torch.as_tensor(np.asarray(bank.X), dtype=torch.float32),
        "bank_Y": torch.as_tensor(np.asarray(bank.Y), dtype=torch.float32),
    }
    m, _ = train_phys_irm(regions, target, seed=seed, gamma_irm=0.1,
                          lambda_cif=0.5, device=DEVICE)
    models["phys"] = m
    m, _ = train_causal_zero_shot(regions, target, seed=seed, device=DEVICE)
    models["causal"] = m
    models["icl"] = train_icl(regions, target, seed=seed, device=DEVICE)
    m = train_hier(regions, target, seed=seed, device=DEVICE)
    models["hier"] = m
    return models


# ---------------------------------------------------------------------------
# Frozen prediction precompute (RAG / ICL — gradient-detached constants)
# ---------------------------------------------------------------------------

def precompute_frozen(frozen_fns, rs_np, origins, config_np, ef_r, ef_nr):
    """Run the frozen numpy predict_fns on each origin's window.

    Returns:
        dict direction -> torch.FloatTensor (n_origins, HORIZON), detached.
    """
    out = {}
    for d, fn in frozen_fns.items():
        preds = np.zeros((len(origins), HORIZON), dtype=np.float32)
        for i, t0 in enumerate(origins):
            x_win = rs_np[t0 - SEQ_LEN:t0][None, :].astype(np.float32)
            try:
                p = fn(x_win, config_np, ef_r, ef_nr)[0]
            except Exception:
                last = rs_np[t0 - HORIZON:t0]
                p = last * ef_r + (1 - last) * ef_nr
            preds[i] = p
        out[d] = torch.as_tensor(preds, dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# Per-origin 5-stack assembly (live forward + frozen constants)
# ---------------------------------------------------------------------------

def assemble_stack(origin_idx, t0, rs_t, config_t, ef_r, ef_nr,
                   live, frozen_preds):
    """Build a (1, 5, HORIZON) CIF stack in DIRECTION_ORDER.

    Live directions are recomputed (graph-carrying); frozen are looked up.
    """
    x_win = rs_t[t0 - SEQ_LEN:t0].unsqueeze(0)   # (1, SEQ_LEN)
    per_dir = {}
    for d in LIVE_DIRS:
        per_dir[d] = live[d].forward_cif(x_win, config_t, ef_r, ef_nr)  # (1, H)
    for d in FROZEN_DIRS:
        per_dir[d] = frozen_preds[d][origin_idx:origin_idx + 1]  # (1, H) const
    stack = torch.stack([per_dir[d].squeeze(0) for d in DIRECTION_ORDER], dim=0)
    return stack.unsqueeze(0)   # (1, 5, H)


# ---------------------------------------------------------------------------
# Head-parameter selection for Stage 2 unfreezing
# ---------------------------------------------------------------------------

def head_modules(native):
    """Return the prediction-head submodules to unfreeze in Stage 2."""
    m = native.model
    if isinstance(native, NativePhys):
        return [m.linear_trend, m.linear_seasonal, m.config_bias]
    if isinstance(native, NativeCausal):
        return [m.predictor]
    if isinstance(native, NativeHier):
        return [m.hourly_head]
    if isinstance(native, NativeRAG):
        # DLinear heads + the RAG branch (proj + gate) that consume retrieval
        return [m.linear_trend, m.linear_seasonal, m.config_bias,
                m.rag_proj, m.rag_gate]
    if isinstance(native, NativeICL):
        # Projection + prediction head; keep the transformer body frozen
        return [m.input_proj, m.pred_head]
    return []


def set_requires_grad(natives, flag):
    for n in natives.values():
        for p in n.model.parameters():
            p.requires_grad = flag


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

def native_stage(
    name, params, zs_plus, fusion, live, frozen_preds_train,
    rs_t, cif_t, config_t, ef_r, ef_nr,
    train_origins, persistence_train, y_true_train,
    n_steps, lr, margin, adv_loss_weight,
    fusion_kind="learned",
):
    """Run one training stage. Returns metrics dict.

    fusion_kind: "learned" (LearnedFusion, takes config) or "softmax"
    (BasisMixFusion global softmax, for the ablation).
    """
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    metrics = {"stage": name, "train_loss": [], "val_mae": []}
    n_origins = len(train_origins)

    for step in range(n_steps):
        opt.zero_grad()
        total_loss = torch.zeros(())
        for o_idx, t0 in enumerate(train_origins):
            stack = assemble_stack(o_idx, t0, rs_t, config_t, ef_r, ef_nr,
                                   live, frozen_preds_train)
            if fusion_kind == "learned":
                fused_cif = fusion(stack, config_t).squeeze(0)
            else:
                fused_cif = fusion(stack).squeeze(0)
            share = _cif_to_share(fused_cif, ef_r, ef_nr).clamp(0.0, 1.0)
            share_fn = _build_share_fn(share)
            pred = zs_plus(rs_t, cif_t, ef_r, ef_nr, [t0], share_fn)
            target_cif = y_true_train[o_idx]
            mae_loss = (pred.squeeze(0) - target_cif).abs().mean()
            adv_loss = adversarial_persistence_loss(
                pred.squeeze(0), persistence_train[o_idx], margin=margin)
            total_loss = total_loss + mae_loss + adv_loss_weight * adv_loss
        total_loss = total_loss / max(1, n_origins)
        total_loss.backward()
        opt.step()
        metrics["train_loss"].append(float(total_loss.item()))

        with torch.no_grad():
            val_maes = []
            for o_idx, t0 in enumerate(train_origins):
                stack = assemble_stack(o_idx, t0, rs_t, config_t, ef_r, ef_nr,
                                       live, frozen_preds_train)
                fused_cif = (fusion(stack, config_t) if fusion_kind == "learned"
                             else fusion(stack)).squeeze(0)
                share = _cif_to_share(fused_cif, ef_r, ef_nr).clamp(0.0, 1.0)
                share_fn = _build_share_fn(share)
                pred = zs_plus(rs_t, cif_t, ef_r, ef_nr, [t0], share_fn)
                val_maes.append(float((pred.squeeze(0) - y_true_train[o_idx]).abs().mean()))
            metrics["val_mae"].append(float(np.mean(val_maes)))
    return metrics


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_held_out(zs_plus, fusion, live, frozen_preds_eval, rs_t, cif_t,
                  config_t, ef_r, ef_nr, eval_origins, y_true_eval,
                  fusion_kind="learned"):
    """Per-origin held-out MAE. Returns (per_origin_mae list, mean)."""
    per_origin = []
    for o_idx, t0 in enumerate(eval_origins):
        stack = assemble_stack(o_idx, t0, rs_t, config_t, ef_r, ef_nr,
                               live, frozen_preds_eval)
        fused_cif = (fusion(stack, config_t) if fusion_kind == "learned"
                     else fusion(stack)).squeeze(0)
        share = _cif_to_share(fused_cif, ef_r, ef_nr).clamp(0.0, 1.0)
        share_fn = _build_share_fn(share)
        pred = zs_plus(rs_t, cif_t, ef_r, ef_nr, [t0], share_fn)
        per_origin.append(float((pred.squeeze(0) - y_true_eval[o_idx]).abs().mean()))
    return per_origin, float(np.mean(per_origin))


# ---------------------------------------------------------------------------
# Snapshot / restore for the internal-val gate
# ---------------------------------------------------------------------------

def _snapshot(zs_plus, fusion, live):
    """Deep-copy state of the modules that change across stages."""
    snap = {
        "zs_plus": {k: v.detach().clone() for k, v in zs_plus.state_dict().items()},
        "fusion": {k: v.detach().clone() for k, v in fusion.state_dict().items()},
    }
    for name, n in live.items():
        snap[name] = {k: v.detach().clone() for k, v in n.model.state_dict().items()}
    return snap


def _restore(snap, zs_plus, fusion, live):
    zs_plus.load_state_dict(snap["zs_plus"])
    fusion.load_state_dict(snap["fusion"])
    for name, n in live.items():
        n.model.load_state_dict(snap[name])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_native_joint_train(
    small_regions, target, seed=0,
    n_train=12, n_eval=12,
    n_steps_s1=30, n_steps_s2=30,
    lr_s1=5e-3, lr_s2=1e-3,
    margin=0.10, adv_loss_weight=0.5,
    fusion_kind="learned",
    gate=None, gate_eps=2.0, n_inner_val=3,
):
    """Train + eval one (target, seed) with the native pipeline.

    fusion_kind: "learned" (LearnedFusion) or "softmax" (BasisMixFusion ablation).
    gate: None (always keep Stage 2) or "internal_val" — split the training
        origins into (n_train - n_inner_val) inner-train + n_inner_val inner-val,
        train both stages on inner-train, evaluate each on inner-val, and keep
        Stage 2 only if it beats Stage 1 on inner-val by ``gate_eps`` MAE;
        otherwise revert to the Stage 1 (frozen-heads) config. This adaptively
        detects head-finetune overfitting on easy grids without peeking at the
        held-out eval origins.
    Returns dict with stage MAEs + held-out per-origin MAEs + gate decision.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = small_regions[target]
    rs_np = data["rs"].astype(np.float32)
    cif_np = data["cif"].astype(np.float32)
    ef_r, ef_nr = float(data["ef_r"]), float(data["ef_nr"])
    rs_t = torch.as_tensor(rs_np, dtype=torch.float32)
    cif_t = torch.as_tensor(cif_np, dtype=torch.float32)

    train_origins, eval_origins = split_origins(rs_np, n_train=n_train, n_eval=n_eval)

    # Internal-val gate: carve ``n_inner_val`` origins from AFTER the eval split
    # (disjoint, never trained on) to decide whether Stage 2 finetuning helps.
    # Keeping eval origins IDENTICAL to the no-gate run makes the comparison
    # apples-to-apples. Stage 1/2 train on the full train_origins; only the
    # gate decision uses inner_val. The gate is CONSERVATIVE: it keeps Stage 2
    # by default and reverts to Stage 1 only when Stage 2 is clearly WORSE on
    # inner-val (by gate_eps), so it preserves the hard-grid gains and only
    # catches clear easy-grid overfit.
    if gate == "internal_val":
        split_pt = int(len(rs_np) * TRAIN_FRACTION)
        candidates = [split_pt + st for st in
                      range(0, len(rs_np) - split_pt - HORIZON + 1, TEST_STRIDE)]
        val_start = n_train + n_eval
        if len(candidates) >= val_start + n_inner_val:
            inner_val = candidates[val_start:val_start + n_inner_val]
        else:
            inner_val = None
            gate = None
    else:
        inner_val = None

    # Stage 1/2 train on the FULL train_origins (the inner-val, when present, is
    # disjoint and only used for the gate decision).
    inner_train = train_origins

    # config: use the 2-D legacy config so it matches the committed fused_five
    # protocol (and avoids the in-progress 12-D config mismatch). Pad to the
    # directions' config_dim.
    config_2d = np.asarray(data["config"], dtype=np.float32)[:2]
    live, frozen_fns, config_dim = build_native_predictors(small_regions, target, seed)
    config_np = np.pad(config_2d, (0, max(0, config_dim - 2)), mode="constant").astype(np.float32)
    config_t = torch.as_tensor(config_np[None, :], dtype=torch.float32)  # (1, config_dim)

    # frozen preds on the relevant origin subsets
    frozen_train = precompute_frozen(frozen_fns, rs_np, inner_train, config_np, ef_r, ef_nr)
    frozen_eval = precompute_frozen(frozen_fns, rs_np, eval_origins, config_np, ef_r, ef_nr)
    frozen_val = (precompute_frozen(frozen_fns, rs_np, inner_val, config_np, ef_r, ef_nr)
                  if gate else None)

    y_true_train = torch.stack([cif_t[o:o + HORIZON] for o in inner_train])
    y_true_eval = torch.stack([cif_t[o:o + HORIZON] for o in eval_origins])
    persistence_train = _persistence_cif_full(rs_np, cif_np, inner_train)
    y_true_val = (torch.stack([cif_t[o:o + HORIZON] for o in inner_val])
                  if gate else None)

    for n in live.values():
        n.model.eval()

    zs_plus = DifferentiableZSPlus()
    if fusion_kind == "learned":
        fusion = LearnedFusion(n_directions=5, config_dim=config_dim, horizon=HORIZON)
    else:
        fusion = BasisMixFusion()

    summary = {"fusion_kind": fusion_kind, "config_dim": config_dim, "gate": gate}

    # Stage 1: directions frozen, train fusion + ZS+
    set_requires_grad(live, False)
    params1 = list(zs_plus.parameters()) + list(fusion.parameters())
    m1 = native_stage("stage1", params1, zs_plus, fusion, live, frozen_train,
                      rs_t, cif_t, config_t, ef_r, ef_nr,
                      inner_train, persistence_train, y_true_train,
                      n_steps_s1, lr_s1, margin, adv_loss_weight, fusion_kind)
    summary["stage1_train_mae"] = m1["val_mae"][-1]

    # Gate: measure Stage 1 on the internal-val split, then snapshot.
    if gate:
        _, s1_val_mae = eval_held_out(
            zs_plus, fusion, live, frozen_val, rs_t, cif_t, config_t, ef_r, ef_nr,
            inner_val, y_true_val, fusion_kind)
        summary["stage1_inner_val_mae"] = s1_val_mae
        snap = _snapshot(zs_plus, fusion, live)

    # Stage 2: unfreeze live heads + fusion + ZS+
    head_params = []
    for n in live.values():
        for mod in head_modules(n):
            for p in mod.parameters():
                p.requires_grad = True
            head_params.extend([p for p in mod.parameters()])
    params2 = (list(zs_plus.parameters()) + list(fusion.parameters())
               + head_params)
    m2 = native_stage("stage2", params2, zs_plus, fusion, live, frozen_train,
                      rs_t, cif_t, config_t, ef_r, ef_nr,
                      inner_train, persistence_train, y_true_train,
                      n_steps_s2, lr_s2, margin, adv_loss_weight, fusion_kind)
    summary["stage2_train_mae"] = m2["val_mae"][-1]

    # Gate decision: keep Stage 2 only if it beats Stage 1 on inner-val.
    if gate:
        _, s2_val_mae = eval_held_out(
            zs_plus, fusion, live, frozen_val, rs_t, cif_t, config_t, ef_r, ef_nr,
            inner_val, y_true_val, fusion_kind)
        summary["stage2_inner_val_mae"] = s2_val_mae
        # Conservative gate: KEEP Stage 2 unless it is clearly WORSE on inner-val
        # (s2 > s1 + gate_eps). Defaulting to keep preserves the hard-grid gains;
        # reverting only on strong overfit evidence catches easy-grid regressions.
        if s2_val_mae > s1_val_mae + gate_eps:
            _restore(snap, zs_plus, fusion, live)
            summary["gate_decision"] = "reverted_to_stage1"
        else:
            summary["gate_decision"] = "stage2_kept"

    per_origin, held_out = eval_held_out(
        zs_plus, fusion, live, frozen_eval, rs_t, cif_t, config_t, ef_r, ef_nr,
        eval_origins, y_true_eval, fusion_kind)
    summary["held_out_mae"] = held_out
    summary["held_out_per_origin"] = per_origin
    summary["n_train_origins"] = len(train_origins)
    summary["n_inner_train_origins"] = len(inner_train)
    summary["n_eval_origins"] = len(eval_origins)
    return summary


# ---------------------------------------------------------------------------
# Validation main: 2 regions × {native-learned, native-softmax} vs baseline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Native joint train validation (Phase 9)")
    ap.add_argument("--regions", nargs="+", default=["QLD1", "UK_08_West_Midlands"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/native_validation.json")
    args = ap.parse_args()

    from transcif.data.loaders import all_region_configs, load_region_data

    print("[LOAD] regions...", flush=True)
    all_configs = all_region_configs()
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}

    # baseline numbers from the existing frozen-proxy joint train (read if present)
    baseline_path = Path("results/joint_train_full.json")
    baseline = {}
    if baseline_path.exists():
        for r in json.loads(baseline_path.read_text()):
            if r.get("target") in args.regions and r.get("seed") == args.seed \
                    and "held_out_mae" in r:
                baseline[r["target"]] = r["held_out_mae"]

    results = {"baseline_frozen_proxy": baseline, "by_region": {}}

    for target in args.regions:
        src_names = [n for n in all_regions if n != target][:3]
        # 2-D config (match committed protocol)
        small_regions = {}
        for n in [target] + src_names:
            rd = dict(all_regions[n])
            rd["config"] = np.asarray(rd["config"], dtype=np.float32)[:2]
            small_regions[n] = rd

        results["by_region"][target] = {}
        for kind in ["learned", "softmax"]:
            print(f"\n=== {target} seed{args.seed} fusion={kind} ===", flush=True)
            t0 = time.time()
            try:
                summ = run_native_joint_train(
                    small_regions, target, seed=args.seed, fusion_kind=kind)
                summ["elapsed_seconds"] = time.time() - t0
                results["by_region"][target][f"native_{kind}"] = summ
                print(f"  held_out_mae={summ['held_out_mae']:.2f}  "
                      f"s1={summ['stage1_train_mae']:.2f} s2={summ['stage2_train_mae']:.2f}  "
                      f"({summ['elapsed_seconds']:.0f}s)", flush=True)
            except Exception as e:
                print(f"  [ERROR] {target}/{kind}: {e}", flush=True)
                traceback.print_exc()
                results["by_region"][target][f"native_{kind}"] = {"error": str(e)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[WRITE] {out_path}")

    # verdict
    print("\n=== VERDICT (native_learned vs baseline frozen-proxy) ===")
    for target in args.regions:
        b = baseline.get(target)
        n = results["by_region"][target].get("native_learned", {}).get("held_out_mae")
        if b is not None and n is not None:
            d = b - n
            flag = "✓ better" if d >= 1.0 else ("~ flat" if abs(d) < 1.0 else "✗ worse")
            print(f"  {target:28s} baseline={b:.2f}  native={n:.2f}  Δ={d:+.2f}  [{flag}]")


if __name__ == "__main__":
    main()
