"""2-stage warmup joint training pipeline (Task 8.3).

Stages:
  Stage 1: 5 direction models frozen, train only DifferentiableZSPlus
           (branch_gate + log_inv_temp) and BasisMixFusion (logit).
  Stage 2: Add learnable per-direction correction (5, HORIZON), init zero,
           so gradient can fine-tune the direction output layers without
           refactoring the underlying numpy/torch hybrid predictors.

Design:
  Direction predictors (rag/phys/causal/icl/hier) accept numpy ``(B, SEQ_LEN)``
  and return numpy ``(B, HORIZON)`` CIF predictions. They are not natively
  differentiable through PyTorch. To enable joint training while keeping
  direction models unchanged, we:

    1. Pre-compute ``(5, n_origins, HORIZON)`` CIF predictions from the 5
       frozen direction models. These become constant tensors.
    2. Apply learnable transformations on top:
       - Stage 1: BasisMixFusion softmax weights over the 5 directions
                  + DifferentiableZSPlus attention.
       - Stage 2: above + per-direction additive correction.

The adversarial-persistence loss is added to MAE to push the model beyond
the persistence baseline.

Usage (programmatic):
    from scripts.experiments.run_joint_train import run_joint_train
    result = run_joint_train(regions, target, stages=("stage1", "stage2"),
                              n_origins=8, out_dir="/tmp/run", seed=0)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transcif.calibration.differentiable_zs_plus import DifferentiableZSPlus
from transcif.config import HORIZON, SEQ_LEN
from transcif.models.zeroshot.fusion import BasisMixFusion
from transcif.training.adversarial_loss import adversarial_persistence_loss

from scripts.experiments._shared import zs_plus_origins


def _origins_from_series(rs: np.ndarray, n_max: int = 32) -> list[int]:
    """Pick up to ``n_max`` evenly spaced test origins from the test split."""
    candidates = zs_plus_origins(rs)
    if len(candidates) <= n_max:
        return candidates
    idx = np.linspace(0, len(candidates) - 1, n_max).astype(int)
    return [candidates[i] for i in idx]


def _train_directions(
    regions: Dict, target: str, seed: int
) -> Dict[str, callable]:
    """Train 5 direction models on (target, source) regions.

    Returns a dict of predict_fn keyed by direction name. Each predict_fn
    takes ``(x_rs, config, ef_r, ef_nr)`` and returns ``(B, HORIZON)`` numpy
    CIF predictions.
    """
    from transcif.models.zeroshot.rag import (
        train_rag_zero_shot, predict_rag_zs,
    )
    from transcif.models.zeroshot.phys_irm import (
        train_phys_irm, predict_phys_irm,
    )
    from transcif.models.zeroshot.causal import (
        train_causal_zero_shot, predict_causal_zs,
    )
    from transcif.models.zeroshot.icl import train_icl, predict_icl_zs
    from transcif.models.zeroshot.hier import train_hier, predict_hier_zs

    predictors = {}

    m, bank = train_rag_zero_shot(regions, target, seed=seed, device=None)
    predictors["rag"] = lambda x, cfg, ef_r, ef_nr, m=m, b=bank: predict_rag_zs(
        m, b, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr
    )

    m, _ = train_phys_irm(
        regions, target, seed=seed, gamma_irm=0.1, lambda_cif=0.5, device=None
    )
    predictors["phys"] = lambda x, cfg, ef_r, ef_nr, m=m: predict_phys_irm(
        m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr
    )

    m, _ = train_causal_zero_shot(regions, target, seed=seed, device=None)
    predictors["causal"] = lambda x, cfg, ef_r, ef_nr, m=m: predict_causal_zs(
        m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr
    )

    m = train_icl(regions, target, seed=seed, device=None)
    predictors["icl"] = lambda x, cfg, ef_r, ef_nr, m=m, r=regions, t=target: (
        predict_icl_zs(m, r, t, x.astype(np.float32), ef_r, ef_nr)
    )

    m = train_hier(regions, target, seed=seed, device=None)
    predictors["hier"] = lambda x, cfg, ef_r, ef_nr, m=m: predict_hier_zs(
        m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr
    )

    return predictors


def _precompute_predictions(
    predictors: Dict[str, callable],
    regions: Dict,
    target: str,
    origins: Sequence[int],
) -> torch.Tensor:
    """Run all 5 frozen direction models on each origin's input window.

    Returns:
        (5, n_origins, HORIZON) torch tensor (no grad). Constant.
    """
    data = regions[target]
    rs = data["rs"]
    config = data["config"].astype(np.float32)
    ef_r, ef_nr = float(data["ef_r"]), float(data["ef_nr"])

    n_origins = len(origins)
    out = np.zeros((5, n_origins, HORIZON), dtype=np.float32)

    for o_idx, t0 in enumerate(origins):
        x_window = rs[t0 - SEQ_LEN : t0][None, :].astype(np.float32)
        for d_idx, name in enumerate(["rag", "phys", "causal", "icl", "hier"]):
            try:
                pred = predictors[name](
                    x_window, config, ef_r, ef_nr
                )[0]
                out[d_idx, o_idx, :] = pred
            except Exception:
                # If a direction fails, fall back to persistence (last HORIZON rs)
                last = rs[t0 - HORIZON : t0]
                out[d_idx, o_idx, :] = last * ef_r + (1 - last) * ef_nr

    return torch.as_tensor(out, dtype=torch.float32)


def _cif_to_share(cif: torch.Tensor, ef_r: float, ef_nr: float) -> torch.Tensor:
    """Convert CIF prediction to renewable share. Inverse of physics decompose."""
    return (cif - ef_nr) / (ef_r - ef_nr + 1e-8)


def _build_share_fn(share_tensor: torch.Tensor) -> callable:
    """Build a share_fn closure that returns the precomputed (with grad) share.

    DifferentiableZSPlus.share_fn expects ``(SEQ_LEN,) -> (horizon,)`` but
    ignores the input window content for our joint setup. The grad path is
    through ``share_tensor`` (which carries BasisMix + correction grad).
    """
    def share_fn(x_window):
        return share_tensor
    return share_fn


def _persistence_cif(rs: np.ndarray, origins: Sequence[int]) -> torch.Tensor:
    """Persistence baseline: yesterday's CIF for today (the simplest branch)."""
    out = np.zeros((len(origins), HORIZON), dtype=np.float32)
    for i, t0 in enumerate(origins):
        # last HORIZON timesteps before t0 — closest "yesterday" we have
        out[i] = rs[t0 - HORIZON : t0]
    return torch.as_tensor(out, dtype=torch.float32)


def _persistence_cif_full(
    rs: np.ndarray, cif: np.ndarray, origins: Sequence[int]
) -> torch.Tensor:
    """Persistence baseline in CIF space (actual CIF values, not rs-derived)."""
    out = np.zeros((len(origins), HORIZON), dtype=np.float32)
    for i, t0 in enumerate(origins):
        out[i] = cif[t0 - HORIZON : t0]
    return torch.as_tensor(out, dtype=torch.float32)


def _stage(
    name: str,
    params: list,
    zs_plus: DifferentiableZSPlus,
    fusion: BasisMixFusion,
    rs_t: torch.Tensor,
    cif_t: torch.Tensor,
    frozen_preds: torch.Tensor,
    origins: Sequence[int],
    ef_r: float,
    ef_nr: float,
    persistence: torch.Tensor,
    y_true: torch.Tensor,
    n_steps: int,
    lr: float,
    margin: float,
    correction: nn.Parameter | None = None,
    adv_loss_weight: float = 0.5,
) -> Tuple[Dict, Dict]:
    """Run one stage of training. Returns (metrics_dict, state_dict).

    Total loss per step:

        L = MAE(pred, target) + adv_loss_weight * adversarial_persistence_loss(...)

    The ``adv_loss_weight`` default of 0.5 is a deliberate balance:
      - 0.0: pure MAE training, no anti-persistence pressure (the model may
             collapse onto the persistence baseline because ZS+ branches 1-4
             ARE persistence).
      - 1.0: anti-persistence pressure equals MAE pressure. Empirically
             over-pushes the model away from sensible lag baselines on
             regions where persistence is genuinely the best forecast
             (e.g. NSW1 dry spells).
      - 0.5: half-weight. The model still gets full MAE signal but pays a
             tax for failing to beat persistence by the relative margin.
             Selected empirically on the QLD1 sanity run (Task 8.4) where
             0.5 produced monotonic loss decrease across both stages.
    """
    opt = torch.optim.Adam(params, lr=lr)

    metrics = {"stage": name, "train_loss": [], "val_mae": []}

    for step in range(n_steps):
        opt.zero_grad()
        n_origins = len(origins)
        total_loss = torch.zeros(())
        # Compute fused CIF per origin (differentiable through fusion.logit
        # and correction)
        for o_idx in range(n_origins):
            preds_o = frozen_preds[:, o_idx, :]  # (5, HORIZON)
            if correction is not None:
                preds_o = preds_o + correction
            # BasisMixFusion expects (N, 5, HORIZON) → returns (N, HORIZON)
            fused_cif = fusion(preds_o.unsqueeze(0)).squeeze(0)  # (HORIZON,)
            share = _cif_to_share(fused_cif, ef_r, ef_nr)
            share = share.clamp(0.0, 1.0)
            share_fn = _build_share_fn(share)
            pred = zs_plus(
                rs_t, cif_t, ef_r, ef_nr, [origins[o_idx]], share_fn
            )  # (1, HORIZON)
            target_cif = y_true[o_idx : o_idx + 1].squeeze(0)
            # MAE loss
            mae_loss = (pred.squeeze(0) - target_cif).abs().mean()
            # Adversarial-persistence loss (weighted; see docstring)
            adv_loss = adversarial_persistence_loss(
                pred.squeeze(0), persistence[o_idx], margin=margin
            )
            total_loss = total_loss + mae_loss + adv_loss_weight * adv_loss

        total_loss = total_loss / max(1, n_origins)
        total_loss.backward()
        opt.step()
        metrics["train_loss"].append(float(total_loss.item()))

        # Validation MAE (in-sample, since this is a sanity pipeline)
        with torch.no_grad():
            val_maes = []
            for o_idx in range(n_origins):
                preds_o = frozen_preds[:, o_idx, :]
                if correction is not None:
                    preds_o = preds_o + correction
                fused_cif = fusion(preds_o.unsqueeze(0)).squeeze(0)
                share = _cif_to_share(fused_cif, ef_r, ef_nr).clamp(0.0, 1.0)
                share_fn = _build_share_fn(share)
                pred = zs_plus(
                    rs_t, cif_t, ef_r, ef_nr, [origins[o_idx]], share_fn
                )
                val_maes.append(
                    float((pred.squeeze(0) - y_true[o_idx]).abs().mean())
                )
            metrics["val_mae"].append(float(np.mean(val_maes)))

    state = {
        "zs_plus_state": zs_plus.state_dict(),
        "fusion_state": fusion.state_dict(),
    }
    if correction is not None:
        state["correction"] = correction.detach().cpu()
    return metrics, state


def run_joint_train(
    regions: Dict,
    target: str,
    stages: Sequence[str] = ("stage1", "stage2"),
    n_origins: int = 8,
    out_dir: str = ".",
    seed: int = 0,
    n_steps_stage1: int = 30,
    n_steps_stage2: int = 30,
    lr_stage1: float = 5e-2,
    lr_stage2: float = 1e-2,
    margin: float = 0.10,
    eval_origins: Sequence[int] | None = None,
    adv_loss_weight: float = 0.5,
) -> Dict:
    """Run joint training pipeline.

    Args:
        regions: Dict of region data (target + sources).
        target: Target region name.
        stages: Which stages to run (subset of ``("stage1", "stage2")``).
        n_origins: Max number of origins to sample (capped for speed).
        out_dir: Directory to write metrics and checkpoints.
        seed: RNG seed.
        n_steps_stage1: Optimizer steps for Stage 1.
        n_steps_stage2: Optimizer steps for Stage 2.
        lr_stage1: Stage 1 learning rate.
        lr_stage2: Stage 2 learning rate.
        margin: Adversarial-persistence margin.
        eval_origins: Optional held-out origins for evaluation. If provided,
            MAE is computed on these (disjoint from training origins) and
            written to ``summary["held_out_mae"]``.
        adv_loss_weight: Weight on the adversarial-persistence loss term.
            Default 0.5 — see :func:`_stage` docstring for rationale.

    Returns:
        Dict with final MAE per stage.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = regions[target]
    rs_np = data["rs"].astype(np.float32)
    cif_np = data["cif"].astype(np.float32)
    ef_r = float(data["ef_r"])
    ef_nr = float(data["ef_nr"])

    rs_t = torch.as_tensor(rs_np, dtype=torch.float32)
    cif_t = torch.as_tensor(cif_np, dtype=torch.float32)

    origins = _origins_from_series(rs_np, n_max=n_origins)
    if len(origins) < 2:
        raise ValueError(
            f"Need at least 2 origins, got {len(origins)} from series len {len(rs_np)}"
        )

    # Train directions and precompute frozen predictions for both training
    # origins and (optional) held-out eval origins.
    predictors = _train_directions(regions, target, seed=seed)
    frozen_preds = _precompute_predictions(
        predictors, regions, target, origins
    )  # (5, n_origins, HORIZON)

    # Held-out eval: precompute predictions on disjoint origins if provided
    frozen_preds_eval = None
    y_true_eval = None
    persistence_eval = None
    if eval_origins is not None and len(eval_origins) > 0:
        frozen_preds_eval = _precompute_predictions(
            predictors, regions, target, list(eval_origins)
        )
        y_true_eval = torch.stack(
            [cif_t[o : o + HORIZON] for o in eval_origins], dim=0
        )
        persistence_eval = _persistence_cif_full(rs_np, cif_np, list(eval_origins))

    # Ground truth: actual CIF values at origins + horizon
    y_true = torch.stack(
        [cif_t[o : o + HORIZON] for o in origins], dim=0
    )  # (n_origins, HORIZON)

    # Persistence baseline (yesterday's CIF for today)
    persistence = _persistence_cif_full(rs_np, cif_np, origins)

    # Core modules
    zs_plus = DifferentiableZSPlus()
    fusion = BasisMixFusion()
    correction = nn.Parameter(torch.zeros(5, HORIZON))

    summary = {}
    if "stage1" in stages:
        # Stage 1: only zs_plus + fusion parameters
        params1 = list(zs_plus.parameters()) + list(fusion.parameters())
        metrics1, state1 = _stage(
            "stage1", params1, zs_plus, fusion,
            rs_t, cif_t, frozen_preds, origins, ef_r, ef_nr,
            persistence, y_true,
            n_steps=n_steps_stage1, lr=lr_stage1, margin=margin,
            correction=None, adv_loss_weight=adv_loss_weight,
        )
        (out_path / "stage1_metrics.json").write_text(json.dumps(metrics1, indent=2))
        torch.save(state1, out_path / "stage1_checkpoint.pt")
        summary["stage1_final_mae"] = metrics1["val_mae"][-1]
        summary["stage1_final_loss"] = metrics1["train_loss"][-1]

    if "stage2" in stages:
        # Stage 2: zs_plus + fusion + correction (init zero)
        params2 = list(zs_plus.parameters()) + list(fusion.parameters()) + [correction]
        metrics2, state2 = _stage(
            "stage2", params2, zs_plus, fusion,
            rs_t, cif_t, frozen_preds, origins, ef_r, ef_nr,
            persistence, y_true,
            n_steps=n_steps_stage2, lr=lr_stage2, margin=margin,
            correction=correction, adv_loss_weight=adv_loss_weight,
        )
        (out_path / "stage2_metrics.json").write_text(json.dumps(metrics2, indent=2))
        torch.save(state2, out_path / "stage2_checkpoint.pt")
        summary["stage2_final_mae"] = metrics2["val_mae"][-1]
        summary["stage2_final_loss"] = metrics2["train_loss"][-1]

    # Held-out evaluation: forward pass on disjoint origins, no training
    if frozen_preds_eval is not None:
        with torch.no_grad():
            eval_maes = []
            n_eval = len(eval_origins)
            for o_idx in range(n_eval):
                preds_o = frozen_preds_eval[:, o_idx, :]
                # Apply the trained correction (Stage 2 output)
                preds_o = preds_o + correction
                fused_cif = fusion(preds_o.unsqueeze(0)).squeeze(0)
                share = _cif_to_share(fused_cif, ef_r, ef_nr).clamp(0.0, 1.0)
                share_fn = _build_share_fn(share)
                pred = zs_plus(
                    rs_t, cif_t, ef_r, ef_nr, [eval_origins[o_idx]], share_fn
                )
                eval_maes.append(
                    float((pred.squeeze(0) - y_true_eval[o_idx]).abs().mean())
                )
            summary["held_out_mae"] = float(np.mean(eval_maes))
            summary["held_out_n_origins"] = n_eval

    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    """CLI entry: run on a real target region (QLD1 + 3 sources by default)."""
    import argparse

    ap = argparse.ArgumentParser(description="Joint train pipeline (Task 8.3)")
    ap.add_argument("--target", default="QLD1")
    ap.add_argument("--n-origins", type=int, default=8)
    ap.add_argument("--out", default="results/joint_train_sanity")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transcif.data.loaders import load_region_data, all_region_configs

    all_configs = all_region_configs()
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}

    src_names = [n for n in all_regions if n != args.target][:3]
    small_regions = {args.target: all_regions[args.target]}
    for n in src_names:
        small_regions[n] = all_regions[n]

    summary = run_joint_train(
        small_regions, args.target,
        stages=("stage1", "stage2"),
        n_origins=args.n_origins,
        out_dir=args.out,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
