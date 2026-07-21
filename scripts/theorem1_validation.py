"""Numerically verifies Theorem 1's exact CI-transfer-error decomposition on real SA1
2023 data, using the already-trained "全部组合" baseline and "+D+E" models from
sa1_domain_adaptation.py:

    CI_pred,t - CI_true,t = (s_hat_t - s_true_t) * (C_renew - C_nonrenew) + (delta_hat_t - epsilon_t)

where epsilon_t := CI_true,t - CIF(s_true_t) is the true residual (real measured CI minus
the physics formula evaluated at the TRUE renewable share -- computable directly from real
data, no model involved), and L_T := |C_renew - C_nonrenew| is the region's exact,
table-derived Lipschitz constant (Corollary 1).

Checks:
  1. The identity holds exactly (up to float precision) on every eval sample -- this is an
     algebraic guarantee, not a hypothesis, so this is a correctness sanity check on the
     derivation and this script, not a test of the model.
  2. Whether Term1 (L_T * |s_hat - s_true|, transfer-error amplification) or Term2
     (|delta_hat - epsilon|, residual-estimation error) dominates SA1's total error, for
     both the baseline and the +D+E variant -- this is the actual empirical question
     Corollary 1 raises.

Item 2 robustness extension (rolling-origin): the split between SA1's calibration and eval
windows was originally a single fixed CALIB_FRACTION=0.7 cut. Since Corollary 1's Term1-
dominance finding is an empirical claim about SA1's error decomposition, not an algebraic
guarantee, it could in principle be an artifact of that one arbitrary split point. `decompose`
now accepts an explicit `calib_fraction` and the `__main__` block sweeps it across several
rolling-origin points, reporting mean +/- std of each metric and whether `dominant_term`
stays constant across splits per variant.

Run with: PYTHONPATH=src python scripts/theorem1_validation.py
"""

import os
import re

import numpy as np
import torch

from transcif.physics.cif import get_emission_factors
from transcif.physics.residual import ResidualCorrectionHead, fit_residual_head
from transcif.calibration.dominant_reweight import recompute_dominant_variable, reweight_lt_mwkc_alpha
from transcif.training.domain_adaptation import fine_tune_on_calibration, train_multi_source_mldg_coral
from transcif.training.train_multi_source import train_multi_source_mldg

from sa1_ablation import CALIB_FRACTION, HORIZON, REGION_TO_FACTOR_CODE, TARGET_REGION, load_source_and_target
from sa1_domain_adaptation import (
    CORAL_WEIGHT,
    FINE_TUNE_EPOCHS_PER_STAGE,
    FINE_TUNE_LR,
    INCLUDE_GENERATION,
    INCLUDE_TEMPERATURE,
    MLDG_EPOCHS,
    build_model,
)

# 3 rolling-origin split points around the original 0.7 cut, spaced 0.1 apart so each shifts
# the calibration/eval boundary by a full window's worth of hours. Kept to 3 (the low end of
# the "3-5 points" range) given each point requires a full ~20-30 min MLDG training run per
# variant and this session has already hit the background-task kill pattern repeatedly.
ROLLING_ORIGIN_SPLITS = [0.6, 0.7, 0.8]


def checkpoint_path_for(name: str, stage: str, calib_fraction: float) -> str:
    """Filename-safe checkpoint path unique per (variant, training stage, split point) so the
    rolling-origin sweep's repeated MLDG/CORAL training runs can each resume independently if
    a background run is killed mid-training, without colliding with each other's state."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    frac_slug = str(round(calib_fraction, 2)).replace(".", "")
    return f"/tmp/transcif_ckpt_theorem1_{slug}_{stage}_{frac_slug}.pt"


def decompose(name: str, use_fine_tune: bool, use_coral: bool, calib_fraction: float = CALIB_FRACTION) -> dict:
    source_windows, x_target, y_target_share, ci_true_target = load_source_and_target(
        INCLUDE_GENERATION, INCLUDE_TEMPERATURE
    )

    n = x_target.shape[0]
    split = int(n * calib_fraction)
    x_calib, x_eval = x_target[:split], x_target[split:]
    y_calib_share, y_eval_share = y_target_share[:split], y_target_share[split:]
    ci_true_calib, ci_true_eval = ci_true_target[:split], ci_true_target[split:]

    model = build_model()

    if use_coral:
        train_multi_source_mldg_coral(
            model,
            source_windows,
            x_calib,
            epochs=MLDG_EPOCHS,
            coral_weight=CORAL_WEIGHT,
            checkpoint_path=checkpoint_path_for(name, "coral", calib_fraction),
        )
    else:
        train_multi_source_mldg(
            model,
            source_windows,
            epochs=MLDG_EPOCHS,
            checkpoint_path=checkpoint_path_for(name, "mldg", calib_fraction),
        )

    if use_fine_tune:
        fine_tune_on_calibration(
            model, x_calib, y_calib_share, epochs_per_stage=FINE_TUNE_EPOCHS_PER_STAGE, lr=FINE_TUNE_LR
        )

    renew_factor, nonrenew_factor = get_emission_factors(REGION_TO_FACTOR_CODE[TARGET_REGION])
    L_T = abs(renew_factor - nonrenew_factor)

    with torch.no_grad():
        s_hat_calib, _ = model(x_calib)
    dominant_idx = recompute_dominant_variable(model, x_calib)
    reweight_lt_mwkc_alpha(model, dominant_idx)

    with torch.no_grad():
        s_hat_calib, _ = model(x_calib)
        s_hat_eval, _ = model(x_eval)

    s_hat_calib_np = s_hat_calib.numpy()
    s_hat_eval_np = s_hat_eval.numpy()

    ci_pred_physics_calib = s_hat_calib_np * renew_factor + (1 - s_hat_calib_np) * nonrenew_factor
    ci_pred_physics_eval = s_hat_eval_np * renew_factor + (1 - s_hat_eval_np) * nonrenew_factor

    residual_head = ResidualCorrectionHead(input_dim=1, hidden_dim=8)
    calib_features = torch.tensor(s_hat_calib_np.reshape(-1, 1), dtype=torch.float32)
    calib_targets = torch.tensor((ci_true_calib - ci_pred_physics_calib).reshape(-1), dtype=torch.float32)
    fit_residual_head(residual_head, calib_features, calib_targets, epochs=100, lr=1e-2)

    eval_features = torch.tensor(s_hat_eval_np.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        delta_hat_eval = residual_head(eval_features).numpy().reshape(ci_pred_physics_eval.shape)

    ci_pred_eval = ci_pred_physics_eval + delta_hat_eval

    s_true_eval = y_eval_share.numpy()[:, :HORIZON]
    ci_true_physics_at_true_share = s_true_eval * renew_factor + (1 - s_true_eval) * nonrenew_factor
    epsilon_eval = ci_true_eval - ci_true_physics_at_true_share

    lhs = ci_pred_eval - ci_true_eval
    term1 = (s_hat_eval_np - s_true_eval) * (renew_factor - nonrenew_factor)
    term2 = delta_hat_eval - epsilon_eval
    rhs = term1 + term2

    identity_max_abs_gap = np.max(np.abs(lhs - rhs))
    mean_abs_term1 = np.mean(np.abs(term1))
    mean_abs_term2 = np.mean(np.abs(term2))
    mean_abs_total = np.mean(np.abs(lhs))

    return {
        "name": name,
        "calib_fraction": calib_fraction,
        "L_T": L_T,
        "identity_max_abs_gap": identity_max_abs_gap,
        "mean_abs_total_error": mean_abs_total,
        "mean_abs_term1_transfer": mean_abs_term1,
        "mean_abs_term2_residual": mean_abs_term2,
        "term1_share_pct": mean_abs_term1 / (mean_abs_term1 + mean_abs_term2) * 100,
        "dominant_term": "Term1(迁移放大)" if mean_abs_term1 > mean_abs_term2 else "Term2(残差估计)",
    }


VARIANTS = [
    dict(name="全部组合(基线)", use_fine_tune=False, use_coral=False),
    dict(name="+D+E", use_fine_tune=True, use_coral=True),
]


def summarize_rolling_origin(variant_name: str, results: list) -> dict:
    term1_share = np.array([r["term1_share_pct"] for r in results])
    term1_abs = np.array([r["mean_abs_term1_transfer"] for r in results])
    term2_abs = np.array([r["mean_abs_term2_residual"] for r in results])
    total_abs = np.array([r["mean_abs_total_error"] for r in results])
    dominant_terms = {r["dominant_term"] for r in results}

    return {
        "name": variant_name,
        "n_splits": len(results),
        "splits": [r["calib_fraction"] for r in results],
        "term1_share_pct_mean": float(term1_share.mean()),
        "term1_share_pct_std": float(term1_share.std()),
        "mean_abs_term1_transfer_mean": float(term1_abs.mean()),
        "mean_abs_term1_transfer_std": float(term1_abs.std()),
        "mean_abs_term2_residual_mean": float(term2_abs.mean()),
        "mean_abs_term2_residual_std": float(term2_abs.std()),
        "mean_abs_total_error_mean": float(total_abs.mean()),
        "mean_abs_total_error_std": float(total_abs.std()),
        "dominant_term_consistent": len(dominant_terms) == 1,
        "dominant_terms_seen": sorted(dominant_terms),
    }


def result_path_for(name: str, calib_fraction: float) -> str:
    """Per-(variant, split) result cache: since a full rolling-origin sweep is 3 splits x 2
    variants = 6 full MLDG/CORAL training runs (~20-30 min each), a background kill partway
    through the sweep should not force re-training combinations already finished -- only the
    per-training-loop checkpoint_path protects against a kill *mid* run, not one *between*
    runs the way this file-level cache does."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    frac_slug = str(round(calib_fraction, 2)).replace(".", "")
    return f"/tmp/transcif_result_theorem1_{slug}_{frac_slug}.json"


if __name__ == "__main__":
    import json

    all_results = {variant["name"]: [] for variant in VARIANTS}

    for calib_fraction in ROLLING_ORIGIN_SPLITS:
        for variant in VARIANTS:
            path = result_path_for(variant["name"], calib_fraction)
            if os.path.exists(path):
                with open(path) as f:
                    result = json.load(f)
                print(f"cached: {variant['name']} @ calib_fraction={calib_fraction}", flush=True)
            else:
                print(f"running variant: {variant['name']} @ calib_fraction={calib_fraction} ...", flush=True)
                result = decompose(**variant, calib_fraction=calib_fraction)
                result = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in result.items()}
                with open(path, "w") as f:
                    json.dump(result, f)
            print(result, flush=True)
            all_results[variant["name"]].append(result)

    print("\n=== rolling-origin summary (splits: {}) ===".format(ROLLING_ORIGIN_SPLITS))
    for variant in VARIANTS:
        summary = summarize_rolling_origin(variant["name"], all_results[variant["name"]])
        print(summary, flush=True)
