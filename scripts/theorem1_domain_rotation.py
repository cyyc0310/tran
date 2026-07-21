"""Item 1 robustness extension: leave-one-domain-out rotation across all four AU regions
(QLD1, NSW1, VIC1, SA1). scripts/theorem1_validation.py and its Item-2 rolling-origin sweep
only ever tested SA1 as the held-out target -- but Corollary 1 ("the region with the smallest
L_T should show Term1(transfer amplification) dominating over Term2(residual estimation)") is
inherently a *cross-region comparison* claim. Confirming it on SA1 alone, no matter how many
splits or variants, never tests whether the L_T ranking across regions actually predicts the
Term1-share ranking across regions -- there is no other region to compare against.

This script rotates the held-out target region through all four AU regions (the other three
become MLDG source regions each time), reruns the exact Theorem 1 decomposition from
theorem1_validation.py's decompose() for each, and reports L_T alongside term1_share_pct per
region so the cross-region correlation can be read off directly.

Run with: PYTHONPATH=src python scripts/theorem1_domain_rotation.py
"""

import json
import os
import re

import numpy as np
import torch

from transcif.calibration.dominant_reweight import recompute_dominant_variable, reweight_lt_mwkc_alpha
from transcif.data.loaders import load_region_hourly_csv, load_region_temperature_csv, load_region_windows, merge_temperature
from transcif.physics.cif import get_emission_factors
from transcif.physics.residual import ResidualCorrectionHead, fit_residual_head
from transcif.training.domain_adaptation import fine_tune_on_calibration, train_multi_source_mldg_coral
from transcif.training.train_multi_source import train_multi_source_mldg

from sa1_ablation import CALIB_FRACTION, DATA_DIR, HORIZON, REGION_TO_FACTOR_CODE, SEQ_LEN, STRIDE
from sa1_domain_adaptation import (
    CORAL_WEIGHT,
    FINE_TUNE_EPOCHS_PER_STAGE,
    FINE_TUNE_LR,
    INCLUDE_GENERATION,
    INCLUDE_TEMPERATURE,
    MLDG_EPOCHS,
    build_model,
)

AU_REGIONS = ["QLD1", "NSW1", "VIC1", "SA1"]

VARIANTS = [
    dict(name="全部组合(基线)", use_fine_tune=False, use_coral=False),
    dict(name="+D+E", use_fine_tune=True, use_coral=True),
]


def extract_ci_true_windows(df, seq_len: int, horizon: int, stride: int) -> np.ndarray:
    window = seq_len + horizon
    ci_real = df["cif_real_gco2_per_kwh"].to_numpy()
    windows = [ci_real[start + seq_len : start + window] for start in range(0, len(ci_real) - window + 1, stride)]
    return np.stack(windows)


def load_source_and_target(target_region: str, include_generation: bool, include_temperature: bool):
    source_regions = [r for r in AU_REGIONS if r != target_region]
    source_windows = {}
    for region in source_regions:
        csv_path = f"{DATA_DIR}/nem_2023_hourly_{region}.csv"
        temp_path = f"{DATA_DIR}/temperature_2023_{region}.csv" if include_temperature else None
        x, y = load_region_windows(
            csv_path,
            seq_len=SEQ_LEN,
            horizon=HORIZON,
            stride=STRIDE,
            include_generation_channels=include_generation,
            temp_csv_path=temp_path,
        )
        source_windows[region] = (x, y)

    target_csv = f"{DATA_DIR}/nem_2023_hourly_{target_region}.csv"
    target_temp_path = f"{DATA_DIR}/temperature_2023_{target_region}.csv" if include_temperature else None
    x_target, y_target_share = load_region_windows(
        target_csv,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        stride=STRIDE,
        include_generation_channels=include_generation,
        temp_csv_path=target_temp_path,
    )

    target_df = load_region_hourly_csv(target_csv)
    if include_temperature:
        temp_df = load_region_temperature_csv(target_temp_path)
        target_df = merge_temperature(target_df, temp_df)
    ci_true_target = extract_ci_true_windows(target_df, SEQ_LEN, HORIZON, STRIDE)

    return source_windows, x_target, y_target_share, ci_true_target


def checkpoint_path_for(target_region: str, name: str, stage: str) -> str:
    """Filename-safe checkpoint path keyed on (target_region, variant, training stage) so each
    of the 4 regions' x 2 variants' MLDG/CORAL training runs can resume independently if a
    background run is killed mid-training, without colliding with each other's state."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    return f"/tmp/transcif_ckpt_rot_{target_region}_{slug}_{stage}.pt"


def result_path_for(target_region: str, name: str) -> str:
    """Per-(target_region, variant) result cache: 4 regions x 2 variants = 8 full MLDG/CORAL
    training runs, each ~20-30 min. A background kill partway through the rotation should not
    force re-training combinations already finished."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    return f"/tmp/transcif_result_rot_{target_region}_{slug}.json"


def decompose(target_region: str, name: str, use_fine_tune: bool, use_coral: bool) -> dict:
    source_windows, x_target, y_target_share, ci_true_target = load_source_and_target(
        target_region, INCLUDE_GENERATION, INCLUDE_TEMPERATURE
    )

    n = x_target.shape[0]
    split = int(n * CALIB_FRACTION)
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
            checkpoint_path=checkpoint_path_for(target_region, name, "coral"),
        )
    else:
        train_multi_source_mldg(
            model,
            source_windows,
            epochs=MLDG_EPOCHS,
            checkpoint_path=checkpoint_path_for(target_region, name, "mldg"),
        )

    if use_fine_tune:
        fine_tune_on_calibration(
            model, x_calib, y_calib_share, epochs_per_stage=FINE_TUNE_EPOCHS_PER_STAGE, lr=FINE_TUNE_LR
        )

    renew_factor, nonrenew_factor = get_emission_factors(REGION_TO_FACTOR_CODE[target_region])
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
        "target_region": target_region,
        "name": name,
        "L_T": L_T,
        "identity_max_abs_gap": identity_max_abs_gap,
        "mean_abs_total_error": mean_abs_total,
        "mean_abs_term1_transfer": mean_abs_term1,
        "mean_abs_term2_residual": mean_abs_term2,
        "term1_share_pct": mean_abs_term1 / (mean_abs_term1 + mean_abs_term2) * 100,
        "dominant_term": "Term1(迁移放大)" if mean_abs_term1 > mean_abs_term2 else "Term2(残差估计)",
    }


def summarize_rotation(variant_name: str, results: list) -> dict:
    """Rank regions by L_T and by term1_share_pct independently; Corollary 1 predicts the two
    rankings should agree (smallest L_T -> highest term1_share_pct)."""
    by_lt = sorted(results, key=lambda r: r["L_T"])
    by_term1_share = sorted(results, key=lambda r: -r["term1_share_pct"])

    lt_order = [r["target_region"] for r in by_lt]
    term1_order = [r["target_region"] for r in by_term1_share]

    return {
        "name": variant_name,
        "regions_ranked_by_L_T_ascending": lt_order,
        "regions_ranked_by_term1_share_pct_descending": term1_order,
        "rankings_match": lt_order == term1_order,
        "per_region": [
            {
                "target_region": r["target_region"],
                "L_T": r["L_T"],
                "term1_share_pct": r["term1_share_pct"],
                "dominant_term": r["dominant_term"],
            }
            for r in results
        ],
    }


if __name__ == "__main__":
    all_results = {variant["name"]: [] for variant in VARIANTS}

    for target_region in AU_REGIONS:
        for variant in VARIANTS:
            path = result_path_for(target_region, variant["name"])
            if os.path.exists(path):
                with open(path) as f:
                    result = json.load(f)
                print(f"cached: {variant['name']} @ target_region={target_region}", flush=True)
            else:
                print(f"running variant: {variant['name']} @ target_region={target_region} ...", flush=True)
                result = decompose(target_region=target_region, **variant)
                result = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in result.items()}
                with open(path, "w") as f:
                    json.dump(result, f)
            print(result, flush=True)
            all_results[variant["name"]].append(result)

    print("\n=== leave-one-domain-out rotation summary ===")
    for variant in VARIANTS:
        summary = summarize_rotation(variant["name"], all_results[variant["name"]])
        print(summary, flush=True)
