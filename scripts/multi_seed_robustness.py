"""Multi-seed robustness check for SA1's "全部组合(基线)" vs "+D+E" comparison.

Both scripts/sa1_domain_adaptation.py and scripts/theorem1_validation.py fix
torch.manual_seed(SEED=42) for model-weight initialization, but the MLDG training loop
(train_multi_source.py:67 / domain_adaptation.py:67) draws its meta-test source region via
Python's un-seeded `random.choice(regions)` every epoch. That is the confirmed source of the
run-to-run corrected_mae variance observed across separate script invocations (e.g. 75.508 vs
74.712 for the same "全部组合" configuration) -- persistence_mae never moves because it depends
only on the real SA1 series itself, not on training.

This script controls that leak explicitly (random.seed(seed) + torch.manual_seed(seed), same
seed value, immediately before each run) and repeats both variants across several seeds to
check three things on real AEMO 2023 data:

  1. Is "全部组合(基线)" always beaten by "+D+E", or was the -2.3% improvement luck?
  2. Is the corrected_mae improvement of +D+E over the baseline consistent in sign/magnitude
     across seeds (reported as a paired difference, not a formal significance test -- n=5 is
     too small for that)?
  3. Is Corollary 1's Term①-dominates-SA1's-error finding (~79%/78.5% in the single-seed run)
     stable across seeds, or was that a single lucky draw?

Run with: PYTHONPATH=src python scripts/multi_seed_robustness.py
"""

import random

import numpy as np
import torch

from transcif.calibration.dominant_reweight import recompute_dominant_variable, reweight_lt_mwkc_alpha
from transcif.evaluation.metrics import mae
from transcif.models.encoder import DomainInvariantEncoder, PersistenceSkipEncoder
from transcif.physics.cif import cif_from_shares, get_emission_factors
from transcif.physics.residual import ResidualCorrectionHead, fit_residual_head
from transcif.training.domain_adaptation import fine_tune_on_calibration, train_multi_source_mldg_coral
from transcif.training.train_multi_source import train_multi_source_mldg

from sa1_ablation import CALIB_FRACTION, HORIZON, LT_FEATURE_DIM, CV_FEATURE_DIM, REGION_TO_FACTOR_CODE, TARGET_REGION, load_source_and_target
from sa1_domain_adaptation import CORAL_WEIGHT, FINE_TUNE_EPOCHS_PER_STAGE, FINE_TUNE_LR, INCLUDE_GENERATION, INCLUDE_TEMPERATURE, MLDG_EPOCHS, NUM_CHANNELS

SEEDS = [0, 1, 2, 3, 4]
VARIANTS = [
    dict(name="全部组合(基线)", use_fine_tune=False, use_coral=False),
    dict(name="+D+E", use_fine_tune=True, use_coral=True),
]


def run_variant_for_seed(seed: int, use_fine_tune: bool, use_coral: bool, data) -> dict:
    source_windows, x_calib, x_eval, y_calib_share, y_eval_share, ci_true_calib, ci_true_eval = data

    random.seed(seed)
    torch.manual_seed(seed)
    base = DomainInvariantEncoder(
        num_variables=NUM_CHANNELS, horizon=HORIZON, lt_feature_dim=LT_FEATURE_DIM, cv_feature_dim=CV_FEATURE_DIM
    )
    model = PersistenceSkipEncoder(base)

    if use_coral:
        train_multi_source_mldg_coral(model, source_windows, x_calib, epochs=MLDG_EPOCHS, coral_weight=CORAL_WEIGHT)
    else:
        train_multi_source_mldg(model, source_windows, epochs=MLDG_EPOCHS)

    if use_fine_tune:
        fine_tune_on_calibration(
            model, x_calib, y_calib_share, epochs_per_stage=FINE_TUNE_EPOCHS_PER_STAGE, lr=FINE_TUNE_LR
        )

    renew_factor, nonrenew_factor = get_emission_factors(REGION_TO_FACTOR_CODE[TARGET_REGION])

    with torch.no_grad():
        s_hat_calib, _ = model(x_calib)
    dominant_idx = recompute_dominant_variable(model, x_calib)
    reweight_lt_mwkc_alpha(model, dominant_idx)

    with torch.no_grad():
        s_hat_calib, _ = model(x_calib)
        s_hat_eval, _ = model(x_eval)

    s_hat_calib_np = s_hat_calib.numpy()
    s_hat_eval_np = s_hat_eval.numpy()

    ci_pred_physics_calib = cif_from_shares(s_hat_calib_np, renew_factor, nonrenew_factor)
    ci_pred_physics_eval = cif_from_shares(s_hat_eval_np, renew_factor, nonrenew_factor)

    residual_head = ResidualCorrectionHead(input_dim=1, hidden_dim=8)
    calib_features = torch.tensor(s_hat_calib_np.reshape(-1, 1), dtype=torch.float32)
    calib_targets = torch.tensor((ci_true_calib - ci_pred_physics_calib).reshape(-1), dtype=torch.float32)
    fit_residual_head(residual_head, calib_features, calib_targets, epochs=100, lr=1e-2)

    eval_features = torch.tensor(s_hat_eval_np.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        delta_hat_eval = residual_head(eval_features).numpy().reshape(ci_pred_physics_eval.shape)
    ci_pred_eval = ci_pred_physics_eval + delta_hat_eval

    corrected_mae = mae(ci_true_eval.reshape(-1), ci_pred_eval.reshape(-1))

    last_observed_share = x_eval[:, -1, 0].numpy()
    persistence_share_pred = np.repeat(last_observed_share[:, None], HORIZON, axis=1)
    ci_persistence_eval = cif_from_shares(persistence_share_pred, renew_factor, nonrenew_factor)
    persistence_mae = mae(ci_true_eval.reshape(-1), ci_persistence_eval.reshape(-1))

    s_true_eval = y_eval_share.numpy()[:, :HORIZON]
    ci_true_physics_at_true_share = cif_from_shares(s_true_eval, renew_factor, nonrenew_factor)
    epsilon_eval = ci_true_eval - ci_true_physics_at_true_share

    term1 = (s_hat_eval_np - s_true_eval) * (renew_factor - nonrenew_factor)
    term2 = delta_hat_eval - epsilon_eval
    mean_abs_term1 = np.mean(np.abs(term1))
    mean_abs_term2 = np.mean(np.abs(term2))

    return {
        "seed": seed,
        "corrected_mae": corrected_mae,
        "persistence_mae": persistence_mae,
        "vs_persistence_pct": (corrected_mae - persistence_mae) / persistence_mae * 100,
        "term1_share_pct": mean_abs_term1 / (mean_abs_term1 + mean_abs_term2) * 100,
    }


def paired_t_stat(diffs: np.ndarray) -> tuple:
    n = len(diffs)
    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)
    t_stat = mean_diff / (std_diff / np.sqrt(n)) if std_diff > 0 else float("inf")
    return mean_diff, std_diff, t_stat, n - 1


if __name__ == "__main__":
    print("loading real AEMO 2023 data once (shared across all seeds/variants)...", flush=True)
    source_windows, x_target, y_target_share, ci_true_target = load_source_and_target(
        INCLUDE_GENERATION, INCLUDE_TEMPERATURE
    )
    n = x_target.shape[0]
    split = int(n * CALIB_FRACTION)
    data = (
        source_windows,
        x_target[:split],
        x_target[split:],
        y_target_share[:split],
        y_target_share[split:],
        ci_true_target[:split],
        ci_true_target[split:],
    )

    all_results = {variant["name"]: [] for variant in VARIANTS}
    for seed in SEEDS:
        for variant in VARIANTS:
            print(f"running seed={seed} variant={variant['name']} ...", flush=True)
            result = run_variant_for_seed(seed, variant["use_fine_tune"], variant["use_coral"], data)
            print(result, flush=True)
            all_results[variant["name"]].append(result)

    print("\n=== per-variant summary across seeds ===")
    for name, results in all_results.items():
        maes = np.array([r["corrected_mae"] for r in results])
        pcts = np.array([r["term1_share_pct"] for r in results])
        print(
            f"{name:20s} corrected_mae mean={maes.mean():.3f} std={maes.std(ddof=1):.3f} "
            f"| term1_share_pct mean={pcts.mean():.1f}% std={pcts.std(ddof=1):.1f}%"
        )

    baseline_maes = np.array([r["corrected_mae"] for r in all_results["全部组合(基线)"]])
    de_maes = np.array([r["corrected_mae"] for r in all_results["+D+E"]])
    diffs = de_maes - baseline_maes
    mean_diff, std_diff, t_stat, df = paired_t_stat(diffs)
    print("\n=== paired D+E vs baseline (corrected_mae, negative = D+E better) ===")
    print(f"per-seed diffs: {diffs.tolist()}")
    print(f"mean_diff={mean_diff:.3f} std_diff={std_diff:.3f} paired_t={t_stat:.3f} df={df}")
    print("note: n=5 seeds is too small for a reliable p-value; this is descriptive evidence of sign/magnitude consistency, not a significance claim.")
