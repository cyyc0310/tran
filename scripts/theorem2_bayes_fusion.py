"""Theorem 2: horizon-specific Bayes-optimal (minimum-variance) linear fusion of three
RenewShare predictors for SA1, under the fixed HORIZON=12 constraint, feeding into Theorem 1's
exact CI-transfer-error decomposition via the fused share estimate.

Three predictors, chosen because they carry genuinely different, largely complementary
information (a precondition for combination to help -- see Bates & Granger 1969):

  1. Persistence: s_hat_pers(t+h) = s_t (last observed share). Good at short horizons, decays
     badly at long horizons (the empirical h=1..12 drift table already computed on real SA1
     data this session shows mean abs share change growing from 4.3% at h=1 to ~20% at h=12).
  2. Network (the "+D+E" model from sa1_domain_adaptation.py -- MLDG+CORAL pretraining, then
     gradual-unfreezing fine-tuning on SA1's calibration split): captures learned dynamics from
     the input window (RenewShare/LoadNorm/RenewOutNorm/NonRenewOutNorm/TempAnomaly trajectory).
  3. Diurnal climatology: mean RenewShare by hour-of-day, estimated directly from the calib
     split's REAL labels (y_calib_share) -- captures the systematic solar/wind daily cycle.
     src/transcif/data/loaders.py confirms NO channel in the current pipeline encodes
     hour-of-day explicitly (only RenewShare/LoadNorm/RenewOutNorm/NonRenewOutNorm/TempAnomaly
     trajectories), so this is information the network cannot already be exploiting from its
     inputs -- a genuinely orthogonal predictor, not a redundant one.

Theorem 2 (closed form): for each horizon step h, given K unbiased (bias-corrected) predictors
with calibration-estimated error covariance Sigma_h (KxK), the affine minimum-variance
combination minimizing E[(s_fused_h - s_true_h)^2] is w*_h = (Sigma_h^-1 1) / (1^T Sigma_h^-1 1)
(Bates-Granger 1969). Weights sum to 1 but are NOT constrained to be non-negative -- this is an
affine combination, not a convex one (fitted weights on real SA1 data include negative
components, e.g. h=1: [1.58, -0.732, 0.152]). Substituting s_fused_h into Theorem 1's exact
decomposition (CI_pred - CI_true = (s_hat-s_true)*(C_renew-C_nonrenew) + (Delta_hat-epsilon))
gives the linear-combination-optimal Term1 for each h, hence the smallest achievable expected
Term1 among all affine combinations of these three predictors.

Overfitting diagnostic (Bates & Granger 1969; Hansen lecture notes; R `ForecastComb::comb_BG`
docs): weights are fit on calib-split residuals only, so a component whose calib-split MAE is
far below its eval-split MAE is a sign the fusion may be overweighting a component that doesn't
generalize -- main() prints each predictor's calib-vs-eval MAE gap (in RenewShare units) to
check for this before trusting the fused result.

Weights and bias corrections are estimated ONLY on the calib split (real SA1 labels, same
"light calibration" paradigm already used elsewhere in this pipeline for the residual head and
dominant-variable reweighting) and applied out-of-sample to the eval split.

Honest expectation, established via real-data analysis before writing this script: at the
fixed HORIZON=12, persistence_mae (67.568, itself already an average over h=1..12 -- confirmed
this session to reconcile with the raw h=12-only persistence-CI-error of ~95.8) is close to the
natural drift-implied floor. This fusion is very unlikely to reach MAE~10; a realistic bar is
roughly 40-55 if the three predictors are meaningfully decorrelated. Report the comparison
honestly regardless of which way it comes out.

Run with: PYTHONPATH=src python scripts/theorem2_bayes_fusion.py
"""

import numpy as np
import torch

from transcif.calibration.dominant_reweight import recompute_dominant_variable, reweight_lt_mwkc_alpha
from transcif.evaluation.metrics import mae
from transcif.physics.cif import cif_from_shares, get_emission_factors
from transcif.physics.residual import ResidualCorrectionHead, fit_residual_head
from transcif.training.domain_adaptation import fine_tune_on_calibration, train_multi_source_mldg_coral

from sa1_ablation import CALIB_FRACTION, DATA_DIR, HORIZON, REGION_TO_FACTOR_CODE, SEED, TARGET_REGION, load_source_and_target
from sa1_domain_adaptation import (
    CORAL_WEIGHT,
    FINE_TUNE_EPOCHS_PER_STAGE,
    FINE_TUNE_LR,
    INCLUDE_GENERATION,
    INCLUDE_TEMPERATURE,
    MLDG_EPOCHS,
    build_model,
)


def extract_hour_of_day_windows(seq_len: int, horizon: int, stride: int) -> np.ndarray:
    """Mirrors sa1_ablation.extract_ci_true_windows' exact (start, window, stride) indexing so
    the real hour-of-day (0-23) of each future horizon step lines up 1:1 with the (x, y)
    windows -- needed to build the diurnal-climatology predictor and to apply it out-of-sample."""
    from transcif.data.loaders import load_region_hourly_csv

    df = load_region_hourly_csv(f"{DATA_DIR}/nem_2023_hourly_{TARGET_REGION}.csv")
    hour_of_day = df["hour"].dt.hour.to_numpy()
    window = seq_len + horizon
    windows = [hour_of_day[start + seq_len : start + window] for start in range(0, len(hour_of_day) - window + 1, stride)]
    return np.stack(windows)


def fit_diurnal_climatology(y_calib_share: np.ndarray, hour_of_day_calib: np.ndarray) -> np.ndarray:
    """mean RenewShare per hour-of-day (0..23), estimated on the calib split's real labels."""
    climatology = np.zeros(24)
    for h in range(24):
        mask = hour_of_day_calib == h
        climatology[h] = y_calib_share[mask].mean() if mask.any() else y_calib_share.mean()
    return climatology


def predict_diurnal_climatology(climatology: np.ndarray, hour_of_day: np.ndarray) -> np.ndarray:
    return climatology[hour_of_day]


OVERFIT_GAP_FLAG_THRESHOLD_PCT = 20.0


def diagnose_component_overfitting(
    label: str, calib_true: np.ndarray, calib_pred: np.ndarray, eval_true: np.ndarray, eval_pred: np.ndarray
) -> dict:
    """Bates-Granger overfitting diagnostic (Hansen lecture notes; R ForecastComb::comb_BG docs):
    weights are fit on calib-split residuals only, so a component whose calib-split MAE is far
    below its eval-split MAE is a candidate for being overweighted by the fusion in a way that
    won't generalize. calib_mae/eval_mae are each predictor's own RenewShare-space MAE (not the
    fused result) -- flat here (rather than per-horizon) to keep the diagnostic readable."""
    calib_mae = mae(calib_true.reshape(-1), calib_pred.reshape(-1))
    eval_mae = mae(eval_true.reshape(-1), eval_pred.reshape(-1))
    gap_pct = (eval_mae - calib_mae) / calib_mae * 100
    return {"label": label, "calib_mae": calib_mae, "eval_mae": eval_mae, "gap_pct": gap_pct}


def fit_bayes_optimal_weights(residuals: np.ndarray) -> tuple:
    """residuals: (N_calib, K) predictor errors (predictor - true) at one horizon step.
    Returns (bias, weights): bias[k] to subtract from each predictor before combining,
    weights (K,) summing to 1, solved via w* = (Sigma^-1 1) / (1^T Sigma^-1 1)."""
    bias = residuals.mean(axis=0)
    debiased = residuals - bias
    sigma = np.cov(debiased, rowvar=False)
    sigma = sigma + np.eye(sigma.shape[0]) * 1e-8
    ones = np.ones(sigma.shape[0])
    sigma_inv_ones = np.linalg.solve(sigma, ones)
    weights = sigma_inv_ones / sigma_inv_ones.sum()
    return bias, weights


def main() -> None:
    source_windows, x_target, y_target_share, ci_true_target = load_source_and_target(
        INCLUDE_GENERATION, INCLUDE_TEMPERATURE
    )
    hour_of_day_windows = extract_hour_of_day_windows(seq_len=48, horizon=HORIZON, stride=6)

    n = x_target.shape[0]
    split = int(n * CALIB_FRACTION)
    x_calib, x_eval = x_target[:split], x_target[split:]
    y_calib_share, y_eval_share = y_target_share[:split].numpy(), y_target_share[split:].numpy()
    ci_true_calib, ci_true_eval = ci_true_target[:split], ci_true_target[split:]
    hour_of_day_calib, hour_of_day_eval = hour_of_day_windows[:split], hour_of_day_windows[split:]

    model = build_model()
    train_multi_source_mldg_coral(model, source_windows, x_calib, epochs=MLDG_EPOCHS, coral_weight=CORAL_WEIGHT)
    fine_tune_on_calibration(
        model, x_calib, torch.tensor(y_calib_share, dtype=torch.float32),
        epochs_per_stage=FINE_TUNE_EPOCHS_PER_STAGE, lr=FINE_TUNE_LR,
    )

    renew_factor, nonrenew_factor = get_emission_factors(REGION_TO_FACTOR_CODE[TARGET_REGION])

    with torch.no_grad():
        s_hat_calib, _ = model(x_calib)
    dominant_idx = recompute_dominant_variable(model, x_calib)
    reweight_lt_mwkc_alpha(model, dominant_idx)

    with torch.no_grad():
        s_hat_calib, _ = model(x_calib)
        s_hat_eval, _ = model(x_eval)
    s_hat_calib_np, s_hat_eval_np = s_hat_calib.numpy(), s_hat_eval.numpy()

    last_observed_share_calib = x_calib[:, -1, 0].numpy()
    last_observed_share_eval = x_eval[:, -1, 0].numpy()
    s_pers_calib = np.repeat(last_observed_share_calib[:, None], HORIZON, axis=1)
    s_pers_eval = np.repeat(last_observed_share_eval[:, None], HORIZON, axis=1)

    climatology = fit_diurnal_climatology(y_calib_share.reshape(-1), hour_of_day_calib.reshape(-1))
    s_clim_calib = predict_diurnal_climatology(climatology, hour_of_day_calib)
    s_clim_eval = predict_diurnal_climatology(climatology, hour_of_day_eval)

    overfit_diagnostics = [
        diagnose_component_overfitting("persistence", y_calib_share, s_pers_calib, y_eval_share, s_pers_eval),
        diagnose_component_overfitting("network", y_calib_share, s_hat_calib_np, y_eval_share, s_hat_eval_np),
        diagnose_component_overfitting("climatology", y_calib_share, s_clim_calib, y_eval_share, s_clim_eval),
    ]

    biases_per_h, weights_per_h = [], []
    s_fused_eval = np.zeros_like(s_hat_eval_np)
    for h in range(HORIZON):
        predictors_calib_h = np.stack([s_pers_calib[:, h], s_hat_calib_np[:, h], s_clim_calib[:, h]], axis=1)
        residuals_h = predictors_calib_h - y_calib_share[:, h : h + 1]
        bias_h, weights_h = fit_bayes_optimal_weights(residuals_h)
        biases_per_h.append(bias_h)
        weights_per_h.append(weights_h)

        predictors_eval_h = np.stack([s_pers_eval[:, h], s_hat_eval_np[:, h], s_clim_eval[:, h]], axis=1)
        debiased_eval_h = predictors_eval_h - bias_h
        s_fused_eval[:, h] = debiased_eval_h @ weights_h

    predictors_calib = np.stack([s_pers_calib, s_hat_calib_np, s_clim_calib], axis=-1)
    s_fused_calib = np.zeros_like(s_hat_calib_np)
    for h in range(HORIZON):
        s_fused_calib[:, h] = (predictors_calib[:, h, :] - biases_per_h[h]) @ weights_per_h[h]

    ci_pred_physics_calib = cif_from_shares(s_fused_calib, renew_factor, nonrenew_factor)
    ci_pred_physics_eval = cif_from_shares(s_fused_eval, renew_factor, nonrenew_factor)

    residual_head = ResidualCorrectionHead(input_dim=1, hidden_dim=8)
    calib_features = torch.tensor(s_fused_calib.reshape(-1, 1), dtype=torch.float32)
    calib_targets = torch.tensor((ci_true_calib - ci_pred_physics_calib).reshape(-1), dtype=torch.float32)
    fit_residual_head(residual_head, calib_features, calib_targets, epochs=100, lr=1e-2)

    eval_features = torch.tensor(s_fused_eval.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        delta_hat_eval = residual_head(eval_features).numpy().reshape(ci_pred_physics_eval.shape)
    ci_pred_eval = ci_pred_physics_eval + delta_hat_eval

    fused_mae = mae(ci_true_eval.reshape(-1), ci_pred_eval.reshape(-1))

    ci_persistence_eval = cif_from_shares(s_pers_eval, renew_factor, nonrenew_factor)
    persistence_mae = mae(ci_true_eval.reshape(-1), ci_persistence_eval.reshape(-1))

    ci_pred_network_only = cif_from_shares(s_hat_eval_np, renew_factor, nonrenew_factor)
    network_only_physics_mae = mae(ci_true_eval.reshape(-1), ci_pred_network_only.reshape(-1))

    ci_pred_climatology_only = cif_from_shares(s_clim_eval, renew_factor, nonrenew_factor)
    climatology_only_mae = mae(ci_true_eval.reshape(-1), ci_pred_climatology_only.reshape(-1))

    epsilon_eval = ci_true_eval - cif_from_shares(y_eval_share[:, :HORIZON], renew_factor, nonrenew_factor)
    term1 = (s_fused_eval - y_eval_share[:, :HORIZON]) * (renew_factor - nonrenew_factor)
    term2 = delta_hat_eval - epsilon_eval
    mean_abs_term1, mean_abs_term2 = np.mean(np.abs(term1)), np.mean(np.abs(term2))

    print("=== Theorem 2 numerical validation on real SA1 2023 data ===")
    print(f"persistence_mae (baseline floor)        = {persistence_mae:.3f}")
    print(f"network-only physics_mae (D+E, no fusion) = {network_only_physics_mae:.3f}")
    print(f"climatology-only physics_mae             = {climatology_only_mae:.3f}")
    print(f"fused (Theorem 2) corrected_mae          = {fused_mae:.3f}")
    print(f"fused vs persistence                     = {(fused_mae - persistence_mae) / persistence_mae * 100:+.1f}%")
    print(f"mean|Term1| (fused transfer amplification) = {mean_abs_term1:.3f}")
    print(f"mean|Term2| (residual estimation)          = {mean_abs_term2:.3f}")
    print(f"Term1 share = {mean_abs_term1 / (mean_abs_term1 + mean_abs_term2) * 100:.1f}%")
    print("\nper-horizon weights [persistence, network, climatology] and bias:")
    for h in range(HORIZON):
        print(f"  h={h + 1:2d}  weights={np.round(weights_per_h[h], 3)}  bias={np.round(biases_per_h[h], 4)}")

    print("\n=== Bates-Granger overfitting diagnostic (per-predictor MAE, RenewShare units) ===")
    for d in overfit_diagnostics:
        flag = " <- overfit signal" if d["gap_pct"] > OVERFIT_GAP_FLAG_THRESHOLD_PCT else ""
        print(
            f"  {d['label']:12s} calib_mae={d['calib_mae']:.4f} eval_mae={d['eval_mae']:.4f} "
            f"gap={d['gap_pct']:+.1f}%{flag}"
        )


if __name__ == "__main__":
    main()
