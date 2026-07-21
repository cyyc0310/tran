"""SA1 domain-adaptation ablation, round 2: on top of the best prior variant (全部组合,
corrected_mae=75.508 vs persistence_mae=67.568), tests two "more fundamental" domain
adaptation techniques surfaced by literature search (AAAI 2024 workshop, WWW'26):

  +编码器微调(D): after MLDG pretraining on QLD1/NSW1/VIC1, gradually-unfreeze fine-tune the
  encoder directly on SA1's real calibration-split (x_calib, y_calib_share) supervised pairs,
  before physics reconstruction + residual correction -- modeled on IBM Research's AAAI 2024
  workshop one-step-fine-tuning method, and validated as high-impact specifically for
  cross-region carbon-intensity forecasting by a WWW'26 paper's ablation (removing
  target-region fine-tuning cost 11.4% MAPE there).

  +CORAL特征对齐(E): during MLDG pretraining, add a Deep CORAL covariance-alignment loss
  between the pooled source-region `fused` features and SA1's calibration-split *inputs*
  (unsupervised -- no target labels needed).

  +D+E: both combined.

Ground truth, split, physics reconstruction, and residual correction are unchanged from
scripts/sa1_ablation.py so results compare directly against its logged persistence_mae and
六-variant summary.

Run with: PYTHONPATH=src python scripts/sa1_domain_adaptation.py
"""

import re

import numpy as np
import torch

from transcif.calibration.dominant_reweight import recompute_dominant_variable, reweight_lt_mwkc_alpha
from transcif.evaluation.metrics import mae
from transcif.models.encoder import DomainInvariantEncoder, PersistenceSkipEncoder
from transcif.physics.cif import cif_from_shares, get_emission_factors
from transcif.physics.residual import ResidualCorrectionHead, fit_residual_head
from transcif.training.domain_adaptation import fine_tune_on_calibration, train_multi_source_mldg_coral
from transcif.training.train_multi_source import train_multi_source_erm, train_multi_source_mldg

from sa1_ablation import (
    CALIB_FRACTION,
    HORIZON,
    LT_FEATURE_DIM,
    CV_FEATURE_DIM,
    MLDG_EPOCHS,
    REGION_TO_FACTOR_CODE,
    SEED,
    TARGET_REGION,
    load_source_and_target,
)

FINE_TUNE_EPOCHS_PER_STAGE = 15
FINE_TUNE_LR = 5e-4
CORAL_WEIGHT = 0.1

# 全部组合 variant's data configuration: 5 channels (RenewShare, LoadNorm, RenewOutNorm,
# NonRenewOutNorm, TempAnomaly), gate conditioning on, MLDG domain-adaptive weighting on.
NUM_CHANNELS = 5
INCLUDE_GENERATION = True
INCLUDE_TEMPERATURE = True


def build_model() -> PersistenceSkipEncoder:
    torch.manual_seed(SEED)
    base = DomainInvariantEncoder(
        num_variables=NUM_CHANNELS, horizon=HORIZON, lt_feature_dim=LT_FEATURE_DIM, cv_feature_dim=CV_FEATURE_DIM
    )
    return PersistenceSkipEncoder(base)


def evaluate(model, x_calib, x_eval, ci_true_calib, ci_true_eval) -> dict:
    renew_factor, nonrenew_factor = get_emission_factors(REGION_TO_FACTOR_CODE[TARGET_REGION])

    with torch.no_grad():
        renew_share_pred_calib, _ = model(x_calib)

    dominant_idx = recompute_dominant_variable(model, x_calib)
    reweight_lt_mwkc_alpha(model, dominant_idx)

    with torch.no_grad():
        renew_share_pred_calib, _ = model(x_calib)
        renew_share_pred_eval, _ = model(x_eval)

    ci_pred_physics_calib = cif_from_shares(renew_share_pred_calib.numpy(), renew_factor, nonrenew_factor)
    ci_pred_physics_eval = cif_from_shares(renew_share_pred_eval.numpy(), renew_factor, nonrenew_factor)

    residual_head = ResidualCorrectionHead(input_dim=1, hidden_dim=8)
    calib_features = torch.tensor(renew_share_pred_calib.numpy().reshape(-1, 1), dtype=torch.float32)
    calib_targets = torch.tensor((ci_true_calib - ci_pred_physics_calib).reshape(-1), dtype=torch.float32)
    fit_residual_head(residual_head, calib_features, calib_targets, epochs=100, lr=1e-2)

    eval_features = torch.tensor(renew_share_pred_eval.numpy().reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        delta_eval = residual_head(eval_features).numpy().reshape(ci_pred_physics_eval.shape)
    ci_pred_corrected_eval = ci_pred_physics_eval + delta_eval

    corrected_mae = mae(ci_true_eval.reshape(-1), ci_pred_corrected_eval.reshape(-1))
    physics_only_mae = mae(ci_true_eval.reshape(-1), ci_pred_physics_eval.reshape(-1))

    last_observed_share = x_eval[:, -1, 0].numpy()
    persistence_share_pred = np.repeat(last_observed_share[:, None], HORIZON, axis=1)
    ci_persistence_eval = cif_from_shares(persistence_share_pred, renew_factor, nonrenew_factor)
    persistence_mae = mae(ci_true_eval.reshape(-1), ci_persistence_eval.reshape(-1))

    return {
        "physics_only_mae": physics_only_mae,
        "corrected_mae": corrected_mae,
        "persistence_mae": persistence_mae,
        "corrected_vs_persistence_pct": (corrected_mae - persistence_mae) / persistence_mae * 100,
    }


def checkpoint_path_for(name: str, stage: str) -> str:
    """Filename-safe checkpoint path so a background run killed mid-training (observed in this
    environment: background tasks can be terminated well before the CORAL variant's 80 MLDG
    epochs finish) can resume from its last saved epoch on relaunch instead of restarting."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    return f"/tmp/transcif_ckpt_{slug}_{stage}.pt"


def run_variant(name: str, use_fine_tune: bool, use_coral: bool, use_erm: bool = False) -> dict:
    source_windows, x_target, y_target_share, ci_true_target = load_source_and_target(
        INCLUDE_GENERATION, INCLUDE_TEMPERATURE
    )

    n = x_target.shape[0]
    split = int(n * CALIB_FRACTION)
    x_calib, x_eval = x_target[:split], x_target[split:]
    y_calib_share = y_target_share[:split]
    ci_true_calib, ci_true_eval = ci_true_target[:split], ci_true_target[split:]

    model = build_model()

    if use_erm:
        train_losses = train_multi_source_erm(
            model, source_windows, epochs=MLDG_EPOCHS, checkpoint_path=checkpoint_path_for(name, "erm")
        )
    elif use_coral:
        train_losses = train_multi_source_mldg_coral(
            model, source_windows, x_calib, epochs=MLDG_EPOCHS, coral_weight=CORAL_WEIGHT,
            checkpoint_path=checkpoint_path_for(name, "coral"),
        )
    else:
        train_losses = train_multi_source_mldg(model, source_windows, epochs=MLDG_EPOCHS)

    fine_tune_losses = None
    if use_fine_tune:
        fine_tune_losses = fine_tune_on_calibration(
            model, x_calib, y_calib_share, epochs_per_stage=FINE_TUNE_EPOCHS_PER_STAGE, lr=FINE_TUNE_LR
        )

    result = evaluate(model, x_calib, x_eval, ci_true_calib, ci_true_eval)
    result["name"] = name
    result["final_train_loss"] = train_losses[-1]
    if fine_tune_losses is not None:
        result["final_fine_tune_loss"] = fine_tune_losses[-1]
    return result


VARIANTS = [
    dict(name="全部组合(基线)", use_fine_tune=False, use_coral=False),
    dict(name="+编码器微调(D)", use_fine_tune=True, use_coral=False),
    dict(name="+CORAL特征对齐(E)", use_fine_tune=False, use_coral=True),
    dict(name="+D+E", use_fine_tune=True, use_coral=True),
    dict(name="纯ERM基线(无MLDG,DomainBed对照)", use_fine_tune=False, use_coral=False, use_erm=True),
]


if __name__ == "__main__":
    results = []
    for variant_config in VARIANTS:
        print(f"running variant: {variant_config['name']} ...", flush=True)
        result = run_variant(**variant_config)
        print(result, flush=True)
        results.append(result)

    print("\n=== summary ===")
    for r in results:
        print(
            f"{r['name']:20s} corrected_mae={r['corrected_mae']:.3f} "
            f"physics_only_mae={r['physics_only_mae']:.3f} "
            f"persistence_mae={r['persistence_mae']:.3f} "
            f"vs_persistence={r['corrected_vs_persistence_pct']:+.1f}%"
        )
