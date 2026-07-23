"""`deploy_region`: run the full TransCIF Stage 1->3 pipeline for a `DeploymentConfig`,
reusing the same training/physics/calibration building blocks the experiment scripts use.
This is the single entry point behind the paper's "adapt by editing a config, not code"
claim: `scripts/sa1_ablation.py`'s hardcoded orchestration becomes one call here.

Nothing about the science changes -- this only removes the copy-pasted glue. The channel
switches and D/E toggles on `DeploymentConfig` select exactly the training paths already
implemented in `training/`; emission factors come from the target `RegionConfig` (inline or
table); ground-truth CIF for residual fitting / evaluation is the real measured
`cif_real_gco2_per_kwh` column when present, and is skipped (prediction-only) when absent."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from transcif.calibration.conformal import (
    compute_nonconformity_scores,
    conformal_interval_halfwidth,
    empirical_coverage,
    predict_with_interval,
)
from transcif.calibration.dominant_reweight import recompute_dominant_variable, reweight_lt_mwkc_alpha
from transcif.config.region_config import DeploymentConfig, RegionConfig
from transcif.data.loaders import (
    load_region_hourly_csv,
    load_region_temperature_csv,
    load_region_windows,
    merge_temperature,
)
from transcif.evaluation.metrics import mae
from transcif.models.encoder import DomainInvariantEncoder, PersistenceSkipEncoder
from transcif.physics.cif import cif_from_shares
from transcif.physics.residual import ResidualCorrectionHead, fit_residual_head
from transcif.training.domain_adaptation import fine_tune_on_calibration, train_multi_source_mldg_coral
from transcif.training.train_multi_source import train_multi_source_erm, train_multi_source_mldg
from transcif.training.train_source import train_source_domain

MEASURED_CI_COLUMN = "cif_real_gco2_per_kwh"


def _load_windows(region: RegionConfig, config: DeploymentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    temp_path = region.temperature_csv if config.include_temperature else None
    return load_region_windows(
        region.hourly_csv,
        seq_len=config.seq_len,
        horizon=config.horizon,
        stride=config.stride,
        include_generation_channels=config.include_generation,
        temp_csv_path=temp_path,
    )


def _load_measured_ci_windows(region: RegionConfig, config: DeploymentConfig) -> np.ndarray | None:
    """Extract the real measured CIF horizon slice aligned 1:1 with `_load_windows`' (x, y)
    windows, or None if the region's CSV has no measured column (prediction-only region).
    Mirrors `build_sliding_windows`' exact (start, seq_len, horizon, stride) indexing."""
    df = load_region_hourly_csv(region.hourly_csv)
    if config.include_temperature and region.temperature_csv is not None:
        df = merge_temperature(df, load_region_temperature_csv(region.temperature_csv))
    if MEASURED_CI_COLUMN not in df.columns:
        return None
    window = config.seq_len + config.horizon
    ci_real = df[MEASURED_CI_COLUMN].to_numpy()
    starts = range(0, len(ci_real) - window + 1, config.stride)
    return np.stack([ci_real[start + config.seq_len : start + window] for start in starts])


def _train_encoder(model: torch.nn.Module, source_windows: dict, x_calib: torch.Tensor, config: DeploymentConfig):
    """Select the training path implied by the config: MLDG meta-learning (optionally with
    Deep CORAL alignment to the target's unlabeled calibration inputs) for >= 2 sources,
    unweighted ERM pooling when domain weighting is switched off, or plain single-source
    training when only one source is provided."""
    n_sources = len(source_windows)
    if n_sources == 0:
        raise ValueError("deployment needs at least one source region")
    if n_sources == 1:
        (x_source, y_source), = source_windows.values()
        return train_source_domain(model, x_source, y_source, epochs=config.mldg_epochs)
    if config.coral:
        return train_multi_source_mldg_coral(
            model, source_windows, x_calib, epochs=config.mldg_epochs, coral_weight=config.coral_weight
        )
    if config.mldg_weighted:
        return train_multi_source_mldg(model, source_windows, epochs=config.mldg_epochs)
    return train_multi_source_erm(model, source_windows, epochs=config.mldg_epochs)


def deploy_region(config: DeploymentConfig, coverage: float = 0.9) -> dict[str, Any]:
    """Run Stage 1 (encoder training) -> Stage 2 (physics + residual) -> Stage 3
    (dominant-variable reweight + split-conformal interval) for `config`, returning a
    results dict. When the target CSV carries the measured CIF column, the returned dict
    includes residual-corrected point metrics and empirical conformal coverage; otherwise
    it is prediction-only (physics point forecast, no ground-truth-dependent fields)."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    source_windows = {s.name: _load_windows(s, config) for s in config.sources}
    x_target, y_target_share = _load_windows(config.target, config)

    n = x_target.shape[0]
    split = int(n * config.calib_fraction)
    x_calib, x_eval = x_target[:split], x_target[split:]
    y_calib_share = y_target_share[:split]

    ci_true = _load_measured_ci_windows(config.target, config)
    ci_true_calib = ci_true[:split] if ci_true is not None else None
    ci_true_eval = ci_true[split:] if ci_true is not None else None

    base = DomainInvariantEncoder(
        num_variables=config.num_channels,
        horizon=config.horizon,
        lt_feature_dim=config.lt_feature_dim,
        cv_feature_dim=config.cv_feature_dim,
    )
    model = PersistenceSkipEncoder(base)
    if not config.gate_conditioning:
        model.volatility_gain_raw.requires_grad_(False)

    losses = _train_encoder(model, source_windows, x_calib, config)

    if config.fine_tune:
        fine_tune_on_calibration(
            model,
            x_calib,
            y_calib_share,
            epochs_per_stage=config.fine_tune_epochs_per_stage,
            lr=config.fine_tune_lr,
        )

    dominant_idx = recompute_dominant_variable(model, x_calib)
    reweight_lt_mwkc_alpha(model, dominant_idx)

    with torch.no_grad():
        renew_share_pred_calib, _ = model(x_calib)
        renew_share_pred_eval, _ = model(x_eval)

    renew_factor, nonrenew_factor = config.target.resolve_emission_factors()
    ci_phys_calib = cif_from_shares(renew_share_pred_calib.numpy(), renew_factor, nonrenew_factor)
    ci_phys_eval = cif_from_shares(renew_share_pred_eval.numpy(), renew_factor, nonrenew_factor)

    result: dict[str, Any] = {
        "target": config.target.name,
        "sources": list(source_windows.keys()),
        "num_channels": config.num_channels,
        "emission_factors": (renew_factor, nonrenew_factor),
        "final_train_loss": losses[-1] if losses else None,
        "renew_share_pred_eval": renew_share_pred_eval.numpy(),
        "ci_pred_physics_eval": ci_phys_eval,
        "has_ground_truth": ci_true is not None,
    }

    if ci_true is None:
        return result

    residual_head = ResidualCorrectionHead(input_dim=1, hidden_dim=8)
    calib_features = torch.tensor(renew_share_pred_calib.numpy().reshape(-1, 1), dtype=torch.float32)
    calib_targets = torch.tensor((ci_true_calib - ci_phys_calib).reshape(-1), dtype=torch.float32)
    fit_residual_head(residual_head, calib_features, calib_targets, epochs=100, lr=1e-2)

    with torch.no_grad():
        delta_calib = residual_head(calib_features).numpy().reshape(ci_phys_calib.shape)
        eval_features = torch.tensor(renew_share_pred_eval.numpy().reshape(-1, 1), dtype=torch.float32)
        delta_eval = residual_head(eval_features).numpy().reshape(ci_phys_eval.shape)
    ci_corrected_calib = ci_phys_calib + delta_calib
    ci_corrected_eval = ci_phys_eval + delta_eval

    scores = compute_nonconformity_scores(ci_true_calib.reshape(-1), ci_corrected_calib.reshape(-1))
    halfwidth = conformal_interval_halfwidth(scores, coverage=coverage)
    lower, upper = predict_with_interval(ci_corrected_eval.reshape(-1), halfwidth)
    coverage_emp = empirical_coverage(ci_true_eval.reshape(-1), lower, upper)

    last_observed_share = x_eval[:, -1, 0].numpy()
    persistence_share_pred = np.repeat(last_observed_share[:, None], config.horizon, axis=1)
    ci_persistence_eval = cif_from_shares(persistence_share_pred, renew_factor, nonrenew_factor)

    corrected_mae = mae(ci_true_eval.reshape(-1), ci_corrected_eval.reshape(-1))
    persistence_mae = mae(ci_true_eval.reshape(-1), ci_persistence_eval.reshape(-1))
    result.update(
        {
            "ci_pred_corrected_eval": ci_corrected_eval,
            "physics_only_mae": mae(ci_true_eval.reshape(-1), ci_phys_eval.reshape(-1)),
            "corrected_mae": corrected_mae,
            "persistence_mae": persistence_mae,
            "corrected_vs_persistence_pct": (corrected_mae - persistence_mae) / persistence_mae * 100,
            "conformal_halfwidth": halfwidth,
            "empirical_coverage": coverage_emp,
        }
    )
    return result
