"""SA1-as-target-region ablation: with QLD1/NSW1/VIC1 as MLDG source regions and SA1 held
out entirely as the unseen deployment target, compares six variants that incrementally turn
on the four SA1-transfer-failure mitigations tried in this round: REG/NEG generation-magnitude
channels, a real temperature-anomaly covariate, volatility-conditioned persistence-skip gating
(Direction A), and MLDG domain-adaptive loss weighting (Direction B).

Ground truth is AEMO's real measured `cif_real_gco2_per_kwh` column (NOT a physics-formula
reconstruction from RenewShare) -- the project's real-data-only mandate calls for the most
direct real signal available, and NEMED's generator-level measurement is strictly more direct
than reconstructing CIF from the two-category renewable/non-renewable share approximation used
elsewhere in the pipeline for cases where no directly-measured column exists.

Run with: PYTHONPATH=src python scripts/sa1_ablation.py
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from transcif.calibration.dominant_reweight import recompute_dominant_variable, reweight_lt_mwkc_alpha
from transcif.data.loaders import load_region_hourly_csv, load_region_temperature_csv, load_region_windows, merge_temperature
from transcif.evaluation.metrics import mae
from transcif.models.encoder import DomainInvariantEncoder, PersistenceSkipEncoder
from transcif.physics.cif import cif_from_shares, get_emission_factors
from transcif.physics.residual import ResidualCorrectionHead, fit_residual_head
from transcif.training.consistency import consistency_loss
from transcif.training.train_multi_source import train_multi_source_mldg

DATA_DIR = "/tmp/nemed_output"
SOURCE_REGIONS = ["QLD1", "NSW1", "VIC1"]
TARGET_REGION = "SA1"
REGION_TO_FACTOR_CODE = {"QLD1": "AU_QLD", "NSW1": "AU_NSW", "VIC1": "AU_VIC", "SA1": "AU_SA"}

SEQ_LEN, HORIZON, STRIDE = 48, 12, 6
LT_FEATURE_DIM, CV_FEATURE_DIM = 16, 8
MLDG_EPOCHS = 80
CALIB_FRACTION = 0.7
SEED = 42


def train_multi_source_mldg_unweighted(
    encoder: nn.Module,
    source_windows: dict,
    epochs: int,
    outer_lr: float = 5e-3,
    inner_lr: float = 1e-2,
    meta_test_weight: float = 1.0,
    consistency_weight: float = 0.05,
) -> list:
    """Direction-B-off ablation switch: mirrors the pre-Task-35 MLDG implementation (a flat
    concatenated-batch MSE over the pooled meta-train regions) rather than production
    train_multi_source_mldg's domain-weighted per-region average."""
    regions = list(source_windows.keys())
    optimizer = torch.optim.Adam(encoder.parameters(), lr=outer_lr)
    mse_loss = nn.MSELoss()
    losses = []

    for _ in range(epochs):
        meta_test_region = random.choice(regions)
        meta_train_regions = [r for r in regions if r != meta_test_region]

        x_meta_train = torch.cat([source_windows[r][0] for r in meta_train_regions], dim=0)
        y_meta_train = torch.cat([source_windows[r][1] for r in meta_train_regions], dim=0)
        x_meta_test, y_meta_test = source_windows[meta_test_region]

        params = dict(encoder.named_parameters())
        buffers = dict(encoder.named_buffers())
        trainable_names = [name for name, p in params.items() if p.requires_grad]

        pred_meta_train, _ = functional_call(encoder, (params, buffers), (x_meta_train,))
        meta_train_loss = mse_loss(pred_meta_train, y_meta_train)
        meta_train_loss = meta_train_loss + consistency_weight * consistency_loss(encoder, x_meta_train)

        grads = torch.autograd.grad(
            meta_train_loss, [params[name] for name in trainable_names], create_graph=True, allow_unused=True
        )
        grad_by_name = dict(zip(trainable_names, grads))
        updated_params = {
            name: p if grad_by_name.get(name) is None else p - inner_lr * grad_by_name[name]
            for name, p in params.items()
        }

        pred_meta_test, _ = functional_call(encoder, (updated_params, buffers), (x_meta_test,))
        meta_test_loss = mse_loss(pred_meta_test, y_meta_test)

        total_loss = meta_train_loss + meta_test_weight * meta_test_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    return losses


def extract_ci_true_windows(df, seq_len: int, horizon: int, stride: int) -> np.ndarray:
    """Mirrors build_sliding_windows' exact (start, window, stride) indexing so the real
    measured cif_real_gco2_per_kwh horizon slice lines up 1:1 with each (x, y) window --
    build_sliding_windows itself never retains this column since it isn't a model input."""
    window = seq_len + horizon
    ci_real = df["cif_real_gco2_per_kwh"].to_numpy()
    windows = [ci_real[start + seq_len : start + window] for start in range(0, len(ci_real) - window + 1, stride)]
    return np.stack(windows)


def load_source_and_target(include_generation: bool, include_temperature: bool):
    source_windows = {}
    for region in SOURCE_REGIONS:
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

    target_csv = f"{DATA_DIR}/nem_2023_hourly_{TARGET_REGION}.csv"
    target_temp_path = f"{DATA_DIR}/temperature_2023_{TARGET_REGION}.csv" if include_temperature else None
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


def run_variant(
    name: str,
    num_channels: int,
    include_generation: bool,
    include_temperature: bool,
    gate_conditioning_on: bool,
    mldg_weighted_on: bool,
) -> dict:
    source_windows, x_target, y_target_share, ci_true_target = load_source_and_target(
        include_generation, include_temperature
    )

    n = x_target.shape[0]
    split = int(n * CALIB_FRACTION)
    x_calib, x_eval = x_target[:split], x_target[split:]
    ci_true_calib, ci_true_eval = ci_true_target[:split], ci_true_target[split:]

    torch.manual_seed(SEED)
    base = DomainInvariantEncoder(
        num_variables=num_channels, horizon=HORIZON, lt_feature_dim=LT_FEATURE_DIM, cv_feature_dim=CV_FEATURE_DIM
    )
    model = PersistenceSkipEncoder(base)
    if not gate_conditioning_on:
        model.volatility_gain_raw.requires_grad_(False)

    train_fn = train_multi_source_mldg if mldg_weighted_on else train_multi_source_mldg_unweighted
    losses = train_fn(model, source_windows, epochs=MLDG_EPOCHS)

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
        "name": name,
        "final_train_loss": losses[-1],
        "physics_only_mae": physics_only_mae,
        "corrected_mae": corrected_mae,
        "persistence_mae": persistence_mae,
        "corrected_vs_persistence_pct": (corrected_mae - persistence_mae) / persistence_mae * 100,
    }


VARIANTS = [
    dict(name="baseline", num_channels=2, include_generation=False, include_temperature=False,
         gate_conditioning_on=False, mldg_weighted_on=False),
    dict(name="+REG/NEG", num_channels=4, include_generation=True, include_temperature=False,
         gate_conditioning_on=False, mldg_weighted_on=False),
    dict(name="+温度协变量(C1)", num_channels=3, include_generation=False, include_temperature=True,
         gate_conditioning_on=False, mldg_weighted_on=False),
    dict(name="+门控条件化(A)", num_channels=2, include_generation=False, include_temperature=False,
         gate_conditioning_on=True, mldg_weighted_on=False),
    dict(name="+MLDG域自适应加权(B)", num_channels=2, include_generation=False, include_temperature=False,
         gate_conditioning_on=False, mldg_weighted_on=True),
    dict(name="全部组合", num_channels=5, include_generation=True, include_temperature=True,
         gate_conditioning_on=True, mldg_weighted_on=True),
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
