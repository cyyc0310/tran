#!/usr/bin/env python3
"""Controlled Neural ODE and grey-box baselines on InTaSet M1<->M2.

The script uses the same file-level cross-configuration split as the linear
ARX/SINDYc benchmark. Candidate width is selected on a chronological tail of
the source training file; final models are refit on the complete source file.
Only target test states at 1/5/10-second block starts initialize evaluation
rollouts, while the recorded 0.1-second pump path is supplied throughout.

The author InTaSet Neural ODE implementation is not publicly available. The
controlled Neural ODE and additive four-tank hybrid model here are independent
implementations, not reproductions of the authors' code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import scipy
import sklearn
import torch
from torch import nn
from scipy.optimize import least_squares

from intaset_causal_transfer_benchmark_v1 import (
    CONFIG_PATHS,
    DATA_PATH,
    MANIFEST_PATH,
    OUTPUT_DIR,
    STATE_NAMES,
    STRIDES,
    load_table,
)


DT_SECONDS = 0.1
TRAIN_VALIDATION_FRACTION = 0.8
TRAINING_HORIZON_STEPS = 10
HIDDEN_WIDTHS = (16, 32)
TRAINING_EPOCHS = 40
BATCH_SIZE = 64
BATCHES_PER_EPOCH = 4
RANDOM_SEED = 20261102
DEFAULT_SEEDS = (20261102, 20261103, 20261104)
HYBRID_CORRELATION_WEIGHT = 0.01
MAX_ABS_LEVEL_CM = 1e4
CALIBRATION_BUDGETS_SECONDS = (60, 300)
CALIBRATION_GAIN_BOUNDS = (0.1, 5.0)
DEVICE = torch.device("cpu")
MODEL_FAMILIES = ("controlled_neural_ode", "four_tank_hybrid_neural_ode", "four_tank_physics_only")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inverse_softplus(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return np.log(np.expm1(value))


class ControlledDynamics(nn.Module):
    """Continuous vector field with an optional four-tank physics component."""

    def __init__(
        self,
        family: str,
        width: int,
        state_mean: np.ndarray,
        state_scale: np.ndarray,
        input_mean: np.ndarray,
        input_scale: np.ndarray,
        rate_scale: np.ndarray,
    ):
        super().__init__()
        self.family = family
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_scale", torch.as_tensor(state_scale, dtype=torch.float32))
        self.register_buffer("input_mean", torch.as_tensor(input_mean, dtype=torch.float32))
        self.register_buffer("input_scale", torch.as_tensor(input_scale, dtype=torch.float32))
        self.register_buffer("rate_scale", torch.as_tensor(rate_scale, dtype=torch.float32))

        if family != "four_tank_physics_only":
            self.residual = nn.Sequential(
                nn.Linear(6, width),
                nn.Tanh(),
                nn.Linear(width, 4),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        else:
            self.residual = None

        if family in ("four_tank_hybrid_neural_ode", "four_tank_physics_only"):
            # Reduced grey-box structure: lower-tank drains, upper-tank drains,
            # upper-to-lower transfers, and the two pump-to-tank paths.
            initial = np.asarray([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.2, 0.2, 0.2, 0.2])
            self.raw_physics = nn.Parameter(torch.as_tensor(inverse_softplus(initial), dtype=torch.float32))
        else:
            self.register_parameter("raw_physics", None)

    def physics_rate(self, state_cm: torch.Tensor, pump_percent: torch.Tensor) -> torch.Tensor:
        if self.raw_physics is None:
            raise RuntimeError("physics component requested for a black-box model")
        coefficients = torch.nn.functional.softplus(self.raw_physics)
        d1, d2, d3, d4, t3, t4, k1, k2, k3, k4 = coefficients
        h = torch.clamp(state_cm, min=0.0)
        root = torch.sqrt(h + 1e-8)
        u1 = pump_percent[:, 0] / 100.0
        u2 = pump_percent[:, 1] / 100.0
        return torch.stack(
            (
                -d1 * root[:, 0] + t3 * root[:, 2] + k1 * u1,
                -d2 * root[:, 1] + t4 * root[:, 3] + k2 * u2,
                -d3 * root[:, 2] + k3 * u2,
                -d4 * root[:, 3] + k4 * u1,
            ),
            dim=1,
        )

    def rate(self, state_cm: torch.Tensor, pump_percent: torch.Tensor) -> torch.Tensor:
        if self.residual is None:
            return self.physics_rate(state_cm, pump_percent)
        state_scaled = (state_cm - self.state_mean) / self.state_scale
        input_scaled = (pump_percent - self.input_mean) / self.input_scale
        residual_rate = self.residual(torch.cat((state_scaled, input_scaled), dim=1)) * self.rate_scale
        if self.raw_physics is None:
            return residual_rate
        return self.physics_rate(state_cm, pump_percent) + residual_rate

    def rk4_step(
        self,
        state_cm: torch.Tensor,
        pump_percent: torch.Tensor,
        dt: float,
        rate_gain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def vector_field(state: torch.Tensor) -> torch.Tensor:
            value = self.rate(state, pump_percent)
            return value if rate_gain is None else value * rate_gain

        k1 = vector_field(state_cm)
        k2 = vector_field(state_cm + 0.5 * dt * k1)
        k3 = vector_field(state_cm + 0.5 * dt * k2)
        k4 = vector_field(state_cm + dt * k3)
        return state_cm + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4_physics_step(self, state_cm: torch.Tensor, pump_percent: torch.Tensor, dt: float) -> torch.Tensor:
        def f(state: torch.Tensor) -> torch.Tensor:
            return self.physics_rate(state, pump_percent)

        k1 = f(state_cm)
        k2 = f(state_cm + 0.5 * dt * k1)
        k3 = f(state_cm + 0.5 * dt * k2)
        k4 = f(state_cm + dt * k3)
        return state_cm + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def training_scales(table: np.ndarray, fit_end: int) -> tuple[np.ndarray, ...]:
    states = table[:fit_end, 1:5]
    inputs = table[:fit_end, 5:7]
    rate = np.diff(states, axis=0) / DT_SECONDS
    state_mean = states.mean(axis=0)
    state_scale = np.std(states, axis=0).clip(1e-6)
    input_mean = inputs.mean(axis=0)
    input_scale = np.std(inputs, axis=0).clip(1e-6)
    rate_scale = np.std(rate, axis=0).clip(1e-6)
    return state_mean, state_scale, input_mean, input_scale, rate_scale


def make_model(family: str, width: int, table: np.ndarray, fit_end: int) -> ControlledDynamics:
    scales = training_scales(table, fit_end)
    return ControlledDynamics(family, width, *scales).to(DEVICE)


def tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=DEVICE)


def rollout_training_batch(
    model: ControlledDynamics,
    initial_states: torch.Tensor,
    input_path: torch.Tensor,
    *,
    return_physics: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    state = initial_states
    physics_state = initial_states
    predictions, physics_predictions = [], []
    for offset in range(input_path.shape[1]):
        pump = input_path[:, offset, :]
        state = model.rk4_step(state, pump, DT_SECONDS)
        predictions.append(state)
        if return_physics:
            physics_state = model.rk4_physics_step(physics_state, pump, DT_SECONDS)
            physics_predictions.append(physics_state)
    pred = torch.stack(predictions, dim=1)
    phys = torch.stack(physics_predictions, dim=1) if return_physics else None
    return pred, phys


def sample_windows(
    table: np.ndarray,
    fit_end: int,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    last_start = fit_end - TRAINING_HORIZON_STEPS - 1
    starts = rng.integers(0, last_start + 1, size=batch_size)
    states, inputs = table[:, 1:5], table[:, 5:7]
    initial = states[starts]
    input_path = np.stack([inputs[starts + step] for step in range(TRAINING_HORIZON_STEPS)], axis=1)
    target_path = np.stack([states[starts + step + 1] for step in range(TRAINING_HORIZON_STEPS)], axis=1)
    return tensor(initial), tensor(input_path), tensor(target_path)


def train_model(
    model: ControlledDynamics,
    table: np.ndarray,
    fit_end: int,
    *,
    seed: int,
    epochs: int = TRAINING_EPOCHS,
) -> list[float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    state_scale = model.state_scale
    losses = []
    for _epoch in range(epochs):
        epoch_losses = []
        model.train()
        for _batch in range(BATCHES_PER_EPOCH):
            initial, input_path, target = sample_windows(table, fit_end, rng, BATCH_SIZE)
            optimizer.zero_grad(set_to_none=True)
            prediction, physics_prediction = rollout_training_batch(
                model,
                initial,
                input_path,
                return_physics=(model.family == "four_tank_hybrid_neural_ode"),
            )
            loss = torch.mean(((prediction - target) / state_scale) ** 2)
            if physics_prediction is not None:
                pred_by_trajectory = prediction.reshape(prediction.shape[0], -1)
                physics_by_trajectory = physics_prediction.reshape(physics_prediction.shape[0], -1)
                numerator = torch.sum(pred_by_trajectory * physics_by_trajectory, dim=1)
                denominator = torch.linalg.vector_norm(pred_by_trajectory, dim=1) * torch.linalg.vector_norm(physics_by_trajectory, dim=1)
                similarity = torch.mean(numerator / (denominator + 1e-8))
                loss = loss - HYBRID_CORRELATION_WEIGHT * similarity
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training objective for {model.family}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return losses


@torch.no_grad()
def evaluate_one_step(model: ControlledDynamics, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    states, inputs = table[:, 1:5], table[:, 5:7]
    prediction = model.rk4_step(tensor(states[:-1]), tensor(inputs[:-1]), DT_SECONDS).cpu().numpy()
    return prediction, states[1:]


@torch.no_grad()
def rollout_blocks(
    model: ControlledDynamics,
    table: np.ndarray,
    stride: int,
    rate_gain: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    model.eval()
    states, inputs = table[:, 1:5], table[:, 5:7]
    blocks = (len(table) - 1) // stride
    starts = np.arange(blocks, dtype=int) * stride
    state = tensor(states[starts])
    gain_tensor = None if rate_gain is None else tensor(np.asarray(rate_gain, dtype=float))
    prediction_steps = []
    diverged = torch.zeros(blocks, dtype=torch.bool, device=DEVICE)
    for offset in range(stride):
        pump = tensor(inputs[starts + offset])
        state = model.rk4_step(state, pump, DT_SECONDS, rate_gain=gain_tensor)
        bad = (~torch.isfinite(state).all(dim=1)) | (torch.max(torch.abs(state), dim=1).values > MAX_ABS_LEVEL_CM)
        diverged |= bad
        state = torch.where(diverged[:, None], torch.zeros_like(state), state)
        prediction_steps.append(state)
    prediction = torch.stack(prediction_steps, dim=1).cpu().numpy()
    diverged_np = diverged.cpu().numpy()
    prediction[diverged_np, :, :] = np.nan
    target_indices = starts[:, None] + np.arange(1, stride + 1)[None, :]
    truth = states[target_indices]
    return prediction.reshape(-1, 4), truth.reshape(-1, 4), blocks, int(np.sum(diverged_np))


def sparse_calibration_blocks(table: np.ndarray, stride: int, budget_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    final_index = int(round(budget_seconds / DT_SECONDS))
    if final_index >= len(table) or final_index % stride:
        raise ValueError(f"calibration budget {budget_seconds}s is unavailable or misaligned for stride {stride}")
    if not np.isclose(table[final_index, 0] - table[0, 0], budget_seconds, atol=1e-6):
        raise ValueError("target training timestamps do not align with requested calibration budget")
    starts = np.arange(0, final_index, stride, dtype=int)
    return starts, starts + stride


def calibration_residual(
    gain: np.ndarray,
    model: ControlledDynamics,
    table: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    stride: int,
    state_scale: np.ndarray,
) -> np.ndarray:
    states, inputs = table[:, 1:5], table[:, 5:7]
    state = tensor(states[starts])
    input_paths = np.stack([inputs[start : start + stride] for start in starts], axis=0)
    gain_tensor = tensor(gain)
    with torch.no_grad():
        for offset in range(stride):
            state = model.rk4_step(state, tensor(input_paths[:, offset, :]), DT_SECONDS, rate_gain=gain_tensor)
            if not torch.isfinite(state).all() or torch.max(torch.abs(state)) > MAX_ABS_LEVEL_CM:
                return np.full(len(starts) * 4, 1e6, dtype=float)
    prediction = state.cpu().numpy()
    return ((prediction - states[ends]) / state_scale).reshape(-1)


def fit_sparse_rate_gain(
    model: ControlledDynamics,
    target_train: np.ndarray,
    stride: int,
    budget_seconds: int,
    state_scale: np.ndarray,
) -> dict:
    starts, ends = sparse_calibration_blocks(target_train, stride, budget_seconds)
    result = least_squares(
        calibration_residual,
        x0=np.ones(4),
        bounds=(np.full(4, CALIBRATION_GAIN_BOUNDS[0]), np.full(4, CALIBRATION_GAIN_BOUNDS[1])),
        args=(model, target_train, starts, ends, stride, state_scale),
        method="trf",
        loss="linear",
        # Models use float32 tensors; scipy's default perturbation near 1.0
        # can round away when converted to float32, yielding a false zero Jacobian.
        diff_step=1e-3,
        max_nfev=150,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    tolerance = singular_values[0] * max(result.jac.shape) * np.finfo(float).eps
    rank = int(np.sum(singular_values > tolerance))
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    return {
        "gain": result.x,
        "transition_blocks": int(len(starts)),
        "sparse_state_observations": int(len(np.unique(np.r_[starts, ends]))),
        "fit_success": bool(result.success),
        "fit_status": int(result.status),
        "endpoint_residual_sum_squares_normalized": float(2.0 * result.cost),
        "jacobian_rank": rank,
        "jacobian_condition_2": condition,
        "gain_at_lower_bound": [bool(value <= CALIBRATION_GAIN_BOUNDS[0] + 1e-6) for value in result.x],
        "gain_at_upper_bound": [bool(value >= CALIBRATION_GAIN_BOUNDS[1] - 1e-6) for value in result.x],
    }


def model_selection_score(model: ControlledDynamics, validation: np.ndarray, state_scale: np.ndarray) -> tuple[float, list[dict]]:
    scores = []
    per_horizon = []
    for stride in STRIDES:
        prediction, truth, blocks, diverged = rollout_blocks(model, validation, stride)
        finite_rows = np.isfinite(prediction).all(axis=1)
        if diverged or not np.any(finite_rows):
            return float("inf"), per_horizon + [{"stride_samples": stride, "diverged_blocks": diverged, "validation_normalized_rmse": None}]
        normalized = (prediction[finite_rows] - truth[finite_rows]) / state_scale
        score = float(np.sqrt(np.mean(normalized**2)))
        scores.append(score)
        per_horizon.append({
            "stride_samples": stride,
            "horizon_seconds": stride * DT_SECONDS,
            "blocks": blocks,
            "diverged_blocks": diverged,
            "validation_normalized_rmse": score,
        })
    return float(np.mean(scores)), per_horizon


def select_and_refit(source: str, source_train: np.ndarray, seed_base: int) -> tuple[dict[str, ControlledDynamics], dict]:
    split = int(len(source_train) * TRAIN_VALIDATION_FRACTION)
    fitting = source_train[:split]
    validation = source_train[split - 1 :]
    state_mean, state_scale, input_mean, input_scale, rate_scale = training_scales(fitting, len(fitting))
    selected, selection = {}, {}
    for family in MODEL_FAMILIES:
        candidates = []
        widths = HIDDEN_WIDTHS if family != "four_tank_physics_only" else (0,)
        for width in widths:
            family_seed = {
                "controlled_neural_ode": 11,
                "four_tank_hybrid_neural_ode": 23,
                "four_tank_physics_only": 37,
            }[family]
            candidate_seed = seed_base + (0 if source == "M1" else 1000) + family_seed + width
            torch.manual_seed(candidate_seed)
            model = ControlledDynamics(family, max(width, 1), state_mean, state_scale, input_mean, input_scale, rate_scale).to(DEVICE)
            losses = train_model(model, fitting, len(fitting), seed=candidate_seed)
            score, horizon_scores = model_selection_score(model, validation, state_scale)
            candidates.append({
                "width": int(width),
                "training_epochs": TRAINING_EPOCHS,
                "final_training_objective": losses[-1],
                "mean_validation_normalized_rollout_rmse_1_5_10s": score if np.isfinite(score) else None,
                "validation_horizons": horizon_scores,
                "eligible": bool(np.isfinite(score)),
                "candidate_seed": int(candidate_seed),
            })
        eligible = [item for item in candidates if item["eligible"]]
        if not eligible:
            raise RuntimeError(f"all {family} candidates diverged on {source} source validation")
        best = min(eligible, key=lambda item: item["mean_validation_normalized_rollout_rmse_1_5_10s"])
        best_width = best["width"]
        refit_seed = seed_base + (0 if source == "M1" else 1000) + 10000 + MODEL_FAMILIES.index(family)
        torch.manual_seed(refit_seed)
        final_model = ControlledDynamics(
            family,
            max(best_width, 1),
            state_mean,
            state_scale,
            input_mean,
            input_scale,
            rate_scale,
        ).to(DEVICE)
        final_losses = train_model(final_model, source_train, len(source_train), seed=refit_seed)
        selected[family] = final_model
        selection[family] = {
            "selected_width": int(best_width),
            "selection_rule": "minimum equal-weight normalized free-rollout RMSE over 1/5/10-second source chronological validation blocks; any diverged horizon is ineligible",
            "candidate_results": candidates,
            "refit_seed": int(refit_seed),
            "refit_epochs": TRAINING_EPOCHS,
            "initial_refit_training_objective": final_losses[0],
            "final_refit_training_objective": final_losses[-1],
        }
    return selected, {
        "source_configuration": source,
        "fit_rows_before_refit": int(split),
        "validation_rows": int(len(validation)),
        "final_refit_rows": int(len(source_train)),
        "source_state_mean_cm": state_mean.tolist(),
        "source_state_scale_cm": state_scale.tolist(),
        "source_input_mean_percent": input_mean.tolist(),
        "source_input_scale_percent": input_scale.tolist(),
        "source_rate_scale_cm_per_second": rate_scale.tolist(),
        "models": selection,
    }


def metrics_for_prediction(
    prediction: np.ndarray,
    truth: np.ndarray,
    scale: np.ndarray,
    *,
    direction: str,
    task: str,
    horizon_seconds: float,
    method: str,
    n_blocks: int,
    diverged_blocks: int,
) -> list[dict]:
    rows = []
    for channels, name in [(np.arange(4), "ALL"), *[(np.array([i]), state) for i, state in enumerate(STATE_NAMES)]]:
        pred = prediction[:, channels]
        actual = truth[:, channels]
        finite = np.isfinite(pred) & np.isfinite(actual)
        p, y = pred[finite], actual[finite]
        err = p - y
        ch_scale = np.broadcast_to(scale[channels], pred.shape)[finite]
        rows.append({
            "direction": direction,
            "task": task,
            "horizon_seconds": float(horizon_seconds),
            "method": method,
            "state": name,
            "n_scored_values": int(pred.size),
            "n_finite_values": int(np.sum(finite)),
            "finite_prediction_fraction": float(np.mean(finite)) if pred.size else float("nan"),
            "n_blocks": int(n_blocks),
            "diverged_blocks": int(diverged_blocks),
            "rmse_cm": float(np.sqrt(np.mean(err**2))) if len(err) else float("nan"),
            "mae_cm": float(np.mean(np.abs(err))) if len(err) else float("nan"),
            "rmse_over_source_train_std": float(np.sqrt(np.mean((err / ch_scale) ** 2))) if len(err) else float("nan"),
            "predicted_below_zero_fraction": float(np.mean(p < 0.0)) if len(p) else float("nan"),
            "minimum_predicted_level_cm": float(np.min(p)) if len(p) else float("nan"),
        })
    return rows


def run(output_dir: Path, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide one or more unique integer training seeds")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    archive_bytes = DATA_PATH.read_bytes()
    archive_md5 = hashlib.md5(archive_bytes).hexdigest()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if archive_md5 != manifest["zenodo_md5_expected"] or archive_sha256 != manifest["sha256_actual"]:
        raise ValueError("InTaSet archive does not match the recorded source hashes")

    with zipfile.ZipFile(DATA_PATH) as archive:
        tables = {
            config: {split: load_table(archive, member) for split, member in paths.items()}
            for config, paths in CONFIG_PATHS.items()
        }

    fit_seconds = 0.0
    calibration_seconds = 0.0
    rows, calibration_records, tuning_by_seed = [], [], {}
    for seed_base in seeds:
        fit_started = time.perf_counter()
        fitted, tuning = {}, {}
        for config in ("M1", "M2"):
            fitted[config], tuning[config] = select_and_refit(config, tables[config]["train"], seed_base)
        fit_seconds += time.perf_counter() - fit_started
        tuning_by_seed[str(seed_base)] = tuning

        for target in ("M1", "M2"):
            source = "M2" if target == "M1" else "M1"
            direction = f"{source}->{target}"
            target_train = tables[target]["train"]
            test = tables[target]["test"]
            scale = np.asarray(tuning[source]["source_state_scale_cm"], dtype=float)
            for family, model in fitted[source].items():
                method = f"source_{family}"
                one_pred, one_truth = evaluate_one_step(model, test)
                one_step_diverged = int(np.sum(~np.isfinite(one_pred).all(axis=1)))
                one_step_rows = metrics_for_prediction(
                    one_pred, one_truth, scale,
                    direction=direction,
                    task="teacher_forced_one_step_diagnostic",
                    horizon_seconds=DT_SECONDS,
                    method=method,
                    n_blocks=0,
                    diverged_blocks=one_step_diverged,
                )
                for row in one_step_rows:
                    row.update(training_seed=seed_base, calibration_budget_seconds=0)
                rows.extend(one_step_rows)

                for stride in STRIDES:
                    prediction, truth, blocks, diverged = rollout_blocks(model, test, stride)
                    baseline_rows = metrics_for_prediction(
                        prediction, truth, scale,
                        direction=direction,
                        task="causal_free_rollout_given_recorded_pump_path",
                        horizon_seconds=stride * DT_SECONDS,
                        method=method,
                        n_blocks=blocks,
                        diverged_blocks=diverged,
                    )
                    for row in baseline_rows:
                        row.update(training_seed=seed_base, calibration_budget_seconds=0)
                    rows.extend(baseline_rows)

                    for budget_seconds in CALIBRATION_BUDGETS_SECONDS:
                        calibration_started = time.perf_counter()
                        fit_record = fit_sparse_rate_gain(
                            model,
                            target_train,
                            stride,
                            budget_seconds,
                            scale,
                        )
                        calibration_seconds += time.perf_counter() - calibration_started
                        calibration_records.append({
                            "training_seed": seed_base,
                            "direction": direction,
                            "source_model": method,
                            "observation_interval_seconds": stride * DT_SECONDS,
                            "calibration_budget_seconds": budget_seconds,
                            **{key: value for key, value in fit_record.items() if key != "gain"},
                            "gain_h1": float(fit_record["gain"][0]),
                            "gain_h2": float(fit_record["gain"][1]),
                            "gain_h3": float(fit_record["gain"][2]),
                            "gain_h4": float(fit_record["gain"][3]),
                        })
                        prediction, truth, blocks, diverged = rollout_blocks(
                            model,
                            test,
                            stride,
                            rate_gain=fit_record["gain"],
                        )
                        adapted_rows = metrics_for_prediction(
                            prediction, truth, scale,
                            direction=direction,
                            task="causal_free_rollout_after_sparse_target_train_calibration_given_recorded_pump_path",
                            horizon_seconds=stride * DT_SECONDS,
                            method=method,
                            n_blocks=blocks,
                            diverged_blocks=diverged,
                        )
                        for row in adapted_rows:
                            row.update(
                                training_seed=seed_base,
                                calibration_budget_seconds=budget_seconds,
                            )
                        rows.extend(adapted_rows)

    detail = pd.DataFrame(rows)
    calibration_detail = pd.DataFrame(calibration_records)
    summary = (
        detail.groupby(
            ["direction", "task", "horizon_seconds", "method", "state", "calibration_budget_seconds"],
            as_index=False,
        )
        .agg(
            training_seeds=("training_seed", "nunique"),
            rmse_cm_mean=("rmse_cm", "mean"),
            rmse_cm_sd_between_seeds=("rmse_cm", "std"),
            mae_cm_mean=("mae_cm", "mean"),
            normalized_rmse_mean=("rmse_over_source_train_std", "mean"),
            normalized_rmse_sd_between_seeds=("rmse_over_source_train_std", "std"),
            predicted_below_zero_fraction_mean=("predicted_below_zero_fraction", "mean"),
            finite_prediction_fraction_mean=("finite_prediction_fraction", "mean"),
            diverged_blocks_mean=("diverged_blocks", "mean"),
        )
        .sort_values(["task", "direction", "horizon_seconds", "state", "normalized_rmse_mean"])
    )
    metadata = {
        "experiment": "InTaSet M1<->M2 controlled Neural ODE and additive four-tank hybrid baselines",
        "source_archive": {
            "path": str(DATA_PATH.relative_to(DATA_PATH.parents[2])),
            "zenodo_md5": archive_md5,
            "sha256": archive_sha256,
            "recorded_manifest_sha256": sha256(MANIFEST_PATH),
        },
        "task_contract": {
            "source_fit_and_model_selection": "Each source configuration's complete training file is split chronologically 80/20 for width selection, then the chosen model is refit on the complete source training file.",
            "target_state_observation_intervals_seconds": [stride * DT_SECONDS for stride in STRIDES],
            "target_pump_input_interval_seconds": DT_SECONDS,
            "target_test": "Target testing files are used only for teacher-forced diagnostics and final blockwise causal rollouts. The recorded pump path is supplied throughout each rollout.",
            "target_calibration": "Calibration uses only the target configuration's separate training file, only at sparse interval endpoints, and only for the first 60 or 300 seconds. Evaluation remains on the separate target testing file.",
            "calibration_parameter_budget": "Four positive per-state vector-field rate multipliers are fitted; all source-trained Neural ODE and hybrid model weights remain frozen.",
            "offline_endpoint_task": "Not included in this neural-model runner; existing endpoint-interpolation scores remain a separate offline task in the linear benchmark.",
            "input_availability": "Recorded target pump paths condition the evaluation. This is a conditional prediction benchmark, not a forecast of unknown future pump commands or a closed-loop stability test.",
        },
        "models": {
            "controlled_neural_ode": "Independent controlled Neural ODE: a one-hidden-layer tanh MLP parameterizes d(h)/dt = f(h,u); the measured pump signal is held constant over each 0.1-second integration interval; RK4 is used for training and inference.",
            "four_tank_hybrid_neural_ode": "Independent additive grey-box model: d(h)/dt = f_reduced_four_tank(h,u;theta) + f_neural(h,u). Positive drainage, upper-to-lower transfer, and pump path coefficients parameterize the reduced phenomenological structure. Source training adds a per-trajectory cosine-correlation regularizer inspired by Osaka et al.; the implementation is not the InTaSet authors' code.",
            "four_tank_physics_only": "Same reduced four-tank ODE without a neural residual; positive coefficients are fit from source training trajectories.",
            "author_neural_ode_status": "The published InTaSet abstract confirms an author Neural ODE baseline, but the public author repository contains only the data archive and README. The present methods are independent implementations and are not claimed to reproduce the unavailable author model.",
        },
        "selection_and_training": {
            "candidate_hidden_widths": list(HIDDEN_WIDTHS),
            "training_horizon_seconds": TRAINING_HORIZON_STEPS * DT_SECONDS,
            "epochs": TRAINING_EPOCHS,
            "batch_size": BATCH_SIZE,
            "batches_per_epoch": BATCHES_PER_EPOCH,
            "optimizer": "Adam, learning rate 1e-3, weight decay 1e-5, gradient norm clipped at 5.",
            "hybrid_correlation_regularizer_weight": HYBRID_CORRELATION_WEIGHT,
            "selection_rule": "Minimum equal-weight normalized 1/5/10-second free-rollout RMSE on the source chronological validation tail; candidates with any diverged horizon are ineligible.",
            "training_seeds": list(seeds),
            "calibration_budgets_seconds": list(CALIBRATION_BUDGETS_SECONDS),
            "calibration_gain_bounds": list(CALIBRATION_GAIN_BOUNDS),
        },
        "tuning_by_seed": tuning_by_seed,
        "limitations": [
            "Each configuration has one training and one testing trajectory; training seeds quantify initialization sensitivity only, while block scores are not independent hardware replicates.",
            "The future pump path is taken from the recorded target test trajectory, so the task is conditional prediction with known recorded input.",
            "The reduced physical equation is a transparent grey-box approximation; its parameters and missing terms are not independently established from hardware documentation.",
            "Target calibration changes only four per-state vector-field gains; it does not fine-tune the Neural ODE weights or establish broad adaptation performance.",
            "No unverified upper water-level bound is imposed; negative levels and numerical divergence are reported as diagnostics.",
        ],
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "openpyxl": openpyxl.__version__,
            "script_sha256": sha256(Path(__file__).resolve()),
            "loader_sha256": sha256(Path(__file__).with_name("intaset_causal_transfer_benchmark_v1.py")),
            "deterministic_algorithms": True,
            "device": str(DEVICE),
        },
        "fit_wall_seconds_across_seeds": fit_seconds,
        "calibration_wall_seconds_across_seeds": calibration_seconds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "intaset_neural_ode_metrics.csv"
    summary_path = output_dir / "intaset_neural_ode_summary.csv"
    calibration_path = output_dir / "intaset_neural_ode_calibration.csv"
    detail.to_csv(detail_path, index=False, float_format="%.10g")
    summary.to_csv(summary_path, index=False, float_format="%.10g")
    calibration_detail.to_csv(calibration_path, index=False, float_format="%.10g")
    metadata["output_sha256"] = {
        detail_path.name: sha256(detail_path),
        summary_path.name: sha256(summary_path),
        calibration_path.name: sha256(calibration_path),
    }
    metadata_path = output_dir / "intaset_neural_ode_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    focus = summary[
        summary.state.eq("ALL")
        & summary.task.isin([
            "causal_free_rollout_given_recorded_pump_path",
            "causal_free_rollout_after_sparse_target_train_calibration_given_recorded_pump_path",
        ])
    ]
    print(focus.to_string(index=False))
    print(f"wrote {detail_path}, {summary_path}, {calibration_path}, {metadata_path}")
    return detail, summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "results/intaset_neural_ode_transfer_v1",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated training initialization seeds (default: three fixed seeds)",
    )
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    run(args.output_dir, seeds=seeds)


if __name__ == "__main__":
    main()
