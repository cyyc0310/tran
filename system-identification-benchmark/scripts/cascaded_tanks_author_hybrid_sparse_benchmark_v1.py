#!/usr/bin/env python3
"""Adapt the published Cascaded Tanks neural state-space model to sparse output forecasts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs/algorithmic-control-v1/external-source/cascaded_tanks"
UPSTREAM_DIR = ROOT / "docs/algorithmic-control-v1/external-source/forgione_piga_sysid_neural_continuous"
UPSTREAM_COMMIT = "956aa7ecd73061b0321c6f6098d74c9e4a7ce67a"
ARCHIVE_PATH = SOURCE_DIR / "CascadedTanksFiles.zip"
PROTOCOL_PATH = ROOT / "docs/algorithmic-control-v1/CASCADED_TANKS_AUTHOR_HYBRID_SPARSE_PROTOCOL_V1_20261103.md"
DEFAULT_OUTPUT_DIR = ROOT / "docs/algorithmic-control-v1/results/cascaded_tanks_author_hybrid_sparse_v1"
EXPECTED_DATA_CSV_SHA256 = "ef2388ed822f3aef4aa80d6b0f2b466dd80b361786b3eafc7a2957c31ea323a7"
DT_SECONDS = 4.0
TARGET_BUDGETS = (0, 1, 3, 5)
HORIZONS = (1, 3, 5)
SOURCE_TRAINING = {
    "n_feat": 100,
    "sequence_length": 128,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "iterations": 10000,
    "burnin_fit_only_iterations": 1001,
    "consistency_weight_after_burnin": 1.0,
    "integrator": "forward_euler",
    "sampling_interval_seconds": DT_SECONDS,
}


sys.path.insert(0, str(UPSTREAM_DIR / "torchid"))
from ssmodels_ct import CascadedTanksOverflowNeuralStateSpaceModel  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    archive_bytes = ARCHIVE_PATH.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        member = "CascadedTanksFiles/dataBenchmark.csv"
        csv_bytes = zf.read(member)
    csv_hash = sha256_bytes(csv_bytes)
    if csv_hash != EXPECTED_DATA_CSV_SHA256:
        raise ValueError(f"upstream CSV/data archive mismatch: {csv_hash}")
    table = pd.read_csv(io.BytesIO(csv_bytes))
    required = {"uEst", "yEst", "uVal", "yVal", "Ts"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"missing Cascaded Tanks columns: {sorted(missing)}")
    dt_values = table["Ts"].dropna().to_numpy(dtype=float)
    if not len(dt_values) or not np.allclose(dt_values, DT_SECONDS, rtol=0.0, atol=1e-8):
        raise ValueError(f"unexpected sampling interval: {np.unique(dt_values)}")
    u_est = table["uEst"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_est = table["yEst"].to_numpy(dtype=np.float32).reshape(-1, 1)
    u_val = table["uVal"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_val = table["yVal"].to_numpy(dtype=np.float32).reshape(-1)
    if not all(len(x) == 1024 for x in (u_est, y_est, u_val, y_val)):
        raise ValueError("official Cascaded Tanks estimation/test record lengths changed")
    return u_est, y_est, u_val, y_val, {
        "archive_sha256": sha256_bytes(archive_bytes),
        "data_csv_member": member,
        "data_csv_sha256": csv_hash,
    }


class ForwardEulerSequence(nn.Module):
    """Upstream simulator rule: x[k+1] = x[k] + dt * f(x[k], u[k])."""

    def __init__(self, model: nn.Module, dt: float):
        super().__init__()
        self.model = model
        self.dt = float(dt)

    def forward(self, x0: torch.Tensor, u_path: torch.Tensor) -> torch.Tensor:
        state = x0
        outputs = []
        for index in range(u_path.shape[1]):
            outputs.append(state)
            state = state + self.dt * self.model(state, u_path[:, index, :])
        return torch.stack(outputs, dim=1)


def train_author_model(
    u_est: np.ndarray,
    y_est: np.ndarray,
    seed: int,
    iterations: int,
) -> tuple[nn.Module, np.ndarray, dict, list[dict]]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CascadedTanksOverflowNeuralStateSpaceModel(n_feat=SOURCE_TRAINING["n_feat"])
    sequence = ForwardEulerSequence(model, DT_SECONDS)
    n_samples = len(y_est)
    seq_len = SOURCE_TRAINING["sequence_length"]
    batch_size = SOURCE_TRAINING["batch_size"]
    if n_samples <= seq_len + batch_size:
        raise ValueError("source record is too short for the author training window/batch contract")

    hidden_initial = np.zeros((n_samples, 2), dtype=np.float32)
    hidden_initial[:, 1] = y_est[:, 0]
    hidden = torch.tensor(hidden_initial, dtype=torch.float32, requires_grad=True)
    u_tensor = torch.as_tensor(u_est, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_est, dtype=torch.float32)

    example_starts = rng.choice(n_samples - seq_len, size=batch_size, replace=False)
    example_indices = example_starts[:, None] + np.arange(seq_len)[None, :]
    example_x0 = hidden[torch.as_tensor(example_starts, dtype=torch.long)]
    example_u = u_tensor[torch.as_tensor(example_indices, dtype=torch.long)]
    with torch.no_grad():
        traced = torch.jit.trace(sequence, (example_x0, example_u), check_trace=False)
        example_x_sim = traced(example_x0, example_u)
        example_y = y_tensor[torch.as_tensor(example_indices, dtype=torch.long)]
        scale_error = torch.sqrt(torch.mean((example_x_sim[:, :, 1:2] - example_y) ** 2)).clamp_min(1e-6)

    optimizer = torch.optim.Adam(
        [
            {"params": list(model.parameters()), "lr": SOURCE_TRAINING["learning_rate"]},
            {"params": [hidden], "lr": SOURCE_TRAINING["learning_rate"]},
        ],
        lr=SOURCE_TRAINING["learning_rate"],
    )

    training_history: list[dict] = []
    start_time = time.perf_counter()
    final_values: dict[str, float] = {}
    for iteration in range(iterations):
        starts = rng.choice(n_samples - seq_len, size=batch_size, replace=False)
        indices = starts[:, None] + np.arange(seq_len)[None, :]
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        x0 = hidden[torch.as_tensor(starts, dtype=torch.long)]
        u_batch = u_tensor[index_tensor]
        y_batch = y_tensor[index_tensor]
        x_hidden_batch = hidden[index_tensor]

        optimizer.zero_grad(set_to_none=True)
        simulated = traced(x0, u_batch)
        fit_loss = torch.mean(((simulated[:, :, 1:2] - y_batch) / scale_error) ** 2)
        consistency_loss = torch.mean(((simulated - x_hidden_batch) / scale_error) ** 2)
        if iteration > SOURCE_TRAINING["burnin_fit_only_iterations"]:
            loss = fit_loss + SOURCE_TRAINING["consistency_weight_after_burnin"] * consistency_loss
        else:
            loss = fit_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite author model loss at seed={seed}, iteration={iteration}")
        loss.backward()
        optimizer.step()

        final_values = {
            "total_loss": float(loss.detach()),
            "fit_loss": float(fit_loss.detach()),
            "consistency_loss": float(consistency_loss.detach()),
        }
        if iteration == 0 or (iteration + 1) % 100 == 0 or iteration + 1 == iterations:
            training_history.append({"seed": seed, "iteration": iteration + 1, **final_values})

    elapsed = time.perf_counter() - start_time
    hidden_np = hidden.detach().cpu().numpy()
    receipt = {
        "seed": int(seed),
        "iterations": int(iterations),
        "sequence_length": int(seq_len),
        "batch_size": int(batch_size),
        "learning_rate": float(SOURCE_TRAINING["learning_rate"]),
        "burnin_fit_only_iterations": int(SOURCE_TRAINING["burnin_fit_only_iterations"]),
        "consistency_weight_after_burnin": float(SOURCE_TRAINING["consistency_weight_after_burnin"]),
        "initial_output_error_scale": float(scale_error.detach()),
        "final_total_loss": final_values["total_loss"],
        "final_fit_loss": final_values["fit_loss"],
        "final_consistency_loss": final_values["consistency_loss"],
        "training_seconds": float(elapsed),
        "initial_latent_state": hidden_np[0].tolist(),
    }
    return model, hidden_np, receipt, training_history


@torch.no_grad()
def euler_step(model: nn.Module, state: np.ndarray, control: float) -> np.ndarray:
    x = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(1, 2))
    u = torch.as_tensor([[float(control)]], dtype=torch.float32)
    next_x = x + DT_SECONDS * model(x, u)
    return next_x[0].cpu().numpy().astype(float)


def rollout_prefix(model: nn.Module, initial_state: np.ndarray, u_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(initial_state, dtype=float).copy()
    states: list[np.ndarray] = []
    for control in np.asarray(u_path, dtype=float).reshape(-1):
        if not np.isfinite(state).all():
            states.append(np.full(2, np.nan))
            state = np.full(2, np.nan)
            continue
        state = euler_step(model, state, float(control))
        states.append(state.copy())
    if not states:
        return np.empty((0, 2), dtype=float), state
    return np.stack(states), state


def fit_prefix_gamma(base_output: np.ndarray, target_prefix: np.ndarray, y0: float) -> dict:
    base = np.asarray(base_output, dtype=float).reshape(-1)
    target = np.asarray(target_prefix, dtype=float).reshape(-1)
    if not len(base):
        return {"gamma": 1.0, "fallback": "zero_budget", "prefix_base_rmse": None, "prefix_calibrated_rmse": None}
    if not np.isfinite(base).all():
        return {"gamma": 1.0, "fallback": "non_finite_prefix_rollout", "prefix_base_rmse": None, "prefix_calibrated_rmse": None}
    source_delta = base - float(y0)
    target_delta = target - float(y0)
    denominator = float(np.dot(source_delta, source_delta))
    if denominator <= 1e-18:
        gamma = 1.0
        fallback = "zero_source_path_displacement"
    else:
        gamma = float(np.dot(source_delta, target_delta) / denominator)
        fallback = None if np.isfinite(gamma) else "non_finite_least_squares_gain"
        if fallback:
            gamma = 1.0
    calibrated = float(y0) + gamma * source_delta
    return {
        "gamma": float(gamma),
        "fallback": fallback,
        "prefix_base_rmse": float(np.sqrt(np.mean((target - base) ** 2))),
        "prefix_calibrated_rmse": float(np.sqrt(np.mean((target - calibrated) ** 2))),
    }


def make_predictions_for_seed(
    model: nn.Module,
    hidden_train: np.ndarray,
    u_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    source_initial_state = np.asarray(hidden_train[0], dtype=float).copy()
    source_initial_state[1] = float(y_val[0])
    block_rows: list[dict] = []
    calibration_rows: list[dict] = []
    source_control_range = [float(np.min(u_val)), float(np.max(u_val))]

    for budget in TARGET_BUDGETS:
        prefix_states, state_after_prefix = rollout_prefix(model, source_initial_state, u_val[:budget])
        base_prefix = prefix_states[:, 1] if len(prefix_states) else np.empty(0, dtype=float)
        gamma_record = fit_prefix_gamma(base_prefix, y_val[1 : budget + 1], float(y_val[0]))
        gamma = float(gamma_record["gamma"])
        calibration_rows.append({
            "seed": int(seed),
            "method": "author_hybrid_output_path_gamma",
            "calibration_budget_transitions": int(budget),
            "calibration_budget_seconds": float(budget * DT_SECONDS),
            "prefix_observations": int(budget + 1),
            "gamma": gamma,
            "fallback": gamma_record["fallback"],
            "prefix_base_rmse": gamma_record["prefix_base_rmse"],
            "prefix_calibrated_rmse": gamma_record["prefix_calibrated_rmse"],
            "base_prefix_prediction_sha256": sha256_bytes(np.asarray(base_prefix, dtype="<f8").tobytes()),
        })

        if budget == 0:
            state_at_budget = source_initial_state.copy()
        else:
            state_at_budget = state_after_prefix.copy()
        if not np.isfinite(state_at_budget).all():
            state_at_budget[1] = float(y_val[budget])
        else:
            state_at_budget[1] = float(y_val[budget])

        for horizon in HORIZONS:
            state = state_at_budget.copy()
            current_index = budget
            starts = np.arange(budget, len(y_val) - horizon, horizon, dtype=int)
            for block_index, start in enumerate(starts):
                if current_index != start:
                    raise RuntimeError(f"block/state alignment error: {current_index} != {start}")
                if np.isfinite(state[0]):
                    state[1] = float(y_val[start])
                base_block: list[np.ndarray] = []
                for offset in range(horizon):
                    if not np.isfinite(state).all():
                        state = np.full(2, np.nan)
                    else:
                        state = euler_step(model, state, float(u_val[start + offset, 0]))
                    base_block.append(state.copy())
                base_outputs = np.asarray([item[1] for item in base_block], dtype=float)
                methods = {
                    "author_hybrid_frozen": 1.0,
                    "author_hybrid_output_path_gamma": gamma,
                }
                for method, method_gamma in methods.items():
                    pred_outputs = float(y_val[start]) + method_gamma * (base_outputs - float(y_val[start]))
                    for step, (pred, actual, latent_state) in enumerate(
                        zip(pred_outputs, y_val[start + 1 : start + horizon + 1], base_block), start=1
                    ):
                        block_rows.append({
                            "seed": int(seed),
                            "task": "causal_block_rollout_given_recorded_pump_path",
                            "method": method,
                            "calibration_budget_transitions": int(budget),
                            "calibration_budget_seconds": float(budget * DT_SECONDS),
                            "horizon_transitions": int(horizon),
                            "horizon_seconds": float(horizon * DT_SECONDS),
                            "block_index": int(block_index),
                            "start_index": int(start),
                            "start_time_seconds": float(start * DT_SECONDS),
                            "step_in_block": int(step),
                            "prediction": float(pred),
                            "truth": float(actual),
                            "gamma": float(method_gamma),
                            "latent_upper_state": float(latent_state[0]),
                            "base_lower_state": float(latent_state[1]),
                            "control_min_in_block": float(np.min(u_val[start : start + horizon, 0])),
                            "control_max_in_block": float(np.max(u_val[start : start + horizon, 0])),
                        })
                current_index = start + horizon
                # Only the observable lower-tank output is reset at sparse block boundaries.
                if np.isfinite(state[0]):
                    state[1] = float(y_val[current_index])

    return block_rows, calibration_rows


def summarize(block_df: pd.DataFrame, source_y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    keys = ["seed", "method", "calibration_budget_transitions", "calibration_budget_seconds", "horizon_transitions", "horizon_seconds"]
    source_std = max(float(np.std(source_y)), 1e-12)
    source_min, source_max = float(np.min(source_y)), float(np.max(source_y))
    for key, group in block_df.groupby(keys, dropna=False, sort=True):
        pred = group["prediction"].to_numpy(dtype=float)
        truth = group["truth"].to_numpy(dtype=float)
        finite = np.isfinite(pred) & np.isfinite(truth)
        diff = pred[finite] - truth[finite]
        rmse = float(np.sqrt(np.mean(diff**2))) if len(diff) else None
        mae = float(np.mean(np.abs(diff))) if len(diff) else None
        latent = group["latent_upper_state"].to_numpy(dtype=float)
        latent_finite = np.isfinite(latent)
        seed, method, budget, budget_seconds, horizon, horizon_seconds = key
        metric_rows.append({
            "seed": int(seed),
            "task": "causal_block_rollout_given_recorded_pump_path",
            "method": method,
            "calibration_budget_transitions": int(budget),
            "calibration_budget_seconds": float(budget_seconds),
            "horizon_transitions": int(horizon),
            "horizon_seconds": float(horizon_seconds),
            "blocks": int(group["block_index"].nunique()),
            "evaluated_steps": int(len(group)),
            "finite_steps": int(np.sum(finite)),
            "finite_fraction": float(np.mean(finite)) if len(finite) else None,
            "diverged_blocks": int(group.loc[~finite, "block_index"].nunique()),
            "rmse": rmse,
            "mae": mae,
            "normalized_rmse": float(rmse / source_std) if rmse is not None else None,
            "negative_output_fraction": float(np.mean(pred[finite] < 0.0)) if np.any(finite) else None,
            "outside_source_observation_range_fraction": float(np.mean((pred[finite] < source_min) | (pred[finite] > source_max))) if np.any(finite) else None,
            "negative_latent_upper_state_fraction": float(np.mean(latent[latent_finite] < 0.0)) if np.any(latent_finite) else None,
            "latent_upper_state_finite_fraction": float(np.mean(latent_finite)) if len(latent) else None,
        })
    metrics = pd.DataFrame(metric_rows)
    summary_keys = ["method", "calibration_budget_transitions", "calibration_budget_seconds", "horizon_transitions", "horizon_seconds"]
    summary_rows = []
    for key, group in metrics.groupby(summary_keys, dropna=False, sort=True):
        row = dict(zip(summary_keys, key))
        row.update({
            "seeds": int(group["seed"].nunique()),
            "mean_normalized_rmse": float(group["normalized_rmse"].mean()),
            "sd_normalized_rmse_across_seeds": float(group["normalized_rmse"].std(ddof=1)) if len(group) > 1 else 0.0,
            "mean_rmse": float(group["rmse"].mean()),
            "mean_mae": float(group["mae"].mean()),
            "mean_finite_fraction": float(group["finite_fraction"].mean()),
            "mean_negative_output_fraction": float(group["negative_output_fraction"].mean()),
            "mean_outside_source_observation_range_fraction": float(group["outside_source_observation_range_fraction"].mean()),
            "mean_negative_latent_upper_state_fraction": float(group["negative_latent_upper_state_fraction"].mean()),
            "interpretation": "optimizer-seed variability on one public test record; not independent physical repeats",
        })
        summary_rows.append(row)
    return metrics, pd.DataFrame(summary_rows)


def run(seeds: tuple[int, ...], iterations: int, output_dir: Path) -> dict:
    u_est, y_est, u_val, y_val, data_receipt = load_data()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    all_blocks: list[dict] = []
    all_calibrations: list[dict] = []
    all_training: list[dict] = []
    all_history: list[dict] = []

    for seed in seeds:
        print(f"training author CTS hybrid seed={seed} iterations={iterations}", flush=True)
        model, hidden, receipt, history = train_author_model(u_est, y_est, int(seed), iterations)
        receipt.update({
            "source_archive_sha256": data_receipt["archive_sha256"],
            "source_data_csv_sha256": data_receipt["data_csv_sha256"],
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_model_file_sha256": sha256_file(UPSTREAM_DIR / "torchid/ssmodels_ct.py"),
            "upstream_training_script_sha256": sha256_file(UPSTREAM_DIR / "examples/CTS/CTS_fit_truncated.py"),
        })
        checkpoint_path = model_dir / f"author_cts_hybrid_seed_{seed}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "source_hidden_states": torch.as_tensor(hidden, dtype=torch.float32),
            "seed": int(seed),
            "upstream_commit": UPSTREAM_COMMIT,
            "training_iterations": int(iterations),
            "training_config": SOURCE_TRAINING,
        }, checkpoint_path)
        receipt["checkpoint_sha256"] = sha256_file(checkpoint_path)
        all_training.append(receipt)
        all_history.extend(history)
        model.eval()
        blocks, calibrations = make_predictions_for_seed(model, hidden, u_val, y_val, int(seed))
        all_blocks.extend(blocks)
        all_calibrations.extend(calibrations)
        print(f"finished seed={seed} elapsed_seconds={receipt['training_seconds']:.1f}", flush=True)

    block_df = pd.DataFrame(all_blocks)
    calibration_df = pd.DataFrame(all_calibrations)
    training_df = pd.DataFrame(all_training)
    history_df = pd.DataFrame(all_history)
    metrics_df, summary_df = summarize(block_df, y_est[:, 0])
    block_path = output_dir / "author_hybrid_blocks.csv"
    calibration_path = output_dir / "author_hybrid_calibration_receipts.csv"
    training_path = output_dir / "author_hybrid_training_receipts.csv"
    history_path = output_dir / "author_hybrid_training_history.csv"
    metrics_path = output_dir / "author_hybrid_metrics.csv"
    summary_path = output_dir / "author_hybrid_summary.csv"
    block_df.to_csv(block_path, index=False, float_format="%.12g")
    calibration_df.to_csv(calibration_path, index=False, float_format="%.12g")
    training_df.to_csv(training_path, index=False, float_format="%.12g")
    history_df.to_csv(history_path, index=False, float_format="%.12g")
    metrics_df.to_csv(metrics_path, index=False, float_format="%.12g")
    summary_df.to_csv(summary_path, index=False, float_format="%.12g")

    metadata = {
        "status": "COMPLETED",
        "scope": "author-architecture/author-training-recipe replication adapted to a sparse-output block-forecast protocol",
        "test_labels_previously_seen": True,
        "independent_test_or_blind_confirmation": False,
        "physical_record_repeats": 1,
        "target_observation_budget_transitions": list(TARGET_BUDGETS),
        "horizons_and_sparse_measurement_intervals_transitions": list(HORIZONS),
        "sampling_interval_seconds": DT_SECONDS,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_model_source_sha256": sha256_file(UPSTREAM_DIR / "torchid/ssmodels_ct.py"),
        "upstream_training_script_sha256": sha256_file(UPSTREAM_DIR / "examples/CTS/CTS_fit_truncated.py"),
        "upstream_license_sha256": sha256_file(UPSTREAM_DIR / "LICENSE.md"),
        "project_script_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "data_receipt": data_receipt,
        "training_config": {**SOURCE_TRAINING, "iterations": int(iterations), "seeds": list(seeds), "torch_version": torch.__version__},
        "prediction_contract": "The latent upper state is propagated from the learned source initial state; only measured lower-tank output is reset at sparse block starts. Recorded future pump input is provided within each block.",
        "output_files": {},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__},
    }
    for path in (block_path, calibration_path, training_path, history_path, metrics_path, summary_path):
        metadata["output_files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    metadata_path = output_dir / "author_hybrid_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=SOURCE_TRAINING["iterations"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    metadata = run(tuple(args.seeds), int(args.iterations), args.output_dir)
    print(json.dumps({"status": metadata["status"], "output_dir": str(args.output_dir), "metrics_sha256": metadata["output_files"]["author_hybrid_metrics.csv"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
