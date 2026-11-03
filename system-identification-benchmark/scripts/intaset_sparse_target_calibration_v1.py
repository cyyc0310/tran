#!/usr/bin/env python3
"""Evaluate sparse target-side calibration of InTaSet source dynamics models.

The source model remains fixed. Target training states are exposed only at the
same 1/5/10 second cadence used at evaluation, while pump inputs stay at 0.1 s.
Calibration fits four positive per-state rate multipliers from sparse endpoint
transitions only. Target test states are never used for fitting or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import zipfile

import numpy as np
import openpyxl
import pandas as pd
import sklearn
import scipy
from scipy.optimize import least_squares

from intaset_causal_transfer_benchmark_v1 import (
    CONFIG_PATHS,
    DATA_PATH,
    MANIFEST_PATH,
    OUTPUT_DIR,
    STATE_NAMES,
    STRIDES,
    DiscreteSINDYc,
    LinearARX,
    ZeroOrderHold,
    load_table,
    metric_rows,
    select_and_fit,
)


CALIBRATION_BUDGETS_SECONDS = (0, 60, 300)
GAIN_LOWER_BOUND = 0.1
GAIN_UPPER_BOUND = 5.0
OUTPUT_SUBDIR = "sparse-target-calibration"


def rollout_with_gain(
    model,
    states: np.ndarray,
    inputs: np.ndarray,
    start: int,
    stride: int,
    gain: np.ndarray,
) -> np.ndarray:
    current = states[start].copy()
    predictions = []
    for index in range(start, start + stride):
        delta = model.predict_delta(current[None, :], inputs[index][None, :])[0]
        current = current + gain * delta
        if not np.isfinite(current).all() or np.max(np.abs(current)) > 1e6:
            raise FloatingPointError("rate-gain rollout became non-finite or exceeded 1e6 cm")
        predictions.append(current.copy())
    return np.asarray(predictions)


def calibration_blocks(
    target_training: np.ndarray,
    stride: int,
    budget_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    if budget_seconds <= 0:
        empty = np.empty(0, dtype=int)
        return empty, empty
    dt = float(np.median(np.diff(target_training[:, 0])))
    final_index = int(round(budget_seconds / dt))
    if final_index >= len(target_training):
        raise ValueError(f"calibration budget {budget_seconds}s exceeds target training data")
    if not np.isclose(target_training[final_index, 0] - target_training[0, 0], budget_seconds, atol=1e-6):
        raise ValueError("target training timestamps do not align with requested calibration budget")
    if final_index % stride:
        raise ValueError("calibration budget must end on a sparse observation boundary")
    starts = np.arange(0, final_index, stride, dtype=int)
    ends = starts + stride
    return starts, ends


def calibration_residual(
    gain: np.ndarray,
    model,
    target_training: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    states, inputs = target_training[:, 1:5], target_training[:, 5:7]
    residuals = []
    try:
        for start, end in zip(starts, ends, strict=True):
            prediction = rollout_with_gain(model, states, inputs, int(start), int(end - start), gain)[-1]
            residuals.append((prediction - states[end]) / scale)
    except FloatingPointError:
        return np.full(len(starts) * 4, 1e6, dtype=float)
    return np.concatenate(residuals)


def fit_rate_gain(
    model,
    target_training: np.ndarray,
    stride: int,
    budget_seconds: int,
    scale: np.ndarray,
) -> tuple[np.ndarray, dict]:
    starts, ends = calibration_blocks(target_training, stride, budget_seconds)
    if budget_seconds == 0:
        return np.ones(4), {
            "budget_seconds": 0,
            "sparse_state_observations": 1,
            "transition_blocks": 0,
            "gain": np.ones(4).tolist(),
            "fit_performed": False,
            "jacobian_rank": 0,
            "jacobian_condition_2": None,
        }

    result = least_squares(
        calibration_residual,
        x0=np.ones(4),
        bounds=(np.full(4, GAIN_LOWER_BOUND), np.full(4, GAIN_UPPER_BOUND)),
        args=(model, target_training, starts, ends, scale),
        method="trf",
        loss="linear",
        max_nfev=200,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    tolerance = singular_values[0] * max(result.jac.shape) * np.finfo(float).eps
    rank = int(np.sum(singular_values > tolerance))
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    summary = {
        "budget_seconds": int(budget_seconds),
        "sparse_state_observations": int(len(np.unique(np.r_[starts, ends]))),
        "transition_blocks": int(len(starts)),
        "gain": result.x.tolist(),
        "fit_performed": True,
        "fit_success": bool(result.success),
        "fit_status": int(result.status),
        "fit_message": str(result.message),
        "endpoint_residual_sum_squares_normalized": float(2 * result.cost),
        "optimality": float(result.optimality),
        "function_evaluations": int(result.nfev),
        "jacobian_rank": rank,
        "jacobian_condition_2": condition,
        "gain_at_lower_bound": [bool(value <= GAIN_LOWER_BOUND + 1e-6) for value in result.x],
        "gain_at_upper_bound": [bool(value >= GAIN_UPPER_BOUND - 1e-6) for value in result.x],
    }
    return result.x, summary


def horizon_errors_with_gain(
    model,
    table: np.ndarray,
    stride: int,
    gain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    states, inputs = table[:, 1:5], table[:, 5:7]
    blocks = (len(table) - 1) // stride
    predictions, truth = [], []
    for block in range(blocks):
        start = block * stride
        end = start + stride
        predictions.append(rollout_with_gain(model, states, inputs, start, stride, gain))
        truth.append(states[start + 1 : end + 1])
    return np.concatenate(predictions), np.concatenate(truth), blocks


def one_step_with_gain(model, table: np.ndarray, gain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states, inputs = table[:, 1:5], table[:, 5:7]
    prediction = states[:-1] + gain * model.predict_delta(states[:-1], inputs[:-1])
    return prediction, states[1:]


def run(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    archive_bytes = DATA_PATH.read_bytes()
    md5 = hashlib.md5(archive_bytes).hexdigest()
    sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if md5 != manifest["zenodo_md5_expected"] or sha256 != manifest["sha256_actual"]:
        raise ValueError("InTaSet archive does not match the recorded source hashes")

    with zipfile.ZipFile(DATA_PATH) as archive:
        tables = {
            config: {split: load_table(archive, member) for split, member in paths.items()}
            for config, paths in CONFIG_PATHS.items()
        }

    fitted, tuning = {}, {}
    fit_started = time.perf_counter()
    for config in ("M1", "M2"):
        fitted[config], tuning[config] = select_and_fit(config, tables[config]["train"])
    source_fit_seconds = time.perf_counter() - fit_started

    rows, calibration_records = [], []
    calibration_started = time.perf_counter()
    source_families = (LinearARX.name, DiscreteSINDYc.name)
    for target in ("M1", "M2"):
        source = "M2" if target == "M1" else "M1"
        direction = f"{source}->{target}"
        target_training = tables[target]["train"]
        target_test = tables[target]["test"]
        scale = np.asarray(tuning[source]["source_state_scale_cm"], dtype=float)

        for stride in STRIDES:
            for budget in CALIBRATION_BUDGETS_SECONDS:
                for family in source_families:
                    model = fitted[source][family]
                    gain, fit_record = fit_rate_gain(model, target_training, stride, budget, scale)
                    fit_record.update({
                        "direction": direction,
                        "target_configuration": target,
                        "source_configuration": source,
                        "sensor_interval_seconds": stride / 10,
                        "model_family": family,
                    })
                    calibration_records.append(fit_record)
                    method = f"source_{family}_rate_gain_cal_{budget}s"

                    one_step_prediction, one_step_truth = one_step_with_gain(model, target_test, gain)
                    metric = metric_rows(
                        one_step_prediction,
                        one_step_truth,
                        scale,
                        direction=direction,
                        task="teacher_forced_one_step_diagnostic",
                        horizon_seconds=0.1,
                        method=method,
                        n_blocks=0,
                    )
                    for record in metric:
                        record.update({
                            "calibration_budget_seconds": budget,
                            "calibration_state_observations": fit_record["sparse_state_observations"],
                            "calibration_transition_blocks": fit_record["transition_blocks"],
                        })
                    rows.extend(metric)

                    prediction, truth, blocks = horizon_errors_with_gain(model, target_test, stride, gain)
                    metric = metric_rows(
                        prediction,
                        truth,
                        scale,
                        direction=direction,
                        task="causal_free_rollout",
                        horizon_seconds=stride / 10,
                        method=method,
                        n_blocks=blocks,
                    )
                    for record in metric:
                        record.update({
                            "calibration_budget_seconds": budget,
                            "calibration_state_observations": fit_record["sparse_state_observations"],
                            "calibration_transition_blocks": fit_record["transition_blocks"],
                        })
                    rows.extend(metric)

                if budget == 0:
                    hold = ZeroOrderHold()
                    prediction, truth, blocks = horizon_errors_with_gain(hold, target_test, stride, np.ones(4))
                    metric = metric_rows(
                        prediction,
                        truth,
                        scale,
                        direction=direction,
                        task="causal_free_rollout",
                        horizon_seconds=stride / 10,
                        method="source_zero_order_hold_cal_0s",
                        n_blocks=blocks,
                    )
                    for record in metric:
                        record.update({
                            "calibration_budget_seconds": 0,
                            "calibration_state_observations": 1,
                            "calibration_transition_blocks": 0,
                        })
                    rows.extend(metric)

    calibration_fit_seconds = time.perf_counter() - calibration_started
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            ["task", "horizon_seconds", "calibration_budget_seconds", "method", "state"],
            as_index=False,
        )[["rmse_cm", "mae_cm", "rmse_over_source_train_std", "predicted_below_zero_fraction"]]
        .mean()
        .sort_values([
            "task", "horizon_seconds", "calibration_budget_seconds", "state",
            "rmse_over_source_train_std",
        ])
    )

    script_path = Path(__file__).resolve()
    base_script_path = Path(__file__).with_name("intaset_causal_transfer_benchmark_v1.py")
    metadata = {
        "experiment": "InTaSet sparse target calibration for cross-configuration dynamics",
        "task_contract": {
            "target_configuration_directions": ["M1->M2", "M2->M1"],
            "target_state_observation_intervals_seconds": [stride / 10 for stride in STRIDES],
            "pump_input_interval_seconds": 0.1,
            "target_calibration_budgets_seconds": list(CALIBRATION_BUDGETS_SECONDS),
            "calibration_data": "Only the target configuration training prefix is used. Four target states are exposed at the same sparse cadence as evaluation; pump inputs remain at native 0.1 s resolution. Between boundary states, only recorded control inputs are used by the model rollout.",
            "test_data": "The target testing file is used only for final teacher-forced one-step diagnostics and causal sparse-start rollouts; never used to fit rate gains or select hyperparameters.",
        },
        "source_model_selection": "The linear ARX and degree-two discrete SINDYc-style source models use the existing chronological source-only final-20% validation procedure, with mean free-rollout error across 1/5/10-second horizons.",
        "target_adapter": {
            "form": "delta_x_adapted = gain_elementwise * delta_x_source",
            "trainable_parameter_count": 4,
            "bounds": [GAIN_LOWER_BOUND, GAIN_UPPER_BOUND],
            "fit_objective": "Least squares of normalized endpoint state errors across sparse target calibration blocks. No target interior state labels are used.",
            "budget_zero": "gain is fixed to one and represents the zero-shot source model.",
            "interpretation": "This is a deliberately low-dimensional calibration reference, not a proposed novel algorithm and not a full target fine-tune.",
        },
        "tuning": tuning,
        "calibration_fits": calibration_records,
        "source_training_wall_seconds": source_fit_seconds,
        "calibration_and_evaluation_wall_seconds": calibration_fit_seconds,
        "source_manifest": manifest,
        "limitations": [
            "This is the first target-calibration baseline; the InTaSet author Neural ODE, four-tank kernel method, and hybrid model are not included.",
            "A positive scalar gain per state cannot express configuration changes that alter the structure or direction of individual flow couplings.",
            "The target training and test files contain one short record per configuration; score differences are descriptive, not independent-trial significance claims.",
            "The target calibration objective uses sparse block endpoints and can be weakly identified when few blocks are available; Jacobian rank and conditioning are recorded.",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "openpyxl": openpyxl.__version__,
            "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
            "source_benchmark_script_sha256": hashlib.sha256(base_script_path.read_bytes()).hexdigest(),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "sparse_calibration_metrics.csv", index=False, float_format="%.10g")
    summary.to_csv(output_dir / "sparse_calibration_summary.csv", index=False, float_format="%.10g")
    metadata["detail_csv_sha256"] = hashlib.sha256((output_dir / "sparse_calibration_metrics.csv").read_bytes()).hexdigest()
    metadata["summary_csv_sha256"] = hashlib.sha256((output_dir / "sparse_calibration_summary.csv").read_bytes()).hexdigest()
    (output_dir / "sparse_calibration_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return detail, summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / OUTPUT_SUBDIR,
    )
    args = parser.parse_args()
    _, summary, metadata = run(args.output_dir)
    selected = summary[(summary["task"] == "causal_free_rollout") & (summary["state"] == "ALL")]
    print(selected.to_string(index=False))
    print(f"script_sha256={metadata['software']['script_sha256']}")
    print(f"detail_csv_sha256={metadata['detail_csv_sha256']}")
    print(f"summary_csv_sha256={metadata['summary_csv_sha256']}")


if __name__ == "__main__":
    main()
