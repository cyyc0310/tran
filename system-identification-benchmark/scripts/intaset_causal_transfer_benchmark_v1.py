#!/usr/bin/env python3
"""Causal, cross-configuration system-identification benchmark on InTaSet.

Source models are trained on the provided dense training trace. The target test
trace exposes four states only at block starts; pump inputs remain visible at
their native 0.1 s rate. Predictions are scored before the next measured state
is used. Endpoint-conditioned bridges are reported as a separate offline task.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs/algorithmic-control-v1/source/InTaSet.zip"
MANIFEST_PATH = ROOT / "docs/algorithmic-control-v1/source_manifest.json"
OUTPUT_DIR = ROOT / "docs/algorithmic-control-v1"
CONFIG_PATHS = {
    "M1": {
        "train": "InTaSet/InTaSet-ID/InTaSet-ID-4T-M1/InTaSet-ID-4T-M1_Training/InTaSet-ID-4T-M1_Training.xlsx",
        "test": "InTaSet/InTaSet-ID/InTaSet-ID-4T-M1/InTaSet-ID-4T-M1_Testing/InTaSet-ID-4T-M1_Testing.xlsx",
    },
    "M2": {
        "train": "InTaSet/InTaSet-ID/InTaSet-ID-4T-M2/InTaSet-ID-4T-M2_Training/InTaSet-ID-4T-M2_Training.xlsx",
        "test": "InTaSet/InTaSet-ID/InTaSet-ID-4T-M2/InTaSet-ID-4T-M2_Testing/InTaSet-ID-4T-M2_Testing.xlsx",
    },
}
STRIDES = (10, 50, 100)  # 0.1 s native data -> 1, 5, 10 s sparse observations.
RIDGE_ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
STLSQ_THRESHOLDS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
STATE_NAMES = ("h1", "h2", "h3", "h4")


def load_table(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    workbook = openpyxl.load_workbook(archive.open(member), read_only=True, data_only=True)
    sheet = workbook.worksheets[0]
    rows = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if not row or not isinstance(row[0], (int, float)):
            continue
        values = np.asarray([np.nan if value is None else float(value) for value in row[:7]])
        if values.shape == (7,) and np.isfinite(values).all():
            rows.append(values)
    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or array.shape[1] != 7 or len(array) < 20:
        raise ValueError(f"invalid four-tank table: {member}")
    if not np.allclose(np.diff(array[:, 0]), 0.1, atol=1e-7, rtol=0):
        raise ValueError(f"expected contiguous 0.1-second samples: {member}")
    return array


def transition_data(table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, inputs = table[:, 1:5], table[:, 5:7]
    return states[:-1], inputs[:-1], np.diff(states, axis=0)


class LinearARX:
    """Fully observed discrete linear state-space / first-order ARX model."""

    name = "linear_state_space_arx"

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, table: np.ndarray) -> "LinearARX":
        states, inputs, deltas = transition_data(table)
        features = np.column_stack([states, inputs])
        self.feature_scaler = StandardScaler().fit(features)
        self.delta_scale = np.std(deltas, axis=0).clip(1e-10)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(self.feature_scaler.transform(features), deltas / self.delta_scale)
        return self

    def predict_delta(self, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        features = np.column_stack([states, inputs])
        return self.model.predict(self.feature_scaler.transform(features)) * self.delta_scale

    def complexity(self) -> int:
        return int(np.count_nonzero(np.abs(self.model.coef_) > 1e-12))


class DiscreteSINDYc:
    """Degree-two discrete SINDYc-style model fit with sequential thresholding."""

    name = "discrete_sindy_c_stlsq_degree2"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def fit(self, table: np.ndarray) -> "DiscreteSINDYc":
        states, inputs, deltas = transition_data(table)
        base = np.column_stack([states, inputs])
        self.base_scaler = StandardScaler().fit(base)
        base_scaled = self.base_scaler.transform(base)
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        raw_library = self.poly.fit_transform(base_scaled)
        self.term_mean = np.mean(raw_library, axis=0)
        self.term_scale = np.std(raw_library, axis=0).clip(1e-10)
        library = (raw_library - self.term_mean) / self.term_scale
        design = np.column_stack([np.ones(len(library)), library])
        self.delta_scale = np.std(deltas, axis=0).clip(1e-10)
        target = deltas / self.delta_scale

        self.coef_ = np.zeros((design.shape[1], target.shape[1]))
        self.active_terms_ = []
        for output in range(target.shape[1]):
            active = np.ones(design.shape[1], dtype=bool)
            active[0] = True  # Keep the constant offset in every output equation.
            for _ in range(20):
                coefficients = np.linalg.lstsq(design[:, active], target[:, output], rcond=None)[0]
                active_indices = np.flatnonzero(active)
                removable = (np.abs(coefficients) < self.threshold) & (active_indices != 0)
                if not np.any(removable):
                    break
                active[active_indices[removable]] = False
                if np.sum(active) == 1:
                    break
            coefficients = np.linalg.lstsq(design[:, active], target[:, output], rcond=None)[0]
            self.coef_[active, output] = coefficients
            self.active_terms_.append(int(np.sum(active)))
        return self

    def predict_delta(self, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        base = np.column_stack([states, inputs])
        base_scaled = self.base_scaler.transform(base)
        raw_library = self.poly.transform(base_scaled)
        library = (raw_library - self.term_mean) / self.term_scale
        design = np.column_stack([np.ones(len(library)), library])
        return (design @ self.coef_) * self.delta_scale

    def complexity(self) -> int:
        return int(np.sum(np.abs(self.coef_) > 1e-12))


class ZeroOrderHold:
    """Causal no-dynamics reference: hold each observed state until next sample."""

    name = "zero_order_hold"

    def predict_delta(self, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        return np.zeros_like(states)

    def complexity(self) -> int:
        return 0


def fit_candidate(name: str, parameter: float, table: np.ndarray):
    if name == LinearARX.name:
        return LinearARX(alpha=parameter).fit(table)
    if name == DiscreteSINDYc.name:
        return DiscreteSINDYc(threshold=parameter).fit(table)
    raise ValueError(f"unknown model family: {name}")


def rollout(model, states: np.ndarray, inputs: np.ndarray, start: int, stride: int) -> np.ndarray:
    current = states[start].copy()
    predictions = []
    for index in range(start, start + stride):
        current = current + model.predict_delta(current[None, :], inputs[index][None, :])[0]
        predictions.append(current.copy())
    return np.asarray(predictions)


def horizon_errors(model, table: np.ndarray, stride: int, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    states, inputs = table[:, 1:5], table[:, 5:7]
    blocks = (len(table) - 1) // stride
    predictions, truth = [], []
    for block in range(blocks):
        start = block * stride
        end = start + stride
        predictions.append(rollout(model, states, inputs, start, stride))
        truth.append(states[start + 1 : end + 1])
    pred = np.concatenate(predictions) if predictions else np.empty((0, 4))
    actual = np.concatenate(truth) if truth else np.empty((0, 4))
    return pred, actual, blocks


def normalized_rollout_score(model, table: np.ndarray, scale: np.ndarray) -> float:
    scores = []
    for stride in STRIDES:
        pred, truth, _ = horizon_errors(model, table, stride, scale)
        scores.append(float(np.sqrt(np.mean(((pred - truth) / scale) ** 2))))
    return float(np.mean(scores))


def select_and_fit(config: str, training: np.ndarray) -> tuple[dict, dict]:
    split = int(len(training) * 0.8)
    fitting = training[:split]
    validation = training[split - 1 :]
    scale = np.std(fitting[:, 1:5], axis=0).clip(1e-10)
    candidates = {
        LinearARX.name: RIDGE_ALPHAS,
        DiscreteSINDYc.name: STLSQ_THRESHOLDS,
    }
    fitted = {}
    selection = {}
    for family, parameters in candidates.items():
        evaluations = []
        for parameter in parameters:
            candidate = fit_candidate(family, parameter, fitting)
            score = normalized_rollout_score(candidate, validation, scale)
            evaluations.append({"parameter": float(parameter), "mean_normalized_rollout_rmse_1_5_10s": score})
        chosen = min(evaluations, key=lambda item: item["mean_normalized_rollout_rmse_1_5_10s"])
        final_model = fit_candidate(family, float(chosen["parameter"]), training)
        fitted[family] = final_model
        selection[family] = {
            "selected_parameter": float(chosen["parameter"]),
            "parameter_name": "ridge_alpha" if family == LinearARX.name else "stlsq_threshold",
            "selection": "minimum mean 1/5/10-second free-rollout RMSE on chronological final 20% of training trace",
            "validation_candidates": evaluations,
            "active_coefficient_count_after_refit": final_model.complexity(),
        }
    return fitted, {
        "config": config,
        "fit_rows": int(split),
        "validation_rows": int(len(validation)),
        "source_state_scale_cm": scale.tolist(),
        "models": selection,
    }


def metric_rows(
    prediction: np.ndarray,
    truth: np.ndarray,
    scale: np.ndarray,
    *,
    direction: str,
    task: str,
    horizon_seconds: float,
    method: str,
    n_blocks: int,
) -> list[dict]:
    rows = []
    for channels, name in [(np.arange(4), "ALL"), *[(np.array([i]), state) for i, state in enumerate(STATE_NAMES)]]:
        pred = prediction[:, channels]
        actual = truth[:, channels]
        error = pred - actual
        normalized = error / scale[channels]
        rows.append({
            "direction": direction,
            "task": task,
            "horizon_seconds": horizon_seconds,
            "method": method,
            "state": name,
            "n_scored_values": int(error.size),
            "n_blocks": int(n_blocks),
            "rmse_cm": float(np.sqrt(np.mean(error**2))),
            "mae_cm": float(np.mean(np.abs(error))),
            "rmse_over_source_train_std": float(np.sqrt(np.mean(normalized**2))),
            "predicted_below_zero_fraction": float(np.mean(pred < 0)),
            "minimum_predicted_level_cm": float(np.min(pred)),
        })
    return rows


def evaluate_one_step(model, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states, inputs = table[:, 1:5], table[:, 5:7]
    prediction = states[:-1] + model.predict_delta(states[:-1], inputs[:-1])
    return prediction, states[1:]


def evaluate_endpoint_interpolation(table: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray, int]:
    states = table[:, 1:5]
    predictions, truth = [], []
    blocks = (len(table) - 1) // stride
    fractions = np.arange(1, stride, dtype=float) / stride
    for block in range(blocks):
        start, end = block * stride, (block + 1) * stride
        predictions.append(states[start] + (states[end] - states[start]) * fractions[:, None])
        truth.append(states[start + 1 : end])
    return np.concatenate(predictions), np.concatenate(truth), blocks


def evaluate_endpoint_bridge(model, table: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray, int]:
    states, inputs = table[:, 1:5], table[:, 5:7]
    predictions, truth = [], []
    blocks = (len(table) - 1) // stride
    fractions = np.arange(1, stride, dtype=float) / stride
    for block in range(blocks):
        start, end = block * stride, (block + 1) * stride
        raw = rollout(model, states, inputs, start, stride)
        correction = states[end] - raw[-1]
        predictions.append(raw[:-1] + fractions[:, None] * correction)
        truth.append(states[start + 1 : end])
    return np.concatenate(predictions), np.concatenate(truth), blocks


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
    started = time.perf_counter()
    for config in ("M1", "M2"):
        fitted[config], tuning[config] = select_and_fit(config, tables[config]["train"])
    fit_seconds = time.perf_counter() - started

    rows = []
    for target in ("M1", "M2"):
        source = "M2" if target == "M1" else "M1"
        direction = f"{source}->{target}"
        test = tables[target]["test"]
        scale = tuning[source]["source_state_scale_cm"]
        scale = np.asarray(scale, dtype=float)

        source_models = dict(fitted[source])
        source_models[ZeroOrderHold.name] = ZeroOrderHold()
        for family, model in source_models.items():
            one_step_pred, one_step_truth = evaluate_one_step(model, test)
            rows.extend(metric_rows(
                one_step_pred, one_step_truth, scale, direction=direction,
                task="teacher_forced_one_step_diagnostic", horizon_seconds=0.1,
                method=f"source_{family}", n_blocks=0,
            ))
            for stride in STRIDES:
                pred, truth, blocks = horizon_errors(model, test, stride, scale)
                rows.extend(metric_rows(
                    pred, truth, scale, direction=direction, task="causal_free_rollout",
                    horizon_seconds=stride / 10, method=f"source_{family}", n_blocks=blocks,
                ))
                if family != ZeroOrderHold.name:
                    bridge_pred, bridge_truth, bridge_blocks = evaluate_endpoint_bridge(model, test, stride)
                    rows.extend(metric_rows(
                        bridge_pred, bridge_truth, scale, direction=direction,
                        task="offline_endpoint_conditioned_bridge", horizon_seconds=stride / 10,
                        method=f"source_{family}", n_blocks=bridge_blocks,
                    ))

        for stride in STRIDES:
            pred, truth, blocks = evaluate_endpoint_interpolation(test, stride)
            rows.extend(metric_rows(
                pred, truth, scale, direction=direction, task="offline_endpoint_conditioned_bridge",
                horizon_seconds=stride / 10, method="linear_endpoint_interpolation", n_blocks=blocks,
            ))

        # A dense target-training reference is included only as an upper reference;
        # it is not a sparse-calibration result and is selected without target test data.
        for family, model in fitted[target].items():
            one_step_pred, one_step_truth = evaluate_one_step(model, test)
            rows.extend(metric_rows(
                one_step_pred, one_step_truth, scale, direction=direction,
                task="teacher_forced_one_step_diagnostic", horizon_seconds=0.1,
                method=f"target_dense_train_reference_{family}", n_blocks=0,
            ))
            for stride in STRIDES:
                pred, truth, blocks = horizon_errors(model, test, stride, scale)
                rows.extend(metric_rows(
                    pred, truth, scale, direction=direction, task="causal_free_rollout",
                    horizon_seconds=stride / 10,
                    method=f"target_dense_train_reference_{family}", n_blocks=blocks,
                ))

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["task", "horizon_seconds", "method", "state"], as_index=False)
        [["rmse_cm", "mae_cm", "rmse_over_source_train_std", "predicted_below_zero_fraction"]]
        .mean()
        .sort_values(["task", "horizon_seconds", "state", "rmse_over_source_train_std"])
    )
    metadata = {
        "experiment": "InTaSet M1<->M2 causal cross-configuration system identification",
        "task_contract": {
            "native_sample_interval_seconds": 0.1,
            "target_state_observation_intervals_seconds": [1, 5, 10],
            "pump_input_interval_seconds": 0.1,
            "causal_rollout": "At each sparse test block start, expose the four measured states and then the recorded high-rate pump sequence. Score every prediction through the block before using the next sparse state.",
            "offline_bridge": "Separately scores interpolation using both observed endpoints; endpoint-conditioned bridges are not causal forecasts.",
            "teacher_forced_one_step": "Diagnostic only: each next-state prediction uses the previous true test state; not a sparse-observation deployment score.",
            "target_dense_reference": "Trained with the full target configuration training trace; an upper reference, not limited target calibration.",
        },
        "models": {
            "linear_state_space_arx": "Fully observed first-order discrete state-space/ARX model x[t+1]=x[t]+g(x[t],u[t]); Ridge regularization is selected on source validation rollouts.",
            "discrete_sindy_c_stlsq_degree2": "Degree-two polynomial library over four states and two pump inputs; sequential thresholded least squares on state increments. This is a discrete SINDYc-style baseline, not a continuous-time SINDYc replication.",
            "zero_order_hold": "Causal reference that keeps the last sparse observed state constant until the next observation.",
            "linear_endpoint_interpolation": "Offline-only interpolation between observed boundary states.",
        },
        "selection": "Each configuration's final 20% training segment is a chronological validation tail. Hyperparameters are selected by the mean normalized 1/5/10-second free-rollout RMSE, then refit on the full configuration training trace.",
        "metrics": "Raw cm errors, errors normalized by source training state standard deviations, negative-prediction fractions, and minimum predicted level are reported. No unverified upper water-level limit is imposed.",
        "tuning": tuning,
        "source_manifest": manifest,
        "training_wall_seconds_for_both_configs": fit_seconds,
        "limitations": [
            "The InTaSet author Neural ODE and the 2024 four-tank kernel interpolation methods are not yet reproduced in this runner.",
            "Target calibration is not included. Target dense training is clearly labeled as an upper reference.",
            "Only two configurations of one water-tank testbed are evaluated here; motor data remain a separate external system.",
            "A full-rank sampled regression matrix does not prove practical or structural identifiability.",
            "The provided test traces are short; block-level scores are descriptive and are not independent experimental replicates.",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "openpyxl": openpyxl.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "causal_transfer_metrics.csv", index=False, float_format="%.10g")
    summary.to_csv(output_dir / "causal_transfer_summary.csv", index=False, float_format="%.10g")
    metadata["detail_csv_sha256"] = hashlib.sha256((output_dir / "causal_transfer_metrics.csv").read_bytes()).hexdigest()
    metadata["summary_csv_sha256"] = hashlib.sha256((output_dir / "causal_transfer_summary.csv").read_bytes()).hexdigest()
    (output_dir / "causal_transfer_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return detail, summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    detail, summary, metadata = run(args.output_dir)
    causal = summary[(summary["task"] == "causal_free_rollout") & (summary["state"] == "ALL")]
    print(causal.to_string(index=False))
    print(f"detail_rows={len(detail)}")
    print(f"script_sha256={metadata['software']['script_sha256']}")
    print(f"detail_csv_sha256={metadata['detail_csv_sha256']}")
    print(f"summary_csv_sha256={metadata['summary_csv_sha256']}")


if __name__ == "__main__":
    main()
