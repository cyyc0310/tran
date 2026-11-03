#!/usr/bin/env python3
"""Run a fixed sparse-output/known-pump-input extension on Cascaded Tanks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs/algorithmic-control-v1/external-source/cascaded_tanks"
MANIFEST_PATH = SOURCE_DIR / "acquisition_manifest.json"
PROTOCOL_PATH = ROOT / "docs/algorithmic-control-v1/CASCADED_TANKS_SPARSE_INPUT_PROTOCOL_V1_20261103.md"
OUTPUT_DIR = ROOT / "docs/algorithmic-control-v1/results/cascaded_tanks_sparse_input_v1"
DT_SECONDS = 4.0
BUDGET_TRANSITIONS = (1, 3, 5)
HORIZON_TRANSITIONS = (1, 3, 5)
SINDY_THRESHOLDS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
VALIDATION_FRACTION = 0.2


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def library(y: np.ndarray, u: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    return np.column_stack((np.ones_like(y), y, u, y**2, y * u, u**2))


def fit_arx(y: np.ndarray, u: np.ndarray) -> dict:
    design = np.column_stack((np.ones(len(y) - 1), y[:-1], u[:-1]))
    coef, _, rank, singular = np.linalg.lstsq(design, y[1:], rcond=None)
    return {
        "family": "linear_arx_1_0",
        "coef": coef,
        "fit_rank": int(rank),
        "fit_singular_values": singular.tolist(),
    }


def sindy_design(y: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi = library(y[:-1], u[:-1])
    derivative = (y[1:] - y[:-1]) / DT_SECONDS
    return phi, derivative


def fit_stlsq(
    y: np.ndarray,
    u: np.ndarray,
    threshold: float,
    selected_support: list[int] | None = None,
) -> dict:
    phi, derivative = sindy_design(y, u)
    scales = np.sqrt(np.mean(phi**2, axis=0))
    scales[scales < 1e-12] = 1.0
    scales[0] = 1.0
    target_scale = max(float(np.std(derivative)), 1e-12)
    normalized = phi / scales
    target = derivative / target_scale
    if selected_support is not None:
        active = np.zeros(normalized.shape[1], dtype=bool)
        active[np.asarray(selected_support, dtype=int)] = True
        active[0] = True
    else:
        active = np.ones(normalized.shape[1], dtype=bool)
        active[0] = True
        for _ in range(normalized.shape[1] + 1):
            candidate = np.zeros(normalized.shape[1], dtype=float)
            candidate[active] = np.linalg.lstsq(normalized[:, active], target, rcond=None)[0]
            keep = np.abs(candidate) >= threshold
            keep[0] = True
            if np.array_equal(keep, active):
                break
            active = keep
    candidate = np.zeros(normalized.shape[1], dtype=float)
    candidate[active] = np.linalg.lstsq(normalized[:, active], target, rcond=None)[0]
    coef = candidate * target_scale / scales
    return {
        "family": "sindy_c_style_poly2_stlsq",
        "coef": coef,
        "threshold": float(threshold),
        "active_terms": np.flatnonzero(active).astype(int).tolist(),
        "feature_scales": scales.tolist(),
        "target_scale": target_scale,
    }


def one_step(model: dict, y: float, u: float) -> float:
    if model["family"] == "linear_arx_1_0":
        return float(np.dot(np.array([1.0, y, u]), model["coef"]))
    derivative = float(np.dot(library(np.array([y]), np.array([u]))[0], model["coef"]))
    return float(y + DT_SECONDS * derivative)


def rollout(model: dict, y0: float, inputs: np.ndarray) -> np.ndarray:
    result = np.empty(len(inputs), dtype=float)
    state = float(y0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for index, value in enumerate(inputs):
            if not np.isfinite(state):
                result[index:] = np.nan
                break
            state = one_step(model, state, float(value))
            result[index] = state
    return result


def source_validation_score(model: dict, y: np.ndarray, u: np.ndarray, first_index: int) -> float:
    scores: list[float] = []
    scale = max(float(np.std(y[:first_index])), 1e-12)
    for horizon in HORIZON_TRANSITIONS:
        starts = np.arange(first_index, len(y) - horizon, horizon, dtype=int)
        if len(starts) == 0:
            continue
        pred_values: list[float] = []
        true_values: list[float] = []
        for start in starts:
            pred = rollout(model, float(y[start]), u[start : start + horizon])
            pred_values.extend(pred.tolist())
            true_values.extend(y[start + 1 : start + horizon + 1].tolist())
        pred_array = np.asarray(pred_values, dtype=float)
        true_array = np.asarray(true_values, dtype=float)
        if not np.isfinite(pred_array).all():
            scores.append(float("inf"))
        else:
            scores.append(float(np.sqrt(np.mean((pred_array - true_array) ** 2)) / scale))
    return float(np.mean(scores)) if scores else float("inf")


def choose_sindy_threshold(y: np.ndarray, u: np.ndarray) -> tuple[float, list[dict], list[int]]:
    transition_count = len(y) - 1
    split = int(np.floor((1.0 - VALIDATION_FRACTION) * transition_count))
    if split < 20 or transition_count - split < 20:
        raise ValueError("source record is too short for the frozen chronological split")
    train_y = y[: split + 1]
    train_u = u[: split + 1]
    validation_rows: list[dict] = []
    validation_scores: list[float] = []
    for threshold in SINDY_THRESHOLDS:
        candidate = fit_stlsq(train_y, train_u, threshold)
        score = source_validation_score(candidate, y, u, split)
        validation_scores.append(score if np.isfinite(score) else float("inf"))
        validation_rows.append({
            "threshold": threshold,
            "source_chronological_validation_macro_nrmse": score if np.isfinite(score) else None,
            "active_terms": candidate["active_terms"],
        })
    selected_index = min(
        range(len(validation_rows)),
        key=lambda index: (validation_scores[index], validation_rows[index]["threshold"]),
    )
    selected_row = validation_rows[selected_index]
    return float(selected_row["threshold"]), validation_rows, list(selected_row["active_terms"])


def fit_gamma(model: dict, y: np.ndarray, u: np.ndarray, budget: int) -> dict:
    base = rollout(model, float(y[0]), u[:budget])
    if not np.isfinite(base).all():
        return {
            "gamma": 1.0,
            "fallback": "non_finite_prefix_rollout",
            "prefix_base_rmse": None,
            "prefix_calibrated_rmse": None,
        }
    source_delta = base - float(y[0])
    target_delta = y[1 : budget + 1] - float(y[0])
    denominator = float(np.dot(source_delta, source_delta))
    if denominator <= 1e-18:
        gamma = 1.0
        fallback = "zero_source_path_displacement"
    else:
        gamma = float(np.dot(source_delta, target_delta) / denominator)
        fallback = None
        if not np.isfinite(gamma):
            gamma = 1.0
            fallback = "non_finite_least_squares_gain"
    calibrated = y[0] + gamma * source_delta
    return {
        "gamma": gamma,
        "fallback": fallback,
        "prefix_base_rmse": float(np.sqrt(np.mean((y[1 : budget + 1] - (y[0] + source_delta)) ** 2))),
        "prefix_calibrated_rmse": float(np.sqrt(np.mean((y[1 : budget + 1] - calibrated) ** 2))),
    }


def make_blocks(
    model: dict | None,
    y: np.ndarray,
    u: np.ndarray,
    horizon: int,
    budget: int,
    *,
    gamma: float = 1.0,
    method: str,
) -> list[dict]:
    starts = np.arange(budget, len(y) - horizon, horizon, dtype=int)
    rows: list[dict] = []
    for block_index, start in enumerate(starts):
        y0 = float(y[start])
        truth = y[start + 1 : start + horizon + 1]
        if method == "zero_order_hold":
            prediction = np.full(horizon, y0, dtype=float)
        elif method == "linear_endpoint_interpolation_oracle":
            y_end = float(y[start + horizon])
            prediction = np.linspace(y0, y_end, horizon + 1, dtype=float)[1:]
        else:
            if model is None:
                raise ValueError("model must be provided for a dynamics method")
            base = rollout(model, y0, u[start : start + horizon])
            prediction = y0 + gamma * (base - y0)
        for step, (pred, actual) in enumerate(zip(prediction, truth), start=1):
            rows.append({
                "task": "offline_endpoint_interpolation_oracle" if method == "linear_endpoint_interpolation_oracle" else "causal_block_rollout_given_recorded_pump_path",
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
                "gamma": float(gamma),
            })
    return rows


def summarize(block_rows: pd.DataFrame, source_y: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["task", "method", "calibration_budget_transitions", "calibration_budget_seconds", "horizon_transitions", "horizon_seconds"]
    source_min, source_max = float(source_y.min()), float(source_y.max())
    for key, group in block_rows.groupby(keys, dropna=False, sort=True):
        pred = group["prediction"].to_numpy(dtype=float)
        truth = group["truth"].to_numpy(dtype=float)
        finite = np.isfinite(pred)
        diff = pred[finite] - truth[finite]
        rmse = float(np.sqrt(np.mean(diff**2))) if len(diff) else float("nan")
        mae = float(np.mean(np.abs(diff))) if len(diff) else float("nan")
        task, method, budget_steps, budget_seconds, horizon_steps, horizon_seconds = key
        rows.append({
            "task": task,
            "method": method,
            "calibration_budget_transitions": int(budget_steps),
            "calibration_budget_seconds": float(budget_seconds),
            "horizon_transitions": int(horizon_steps),
            "horizon_seconds": float(horizon_seconds),
            "n_scored_values": int(len(pred)),
            "n_finite_values": int(finite.sum()),
            "finite_prediction_fraction": float(finite.mean()) if len(finite) else float("nan"),
            "n_blocks": int(group["block_index"].nunique()),
            "diverged_blocks": int(group.loc[~finite, "block_index"].nunique()),
            "rmse_sensor_units": rmse,
            "mae_sensor_units": mae,
            "nrmse_over_estimation_output_std": rmse / max(float(np.std(source_y)), 1e-12),
            "prediction_outside_source_output_range_fraction": float(np.mean((pred < source_min) | (pred > source_max))) if len(pred) else float("nan"),
            "prediction_beyond_observed_saturation_10_fraction": float(np.mean(pred > 10.0)) if len(pred) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["task", "horizon_transitions", "calibration_budget_transitions", "method"])


def excitation_record(y: np.ndarray, u: np.ndarray, label: str, transitions: int | None = None) -> dict:
    if transitions is None:
        transitions = len(y) - 1
    transitions = min(int(transitions), len(y) - 1)
    yt = y[: transitions + 1]
    ut = u[:transitions]
    arx = np.column_stack((np.ones(transitions), yt[:-1], ut))
    phi = library(yt[:-1], ut)
    output: dict[str, dict] = {}
    for name, design in (("linear_arx_update", arx), ("sindy_c_style_poly2", phi)):
        scales = np.linalg.norm(design, axis=0)
        scales[scales < 1e-12] = 1.0
        normalized = design / scales
        singular = np.linalg.svd(normalized, compute_uv=False)
        rank = int(np.linalg.matrix_rank(normalized))
        condition = float(singular[0] / singular[-1]) if rank == normalized.shape[1] and len(singular) >= normalized.shape[1] else None
        output[name] = {
            "rows": int(design.shape[0]),
            "columns": int(design.shape[1]),
            "rank": rank,
            "condition_2_if_full_column_rank": condition,
            "singular_values_column_normalized": singular.tolist(),
        }
    spectrum = np.abs(np.fft.rfft(ut - np.mean(ut))) / max(len(ut), 1)
    frequencies = np.fft.rfftfreq(len(ut), d=DT_SECONDS)
    dominant = np.argsort(spectrum)[-10:][::-1]
    return {
        "record": label,
        "transitions": transitions,
        "output_observations": transitions + 1,
        "input_min": float(ut.min()) if len(ut) else None,
        "input_max": float(ut.max()) if len(ut) else None,
        "input_std": float(np.std(ut)) if len(ut) else None,
        "linear_algebra_diagnostics": output,
        "dominant_input_fft_bins": [
            {"frequency_hz": float(frequencies[i]), "amplitude": float(spectrum[i])}
            for i in dominant if spectrum[i] > 0
        ],
    }


def run(output_dir: Path = OUTPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text())
    archive_path = SOURCE_DIR / manifest["archive"]["path"]
    archive_bytes = archive_path.read_bytes()
    if hashlib.md5(archive_bytes).hexdigest() != manifest["archive"]["expected_md5_from_4tu_record"]:
        raise ValueError("Cascaded Tanks source archive MD5 check failed")
    if sha256_bytes(archive_bytes) != manifest["archive"]["sha256"]:
        raise ValueError("Cascaded Tanks source archive SHA-256 check failed")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        data = loadmat(io.BytesIO(archive.read("CascadedTanksFiles/dataBenchmark.mat")))
    u_est = np.asarray(data["uEst"], dtype=float).reshape(-1)
    y_est = np.asarray(data["yEst"], dtype=float).reshape(-1)
    u_val = np.asarray(data["uVal"], dtype=float).reshape(-1)
    y_val = np.asarray(data["yVal"], dtype=float).reshape(-1)
    if len({len(u_est), len(y_est), len(u_val), len(y_val)}) != 1 or float(data["Ts"][0, 0]) != DT_SECONDS:
        raise ValueError("source data schema differs from the frozen protocol")

    threshold, threshold_validation, selected_support = choose_sindy_threshold(y_est, u_est)
    arx_model = fit_arx(y_est, u_est)
    sindy_model = fit_stlsq(y_est, u_est, threshold, selected_support=selected_support)
    models = {"source_linear_arx_1_0": arx_model, "source_sindy_c_style_poly2_stlsq": sindy_model}

    rows: list[dict] = []
    receipts: list[dict] = []
    for budget in (0,) + BUDGET_TRANSITIONS:
        for model_name, model in models.items():
            base_method = f"{model_name}_frozen"
            gammas = [(1.0, base_method)]
            if budget > 0:
                fit = fit_gamma(model, y_val, u_val, budget)
                gamma_method = f"{model_name}_gamma_calibrated"
                gammas.append((float(fit["gamma"]), gamma_method))
                receipts.append({
                    "model": model_name,
                    "method": gamma_method,
                    "calibration_budget_transitions": int(budget),
                    "calibration_budget_seconds": float(budget * DT_SECONDS),
                    "target_state_indices_used": list(range(budget + 1)),
                    "target_input_indices_used": list(range(budget)),
                    **fit,
                })
            for horizon in HORIZON_TRANSITIONS:
                for gamma, method in gammas:
                    rows.extend(make_blocks(
                        model, y_val, u_val, horizon, budget,
                        gamma=gamma,
                        method=method,
                    ))
        for horizon in HORIZON_TRANSITIONS:
            rows.extend(make_blocks(None, y_val, u_val, horizon, budget, method="zero_order_hold"))
            rows.extend(make_blocks(None, y_val, u_val, horizon, budget, method="linear_endpoint_interpolation_oracle"))

    block_rows = pd.DataFrame(rows)
    metrics = summarize(block_rows, y_est)
    gamma_receipts = pd.DataFrame(receipts)
    diagnostics = {
        "sampling_interval_seconds": DT_SECONDS,
        "source_input_and_regressor_diagnostics": excitation_record(y_est, u_est, "estimation", None),
        "target_prefix_diagnostics": {
            str(budget): excitation_record(y_val, u_val, f"validation_prefix_{budget}_transitions", budget)
            for budget in BUDGET_TRANSITIONS
        },
        "source_output_range": [float(y_est.min()), float(y_est.max())],
        "target_output_range": [float(y_val.min()), float(y_val.max())],
        "target_observed_value_10_count": int(np.sum(y_val == 10.0)),
        "interpretation": "matrix rank and condition describe these regressors and records only; 10.0 is observed saturation, not an independently verified physical limit",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cascaded_tanks_metrics.csv": metrics,
        "cascaded_tanks_blocks.csv": block_rows,
        "cascaded_tanks_gamma_receipts.csv": gamma_receipts,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False, float_format="%.12g")
    diagnostics_path = output_dir / "cascaded_tanks_excitation_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    script_path = Path(__file__).resolve()
    metadata = {
        "experiment": "Cascaded Tanks sparse-output known-pump-input transfer extension",
        "status": "development_evidence_public_target_outputs_were_inspected_during_source_admission",
        "official_benchmark_policy_boundary": "uVal/yVal are the official test record; this custom extension explicitly allows only the first 1/3/5 native transitions for scalar gamma calibration, then scores later block queries; these are not the official untouched-test benchmark scores",
        "raw_data": {
            "path": str(archive_path.relative_to(ROOT)),
            "bytes": len(archive_bytes),
            "md5": hashlib.md5(archive_bytes).hexdigest(),
            "sha256": sha256_bytes(archive_bytes),
            "doi": manifest["doi"],
            "license": manifest["license"],
        },
        "data_contract": {
            "directions": ["estimation_to_validation_trajectory"],
            "state_observation_interval_seconds": DT_SECONDS,
            "target_calibration_budgets_seconds": [int(b * DT_SECONDS) for b in BUDGET_TRANSITIONS],
            "prediction_horizons_seconds": [int(h * DT_SECONDS) for h in HORIZON_TRANSITIONS],
            "control_input": "recorded pump input uVal, zero-order held over 4-second samples; future input is known offline",
            "initialization": "measured lower-tank output at each scored block start",
            "test_output_after_prefix_used_for_training_or_model_selection": False,
            "calibration_task": "one scalar output-path gain per source model fitted to the target prefix",
            "statistical_unit": "one official validation record; blocks are within-record queries, not independent replicates",
        },
        "methods": {
            "source_linear_arx_1_0": "OLS fit y[k+1] = a*y[k] + b*u[k] + c on estimation record",
            "source_sindy_c_style_poly2_stlsq": "sparse quadratic regression of dy/dt on [1,y,u,y^2,y*u,u^2], threshold chosen only on a chronological estimation-record validation tail",
            "zero_order_hold": "block-start output held constant",
            "linear_endpoint_interpolation_oracle": "uses the measured block endpoint; offline oracle only, excluded from causal model ranking",
            "gamma_calibration": "least-squares scalar path multiplier on target prefix; no model weights refit",
            "author_model": "no dataset-specific author neural checkpoint was included; do not interpret independent project baselines as author-model reproduction",
        },
        "source_model_selection": {
            "sindy_candidate_thresholds": list(SINDY_THRESHOLDS),
            "chronological_validation_fraction": VALIDATION_FRACTION,
            "selected_sindy_threshold": threshold,
            "selected_sindy_support_from_validation": selected_support,
            "validation_scores": threshold_validation,
            "final_refit": "selected threshold refit on full official estimation record",
        },
        "source_model_fit_receipts": {
            "linear_arx_1_0": {
                "equation": "y[k+1] = c + a*y[k] + b*u[k]",
                "coefficients_intercept_y_u": arx_model["coef"].tolist(),
                "fit_rank": arx_model["fit_rank"],
                "fit_singular_values": arx_model["fit_singular_values"],
            },
            "sindy_c_style_poly2_stlsq": {
                "feature_order": ["1", "y", "u", "y^2", "y*u", "u^2"],
                "derivative_coefficients": sindy_model["coef"].tolist(),
                "selected_threshold": sindy_model["threshold"],
                "active_terms": sindy_model["active_terms"],
                "feature_scales": sindy_model["feature_scales"],
                "target_scale": sindy_model["target_scale"],
            },
        },
        "training_seeds_are_not_replicates": True,
        "interpretation_limits": [
            "only one official estimation record and one validation record",
            "public validation outputs were inspected in data admission; this is not blind confirmation",
            "the custom target-prefix calibration extension differs from the official untouched-test protocol",
            "the y=10 observation saturation is not a verified engineering safety limit",
            "recorded future pump input makes this offline conditional block prediction, not unknown-input forecasting or closed-loop validation",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_files": [
            {"path": str(script_path.relative_to(ROOT)), "sha256": sha256_file(script_path)},
            {"path": str(MANIFEST_PATH.relative_to(ROOT)), "sha256": sha256_file(MANIFEST_PATH)},
            {"path": str(PROTOCOL_PATH.relative_to(ROOT)), "sha256": sha256_file(PROTOCOL_PATH)},
        ],
        "outputs": {},
        "data_diagnostics": diagnostics,
    }
    for path in list(output_dir.glob("cascaded_tanks_*.csv")) + [diagnostics_path]:
        metadata["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    metadata_path = output_dir / "cascaded_tanks_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metrics, block_rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new or empty directory for this replay's outputs",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("--output-dir must be new or empty")

    metrics, blocks, metadata = run(output_dir)
    focus = metrics[metrics["horizon_transitions"].eq(5)]
    print(f"metric_rows={len(metrics)} block_rows={len(blocks)} gamma_receipts={len(pd.read_csv(output_dir / 'cascaded_tanks_gamma_receipts.csv'))}")
    print(focus[["method", "calibration_budget_seconds", "rmse_sensor_units", "nrmse_over_estimation_output_std", "n_blocks"]].to_string(index=False))
    print(f"output={output_dir}")
    print(f"raw_sha256={metadata['raw_data']['sha256']}")


if __name__ == "__main__":
    main()
