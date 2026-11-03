#!/usr/bin/env python3
"""Run a source-selected ridge ARX(2,2) sparse-output ablation on Cascaded Tanks."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs/algorithmic-control-v1/external-source/cascaded_tanks"
ARCHIVE_PATH = SOURCE_DIR / "CascadedTanksFiles.zip"
PROTOCOL_PATH = ROOT / "docs/algorithmic-control-v1/CASCADED_TANKS_ARX2_REGULARIZED_PROTOCOL_V1_20261103.md"
OUTPUT_DIR = ROOT / "docs/algorithmic-control-v1/results/cascaded_tanks_arx2_regularized_v1"
EXPECTED_ARCHIVE_SHA256 = "eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51"
EXPECTED_MAT_SHA256 = "cb2f88d4388be4d3f2a24c6402fba804976aac5f2e1f26cda59ea0a38d016eab"
DT_SECONDS = 4.0
HORIZONS = (1, 3, 5)
RIDGE_ALPHAS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SOURCE_VALIDATION_FRACTION = 0.2
OUTPUT_FILES = (
    "arx2_ridge_selection.csv",
    "arx2_coefficients.json",
    "arx2_sparse_blocks.csv",
    "arx2_sparse_metrics.csv",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    archive_bytes = ARCHIVE_PATH.read_bytes()
    archive_hash = sha256_bytes(archive_bytes)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"unexpected archive SHA-256: {archive_hash}")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = "CascadedTanksFiles/dataBenchmark.mat"
        member_bytes = archive.read(member)
    member_hash = sha256_bytes(member_bytes)
    if member_hash != EXPECTED_MAT_SHA256:
        raise ValueError(f"unexpected MAT member SHA-256: {member_hash}")
    data = loadmat(io.BytesIO(member_bytes))
    u_est = np.asarray(data["uEst"], dtype=float).reshape(-1)
    y_est = np.asarray(data["yEst"], dtype=float).reshape(-1)
    u_val = np.asarray(data["uVal"], dtype=float).reshape(-1)
    y_val = np.asarray(data["yVal"], dtype=float).reshape(-1)
    if not all(len(values) == 1024 and np.isfinite(values).all() for values in (u_est, y_est, u_val, y_val)):
        raise ValueError("Cascaded Tanks records must each contain 1,024 finite samples")
    if float(np.asarray(data["Ts"]).reshape(-1)[0]) != DT_SECONDS:
        raise ValueError("Cascaded Tanks sample interval changed")
    return u_est, y_est, u_val, y_val, {
        "archive_sha256": archive_hash,
        "mat_member": member,
        "mat_member_sha256": member_hash,
        "sampling_interval_seconds": DT_SECONDS,
        "source_samples": len(y_est),
        "target_samples": len(y_val),
    }


def regression_arrays(u: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # k = 1 .. N-2, so all ARX(2,2) regressors and y[k+1] exist.
    rows = [
        [y[k], y[k - 1], u[k], u[k - 1]]
        for k in range(1, len(y) - 1)
    ]
    return np.asarray(rows, dtype=float), np.asarray(y[2:], dtype=float)


def fit_ridge_arx2(u: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    x, target = regression_arrays(u, y)
    x_mean = np.mean(x, axis=0)
    x_scale = np.std(x, axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    target_mean = float(np.mean(target))
    target_scale = max(float(np.std(target)), 1e-12)
    normalized_x = (x - x_mean) / x_scale
    normalized_target = (target - target_mean) / target_scale
    if alpha == 0:
        normalized_coef = np.linalg.lstsq(normalized_x, normalized_target, rcond=None)[0]
    else:
        normal_matrix = normalized_x.T @ normalized_x + float(alpha) * np.eye(normalized_x.shape[1])
        normalized_coef = np.linalg.solve(normal_matrix, normalized_x.T @ normalized_target)
    coefficients = normalized_coef * target_scale / x_scale
    intercept = target_mean - float(np.dot(x_mean, coefficients))
    design = np.column_stack((np.ones(len(x)), x))
    return {
        "alpha": float(alpha),
        "intercept": intercept,
        "coefficients": coefficients,
        "rank": int(np.linalg.matrix_rank(design)),
        "condition_number_column_normalized": float(np.linalg.cond(normalized_x)),
        "n_training_transitions": int(len(target)),
        "feature_means": x_mean.tolist(),
        "feature_scales": x_scale.tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


def block_forecast(
    u_all: np.ndarray,
    y_all: np.ndarray,
    coefficient: dict,
    horizon: int,
    *,
    offset: int = 0,
    available_length: int | None = None,
) -> list[dict]:
    stop = len(y_all) if available_length is None else offset + int(available_length)
    starts = np.arange(offset, stop - horizon, horizon, dtype=int)
    rows: list[dict] = []
    previous_output = float(y_all[offset - 1]) if offset > 0 else float(y_all[0])
    current_output = float(y_all[offset])
    for block_index, start in enumerate(starts):
        # Only y[start] is read at block boundaries; carry the hidden lag from prior model predictions.
        observed_start = float(y_all[start])
        current_output = observed_start
        initial_previous_output = previous_output
        for step in range(horizon):
            index = start + step
            previous_input = float(u_all[index - 1]) if index > 0 else float(u_all[0])
            features = np.asarray(
                [current_output, previous_output, float(u_all[index]), previous_input],
                dtype=float,
            )
            prediction = float(coefficient["intercept"] + np.dot(coefficient["coefficients"], features))
            truth = float(y_all[index + 1])
            rows.append({
                "task": "causal_block_rollout_given_recorded_pump_path",
                "method": "source_ridge_arx_2_2_frozen",
                "validation_or_test": "source_chronological_validation" if offset else "public_test_development",
                "alpha": float(coefficient["alpha"]),
                "horizon_transitions": int(horizon),
                "horizon_seconds": float(horizon * DT_SECONDS),
                "block_index": int(block_index),
                "start_index": int(start),
                "step_in_block": int(step + 1),
                "prediction": prediction,
                "truth": truth,
                "input_current": float(u_all[index]),
                "input_previous": previous_input,
                "output_initial_current": observed_start,
                "output_initial_previous": initial_previous_output,
            })
            previous_output, current_output = current_output, prediction
    return rows


def summarize_blocks(blocks: pd.DataFrame, source_std: float, evaluation: str) -> pd.DataFrame:
    rows = []
    for horizon, group in blocks.loc[blocks.validation_or_test.eq(evaluation)].groupby("horizon_transitions", sort=True):
        prediction = group["prediction"].to_numpy(dtype=float)
        truth = group["truth"].to_numpy(dtype=float)
        finite = np.isfinite(prediction) & np.isfinite(truth)
        error = prediction[finite] - truth[finite]
        rmse = float(np.sqrt(np.mean(error**2))) if len(error) else float("nan")
        rows.append({
            "method": "source_ridge_arx_2_2_frozen",
            "evaluation": evaluation,
            "alpha": float(group["alpha"].iloc[0]),
            "horizon_transitions": int(horizon),
            "horizon_seconds": float(horizon * DT_SECONDS),
            "n_blocks": int(group["block_index"].nunique()),
            "n_scored_steps": int(len(group)),
            "finite_prediction_fraction": float(finite.mean()) if len(finite) else float("nan"),
            "rmse_sensor_units": rmse,
            "mae_sensor_units": float(np.mean(np.abs(error))) if len(error) else float("nan"),
            "nrmse_over_full_estimation_output_std": rmse / max(source_std, 1e-12),
            "prediction_min": float(np.min(prediction)) if len(prediction) else float("nan"),
            "prediction_max": float(np.max(prediction)) if len(prediction) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["evaluation", "horizon_transitions"]).reset_index(drop=True)


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    u_est, y_est, u_val, y_val, data_receipt = load_data()
    split = int(np.floor((1.0 - SOURCE_VALIDATION_FRACTION) * (len(y_est) - 1)))
    if split < 20 or len(y_est) - split < 20:
        raise ValueError("source record too short for the frozen chronological validation split")
    u_train = u_est[: split + 1]
    y_train = y_est[: split + 1]
    source_train_std = max(float(np.std(y_train)), 1e-12)

    selection_rows: list[dict] = []
    for alpha in RIDGE_ALPHAS:
        candidate = fit_ridge_arx2(u_train, y_train, alpha)
        validation_blocks: list[dict] = []
        for horizon in HORIZONS:
            validation_blocks.extend(block_forecast(
                u_est,
                y_est,
                candidate,
                horizon,
                offset=split,
                available_length=len(y_est) - split,
            ))
        validation_metrics = summarize_blocks(pd.DataFrame(validation_blocks), source_train_std, "source_chronological_validation")
        by_horizon = {
            int(row.horizon_transitions): float(row.nrmse_over_full_estimation_output_std)
            for row in validation_metrics.itertuples(index=False)
        }
        selection_rows.append({
            "alpha": float(alpha),
            "source_split_transition_index": int(split),
            "source_training_samples": int(split + 1),
            "source_validation_samples": int(len(y_est) - split),
            "source_train_output_std": source_train_std,
            "validation_nrmse_h1": by_horizon[1],
            "validation_nrmse_h3": by_horizon[3],
            "validation_nrmse_h5": by_horizon[5],
            "validation_macro_nrmse": float(np.mean(list(by_horizon.values()))),
            "design_rank": int(candidate["rank"]),
            "design_condition_number_column_normalized": float(candidate["condition_number_column_normalized"]),
        })
    selection = pd.DataFrame(selection_rows).sort_values(["validation_macro_nrmse", "alpha"]).reset_index(drop=True)
    selected_alpha = float(selection.iloc[0]["alpha"])
    final_fit = fit_ridge_arx2(u_est, y_est, selected_alpha)

    test_rows: list[dict] = []
    for horizon in HORIZONS:
        test_rows.extend(block_forecast(u_val, y_val, final_fit, horizon))
    test_blocks = pd.DataFrame(test_rows)
    test_metrics = summarize_blocks(test_blocks, float(np.std(y_est)), "public_test_development")

    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_dir / OUTPUT_FILES[0], index=False, float_format="%.12g")
    coefficient_record = {
        "model": "ARX(2,2): y[k+1] = c + a1*y[k] + a2*y[k-1] + b0*u[k] + b1*u[k-1]",
        "feature_order": ["y[k]", "y[k-1]", "u[k]", "u[k-1]"],
        "intercept": float(final_fit["intercept"]),
        "coefficients": [float(x) for x in final_fit["coefficients"]],
        "selected_ridge_alpha": selected_alpha,
        "fit_rows": final_fit["n_training_transitions"],
        "design_rank": final_fit["rank"],
        "design_condition_number_column_normalized": final_fit["condition_number_column_normalized"],
        "training_record": "uEst/yEst only; refit on the full official estimation record after source-only alpha selection",
        "target_test_labels_previously_seen": True,
        "sparse_block_output_initialization": "overwrite y[k] with the allowed block-start measurement; carry the y[k-1] estimate propagated from prior model steps",
        "input_lag_boundary": "u[k-1] is read from the recorded input path; at k=0 use u[0]",
    }
    (output_dir / OUTPUT_FILES[1]).write_text(json.dumps(coefficient_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    test_blocks.to_csv(output_dir / OUTPUT_FILES[2], index=False, float_format="%.12g")
    test_metrics.to_csv(output_dir / OUTPUT_FILES[3], index=False, float_format="%.12g")

    manifest = {
        "status": "COMPLETED_PUBLIC_TEST_DEVELOPMENT_ABLATION",
        "target_labels_previously_seen": True,
        "independent_or_blind_confirmation": False,
        "physical_test_records": 1,
        "model": coefficient_record["model"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "data_receipt": data_receipt,
        "source_validation": {
            "split_transition_index": int(split),
            "fraction": SOURCE_VALIDATION_FRACTION,
            "candidate_alphas": list(RIDGE_ALPHAS),
            "selected_alpha": selected_alpha,
            "selection_rule": "minimum mean source-validation NRMSE over horizons 1, 3, and 5; ties choose smaller alpha",
        },
        "target_protocol": {
            "target_calibration_budget_transitions": 0,
            "horizons_transitions": list(HORIZONS),
            "sampling_interval_seconds": DT_SECONDS,
            "output_reset": "at each non-overlapping block start only",
            "incomplete_final_block_scored": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "output_files": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_file(output_dir / name),
            }
            for name in OUTPUT_FILES
        },
    }
    (output_dir / "arx2_ridge_metadata.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "selected_alpha": selected_alpha,
        "validation_macro_nrmse": float(selection.iloc[0]["validation_macro_nrmse"]),
        "test_metrics": test_metrics.to_dict(orient="records"),
        "output_dir": str(output_dir),
    }, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    run()
