#!/usr/bin/env python3
"""Evaluate a pinned two-state Python N4SID model under sparse output observations."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs/algorithmic-control-v1/external-source/cascaded_tanks"
UPSTREAM_DIR = ROOT / "docs/algorithmic-control-v1/external-source/nfoursid"
UPSTREAM_COMMIT = "1e306879f9e697cd275f17d01dc96182b74ac97d"
ARCHIVE_PATH = SOURCE_DIR / "CascadedTanksFiles.zip"
PROTOCOL_PATH = ROOT / "docs/algorithmic-control-v1/CASCADED_TANKS_PYTHON_N4SID_SPARSE_PROTOCOL_V1_20261103.md"
DEFAULT_OUTPUT_DIR = ROOT / "docs/algorithmic-control-v1/results/cascaded_tanks_python_n4sid_sparse_v1"
EXPECTED_ARCHIVE_SHA256 = "eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51"
EXPECTED_MAT_SHA256 = "cb2f88d4388be4d3f2a24c6402fba804976aac5f2e1f26cda59ea0a38d016eab"
DT_SECONDS = 4.0
SPLIT_FRACTION = 0.8
STATE_ORDER = 2
BLOCK_ROWS = (2, 3, 4, 6, 8, 10, 12, 16)
HORIZONS = (1, 3, 5)
INITIAL_STATE_FIT_SAMPLES = 40
INITIAL_COVARIANCE_SAMPLES = 100
OUTPUT_FILES = (
    "python_n4sid_source_validation.csv",
    "python_n4sid_model.json",
    "python_n4sid_sparse_blocks.csv",
    "python_n4sid_sparse_metrics.csv",
)


sys.path.insert(0, str(UPSTREAM_DIR / "src"))
from nfoursid.kalman import Kalman  # noqa: E402
from nfoursid.nfoursid import NFourSID  # noqa: E402


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


def initial_condition_fit(state_space, u_norm: np.ndarray, y_norm: np.ndarray, n_samples: int) -> tuple[np.ndarray, float]:
    a, b, c, d = state_space.a, state_space.b, state_space.c, state_space.d
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    forced_state = np.zeros((STATE_ORDER, 1), dtype=float)
    observability_power = np.eye(STATE_ORDER, dtype=float)
    count = min(int(n_samples), len(y_norm))
    for index in range(count):
        rows.append(c @ observability_power)
        targets.append(np.asarray([[y_norm[index]]], dtype=float) - d * float(u_norm[index]) - c @ forced_state)
        forced_state = a @ forced_state + b * float(u_norm[index])
        observability_power = observability_power @ a
    design = np.concatenate(rows, axis=0)
    target = np.concatenate(targets, axis=0)
    state0 = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = design @ state0 - target
    rmse = float(np.sqrt(np.mean(residual**2)))
    if not np.isfinite(state0).all() or not np.isfinite(rmse):
        raise FloatingPointError("source initial-state estimate is not finite")
    return state0, rmse


def fit_python_n4sid(u: np.ndarray, y: np.ndarray, block_rows: int) -> dict:
    u_mean, u_scale = float(np.mean(u)), max(float(np.std(u)), 1e-12)
    y_mean, y_scale = float(np.mean(y)), max(float(np.std(y)), 1e-12)
    u_norm = (np.asarray(u, dtype=float) - u_mean) / u_scale
    y_norm = (np.asarray(y, dtype=float) - y_mean) / y_scale
    frame = pd.DataFrame({"u": u_norm, "y": y_norm})
    identifier = NFourSID(frame, output_columns=["y"], input_columns=["u"], num_block_rows=int(block_rows))
    identifier.subspace_identification()
    state_space, noise_covariance = identifier.system_identification(rank=STATE_ORDER)
    noise_covariance = (noise_covariance + noise_covariance.T) / 2.0
    if not all(np.isfinite(matrix).all() for matrix in (state_space.a, state_space.b, state_space.c, state_space.d, noise_covariance)):
        raise FloatingPointError(f"NFourSID produced non-finite matrices for block_rows={block_rows}")

    covariance_eigenvalues = np.linalg.eigvalsh(noise_covariance)
    if float(np.min(covariance_eigenvalues)) < -1e-8:
        raise FloatingPointError(f"NFourSID noise covariance is not positive semidefinite: {covariance_eigenvalues}")

    initial_state, initial_state_fit_rmse = initial_condition_fit(
        state_space,
        u_norm,
        y_norm,
        INITIAL_STATE_FIT_SAMPLES,
    )
    decomposition = identifier._get_observability_matrix_decomposition()
    state_values = np.maximum(np.diag(decomposition.eigenvalues), 0.0)
    state_sequence = np.sqrt(state_values)[:, None] * decomposition.right_orthogonal
    covariance_count = min(INITIAL_COVARIANCE_SAMPLES, state_sequence.shape[1])
    if covariance_count < 2:
        raise ValueError("too few NFourSID state estimates for initial covariance")
    initial_covariance = np.cov(state_sequence[:, :covariance_count], ddof=1)
    initial_covariance = (initial_covariance + initial_covariance.T) / 2.0
    covariance_floor = max(1e-9, abs(float(np.trace(initial_covariance))) * 1e-9)
    initial_covariance += covariance_floor * np.eye(STATE_ORDER)
    p_eigenvalues = np.linalg.eigvalsh(initial_covariance)
    if float(np.min(p_eigenvalues)) <= 0:
        initial_covariance += (abs(float(np.min(p_eigenvalues))) + covariance_floor) * np.eye(STATE_ORDER)

    return {
        "state_space": state_space,
        "noise_covariance": noise_covariance,
        "scaler": {"u_mean": u_mean, "u_scale": u_scale, "y_mean": y_mean, "y_scale": y_scale},
        "initial_state": initial_state,
        "initial_covariance": initial_covariance,
        "initial_state_fit_rmse_normalized": initial_state_fit_rmse,
        "noise_covariance_eigenvalues": covariance_eigenvalues,
        "initial_covariance_eigenvalues": np.linalg.eigvalsh(initial_covariance),
        "state_sequence": state_sequence,
    }


def initialized_filter(fit: dict) -> Kalman:
    kalman = Kalman(fit["state_space"], fit["noise_covariance"])
    # The pinned upstream Kalman class initializes x and P to zero and identity.
    # Seed its prior arrays with the source-estimated initial state/covariance.
    kalman.x_predicteds = [np.asarray(fit["initial_state"], dtype=float).copy()]
    kalman.p_predicteds = [np.asarray(fit["initial_covariance"], dtype=float).copy()]
    return kalman


def sparse_rollout(
    u_all: np.ndarray,
    y_all: np.ndarray,
    fit: dict,
    horizon: int,
    *,
    score_start: int,
    score_stop: int,
    fully_observed_prefix_stop: int,
    evaluation: str,
) -> list[dict]:
    scaler = fit["scaler"]
    u_norm = (np.asarray(u_all, dtype=float) - scaler["u_mean"]) / scaler["u_scale"]
    y_norm = (np.asarray(y_all, dtype=float) - scaler["y_mean"]) / scaler["y_scale"]
    kalman = initialized_filter(fit)
    for index in range(fully_observed_prefix_stop):
        kalman.step(
            np.asarray([[y_norm[index]]], dtype=float),
            np.asarray([[u_norm[index]]], dtype=float),
        )

    state_space = fit["state_space"]
    starts = np.arange(score_start, score_stop - horizon, horizon, dtype=int)
    rows: list[dict] = []
    for block_index, start in enumerate(starts):
        for step in range(horizon):
            index = int(start + step)
            observed = step == 0
            kalman.step(
                np.asarray([[y_norm[index]]], dtype=float) if observed else None,
                np.asarray([[u_norm[index]]], dtype=float),
            )
            next_index = min(index + 1, len(u_norm) - 1)
            next_prediction_norm = (
                state_space.c @ kalman.x_predicteds[-1]
                + state_space.d * float(u_norm[next_index])
            ).item()
            prediction = float(next_prediction_norm * scaler["y_scale"] + scaler["y_mean"])
            truth = float(y_all[index + 1])
            rows.append({
                "evaluation": evaluation,
                "method": "source_python_n4sid_order2_frozen",
                "state_order": STATE_ORDER,
                "num_block_rows": int(fit["num_block_rows"]),
                "horizon_transitions": int(horizon),
                "horizon_seconds": float(horizon * DT_SECONDS),
                "block_index": int(block_index),
                "start_index": index - step,
                "step_in_block": int(step + 1),
                "target_index": index + 1,
                "sparse_output_observed_at_step": bool(observed),
                "prediction": prediction,
                "truth": truth,
                "input_transition": float(u_all[index]),
                "input_for_direct_output_term": float(u_all[next_index]),
            })
    return rows


def summarize(blocks: pd.DataFrame, source_output_std: float, source_output_range: tuple[float, float]) -> pd.DataFrame:
    source_min, source_max = source_output_range
    rows: list[dict] = []
    for evaluation in sorted(blocks["evaluation"].unique()):
        subset = blocks.loc[blocks.evaluation.eq(evaluation)]
        for horizon, group in subset.groupby("horizon_transitions", sort=True):
            prediction = group["prediction"].to_numpy(dtype=float)
            truth = group["truth"].to_numpy(dtype=float)
            finite = np.isfinite(prediction) & np.isfinite(truth)
            errors = prediction[finite] - truth[finite]
            rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan")
            rows.append({
                "evaluation": evaluation,
                "method": "source_python_n4sid_order2_frozen",
                "state_order": STATE_ORDER,
                "num_block_rows": int(group["num_block_rows"].iloc[0]),
                "horizon_transitions": int(horizon),
                "horizon_seconds": float(horizon * DT_SECONDS),
                "n_blocks": int(group["block_index"].nunique()),
                "n_scored_steps": int(len(group)),
                "finite_prediction_fraction": float(finite.mean()) if len(finite) else float("nan"),
                "diverged_blocks_nonfinite": int(group.loc[~finite, "block_index"].nunique()),
                "rmse_sensor_units": rmse,
                "mae_sensor_units": float(np.mean(np.abs(errors))) if len(errors) else float("nan"),
                "nrmse_over_source_output_std": rmse / max(float(source_output_std), 1e-12),
                "negative_prediction_fraction": float(np.mean(prediction < 0.0)) if len(prediction) else float("nan"),
                "prediction_outside_source_range_fraction": float(np.mean((prediction < source_min) | (prediction > source_max))) if len(prediction) else float("nan"),
                "prediction_min": float(np.min(prediction)) if len(prediction) else float("nan"),
                "prediction_max": float(np.max(prediction)) if len(prediction) else float("nan"),
            })
    return pd.DataFrame(rows).sort_values(["evaluation", "horizon_transitions"]).reset_index(drop=True)


def summarize_with_source_range(blocks: pd.DataFrame, source_y: np.ndarray, source_output_std: float) -> pd.DataFrame:
    return summarize(
        blocks,
        source_output_std,
        (float(np.min(source_y)), float(np.max(source_y))),
    )


def fit_candidate_and_validate(u_est: np.ndarray, y_est: np.ndarray, split: int, block_rows: int) -> tuple[dict, list[dict]]:
    candidate = fit_python_n4sid(u_est[: split + 1], y_est[: split + 1], block_rows)
    candidate["num_block_rows"] = int(block_rows)
    validation_rows: list[dict] = []
    for horizon in HORIZONS:
        validation_rows.extend(sparse_rollout(
            u_est,
            y_est,
            candidate,
            horizon,
            score_start=split,
            score_stop=len(y_est),
            fully_observed_prefix_stop=split,
            evaluation="source_chronological_validation",
        ))
    validation_blocks = pd.DataFrame(validation_rows)
    metrics = summarize_with_source_range(validation_blocks, y_est[: split + 1], float(np.std(y_est[: split + 1])))
    candidate["validation_metrics"] = metrics
    return candidate, validation_rows


def select_block_rows(u_est: np.ndarray, y_est: np.ndarray) -> tuple[pd.DataFrame, dict[int, dict]]:
    split = int(np.floor(SPLIT_FRACTION * (len(y_est) - 1)))
    if split < 100 or len(y_est) - split < 20:
        raise ValueError("source record too short for the frozen N4SID chronological validation split")
    candidate_fits: dict[int, dict] = {}
    selection_rows: list[dict] = []
    for block_rows in BLOCK_ROWS:
        fit, _ = fit_candidate_and_validate(u_est, y_est, split, block_rows)
        candidate_fits[int(block_rows)] = fit
        by_horizon = {
            int(row.horizon_transitions): float(row.nrmse_over_source_output_std)
            for row in fit["validation_metrics"].itertuples(index=False)
        }
        selection_rows.append({
            "state_order": STATE_ORDER,
            "num_block_rows": int(block_rows),
            "source_validation_start_index": int(split),
            "source_training_samples_including_boundary": int(split + 1),
            "source_validation_samples": int(len(y_est) - split),
            "validation_nrmse_h1": by_horizon[1],
            "validation_nrmse_h3": by_horizon[3],
            "validation_nrmse_h5": by_horizon[5],
            "validation_macro_nrmse": float(np.mean(list(by_horizon.values()))),
            "spectral_radius_A": float(np.max(np.abs(np.linalg.eigvals(fit["state_space"].a)))),
            "D_direct_feedthrough_normalized": float(fit["state_space"].d.item()),
            "noise_covariance_min_eigenvalue": float(np.min(fit["noise_covariance_eigenvalues"])),
        })
    selection = pd.DataFrame(selection_rows).sort_values(["validation_macro_nrmse", "num_block_rows"]).reset_index(drop=True)
    return selection, candidate_fits


def serialize_fit(fit: dict, selected_block_rows: int) -> dict:
    state_space = fit["state_space"]
    scaler = fit["scaler"]
    poles = np.linalg.eigvals(state_space.a)
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "model_name": "two-state Python N4SID (NFourSID 1.0.2), frozen after source-only block-row selection",
        "state_order": STATE_ORDER,
        "num_block_rows": int(selected_block_rows),
        "A": state_space.a.tolist(),
        "B": state_space.b.tolist(),
        "C": state_space.c.tolist(),
        "D": state_space.d.tolist(),
        "noise_covariance": fit["noise_covariance"].tolist(),
        "initial_state_source_estimate": fit["initial_state"].tolist(),
        "initial_state_covariance_source_estimate": fit["initial_covariance"].tolist(),
        "source_scaler": scaler,
        "initial_state_fit_rmse_normalized": fit["initial_state_fit_rmse_normalized"],
        "noise_covariance_eigenvalues": fit["noise_covariance_eigenvalues"].tolist(),
        "initial_state_covariance_eigenvalues": fit["initial_covariance_eigenvalues"].tolist(),
        "spectral_radius_A": float(np.max(np.abs(np.linalg.eigvals(state_space.a)))),
        "A_eigenvalues": [
            {"real": float(pole.real), "imag": float(pole.imag)}
            for pole in poles
        ],
        "target_labels_previously_seen": True,
        "target_calibration_budget_transitions": 0,
    }


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    u_est, y_est, u_val, y_val, data_receipt = load_data()
    split = int(np.floor(SPLIT_FRACTION * (len(y_est) - 1)))
    selection, _ = select_block_rows(u_est, y_est)
    selected_block_rows = int(selection.iloc[0]["num_block_rows"])
    final_fit = fit_python_n4sid(u_est, y_est, selected_block_rows)
    final_fit["num_block_rows"] = selected_block_rows

    block_rows: list[dict] = []
    for horizon in HORIZONS:
        block_rows.extend(sparse_rollout(
            u_val,
            y_val,
            final_fit,
            horizon,
            score_start=0,
            score_stop=len(y_val),
            fully_observed_prefix_stop=0,
            evaluation="public_test_development",
        ))
    blocks = pd.DataFrame(block_rows)
    metrics = summarize_with_source_range(blocks, y_est, float(np.std(y_est)))

    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_dir / OUTPUT_FILES[0], index=False, float_format="%.12g")
    model_record = serialize_fit(final_fit, selected_block_rows)
    (output_dir / OUTPUT_FILES[1]).write_text(json.dumps(model_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    blocks.to_csv(output_dir / OUTPUT_FILES[2], index=False, float_format="%.12g")
    metrics.to_csv(output_dir / OUTPUT_FILES[3], index=False, float_format="%.12g")

    metadata = {
        "status": "COMPLETED_PUBLIC_TEST_DEVELOPMENT_BASELINE",
        "target_labels_previously_seen": True,
        "independent_or_blind_confirmation": False,
        "physical_target_records": 1,
        "matlab_author_n4sid_exact_reproduction": False,
        "upstream_commit": UPSTREAM_COMMIT,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "verification_script_sha256": sha256_file(ROOT / "scripts/verify_cascaded_tanks_python_n4sid_sparse_benchmark_v1.py"),
        "data_receipt": data_receipt,
        "source_selection": {
            "split_transition_index": split,
            "candidate_num_block_rows": list(BLOCK_ROWS),
            "fixed_state_order": STATE_ORDER,
            "selected_num_block_rows": selected_block_rows,
            "horizons_transitions": list(HORIZONS),
            "selection_rule": "minimum source-validation macro NRMSE over h=1,3,5; ties choose smaller num_block_rows",
        },
        "target_protocol": {
            "observation_budget_transitions": 0,
            "horizons_and_sparse_output_intervals_transitions": list(HORIZONS),
            "sampling_interval_seconds": DT_SECONDS,
            "known_recorded_input_path_within_each_block": True,
        },
        "upstream_source_files": {
            "kalman.py": sha256_file(UPSTREAM_DIR / "src/nfoursid/kalman.py"),
            "nfoursid.py": sha256_file(UPSTREAM_DIR / "src/nfoursid/nfoursid.py"),
            "state_space.py": sha256_file(UPSTREAM_DIR / "src/nfoursid/state_space.py"),
            "utils.py": sha256_file(UPSTREAM_DIR / "src/nfoursid/utils.py"),
            "license": sha256_file(UPSTREAM_DIR / "LICENSE"),
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
    (output_dir / "python_n4sid_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": metadata["status"],
        "selected_num_block_rows": selected_block_rows,
        "source_validation_macro_nrmse": float(selection.iloc[0]["validation_macro_nrmse"]),
        "metrics": metrics.to_dict(orient="records"),
        "output_dir": str(output_dir),
    }, indent=2, ensure_ascii=False))
    return metadata


if __name__ == "__main__":
    run()
