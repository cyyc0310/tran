#!/usr/bin/env python3
"""Independently refit source N4SID candidates and replay saved sparse predictions."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs/algorithmic-control-v1/external-source/cascaded_tanks"
UPSTREAM_DIR = ROOT / "docs/algorithmic-control-v1/external-source/nfoursid"
OUTPUT_DIR = ROOT / "docs/algorithmic-control-v1/results/cascaded_tanks_python_n4sid_sparse_v1"
ARCHIVE_PATH = SOURCE_DIR / "CascadedTanksFiles.zip"
PROTOCOL_PATH = ROOT / "docs/algorithmic-control-v1/CASCADED_TANKS_PYTHON_N4SID_SPARSE_PROTOCOL_V1_20261103.md"
UPSTREAM_COMMIT = "1e306879f9e697cd275f17d01dc96182b74ac97d"
EXPECTED_ARCHIVE_SHA256 = "eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51"
EXPECTED_MAT_SHA256 = "cb2f88d4388be4d3f2a24c6402fba804976aac5f2e1f26cda59ea0a38d016eab"
BLOCK_ROWS = (2, 3, 4, 6, 8, 10, 12, 16)
HORIZONS = (1, 3, 5)
STATE_ORDER = 2
DT_SECONDS = 4.0
OUTPUT_FILES = (
    "python_n4sid_source_validation.csv",
    "python_n4sid_model.json",
    "python_n4sid_sparse_blocks.csv",
    "python_n4sid_sparse_metrics.csv",
)

sys.path.insert(0, str(UPSTREAM_DIR / "src"))
from nfoursid.nfoursid import NFourSID  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VERIFICATION_FAILED: {message}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_source_and_target() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    archive_bytes = ARCHIVE_PATH.read_bytes()
    archive_hash = sha256_bytes(archive_bytes)
    require(archive_hash == EXPECTED_ARCHIVE_SHA256, f"unexpected source archive SHA-256: {archive_hash}")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        mat_member = "CascadedTanksFiles/dataBenchmark.mat"
        mat_bytes = archive.read(mat_member)
    mat_hash = sha256_bytes(mat_bytes)
    require(mat_hash == EXPECTED_MAT_SHA256, f"unexpected MAT member SHA-256: {mat_hash}")
    source = loadmat(io.BytesIO(mat_bytes))
    arrays = [np.asarray(source[key], dtype=float).reshape(-1) for key in ("uEst", "yEst", "uVal", "yVal")]
    require(all(array.size == 1024 and np.isfinite(array).all() for array in arrays), "source record shape/finite check failed")
    require(float(np.asarray(source["Ts"]).reshape(-1)[0]) == DT_SECONDS, "sample interval changed")
    return (*arrays, {
        "archive_sha256": archive_hash,
        "mat_member": mat_member,
        "mat_member_sha256": mat_hash,
        "sampling_interval_seconds": DT_SECONDS,
        "source_samples": int(arrays[1].size),
        "target_samples": int(arrays[3].size),
    })


def estimate_initial_state(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
                           u: np.ndarray, y: np.ndarray, count: int = 40) -> tuple[np.ndarray, float]:
    forced = np.zeros((STATE_ORDER, 1), dtype=float)
    power = np.eye(STATE_ORDER, dtype=float)
    design_rows, target_rows = [], []
    for index in range(min(count, y.size)):
        design_rows.append(c @ power)
        target_rows.append(np.asarray([[y[index]]]) - d * float(u[index]) - c @ forced)
        forced = a @ forced + b * float(u[index])
        power = power @ a
    design = np.concatenate(design_rows, axis=0)
    target = np.concatenate(target_rows, axis=0)
    state0 = np.linalg.lstsq(design, target, rcond=None)[0]
    rmse = float(np.sqrt(np.mean((design @ state0 - target) ** 2)))
    return state0, rmse


def fit_n4sid(u: np.ndarray, y: np.ndarray, block_rows: int) -> dict:
    scaler = {
        "u_mean": float(np.mean(u)),
        "u_scale": max(float(np.std(u)), 1e-12),
        "y_mean": float(np.mean(y)),
        "y_scale": max(float(np.std(y)), 1e-12),
    }
    un = (u - scaler["u_mean"]) / scaler["u_scale"]
    yn = (y - scaler["y_mean"]) / scaler["y_scale"]
    identifier = NFourSID(
        pd.DataFrame({"u": un, "y": yn}),
        output_columns=["y"],
        input_columns=["u"],
        num_block_rows=int(block_rows),
    )
    identifier.subspace_identification()
    state_space, covariance = identifier.system_identification(rank=STATE_ORDER)
    covariance = (covariance + covariance.T) / 2.0
    a, b, c, d = state_space.a, state_space.b, state_space.c, state_space.d
    x0, x0_rmse = estimate_initial_state(a, b, c, d, un, yn)

    decomposition = identifier._get_observability_matrix_decomposition()
    singular_values = np.maximum(np.diag(decomposition.eigenvalues), 0.0)
    state_sequence = np.sqrt(singular_values)[:, None] * decomposition.right_orthogonal
    covariance_count = min(100, state_sequence.shape[1])
    p0 = np.cov(state_sequence[:, :covariance_count], ddof=1)
    p0 = (p0 + p0.T) / 2.0
    floor = max(1e-9, abs(float(np.trace(p0))) * 1e-9)
    p0 += floor * np.eye(STATE_ORDER)
    eigenvalues = np.linalg.eigvalsh(p0)
    if float(np.min(eigenvalues)) <= 0:
        p0 += (abs(float(np.min(eigenvalues))) + floor) * np.eye(STATE_ORDER)
    return {
        "a": a, "b": b, "c": c, "d": d, "noise_covariance": covariance,
        "scaler": scaler, "x0": x0, "p0": p0, "x0_rmse": x0_rmse,
        "noise_eigenvalues": np.linalg.eigvalsh(covariance),
        "p0_eigenvalues": np.linalg.eigvalsh(p0),
        "num_block_rows": int(block_rows),
    }


def filter_step(fit: dict, x: np.ndarray, p: np.ndarray, u: float, y: float | None) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d = fit["a"], fit["b"], fit["c"], fit["d"]
    covariance = fit["noise_covariance"]
    r = covariance[:1, :1]
    s = covariance[1:, :1]
    q = covariance[1:, 1:]
    innovation_covariance = r + c @ p @ c.T
    gain = (s + a @ p @ c.T) @ np.linalg.pinv(innovation_covariance)
    x_next = a @ x + b * float(u)
    if y is not None:
        innovation = np.asarray([[float(y)]]) - d * float(u) - c @ x
        x_next = x_next + gain @ innovation
    p_next = a @ p @ a.T + q - gain @ (s + a @ p @ c.T).T
    return x_next, p_next


def forecast_blocks(u_original: np.ndarray, y_original: np.ndarray, fit: dict, horizon: int,
                    score_start: int, prefix_stop: int, evaluation: str) -> list[dict]:
    scale = fit["scaler"]
    u = (u_original - scale["u_mean"]) / scale["u_scale"]
    y = (y_original - scale["y_mean"]) / scale["y_scale"]
    starts = np.arange(score_start, len(y) - horizon, horizon, dtype=int)
    rows = []
    x, p = fit["x0"].copy(), fit["p0"].copy()
    for index in range(prefix_stop):
        x, p = filter_step(fit, x, p, float(u[index]), float(y[index]))
    for start in starts:
        for step in range(horizon):
            index = int(start + step)
            observed = step == 0
            x, p = filter_step(fit, x, p, float(u[index]), float(y[index]) if observed else None)
            next_index = index + 1
            prediction_n = (fit["c"] @ x + fit["d"] * float(u[next_index])).item()
            prediction = float(prediction_n * scale["y_scale"] + scale["y_mean"])
            rows.append({
                "evaluation": evaluation,
                "horizon_transitions": horizon,
                "block_index": int((start - score_start) // horizon),
                "step_in_block": step + 1,
                "start_index": int(start),
                "target_index": next_index,
                "sparse_output_observed_at_step": bool(observed),
                "prediction": prediction,
                "truth": float(y_original[next_index]),
                "input_transition": float(u_original[index]),
                "input_for_direct_output_term": float(u_original[next_index]),
            })
    return rows


def source_selection(u_est: np.ndarray, y_est: np.ndarray, split: int) -> pd.DataFrame:
    rows = []
    for block_rows in BLOCK_ROWS:
        fit = fit_n4sid(u_est[:split + 1], y_est[:split + 1], block_rows)
        values = []
        for horizon in HORIZONS:
            predictions = forecast_blocks(
                u_est, y_est, fit, horizon, score_start=split,
                prefix_stop=split, evaluation="source_chronological_validation",
            )
            group = [row for row in predictions if row["horizon_transitions"] == horizon]
            errors = np.asarray([row["prediction"] - row["truth"] for row in group])
            rmse = float(np.sqrt(np.mean(errors**2)))
            values.append(rmse / max(float(np.std(y_est[:split + 1])), 1e-12))
        rows.append({
            "num_block_rows": block_rows,
            "validation_nrmse_h1": values[0],
            "validation_nrmse_h3": values[1],
            "validation_nrmse_h5": values[2],
            "validation_macro_nrmse": float(np.mean(values)),
        })
    return pd.DataFrame(rows).sort_values(["validation_macro_nrmse", "num_block_rows"]).reset_index(drop=True)


def verify() -> dict:
    metadata_path = OUTPUT_DIR / "python_n4sid_metadata.json"
    require(metadata_path.is_file(), "missing benchmark metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata.get("status") == "COMPLETED_PUBLIC_TEST_DEVELOPMENT_BASELINE", "unexpected run status")
    require(metadata.get("target_labels_previously_seen") is True, "test-label history is absent")
    require(metadata.get("independent_or_blind_confirmation") is False, "test was mislabeled as independent/blind")
    require(metadata.get("physical_target_records") == 1, "physical record count changed")
    require(metadata.get("matlab_author_n4sid_exact_reproduction") is False, "Python result was mislabeled as MATLAB reproduction")
    require(metadata.get("upstream_commit") == UPSTREAM_COMMIT, "upstream source revision mismatch")
    require(metadata.get("protocol_sha256") == sha256_file(PROTOCOL_PATH), "protocol hash mismatch")
    require(metadata.get("script_sha256") == sha256_file(ROOT / "scripts/cascaded_tanks_python_n4sid_sparse_benchmark_v1.py"), "benchmark script hash mismatch")
    require(metadata.get("verification_script_sha256") == sha256_file(Path(__file__).resolve()), "verification script hash mismatch")

    upstream_manifest = metadata.get("upstream_source_files", {})
    for key, relative_path in {
        "kalman.py": "src/nfoursid/kalman.py",
        "nfoursid.py": "src/nfoursid/nfoursid.py",
        "state_space.py": "src/nfoursid/state_space.py",
        "utils.py": "src/nfoursid/utils.py",
        "license": "LICENSE",
    }.items():
        require(upstream_manifest.get(key) == sha256_file(UPSTREAM_DIR / relative_path), f"upstream source hash mismatch: {key}")

    for name, receipt in metadata["output_files"].items():
        path = OUTPUT_DIR / name
        require(path.is_file(), f"missing output: {name}")
        require(path.stat().st_size == receipt["bytes"], f"output length mismatch: {name}")
        require(sha256_file(path) == receipt["sha256"], f"output hash mismatch: {name}")
    require(tuple(metadata["output_files"].keys()) == OUTPUT_FILES, "output manifest ordering/content changed")

    u_est, y_est, u_val, y_val, data_receipt = load_source_and_target()
    require(metadata.get("data_receipt") == data_receipt, "data receipt mismatch")
    selection = pd.read_csv(OUTPUT_DIR / OUTPUT_FILES[0])
    model = json.loads((OUTPUT_DIR / OUTPUT_FILES[1]).read_text(encoding="utf-8"))
    blocks = pd.read_csv(OUTPUT_DIR / OUTPUT_FILES[2])
    metrics = pd.read_csv(OUTPUT_DIR / OUTPUT_FILES[3])

    split = int(np.floor(0.8 * (len(y_est) - 1)))
    replayed_selection = source_selection(u_est, y_est, split)
    expected_selection = selection.sort_values(["validation_macro_nrmse", "num_block_rows"]).reset_index(drop=True)
    require(len(selection) == len(BLOCK_ROWS), "source candidate table incomplete")
    for column in ("validation_nrmse_h1", "validation_nrmse_h3", "validation_nrmse_h5", "validation_macro_nrmse"):
        require(np.allclose(replayed_selection[column], expected_selection[column], rtol=2e-8, atol=2e-10), f"source validation replay mismatch: {column}")
    selected = int(replayed_selection.iloc[0]["num_block_rows"])
    require(selected == int(model["num_block_rows"]), "saved model did not use source-selected block rows")
    require(selected == int(metadata["source_selection"]["selected_num_block_rows"]), "source selection metadata mismatch")

    fit = fit_n4sid(u_est, y_est, selected)
    for key, matrix in (("A", fit["a"]), ("B", fit["b"]), ("C", fit["c"]), ("D", fit["d"]),
                        ("noise_covariance", fit["noise_covariance"]),
                        ("initial_state_source_estimate", fit["x0"]),
                        ("initial_state_covariance_source_estimate", fit["p0"])):
        require(np.allclose(np.asarray(model[key], dtype=float), matrix, rtol=2e-8, atol=2e-10), f"full-source refit mismatch: {key}")
    for key in ("u_mean", "u_scale", "y_mean", "y_scale"):
        require(np.isclose(float(model["source_scaler"][key]), fit["scaler"][key], rtol=1e-12, atol=1e-12), f"source scaler mismatch: {key}")
    expected_poles = np.linalg.eigvals(fit["a"])
    saved_poles = sorted(
        ((float(item["real"]), float(item["imag"])) for item in model["A_eigenvalues"]),
        key=lambda pair: (pair[0], pair[1]),
    )
    refit_poles = sorted(((float(item.real), float(item.imag)) for item in expected_poles), key=lambda pair: (pair[0], pair[1]))
    require(np.allclose(np.asarray(saved_poles), np.asarray(refit_poles), rtol=2e-8, atol=2e-10), "saved model poles do not match refit")

    expected_rows = []
    for horizon in HORIZONS:
        expected_rows.extend(forecast_blocks(
            u_val, y_val, fit, horizon, score_start=0, prefix_stop=0,
            evaluation="public_test_development",
        ))
    expected = pd.DataFrame(expected_rows).sort_values(["horizon_transitions", "block_index", "step_in_block"]).reset_index(drop=True)
    actual = blocks.sort_values(["horizon_transitions", "block_index", "step_in_block"]).reset_index(drop=True)
    require(len(expected) == len(actual), "target block coverage mismatch")
    for column in ("prediction", "truth", "input_transition", "input_for_direct_output_term"):
        require(np.allclose(expected[column].to_numpy(float), actual[column].to_numpy(float), rtol=2e-8, atol=2e-9), f"target prediction replay mismatch: {column}")
    for column in ("evaluation", "horizon_transitions", "block_index", "step_in_block", "start_index", "target_index", "sparse_output_observed_at_step"):
        require(expected[column].astype(str).tolist() == actual[column].astype(str).tolist(), f"target row contract mismatch: {column}")

    for row in metrics.itertuples(index=False):
        group = actual.loc[actual.horizon_transitions.eq(row.horizon_transitions)]
        error = group.prediction.to_numpy(float) - group.truth.to_numpy(float)
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        require(np.isclose(float(row.rmse_sensor_units), rmse, rtol=2e-8, atol=2e-9), f"RMSE mismatch: h={row.horizon_transitions}")
        require(np.isclose(float(row.mae_sensor_units), mae, rtol=2e-8, atol=2e-9), f"MAE mismatch: h={row.horizon_transitions}")
        require(np.isclose(float(row.nrmse_over_source_output_std), rmse / np.std(y_est), rtol=2e-8, atol=2e-9), f"NRMSE mismatch: h={row.horizon_transitions}")
        require(np.isclose(float(row.finite_prediction_fraction), np.isfinite(group.prediction.to_numpy(float)).mean(), rtol=1e-12, atol=1e-12), f"finite fraction mismatch: h={row.horizon_transitions}")
        require(np.isclose(float(row.negative_prediction_fraction), (group.prediction.to_numpy(float) < 0.0).mean(), rtol=1e-12, atol=1e-12), f"negative prediction fraction mismatch: h={row.horizon_transitions}")
        require(np.isclose(float(row.prediction_outside_source_range_fraction), ((group.prediction.to_numpy(float) < np.min(y_est)) | (group.prediction.to_numpy(float) > np.max(y_est))).mean(), rtol=1e-12, atol=1e-12), f"source range fraction mismatch: h={row.horizon_transitions}")

    return {
        "status": "VERIFIED_CASCADED_TANKS_PYTHON_N4SID_SPARSE_RESULTS",
        "source_candidates_refit": len(replayed_selection),
        "selected_block_rows": selected,
        "target_prediction_rows_replayed_from_saved_matrices": len(actual),
        "horizons": list(HORIZONS),
        "test_labels_previously_seen": True,
        "independent_physical_records": 1,
        "target_replay_used_manual_kalman_equations": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
