#!/usr/bin/env python3
"""Run InTaSet under the horizons and sparse calibration budgets used on motors.

This wrapper leaves the existing historical runners unchanged. It runs their
ARX/state-space, discrete SINDYc-style, controlled Neural ODE and grey-box
hybrid implementations on 0.5/1 s state-observation intervals and 0/1/3/5 s
target calibration budgets, then writes a direction-preserving harmonized
summary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HORIZON_STRIDES = (5, 10)  # native 0.1 s sampling -> 0.5 s and 1 s
CALIBRATION_BUDGETS = (0, 1, 3, 5)
DEFAULT_SEEDS = (20261102, 20261103, 20261104)
DEFAULT_OUTPUT_DIR = ROOT / "docs/algorithmic-control-v1/results/aligned_control_budget_v1/intaset"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_metadata_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            replace_strings(key, replacements): replace_strings(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def rewrite_runner_metadata(path: Path, wrapper_hash: str, runner_name: str) -> dict:
    metadata = json.loads(path.read_text())
    metadata = replace_strings(metadata, {
        "1/5/10-second": "0.5/1-second",
        "1/5/10 second": "0.5/1 second",
        "1, 5, and 10 seconds": "0.5 and 1 second",
        "1/5/10-second horizons": "0.5/1-second horizons",
        "first 60 or 300 seconds": "first 1, 3, or 5 seconds",
        "60 or 300 seconds": "1, 3, or 5 seconds",
        "60/300-second": "1/3/5-second",
        "60/300 seconds": "1/3/5 seconds",
        "mean_normalized_rollout_rmse_1_5_10s": "mean_normalized_rollout_rmse_0.5_1s",
        "mean_validation_normalized_rollout_rmse_1_5_10s": "mean_validation_normalized_rollout_rmse_0.5_1s",
    })
    task_contract = metadata.get("task_contract")
    if isinstance(task_contract, dict):
        task_contract["target_state_observation_intervals_seconds"] = [stride / 10 for stride in HORIZON_STRIDES]
    metadata["aligned_budget_wrapper"] = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": wrapper_hash,
        "runner": runner_name,
        "horizon_strides_native_samples": list(HORIZON_STRIDES),
        "target_state_observation_intervals_seconds": [stride / 10 for stride in HORIZON_STRIDES],
        "target_calibration_budgets_seconds": list(CALIBRATION_BUDGETS),
        "note": "The base runner source is unchanged; this wrapper sets the global experiment constants before invoking its run() function.",
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def summarize_in_dataset(
    *,
    causal_detail: pd.DataFrame,
    calibration_detail: pd.DataFrame,
    neural_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    causal = causal_detail[causal_detail["state"].eq("ALL")].copy()
    if len(causal):
        grouped = causal.groupby(["direction", "task", "horizon_seconds", "method"], as_index=False).agg(
            rmse_raw_cm=("rmse_cm", "mean"),
            mae_raw_cm=("mae_cm", "mean"),
            normalized_rmse=("rmse_over_source_train_std", "mean"),
            below_zero_fraction=("predicted_below_zero_fraction", "mean"),
        )
        for row in grouped.to_dict("records"):
            rows.append({
                "dataset": "InTaSet_4T",
                "condition": row["direction"],
                "task": row["task"],
                "observation_horizon_seconds": row["horizon_seconds"],
                "calibration_budget_seconds": 0,
                "method": row["method"],
                "rmse_raw": row["rmse_raw_cm"],
                "raw_unit": "cm",
                "normalized_rmse": row["normalized_rmse"],
                "mae_raw": row["mae_raw_cm"],
                "below_zero_fraction": row["below_zero_fraction"],
                "training_seed_sd": np.nan,
            })

    calibration = calibration_detail[calibration_detail["state"].eq("ALL")].copy()
    if len(calibration):
        grouped = calibration.groupby(
            ["direction", "task", "horizon_seconds", "calibration_budget_seconds", "method"],
            as_index=False,
        ).agg(
            rmse_raw_cm=("rmse_cm", "mean"),
            mae_raw_cm=("mae_cm", "mean"),
            normalized_rmse=("rmse_over_source_train_std", "mean"),
            below_zero_fraction=("predicted_below_zero_fraction", "mean"),
        )
        for row in grouped.to_dict("records"):
            rows.append({
                "dataset": "InTaSet_4T",
                "condition": row["direction"],
                "task": row["task"],
                "observation_horizon_seconds": row["horizon_seconds"],
                "calibration_budget_seconds": row["calibration_budget_seconds"],
                "method": row["method"],
                "rmse_raw": row["rmse_raw_cm"],
                "raw_unit": "cm",
                "normalized_rmse": row["normalized_rmse"],
                "mae_raw": row["mae_raw_cm"],
                "below_zero_fraction": row["below_zero_fraction"],
                "training_seed_sd": np.nan,
            })

    neural = neural_summary[neural_summary["state"].eq("ALL")].copy()
    for row in neural.to_dict("records"):
        task = row["task"]
        rows.append({
            "dataset": "InTaSet_4T",
            "condition": row["direction"],
            "task": task,
            "observation_horizon_seconds": row["horizon_seconds"],
            "calibration_budget_seconds": row["calibration_budget_seconds"],
            "method": row["method"],
            "rmse_raw": row["rmse_cm_mean"],
            "raw_unit": "cm",
            "normalized_rmse": row["normalized_rmse_mean"],
            "mae_raw": row["mae_cm_mean"],
            "below_zero_fraction": row["predicted_below_zero_fraction_mean"],
            "training_seed_sd": row["normalized_rmse_sd_between_seeds"],
        })
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["dataset", "condition", "task", "observation_horizon_seconds", "calibration_budget_seconds", "method"],
        na_position="first",
    ).reset_index(drop=True)


def run(output_dir: Path, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> tuple[pd.DataFrame, dict]:
    started = time.time()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_hash = sha256(Path(__file__))

    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    causal = importlib.import_module("intaset_causal_transfer_benchmark_v1")
    sparse = importlib.import_module("intaset_sparse_target_calibration_v1")
    neural = importlib.import_module("intaset_neural_ode_transfer_benchmark_v1")

    # Keep all three runners on the exact same state horizons. Their own raw
    # data and source/target splits remain unchanged.
    causal.STRIDES = HORIZON_STRIDES
    sparse.STRIDES = HORIZON_STRIDES
    sparse.CALIBRATION_BUDGETS_SECONDS = CALIBRATION_BUDGETS
    neural.STRIDES = HORIZON_STRIDES
    neural.CALIBRATION_BUDGETS_SECONDS = CALIBRATION_BUDGETS[1:]

    causal_dir = output_dir / "causal_baselines"
    sparse_dir = output_dir / "sparse_calibration"
    neural_dir = output_dir / "neural_models"
    causal_detail, _, _ = causal.run(causal_dir)
    calibration_detail, _, _ = sparse.run(sparse_dir)
    _, neural_summary, _ = neural.run(neural_dir, seeds=seeds)

    runner_metadata = {}
    for name, directory, metadata_name in (
        ("causal_baselines", causal_dir, "causal_transfer_metadata.json"),
        ("sparse_calibration", sparse_dir, "sparse_calibration_metadata.json"),
        ("neural_models", neural_dir, "intaset_neural_ode_metadata.json"),
    ):
        runner_metadata[name] = rewrite_runner_metadata(directory / metadata_name, script_hash, name)

    master = summarize_in_dataset(
        causal_detail=causal_detail,
        calibration_detail=calibration_detail,
        neural_summary=neural_summary,
    )
    master_path = output_dir / "intaset_aligned_master_summary.csv"
    master.to_csv(master_path, index=False, float_format="%.10g")
    run_metadata = {
        "experiment": "InTaSet aligned to external motor sparse-state horizons and calibration budgets",
        "task_contract": {
            "horizons_seconds": [stride / 10 for stride in HORIZON_STRIDES],
            "target_calibration_budgets_seconds": list(CALIBRATION_BUDGETS),
            "target_pump_input_interval_seconds": 0.1,
            "target_input_availability": "full recorded future pump path is conditionally supplied; this is not an online command-availability claim",
            "directional_split": "M1->M2 and M2->M1, with the target test trace held out from fitting and calibration",
            "baseline_and_offline_tasks": "endpoint interpolation remains offline; teacher-forced one-step metrics remain diagnostics; neither is mixed with causal sparse rollouts",
            "calibration": "source model weights frozen; linear and discrete sparse methods use per-state delta gains; neural/grey-box models use per-state vector-field rate gains",
        },
        "seeds": list(seeds),
        "wrapper_sha256": script_hash,
        "base_runners": {
            "causal": sha256(Path(causal.__file__)),
            "sparse_calibration": sha256(Path(sparse.__file__)),
            "neural": sha256(Path(neural.__file__)),
        },
        "runner_metadata": {
            name: format_metadata_path(directory / metadata_name)
            for name, directory, metadata_name in (
                ("causal_baselines", causal_dir, "causal_transfer_metadata.json"),
                ("sparse_calibration", sparse_dir, "sparse_calibration_metadata.json"),
                ("neural_models", neural_dir, "intaset_neural_ode_metadata.json"),
            )
        },
        "master_summary": format_metadata_path(master_path),
        "master_summary_sha256": sha256(master_path),
        "wall_seconds": time.time() - started,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "interpretation": "This run aligns time horizons and observed target-data budgets across datasets; it does not remove the recorded-future-input limitation or make water and motor raw errors directly comparable.",
    }
    metadata_path = output_dir / "aligned_budget_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n")
    return master, run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    master, metadata = run(args.output_dir, seeds=seeds)
    print(master.groupby(["condition", "task", "observation_horizon_seconds", "method"], dropna=False)["normalized_rmse"].mean().to_string())
    print(f"\nmaster rows={len(master)}; runtime={metadata['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
