#!/usr/bin/env python3
"""Read-only verification of Cascaded Tanks experiment outputs and receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_cascaded_tanks_source_v1 import verify as verify_source_archive


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs/algorithmic-control-v1/results/cascaded_tanks_sparse_input_v1"
PROTOCOL_PATH = ROOT / "docs/algorithmic-control-v1/CASCADED_TANKS_SPARSE_INPUT_PROTOCOL_V1_20261103.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(output_dir: Path = OUTPUT_DIR) -> dict:
    source_receipt = verify_source_archive()
    if source_receipt["status"] != "VERIFIED_CASCADED_TANKS_SOURCE_ARCHIVE":
        raise ValueError("source archive verification failed")

    metadata_path = output_dir / "cascaded_tanks_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("experiment") != "Cascaded Tanks sparse-output known-pump-input transfer extension":
        raise ValueError("unexpected experiment metadata")
    if metadata.get("status") != "development_evidence_public_target_outputs_were_inspected_during_source_admission":
        raise ValueError("experiment leakage/confirmation status is not explicit")
    if metadata.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise ValueError("protocol hash differs from the frozen file")

    for receipt in metadata["source_files"]:
        path = ROOT / receipt["path"]
        if not path.is_file() or sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"source file hash mismatch: {receipt['path']}")

    for name, receipt in metadata["outputs"].items():
        path = output_dir / name
        if not path.is_file() or path.stat().st_size != receipt["bytes"] or sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"output hash mismatch: {name}")

    metrics = pd.read_csv(output_dir / "cascaded_tanks_metrics.csv")
    blocks = pd.read_csv(output_dir / "cascaded_tanks_blocks.csv")
    gamma = pd.read_csv(output_dir / "cascaded_tanks_gamma_receipts.csv")
    diagnostics = json.loads((output_dir / "cascaded_tanks_excitation_diagnostics.json").read_text())
    required_metric_columns = {
        "task", "method", "calibration_budget_transitions", "calibration_budget_seconds",
        "horizon_transitions", "horizon_seconds", "n_scored_values", "n_finite_values",
        "finite_prediction_fraction", "n_blocks", "diverged_blocks", "rmse_sensor_units",
        "mae_sensor_units", "nrmse_over_estimation_output_std",
    }
    if not required_metric_columns.issubset(metrics.columns):
        raise ValueError("metrics file is missing required columns")
    if set(metrics["calibration_budget_transitions"].unique()) != {0, 1, 3, 5}:
        raise ValueError("unexpected calibration budgets")
    if set(metrics["horizon_transitions"].unique()) != {1, 3, 5}:
        raise ValueError("unexpected prediction horizons")
    if set(metrics["task"].unique()) != {
        "causal_block_rollout_given_recorded_pump_path",
        "offline_endpoint_interpolation_oracle",
    }:
        raise ValueError("unexpected task coverage")
    if len(gamma) != 6 or set(gamma["calibration_budget_transitions"].unique()) != {1, 3, 5}:
        raise ValueError("target scalar calibration receipts are incomplete")
    for row in gamma.itertuples(index=False):
        if row.target_state_indices_used != str(list(range(int(row.calibration_budget_transitions) + 1))):
            raise ValueError("target calibration state indices exceed the stated prefix budget")
        if row.target_input_indices_used != str(list(range(int(row.calibration_budget_transitions)))):
            raise ValueError("target calibration input indices exceed the stated prefix budget")
        if not np.isfinite(float(row.gamma)):
            raise ValueError("non-finite target calibration parameter")
    if not np.isfinite(blocks["truth"].to_numpy(dtype=float)).all():
        raise ValueError("block scoring labels include non-finite values")
    if len(blocks) == 0 or len(metrics) != 66:
        raise ValueError("unexpected result size")
    if diagnostics.get("source_input_and_regressor_diagnostics", {}).get("transitions") != 1023:
        raise ValueError("source excitation diagnostics do not cover the estimation record")

    return {
        "status": "VERIFIED_CASCADED_TANKS_SPARSE_INPUT_RESULTS",
        "source_archive_sha256": source_receipt["archive_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "source_files_hashed": len(metadata["source_files"]),
        "output_files_hashed": len(metadata["outputs"]),
        "metrics_rows": len(metrics),
        "block_rows": len(blocks),
        "gamma_receipts": len(gamma),
        "budgets_transitions": sorted(metrics["calibration_budget_transitions"].unique().astype(int).tolist()),
        "horizons_transitions": sorted(metrics["horizon_transitions"].unique().astype(int).tolist()),
        "finite_predictions": bool(np.isfinite(blocks["prediction"].to_numpy(dtype=float)).all()),
        "note": "hash/schema/prefix-budget verifier; does not retrain or select models",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="run directory to verify (defaults to the archived result directory)",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), indent=2, sort_keys=True))
