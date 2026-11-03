#!/usr/bin/env python3
"""Verify and replay the regularized ARX(2,2) Cascaded Tanks ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascaded_tanks_arx2_regularized_ablation_v1 import (
    HORIZONS,
    OUTPUT_DIR,
    OUTPUT_FILES,
    PROTOCOL_PATH,
    block_forecast,
    load_data,
    sha256_file,
    summarize_blocks,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VERIFICATION_FAILED: {message}")


def main() -> None:
    metadata_path = OUTPUT_DIR / "arx2_ridge_metadata.json"
    require(metadata_path.is_file(), "missing run metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata.get("status") == "COMPLETED_PUBLIC_TEST_DEVELOPMENT_ABLATION", "unexpected run status")
    require(metadata.get("target_labels_previously_seen") is True, "label history not recorded")
    require(metadata.get("independent_or_blind_confirmation") is False, "test was mislabeled independent/blind")
    require(metadata.get("physical_test_records") == 1, "physical test-record count changed")
    require(metadata.get("protocol_sha256") == sha256_file(PROTOCOL_PATH), "protocol hash mismatch")
    for name, receipt in metadata["output_files"].items():
        path = OUTPUT_DIR / name
        require(path.is_file(), f"missing output file: {name}")
        require(path.stat().st_size == receipt["bytes"], f"output size mismatch: {name}")
        require(sha256_file(path) == receipt["sha256"], f"output hash mismatch: {name}")
    require(tuple(metadata["output_files"].keys()) == OUTPUT_FILES, "output file manifest changed")

    u_est, y_est, u_val, y_val, data_receipt = load_data()
    require(metadata.get("data_receipt") == data_receipt, "source archive/member receipt mismatch")
    selection = pd.read_csv(OUTPUT_DIR / OUTPUT_FILES[0])
    coefficients = json.loads((OUTPUT_DIR / OUTPUT_FILES[1]).read_text(encoding="utf-8"))
    blocks = pd.read_csv(OUTPUT_DIR / OUTPUT_FILES[2])
    metrics = pd.read_csv(OUTPUT_DIR / OUTPUT_FILES[3])
    require(len(selection) == 8, "ridge alpha candidate set incomplete")
    require(set(selection.alpha.astype(float)) == {0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0}, "ridge alpha candidates changed")
    expected_alpha = float(selection.sort_values(["validation_macro_nrmse", "alpha"]).iloc[0].alpha)
    require(expected_alpha == float(coefficients["selected_ridge_alpha"]), "selected alpha differs from source-only validation")
    require(expected_alpha == float(metadata["source_validation"]["selected_alpha"]), "metadata selected alpha mismatch")
    require(coefficients["target_test_labels_previously_seen"] is True, "coefficient receipt omits label history")
    require(coefficients["feature_order"] == ["y[k]", "y[k-1]", "u[k]", "u[k-1]"], "feature order changed")

    coefficient_record = {
        "alpha": float(coefficients["selected_ridge_alpha"]),
        "intercept": float(coefficients["intercept"]),
        "coefficients": np.asarray(coefficients["coefficients"], dtype=float),
    }
    replay_rows = []
    for horizon in HORIZONS:
        replay_rows.extend(block_forecast(u_val, y_val, coefficient_record, horizon))
    replay = pd.DataFrame(replay_rows)
    key = ["horizon_transitions", "block_index", "step_in_block"]
    replay = replay.sort_values(key).reset_index(drop=True)
    actual = blocks.sort_values(key).reset_index(drop=True)
    require(len(replay) == len(actual) == int(metrics.n_scored_steps.sum()), "block coverage mismatch")
    for column in ("prediction", "truth", "input_current", "input_previous", "output_initial_current", "output_initial_previous"):
        require(np.allclose(replay[column].to_numpy(float), actual[column].to_numpy(float), rtol=1e-10, atol=1e-11), f"replayed block mismatch: {column}")
    expected_metrics = summarize_blocks(blocks, float(np.std(y_est)), "public_test_development")
    expected_metrics = expected_metrics.sort_values("horizon_transitions").reset_index(drop=True)
    actual_metrics = metrics.sort_values("horizon_transitions").reset_index(drop=True)
    require(len(expected_metrics) == len(actual_metrics) == len(HORIZONS), "metric horizon coverage mismatch")
    for column in ("rmse_sensor_units", "mae_sensor_units", "nrmse_over_full_estimation_output_std", "finite_prediction_fraction"):
        require(np.allclose(expected_metrics[column], actual_metrics[column], rtol=1e-10, atol=1e-11), f"metric replay mismatch: {column}")

    print(json.dumps({
        "status": "VERIFIED_CASCADED_TANKS_REGULARIZED_ARX2_ABLATION",
        "selected_alpha": expected_alpha,
        "source_validation_candidates": len(selection),
        "horizons": list(HORIZONS),
        "prediction_rows_replayed": len(replay),
        "metric_rows_recomputed": len(expected_metrics),
        "test_labels_previously_seen": True,
        "physical_test_records": 1,
    }, indent=2))


if __name__ == "__main__":
    main()
