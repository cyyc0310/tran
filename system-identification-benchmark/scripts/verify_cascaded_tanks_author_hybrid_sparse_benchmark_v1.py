#!/usr/bin/env python3
"""Verify author-hybrid Cascaded Tanks receipts and replay stored predictions without training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cascaded_tanks_author_hybrid_sparse_benchmark_v1 import (
    DEFAULT_OUTPUT_DIR,
    HORIZONS,
    SOURCE_TRAINING,
    TARGET_BUDGETS,
    UPSTREAM_COMMIT,
    UPSTREAM_DIR,
    load_data,
    make_predictions_for_seed,
    sha256_file,
)
from ssmodels_ct import CascadedTanksOverflowNeuralStateSpaceModel


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VERIFICATION_FAILED: {message}")


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    metadata_path = output_dir / "author_hybrid_metadata.json"
    require(metadata_path.is_file(), f"missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata.get("status") == "COMPLETED", "metadata status is not COMPLETED")
    require(metadata.get("test_labels_previously_seen") is True, "test-label history is missing/false")
    require(metadata.get("independent_test_or_blind_confirmation") is False, "test was mislabeled independent/blind")
    require(metadata.get("physical_record_repeats") == 1, "physical repeat count changed")
    require(metadata.get("upstream_commit") == UPSTREAM_COMMIT, "upstream source commit changed")
    require(metadata.get("training_config", {}).get("n_feat") == SOURCE_TRAINING["n_feat"], "model width mismatch")
    require(metadata.get("training_config", {}).get("sequence_length") == SOURCE_TRAINING["sequence_length"], "training sequence length mismatch")
    require(metadata.get("training_config", {}).get("batch_size") == SOURCE_TRAINING["batch_size"], "training batch size mismatch")
    require(metadata.get("training_config", {}).get("iterations") == 10000, "author training iteration count mismatch")

    u_est, y_est, u_val, y_val, data_receipt = load_data()
    require(metadata.get("data_receipt") == data_receipt, "source archive/member receipt mismatch")
    require(metadata.get("protocol_sha256") == sha256_file(Path(__file__).resolve().parents[1] / "docs/algorithmic-control-v1/CASCADED_TANKS_AUTHOR_HYBRID_SPARSE_PROTOCOL_V1_20261103.md"), "protocol hash mismatch")
    require(metadata.get("upstream_model_source_sha256") == sha256_file(UPSTREAM_DIR / "torchid/ssmodels_ct.py"), "upstream model source hash mismatch")
    require(metadata.get("upstream_training_script_sha256") == sha256_file(UPSTREAM_DIR / "examples/CTS/CTS_fit_truncated.py"), "upstream training script hash mismatch")
    require(metadata.get("upstream_license_sha256") == sha256_file(UPSTREAM_DIR / "LICENSE.md"), "upstream license hash mismatch")

    for name, receipt in metadata["output_files"].items():
        path = output_dir / name
        require(path.is_file(), f"missing output file: {name}")
        require(path.stat().st_size == receipt["bytes"], f"output size mismatch: {name}")
        require(sha256_file(path) == receipt["sha256"], f"output hash mismatch: {name}")

    training = pd.read_csv(output_dir / "author_hybrid_training_receipts.csv")
    blocks = pd.read_csv(output_dir / "author_hybrid_blocks.csv")
    calibrations = pd.read_csv(output_dir / "author_hybrid_calibration_receipts.csv")
    metrics = pd.read_csv(output_dir / "author_hybrid_metrics.csv")
    summary = pd.read_csv(output_dir / "author_hybrid_summary.csv")
    seeds = sorted(int(x) for x in metadata["training_config"]["seeds"])
    methods = ("author_hybrid_frozen", "author_hybrid_output_path_gamma")
    require(training["seed"].astype(int).tolist() == seeds, "training seed coverage mismatch")
    require(calibrations["seed"].nunique() == len(seeds), "calibration seed coverage mismatch")
    require(len(calibrations) == len(seeds) * len(TARGET_BUDGETS), "calibration budget coverage mismatch")
    require(len(metrics) == len(seeds) * len(TARGET_BUDGETS) * len(HORIZONS) * len(methods), "per-seed metric grid incomplete")
    require(len(summary) == len(TARGET_BUDGETS) * len(HORIZONS) * len(methods), "summary grid incomplete")

    for seed in seeds:
        receipt = training.loc[training.seed.eq(seed)].iloc[0]
        checkpoint = output_dir / "models" / f"author_cts_hybrid_seed_{seed}.pt"
        require(checkpoint.is_file(), f"missing seed checkpoint: {seed}")
        require(sha256_file(checkpoint) == receipt["checkpoint_sha256"], f"checkpoint hash mismatch: seed {seed}")
        require(receipt["source_archive_sha256"] == data_receipt["archive_sha256"], f"training archive mismatch: seed {seed}")
        require(receipt["source_data_csv_sha256"] == data_receipt["data_csv_sha256"], f"training CSV mismatch: seed {seed}")
        require(np.isfinite([receipt["final_total_loss"], receipt["final_fit_loss"], receipt["final_consistency_loss"]]).all(), f"non-finite training loss: seed {seed}")

        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(checkpoint_data["upstream_commit"] == UPSTREAM_COMMIT, f"checkpoint source commit mismatch: seed {seed}")
        model = CascadedTanksOverflowNeuralStateSpaceModel(n_feat=SOURCE_TRAINING["n_feat"])
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.eval()
        hidden = checkpoint_data["source_hidden_states"].detach().cpu().numpy()
        replay_blocks, replay_calibrations = make_predictions_for_seed(model, hidden, u_val, y_val, seed)

        expected_blocks = pd.DataFrame(replay_blocks)
        actual_blocks = blocks.loc[blocks.seed.eq(seed)].copy()
        block_key = ["seed", "method", "calibration_budget_transitions", "horizon_transitions", "block_index", "step_in_block"]
        expected_blocks = expected_blocks.sort_values(block_key).reset_index(drop=True)
        actual_blocks = actual_blocks.sort_values(block_key).reset_index(drop=True)
        require(len(expected_blocks) == len(actual_blocks), f"replayed block count mismatch: seed {seed}")
        for column in ("prediction", "truth", "gamma", "latent_upper_state", "base_lower_state", "control_min_in_block", "control_max_in_block"):
            require(np.allclose(expected_blocks[column].to_numpy(float), actual_blocks[column].to_numpy(float), rtol=1e-7, atol=1e-8, equal_nan=True), f"prediction replay mismatch: seed={seed}, column={column}")

        expected_cal = pd.DataFrame(replay_calibrations).sort_values(["seed", "calibration_budget_transitions"]).reset_index(drop=True)
        actual_cal = calibrations.loc[calibrations.seed.eq(seed)].sort_values(["seed", "calibration_budget_transitions"]).reset_index(drop=True)
        require(len(expected_cal) == len(actual_cal), f"calibration receipt count mismatch: seed {seed}")
        for column in ("gamma", "prefix_base_rmse", "prefix_calibrated_rmse"):
            require(np.allclose(expected_cal[column].to_numpy(float), actual_cal[column].to_numpy(float), rtol=1e-7, atol=1e-8, equal_nan=True), f"calibration replay mismatch: seed={seed}, column={column}")

    require(blocks["seed"].nunique() == len(seeds), "block seed coverage mismatch")
    require(blocks["method"].nunique() == len(methods), "method coverage mismatch")
    require(set(blocks["method"].unique()) == set(methods), "unexpected author-hybrid method labels")
    require(set(blocks["calibration_budget_transitions"].unique()) == set(TARGET_BUDGETS), "target budget coverage mismatch")
    require(set(blocks["horizon_transitions"].unique()) == set(HORIZONS), "horizon coverage mismatch")
    require(np.isfinite(blocks[["prediction", "truth"]].to_numpy(dtype=float)).all(), "non-finite prediction or truth in archived blocks")

    source_std = max(float(np.std(y_est[:, 0])), 1e-12)
    for row in metrics.itertuples(index=False):
        group = blocks.loc[
            blocks.seed.eq(row.seed)
            & blocks.method.eq(row.method)
            & blocks.calibration_budget_transitions.eq(row.calibration_budget_transitions)
            & blocks.horizon_transitions.eq(row.horizon_transitions)
        ]
        diff = group["prediction"].to_numpy(float) - group["truth"].to_numpy(float)
        expected_rmse = float(np.sqrt(np.mean(diff**2)))
        expected_mae = float(np.mean(np.abs(diff)))
        require(np.isclose(row.rmse, expected_rmse, rtol=1e-8, atol=1e-10), f"RMSE mismatch: seed={row.seed}, method={row.method}")
        require(np.isclose(row.mae, expected_mae, rtol=1e-8, atol=1e-10), f"MAE mismatch: seed={row.seed}, method={row.method}")
        require(np.isclose(row.normalized_rmse, expected_rmse / source_std, rtol=1e-8, atol=1e-10), f"NRMSE mismatch: seed={row.seed}, method={row.method}")

    old_metrics_path = Path(__file__).resolve().parents[1] / "docs/algorithmic-control-v1/results/cascaded_tanks_sparse_input_v1/cascaded_tanks_metrics.csv"
    old_metrics = pd.read_csv(old_metrics_path)
    frozen_old = old_metrics.loc[
        old_metrics.calibration_budget_transitions.eq(0)
        & old_metrics.method.isin(["source_linear_arx_1_0_frozen", "source_sindy_c_style_poly2_stlsq_frozen"])
    ]
    require(len(frozen_old) == len(HORIZONS) * 2, "missing frozen ARX/SINDYc comparison rows")
    require(old_metrics_path.is_file(), "existing ARX/SINDYc baseline metrics missing")

    print(json.dumps({
        "status": "VERIFIED_CASCADED_TANKS_AUTHOR_HYBRID_SPARSE_RESULTS",
        "seeds": seeds,
        "calibration_receipts": len(calibrations),
        "metric_rows": len(metrics),
        "summary_rows": len(summary),
        "prediction_rows": len(blocks),
        "model_predictions_replayed_without_training": True,
        "existing_frozen_arx_sindyc_rows": len(frozen_old),
        "notes": "Repeated blocks and optimizer seeds are not independent physical test records.",
    }, indent=2))


if __name__ == "__main__":
    main()
