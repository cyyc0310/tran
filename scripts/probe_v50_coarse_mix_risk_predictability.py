"""Source-held-out probe of legal monthly POS share as an hourly-risk predictor."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/disjoint-calibration-v50"
OUT = ROOT / "results/disjoint-calibration-v50-coarse-mix-probe"
sys.path.insert(0, str(ROOT))
from scripts import disjoint_calibration_v50 as helper  # noqa: E402

SOURCES, EXTERNAL = helper.LEGACY, helper.EXTERNAL


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def inverse(values: np.ndarray, level: float) -> float:
    ordered = np.sort(values.reshape(-1), kind="stable")
    return float(ordered[max(0, math.ceil(level * len(ordered)) - 1)])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve existing probe output: {OUT}")
    freeze = read_json(RUN / "freeze.json")
    pred_seal = read_json(RUN / "prediction_seal.json")
    cal_seal = read_json(RUN / "calibration_seal.json")
    completed = read_json(RUN / "completed.json")
    if (completed.get("status") != "COMPLETE" or completed.get("target_values_opened") is not False
            or cal_seal.get("risk_refits") != 0
            or pred_seal.get("freeze_sha256") != sha(RUN / "freeze.json")
            or cal_seal.get("freeze_sha256") != sha(RUN / "freeze.json")
            or cal_seal.get("prediction_seal_sha256") != sha(RUN / "prediction_seal.json")):
        raise ValueError("Completed and sealed V50 calibration source run required")

    prediction_entries = {(e["outer"], e["region"]): e for e in pred_seal["entries"]
                          if e.get("role") == "calibration" and e.get("region") in EXTERNAL}
    if set(prediction_entries) != {(outer, source) for outer in SOURCES for source in EXTERNAL}:
        raise ValueError("Exact fixed 70 external calibration prediction packages required")
    label_entries = {e["region"]: e for e in cal_seal["labels"]}
    if set(label_entries) != set(EXTERNAL):
        raise ValueError("Exact fixed 10 external calibration label packages required")

    outcome = {source: {"p90": [], "p99": []} for source in EXTERNAL}
    pos_feature: dict[str, float] = {}
    opened_prediction_paths, opened_label_paths = [], []
    for source in EXTERNAL:
        label_path = RUN / "external_labels" / f"{source}.npz"
        label_entry = label_entries[source]
        if label_entry["path"] != str(label_path) or sha(label_path) != label_entry["sha256"]:
            raise ValueError("External label archive fails its registered hash")
        with np.load(label_path, allow_pickle=False) as z:
            positions = z["positions"].astype(int, copy=True)
            truth = z["truth"].astype(float, copy=True)
            origins = z["origins"].astype(np.int64, copy=True)
        if (truth.shape != (len(positions), 24) or not np.isfinite(truth).all()
                or not np.array_equal(origins, helper.CAL_DAYS.asi8[positions])):
            raise ValueError("External calibration label support or values invalid")
        eligible = helper.eligible(pd.DatetimeIndex(pd.to_datetime(origins, utc=True)), 90)
        if not eligible.any() or int(eligible.sum()) != freeze["lag_counts"]["90"][source]:
            raise ValueError("Registered 90-day label support mismatch")
        opened_label_paths.append(str(label_path))

        expected_context = None
        for outer in SOURCES:
            entry = prediction_entries[(outer, source)]
            prediction_path = RUN / "predictions" / f"{outer}__{source}.npz"
            if entry["path"] != str(prediction_path) or sha(prediction_path) != entry["sha256"]:
                raise ValueError("External prediction archive fails its registered hash")
            with np.load(prediction_path, allow_pickle=False) as z:
                if (z["outer"].item() != outer or z["region"].item() != source
                        or z["arms"].tolist() != list(helper.BASES)
                        or z["methods"].tolist() != ["GlobalABS", "MCStd", "Component", "HETScale",
                                                       "HETLocationScale", "CQRplus"]
                        or not np.array_equal(z["origins"], helper.CAL_DAYS.asi8)):
                    raise ValueError("External archive is not on the registered calibration grid")
                contexts = np.asarray(z["context_g"], float)
                if contexts.shape != (len(helper.CAL_DAYS), 3, 5) or not np.isfinite(contexts).all():
                    raise ValueError("Visible legal monthly contexts have invalid shape or values")
                if expected_context is None:
                    expected_context = contexts.copy()
                elif not np.array_equal(contexts, expected_context):
                    raise ValueError("Point-function outer changed the same source's legal context history")
                lo = np.take(z["lo0"][0, 1], positions, axis=0)[eligible]
                hi = np.take(z["hi0"][0, 1], positions, axis=0)[eligible]
                scale = np.take(z["scales"][0, 1], positions, axis=0)[eligible]
            y = truth[eligible]
            scores = np.maximum(np.maximum((lo - y) / scale, (y - hi) / scale), 0.0)
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite standardized hourly calibration scores")
            outcome[source]["p90"].append(inverse(scores, .90))
            outcome[source]["p99"].append(inverse(scores, .99))
            opened_prediction_paths.append(str(prediction_path))
        # Average the three latest legally released monthly POS shares over the registered query dates.
        pos_feature[source] = float(expected_context[:, :, 0].mean())

    rows = []
    for source in EXTERNAL:
        rows.append({
            "source": source,
            "mean_legal_context_POS_share": pos_feature[source],
            "median_outer_p90_score": float(np.median(outcome[source]["p90"])),
            "median_outer_p99_score": float(np.median(outcome[source]["p99"])),
            "outer_p90_scores": json.dumps(outcome[source]["p90"], separators=(",", ":")),
            "outer_p99_scores": json.dumps(outcome[source]["p99"], separators=(",", ":")),
        })
    table = pd.DataFrame(rows)
    x = table["mean_legal_context_POS_share"].to_numpy(float)
    if np.ptp(x) <= 1e-12:
        raise ValueError("Registered single coarse predictor has no cross-source variation")

    cv_rows = []
    for target_i, target in enumerate(EXTERNAL):
        train = np.arange(len(EXTERNAL)) != target_i
        train_x, train_y = x[train], table["median_outer_p90_score"].to_numpy(float)[train]
        train_design = np.column_stack([np.ones(train.sum()), train_x])
        coef = np.linalg.lstsq(train_design, train_y, rcond=None)[0]
        linear_prediction = float(np.array([1.0, x[target_i]]) @ coef)
        mean_prediction = float(train_y.mean())
        truth = float(table.loc[target_i, "median_outer_p90_score"])
        tail_y = table["median_outer_p99_score"].to_numpy(float)
        tail_design = np.column_stack([np.ones(train.sum()), train_x])
        tail_coef = np.linalg.lstsq(tail_design, tail_y[train], rcond=None)[0]
        tail_prediction = float(np.array([1.0, x[target_i]]) @ tail_coef)
        cv_rows.append({
            "held_source": target, "train_source_count": int(train.sum()),
            "mean_POS_share": x[target_i], "observed_p90_score": truth,
            "null_mean_prediction_p90": mean_prediction, "POS_linear_prediction_p90": linear_prediction,
            "null_abs_error_p90": abs(truth - mean_prediction),
            "POS_abs_error_p90": abs(truth - linear_prediction),
            "observed_p99_score": float(tail_y[target_i]), "POS_linear_prediction_p99": tail_prediction,
            "POS_fold_slope_p90": float(coef[1]), "POS_fold_slope_p99": float(tail_coef[1]),
        })

    cv = pd.DataFrame(cv_rows)
    OUT.mkdir()
    source_path, cv_path = OUT / "source_risk_and_mix.csv", OUT / "source_heldout_predictions.csv"
    table.to_csv(source_path, index=False)
    cv.to_csv(cv_path, index=False)
    summary = {
        "status": "COMPLETE_EXPLORATORY_INFORMATION_VALUE_PROBE",
        "protocol": "docs/V50_COARSE_MIX_RISK_PREDICTABILITY_PROTOCOL.md",
        "protocol_sha256": sha(ROOT / "docs/V50_COARSE_MIX_RISK_PREDICTABILITY_PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)),
        "prediction_packages_opened": len(opened_prediction_paths),
        "external_label_packages_opened": len(opened_label_paths),
        "target_prediction_packages_opened": 0,
        "target_scoring_files_opened": 0,
        "point_or_risk_refits": 0,
        "diagnostic_regressions": 20,
        "source_units": 10,
        "POS_feature_definition": "mean of visible POS share over three latest legally released monthly contexts, averaged across query dates",
        "primary_p90": {
            "null_mae": float(cv.null_abs_error_p90.mean()),
            "POS_linear_mae": float(cv.POS_abs_error_p90.mean()),
            "null_rmse": float(np.sqrt(np.mean(cv.null_abs_error_p90.to_numpy() ** 2))),
            "POS_linear_rmse": float(np.sqrt(np.mean(cv.POS_abs_error_p90.to_numpy() ** 2))),
            "fold_slope_positive_fraction": float(np.mean(cv.POS_fold_slope_p90 > 0)),
        },
        "p99_secondary": {
            "POS_linear_mae": float(np.mean(np.abs(
                cv.observed_p99_score.to_numpy() - cv.POS_linear_prediction_p99.to_numpy()))),
            "POS_linear_rmse": float(np.sqrt(np.mean((cv.observed_p99_score.to_numpy()
                - cv.POS_linear_prediction_p99.to_numpy()) ** 2))),
            "fold_slope_positive_fraction": float(np.mean(cv.POS_fold_slope_p99 > 0)),
            "not_a_selection_metric": True,
        },
        "input_hashes": {
            "freeze": sha(RUN / "freeze.json"), "prediction_seal": sha(RUN / "prediction_seal.json"),
            "calibration_seal": sha(RUN / "calibration_seal.json"),
        },
        "source_table_sha256": sha(source_path), "heldout_table_sha256": sha(cv_path),
        "interpretation_limit": "Only ten previously used source regions; hypothesis-generating and not evidence of target-region calibration or interval efficiency.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "source_units": summary["source_units"],
        "p90_null_mae": summary["primary_p90"]["null_mae"],
        "p90_POS_linear_mae": summary["primary_p90"]["POS_linear_mae"],
        "p90_null_rmse": summary["primary_p90"]["null_rmse"],
        "p90_POS_linear_rmse": summary["primary_p90"]["POS_linear_rmse"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
