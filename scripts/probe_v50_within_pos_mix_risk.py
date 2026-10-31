"""Source-held-out probe of legal monthly within-POS mix for hourly-risk heterogeneity."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/disjoint-calibration-v50"
PRIOR = ROOT / "results/disjoint-calibration-v50-coarse-mix-probe"
OUT = ROOT / "results/disjoint-calibration-v50-within-pos-probe"
sys.path.insert(0, str(ROOT))
from scripts import disjoint_calibration_v50 as helper  # noqa: E402

EXTERNAL = helper.EXTERNAL
FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
POS_EF = {"coal": 1000.0, "gas": 400.0, "oil": 800.0, "other": 400.0}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def metric(errors: np.ndarray) -> dict[str, float]:
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2)))}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve existing probe output: {OUT}")
    freeze = read_json(RUN / "freeze.json")
    if not freeze.get("input_hashes"):
        raise ValueError("Hash-bound V50 source inputs required")
    prior_summary = read_json(PRIOR / "summary.json")
    source_table_path = PRIOR / "source_risk_and_mix.csv"
    if (prior_summary.get("status") != "COMPLETE_EXPLORATORY_INFORMATION_VALUE_PROBE"
            or sha(source_table_path) != prior_summary.get("source_table_sha256")):
        raise ValueError("Completed, hash-matched prior source-level outcome required")
    source_outcome = pd.read_csv(source_table_path)
    if set(source_outcome.source) != set(EXTERNAL) or len(source_outcome) != 10:
        raise ValueError("Exact ten registered source outcomes required")
    outcome = source_outcome.set_index("source")

    monthly_paths = [Path(path) for path in freeze["external_monthly"]]
    monthly_rows: dict[str, list[dict[str, str]]] = {source: [] for source in EXTERNAL}
    for path in monthly_paths:
        expected_hash = freeze["input_hashes"].get(str(path))
        if expected_hash is None or sha(path) != expected_hash:
            raise ValueError("Monthly input path/hash is not the original V50 frozen input")
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"region", "period_start", "period_end", "available_at",
                        *("fuelshare_" + fuel for fuel in FUELS)}
            if not required <= set(reader.fieldnames or []):
                raise ValueError("Frozen monthly fuel mix fields are incomplete")
            for row in reader:
                source = row["region"]
                if source in EXTERNAL:
                    monthly_rows[source].append(row)

    features = {}
    support_summary = {}
    for source in EXTERNAL:
        records = []
        seen = set()
        for row in monthly_rows[source]:
            start = pd.Timestamp(row["period_start"])
            end = pd.Timestamp(row["period_end"])
            release = pd.Timestamp(row["available_at"])
            for at in (start, end, release):
                if at.tzinfo is None:
                    raise ValueError("Timezone-aware monthly period and release timestamps required")
            start, end, release = (at.tz_convert("UTC") for at in (start, end, release))
            if (start != start.normalize() or start.day != 1 or end != start + pd.offsets.MonthBegin(1)
                    or release < end):
                raise ValueError("Whole-month and post-period release metadata required")
            key = start.value
            if key in seen:
                raise ValueError("Duplicate region-month vintage in frozen monthly support")
            seen.add(key)
            share = np.array([float(row["fuelshare_" + fuel]) for fuel in FUELS], dtype=float)
            if (not np.isfinite(share).all() or np.any(share < 0)
                    or not np.isclose(share.sum(), 1.0, atol=1e-8, rtol=0)):
                raise ValueError("Finite normalized eight-fuel monthly share required")
            pos_mass = sum(share[FUELS.index(fuel)] for fuel in POS_EF)
            if pos_mass <= 0:
                intensity = None
            else:
                intensity = sum(POS_EF[fuel] * share[FUELS.index(fuel)] for fuel in POS_EF) / pos_mass
            records.append((start, end, release, intensity))
        records.sort(key=lambda item: item[0])
        if not records:
            raise ValueError("No frozen monthly records for external source")
        observed, contexts_used = [], 0
        for query_day in helper.CAL_DAYS:
            legal = [r for r in records if r[0] >= pd.Timestamp("2022-01-01", tz="UTC")
                     and r[0] < query_day and r[1] <= query_day and r[2] <= query_day]
            selected = legal[-3:][::-1]
            if len(selected) != 3 or any(r[3] is None for r in selected):
                raise ValueError("Every registered query date needs three legal positive-POS monthly contexts")
            observed.extend(r[3] for r in selected)
            contexts_used += len(selected)
        features[source] = float(np.mean(observed))
        support_summary[source] = {"query_dates": len(helper.CAL_DAYS), "monthly_contexts": contexts_used}

    ordered = list(EXTERNAL)
    x = np.array([features[source] for source in ordered], dtype=float)
    y90 = outcome.loc[ordered, "median_outer_p90_score"].to_numpy(float)
    y99 = outcome.loc[ordered, "median_outer_p99_score"].to_numpy(float)
    if not np.isfinite(x).all() or np.ptp(x) <= 1e-12:
        raise ValueError("Registered within-POS predictor must vary across sources")

    folds = []
    for i, source in enumerate(ordered):
        train = np.arange(len(ordered)) != i
        design = np.column_stack([np.ones(train.sum()), x[train]])
        coef90 = np.linalg.lstsq(design, y90[train], rcond=None)[0]
        coef99 = np.linalg.lstsq(design, y99[train], rcond=None)[0]
        predicted90, predicted99 = (np.array([1.0, x[i]]) @ coef90,
                                    np.array([1.0, x[i]]) @ coef99)
        folds.append({
            "held_source": source, "mean_legal_within_POS_EF": x[i],
            "observed_p90_score": y90[i], "null_mean_p90": float(y90[train].mean()),
            "within_POS_linear_p90": float(predicted90),
            "null_abs_error_p90": abs(y90[i] - y90[train].mean()),
            "within_POS_abs_error_p90": abs(y90[i] - predicted90),
            "observed_p99_score": y99[i], "null_mean_p99": float(y99[train].mean()),
            "within_POS_linear_p99": float(predicted99),
            "null_abs_error_p99": abs(y99[i] - y99[train].mean()),
            "within_POS_abs_error_p99": abs(y99[i] - predicted99),
            "fold_slope_p90": float(coef90[1]), "fold_slope_p99": float(coef99[1]),
        })
    folds_df = pd.DataFrame(folds)
    source_df = pd.DataFrame([{"source": source, "mean_legal_within_POS_EF": features[source],
                               **support_summary[source]} for source in ordered])
    OUT.mkdir()
    feature_path, prediction_path = OUT / "source_feature.csv", OUT / "source_heldout_predictions.csv"
    source_df.to_csv(feature_path, index=False)
    folds_df.to_csv(prediction_path, index=False)
    p90_null = np.array([folds_df.null_abs_error_p90.mean(),
                         np.sqrt(np.mean(folds_df.null_abs_error_p90.to_numpy()**2))])
    p90_mix = np.array([folds_df.within_POS_abs_error_p90.mean(),
                        np.sqrt(np.mean(folds_df.within_POS_abs_error_p90.to_numpy()**2))])
    p99_null = np.array([folds_df.null_abs_error_p99.mean(),
                         np.sqrt(np.mean(folds_df.null_abs_error_p99.to_numpy()**2))])
    p99_mix = np.array([folds_df.within_POS_abs_error_p99.mean(),
                        np.sqrt(np.mean(folds_df.within_POS_abs_error_p99.to_numpy()**2))])
    summary = {
        "status": "COMPLETE_EXPLORATORY_INFORMATION_VALUE_PROBE",
        "protocol": "docs/V50_WITHIN_POS_MIX_RISK_PROTOCOL.md",
        "protocol_sha256": sha(ROOT / "docs/V50_WITHIN_POS_MIX_RISK_PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)),
        "source_outcome_sha256": sha(source_table_path),
        "monthly_input_hashes": {str(path): sha(path) for path in monthly_paths},
        "source_units": 10, "source_heldout_folds": 10,
        "point_or_risk_refits": 0, "diagnostic_regressions": 20,
        "target_or_unseen_source_read": False,
        "within_POS_EF_definition": "monthly coal/gas/oil/other share-weighted project EF; normalized within POS; averaged over three latest legal monthly contexts and query dates",
        "p90_LOO": {"null_MAE": float(p90_null[0]), "within_POS_MAE": float(p90_mix[0]),
                    "null_RMSE": float(p90_null[1]), "within_POS_RMSE": float(p90_mix[1]),
                    "positive_slope_fold_fraction": float(np.mean(folds_df.fold_slope_p90 > 0))},
        "p99_LOO_secondary": {"null_MAE": float(p99_null[0]), "within_POS_MAE": float(p99_mix[0]),
                    "null_RMSE": float(p99_null[1]), "within_POS_RMSE": float(p99_mix[1]),
                    "positive_slope_fold_fraction": float(np.mean(folds_df.fold_slope_p99 > 0))},
        "feature_csv_sha256": sha(feature_path), "heldout_csv_sha256": sha(prediction_path),
        "interpretation_limit": "Ten previously used source regions and a proxy EF convention; exploratory only, not Chinese CIF validation or proof of interval efficiency.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "p90_null_MAE": summary["p90_LOO"]["null_MAE"],
        "p90_within_POS_MAE": summary["p90_LOO"]["within_POS_MAE"],
        "p90_null_RMSE": summary["p90_LOO"]["null_RMSE"],
        "p90_within_POS_RMSE": summary["p90_LOO"]["within_POS_RMSE"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
