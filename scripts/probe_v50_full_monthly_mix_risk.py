"""Source-held-out probe of a legal full monthly fuel-mix proxy for hourly risk."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/disjoint-calibration-v50"
PRIOR = ROOT / "results/disjoint-calibration-v50-coarse-mix-probe"
OUT = ROOT / "results/disjoint-calibration-v50-full-mix-probe"
sys.path.insert(0, str(ROOT))
from scripts import disjoint_calibration_v50 as helper  # noqa: E402

EXTERNAL = helper.EXTERNAL
FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve existing probe output: {OUT}")
    freeze = read_json(RUN / "freeze.json")
    prior_summary = read_json(PRIOR / "summary.json")
    prior_table = PRIOR / "source_risk_and_mix.csv"
    if (prior_summary.get("status") != "COMPLETE_EXPLORATORY_INFORMATION_VALUE_PROBE"
            or sha(prior_table) != prior_summary.get("source_table_sha256")):
        raise ValueError("Hash-matched completed source-risk outcomes required")
    outcomes = pd.read_csv(prior_table).set_index("source")
    if set(outcomes.index) != set(EXTERNAL) or len(outcomes) != 10:
        raise ValueError("Exact ten source-level outcomes required")

    monthly_paths = [Path(path) for path in freeze["external_monthly"]]
    rows_by_source: dict[str, list[dict[str, str]]] = {source: [] for source in EXTERNAL}
    for path in monthly_paths:
        if sha(path) != freeze["input_hashes"].get(str(path)):
            raise ValueError("Monthly source fails the original V50 frozen hash")
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"region", "period_start", "period_end", "available_at",
                        *("fuelshare_" + fuel for fuel in FUELS)}
            if not required <= set(reader.fieldnames or []):
                raise ValueError("Eight-fuel monthly share fields absent")
            for row in reader:
                if row["region"] in EXTERNAL:
                    rows_by_source[row["region"]].append(row)

    features, supports = {}, {}
    for source in EXTERNAL:
        records, seen = [], set()
        for row in rows_by_source[source]:
            start, end, released = (pd.Timestamp(row[name]) for name in ("period_start", "period_end", "available_at"))
            if any(value.tzinfo is None for value in (start, end, released)):
                raise ValueError("Timezone-aware calendar and release timestamps required")
            start, end, released = (value.tz_convert("UTC") for value in (start, end, released))
            if (start != start.normalize() or start.day != 1 or end != start + pd.offsets.MonthBegin(1)
                    or released < end):
                raise ValueError("Whole calendar months with post-period release required")
            if start.value in seen:
                raise ValueError("Duplicate monthly region vintage")
            seen.add(start.value)
            shares = np.array([float(row["fuelshare_" + fuel]) for fuel in FUELS], dtype=float)
            if (not np.isfinite(shares).all() or np.any(shares < 0)
                    or not np.isclose(shares.sum(), 1., atol=1e-8, rtol=0)):
                raise ValueError("Finite nonnegative full fuel simplex required")
            records.append((start, end, released, float(shares @ EF)))
        records.sort(key=lambda item: item[0])
        if not records:
            raise ValueError("No fixed external monthly rows for source")
        context_values = []
        for day in helper.CAL_DAYS:
            legal = [r for r in records if r[0] >= pd.Timestamp("2022-01-01", tz="UTC")
                     and r[0] < day and r[1] <= day and r[2] <= day]
            selected = legal[-3:][::-1]
            if len(selected) != 3:
                raise ValueError("Missing three fully released monthly contexts on registered query date")
            context_values.extend(r[3] for r in selected)
        features[source] = float(np.mean(context_values))
        supports[source] = {"query_dates": len(helper.CAL_DAYS), "monthly_contexts": len(context_values)}

    ordered = list(EXTERNAL)
    x = np.array([features[source] for source in ordered])
    y90 = outcomes.loc[ordered, "median_outer_p90_score"].to_numpy(float)
    y99 = outcomes.loc[ordered, "median_outer_p99_score"].to_numpy(float)
    if not np.isfinite(x).all() or np.ptp(x) <= 1e-12:
        raise ValueError("Registered monthly proxy has no finite cross-region variation")
    predictions = []
    for i, source in enumerate(ordered):
        train = np.arange(len(ordered)) != i
        design = np.column_stack([np.ones(train.sum()), x[train]])
        beta90 = np.linalg.lstsq(design, y90[train], rcond=None)[0]
        beta99 = np.linalg.lstsq(design, y99[train], rcond=None)[0]
        pred90, pred99 = float(np.array([1., x[i]]) @ beta90), float(np.array([1., x[i]]) @ beta99)
        predictions.append({
            "held_source": source, "monthly_EF_proxy": x[i],
            "observed_p90_score": y90[i], "null_mean_p90": float(y90[train].mean()),
            "monthly_mix_prediction_p90": pred90,
            "null_abs_error_p90": abs(y90[i] - y90[train].mean()),
            "monthly_mix_abs_error_p90": abs(y90[i] - pred90),
            "observed_p99_score": y99[i], "null_mean_p99": float(y99[train].mean()),
            "monthly_mix_prediction_p99": pred99,
            "null_abs_error_p99": abs(y99[i] - y99[train].mean()),
            "monthly_mix_abs_error_p99": abs(y99[i] - pred99),
            "fold_slope_p90": float(beta90[1]), "fold_slope_p99": float(beta99[1]),
        })
    pred = pd.DataFrame(predictions)
    feature_table = pd.DataFrame([{"source": source, "mean_legal_monthly_EF_proxy": features[source],
                                   **supports[source]} for source in ordered])
    OUT.mkdir()
    feature_path, pred_path = OUT / "source_feature.csv", OUT / "source_heldout_predictions.csv"
    feature_table.to_csv(feature_path, index=False)
    pred.to_csv(pred_path, index=False)

    def compare(field: str) -> dict[str, float]:
        null_error = pred[f"null_abs_error_{field}"].to_numpy(float)
        mix_error = pred[f"monthly_mix_abs_error_{field}"].to_numpy(float)
        return {"null_MAE": float(np.mean(null_error)), "monthly_mix_MAE": float(np.mean(mix_error)),
                "null_RMSE": float(np.sqrt(np.mean(null_error**2))),
                "monthly_mix_RMSE": float(np.sqrt(np.mean(mix_error**2))),
                "positive_slope_fold_fraction": float(np.mean(pred[f"fold_slope_{field}"] > 0))}
    summary = {
        "status": "COMPLETE_EXPLORATORY_INFORMATION_VALUE_PROBE",
        "protocol": "docs/V50_MONTHLY_FULL_MIX_RISK_PROTOCOL.md",
        "protocol_sha256": sha(ROOT / "docs/V50_MONTHLY_FULL_MIX_RISK_PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)),
        "source_outcome_sha256": sha(prior_table),
        "monthly_input_hashes": {str(path): sha(path) for path in monthly_paths},
        "source_units": 10, "source_heldout_folds": 10,
        "point_or_risk_refits": 0, "diagnostic_regressions": 20,
        "target_or_unseen_source_read": False,
        "feature_definition": "fixed EF-weighted monthly generation shares, averaged across three latest fully released contexts and July-Dec 2024 query dates",
        "p90_LOO": compare("p90"), "p99_LOO_secondary": compare("p99"),
        "feature_csv_sha256": sha(feature_path), "heldout_csv_sha256": sha(pred_path),
        "interpretation_limit": "Ten previously used source regions with a project proxy-EF convention; exploratory only, not Chinese CIF validation or proof of interval efficiency.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "p90_LOO": summary["p90_LOO"],
        "p99_LOO_secondary": summary["p99_LOO_secondary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
