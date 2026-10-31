"""Nested source-held-out selection of monthly fossil-role correction strength."""
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
OUT = ROOT / "results/disjoint-calibration-v50-nested-fossil-role"
sys.path.insert(0, str(ROOT))
from scripts import disjoint_calibration_v50 as helper  # noqa: E402

SOURCES, EXTERNAL = helper.LEGACY, helper.EXTERNAL
BASES = helper.BASES
METHODS = ("GlobalABS", "MCStd", "Component", "HETScale", "HETLocationScale", "CQRplus")
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


def crps(samples: np.ndarray, observation: np.ndarray) -> np.ndarray:
    x, y = np.asarray(samples, float), np.asarray(observation, float)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite case/sample CRPS arrays required")
    ordered = np.sort(x, axis=1, kind="stable")
    n = ordered.shape[1]
    rank_weight = 2 * np.arange(1, n + 1) - n - 1
    half_pairwise = np.sum(ordered * rank_weight[None, :], axis=1) / (n * n)
    return np.mean(np.abs(x - y[:, None]), axis=1) - half_pairwise


def metrics(raw: np.ndarray, truth: np.ndarray, daily_offset: np.ndarray) -> dict[str, float]:
    shifted = np.clip(raw + daily_offset[:, None, None], 0.0, 1000.0)
    center = shifted.mean(axis=1, keepdims=True)
    expanded = np.sort(np.clip(center + 2.0 * (shifted - center), 0.0, 1000.0), axis=1)
    lo, hi = expanded[:, 6], expanded[:, 121]
    hit = (truth >= lo) & (truth <= hi)
    lower, upper = truth < lo, truth > hi
    width = hi - lo
    score = width + 20 * np.maximum(lo - truth, 0) + 20 * np.maximum(truth - hi, 0)
    hourly_samples = shifted.transpose(0, 2, 1).reshape(-1, shifted.shape[1])
    hourly_score = crps(hourly_samples, truth.reshape(-1)).mean()
    daily_samples = shifted.mean(axis=2)
    truth_daily = truth.mean(axis=1)
    center_daily = shifted.mean(axis=1).mean(axis=1)
    return {
        "hourly_crps": float(hourly_score), "daily_mean_crps": float(crps(daily_samples, truth_daily).mean()),
        "daily_level_mae": float(np.mean(np.abs(center_daily - truth_daily))),
        "daily_level_bias_prediction_minus_truth": float(np.mean(center_daily - truth_daily)),
        "coverage90": float(hit.mean()), "width90": float(width.mean()),
        "interval_score90": float(score.mean()), "lower_miss": float(lower.mean()),
        "upper_miss": float(upper.mean()),
        "fraction_clipped_after_shift": float(np.mean((raw + daily_offset[:, None, None] < 0)
                                                        | (raw + daily_offset[:, None, None] > 1000))),
    }


def monthly_rows(freeze: dict) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float, np.ndarray]]]:
    values = {source: [] for source in EXTERNAL}
    paths = [Path(path) for path in freeze["external_monthly"]]
    for path in paths:
        if sha(path) != freeze["input_hashes"].get(str(path)):
            raise ValueError("External monthly file hash differs from V50 freeze")
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"region", "period_start", "period_end", "available_at",
                        *("fuelshare_" + fuel for fuel in FUELS)}
            if not required <= set(reader.fieldnames or []):
                raise ValueError("Registered full fuel-share columns missing")
            for row in reader:
                source = row["region"]
                if source not in EXTERNAL:
                    continue
                start, end, released = (pd.Timestamp(row[key]) for key in ("period_start", "period_end", "available_at"))
                if any(stamp.tzinfo is None for stamp in (start, end, released)):
                    raise ValueError("Timezone-aware monthly timing required")
                start, end, released = (stamp.tz_convert("UTC") for stamp in (start, end, released))
                if (start != start.normalize() or start.day != 1 or end != start + pd.offsets.MonthBegin(1)
                        or released < end):
                    raise ValueError("Whole calendar month with release after period end required")
                share = np.array([float(row["fuelshare_" + fuel]) for fuel in FUELS], float)
                if (not np.isfinite(share).all() or np.any(share < 0)
                        or not np.isclose(share.sum(), 1., atol=1e-8, rtol=0)):
                    raise ValueError("Finite normalized eight-fuel shares required")
                values[source].append((start, end, released, float(share @ EF), share))
    for source in EXTERNAL:
        values[source].sort(key=lambda record: record[0])
        starts = [record[0].value for record in values[source]]
        if len(starts) != len(set(starts)):
            raise ValueError("Duplicate source-month vintages are not allowed")
        if not values[source]:
            raise ValueError("Missing external monthly history")
    return values


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve existing daily-as-of experiment: {OUT}")
    freeze = read_json(RUN / "freeze.json")
    prediction_seal, calibration_seal = read_json(RUN / "prediction_seal.json"), read_json(RUN / "calibration_seal.json")
    completed = read_json(RUN / "completed.json")
    if (completed.get("status") != "COMPLETE"
            or prediction_seal.get("freeze_sha256") != sha(RUN / "freeze.json")
            or calibration_seal.get("freeze_sha256") != sha(RUN / "freeze.json")
            or calibration_seal.get("prediction_seal_sha256") != sha(RUN / "prediction_seal.json")
            or calibration_seal.get("risk_refits") != 0):
        raise ValueError("Completed sealed V50 run required")

    records = monthly_rows(freeze)
    daily_feature, daily_groups = {}, {}
    for source in EXTERNAL:
        source_rows = records[source]
        feature, groups = [], []
        for day in helper.CAL_DAYS:
            available = [row for row in source_rows if row[0] >= pd.Timestamp("2022-01-01", tz="UTC")
                         and row[0] < day and row[1] <= day and row[2] <= day]
            chosen = available[-3:][::-1]
            if len(chosen) != 3:
                raise ValueError("Every prediction day must have three legally released monthly contexts")
            feature.append(float(np.mean([row[3] for row in chosen])))
            share = np.mean(np.stack([row[4] for row in chosen]), axis=0)
            groups.append(np.array([share[[0, 1, 2, 7]].sum(), share[[5, 6]].sum(), share[3]], dtype=float))
        daily_feature[source] = np.asarray(feature)
        daily_groups[source] = np.stack(groups)

    entries = {(entry["outer"], entry["region"]): entry for entry in prediction_seal["entries"]
               if entry.get("role") == "calibration" and entry.get("region") in EXTERNAL}
    labels_entries = {entry["region"]: entry for entry in calibration_seal["labels"]}
    if (set(entries) != {(outer, source) for outer in SOURCES for source in EXTERNAL}
            or set(labels_entries) != set(EXTERNAL)):
        raise ValueError("Exact fixed external grid is required")

    label_cache, raw_cache, opened_paths = {}, {}, []
    for source in EXTERNAL:
        label_path = RUN / "external_labels" / f"{source}.npz"
        if labels_entries[source]["path"] != str(label_path) or sha(label_path) != labels_entries[source]["sha256"]:
            raise ValueError("External source label hash mismatch")
        with np.load(label_path, allow_pickle=False) as z:
            origins, positions, truth = z["origins"].astype(np.int64), z["positions"].astype(int), z["truth"].astype(float)
        if (not np.array_equal(origins, helper.CAL_DAYS.asi8[positions])
                or truth.shape != (len(positions), 24) or not np.isfinite(truth).all()):
            raise ValueError("External labels do not match fixed calendar support")
        keep = helper.eligible(pd.DatetimeIndex(pd.to_datetime(origins, utc=True)), 90)
        if int(keep.sum()) != freeze["lag_counts"]["90"][source]:
            raise ValueError("90-day lag support does not match freeze")
        label_cache[source] = (positions[keep], truth[keep], daily_feature[source][positions[keep]],
                               daily_groups[source][positions[keep]])
        raw_cache[source] = {}
        for outer in SOURCES:
            entry = entries[(outer, source)]
            path = RUN / "predictions" / f"{outer}__{source}.npz"
            if entry["path"] != str(path) or sha(path) != entry["sha256"]:
                raise ValueError("External prediction package hash mismatch")
            with np.load(path, allow_pickle=False) as z:
                if (z["outer"].item() != outer or z["region"].item() != source
                        or z["arms"].tolist() != list(BASES) or z["methods"].tolist() != list(METHODS)
                        or not np.array_equal(z["origins"], helper.CAL_DAYS.asi8)):
                    raise ValueError("Prediction package identity mismatch")
                raw_cache[source][outer] = np.take(z["raw"][0], label_cache[source][0], axis=0).astype(float)
            opened_paths.append(str(path))

    def fit_increment(train_sources: tuple[str, ...], validation: str, outer: str,
                      train_bias: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        xs, fossil_parts, ys, weights = [], [], [], []
        for source in train_sources:
            xs.append(label_cache[source][2])
            fossil_parts.append(label_cache[source][3][:, 0])
            ys.append(train_bias[source])
            weights.append(np.full(len(train_bias[source]), 1.0 / len(train_bias[source])))
        x, fossil, y, weight = (np.concatenate(xs), np.concatenate(fossil_parts),
                                np.concatenate(ys), np.concatenate(weights))
        root_weight = np.sqrt(weight)
        x_mean, f_mean = float(np.average(x, weights=weight)), float(np.average(fossil, weights=weight))
        x_scale = float(np.sqrt(np.average((x - x_mean) ** 2, weights=weight)))
        f_scale = float(np.sqrt(np.average((fossil - f_mean) ** 2, weights=weight)))
        x_scale = x_scale if x_scale > 1e-12 else 1.0
        f_scale = f_scale if f_scale > 1e-12 else 1.0
        zx, zf = (x - x_mean) / x_scale, (fossil - f_mean) / f_scale
        design = np.column_stack([np.ones(len(x)), zx])
        beta_ef = np.linalg.lstsq(design * root_weight[:, None], y * root_weight, rcond=None)[0]
        residual_y = y - design @ beta_ef
        beta_project = np.linalg.lstsq(design * root_weight[:, None], zf * root_weight, rcond=None)[0]
        residual_f = zf - design @ beta_project
        denominator = float(np.sum(weight * residual_f ** 2))
        beta_fossil = 0.0 if denominator <= 1e-12 else float(np.sum(weight * residual_f * residual_y) / denominator)
        held_x = label_cache[validation][2]
        held_fossil = label_cache[validation][3][:, 0]
        held_design = np.column_stack([np.ones(len(held_x)), (held_x - x_mean) / x_scale])
        held_residual_f = (held_fossil - f_mean) / f_scale - held_design @ beta_project
        return held_design @ beta_ef, beta_fossil * held_residual_f

    rows, bias_rows, selection_rows = [], [], []
    lambdas = (0.0, 0.25, 0.50, 0.75, 1.0)
    for outer in SOURCES:
        train_bias = {}
        for source in EXTERNAL:
            raw, truth = raw_cache[source][outer], label_cache[source][1]
            train_bias[source] = (truth - raw.mean(axis=1)).mean(axis=1)
        for held in EXTERNAL:
            outer_train = tuple(source for source in EXTERNAL if source != held)
            inner_metrics = {lam: [] for lam in lambdas}
            for inner_held in outer_train:
                inner_train = tuple(source for source in outer_train if source != inner_held)
                ef_prediction, increment = fit_increment(inner_train, inner_held, outer, train_bias)
                raw_inner, truth_inner = raw_cache[inner_held][outer], label_cache[inner_held][1]
                for lam in lambdas:
                    inner_metrics[lam].append(metrics(raw_inner, truth_inner, ef_prediction + lam * increment))
            diagnostics, feasible = {}, []
            for lam in lambdas:
                scores = inner_metrics[lam]
                mean_coverage = float(np.mean([score["coverage90"] for score in scores]))
                worst_coverage = float(np.min([score["coverage90"] for score in scores]))
                mean_is = float(np.mean([score["interval_score90"] for score in scores]))
                diagnostics[lam] = {"mean_coverage": mean_coverage, "worst_coverage": worst_coverage,
                                    "mean_interval_score": mean_is}
                if mean_coverage >= 0.90 and worst_coverage >= 0.85:
                    feasible.append(lam)
            selected = min(feasible, key=lambda lam: diagnostics[lam]["mean_interval_score"]) if feasible else 0.0
            global_offset = float(np.mean([train_bias[source].mean() for source in outer_train]))
            ef_prediction, increment = fit_increment(outer_train, held, outer, train_bias)
            full_prediction = ef_prediction + increment
            selected_prediction = ef_prediction + selected * increment
            selection_rows.append({"outer": outer, "held_source": held, "selected_lambda": selected,
                                   "inner_feasible_count": len(feasible),
                                   "inner_diagnostics_json": json.dumps(diagnostics)})
            held_bias = train_bias[held]
            bias_rows.append({"outer": outer, "held_source": held, "eligible_days": len(held_bias),
                              "mean_daily_legal_EF_proxy": float(label_cache[held][2].mean()),
                              "observed_source_level_bias": float(held_bias.mean()), "selected_lambda": selected,
                              "ef_only_abs_source_bias_after": abs(float(np.mean(held_bias - ef_prediction))),
                              "nested_abs_source_bias_after": abs(float(np.mean(held_bias - selected_prediction)))})
            raw, truth = raw_cache[held][outer], label_cache[held][1]
            offsets = {"no_correction_factor2_anchor": np.zeros(len(held_bias)),
                       "global_source_bias_shift": np.full(len(held_bias), global_offset),
                       "ef_only_monthly_mix": ef_prediction,
                       "full_fossil_increment_lambda1": full_prediction,
                       "nested_selected_fossil_increment": selected_prediction}
            for method, daily_offset in offsets.items():
                rows.append({"outer": outer, "held_source": held, "method": method,
                             "selected_lambda": selected, "mean_predicted_offset": float(daily_offset.mean()),
                             **metrics(raw, truth, daily_offset)})

    if len(rows) != len(SOURCES) * len(EXTERNAL) * 5:
        raise ValueError("Expected all 350 outer × held-source × method rows")
    OUT.mkdir()
    fold_path, bias_path = OUT / "held_source_fold_metrics.csv", OUT / "held_source_bias_summary.csv"
    pd.DataFrame(rows).to_csv(fold_path, index=False)
    pd.DataFrame(bias_rows).to_csv(bias_path, index=False)
    selection_path = OUT / "nested_selection.csv"
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    fold = pd.DataFrame(rows)
    source_macro = fold.groupby(["held_source", "method"], sort=False)[
        ["hourly_crps", "daily_mean_crps", "daily_level_mae", "daily_level_bias_prediction_minus_truth",
         "coverage90", "width90", "interval_score90", "lower_miss", "upper_miss",
         "fraction_clipped_after_shift"]].mean().reset_index()
    source_path = OUT / "source_macro_metrics.csv"
    source_macro.to_csv(source_path, index=False)
    overall = source_macro.groupby("method", sort=False)[
        ["hourly_crps", "daily_mean_crps", "daily_level_mae", "coverage90", "width90", "interval_score90",
         "lower_miss", "upper_miss", "fraction_clipped_after_shift"]].mean().reset_index()
    bias = pd.DataFrame(bias_rows).groupby("held_source", sort=False)[
        ["observed_source_level_bias", "ef_only_abs_source_bias_after", "nested_abs_source_bias_after"]].mean().reset_index()
    summary = {
        "status": "COMPLETE_NESTED_FOSSIL_ROLE_SELECTION",
        "protocol": "docs/V50_NESTED_FOSSIL_ROLE_PROTOCOL.md",
        "protocol_sha256": sha(ROOT / "docs/V50_NESTED_FOSSIL_ROLE_PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)), "source_units": len(EXTERNAL),
        "prediction_packages_opened": len(opened_paths), "external_label_packages_opened": len(label_cache),
        "target_prediction_packages_opened": 0, "target_or_unseen_scores_opened": 0,
        "point_or_risk_refits": 0, "outer_source_heldout_fits": len(SOURCES) * len(EXTERNAL),
        "nested_inner_source_heldout_fits": len(SOURCES) * len(EXTERNAL) * (len(EXTERNAL) - 1),
        "selected_lambda_counts": {str(k): int(v) for k, v in
                                    pd.Series([row["selected_lambda"] for row in selection_rows]).value_counts().items()},
        "source_heldout_rows": len(rows), "lag_days": 90,
        "full_mix_EF_definition": "mean fixed-EF intensity of three latest legally released monthly eight-fuel records at each query date",
        "composition_groups": {"fossil": ["coal", "gas", "oil", "other"]},
        "nested_lambda_grid": list(lambdas),
        "selection_rule": "min mean inner-source interval score subject to mean coverage>=0.90 and worst source>=0.85; fallback lambda=0",
        "daily_source_bias_mae": {"ef_only": float(bias.ef_only_abs_source_bias_after.mean()),
                                  "nested_selected": float(bias.nested_abs_source_bias_after.mean())},
        "source_macro_metrics": overall.to_dict(orient="records"),
        "frozen_input_hashes": {"freeze": sha(RUN / "freeze.json"),
                                 "prediction_seal": sha(RUN / "prediction_seal.json"),
                                 "calibration_seal": sha(RUN / "calibration_seal.json"),
                                 **{str(path): sha(path) for path in map(Path, freeze["external_monthly"])}},
        "held_source_fold_metrics_sha256": sha(fold_path), "held_source_bias_sha256": sha(bias_path),
        "nested_selection_sha256": sha(selection_path),
        "source_macro_metrics_sha256": sha(source_path),
        "interpretation_limit": "Ten previously used sources only; source-held-out descriptive evidence, not China transfer, novel method proof, or a coverage guarantee.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "daily_source_bias_mae": summary["daily_source_bias_mae"],
        "source_macro_metrics": summary["source_macro_metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
