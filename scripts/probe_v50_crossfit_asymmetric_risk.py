"""Cross-fitted symmetric vs asymmetric tail calibration after monthly location correction."""
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
OUT = ROOT / "results/disjoint-calibration-v50-crossfit-asymmetric-risk"
sys.path.insert(0, str(ROOT))
from scripts import disjoint_calibration_v50 as helper  # noqa: E402
from scripts import risk_baselines_v50 as risk  # noqa: E402

SOURCES, EXTERNAL = helper.LEGACY, helper.EXTERNAL
BASES = helper.BASES
METHODS = ("GlobalABS", "MCStd", "Component", "HETScale", "HETLocationScale", "CQRplus")
ARMS = {"Scale-anchor": 0, "B0-SizeET-AlignedKeep": 2, "B1-SizeCapRidge-AlignedKeep": 4}
RISK_METHODS = {"MCStd": 1, "HETScale": 3, "CQRplus": 5}
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


def interval_metrics(lo: np.ndarray, hi: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    hit = (truth >= lo) & (truth <= hi)
    lower, upper = truth < lo, truth > hi
    width = hi - lo
    score = width + 20 * np.maximum(lo - truth, 0) + 20 * np.maximum(truth - hi, 0)
    return {"coverage90": float(hit.mean()), "width90": float(width.mean()),
            "interval_score90": float(score.mean()), "lower_miss": float(lower.mean()),
            "upper_miss": float(upper.mean())}


def equal_source_quantile(scores: dict[str, np.ndarray], level: float) -> float:
    counts = {source: scores[source].shape[0] for source in scores}
    denominator = math.lcm(*(count * 24 for count in counts.values()))
    total_mass = len(scores) * denominator
    values, weights = [], []
    for source, score in scores.items():
        values.append(score.reshape(-1))
        mass = denominator // (score.shape[0] * 24)
        weights.append(np.full(score.size, mass, dtype=object))
    values, weights = np.concatenate(values), np.concatenate(weights)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=object)
    threshold = math.ceil(level * total_mass - 1e-12)
    return float(values[order[int(np.searchsorted(cumulative, threshold, side="left"))]])


def fit_location(train_sources: tuple[str, ...], held: str, train_bias: dict[str, np.ndarray],
                 label_cache: dict[str, tuple]) -> tuple[np.ndarray, np.ndarray]:
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
    held_x = label_cache[held][2]
    held_fossil = label_cache[held][3][:, 0]
    held_design = np.column_stack([np.ones(len(held_x)), (held_x - x_mean) / x_scale])
    held_residual_f = (held_fossil - f_mean) / f_scale - held_design @ beta_project
    return held_design @ beta_ef, beta_fossil * held_residual_f


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve existing cross-fitted risk experiment: {OUT}")
    freeze = read_json(RUN / "freeze.json")
    prediction_seal = read_json(RUN / "prediction_seal.json")
    calibration_seal = read_json(RUN / "calibration_seal.json")
    if (read_json(RUN / "completed.json").get("status") != "COMPLETE"
            or prediction_seal.get("freeze_sha256") != sha(RUN / "freeze.json")
            or calibration_seal.get("freeze_sha256") != sha(RUN / "freeze.json")
            or calibration_seal.get("prediction_seal_sha256") != sha(RUN / "prediction_seal.json")
            or calibration_seal.get("risk_refits") != 0):
        raise ValueError("Completed and hash-consistent V50 seals required")
    records = monthly_rows(freeze)
    daily_feature, daily_groups = {}, {}
    for source in EXTERNAL:
        features, groups = [], []
        for day in helper.CAL_DAYS:
            available = [row for row in records[source] if row[0] >= pd.Timestamp("2022-01-01", tz="UTC")
                         and row[0] < day and row[1] <= day and row[2] <= day]
            chosen = available[-3:][::-1]
            if len(chosen) != 3:
                raise ValueError("Each query needs three released full-month records")
            features.append(float(np.mean([row[3] for row in chosen])))
            share = np.mean(np.stack([row[4] for row in chosen]), axis=0)
            groups.append(np.array([share[[0, 1, 2, 7]].sum(), share[[5, 6]].sum(), share[3]], float))
        daily_feature[source], daily_groups[source] = np.asarray(features), np.stack(groups)

    prediction_entries = {(e["outer"], e["region"]): e for e in prediction_seal["entries"]
                          if e.get("role") == "calibration" and e.get("region") in EXTERNAL}
    label_entries = {e["region"]: e for e in calibration_seal["labels"]}
    if set(prediction_entries) != {(o, s) for o in SOURCES for s in EXTERNAL} or set(label_entries) != set(EXTERNAL):
        raise ValueError("Exact sealed external source grid required")
    label_cache, raw_cache, bounds_cache, opened = {}, {}, {}, []
    for source in EXTERNAL:
        label_path = RUN / "external_labels" / f"{source}.npz"
        if label_entries[source]["path"] != str(label_path) or sha(label_path) != label_entries[source]["sha256"]:
            raise ValueError("External label hash mismatch")
        with np.load(label_path, allow_pickle=False) as z:
            origins, positions, truth = z["origins"].astype(np.int64), z["positions"].astype(int), z["truth"].astype(float)
        keep = helper.eligible(pd.DatetimeIndex(pd.to_datetime(origins, utc=True)), 90)
        if (not np.array_equal(origins, helper.CAL_DAYS.asi8[positions]) or truth.shape != (len(positions), 24)
                or int(keep.sum()) != freeze["lag_counts"]["90"][source]):
            raise ValueError("External labels/support mismatch")
        positions, truth = positions[keep], truth[keep]
        label_cache[source] = (positions, truth, daily_feature[source][positions], daily_groups[source][positions])
        raw_cache[source], bounds_cache[source] = {}, {}
        for outer in SOURCES:
            path = RUN / "predictions" / f"{outer}__{source}.npz"
            entry = prediction_entries[(outer, source)]
            if entry["path"] != str(path) or sha(path) != entry["sha256"]:
                raise ValueError("Prediction package hash mismatch")
            with np.load(path, allow_pickle=False) as z:
                if (z["outer"].item() != outer or z["region"].item() != source
                        or z["arms"].tolist() != list(BASES) or z["methods"].tolist() != list(METHODS)
                        or not np.array_equal(z["origins"], helper.CAL_DAYS.asi8)):
                    raise ValueError("Prediction package identity/support mismatch")
                raw_cache[source][outer] = {arm_name: np.take(z["raw"][arm_index], positions, axis=0)
                                            for arm_name, arm_index in ARMS.items()}
                bounds_cache[source][outer] = {}
                for arm_name, arm_index in ARMS.items():
                    bounds_cache[source][outer][arm_name] = {}
                    for method_name, method_index in RISK_METHODS.items():
                        bounds_cache[source][outer][arm_name][method_name] = {
                            name: np.take(z[key][arm_index, method_index], positions, axis=0)
                            for name, key in (("lo", "lo0"), ("hi", "hi0"), ("scale", "scales"))}
            opened.append(str(path))

    rows, q_rows = [], []
    correction_names = ("unshifted", "ef_only")
    for outer in SOURCES:
        for arm_name in ARMS:
            train_bias = {}
            for source in EXTERNAL:
                raw = raw_cache[source][outer][arm_name]
                truth = label_cache[source][1]
                train_bias[source] = (truth - raw.mean(axis=1)).mean(axis=1)
            for held in EXTERNAL:
                train_sources = tuple(s for s in EXTERNAL if s != held)
                calibration_scores = {name: {method: {} for method in RISK_METHODS} for name in correction_names}
                # Cross-fit the location correction inside the nine-source calibration pool.
                for cal_source in train_sources:
                    offset_train = tuple(s for s in train_sources if s != cal_source)
                    ef_offset, fossil_increment = fit_location(offset_train, cal_source, train_bias, label_cache)
                    offsets = {"unshifted": np.zeros_like(ef_offset), "ef_only": ef_offset}
                    truth_cal = label_cache[cal_source][1]
                    for method_name in RISK_METHODS:
                        bound = bounds_cache[cal_source][outer][arm_name][method_name]
                        for correction_name, offset in offsets.items():
                            lower = np.maximum((bound["lo"] + offset[:, None] - truth_cal) / bound["scale"], 0.0)
                            upper = np.maximum((truth_cal - bound["hi"] - offset[:, None]) / bound["scale"], 0.0)
                            calibration_scores[correction_name][method_name][cal_source] = {
                                "symmetric": np.maximum(lower, upper), "lower": lower, "upper": upper}

                held_ef, held_fossil_increment = fit_location(train_sources, held, train_bias, label_cache)
                held_offsets = {"unshifted": np.zeros_like(held_ef), "ef_only": held_ef}
                truth_held = label_cache[held][1]
                for method_name in RISK_METHODS:
                    held_bound = bounds_cache[held][outer][arm_name][method_name]
                    for correction_name in correction_names:
                        by_source = calibration_scores[correction_name][method_name]
                        symmetric_by_source = {s: by_source[s]["symmetric"] for s in train_sources}
                        lower_by_source = {s: by_source[s]["lower"] for s in train_sources}
                        upper_by_source = {s: by_source[s]["upper"] for s in train_sources}
                        score_matrix = np.concatenate([symmetric_by_source[s] for s in train_sources], axis=0)
                        region_labels = [s for s in train_sources
                                         for _ in range(by_source[s]["symmetric"].shape[0])]
                        calibrated = risk.calibrate(score_matrix, region_labels, "equal_region")
                        q_symmetric = calibrated["q"]
                        q_lower = equal_source_quantile(lower_by_source, .95)
                        q_upper = equal_source_quantile(upper_by_source, .95)
                        offset = held_offsets[correction_name]
                        shifted_lo = held_bound["lo"] + offset[:, None]
                        shifted_hi = held_bound["hi"] + offset[:, None]
                        sym = risk.final_bounds(shifted_lo, shifted_hi, held_bound["scale"], q_symmetric)
                        asym_lo = np.clip(shifted_lo - q_lower * held_bound["scale"], 0.0, 1000.0)
                        asym_hi = np.clip(shifted_hi + q_upper * held_bound["scale"], 0.0, 1000.0)
                        for tail_name, lo, hi in (("symmetric_90", sym["lo"], sym["hi"]),
                                                  ("separate_95_each_tail", asym_lo, asym_hi)):
                            rows.append({"outer": outer, "arm": arm_name, "held_source": held,
                                         "risk_method": method_name, "location_method": correction_name,
                                         "tail_calibration": tail_name, "q_symmetric": q_symmetric,
                                         "q_lower": q_lower, "q_upper": q_upper,
                                         "eligible_days": len(truth_held),
                                         **interval_metrics(lo, hi, truth_held)})
                        q_rows.append({"outer": outer, "arm": arm_name, "held_source": held,
                                       "risk_method": method_name, "location_method": correction_name,
                                       "q_symmetric": q_symmetric, "q_lower": q_lower, "q_upper": q_upper,
                                       "calibration_regions": len(train_sources),
                                       "calibration_days": int(len(score_matrix))})

    expected_rows = len(SOURCES) * len(ARMS) * len(EXTERNAL) * len(RISK_METHODS) * len(correction_names) * 2
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} outer × arm × source × risk × location rows")
    OUT.mkdir()
    fold_path, q_path = OUT / "held_source_fold_metrics.csv", OUT / "calibration_q.csv"
    pd.DataFrame(rows).to_csv(fold_path, index=False)
    pd.DataFrame(q_rows).to_csv(q_path, index=False)
    fold = pd.DataFrame(rows)
    source_macro = fold.groupby(["arm", "risk_method", "held_source", "location_method", "tail_calibration"], sort=False)[
        ["coverage90", "width90", "interval_score90", "lower_miss", "upper_miss"]].mean().reset_index()
    source_path = OUT / "source_macro_metrics.csv"
    source_macro.to_csv(source_path, index=False)
    overall = source_macro.groupby(["arm", "risk_method", "location_method", "tail_calibration"], sort=False)[
        ["coverage90", "width90", "interval_score90", "lower_miss", "upper_miss"]].mean().reset_index()
    summary = {
        "status": "COMPLETE_CROSSFIT_ASYMMETRIC_TAIL_SOURCE_HELDOUT",
        "protocol": "docs/V50_CROSSFIT_ASYMMETRIC_RISK_PROTOCOL.md",
        "protocol_sha256": sha(ROOT / "docs/V50_CROSSFIT_ASYMMETRIC_RISK_PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)), "source_units": len(EXTERNAL),
        "prediction_packages_opened": len(opened), "external_label_packages_opened": len(label_cache),
        "target_prediction_packages_opened": 0, "target_or_unseen_scores_opened": 0,
        "point_or_risk_head_refits": 0, "crossfit_location_fits": len(SOURCES) * len(ARMS) * len(EXTERNAL) * len(EXTERNAL),
        "held_source_rows": len(rows), "risk_methods": list(RISK_METHODS), "point_arms": list(ARMS),
        "location_methods": list(correction_names),
        "calibration_rules": {"symmetric": "equal-region 90% nonconformity quantile",
                              "asymmetric": "separate equal-region 95% quantiles for lower and upper excess"},
        "source_macro_metrics": overall.to_dict(orient="records"),
        "frozen_hashes": {"freeze": sha(RUN / "freeze.json"),
                          "prediction_seal": sha(RUN / "prediction_seal.json"),
                          "calibration_seal": sha(RUN / "calibration_seal.json"),
                          **{str(path): sha(path) for path in map(Path, freeze["external_monthly"])}},
        "held_source_fold_metrics_sha256": sha(fold_path), "calibration_q_sha256": sha(q_path),
        "source_macro_metrics_sha256": sha(source_path),
        "interpretation_limit": "Ten previously used sources only; cross-fitted development comparison, no conditional coverage guarantee or China transfer evidence.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "source_macro_metrics": summary["source_macro_metrics"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
