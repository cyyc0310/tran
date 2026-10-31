"""Source-held-out monthly-mix level correction with feature support guard."""
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
OUT = ROOT / "results/disjoint-calibration-v50-mix-support-guard"
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


def monthly_rows(freeze: dict) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]]]:
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
                values[source].append((start, end, released, float(share @ EF)))
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
    daily_feature = {}
    for source in EXTERNAL:
        source_rows = records[source]
        feature = []
        for day in helper.CAL_DAYS:
            available = [row for row in source_rows if row[0] >= pd.Timestamp("2022-01-01", tz="UTC")
                         and row[0] < day and row[1] <= day and row[2] <= day]
            chosen = available[-3:][::-1]
            if len(chosen) != 3:
                raise ValueError("Every prediction day must have three legally released monthly contexts")
            feature.append(float(np.mean([row[3] for row in chosen])))
        daily_feature[source] = np.asarray(feature)

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
        label_cache[source] = (positions[keep], truth[keep], daily_feature[source][positions[keep]])
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

    rows, bias_rows = [], []
    def weighted_fit(design: np.ndarray, response: np.ndarray, weight: np.ndarray) -> np.ndarray:
        root_weight = np.sqrt(weight)
        return np.linalg.lstsq(design * root_weight[:, None], response * root_weight, rcond=None)[0]

    for outer in SOURCES:
        train_bias = {}
        for source in EXTERNAL:
            if source == EXTERNAL[0]:  # no-op; explicit loop below fills every fixed source
                pass
            raw, truth, _ = raw_cache[source][outer], label_cache[source][1], label_cache[source][2]
            train_bias[source] = (truth - raw.mean(axis=1)).mean(axis=1)
        for held in EXTERNAL:
            train_sources = tuple(source for source in EXTERNAL if source != held)
            global_offset = float(np.mean([train_bias[source].mean() for source in train_sources]))
            x_parts, season_parts, y_parts, w_parts = [], [], [], []
            for source in train_sources:
                positions = label_cache[source][0]
                x_parts.append(label_cache[source][2])
                day_of_year = helper.CAL_DAYS[positions].dayofyear.to_numpy(dtype=float)
                angle = 2 * np.pi * (day_of_year - 1) / 365.25
                season_parts.append(np.column_stack([np.sin(angle), np.cos(angle)]))
                y_parts.append(train_bias[source])
                w_parts.append(np.full(len(train_bias[source]), 1.0 / len(train_bias[source])))
            train_x = np.concatenate(x_parts)
            train_season = np.concatenate(season_parts)
            train_y, train_weight = np.concatenate(y_parts), np.concatenate(w_parts)
            held_x = label_cache[held][2]
            held_positions = label_cache[held][0]
            held_day_of_year = helper.CAL_DAYS[held_positions].dayofyear.to_numpy(dtype=float)
            held_angle = 2 * np.pi * (held_day_of_year - 1) / 365.25
            held_season = np.column_stack([np.sin(held_angle), np.cos(held_angle)])
            mix_beta = weighted_fit(np.column_stack([np.ones(len(train_x)), train_x]), train_y, train_weight)
            season_beta = weighted_fit(np.column_stack([np.ones(len(train_x)), train_season]), train_y, train_weight)
            joint_beta = weighted_fit(np.column_stack([np.ones(len(train_x)), train_x, train_season]), train_y, train_weight)
            local_offsets = np.column_stack([np.ones(len(held_x)), held_x]) @ mix_beta
            support_min, support_max = float(train_x.min()), float(train_x.max())
            support_mask = (held_x >= support_min) & (held_x <= support_max)
            guarded_offsets = np.where(support_mask, 0.75 * local_offsets, 0.0)
            season_offsets = np.column_stack([np.ones(len(held_x)), held_season]) @ season_beta
            joint_offsets = np.column_stack([np.ones(len(held_x)), held_x, held_season]) @ joint_beta
            held_bias = train_bias[held]
            bias_rows.append({
                "outer": outer, "held_source": held, "eligible_days": len(held_x),
                "mean_daily_legal_EF_proxy": float(held_x.mean()),
                "observed_source_level_bias": float(held_bias.mean()),
                "global_source_offset": global_offset,
                "daily_mix_offset_mean": float(local_offsets.mean()),
                "training_feature_support_min": support_min,
                "training_feature_support_max": support_max,
                "held_days_outside_feature_support_fraction": float((~support_mask).mean()),
                "no_correction_abs_source_bias": abs(float(held_bias.mean())),
                "global_abs_source_bias_after": abs(float(held_bias.mean() - global_offset)),
                "daily_mix_abs_source_bias_after": abs(float(np.mean(held_bias - local_offsets))),
                "season_only_abs_source_bias_after": abs(float(np.mean(held_bias - season_offsets))),
                "mix_season_abs_source_bias_after": abs(float(np.mean(held_bias - joint_offsets))),
            })
            raw, truth = raw_cache[held][outer], label_cache[held][1]
            offsets = {
                "no_correction_factor2_anchor": np.zeros(len(held_x)),
                "global_source_bias_shift": np.full(len(held_x), global_offset),
                "daily_asof_monthly_mix_shift": local_offsets,
                "monthly_mix_shrink_025": 0.25 * local_offsets,
                "monthly_mix_shrink_050": 0.50 * local_offsets,
                "monthly_mix_shrink_075": 0.75 * local_offsets,
                "monthly_mix_shrink_075_support_guard": guarded_offsets,
                "season_only_shift": season_offsets,
                "monthly_mix_plus_season_shift": joint_offsets,
            }
            for method, daily_offset in offsets.items():
                rows.append({"outer": outer, "held_source": held, "method": method,
                             "fraction_held_days_outside_training_feature_support": float((~support_mask).mean()),
                             "mean_predicted_offset": float(daily_offset.mean()),
                             **metrics(raw, truth, daily_offset)})

    if len(rows) != len(SOURCES) * len(EXTERNAL) * 9:
        raise ValueError("Expected all 630 outer × held-source × method rows")
    OUT.mkdir()
    fold_path, bias_path = OUT / "held_source_fold_metrics.csv", OUT / "held_source_bias_summary.csv"
    pd.DataFrame(rows).to_csv(fold_path, index=False)
    pd.DataFrame(bias_rows).to_csv(bias_path, index=False)
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
        ["observed_source_level_bias", "no_correction_abs_source_bias", "global_abs_source_bias_after",
         "daily_mix_abs_source_bias_after"]].mean().reset_index()
    summary = {
        "status": "COMPLETE_MONTHLY_MIX_SUPPORT_GUARD",
        "protocol": "docs/V50_MIX_SUPPORT_GUARD_PROTOCOL.md",
        "protocol_sha256": sha(ROOT / "docs/V50_MIX_SUPPORT_GUARD_PROTOCOL.md"),
        "script_sha256": sha(Path(__file__)), "source_units": len(EXTERNAL),
        "prediction_packages_opened": len(opened_paths), "external_label_packages_opened": len(label_cache),
        "target_prediction_packages_opened": 0, "target_or_unseen_scores_opened": 0,
        "point_or_risk_refits": 0, "source_heldout_wls_fits": len(SOURCES) * len(EXTERNAL),
        "source_heldout_rows": len(rows), "lag_days": 90,
        "season_basis": "annual sin/cos, period 365.25 days, query-date calendar day",
        "fixed_mix_offset_shrinkage": [0.75],
        "support_guard": "apply zero offset when held query-date feature is outside eligible training-source min/max",
        "full_mix_EF_definition": "mean fixed-EF intensity of three latest legally released monthly eight-fuel records at each query date",
        "daily_source_bias_mae": {
            "no_correction": float(bias.no_correction_abs_source_bias.mean()),
            "global_correction": float(bias.global_abs_source_bias_after.mean()),
            "daily_asof_mix_correction": float(bias.daily_mix_abs_source_bias_after.mean()),
            "mix_beats_global_source_count": int((bias.daily_mix_abs_source_bias_after
                                                   < bias.global_abs_source_bias_after).sum()),
        },
        "source_macro_metrics": overall.to_dict(orient="records"),
        "frozen_input_hashes": {"freeze": sha(RUN / "freeze.json"),
                                 "prediction_seal": sha(RUN / "prediction_seal.json"),
                                 "calibration_seal": sha(RUN / "calibration_seal.json"),
                                 **{str(path): sha(path) for path in map(Path, freeze["external_monthly"])}},
        "held_source_fold_metrics_sha256": sha(fold_path), "held_source_bias_sha256": sha(bias_path),
        "source_macro_metrics_sha256": sha(source_path),
        "interpretation_limit": "Ten previously used sources only; source-held-out descriptive evidence, not China transfer, novel method proof, or a coverage guarantee.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "daily_source_bias_mae": summary["daily_source_bias_mae"],
        "source_macro_metrics": summary["source_macro_metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
