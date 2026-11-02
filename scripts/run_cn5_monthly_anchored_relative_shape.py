#!/usr/bin/env python3
"""Nested strict-CN5 LOCO test of monthly anchored relative shape transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402
from run_cn5_fuelshare_component_ablation import (  # noqa: E402
    fit_head as fit_fuelshare_head,
    predict_points as predict_with_fuelshare,
)

PROTOCOL = ROOT / "docs/CN5_MONTHLY_ANCHORED_RELATIVE_SHAPE_PROTOCOL_20261102.md"
METHODS = (
    "anchor700", "ratio_recomposition", "daily_anchor_shape",
    "monthly_anchor_relshape", "factor_history", "direct_history",
    "fuelshare_history",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fuel_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)


def predict_all(base: dict, head: dict, public: pd.DataFrame) -> tuple[dict, dict]:
    points, compositions = predict_with_fuelshare(base, head, public)
    anchor = points["anchor700"]
    ratio = points["ratio_recomposition"]
    day_key = public.clock.dt.strftime("%Y-%m-%d")
    points["daily_anchor_shape"] = anchor + ratio - pd.Series(ratio).groupby(
        day_key.to_numpy()).transform("mean").to_numpy()

    monthly = np.empty(len(public), dtype=float)
    diagnostics = []
    grouped = public.groupby(["country", "region", "period"], sort=False).indices
    for (country, region, period), indices in grouped.items():
        part = public.iloc[indices]
        expected_clock = pd.date_range(period.start_time, periods=24, freq="h")
        expected = pd.DataFrame({"clock": expected_clock})
        for group_name in model.GROUPS:
            expected[group_name] = float(part[group_name].iloc[0])
        expected_q = expected[list(model.GROUPS)].to_numpy(float) * base["profile"][
            expected.clock.dt.hour.to_numpy()]
        expected_shape = base["ratio_ef"] * expected_q[:, 0] / expected_q.sum(axis=1)
        denominator = float(np.mean(expected_shape))
        month_anchor = 700. * float(part.thermal.iloc[0])
        if (not np.isfinite(expected_shape).all() or (expected_shape < -1e-10).any()
                or denominator <= 0. or not np.isfinite(denominator)):
            values = np.full(len(indices), month_anchor, dtype=float)
            status = "anchor_fallback"
        else:
            values = month_anchor * ratio[indices] / denominator
            status = "normalized"
        monthly[indices] = values
        expected_candidate_mean = (month_anchor * float(np.mean(expected_shape)) / denominator
                                   if status == "normalized" else month_anchor)
        if (not np.isfinite(values).all() or (values < -1e-10).any()
                or abs(expected_candidate_mean - month_anchor) > 1e-8):
            raise ValueError(f"monthly anchored relative-shape invariant failed for {(country, region, period)}")
        diagnostics.append({
            "country": country, "region": region, "period": str(period),
            "rows": len(indices), "expected_hours": 24,
            "anchor_month_mean": month_anchor,
            "expected_recomposition_mean": denominator,
            "expected_candidate_24h_mean": expected_candidate_mean,
            "expected_candidate_mean_error": expected_candidate_mean - month_anchor,
            "observed_prediction_mean": float(np.mean(values)),
            "observed_support_mean_error": float(np.mean(values) - month_anchor),
            "normalization_status": status,
        })
    points["monthly_anchor_relshape"] = monthly
    return {name: points[name] for name in METHODS}, {"composition": compositions,
                                                       "monthly_diagnostics": diagnostics}


def metric_row(country: str, method: str, query: pd.DataFrame,
               truth: np.ndarray, prediction: np.ndarray, radius: float) -> dict:
    error = truth - prediction
    daily = pd.DataFrame({"truth": truth, "prediction": prediction,
                          "day": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
    centered = daily[["truth", "prediction"]] - daily.groupby("day")[
        ["truth", "prediction"]].transform("mean")
    return {
        "target_country": country.upper(), "method": method, "hours": int(len(query)),
        "mae": float(np.mean(np.abs(error))),
        "shape_mae": float(np.mean(np.abs(centered.truth - centered.prediction))),
        "coverage90": float(np.mean(np.abs(error) <= radius)),
        "width90": float(2. * radius),
        "winkler90": float(np.mean(2. * radius + 20. * np.maximum(np.abs(error) - radius, 0.))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for country in nine.COUNTRIES:
        folder, manifest, _ = nine.six_group.data_locations(country)
        input_paths.extend([manifest, folder / f"{country}_2022_2024.json"])
        if country in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_hashes = {str(path.resolve()): sha256(path) for path in sorted(set(input_paths))}
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(nine.__file__).resolve(),
                  Path(nine.six_group.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), Path(weighted_quantile.__code__.co_filename).resolve(),
                  Path(__import__("run_cn5_fuelshare_component_ablation").__file__).resolve()]
    code_hashes = {str(path.resolve()): sha256(path) for path in sorted(set(code_paths))}

    prepared, audits = {}, {}
    for country in nine.COUNTRIES:
        raw, features, audits[country] = nine.prepare_country(country)
        prepared[country] = raw, features
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")

    fold_data, seals = {}, []
    monthly_diagnostics = []
    for target in nine.COUNTRIES:
        sources = [country for country in nine.COUNTRIES if country != target]
        train_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([fuel_for(prepared[c][0], part)
                                     for c, part in zip(sources, train_parts)])
        train_coarse = np.concatenate([
            model.coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
            for c, part in zip(sources, train_parts)
        ])
        if not len(train) or len(train) != len(train_fuel) or len(train) != len(train_coarse):
            raise ValueError(f"source training support mismatch for {target}")
        base = model.fit(train, train_fuel, train_coarse)
        head = fit_fuelshare_head(train, train_fuel)

        calibration_parts = [prepared[c][1].loc[
            prepared[c][1].year.eq(2023) & prepared[c][1].clock.dt.month.le(6)]
            for c in sources]
        calibration = pd.concat(calibration_parts, ignore_index=True)
        calibration_fuel = np.concatenate([fuel_for(prepared[c][0], part)
                                           for c, part in zip(sources, calibration_parts)])
        cal_truth = calibration_fuel @ model.FACTORS / calibration_fuel.sum(axis=1)
        cal_weights = model.weights(calibration)
        cal_points, cal_aux = predict_all(base, head, calibration)
        radii = {name: float(weighted_quantile(np.abs(cal_truth - cal_points[name]),
                                               cal_weights, .9)) for name in METHODS}

        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        points, aux = predict_all(base, head, query)
        monthly_diagnostics.extend({"target_country": target.upper(), "split": "target", **item}
                                   for item in aux["monthly_diagnostics"])
        monthly_diagnostics.extend({"target_country": target.upper(), "split": "source_calibration",
                                   **item} for item in cal_aux["monthly_diagnostics"])
        prediction_path = out / f"{target.upper()}_predictions.csv.gz"
        pred = pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                             "period": query.period.astype(str).to_numpy(),
                             **{method: points[method] for method in METHODS}})
        pred.to_csv(prediction_path, index=False, compression="gzip")
        radii_path = out / f"{target.upper()}_source_radii.json"
        radii_path.write_text(json.dumps(radii, indent=2) + "\n")
        month_audit_path = out / f"{target.upper()}_monthly_anchor_audit.json"
        month_audit_path.write_text(json.dumps(aux["monthly_diagnostics"], indent=2) + "\n")
        seals.append({"target": target.upper(), "sources": [x.upper() for x in sources],
                      "prediction_file": prediction_path.name,
                      "prediction_sha256": sha256(prediction_path), "rows": len(query),
                      "radii_file": radii_path.name, "radii_sha256": sha256(radii_path),
                      "monthly_audit_file": month_audit_path.name,
                      "monthly_audit_sha256": sha256(month_audit_path),
                      "target_labels_opened_before_seal": False})
        fold_data[target] = {"raw": raw, "query": query, "radii": radii}
        print(f"sealed {target.upper()}: {len(query)} target hours", flush=True)

    seal_path = out / "prediction_seal.json"
    seal_path.write_text(json.dumps(seals, indent=2) + "\n")
    (out / "startup_hashes.json").write_text(json.dumps(
        {"input_sha256": input_hashes, "code_sha256": code_hashes}, indent=2) + "\n")
    if any(sha256(Path(path)) != digest for path, digest in {**input_hashes, **code_hashes}.items()):
        raise RuntimeError("input, frozen protocol, or code changed during experiment")
    for entry in seals:
        for file_key, hash_key in (("prediction_file", "prediction_sha256"),
                                   ("radii_file", "radii_sha256"),
                                   ("monthly_audit_file", "monthly_audit_sha256")):
            if sha256(out / entry[file_key]) != entry[hash_key]:
                raise RuntimeError(f"seal mismatch: {entry[file_key]}")

    rows, month_rows = [], []
    for target in nine.COUNTRIES:
        raw, query, radii = (fold_data[target][k] for k in ("raw", "query", "radii"))
        truth_fuel = fuel_for(raw, query)
        truth = truth_fuel @ model.FACTORS / truth_fuel.sum(axis=1)
        stored = pd.read_csv(out / f"{target.upper()}_predictions.csv.gz")
        for method in METHODS:
            prediction = stored[method].to_numpy(float)
            rows.append(metric_row(target, method, query, truth, prediction, radii[method]))
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                month_rows.append({**metric_row(target, method, query.loc[mask], truth[mask],
                                                prediction[mask], radii[method]),
                                   "month": int(month)})

    country_metrics = pd.DataFrame(rows)
    country_metrics.to_csv(out / "country_metrics.csv", index=False)
    country_metrics.groupby("method").mean(numeric_only=True).reset_index().to_csv(
        out / "macro_metrics.csv", index=False)
    pd.DataFrame(month_rows).to_csv(out / "monthly_metrics.csv", index=False)
    pd.DataFrame(monthly_diagnostics).to_csv(out / "monthly_anchor_diagnostics.csv", index=False)
    status = {
        "status": "DEVELOPMENT_STRICT_CN5_NESTED_LOCO_MONTHLY_ANCHORED_SHAPE",
        "outer_targets": [country.upper() for country in nine.COUNTRIES],
        "target_year": 2024, "training_year": 2022, "source_calibration": "2023H1",
        "target_budget": "strict CN5 t-2 through t-7 monthly shares plus calendar",
        "target_label": "common fixed-factor generation proxy, not measured CIF",
        "target_labels_opened_after_all_fold_predictions_sealed": True,
        "new_candidate_formula": "anchor700 * ratio_recomposition_hour / full_expected_24h_mean(ratio_recomposition for same target-month shares)",
        "monthly_mean_invariant": "full expected 24-hour mean equals anchor700; realized target rows can differ when target-hour support is incomplete",
        "country_level_confirmation": False,
        "methods": list(METHODS),
    }
    (out / "run_manifest.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    macro = pd.read_csv(out / "macro_metrics.csv")
    print(macro.round(4).to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
