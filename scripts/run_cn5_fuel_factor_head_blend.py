#!/usr/bin/env python3
"""Nested source-pseudotarget blend of CN5 fuel-share and factor heads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as response  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = nine.COUNTRIES
PROTOCOL = ROOT / "docs/CN5_FUEL_FACTOR_HEAD_BLEND_PROTOCOL_20261102.md"
ALPHAS = (0., .25, .5, .75, 1.)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truth_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    fuel = response.fuel_for(raw, part)
    return fuel @ model.FACTORS / fuel.sum(axis=1)


def fit_pair(train_countries: list[str], prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
    parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in train_countries]
    train = pd.concat(parts, ignore_index=True)
    fuels = np.concatenate([response.fuel_for(prepared[c][0], part)
                            for c, part in zip(train_countries, parts)])
    coarse = np.concatenate([model.coarse_values(
        prepared[c][0].iloc[part.raw_row.to_numpy()])
        for c, part in zip(train_countries, parts)])
    base = model.fit(train, fuels, coarse)
    fine_head = response.fit_fuelshare(train, fuels)
    return {"base": base, "fuelshare": fine_head}


def predict_pair(fitted: dict, raw: pd.DataFrame, part: pd.DataFrame) -> dict[str, np.ndarray]:
    base_points = model.predict(fitted["base"], part)
    return {
        "factor_history": base_points["factor_history"],
        "fuelshare_history": response.predict_fuelshare(fitted["fuelshare"],
                                                          fitted["base"], part),
        "ratio_recomposition": base_points["ratio_recomposition"],
    }


def local_day_shape_mae(part: pd.DataFrame, pred: np.ndarray, truth: np.ndarray) -> float:
    frame = pd.DataFrame({"pred": pred, "truth": truth,
                          "day": part.clock.dt.strftime("%Y-%m-%d").to_numpy()})
    centered = frame[["pred", "truth"]] - frame.groupby("day")[["pred", "truth"]].transform("mean")
    return float(np.abs(centered.pred - centered.truth).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for code in COUNTRIES:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_paths = sorted(set(input_paths))
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(nine.__file__).resolve(),
                  Path(nine.six_group.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), Path(response.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(p.resolve()): sha(p) for p in input_paths}
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (out / "startup_hashes.json").write_text(
        json.dumps({"inputs": input_hashes, "code": code_hashes}, indent=2) + "\n", encoding="utf-8")

    prepared, audits = {}, {}
    for code in COUNTRIES:
        raw, features, audits[code] = nine.prepare_country(code)
        prepared[code] = (raw, features)
    (out / "input_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Each pairwise fit excludes the outer target and inner pseudo-target. The same
    # fit yields independent H1 selection and H2 calibration predictions.
    pair_predictions: dict[tuple[str, str, str], dict[str, pd.DataFrame]] = {}
    pair_rows = []
    for i, first in enumerate(COUNTRIES):
        for second in COUNTRIES[i + 1:]:
            train_countries = [c for c in COUNTRIES if c not in (first, second)]
            fitted = fit_pair(train_countries, prepared)
            for heldout in (first, second):
                raw, features = prepared[heldout]
                yearly = {}
                for period, month_mask in (("h1", features.clock.dt.month.le(6)),
                                           ("h2", features.clock.dt.month.ge(7))):
                    part = features.loc[features.year.eq(2023) & month_mask].copy()
                    points = predict_pair(fitted, raw, part)
                    stored = pd.DataFrame({"time": raw.iloc[part.raw_row.to_numpy()].time.astype(str).to_numpy(),
                                           **points})
                    yearly[period] = stored
                    truth = truth_for(raw, part)
                    pair_rows.append({"excluded_outer": first if heldout == second else second,
                                      "pseudo_target": heldout, "period": period,
                                      "n_hours": len(part),
                                      "factor_mae": float(np.abs(truth-points["factor_history"]).mean()),
                                      "fuelshare_mae": float(np.abs(truth-points["fuelshare_history"]).mean()),
                                      "ratio_mae": float(np.abs(truth-points["ratio_recomposition"]).mean())})
                pair_predictions[(*tuple(sorted((first, second))), heldout)] = yearly
            print(f"crossfit pair {first.upper()}×{second.upper()} complete", flush=True)

    # Hash the complete nested source-pseudotarget evidence before evaluating outer 2024 targets.
    pair_dir = out / "source_pseudotarget_predictions"
    pair_dir.mkdir()
    pair_seals = []
    for (first, second, heldout), years in sorted(pair_predictions.items()):
        for period, frame in years.items():
            path = pair_dir / f"exclude_{first}_{second}_{heldout}_{period}.csv.gz"
            frame.to_csv(path, index=False, compression="gzip")
            pair_seals.append({"path": str(path.relative_to(out)), "sha256": sha(path),
                               "excluded": [first, second], "pseudo_target": heldout,
                               "period": period, "rows": len(frame)})
    (out / "source_pseudotarget_seal.json").write_text(
        json.dumps(pair_seals, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(pair_rows).to_csv(out / "source_pseudotarget_metrics.csv", index=False)

    target_predictions, target_meta = {}, []
    for target in COUNTRIES:
        pseudo_scores = []
        cal_residuals = {"factor_history": [], "fuelshare_history": [],
                         "ratio_recomposition": []}
        cal_weights = {name: [] for name in cal_residuals}
        for pseudo in COUNTRIES:
            if pseudo == target:
                continue
            key = tuple(sorted((target, pseudo)))
            source_pred = pair_predictions[(*key, pseudo)]
            source_raw, source_features = prepared[pseudo]

            h1 = source_features.loc[source_features.year.eq(2023) &
                                     source_features.clock.dt.month.le(6)].copy()
            h1_truth = truth_for(source_raw, h1)
            h1_pred = source_pred["h1"].drop(columns="time")
            for alpha in ALPHAS:
                mixed = (1-alpha)*h1_pred.factor_history.to_numpy() + alpha*h1_pred.fuelshare_history.to_numpy()
                pseudo_scores.append({"pseudo_target": pseudo, "alpha": alpha,
                                      "mae": float(np.abs(h1_truth-mixed).mean())})

            h2 = source_features.loc[source_features.year.eq(2023) &
                                     source_features.clock.dt.month.ge(7)].copy()
            h2_truth = truth_for(source_raw, h2)
            h2_pred = source_pred["h2"].drop(columns="time")
            for name, values in (("factor_history", h2_pred.factor_history.to_numpy()),
                                 ("fuelshare_history", h2_pred.fuelshare_history.to_numpy()),
                                 ("ratio_recomposition", h2_pred.ratio_recomposition.to_numpy())):
                cal_residuals[name].extend(np.abs(h2_truth-values).tolist())
                cal_weights[name].extend(np.full(len(h2), 1/(len(COUNTRIES)-1)/len(h2)).tolist())
        score_table = pd.DataFrame(pseudo_scores).groupby("alpha", as_index=False).agg(
            macro_pseudotarget_mae=("mae", "mean"))
        alpha = float(score_table.sort_values(["macro_pseudotarget_mae", "alpha"]).iloc[0].alpha)
        # The source pseudo-target calibration block is H2 and is independent of H1 selection.
        radii = {}
        for name in ("factor_history", "fuelshare_history", "ratio_recomposition"):
            radii[name] = float(weighted_quantile(np.asarray(cal_residuals[name]),
                                                  np.asarray(cal_weights[name]), .9))
        # Recompute the selected blend H2 radius from stored per-country predictions.
        blend_abs, blend_w = [], []
        for pseudo in COUNTRIES:
            if pseudo == target:
                continue
            key = tuple(sorted((target, pseudo)))
            h2_pred = pair_predictions[(*key, pseudo)]["h2"].drop(columns="time")
            part = prepared[pseudo][1].loc[prepared[pseudo][1].year.eq(2023) &
                                           prepared[pseudo][1].clock.dt.month.ge(7)]
            truth = truth_for(prepared[pseudo][0], part)
            mix = ((1-alpha)*h2_pred.factor_history.to_numpy() +
                   alpha*h2_pred.fuelshare_history.to_numpy())
            blend_abs.extend(np.abs(truth-mix).tolist())
            blend_w.extend(np.full(len(part), 1/(len(COUNTRIES)-1)/len(part)).tolist())
        radii["blend"] = float(weighted_quantile(np.asarray(blend_abs), np.asarray(blend_w), .9))

        sources = [c for c in COUNTRIES if c != target]
        fitted = fit_pair(sources, prepared)
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        point = predict_pair(fitted, raw, query)
        point["blend"] = ((1-alpha)*point["factor_history"] +
                          alpha*point["fuelshare_history"])
        target_predictions[target] = (query, point, alpha, radii)
        path = out / f"{target.upper()}_outer_2024_predictions.csv.gz"
        pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                      **point}).to_csv(path, index=False, compression="gzip")
        radius_path = out / f"{target.upper()}_selection_calibration.json"
        radius_path.write_text(json.dumps({"selected_alpha": alpha,
            "alpha_selection_h1": score_table.to_dict("records"), "radii_h2": radii,
            "pseudo_targets": [c for c in COUNTRIES if c != target]}, indent=2) + "\n", encoding="utf-8")
        target_meta.append({"target": target, "prediction_file": path.name,
                            "prediction_sha256": sha(path), "calibration_file": radius_path.name,
                            "calibration_sha256": sha(radius_path), "hours": len(query)})
        print(f"sealed target {target.upper()}: alpha={alpha:.2f}, {len(query)} hours", flush=True)

    (out / "outer_prediction_seal.json").write_text(json.dumps(target_meta, indent=2) + "\n", encoding="utf-8")
    if any(sha(Path(path)) != value for path, value in input_hashes.items()):
        raise RuntimeError("input changed during run")
    if any(sha(Path(path)) != value for path, value in code_hashes.items()):
        raise RuntimeError("code/protocol changed during run")
    if any(sha(out / row["path"]) != row["sha256"] for row in pair_seals):
        raise RuntimeError("source pseudo-target prediction hash mismatch")
    if any(sha(out / row["prediction_file"]) != row["prediction_sha256"] or
           sha(out / row["calibration_file"]) != row["calibration_sha256"] for row in target_meta):
        raise RuntimeError("outer prediction/calibration hash mismatch")

    metrics = []
    for target in COUNTRIES:
        raw, _ = prepared[target]
        query, point, alpha, radii = target_predictions[target]
        truth = truth_for(raw, query)
        for name in ("ratio_recomposition", "factor_history", "fuelshare_history", "blend"):
            pred = point[name]
            radius = radii[name]
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                local_part = query.loc[mask]
                error = truth[mask]-pred[mask]
                metrics.append({"target_country": target.upper(), "method": name,
                    "blend_alpha": alpha if name == "blend" else np.nan,
                    "month": int(month), "hours": int(mask.sum()),
                    "mae": float(np.abs(error).mean()),
                    "shape_mae": local_day_shape_mae(local_part, pred[mask], truth[mask]),
                    "coverage90": float((np.abs(error) <= radius).mean()),
                    "width90": float(2*radius),
                    "winkler90": float(np.mean(2*radius + 20*np.maximum(np.abs(error)-radius, 0)))})
    monthly = pd.DataFrame(metrics)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    detail = monthly.groupby(["target_country", "method"], as_index=False).agg(
        mae=("mae", "mean"), shape_mae=("shape_mae", "mean"), coverage90=("coverage90", "mean"),
        width90=("width90", "mean"), winkler90=("winkler90", "mean"),
        blend_alpha=("blend_alpha", "first"))
    detail.to_csv(out / "country_metrics.csv", index=False)
    detail.groupby("method", as_index=False).agg(
        macro_mae=("mae", "mean"), macro_shape_mae=("shape_mae", "mean"),
        macro_coverage90=("coverage90", "mean"), macro_width90=("width90", "mean"),
        macro_winkler90=("winkler90", "mean"), worst_country_mae=("mae", "max"),
        min_country_coverage90=("coverage90", "min")).to_csv(out / "macro_metrics.csv", index=False)
    (out / "status.json").write_text(json.dumps({
        "status": "COMPLETED_DEVELOPMENT_NESTED_SOURCE_PSEUDOTARGET_BLEND",
        "countries": [c.upper() for c in COUNTRIES], "outer_year": 2024,
        "fit_year": 2022, "weight_selection": "nested pseudo-target 2023H1; equal-country MAE",
        "interval_calibration": "double-excluded pseudo-target 2023H2; equal-country weighted q90",
        "blend_alpha_definition": "(1-alpha)*factor_history + alpha*fuelshare_history",
        "target_budget": "lagged t-2..t-7 five-group monthly shares plus calendar",
        "label": "fixed-factor generation proxy; not measured CIF",
        "targets_previously_viewed": True, "target_labels_used_before_outer_seal": False,
        "target_generation": "historical monthly totals reconstructed from revised hourly series",
        "selected_alpha_by_country": {c: target_predictions[c][2] for c in COUNTRIES},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("completed nested fuelshare/factor blend experiment", flush=True)


if __name__ == "__main__":
    main()
