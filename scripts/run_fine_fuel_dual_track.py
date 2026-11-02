#!/usr/bin/env python3
"""Paired US/GB fine-fuel profile transfer on generation and consumption labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_cross_country as common
import run_official_intensity_transfer as official
import run_cross_country_fine_fuel_bidirectional as fine
from transcif.research import cross_country as cc

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/FINE_FUEL_DUAL_TRACK_PROTOCOL_20261101.md"
MONTHS = ("2024-02", "2024-04", "2024-06")
TRACKS = ("generation_proxy", "consumption_official")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fuel_vector(raw: dict[str, pd.DataFrame], frame: pd.DataFrame) -> np.ndarray:
    return np.concatenate([
        raw[region].iloc[part.raw_row.to_numpy()][list(cc.FUELS)].to_numpy(float)
        for region, part in frame.groupby("region", sort=False)
    ])


def source_response_vector(frame: pd.DataFrame, labels: dict[str, pd.Series]) -> np.ndarray:
    values = []
    for region, part in frame.groupby("region", sort=False):
        values.append(official.aligned_labels(part, labels[region]))
    return np.concatenate(values)


def points(gen_models: dict, response_models: dict, profile: np.ndarray,
           frame: pd.DataFrame, track: str) -> dict[str, np.ndarray]:
    generation = cc.predict_models(gen_models, frame)
    fine_static = np.clip(fine.static_fine_prediction(frame), 0., 1000.)
    fine_shape = np.clip(fine.fine_profile_prediction(profile, frame), 0., 1000.)
    if track == "generation_proxy":
        output = {"anchor700": generation["anchor700"],
                  "source_thermal_ef": generation["source_thermal_ef"],
                  "calendar_ridge": generation["calendar_ridge"],
                  "coarse_ridge": generation["coarse_ridge"],
                  "coarse_boosting": generation["coarse_boosting"],
                  "M5_strong_recomposition": generation["coarse_recomposition"]}
    else:
        consumer = official.point_predictions(response_models, gen_models, frame)
        output = {"source_mean": consumer["source_mean"],
                  "calendar_ridge": consumer["calendar_ridge"],
                  "coarse_ridge": consumer["coarse_ridge"],
                  "coarse_boosting": consumer["coarse_boosting"],
                  "M5_strong_recomposition": consumer["fixed_ef_recomposition"]}
    output["MplusF_static_EF"] = fine_static
    output["MplusF_fuel_profile"] = fine_shape
    return output


def calibrate(residuals: dict[str, list[np.ndarray]]) -> dict[str, tuple[float, float]]:
    result = {}
    for method, regions in residuals.items():
        bias = float(np.mean([value.mean() for value in regions]))
        sigma = max(float(np.sqrt(np.mean([np.mean((value - bias) ** 2)
                                          for value in regions]))), 1.)
        result[method] = (bias, sigma)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    capacity = official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    raw, prepared_base, _, audit = official.prepare(ROOT, capacity)
    prepared = {}
    for region, frame in raw.items():
        base = prepared_base[region].copy().reset_index(drop=True)
        # Keep the official transfer features (including capacity/calendar) and add the
        # fine-fuel lagged shares from the paired generation-profile experiment.
        fine_full = fine.fine_monthly_features(frame)
        extra = fine_full.iloc[base.raw_row.to_numpy()][
            [f"fine_{fuel}" for fuel in cc.FUELS]].reset_index(drop=True)
        prepared[region] = base.join(extra)

    source_regions = {"US": list(official.US_REGIONS), "GB": ["GB"]}
    all_regions = list(raw)
    supported_columns = [*cc.COARSE, *(f"fine_{fuel}" for fuel in cc.FUELS)]
    supported = {}
    for region, frame in prepared.items():
        vals = frame[supported_columns].to_numpy(float)
        supported[region] = np.isfinite(vals).all(axis=1)

    source_consumption = {r: official.source_labels(r, years=(2022, 2023)) for r in all_regions}
    pending = []
    seals = []
    calibration_log = {}
    for source_country, target_country in (("US", "GB"), ("GB", "US")):
        sources, targets = source_regions[source_country], source_regions[target_country]
        train_parts, fuel_parts = [], []
        for region in sources:
            frame = prepared[region]
            mask = (pd.to_datetime(frame.clock).dt.year.eq(2022).to_numpy()
                    & supported[region])
            part = frame.loc[mask].copy()
            train_parts.append(part)
            fuel_parts.append(raw[region].iloc[part.raw_row.to_numpy()][list(cc.FUELS)].to_numpy(float))
        train = pd.concat(train_parts, ignore_index=True)
        fuel_y = np.concatenate(fuel_parts)
        gen_models = cc.fit_models(train, fuel_y)
        profile_raw = pd.concat([raw[r].assign(region=r) for r in sources], ignore_index=True)
        profile_mask_parts = []
        for region in sources:
            raw_support = np.zeros(len(raw[region]), dtype=bool)
            raw_support[prepared[region].raw_row.to_numpy()] = supported[region]
            profile_mask_parts.append(pd.to_datetime(raw[region].clock).dt.year.eq(2022).to_numpy()
                                      & raw_support)
        profile, profile_support = fine.source_fine_profiles(
            profile_raw, np.concatenate(profile_mask_parts))
        response_y = source_response_vector(train, source_consumption)
        response_models = official.train_official(train, response_y)

        calibration_by_track = {}
        for track in TRACKS:
            residuals: dict[str, list[np.ndarray]] = {}
            for region in sources:
                frame = prepared[region]
                year23 = (pd.to_datetime(frame.clock).dt.year.eq(2023).to_numpy()
                          & supported[region])
                cal = frame.loc[year23].copy()
                if not len(cal):
                    continue
                pred = points(gen_models, response_models, profile, cal, track)
                if track == "generation_proxy":
                    truth = cc.labels(raw[region])[cal.raw_row.to_numpy()]
                else:
                    truth = official.aligned_labels(cal, source_consumption[region])
                for method, estimate in pred.items():
                    good = np.isfinite(truth) & np.isfinite(estimate)
                    if good.any():
                        residuals.setdefault(method, []).append(truth[good] - estimate[good])
            calibration_by_track[track] = calibrate(residuals)
        calibration_log[f"{source_country}_to_{target_country}"] = {
            "train_hours": int(len(train)), "source_regions": sources,
            "source_fuel_profile_support": profile_support,
            "calibration": {track: {method: {"bias": b, "sigma": s}
                                      for method, (b, s) in values.items()}
                            for track, values in calibration_by_track.items()}}

        for region in targets:
            frame = prepared[region]
            date = pd.to_datetime(frame.clock)
            use = (date.dt.year.eq(2024).to_numpy()
                   & date.dt.strftime("%Y-%m").isin(MONTHS).to_numpy()
                   & supported[region])
            query = frame.loc[use].copy().reset_index(drop=True)
            if query.empty:
                continue
            for track in TRACKS:
                pred = points(gen_models, response_models, profile, query, track)
                path = out / f"{source_country}_to_{region}_{track}_predictions.csv.gz"
                pd.DataFrame({"time_utc": pd.to_datetime(query.time, utc=True).astype(str),
                              "observed_month": query.observed_month.astype(str),
                              **pred}).to_csv(path, index=False, compression="gzip")
                seal = {"source": source_country, "target": region, "track": track,
                        "path": path.name, "sha256": sha(path), "rows": len(query)}
                seals.append(seal)
                pending.append((source_country, region, track, query, path,
                                calibration_by_track[track]))

    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    (out / "source_calibration.json").write_text(json.dumps(calibration_log, indent=2) + "\n")

    # Score target responses only after every prediction file has been sealed.
    score_rows = []
    target_hashes = {}
    for source_country, region, track, query, path, cal in pending:
        if track == "generation_proxy":
            truth_all = cc.labels(raw[region])
            truth = truth_all[query.raw_row.to_numpy()]
            label_path = ROOT / ("data/neso-gb/df_fuel_ckan.csv" if region == "GB"
                                 else f"results/research/data/{region}_hourly.csv")
        else:
            label_series = official.source_labels(region, years=(2024,))
            truth = official.aligned_labels(query, label_series)
            label_path = official.TARGET_LABEL_ROOT / (
                "GB_consumption_2024_halfhour.csv.gz" if region == "GB"
                else f"{region}_consumption_2024.csv.gz")
        target_hashes[f"{track}:{region}"] = sha(label_path)
        predictions = pd.read_csv(path)
        local_month = pd.to_datetime(query.clock).dt.strftime("%Y-%m")
        for month in MONTHS:
            take = local_month.eq(month).to_numpy() & np.isfinite(truth)
            if not take.any():
                continue
            for method in [c for c in predictions.columns
                           if c not in {"time_utc", "observed_month"}]:
                bias, sigma = cal[method]
                scored = cc.metrics(query.loc[take], truth[take],
                                    predictions.loc[take, method].to_numpy(float), bias, sigma)
                score_rows.append({"source": source_country, "target": region,
                                   "track": track, "month": month,
                                   "method": method, **scored})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(out / "region_month_metrics.csv", index=False)
    macro = (scores.groupby(["source", "target", "track", "method"], as_index=False)
             .agg(region_months=("month", "size"), mae=("mae", "mean"),
                  shape_mae=("shape_mae", "mean"), crps=("crps_gaussian", "mean"),
                  coverage90=("coverage90", "mean"), width90=("width90", "mean"),
                  interval_score90=("interval_score90", "mean")))
    macro.to_csv(out / "macro_metrics.csv", index=False)
    (out / "target_label_hashes.json").write_text(json.dumps(target_hashes, indent=2) + "\n")

    input_paths = [PROTOCOL, Path(__file__), Path(fine.__file__), Path(official.__file__),
                   Path(cc.__file__), official.CAPACITY_TABLE_PATH,
                   ROOT / "data/neso-gb/df_fuel_ckan.csv"]
    input_paths += [ROOT / f"results/research/data/{r}_hourly.csv" for r in official.US_REGIONS]
    input_paths += [official.SOURCE_LABEL_ROOT / f"{r}_consumption_2022_2023.csv.gz"
                    for r in official.US_REGIONS]
    input_paths += [official.SOURCE_LABEL_ROOT / "GB_consumption_2022_2023_halfhour.csv.gz",
                    official.TARGET_LABEL_ROOT / "GB_consumption_2024_halfhour.csv.gz"]
    input_paths += [official.TARGET_LABEL_ROOT / f"{r}_consumption_2024.csv.gz"
                    for r in official.US_REGIONS]
    (out / "input_code_hashes.json").write_text(json.dumps(
        {str(p): sha(p) for p in dict.fromkeys(input_paths)}, indent=2) + "\n")
    summary = {"status": "DEVELOPMENT_FINE_FUEL_DUAL_TRACK",
               "source_train_year": 2022, "source_calibration_year": 2023,
               "target_year": 2024, "months": list(MONTHS),
               "tracks": list(TRACKS), "prediction_seals": seals,
               "input_audit": audit, "target_label_hashes": target_hashes,
               "target_is_unseen": False,
               "MplusF_is_china_budget": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
