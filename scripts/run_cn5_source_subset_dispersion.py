#!/usr/bin/env python3
"""Strict-CN5 LOCO evaluation of source-subset disagreement as an interval scale."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
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
PROTOCOL = ROOT / "docs/CN5_SOURCE_SUBSET_DISPERSION_PROTOCOL_20261102.md"
SCALE_FLOOR = 1.0


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_for(frame: pd.DataFrame, coarse: np.ndarray) -> np.ndarray:
    """Reproduce the country-equal hourly profile in multicountry_cn5.fit."""
    frame = frame.reset_index(drop=True)
    profiles: dict[str, list[np.ndarray]] = {}
    for (country, _region), rows in frame.groupby(["country", "region"], sort=False).groups.items():
        positions = np.asarray(list(rows), dtype=int)
        local = pd.DataFrame(coarse[positions], columns=model.GROUPS)
        local["hour"] = frame.iloc[positions].clock.dt.hour.to_numpy()
        profile = local.groupby("hour")[list(model.GROUPS)].mean().reindex(range(24)).fillna(0.)
        profile = profile.div(profile.mean().replace(0., np.nan)).fillna(1.).to_numpy()
        profiles.setdefault(country, []).append(profile)
    country_profiles = [np.mean(values, axis=0) for _, values in sorted(profiles.items())]
    if not country_profiles:
        raise ValueError("no source profiles")
    return np.mean(country_profiles, axis=0)


def calibration_part(prepared: dict, country: str) -> tuple[pd.DataFrame, np.ndarray]:
    raw, features = prepared[country]
    part = features.loc[features.year.eq(2023) & features.clock.dt.month.le(6)].copy()
    truth = response.truth_for(raw, part)
    return part.reset_index(drop=True), truth


def fit_subset(prepared: dict, countries: list[str]) -> tuple[dict, np.ndarray]:
    parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in countries]
    train = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([response.fuel_for(prepared[c][0], p)
                           for c, p in zip(countries, parts)])
    coarse = np.concatenate([model.coarse_values(prepared[c][0].iloc[p.raw_row.to_numpy()])
                             for c, p in zip(countries, parts)])
    if len(train) != len(fuel) or len(train) != len(coarse) or not len(train):
        raise ValueError("source training arrays are empty or misaligned")
    profile = profile_for(train, coarse)
    head = response.fit_fuelshare(train, fuel)
    return {"profile": profile, "fuelshare": head}, fuel


def predict(head: dict, frame: pd.DataFrame) -> np.ndarray:
    return response.predict_fuelshare(head["fuelshare"], head, frame)


def country_weights(lengths: dict[str, int]) -> np.ndarray:
    total = len(lengths)
    return np.concatenate([np.full(n, 1. / (total * n)) for n in lengths.values()])


def row_country_weights(countries: pd.Series) -> np.ndarray:
    counts = countries.value_counts().to_dict()
    weights = countries.map(lambda country: 1. / (len(counts) * counts[country])).to_numpy(float)
    return weights / weights.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for code in COUNTRIES:
        folder, manifest, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest, folder / f"{code}_2022_2024.json"])
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
    (out / "startup_hashes.json").write_text(json.dumps(
        {"input_sha256": input_hashes, "code_sha256": code_hashes}, indent=2) + "\n")

    prepared, audits = {}, {}
    for code in COUNTRIES:
        raw, features, audits[code] = nine.prepare_country(code)
        prepared[code] = raw, features
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")

    seals, calibration_records = [], {}
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy().reset_index(drop=True)
        query_pred: list[np.ndarray] = []
        query_members: list[str] = []
        pseudo_predictions: dict[str, list[np.ndarray]] = {c: [] for c in sources}
        pseudo_truth: dict[str, np.ndarray] = {}
        pseudo_frames: dict[str, pd.DataFrame] = {}
        for country in sources:
            pseudo_frames[country], pseudo_truth[country] = calibration_part(prepared, country)

        for excluded in combinations(sources, 2):
            train_countries = [c for c in sources if c not in excluded]
            fitted, _ = fit_subset(prepared, train_countries)
            member_name = "exclude_" + "_".join(excluded)
            query_pred.append(predict(fitted, query))
            query_members.append(member_name)
            for pseudo in excluded:
                pseudo_predictions[pseudo].append(predict(fitted, pseudo_frames[pseudo]))
        member_array = np.asarray(query_pred, dtype=float)
        center = member_array.mean(axis=0)
        scale = np.maximum(member_array.std(axis=0, ddof=1), SCALE_FLOOR)

        cal_rows = []
        for pseudo in sources:
            member_values = np.asarray(pseudo_predictions[pseudo], dtype=float)
            if member_values.shape[0] != len(sources) - 1:
                raise RuntimeError(f"wrong pseudo-target ensemble size for {target}/{pseudo}")
            pseudo_center = member_values.mean(axis=0)
            pseudo_scale = np.maximum(member_values.std(axis=0, ddof=1), SCALE_FLOOR)
            error = pseudo_truth[pseudo] - pseudo_center
            for i, (e, s) in enumerate(zip(error, pseudo_scale)):
                cal_rows.append({"pseudo_country": pseudo, "row": i, "error": float(e),
                                 "scale": float(s), "normalized_abs_error": float(abs(e) / s)})
        cal_frame = pd.DataFrame(cal_rows)
        lengths = cal_frame.groupby("pseudo_country", sort=False).size().to_dict()
        weights = row_country_weights(cal_frame.pseudo_country)
        pooled_radius = weighted_quantile(np.abs(cal_frame.error.to_numpy()), weights, .90)
        normalized_radius = weighted_quantile(
            cal_frame.normalized_abs_error.to_numpy(), weights, .90)
        calibration_records[target] = {
            "pooled_radius": pooled_radius,
            "normalized_radius": normalized_radius,
            "pseudo_country_hours": lengths,
        }

        # Full-source fuel-share model is the strong deployment reference.
        full_head, _ = fit_subset(prepared, sources)
        full_center = predict(full_head, query)
        full_cal_error, full_lengths = [], {}
        for pseudo in sources:
            part, truth = calibration_part(prepared, pseudo)
            full_cal_error.append(np.abs(truth - predict(full_head, part)))
            full_lengths[pseudo] = len(part)
        full_weights = country_weights(full_lengths)
        full_radius = weighted_quantile(np.concatenate(full_cal_error), full_weights, .90)
        anchor_center = 700. * query.thermal.to_numpy(float)
        anchor_cal_error, anchor_lengths = [], {}
        for pseudo in sources:
            part, truth = calibration_part(prepared, pseudo)
            anchor_cal_error.append(np.abs(truth - 700. * part.thermal.to_numpy(float)))
            anchor_lengths[pseudo] = len(part)
        anchor_radius = weighted_quantile(np.concatenate(anchor_cal_error),
                                          country_weights(anchor_lengths), .90)

        artifact = pd.DataFrame({
            "time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
            "target_mean": center,
            "target_scale": scale,
            "target_pooled_lo": np.clip(center - pooled_radius, 0., 1000.),
            "target_pooled_hi": np.clip(center + pooled_radius, 0., 1000.),
            "target_scaled_lo": np.clip(center - normalized_radius * scale, 0., 1000.),
            "target_scaled_hi": np.clip(center + normalized_radius * scale, 0., 1000.),
            "fuelshare_full": full_center,
            "fuelshare_full_lo": np.clip(full_center - full_radius, 0., 1000.),
            "fuelshare_full_hi": np.clip(full_center + full_radius, 0., 1000.),
            "anchor700": anchor_center,
            "anchor700_lo": np.clip(anchor_center - anchor_radius, 0., 1000.),
            "anchor700_hi": np.clip(anchor_center + anchor_radius, 0., 1000.),
        })
        members = pd.DataFrame(member_array.T, columns=query_members)
        predictions_path = out / f"{target.upper()}_predictions.csv.gz"
        members_path = out / f"{target.upper()}_member_predictions.csv.gz"
        artifact.to_csv(predictions_path, index=False, compression="gzip")
        members.to_csv(members_path, index=False, compression="gzip")
        cal_path = out / f"{target.upper()}_source_calibration.csv.gz"
        cal_frame.to_csv(cal_path, index=False, compression="gzip")
        seals.append({"target": target.upper(), "hours": len(query),
                      "members": len(query_members), "predictions": predictions_path.name,
                      "predictions_sha256": sha(predictions_path),
                      "members_file": members_path.name, "members_sha256": sha(members_path),
                      "source_calibration": cal_path.name, "calibration_sha256": sha(cal_path)})
        print(f"sealed {target.upper()}: {len(query)} hours, {len(query_members)} source subsets",
              flush=True)

    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    (out / "calibration_summary.json").write_text(
        json.dumps(calibration_records, indent=2) + "\n")
    if any(sha(Path(p)) != h for p, h in input_hashes.items()):
        raise RuntimeError("input changed during run")
    if any(sha(Path(p)) != h for p, h in code_hashes.items()):
        raise RuntimeError("code or protocol changed during run")
    if any(sha(out / row[key]) != row[digest] for row in seals for key, digest in (
            ("predictions", "predictions_sha256"),
            ("members_file", "members_sha256"),
            ("source_calibration", "calibration_sha256"))):
        raise RuntimeError("prediction/calibration seal mismatch")

    methods = {
        "subset_pooled": ("target_pooled_lo", "target_pooled_hi"),
        "subset_disagreement_scaled": ("target_scaled_lo", "target_scaled_hi"),
        "fuelshare_full": ("fuelshare_full_lo", "fuelshare_full_hi"),
        "anchor700": ("anchor700_lo", "anchor700_hi"),
    }
    score_rows = []
    for target in COUNTRIES:
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy().reset_index(drop=True)
        truth = response.truth_for(raw, query)
        saved = pd.read_csv(out / f"{target.upper()}_predictions.csv.gz")
        if len(saved) != len(query):
            raise RuntimeError(f"target row mismatch for {target}")
        for method, (lower_col, upper_col) in methods.items():
            center_col = {"subset_pooled": "target_mean",
                          "subset_disagreement_scaled": "target_mean",
                          "fuelshare_full": "fuelshare_full",
                          "anchor700": "anchor700"}[method]
            center = saved[center_col].to_numpy(float)
            lower = saved[lower_col].to_numpy(float)
            upper = saved[upper_col].to_numpy(float)
            error = truth - center
            width = upper - lower
            score = width + 20. * (np.maximum(lower-truth, 0.) + np.maximum(truth-upper, 0.))
            frame = pd.DataFrame({"truth": truth, "prediction": center,
                                  "lower": lower, "upper": upper,
                                  "width": width, "score": score})
            frame["day"] = query.clock.dt.strftime("%Y-%m-%d").to_numpy()
            centered = frame[["truth", "prediction"]] - frame.groupby("day")[
                ["truth", "prediction"]].transform("mean")
            score_rows.append({
                "target_country": target.upper(), "method": method, "hours": len(query),
                "mae": float(np.mean(np.abs(error))),
                "shape_mae": float(np.abs(centered.truth-centered.prediction).mean()),
                "coverage90": float(np.mean((truth >= lower) & (truth <= upper))),
                "width90": float(width.mean()), "winkler90": float(score.mean()),
            })
    country_metrics = pd.DataFrame(score_rows)
    country_metrics.to_csv(out / "country_metrics.csv", index=False)
    country_metrics.groupby("method").agg(
        macro_mae=("mae", "mean"), macro_shape_mae=("shape_mae", "mean"),
        macro_coverage90=("coverage90", "mean"), macro_width90=("width90", "mean"),
        macro_winkler90=("winkler90", "mean"), min_country_coverage90=("coverage90", "min"),
        max_country_mae=("mae", "max"),
    ).reset_index().to_csv(out / "macro_metrics.csv", index=False)
    (out / "status.json").write_text(json.dumps({
        "status": "DEVELOPMENT_STRICT_CN5_LOCO_SOURCE_SUBSET_SCALE",
        "countries": [c.upper() for c in COUNTRIES], "training_year": 2022,
        "source_calibration": "2023H1 pseudo-targets, two-country exclusion models",
        "target_year": 2024, "subset_size": 6, "target_budget": "strict CN5 t-2..t-7 + calendar",
        "target_labels_read_after_prediction_seal": True, "blind_confirmation": False,
        "label": "Energy-Charts fixed-factor generation-side proxy",
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"completed {len(country_metrics)} country-method scores", flush=True)


if __name__ == "__main__":
    main()
