"""Pseudo-target leave-one-source-out selection for nine-country transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import run_energy_charts_nine_country_cn5_loco as nine
from transcif.research import multicountry_cn5 as model
from transcif.research.monthly_adaptation import weighted_quantile

ROOT = nine.ROOT
PROTOCOL = ROOT / "docs/ENERGY_CHARTS_PSEUDOTARGET_SOURCE_DROP_CN5_PROTOCOL_20261101.md"
SOURCES = nine.COUNTRIES


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for code in SOURCES:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    code_paths = [Path(__file__).resolve(), Path(nine.__file__).resolve(),
                  Path(nine.six_group.base.__file__).resolve(), Path(model.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py", PROTOCOL]
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (output / "startup_hashes.json").write_text(
        json.dumps({"code": code_hashes, "inputs": input_hashes}, indent=2) + "\n",
        encoding="utf-8",
    )

    prepared, audits = {}, {}
    for code in SOURCES:
        raw, features, audits[code] = nine.prepare_country(code)
        prepared[code] = (raw, features)
    (output / "input_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    @lru_cache(maxsize=None)
    def fit_ratio(country_key: tuple[str, ...]) -> dict:
        parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in country_key]
        frame = pd.concat(parts, ignore_index=True)
        fuel = np.concatenate([nine.fuel_for(prepared[c][0], part)
                               for c, part in zip(country_key, parts)])
        w = model.weights(frame)
        thermal = fuel[:, :4].sum(axis=1)
        ratio_ef = float(np.sum(w * (fuel @ model.FACTORS)) / np.sum(w * thermal))
        country_profiles = []
        for c, part in zip(country_key, parts):
            source_coarse = model.coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
            data = pd.DataFrame(source_coarse, columns=model.GROUPS)
            data["hour"] = part.clock.dt.hour.to_numpy()
            profile = data.groupby("hour")[list(model.GROUPS)].mean().reindex(range(24)).fillna(0.)
            profile = profile.div(profile.mean().replace(0., np.nan)).fillna(1.).to_numpy()
            country_profiles.append(profile)
        return {"ratio_ef": ratio_ef, "profile": np.mean(country_profiles, axis=0)}

    def predict(ratio_model: dict, query: pd.DataFrame) -> np.ndarray:
        q = query[list(model.GROUPS)].to_numpy(float) * ratio_model["profile"][query.clock.dt.hour.to_numpy()]
        total = q.sum(axis=1)
        return ratio_model["ratio_ef"] * np.divide(q[:, 0], total,
                                                    out=np.zeros(len(q), dtype=float), where=total > 0)

    def response(code: str, query: pd.DataFrame) -> np.ndarray:
        raw = prepared[code][0]
        fuel = raw.iloc[query.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)
        return fuel @ model.FACTORS / fuel.sum(axis=1)

    @lru_cache(maxsize=None)
    def pseudo_month_mae(pseudo: str, train_key: tuple[str, ...], half: int) -> float:
        query = prepared[pseudo][1]
        query = query.loc[
            query.year.eq(2023) &
            (query.clock.dt.month.le(6) if half == 1 else query.clock.dt.month.gt(6))
        ].copy()
        if query.empty:
            raise ValueError(f"empty pseudo-target H{half} period for {pseudo}")
        y, p = response(pseudo, query), predict(fit_ratio(train_key), query)
        frame = pd.DataFrame({"month": query.clock.dt.month.to_numpy(), "error": np.abs(y - p)})
        return float(frame.groupby("month").error.mean().mean())

    def select_drop(source_countries: tuple[str, ...]) -> tuple[str | None, list[dict]]:
        contributions = []
        for dropped in source_countries:
            deltas = []
            for pseudo in source_countries:
                if pseudo == dropped:
                    continue
                baseline_train = tuple(sorted(c for c in source_countries if c != pseudo))
                reduced_train = tuple(c for c in baseline_train if c != dropped)
                baseline = pseudo_month_mae(pseudo, baseline_train, 1)
                reduced = pseudo_month_mae(pseudo, reduced_train, 1)
                deltas.append(reduced - baseline)
            contributions.append({
                "candidate_drop": dropped, "pseudo_targets": len(deltas),
                "mean_paired_mae_delta": float(np.mean(deltas)),
                "pseudo_target_deltas": deltas,
            })
        best = min(contributions, key=lambda row: (row["mean_paired_mae_delta"], row["candidate_drop"]))
        selected = best["candidate_drop"] if best["mean_paired_mae_delta"] < 0. else None
        return selected, contributions

    radii_by_target, selections, seals = {}, {}, []
    # Freeze point forecasts and H2 cross-fitted calibration radii for all folds first.
    for target in SOURCES:
        sources = tuple(c for c in SOURCES if c != target)
        selected_drop, contribution_rows = select_drop(sources)
        selections[target] = {"selected_drop": selected_drop,
                              "selection_rule": "drop the lowest mean paired pseudo-target MAE delta if it is negative; otherwise keep all",
                              "candidate_source_contributions": contribution_rows}

        cal_errors = {"ratio_all": [], "ratio_source_drop": [], "anchor700": []}
        cal_frames = []
        for pseudo in sources:
            pseudo_sources = tuple(c for c in sources if c != pseudo)
            nested_drop, _ = select_drop(pseudo_sources)
            all_model = fit_ratio(tuple(sorted(pseudo_sources)))
            reduced_sources = tuple(sorted(c for c in pseudo_sources if c != nested_drop))
            drop_model = fit_ratio(reduced_sources)
            query = prepared[pseudo][1]
            query = query.loc[
                query.year.eq(2023) & query.clock.dt.month.gt(6)
            ].copy()
            y = response(pseudo, query)
            cal_errors["ratio_all"].append(np.abs(y - predict(all_model, query)))
            cal_errors["ratio_source_drop"].append(np.abs(y - predict(drop_model, query)))
            cal_errors["anchor700"].append(np.abs(y - 700. * query.thermal.to_numpy(float)))
            cal_frames.append(pd.DataFrame({"country": pseudo, "region": pseudo}, index=np.arange(len(query))))

        calibration_frame = pd.concat(cal_frames, ignore_index=True)
        cal_weights = model.weights(calibration_frame)
        radii = {method: float(weighted_quantile(np.concatenate(errors), cal_weights, .9))
                 for method, errors in cal_errors.items()}
        radii_by_target[target] = radii

        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        all_prediction = predict(fit_ratio(tuple(sorted(sources))), query)
        retained = tuple(sorted(c for c in sources if c != selected_drop))
        drop_prediction = predict(fit_ratio(retained), query)
        point_path = output / f"{target.upper()}_heldout_predictions.csv.gz"
        pd.DataFrame({
            "time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
            "ratio_all": all_prediction,
            "ratio_source_drop": drop_prediction,
            "anchor700": 700. * query.thermal.to_numpy(float),
        }).to_csv(point_path, index=False, compression="gzip")
        radius_path = output / f"{target.upper()}_source_radii.json"
        radius_path.write_text(json.dumps(radii, indent=2) + "\n", encoding="utf-8")
        seals.append({"target": target.upper(), "path": point_path.name,
                      "sha256": sha(point_path), "radii_path": radius_path.name,
                      "radii_sha256": sha(radius_path), "hours": len(query),
                      "selected_drop": selected_drop,
                      "retained_sources": [c.upper() for c in retained]})
        print(f"sealed {target.upper()}: drop={selected_drop or 'none'}; {len(query)} hours", flush=True)

    (output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n", encoding="utf-8")
    (output / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n", encoding="utf-8")
    if any(sha(Path(path)) != value for path, value in input_hashes.items()):
        raise RuntimeError("input changed during experiment")
    if any(sha(Path(path)) != value for path, value in code_hashes.items()):
        raise RuntimeError("code or protocol changed during experiment")
    if any(sha(output / row[artifact]) != row[hash_field] for row in seals
           for artifact, hash_field in (("path", "sha256"),
                                        ("radii_path", "radii_sha256"))):
        raise RuntimeError("sealed prediction or calibration-radius hash mismatch")

    rows = []
    for target in SOURCES:
        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        truth = response(target, query)
        stored = pd.read_csv(output / f"{target.upper()}_heldout_predictions.csv.gz")
        for method, radius in radii_by_target[target].items():
            pred = (700. * query.thermal.to_numpy(float) if method == "anchor700"
                    else stored[method].to_numpy(float))
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = truth[mask] - pred[mask]
                daily = pd.DataFrame({
                    "truth": truth[mask], "prediction": pred[mask],
                    "day": query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy(),
                })
                center = daily[["truth", "prediction"]] - daily.groupby("day")[
                    ["truth", "prediction"]].transform("mean")
                rows.append({
                    "target_country": target.upper(), "method": method, "month": int(month),
                    "hours": int(mask.sum()), "mae": float(np.abs(error).mean()),
                    "shape_mae": float(np.abs(center.truth - center.prediction).mean()),
                    "coverage90": float((np.abs(error) <= radius).mean()),
                    "width90": float(2 * radius),
                    "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(np.abs(error) - radius, 0))),
                })
    monthly = pd.DataFrame(rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    metric_cols = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    country = monthly.groupby(["target_country", "method"])[metric_cols].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    country.groupby("method")[metric_cols].mean().reset_index().to_csv(
        output / "macro_metrics.csv", index=False
    )
    (output / "status.json").write_text(json.dumps({
        "status": "DEVELOPMENT_NINE_COUNTRY_CN5_PSEUDOTARGET_SOURCE_DROP",
        "countries": [c.upper() for c in SOURCES], "target_year": 2024,
        "training_year": 2022, "selection_period": "2023H1",
        "source_crossfit_calibration_period": "2023H2",
        "target_budget": "five coarse monthly generation shares t-2 through t-7 plus calendar",
        "other_renewable": "excluded from public input; retained in fixed-factor proxy denominator",
        "labels": "fixed-factor generation-side proxy, not measured CIF",
        "target_labels_used_before_prediction_seal": False,
        "final_blind_confirmation": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"completed source-drop LOCO: {len(monthly)} monthly rows", flush=True)


if __name__ == "__main__":
    main()
