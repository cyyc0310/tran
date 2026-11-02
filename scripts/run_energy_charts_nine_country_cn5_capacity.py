"""Compare lagged annual thermal-capacity information on nine-country LOCO."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_energy_charts_nine_country_cn5_loco as nine
from transcif.research import multicountry_cn5 as model
from transcif.research.monthly_adaptation import weighted_quantile

ROOT = nine.ROOT
PROTOCOL = ROOT / "docs/ENERGY_CHARTS_NINE_COUNTRY_CN5_CAPACITY_PROTOCOL_20261101.md"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-csv", type=Path, required=True)
    parser.add_argument("--capacity-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    capacity = pd.read_csv(args.capacity_csv)
    required = {"country", "stat_year", "thermal_capacity_share"}
    if not required.issubset(capacity.columns):
        raise ValueError(f"capacity file needs columns {sorted(required)}")
    capacity["country"] = capacity.country.astype(str).str.upper()
    capacity["stat_year"] = capacity.stat_year.astype(int)
    capacity = capacity.loc[capacity.country.isin([c.upper() for c in nine.COUNTRIES])]
    expected_pairs = {(c.upper(), year) for c in nine.COUNTRIES for year in (2020, 2021, 2022)}
    observed_pairs = set(zip(capacity.country, capacity.stat_year))
    if expected_pairs - observed_pairs:
        raise ValueError(f"missing capacity rows: {sorted(expected_pairs - observed_pairs)}")
    if capacity.duplicated(["country", "stat_year"]).any():
        raise ValueError("duplicate country/year capacity rows")

    input_paths = [PROTOCOL, args.capacity_csv.resolve(), args.capacity_manifest.resolve()]
    for code in nine.COUNTRIES:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    code_paths = [
        Path(__file__).resolve(), Path(nine.__file__).resolve(), 
        Path(nine.six_group.base.__file__).resolve(), Path(model.__file__).resolve(),
        ROOT / "src/transcif/research/cross_country.py",
        ROOT / "src/transcif/research/monthly_adaptation.py", PROTOCOL,
        ROOT / "scripts/fetch_eurostat_capacity_expansion.py",
        ROOT / "scripts/parse_eurostat_energy_capacity.py",
        ROOT / "scripts/merge_eurostat_capacity_panel.py",
    ]
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (output / "startup_hashes.json").write_text(
        json.dumps({"code": code_hashes, "inputs": input_hashes}, indent=2) + "\n",
        encoding="utf-8",
    )

    prepared, audits = {}, {}
    for code in nine.COUNTRIES:
        raw, features, audits[code] = nine.prepare_country(code)
        features = model.add_capacity_features(features, capacity)
        prepared[code] = (raw, features)
    (output / "input_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    sealed, radii_by_target = [], {}
    for target in nine.COUNTRIES:
        sources = [c for c in nine.COUNTRIES if c != target]
        train_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([nine.fuel_for(prepared[c][0], part)
                                     for c, part in zip(sources, train_parts)])
        train_coarse = np.concatenate([
            model.coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
            for c, part in zip(sources, train_parts)
        ])
        if not len(train) or len(train) != len(train_fuel) or len(train) != len(train_coarse):
            raise ValueError(f"empty or misaligned training cohort for {target}")
        base_models = model.fit(train, train_fuel, train_coarse)
        capacity_models = model.fit_capacity_heads(train, train_fuel)

        cal_parts = [prepared[c][1].loc[
            prepared[c][1].year.eq(2023) & prepared[c][1].clock.dt.month.le(6)
        ] for c in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([nine.fuel_for(prepared[c][0], part)
                                   for c, part in zip(sources, cal_parts)])
        cal_y = cal_fuel @ model.FACTORS / cal_fuel.sum(axis=1)
        cal_weights = model.weights(calibration)
        cal_predictions = model.predict(base_models, calibration)
        cal_predictions.update(model.predict_capacity_heads(capacity_models, calibration, base_models))
        radii = {name: float(weighted_quantile(np.abs(cal_y - value), cal_weights, .9))
                 for name, value in cal_predictions.items()}

        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        point = model.predict(base_models, query)
        point.update(model.predict_capacity_heads(capacity_models, query, base_models))
        path = output / f"{target.upper()}_heldout_predictions.csv.gz"
        pd.DataFrame({
            "time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(), **point,
        }).to_csv(path, index=False, compression="gzip")
        sealed.append({"target": target.upper(), "path": path.name, "sha256": sha(path),
                       "hours": len(query), "source_countries": [c.upper() for c in sources]})
        radii_by_target[target] = radii
        radii_path = output / f"{target.upper()}_source_radii.json"
        radii_path.write_text(
            json.dumps(radii, indent=2) + "\n", encoding="utf-8"
        )
        sealed[-1]["radii_path"] = radii_path.name
        sealed[-1]["radii_sha256"] = sha(radii_path)
        print(f"sealed {target.upper()}: {len(query)} hours", flush=True)

    (output / "prediction_seal.json").write_text(
        json.dumps(sealed, indent=2) + "\n", encoding="utf-8"
    )
    if any(sha(Path(p)) != expected for p, expected in input_hashes.items()):
        raise RuntimeError("input changed during experiment")
    if any(sha(Path(p)) != expected for p, expected in code_hashes.items()):
        raise RuntimeError("code or protocol changed during experiment")
    if any(sha(output / row[artifact]) != row[hash_field] for row in sealed
           for artifact, hash_field in (("path", "sha256"),
                                        ("radii_path", "radii_sha256"))):
        raise RuntimeError("sealed prediction or calibration-radius hash mismatch")

    rows = []
    for target in nine.COUNTRIES:
        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        query_rows = raw.iloc[query.raw_row.to_numpy()]
        fuel = query_rows[list(model.FINE)].to_numpy(float)
        truth = fuel @ model.FACTORS / fuel.sum(axis=1)
        stored = pd.read_csv(output / f"{target.upper()}_heldout_predictions.csv.gz")
        for method, radius in radii_by_target[target].items():
            prediction = stored[method].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = truth[mask] - prediction[mask]
                daily = pd.DataFrame({
                    "truth": truth[mask], "prediction": prediction[mask],
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

    pairs = [
        ("capacity_direct", "direct_history"),
        ("capacity_factor", "factor_history"),
        ("capacity_ridge", "ridge_history"),
    ]
    deltas = []
    for treatment, control in pairs:
        a = country.loc[country.method.eq(treatment)].set_index("target_country")
        b = country.loc[country.method.eq(control)].set_index("target_country")
        for target in a.index:
            deltas.append({
                "target_country": target, "treatment": treatment, "control": control,
                **{metric: float(a.loc[target, metric] - b.loc[target, metric])
                   for metric in metric_cols},
            })
    pd.DataFrame(deltas).to_csv(output / "capacity_increment_pair.csv", index=False)
    (output / "status.json").write_text(json.dumps({
        "status": "DEVELOPMENT_NINE_COUNTRY_STRICT_CN5_CAPACITY_LOCO",
        "countries": [c.upper() for c in nine.COUNTRIES], "target_year": 2024,
        "training_year": 2022, "source_calibration": "2023H1",
        "capacity_budget": "Eurostat nrg_inf_epc CF/TOTAL net maximum capacity; 2020→2022, 2021→2023, 2022→2024",
        "capacity_vintage": "current revised values; lag simulation, not historical as-issued",
        "target_budget": "thermal, hydro, nuclear, wind, solar monthly shares t-2 through t-7, calendar, and annual capacity share",
        "other_renewable": "excluded from public input; retained in fixed-factor hourly proxy denominator",
        "labels": "fixed-factor generation-side proxy, not measured CIF",
        "target_labels_used_before_prediction_seal": False,
        "final_blind_confirmation": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"completed nine-country capacity LOCO: {len(monthly)} monthly rows", flush=True)


if __name__ == "__main__":
    main()
