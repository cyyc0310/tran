"""Paired US-GB test of annual coal/gas capacity on top of official monthly mixes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from audit_eia923_ba_mix_alignment import DEFAULT_BAS
from run_cross_country import ROOT
from run_eia923_crosscountry_monthly import digest, eia923_month_table
from transcif.research.budget_transfer import budget_views
from transcif.research.cross_country import FUELS, fit_models, labels, load_inputs, monthly_inputs
from transcif.research.monthly_adaptation import (
    METHODS, fit_risks, fit_source, monthly_table, predict_hours, risk_bounds,
)


def read_capacity(path: Path) -> dict[str, float]:
    source = json.loads(path.read_text())
    regions = source["regions"]
    result = {}
    for region, row in regions.items():
        total = float(row["coal_mw"]) + float(row["gas_mw"])
        if total > 0:
            result[region] = .4 + .6 * float(row["coal_mw"]) / total
    missing = (set(DEFAULT_BAS) | {"GB"}) - set(result)
    if missing:
        raise ValueError(f"missing annual coal/gas capacity for {sorted(missing)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-json", type=Path, required=True)
    parser.add_argument("--eia923-2022", type=Path, required=True)
    parser.add_argument("--eia923-2023", type=Path, required=True)
    parser.add_argument("--eia923-2024", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    eia_paths = {2022: args.eia923_2022, 2023: args.eia923_2023, 2024: args.eia923_2024}
    code_files = [Path(__file__), Path(__file__).with_name("run_eia923_crosscountry_monthly.py"),
                  Path(__file__).with_name("audit_eia923_ba_mix_alignment.py"),
                  ROOT/"src/transcif/research/monthly_adaptation.py",
                  ROOT/"src/transcif/research/cross_country.py",
                  ROOT/"src/transcif/research/budget_transfer.py",
                  ROOT/"docs/EIA923_CAPACITY_INCREMENT_PROTOCOL_20261101.md"]
    startup_hashes = {str(p.relative_to(ROOT)): digest(p) for p in code_files}
    capacity_ef = read_capacity(args.capacity_json)
    input_hashes = {str(path.resolve()): digest(path) for path in [*eia_paths.values(), args.capacity_json]}
    (out/"startup_code_hashes.json").write_text(json.dumps(startup_hashes, indent=2))
    (out/"capacity_ef.json").write_text(json.dumps(capacity_ef, indent=2))

    raw, input_audit = load_inputs(ROOT)
    input_hashes.update({str((ROOT/audit["path"]).resolve()): digest(ROOT/audit["path"])
                         for audit in input_audit.values()})
    (out/"input_hashes.json").write_text(json.dumps(input_hashes, indent=2))
    tables = {"GB": monthly_table(raw["GB"]), **eia923_month_table(eia_paths)}
    lagged = {region: {lag: monthly_inputs(frame, lag) for lag in (2, 3, 4)}
              for region, frame in raw.items()}
    all_frames = {region: features[2].dropna(subset=["thermal", "hydro", "nuclear", "wind", "solar"])
                  for region, features in lagged.items()}
    records, seals = [], []

    for source_country, target_country in (("US", "GB"), ("GB", "US")):
        sources = [r for r in raw if raw[r].country.iloc[0] == source_country and r != "BPAT"]
        targets = [r for r in raw if raw[r].country.iloc[0] == target_country]
        source_frames = [budget_views(lagged[r], 2022, False).drop_duplicates("raw_row") for r in sources]
        train = pd.concat(source_frames, ignore_index=True)
        fuel = np.concatenate([raw[r].loc[f.raw_row, list(FUELS)].to_numpy() for r, f in zip(sources, source_frames)])
        native = fit_models(train, fuel)
        source_model = fit_source({r: tables[r] for r in sources})
        for arm in ("monthly", "monthly_plus_capacity"):
            use_capacity = arm == "monthly_plus_capacity"
            (out/f"{arm}_{source_country}_model.json").write_text(json.dumps(
                {"source_regions": sources, "capacity_adjustment": use_capacity,
                 "capacity_ef": {r: capacity_ef[r] for r in [*sources, *targets]},
                 "source_thermal_ef": native["thermal_ef"],
                 "profile": native["profile"].tolist()}, indent=2))
            for method in METHODS:
                halves = {h: [] for h in (1, 2)}
                for region in sources:
                    frame = all_frames[region].loc[all_frames[region].year.eq(2023)]
                    point, features = predict_hours(frame, tables[region], source_model, method, native)
                    if use_capacity:
                        point = point * capacity_ef[region] / native["thermal_ef"]
                    error = labels(raw[region].loc[frame.index]) - point
                    for half in (1, 2):
                        mask = (frame.clock.dt.month.le(6) if half == 1 else frame.clock.dt.month.gt(6)).to_numpy()
                        halves[half].append((features[mask], error[mask], np.repeat(region, mask.sum())))
                parts = []
                for half in (1, 2):
                    parts.extend(np.concatenate([entry[i] for entry in halves[half]]) for i in range(3))
                fitted = fit_risks(*parts)
                joblib.dump(fitted, out/f"{arm}_{source_country}_{method}_risk.joblib")
                for region in targets:
                    frame = all_frames[region].loc[all_frames[region].year.eq(2024)]
                    point, features = predict_hours(frame, tables[region], source_model, method, native)
                    if use_capacity:
                        point = point * capacity_ef[region] / native["thermal_ef"]
                    bounds = risk_bounds(fitted, features, point)
                    values = {"time": frame.time.astype(str), "point_raw": point,
                              "point": point+fitted["bias"]}
                    for risk, (lo, hi) in bounds.items():
                        values[f"{risk}_lo"], values[f"{risk}_hi"] = lo, hi
                    path = out/f"{arm}_{source_country}_to_{region}_{method}.csv"
                    pd.DataFrame(values).to_csv(path, index=False)
                    seals.append({"path": path.name, "sha256": digest(path)})
                    records.append((arm, source_country, target_country, region, method, frame.copy(), path))
                print(arm, source_country, method, "predictions written", flush=True)

    (out/"prediction_seal.json").write_text(json.dumps(seals, indent=2))
    (out/"input_audit.json").write_text(json.dumps(input_audit, indent=2))
    rows = []
    for arm, source_country, target_country, region, method, frame, path in records:
        prediction = pd.read_csv(path)
        target = raw[region].loc[frame.index]
        y = labels(target)
        for month in sorted(frame.clock.dt.month.unique()):
            mask = frame.clock.dt.month.eq(month).to_numpy()
            target_month, point, actual = frame.loc[mask], prediction.point.to_numpy()[mask], y[mask]
            dates = target_month.clock.dt.strftime("%Y-%m-%d").to_numpy()
            centered = pd.DataFrame({"y": actual, "p": point})
            centered = centered-centered.groupby(dates).transform("mean")
            for risk in ("global", "hetero", "cqr"):
                lo, hi = prediction[f"{risk}_lo"].to_numpy()[mask], prediction[f"{risk}_hi"].to_numpy()[mask]
                rows.append({"arm": arm, "source": source_country, "target": target_country,
                             "region": region, "method": method, "risk": risk, "month": int(month),
                             "common_month": int(month) in (2, 4, 6), "hours": int(mask.sum()),
                             "mae": float(np.abs(actual-point).mean()),
                             "shape_mae": float(np.abs(centered.y-centered.p).mean()),
                             "coverage90": float(((actual >= lo)&(actual <= hi)).mean()),
                             "width90": float(np.mean(hi-lo)),
                             "interval_score90": float(np.mean(hi-lo+20*np.maximum(lo-actual, 0)
                                                                 +20*np.maximum(actual-hi, 0)))})
    scores = pd.DataFrame(rows)
    scores.to_csv(out/"monthly_metrics.csv", index=False)
    fields = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    groups = ["arm", "source", "target", "region", "method", "risk"]
    regional = scores.loc[scores.common_month].groupby(groups)[fields].mean().reset_index()
    regional.to_csv(out/"common_region_metrics.csv", index=False)
    country = regional.groupby(["arm", "source", "target", "method", "risk"])[fields].mean().reset_index()
    country.to_csv(out/"common_country_metrics.csv", index=False)
    if any(digest(ROOT/path) != value for path, value in startup_hashes.items()):
        raise RuntimeError("code or protocol changed during execution")
    print(country.to_string(index=False))


if __name__ == "__main__":
    main()
