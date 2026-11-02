"""Compare official EIA-923 BA monthly inputs with hourly-reconstructed mixes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from audit_eia923_ba_mix_alignment import DEFAULT_BAS, MIX_FIELDS, parse_eia923
from run_cross_country import ROOT
from transcif.research.budget_transfer import budget_views
from transcif.research.cross_country import COARSE, FUELS, fit_models, labels, load_inputs, monthly_inputs
from transcif.research.monthly_adaptation import (
    METHODS, fit_risks, fit_source, monthly_table, predict_hours, risk_bounds,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def eia923_month_table(paths: dict[int, Path]) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {ba: [] for ba in DEFAULT_BAS}
    for year, path in sorted(paths.items()):
        panel = parse_eia923(path, DEFAULT_BAS, year)
        for row in panel.to_dict("records"):
            values = {field: float(row[f"eia923_{field}_mwh"]) for field in MIX_FIELDS}
            total = sum(values.values())
            if total <= 0:
                continue
            coarse = {
                "thermal": values["coal"] + values["gas"] + values["oil"] + values["other"],
                "hydro": values["hydro"], "nuclear": values["nuclear"],
                "wind": values["wind"], "solar": values["solar"],
            }
            frame = pd.DataFrame([coarse], index=[pd.Period(row["month"], freq="M")])
            frames[row["ba"]].append(frame.div(total, axis=1))
    result = {}
    for ba, rows in frames.items():
        if not rows:
            raise ValueError(f"EIA-923 contains no months for {ba}")
        table = pd.concat(rows).sort_index()
        if table.index.duplicated().any() or not table[list(COARSE)].notna().all().all():
            raise ValueError(f"duplicate/missing EIA-923 BA-month rows for {ba}")
        result[ba] = table[list(COARSE)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eia923-2022", type=Path, required=True)
    parser.add_argument("--eia923-2023", type=Path, required=True)
    parser.add_argument("--eia923-2024", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    eia_paths = {2022: args.eia923_2022, 2023: args.eia923_2023, 2024: args.eia923_2024}
    code_files = [Path(__file__), Path(__file__).with_name("audit_eia923_ba_mix_alignment.py"),
                  ROOT/"src/transcif/research/monthly_adaptation.py",
                  ROOT/"src/transcif/research/cross_country.py",
                  ROOT/"src/transcif/research/budget_transfer.py",
                  ROOT/"docs/EIA923_CROSS_COUNTRY_EXPERIMENT_PROTOCOL_20261101.md"]
    startup_hashes = {str(p.relative_to(ROOT)): digest(p) for p in code_files}
    input_hashes = {str(path.resolve()): digest(path) for path in eia_paths.values()}
    (out/"startup_code_hashes.json").write_text(json.dumps(startup_hashes, indent=2))
    (out/"environment.json").write_text(json.dumps({"numpy": np.__version__, "pandas": pd.__version__,
                                                       "sklearn": sklearn.__version__}, indent=2))

    raw, input_audit = load_inputs(ROOT)
    input_hashes.update({str((ROOT / audit["path"]).resolve()): digest(ROOT / audit["path"])
                         for audit in input_audit.values()})
    (out/"input_hashes.json").write_text(json.dumps(input_hashes, indent=2))
    hourly_tables = {region: monthly_table(frame) for region, frame in raw.items()}
    official_us_tables = eia923_month_table(eia_paths)
    arms = {
        "hourly_reconstructed": hourly_tables,
        "eia923_us": {**hourly_tables, **official_us_tables},
    }
    lagged = {region: {lag: monthly_inputs(frame, lag) for lag in (2, 3, 4)}
              for region, frame in raw.items()}
    all_frames = {region: features[2].dropna(subset=list(COARSE)) for region, features in lagged.items()}
    records, seals = [], []

    for arm_name, tables in arms.items():
        (out/f"{arm_name}_table_availability.json").write_text(json.dumps(
            {region: {"months": [str(p) for p in table.index], "rows": len(table),
                      "first": str(table.index.min()), "last": str(table.index.max())}
             for region, table in tables.items()}, indent=2))
        for source_country, target_country in (("US", "GB"), ("GB", "US")):
            sources = [r for r in raw if raw[r].country.iloc[0] == source_country and r != "BPAT"]
            targets = [r for r in raw if raw[r].country.iloc[0] == target_country]
            source_frames = [budget_views(lagged[r], 2022, False).drop_duplicates("raw_row") for r in sources]
            train = pd.concat(source_frames, ignore_index=True)
            fuel = np.concatenate([raw[r].loc[f.raw_row, list(FUELS)].to_numpy() for r, f in zip(sources, source_frames)])
            native = fit_models(train, fuel)
            source_model = fit_source({r: tables[r] for r in sources})
            (out/f"{arm_name}_{source_country}_model.json").write_text(json.dumps(
                {"source_regions": sources, "ridge_coef": source_model.coef_.tolist(),
                 "ridge_intercept": source_model.intercept_.tolist(),
                 "thermal_ef": native["thermal_ef"], "profile": native["profile"].tolist()}, indent=2))

            for method in METHODS:
                split = {half: [] for half in (1, 2)}
                for region in sources:
                    frame = all_frames[region].loc[all_frames[region].year.eq(2023)]
                    point, features = predict_hours(frame, tables[region], source_model, method, native)
                    error = labels(raw[region].loc[frame.index]) - point
                    for half in (1, 2):
                        mask = (frame.clock.dt.month.le(6) if half == 1 else frame.clock.dt.month.gt(6)).to_numpy()
                        split[half].append((features[mask], error[mask], np.repeat(region, mask.sum())))
                parts = []
                for half in (1, 2):
                    parts.extend(np.concatenate([entry[i] for entry in split[half]]) for i in range(3))
                fitted = fit_risks(*parts)
                risk_path = out/f"{arm_name}_{source_country}_{method}_risk.joblib"
                joblib.dump(fitted, risk_path)
                restored = joblib.load(risk_path)
                for kind in ("global", "hetero", "cqr"):
                    left = risk_bounds(fitted, parts[3][:32], np.zeros(min(32, len(parts[3]))))[kind]
                    right = risk_bounds(restored, parts[3][:32], np.zeros(min(32, len(parts[3]))))[kind]
                    for a, b in zip(left, right):
                        np.testing.assert_array_equal(a, b)
                (out/f"{arm_name}_{source_country}_{method}_risk.json").write_text(json.dumps(
                    {**{k: fitted[k] for k in ("bias", "global", "hetero", "cqr")},
                     "fit_iterations": {k: int(fitted[k].n_iter_) for k in ("scale", "lo", "hi")}}, indent=2))
                for region in targets:
                    frame = all_frames[region].loc[all_frames[region].year.eq(2024)]
                    point, features = predict_hours(frame, tables[region], source_model, method, native)
                    bounds = risk_bounds(fitted, features, point)
                    values = {"time": frame.time.astype(str), "point_raw": point, "point": point+fitted["bias"]}
                    for risk, (lo, hi) in bounds.items():
                        values[f"{risk}_lo"], values[f"{risk}_hi"] = lo, hi
                    path = out/f"{arm_name}_{source_country}_to_{region}_{method}.csv"
                    pd.DataFrame(values).to_csv(path, index=False)
                    seals.append({"path": path.name, "sha256": digest(path)})
                    records.append((arm_name, source_country, target_country, region, method, frame.copy(), path))
                print(arm_name, source_country, method, "predictions written", flush=True)

    (out/"prediction_seal.json").write_text(json.dumps(seals, indent=2))
    (out/"input_audit.json").write_text(json.dumps(input_audit, indent=2))
    rows = []
    for arm_name, source_country, target_country, region, method, frame, path in records:
        prediction = pd.read_csv(path)
        target = raw[region].loc[frame.index]
        y = labels(target)
        for month in sorted(frame.clock.dt.month.unique()):
            mask = frame.clock.dt.month.eq(month).to_numpy()
            target_month, point, actual = frame.loc[mask], prediction.point.to_numpy()[mask], y[mask]
            centered = pd.DataFrame({"day": target_month.clock.dt.strftime("%Y-%m-%d").to_numpy(),
                                     "y": actual, "p": point})
            centered = centered[["y", "p"]]-centered.groupby(
                target_month.clock.dt.strftime("%Y-%m-%d").to_numpy())[["y", "p"]].transform("mean")
            for risk in ("global", "hetero", "cqr"):
                lo, hi = prediction[f"{risk}_lo"].to_numpy()[mask], prediction[f"{risk}_hi"].to_numpy()[mask]
                rows.append({"arm": arm_name, "source": source_country, "target": target_country,
                             "region": region, "method": method, "risk": risk, "month": int(month),
                             "common_month": int(month) in (2, 4, 6), "hours": int(mask.sum()),
                             "mae": float(np.abs(actual-point).mean()),
                             "raw_mae": float(np.abs(actual-prediction.point_raw.to_numpy()[mask]).mean()),
                             "shape_mae": float(np.abs(centered.y-centered.p).mean()),
                             "coverage90": float(((actual >= lo)&(actual <= hi)).mean()),
                             "width90": float(np.mean(hi-lo)),
                             "interval_score90": float(np.mean(hi-lo+20*np.maximum(lo-actual, 0)
                                                                 +20*np.maximum(actual-hi, 0)))})
    scores = pd.DataFrame(rows)
    scores.to_csv(out/"monthly_metrics.csv", index=False)
    fields = ["mae", "raw_mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    groups = ["arm", "source", "target", "region", "method", "risk"]
    regional = scores.loc[scores.common_month].groupby(groups)[fields].mean().reset_index()
    regional.to_csv(out/"common_region_metrics.csv", index=False)
    country = regional.groupby(["arm", "source", "target", "method", "risk"])[fields].mean().reset_index()
    country.to_csv(out/"common_country_metrics.csv", index=False)
    if any(digest(ROOT/path) != value for path, value in startup_hashes.items()):
        raise RuntimeError("code or protocol changed during execution")
    provenance = {**startup_hashes, **{path.resolve().relative_to(ROOT).as_posix(): value
                                        for path, value in ((args.eia923_2022, input_hashes[str(args.eia923_2022.resolve())]),
                                                            (args.eia923_2023, input_hashes[str(args.eia923_2023.resolve())]),
                                                            (args.eia923_2024, input_hashes[str(args.eia923_2024.resolve())]))}}
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2))
    print(country.to_string(index=False))


if __name__ == "__main__":
    main()
