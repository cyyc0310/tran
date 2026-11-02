#!/usr/bin/env python3
"""Source-only LOCO test of coarse-input disagreement-conditioned intervals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as response  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = nine.COUNTRIES
PROTOCOL = ROOT / "docs/CN5_DISAGREEMENT_RISK_PSEUDOTARGET_PROTOCOL_20261101.md"
BINS = 3
MIN_BIN_COUNTRIES = 4


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fit(prepared: dict, countries: list[str]):
    parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in countries]
    train = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([nine.fuel_for(prepared[c][0], part)
                           for c, part in zip(countries, parts)])
    coarse = np.concatenate([model.coarse_values(
        prepared[c][0].iloc[part.raw_row.to_numpy()])
        for c, part in zip(countries, parts)])
    fitted = model.fit(train, fuel, coarse)
    fuel_head = response.fit_fuelshare(train, fuel)
    return fitted, fuel_head


def equal_country_weights(frame: pd.DataFrame) -> np.ndarray:
    out = model.weights(pd.DataFrame({"country": frame.country, "region": frame.country}))
    return out / out.sum()


def metrics(frame: pd.DataFrame, radii: np.ndarray) -> dict:
    y = frame.truth.to_numpy(float)
    p = frame.ratio.to_numpy(float)
    low, high = p - radii, p + radii
    error = np.abs(y - p)
    outside = np.maximum(low - y, 0.) + np.maximum(y - high, 0.)
    return {"hours": len(frame), "mae": float(error.mean()),
            "coverage90": float(((y >= low) & (y <= high)).mean()),
            "width90": float(np.mean(high-low)),
            "interval_score90": float(np.mean(high-low+20.*outside))}


def calibrate(train: pd.DataFrame, test: pd.DataFrame, signal: str) -> tuple[np.ndarray, dict]:
    residual = np.abs(train.truth.to_numpy(float) - train.ratio.to_numpy(float))
    weights = equal_country_weights(train)
    global_radius = float(weighted_quantile(residual, weights, .9))
    edges = np.unique(np.quantile(train[signal].to_numpy(float), np.linspace(0, 1, BINS + 1)))
    assignment = np.digitize(test[signal].to_numpy(float), edges[1:-1], right=True)
    train_assignment = np.digitize(train[signal].to_numpy(float), edges[1:-1], right=True)
    radii = np.full(len(test), global_radius, dtype=float)
    bin_details = []
    for index in range(len(edges)-1):
        take = train_assignment == index
        countries = train.loc[take, "country"].nunique()
        if countries < MIN_BIN_COUNTRIES or not take.any():
            bin_details.append({"bin": index, "countries": int(countries),
                                "fallback": True, "radius": global_radius})
            continue
        bin_weights = equal_country_weights(train.loc[take])
        radius = float(weighted_quantile(residual[take], bin_weights, .9))
        radii[assignment == index] = radius
        bin_details.append({"bin": index, "countries": int(countries),
                            "fallback": False, "radius": radius,
                            "train_signal_min": float(train.loc[take, signal].min()),
                            "train_signal_max": float(train.loc[take, signal].max())})
    return radii, {"global_radius": global_radius, "edges": edges.tolist(), "bins": bin_details}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for country in COUNTRIES:
        folder, manifest, _ = nine.six_group.data_locations(country)
        input_paths.extend([manifest, folder / f"{country}_2022_2024.json"])
        if country in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    code_paths = [Path(__file__).resolve(), Path(response.__file__).resolve(),
                  Path(nine.__file__).resolve(), Path(nine.six_group.__file__).resolve(),
                  Path(nine.six_group.base.__file__).resolve(), Path(model.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(path.resolve()): sha(path) for path in sorted(set(input_paths))}
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}

    prepared, audits = {}, {}
    for country in COUNTRIES:
        raw, features, audits[country] = nine.prepare_country(country)
        prepared[country] = (raw, features)

    pseudo = {}
    for country in COUNTRIES:
        sources = [x for x in COUNTRIES if x != country]
        fitted, fuel_head = source_fit(prepared, sources)
        raw, features = prepared[country]
        query = features.loc[features.year.eq(2023) & features.clock.dt.month.gt(6)].copy()
        if query.empty:
            raise ValueError(f"missing H2 pseudo-target rows for {country}")
        fuel = nine.fuel_for(raw, query)
        truth = fuel @ model.FACTORS / fuel.sum(axis=1)
        predictions = model.predict(fitted, query)
        predictions["fuelshare_history"] = response.predict_fuelshare(fuel_head, fitted, query)
        ratio = predictions["ratio_recomposition"]
        factor = predictions["factor_history"]
        fuelshare = predictions["fuelshare_history"]
        frame = pd.DataFrame({"country": country.upper(), "truth": truth, "ratio": ratio,
            "factor_ratio_disagreement": np.abs(factor-ratio),
            "three_model_spread": np.maximum.reduce([ratio,factor,fuelshare])-
                                 np.minimum.reduce([ratio,factor,fuelshare]),
            "time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()})
        pseudo[country] = frame
        print(f"source pseudo-target {country.upper()}: {len(frame)} H2 hours", flush=True)

    all_rows, calibration_records = [], []
    signals = ("factor_ratio_disagreement", "three_model_spread")
    for held in COUNTRIES:
        train = pd.concat([pseudo[c] for c in COUNTRIES if c != held], ignore_index=True)
        test = pseudo[held]
        baseline_radius = float(weighted_quantile(
            np.abs(train.truth-train.ratio), equal_country_weights(train), .9))
        baseline = metrics(test, np.full(len(test), baseline_radius))
        all_rows.append({"heldout_country": held.upper(), "method": "global_radius", **baseline})
        calibration_records.append({"heldout_country": held.upper(), "method": "global_radius",
                                    "radius": baseline_radius})
        for signal in signals:
            radii, details = calibrate(train, test, signal)
            score = metrics(test, radii)
            all_rows.append({"heldout_country": held.upper(), "method": f"binned_{signal}", **score})
            calibration_records.append({"heldout_country": held.upper(),
                "method": f"binned_{signal}", "details": details})
        daily = test.copy()
        daily["date"] = pd.to_datetime(daily.time).dt.strftime("%Y-%m-%d")
        daily["absolute_error"] = np.abs(daily.truth-daily.ratio)
        daily = daily.groupby("date", as_index=False).agg(
            absolute_error=("absolute_error", "mean"),
            **{signal: (signal, "mean") for signal in signals})
        for signal in signals:
            rho = daily[signal].corr(daily.absolute_error, method="spearman")
            calibration_records.append({"heldout_country": held.upper(), "method": f"risk_rank_{signal}",
                                        "daily_spearman": float(rho)})

    results = pd.DataFrame(all_rows)
    results.to_csv(out / "heldout_pseudotarget_metrics.csv", index=False)
    macro = results.groupby("method", as_index=False).agg(
        macro_mae=("mae", "mean"), macro_coverage90=("coverage90", "mean"),
        macro_width90=("width90", "mean"), macro_interval_score90=("interval_score90", "mean"),
        minimum_country_coverage90=("coverage90", "min"))
    macro.to_csv(out / "macro_metrics.csv", index=False)
    pd.DataFrame(calibration_records).to_json(out / "calibration_details.json", orient="records",
                                               indent=2)
    prediction_path = out / "pseudotarget_h2_predictions.csv.gz"
    pd.concat(pseudo.values(), ignore_index=True).to_csv(prediction_path, index=False,
                                                         compression="gzip")
    audit = {"status": "SOURCE_ONLY_STRICT_CN5_H2_LOCO_RISK_TEST",
        "countries": [c.upper() for c in COUNTRIES], "train_year": 2022,
        "pseudo_target_period": "2023H2", "risk_calibration": "leave-one-pseudo-country-out",
        "target_inputs": "strict CN5 t-2 through t-7 monthly shares and calendar",
        "label": "fixed-factor generation-side proxy; not measured CIF",
        "risk_outcome": "absolute error of ratio_recomposition",
        "conditional_risk_bins": BINS, "min_countries_per_bin": MIN_BIN_COUNTRIES,
        "target_labels_used": False, "prediction_file_sha256": sha(prediction_path),
        "input_sha256": input_hashes, "code_sha256": code_hashes,
        "country_audits": audits}
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n",
                                    encoding="utf-8")
    print(macro.to_string(index=False), flush=True)
    if any(sha(Path(p)) != digest for p, digest in input_hashes.items()):
        raise RuntimeError("input changed during experiment")
    if any(sha(Path(p)) != digest for p, digest in code_hashes.items()):
        raise RuntimeError("code changed during experiment")


if __name__ == "__main__":
    main()
