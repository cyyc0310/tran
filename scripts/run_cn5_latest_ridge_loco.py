"""Frozen nine-country strict-CN5 LOCO check for a latest-month Ridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_energy_charts_nine_country_cn5_loco as base  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_LATEST_COARSE_RIDGE_PSEUDOTARGET_PROTOCOL_20261102.md"
COUNTRIES = base.COUNTRIES


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def latest_ridge(train: pd.DataFrame, y: np.ndarray):
    w = model.weights(train)
    scaler = StandardScaler().fit(train[list(model.LATEST)], sample_weight=w)
    estimator = Ridge(alpha=10.).fit(scaler.transform(train[list(model.LATEST)]), y,
                                     sample_weight=w * len(w))
    return scaler, estimator


def ridge_predict(fitted, frame: pd.DataFrame) -> np.ndarray:
    scaler, estimator = fitted
    return np.clip(estimator.predict(scaler.transform(frame[list(model.LATEST)])), 0., 1000.)


def score_group(method: str, target: str, stratum: str, month: int,
                query: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                radius: float) -> dict:
    err = truth - prediction
    daily = pd.DataFrame({"truth": truth, "prediction": prediction,
                          "day": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
    centered = daily[["truth", "prediction"]] - daily.groupby("day")[
        ["truth", "prediction"]].transform("mean")
    return {"target_country": target.upper(), "stratum": stratum, "method": method,
            "month": month, "hours": len(query), "mae": float(np.abs(err).mean()),
            "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
            "coverage90": float((np.abs(err) <= radius).mean()), "width90": 2 * radius,
            "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(np.abs(err)-radius, 0)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    prepared, audits = {}, {}
    for code in COUNTRIES:
        prepared[code] = base.prepare_country(code)[:2]

    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(base.__file__).resolve(),
                  Path(base.six_group.__file__).resolve(), Path(base.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_paths = [PROTOCOL]
    for code in COUNTRIES:
        folder, manifest, _ = base.six_group.data_locations(code)
        input_paths.extend([manifest, folder / f"{code}_2022_2024.json"])
        if code in base.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    hashes = {"code": {str(p.resolve()): sha(p) for p in code_paths},
              "inputs": {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}}
    (out / "startup_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")

    sealed, radii_by_target = [], {}
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        train = pd.concat(parts, ignore_index=True)
        fuels = np.concatenate([base.fuel_for(prepared[c][0], part) for c, part in zip(sources, parts)])
        coarse = np.concatenate([model.coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
                                 for c, part in zip(sources, parts)])
        models = model.fit(train, fuels, coarse)
        y_train = fuels @ model.FACTORS / fuels.sum(axis=1)
        models["ridge_latest"] = latest_ridge(train, y_train)

        cal_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2023) &
                                      prepared[c][1].clock.dt.month.le(6)] for c in sources]
        cal = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([base.fuel_for(prepared[c][0], p) for c, p in zip(sources, cal_parts)])
        cal_y = cal_fuel @ model.FACTORS / cal_fuel.sum(axis=1)
        cal_weights = model.weights(cal)
        cal_pred = model.predict(models, cal)
        cal_pred["ridge_latest"] = ridge_predict(models["ridge_latest"], cal)
        radii = {name: float(weighted_quantile(np.abs(cal_y - value), cal_weights, .9))
                 for name, value in cal_pred.items()}

        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        predictions = model.predict(models, query)
        predictions["ridge_latest"] = ridge_predict(models["ridge_latest"], query)
        pred = pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                             "input_thermal": query.thermal.to_numpy(float), **predictions})
        path = out / f"{target.upper()}_heldout_predictions.csv.gz"
        pred.to_csv(path, index=False, compression="gzip")
        rp = out / f"{target.upper()}_source_radii.json"
        rp.write_text(json.dumps(radii, indent=2) + "\n")
        sealed.append({"target": target.upper(), "path": path.name, "sha256": sha(path),
                       "radii_path": rp.name, "radii_sha256": sha(rp), "hours": len(query)})
        radii_by_target[target] = radii
        print(f"sealed {target.upper()}: {len(query)} hours", flush=True)

    seal_path = out / "prediction_seal.json"
    seal_path.write_text(json.dumps(sealed, indent=2) + "\n")
    if any(sha(Path(path)) != digest for key in ("code", "inputs")
           for path, digest in hashes[key].items()):
        raise RuntimeError("code, protocol, or inputs changed during run")
    if any(sha(out / row[key]) != row[hash_key] for row in sealed
           for key, hash_key in (("path", "sha256"), ("radii_path", "radii_sha256"))):
        raise RuntimeError("prediction seal verification failed")

    rows = []
    for target in COUNTRIES:
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        fuel = base.fuel_for(raw, query)
        truth = fuel @ model.FACTORS / fuel.sum(axis=1)
        saved = pd.read_csv(out / f"{target.upper()}_heldout_predictions.csv.gz")
        strata = {"all": np.ones(len(query), dtype=bool),
                  "thermal_ge_0.5": query.thermal.to_numpy() >= .5,
                  "thermal_lt_0.5": query.thermal.to_numpy() < .5}
        for method, radius in radii_by_target[target].items():
            prediction = saved[method].to_numpy(float)
            for stratum, stratum_mask in strata.items():
                for month in sorted(query.clock.dt.month.unique()):
                    mask = stratum_mask & query.clock.dt.month.eq(month).to_numpy()
                    if not mask.any():
                        continue
                    rows.append(score_group(method, target, stratum, int(month), query.loc[mask],
                                            truth[mask], prediction[mask], radius))
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    country = monthly.groupby(["target_country", "stratum", "method"])[metrics].mean().reset_index()
    country.to_csv(out / "country_strata_metrics.csv", index=False)
    macro = country.groupby(["stratum", "method"])[metrics].mean().reset_index()
    macro.to_csv(out / "macro_strata_metrics.csv", index=False)
    macro.loc[macro.stratum.eq("all")].drop(columns="stratum").to_csv(out / "macro_metrics.csv", index=False)
    (out / "status.json").write_text(json.dumps({
        "status": "DEVELOPMENT_NINE_COUNTRY_STRICT_CN5_LOCO",
        "countries": [c.upper() for c in COUNTRIES], "training_year": 2022,
        "source_calibration": "2023H1", "target_year": 2024,
        "candidate": "weighted StandardScaler + Ridge(alpha=10), LATEST features only",
        "label": "strict-CN5 fixed-factor generation proxy; other renewables in denominator",
        "target_labels_used_before_prediction_seal": False, "unseen_confirmation": False,
    }, indent=2) + "\n")
    print(f"scored {len(monthly)} country/method/month/stratum rows", flush=True)


if __name__ == "__main__":
    main()
