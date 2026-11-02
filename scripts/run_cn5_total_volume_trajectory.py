"""Strict CN5 LOCO test of lagged total-generation level changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_energy_charts_nine_country_cn5_loco as loader  # noqa: E402
import run_cn5_fuel_resolved_response as fuel_model  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = list(loader.COUNTRIES)
PROTOCOL = ROOT / "docs/CN5_TOTAL_VOLUME_TRAJECTORY_PROTOCOL_20261101.md"
FEATURES = ("log_total_slope6", "log_total_recent_shift3")
METHODS = ("anchor700", "ratio_recomposition", "factor_history", "fuelshare_hgb",
           "factor_volume_trajectory", "fuelshare_volume_trajectory")
FACTORS = fuel_model.FUEL_FACTORS
METRICS = ("mae", "shape_mae", "coverage90", "width90", "interval_score90")
OPTIONS = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
               l2_regularization=10., random_state=416, early_stopping=False)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def augment(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    months = model.public_months(raw)
    period = raw.clock.dt.to_period("M")
    hours = raw.groupby(period).expected_hours.first().reindex(months.index)
    mean_total = months.sum(axis=1).div(hours)
    if (mean_total <= 0).any() or mean_total.isna().any():
        raise ValueError("monthly total generation must be positive and complete")
    log_total = np.log(mean_total)
    trajectory: dict[pd.Period, dict[str, float]] = {}
    for p in features.clock.dt.to_period("M").unique():
        history = log_total.loc[(log_total.index >= p - 7) & (log_total.index <= p - 2)]
        if len(history) != 6 or not history.index.is_monotonic_increasing:
            continue
        y = history.to_numpy(float)
        trajectory[p] = {
            FEATURES[0]: float(np.polyfit(np.arange(6, dtype=float), y, 1)[0]),
            FEATURES[1]: float(y[3:].mean() - y[:3].mean()),
        }
    periods = features.clock.dt.to_period("M")
    out = features.copy()
    for name in FEATURES:
        out[name] = periods.map({p: x[name] for p, x in trajectory.items()}).to_numpy()
    return out


def collect(prepared: dict, countries: list[str], year: int,
            months: range | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    parts = []
    for country in countries:
        frame = prepared[country][1].loc[prepared[country][1].year.eq(year)]
        frame = frame.loc[frame[list(FEATURES)].notna().all(axis=1)]
        if months is not None:
            frame = frame.loc[frame.clock.dt.month.isin(months)]
        if len(frame):
            parts.append((country, frame))
    combined = pd.concat([f for _, f in parts], ignore_index=True)
    fine = np.concatenate([fuel_model.fuel_for(prepared[c][0], f) for c, f in parts])
    coarse = np.concatenate([model.coarse_values(
        prepared[c][0].iloc[f.raw_row.to_numpy()]) for c, f in parts])
    return combined, fine, coarse


def fit_volume_heads(frame: pd.DataFrame, fine: np.ndarray) -> dict:
    thermal = fine[:, :4].sum(axis=1)
    active = thermal > 0
    x = frame.loc[active, [*model.HISTORY, *FEATURES]]
    weights = model.weights(frame.loc[active]).astype(float)
    weights /= weights.mean()
    shares = fine[active, :4] / thermal[active, None]
    heads = []
    for k in range(4):
        head = HistGradientBoostingRegressor(**OPTIONS)
        head.fit(x, shares[:, k], sample_weight=weights)
        heads.append(head)
    prior = np.average(shares, axis=0, weights=weights)
    return {"heads": heads, "prior": prior / prior.sum()}


def predict_volume_shares(head: dict, frame: pd.DataFrame) -> np.ndarray:
    raw = np.column_stack([h.predict(frame[[*model.HISTORY, *FEATURES]])
                           for h in head["heads"]])
    raw = np.maximum(raw, 0.)
    total = raw.sum(axis=1)
    return np.divide(raw, total[:, None], out=np.tile(head["prior"], (len(raw), 1)),
                     where=total[:, None] > 0.)


def thermal_share(frame: pd.DataFrame, profile: np.ndarray) -> np.ndarray:
    q = frame[list(model.GROUPS)].to_numpy(float) * profile[frame.clock.dt.hour.to_numpy()]
    total = q.sum(axis=1)
    return np.divide(q[:, 0], total, out=np.zeros(len(q)), where=total > 0.)


def run_fold(target: str, prepared: dict, out: Path) -> dict:
    sources = [c for c in COUNTRIES if c != target]
    train, fine, coarse = collect(prepared, sources, 2022)
    base = model.fit(train, fine, coarse)
    base_shares = fuel_model.fit_fuelshare(train, fine)
    thermal = fine[:, :4].sum(axis=1)
    active = thermal > 0
    thermal_ef = np.divide(fine @ model.FACTORS, thermal,
                           out=np.full(len(fine), 400.), where=active)
    x = train.loc[active, [*model.HISTORY, *FEATURES]]
    factor = HistGradientBoostingRegressor(**OPTIONS)
    factor.fit(x, thermal_ef[active],
               sample_weight=model.weights(train.loc[active]) * len(train))
    volume_shares = fit_volume_heads(train, fine)

    cal, cal_fine, _ = collect(prepared, sources, 2023, range(7, 13))
    cal_truth = cal_fine @ model.FACTORS / cal_fine.sum(axis=1)
    weights = model.weights(cal)
    cal_pred = model.predict(base, cal)
    cal_q = thermal_share(cal, base["profile"])
    cal_pred["fuelshare_hgb"] = fuel_model.predict_fuelshare(base_shares, base, cal)
    cal_pred["factor_volume_trajectory"] = np.clip(
        factor.predict(cal[[*model.HISTORY, *FEATURES]]), 400., 1000.) * cal_q
    cal_pred["fuelshare_volume_trajectory"] = (
        predict_volume_shares(volume_shares, cal) @ FACTORS) * cal_q
    radii = {m: float(weighted_quantile(np.abs(cal_truth - p), weights, .9))
             for m, p in cal_pred.items() if m in METHODS}

    raw, features = prepared[target]
    query = features.loc[features.year.eq(2024) &
                         features[list(FEATURES)].notna().all(axis=1)].copy()
    point = model.predict(base, query)
    q = thermal_share(query, base["profile"])
    point["fuelshare_hgb"] = fuel_model.predict_fuelshare(base_shares, base, query)
    point["factor_volume_trajectory"] = np.clip(
        factor.predict(query[[*model.HISTORY, *FEATURES]]), 400., 1000.) * q
    point["fuelshare_volume_trajectory"] = (predict_volume_shares(volume_shares, query) @ FACTORS) * q
    base_comp = np.column_stack([np.maximum(h.predict(query[list(model.HISTORY)]), 0.)
                                 for h in base_shares["heads"]])
    base_comp = np.divide(base_comp, base_comp.sum(axis=1)[:, None],
        out=np.tile(base_shares["prior"], (len(query), 1)),
        where=base_comp.sum(axis=1)[:, None] > 0.)
    vol_comp = predict_volume_shares(volume_shares, query)
    for i, fuel in enumerate(("coal", "gas", "oil", "other")):
        point[f"base_share_{fuel}"] = base_comp[:, i]
        point[f"volume_share_{fuel}"] = vol_comp[:, i]
    pred_path = out / f"{target}_predictions.csv.gz"
    pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                  **point}).to_csv(pred_path, index=False, compression="gzip")
    radius_path = out / f"{target}_source_radii.json"
    radius_path.write_text(json.dumps(radii, indent=2) + "\n")
    full = int(features.year.eq(2024).sum())
    return {"target": target, "prediction_file": pred_path.name,
            "radius_file": radius_path.name, "scored_hours": len(query),
            "available_hours": full, "scored_fraction": len(query) / full,
            "sources": sources}


def score(target: str, prepared: dict, out: Path, selection: dict):
    raw, features = prepared[target]
    query = features.loc[features.year.eq(2024) &
                         features[list(FEATURES)].notna().all(axis=1)].copy()
    pred = pd.read_csv(out / selection["prediction_file"])
    radii = json.loads((out / selection["radius_file"]).read_text())
    fine = fuel_model.fuel_for(raw, query)
    truth = fuel_model.truth_for(raw, query)
    active = fine[:, :4].sum(axis=1) > 0
    true_shares = np.divide(fine[:, :4], fine[:, :4].sum(axis=1)[:, None],
                            out=np.zeros((len(fine), 4)),
                            where=fine[:, :4].sum(axis=1)[:, None] > 0.)
    rows = []
    for method, radius in radii.items():
        yhat = pred[method].to_numpy(float)
        err = np.abs(truth - yhat)
        daily = pd.DataFrame({"y": truth, "p": yhat,
                              "d": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
        centered = daily[["y", "p"]] - daily.groupby("d")[["y", "p"]].transform("mean")
        for month in sorted(query.clock.dt.month.unique()):
            m = query.clock.dt.month.eq(month).to_numpy()
            e = err[m]
            rows.append({"target_country": target.upper(), "method": method,
                "month": int(month), "hours": int(m.sum()), "mae": float(e.mean()),
                "shape_mae": float(np.abs(centered.y.to_numpy()[m] -
                                           centered.p.to_numpy()[m]).mean()),
                "coverage90": float((e <= radius).mean()), "width90": 2 * radius,
                "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(e-radius, 0)))})
    mechanism = []
    for model_name, names in (("fuelshare_hgb", "base_share"),
                              ("fuelshare_volume_trajectory", "volume_share")):
        shares = pred[[f"{names}_{f}" for f in ("coal", "gas", "oil", "other")]].to_numpy()
        for i, f in enumerate(("coal", "gas", "oil", "other")):
            mechanism.append({"target_country": target.upper(), "model": model_name,
                "target": f, "mae": float(np.abs(true_shares[active, i] - shares[active, i]).mean())})
        mechanism.append({"target_country": target.upper(), "model": model_name,
            "target": "thermal_effective_ef", "mae": float(np.abs(
                true_shares[active] @ FACTORS - shares[active] @ FACTORS).mean())})
    return rows, mechanism


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs, code = [PROTOCOL], [Path(__file__).resolve(), PROTOCOL,
        Path(loader.__file__).resolve(), Path(loader.six_group.__file__).resolve(),
        Path(loader.six_group.base.__file__).resolve(), Path(fuel_model.__file__).resolve(),
        Path(model.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
        ROOT / "src/transcif/research/monthly_adaptation.py"]
    for c in COUNTRIES:
        folder, manifest, _ = loader.six_group.data_locations(c)
        inputs += [manifest, folder / f"{c}_2022_2024.json"]
        if c in loader.six_group.OLD_COUNTRIES:
            inputs.append(folder / "audit.json")
    hashes = {"inputs": {str(p.resolve()): sha(p) for p in sorted(set(inputs))},
              "code": {str(p.resolve()): sha(p) for p in sorted(set(code))}}
    (out / "startup_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    prepared = {}
    for c in COUNTRIES:
        raw, features, _ = loader.prepare_country(c)
        prepared[c] = raw, augment(raw, features)
    selections = []
    for target in COUNTRIES:
        selections.append(run_fold(target, prepared, out))
        print(f"sealed {target.upper()}", flush=True)
    seals = []
    for selection in selections:
        for key in ("prediction_file", "radius_file"):
            path = out / selection[key]
            seals.append({"target": selection["target"], "file": path.name,
                          "sha256": sha(path), "kind": key})
    (out / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    for group in ("inputs", "code"):
        for path, expected in hashes[group].items():
            if sha(Path(path)) != expected:
                raise RuntimeError(f"{group} changed during run: {path}")
    for seal in seals:
        if sha(out / seal["file"]) != seal["sha256"]:
            raise RuntimeError(f"prediction seal mismatch: {seal['file']}")
    rows, mechanisms = [], []
    for target, selection in zip(COUNTRIES, selections):
        scored, mechanism = score(target, prepared, out, selection)
        rows += scored
        mechanisms += mechanism
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    country = monthly.groupby(["target_country", "method"])[list(METRICS)].mean().reset_index()
    country.to_csv(out / "country_metrics.csv", index=False)
    macro = country.groupby("method")[list(METRICS)].mean().reset_index()
    macro.to_csv(out / "macro_metrics.csv", index=False)
    pd.DataFrame(mechanisms).to_csv(out / "mechanism_metrics.csv", index=False)
    summary = {"status": "DEVELOPMENT_STRICT_CN5_TOTAL_VOLUME_TRAJECTORY",
        "features": list(FEATURES), "countries": [c.upper() for c in COUNTRIES],
        "macro_metrics": macro.set_index("method").to_dict(orient="index"),
        "limitations": ["All countries and target years were previously viewed.",
            "Monthly inputs are reconstructed from revised hourly series.",
            "Labels are fixed-factor generation-side proxies, not measured CIF.",
            "Trajectory features use only complete six-month coarse histories."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(macro.to_string(index=False), flush=True)
    print(f"completed: {out}", flush=True)


if __name__ == "__main__":
    main()
