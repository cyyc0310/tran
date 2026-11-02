#!/usr/bin/env python3
"""Strict CN5 LOCO: reconcile transferred hourly profiles to lagged monthly margins."""
from __future__ import annotations

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
PROTOCOL = ROOT / "docs/CN5_MONTHLY_MARGIN_RECONCILIATION_PROTOCOL_20261102.md"
METHODS = ("anchor700", "ratio_recomposition", "factor_history", "fuelshare_history",
           "monthly_margin_ipf")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ipf_month(prior: np.ndarray, row_margin: np.ndarray, column_margin: np.ndarray,
              *, tolerance: float = 1e-10, max_iter: int = 5000) -> tuple[np.ndarray, dict]:
    """KL-project a strictly positive prior onto prescribed row and column margins."""
    p = np.asarray(prior, dtype=float).copy()
    r = np.asarray(row_margin, dtype=float).copy()
    c = np.asarray(column_margin, dtype=float).copy()
    if p.ndim != 2 or len(r) != len(p) or len(c) != p.shape[1]:
        raise ValueError("IPF margin dimensions do not match prior")
    if not np.isfinite(p).all() or not np.isfinite(r).all() or not np.isfinite(c).all():
        raise ValueError("IPF inputs must be finite")
    if (p < 0).any() or (r < 0).any() or (c < 0).any() or r.sum() <= 0 or c.sum() <= 0:
        raise ValueError("IPF inputs must be nonnegative with positive totals")
    r /= r.sum()
    c /= c.sum()
    if not np.isclose(r.sum(), c.sum(), atol=1e-12, rtol=0):
        raise ValueError("row and column margins have different total mass")
    epsilon_cells = p <= 0
    p = np.maximum(p, 1e-12)
    p /= p.sum()
    for iteration in range(1, max_iter + 1):
        row_sum = p.sum(axis=1)
        p *= np.divide(r, row_sum, out=np.zeros_like(r), where=row_sum > 0)[:, None]
        col_sum = p.sum(axis=0)
        p *= np.divide(c, col_sum, out=np.zeros_like(c), where=col_sum > 0)[None, :]
        row_error = float(np.max(np.abs(p.sum(axis=1) - r)))
        col_error = float(np.max(np.abs(p.sum(axis=0) - c)))
        error = max(row_error, col_error)
        if error <= tolerance:
            return p, {"iterations": iteration, "max_margin_error": error,
                       "epsilon_cell_count": int(epsilon_cells.sum()),
                       "epsilon_mass_after_projection": float(p[epsilon_cells].sum())}
    raise RuntimeError(f"IPF did not converge after {max_iter} iterations; error={error}")


def predict(models: dict, public: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    out = model.predict(models, public)
    q = public[list(model.GROUPS)].to_numpy(float) * models["profile"][
        public.clock.dt.hour.to_numpy()]
    if not np.isfinite(q).all() or (q < 0).any() or (q.sum(axis=1) <= 0).any():
        raise ValueError("source recomposition prior has invalid hourly support")
    closed = np.zeros_like(q)
    diagnostics = []
    query = public.reset_index(drop=True)
    periods = query.clock.dt.to_period("M")
    if "region" not in query:
        raise ValueError("region identifiers are required for country-by-month reconciliation")
    keyed = query.assign(_period=periods)
    for (region, period), group in keyed.groupby(["region", "_period"], sort=True):
        positions = group.index.to_numpy(dtype=int)
        month = query.iloc[positions]
        mix = month.iloc[0][list(model.GROUPS)].to_numpy(float)
        if not np.isfinite(mix).all() or (mix < 0).any() or not np.isclose(mix.sum(), 1., atol=1e-9):
            raise ValueError(f"invalid lagged five-group margin in {period}")
        if not np.allclose(month[list(model.GROUPS)].to_numpy(float), mix[None, :],
                           atol=1e-12, rtol=0):
            raise ValueError(f"lagged monthly composition varies within {period}")
        hourly_total_prior = q[positions].sum(axis=1)
        matrix, audit = ipf_month(q[positions], hourly_total_prior, mix)
        closed[positions] = matrix
        diagnostics.append({"region": str(region), "period": str(period), "hours": len(positions),
                            "predicted_month_margin": mix.tolist(),
                            "reconciled_margin": matrix.sum(axis=0).tolist(),
                            "source_hourly_total_weight_cv": float(
                                hourly_total_prior.std() / hourly_total_prior.mean()), **audit})
    thermal_share = closed[:, 0] / closed.sum(axis=1)
    out["monthly_margin_ipf"] = np.clip(models["ratio_ef"] * thermal_share, 0., 1000.)
    return out, pd.DataFrame(diagnostics)


def score_month(query: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                radius: float) -> dict:
    error = truth - prediction
    daily = pd.DataFrame({"truth": truth, "prediction": prediction,
                          "day": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
    centered = daily[["truth", "prediction"]] - daily.groupby("day")[
        ["truth", "prediction"]].transform("mean")
    abs_error = np.abs(error)
    return {"hours": len(error), "mae": float(abs_error.mean()),
            "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
            "coverage90": float((abs_error <= radius).mean()), "width90": float(2 * radius),
            "winkler90": float(np.mean(2 * radius + 20 * np.maximum(abs_error - radius, 0.)))}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

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
    (output / "startup_hashes.json").write_text(json.dumps(
        {"input_sha256": input_hashes, "code_sha256": code_hashes}, indent=2) + "\n")

    prepared, audits = {}, {}
    for code in COUNTRIES:
        raw, features, audits[code] = nine.prepare_country(code)
        prepared[code] = raw, features
    (output / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")

    seals, radii_by_target, diagnostics_by_target = [], {}, {}
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        train_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([nine.fuel_for(prepared[c][0], part)
                                     for c, part in zip(sources, train_parts)])
        train_coarse = np.concatenate([model.coarse_values(prepared[c][0].iloc[
            part.raw_row.to_numpy()]) for c, part in zip(sources, train_parts)])
        if not len(train) or len(train) != len(train_fuel) or len(train) != len(train_coarse):
            raise ValueError(f"empty or misaligned training cohort for {target}")
        fitted = model.fit(train, train_fuel, train_coarse)
        fuel_head = response.fit_fuelshare(train, train_fuel)

        cal_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2023)
                                      & prepared[c][1].clock.dt.month.le(6)] for c in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([nine.fuel_for(prepared[c][0], part)
                                   for c, part in zip(sources, cal_parts)])
        cal_y = cal_fuel @ model.FACTORS / cal_fuel.sum(axis=1)
        cal_weights = model.weights(calibration)
        cal_points, cal_diagnostics = predict(fitted, calibration)
        cal_points["fuelshare_history"] = response.predict_fuelshare(
            fuel_head, fitted, calibration)
        radii = {name: float(weighted_quantile(np.abs(cal_y - value), cal_weights, .9))
                 for name, value in cal_points.items()}

        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        points, target_diagnostics = predict(fitted, query)
        points["fuelshare_history"] = response.predict_fuelshare(fuel_head, fitted, query)
        point_path = output / f"{target.upper()}_2024_predictions.csv.gz"
        pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                      **points}).to_csv(point_path, index=False, compression="gzip")
        radius_path = output / f"{target.upper()}_source_radii.json"
        radius_path.write_text(json.dumps(radii, indent=2) + "\n")
        diag_path = output / f"{target.upper()}_projection_audit.json"
        diag_path.write_text(json.dumps({"calibration": cal_diagnostics.to_dict("records"),
                                         "target": target_diagnostics.to_dict("records")},
                                        indent=2) + "\n")
        seals.append({"target": target.upper(), "prediction_file": point_path.name,
                      "prediction_sha256": sha(point_path), "radius_file": radius_path.name,
                      "radius_sha256": sha(radius_path), "projection_audit_file": diag_path.name,
                      "projection_audit_sha256": sha(diag_path), "hours": len(query),
                      "source_countries": [c.upper() for c in sources]})
        radii_by_target[target] = radii
        diagnostics_by_target[target] = {"calibration": cal_diagnostics, "target": target_diagnostics}
        print(f"sealed {target.upper()}: {len(query)} hours", flush=True)

    seal_path = output / "prediction_seal.json"
    seal_path.write_text(json.dumps(seals, indent=2) + "\n")
    if any(sha(Path(p)) != value for p, value in {**input_hashes, **code_hashes}.items()):
        raise RuntimeError("input, protocol, or code changed during run")
    for entry in seals:
        for file_key, hash_key in (("prediction_file", "prediction_sha256"),
                                   ("radius_file", "radius_sha256"),
                                   ("projection_audit_file", "projection_audit_sha256")):
            if sha(output / entry[file_key]) != entry[hash_key]:
                raise RuntimeError(f"sealed output hash mismatch: {entry[file_key]}")

    monthly_rows = []
    for target in COUNTRIES:
        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        truth_fuel = nine.fuel_for(raw, query)
        truth = truth_fuel @ model.FACTORS / truth_fuel.sum(axis=1)
        sealed = pd.read_csv(output / f"{target.upper()}_2024_predictions.csv.gz")
        for method in METHODS:
            radius = radii_by_target[target][method]
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                row = {"target_country": target.upper(), "method": method,
                       "month": int(month), **score_month(query.loc[mask], truth[mask],
                           sealed.loc[mask, method].to_numpy(float), radius)}
                monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "winkler90"]
    country = monthly.groupby(["target_country", "method"])[metrics].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    macro = country.groupby("method")[metrics].mean().reset_index()
    macro.to_csv(output / "macro_metrics.csv", index=False)
    status = {"status": "DEVELOPMENT_ONLY_STRICT_CN5_NINE_COUNTRY_LOCO",
              "target_year": 2024, "train_year": 2022, "calibration": "2023H1 source country equal",
              "target_budget": "five-group lagged t-2 through t-7 mix and calendar",
              "target_label": "fixed-factor generation-side hourly proxy",
              "target_outcomes_opened_after_prediction_and_radius_seals": True,
              "candidate": "KL/IPF projection of source-hourly profile to lagged t-2 monthly five-group margins",
              "not_confirmatory": ["unseen-region generalization", "measured CIF", "as-issued replay"]}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
