"""Nested CN5 pseudo-target transfer of monthly mean residual corrections."""
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as fuel_model  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_PSEUDOTARGET_MONTHLY_LEVEL_RESIDUAL_PROTOCOL_20261102.md"
MONTHLY_INPUTS = (*model.GROUPS, "thermal_mean", "thermal_std", "thermal_change",
                  "total_cv", "observations")
CALENDAR_INPUTS = ("month_sin", "month_cos")
ALPHAS = (1.0, 10.0, 100.0)
FAMILIES = ("season_only", "cn5_monthly")
METHODS = ("anchor700", "ratio_recomposition", "fuelshare_hgb",
           "season_level_correction", "cn5_level_correction")
SOURCE_MODEL_CACHE: dict[tuple[str, ...], dict] = {}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_all() -> tuple[dict, dict]:
    prepared, audits = {}, {}
    for country in nine.COUNTRIES:
        raw, features, audits[country] = nine.prepare_country(country)
        prepared[country] = raw, features
    return prepared, audits


def source_fit(prepared: dict, countries: list[str]) -> dict:
    key = tuple(countries)
    if key in SOURCE_MODEL_CACHE:
        return SOURCE_MODEL_CACHE[key]
    parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in countries]
    train = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([nine.fuel_for(prepared[c][0], p)
                           for c, p in zip(countries, parts)])
    coarse = np.concatenate([
        model.coarse_values(prepared[c][0].iloc[p.raw_row.to_numpy()])
        for c, p in zip(countries, parts)
    ])
    if train.empty or len(train) != len(fuel) or len(train) != len(coarse):
        raise ValueError(f"invalid source rows for {countries}")
    base = model.fit(train, fuel, coarse)
    base["fuelshare_head"] = fuel_model.fit_fuelshare(train, fuel)
    SOURCE_MODEL_CACHE[key] = base
    return base


def source_predict(fitted: dict, query: pd.DataFrame) -> dict[str, np.ndarray]:
    points = model.predict(fitted, query)
    points["fuelshare_hgb"] = fuel_model.predict_fuelshare(
        fitted["fuelshare_head"], fitted, query)
    return points


def target_part(prepared: dict, country: str, year: int, half: int):
    raw, features = prepared[country]
    query = features.loc[features.year.eq(year)].copy()
    if half == 1:
        query = query.loc[query.clock.dt.month.le(6)].copy()
    elif half == 2:
        query = query.loc[query.clock.dt.month.gt(6)].copy()
    if query.empty:
        raise ValueError(f"empty pseudo-target support: {country} {year} H{half}")
    fuel = nine.fuel_for(raw, query)
    truth = fuel @ model.FACTORS / fuel.sum(axis=1)
    return raw, query, truth


def make_month_features(query: pd.DataFrame) -> pd.DataFrame:
    columns = [*MONTHLY_INPUTS]
    rows = query.groupby(query.clock.dt.to_period("M"), sort=True)[columns].first()
    month = rows.index.month.to_numpy(float)
    rows["month_sin"] = np.sin(2 * np.pi * month / 12.)
    rows["month_cos"] = np.cos(2 * np.pi * month / 12.)
    rows.index.name = "period"
    if rows.isna().any().any():
        raise ValueError("missing legal monthly inputs in scored feature support")
    return rows


def monthly_residual_table(country: str, query: pd.DataFrame, truth: np.ndarray,
                           prediction: np.ndarray) -> pd.DataFrame:
    frame = make_month_features(query)
    work = pd.DataFrame({"period": query.clock.dt.to_period("M").to_numpy(),
                         "residual": truth - prediction})
    residual = work.groupby("period", sort=True).residual.mean()
    frame["country"] = country
    frame["residual"] = residual.reindex(frame.index).to_numpy(float)
    frame["n_hours"] = work.groupby("period").size().reindex(frame.index).to_numpy(int)
    if frame.residual.isna().any():
        raise ValueError(f"missing residual months for {country}")
    return frame.reset_index()


def columns_for(family: str) -> tuple[str, ...]:
    return CALENDAR_INPUTS if family == "season_only" else (*MONTHLY_INPUTS, *CALENDAR_INPUTS)


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("country").country.transform("size").to_numpy(float)
    return 1. / counts


def fit_residual_model(frame: pd.DataFrame, family: str, alpha: float) -> tuple[StandardScaler, Ridge]:
    columns = list(columns_for(family))
    x = frame[columns].to_numpy(float)
    y = frame.residual.to_numpy(float)
    w = sample_weights(frame)
    scaler = StandardScaler().fit(x, sample_weight=w)
    estimator = Ridge(alpha=alpha)
    estimator.fit(scaler.transform(x), y, sample_weight=w * len(w))
    return scaler, estimator


def predict_residual(frame: pd.DataFrame, fitted: tuple[StandardScaler, Ridge],
                     family: str) -> np.ndarray:
    scaler, estimator = fitted
    return estimator.predict(scaler.transform(frame[list(columns_for(family))].to_numpy(float)))


def tune_family(frame: pd.DataFrame, family: str) -> tuple[float, dict]:
    scores = {}
    countries = sorted(frame.country.unique())
    if len(countries) < 3:
        raise ValueError("too few pseudo-target countries for nested tuning")
    for alpha in ALPHAS:
        heldout_losses = []
        for heldout in countries:
            train = frame.loc[frame.country.ne(heldout)]
            test = frame.loc[frame.country.eq(heldout)]
            fitted = fit_residual_model(train, family, alpha)
            pred = predict_residual(test, fitted, family)
            heldout_losses.append(float(np.mean(np.abs(test.residual.to_numpy() - pred))))
        scores[str(alpha)] = {"country_mae": heldout_losses,
                              "macro_month_residual_mae": float(np.mean(heldout_losses))}
    alpha = min(ALPHAS, key=lambda a: (scores[str(a)]["macro_month_residual_mae"], a))
    return alpha, scores


def nested_validate_family(frames_by_holdout: dict[str, pd.DataFrame],
                           family: str) -> tuple[float, list[dict]]:
    """Evaluate on countries excluded from residual generation, tuning, and fit."""
    rows = []
    for heldout in sorted(frames_by_holdout):
        frame = frames_by_holdout[heldout]
        train = frame.loc[frame.country.ne(heldout)]
        test = frame.loc[frame.country.eq(heldout)]
        alpha, _ = tune_family(train, family)
        fitted = fit_residual_model(train, family, alpha)
        prediction = predict_residual(test, fitted, family)
        rows.append({"country": heldout, "alpha_selected_without_country": alpha,
                     "month_residual_mae": float(np.mean(np.abs(
                         test.residual.to_numpy(float) - prediction)))})
    return float(np.mean([row["month_residual_mae"] for row in rows])), rows


def fit_predict_months(train: pd.DataFrame, test: pd.DataFrame, family: str,
                       alpha: float) -> pd.Series:
    fitted = fit_residual_model(train, family, alpha)
    values = predict_residual(test, fitted, family)
    return pd.Series(values, index=pd.PeriodIndex(test.period, freq="M"))


def expand_to_hours(query: pd.DataFrame, month_values: pd.Series) -> np.ndarray:
    periods = query.clock.dt.to_period("M")
    result = periods.map(month_values).to_numpy(float)
    if not np.isfinite(result).all():
        raise ValueError("monthly residual forecast did not cover all target hours")
    return result


def oof_source_tables(prepared: dict, sources: list[str], outer: str,
                      heldout: str | None = None) -> pd.DataFrame:
    h1_tables = []
    for pseudo in sources:
        pseudo_sources = [c for c in sources if c != pseudo and c != heldout]
        fitted = source_fit(prepared, pseudo_sources)
        _, q1, y1 = target_part(prepared, pseudo, 2023, 1)
        p1 = source_predict(fitted, q1)
        h1_tables.append(monthly_residual_table(pseudo, q1, y1, p1["fuelshare_hgb"]))
        print(f"cross-fit pseudo target {outer.upper()} <- {pseudo.upper()}", flush=True)
    return pd.concat(h1_tables, ignore_index=True)


def calibrate_radii(prepared: dict, h1_by_holdout: dict[str, pd.DataFrame],
                    sources: list[str]) -> tuple[dict, list[dict]]:
    residuals = {m: [] for m in METHODS}
    frames = []
    calibration_rows = []
    for pseudo in sources:
        h1 = h1_by_holdout[pseudo]
        train_months = h1.loc[h1.country.ne(pseudo)]
        _, q, truth = target_part(prepared, pseudo, 2023, 2)
        pseudo_base = source_fit(prepared, [c for c in sources if c != pseudo])
        pred = source_predict(pseudo_base, q)
        pred = {m: pred[m] for m in ("anchor700", "ratio_recomposition", "fuelshare_hgb")}
        corrections = {}
        for family in FAMILIES:
            alpha, cv = tune_family(train_months, family)
            month_frame = make_month_features(q).reset_index()
            vals = fit_predict_months(train_months, month_frame, family, alpha)
            correction = expand_to_hours(q, vals)
            name = "season_level_correction" if family == "season_only" else "cn5_level_correction"
            pred[name] = pred["fuelshare_hgb"] + correction
            corrections[name] = {"alpha": alpha, "cv": cv}
        frames.append(q)
        for method in METHODS:
            residuals[method].append(np.abs(truth - pred[method]))
        calibration_rows.append({"pseudo_target": pseudo.upper(),
                                 "hours": len(q), "correction_models": corrections})
    calibration = pd.concat(frames, ignore_index=True)
    weights = model.weights(calibration)
    radii = {method: float(weighted_quantile(np.concatenate(values), weights, .90))
             for method, values in residuals.items()}
    return radii, calibration_rows


def score_months(target: str, query: pd.DataFrame, truth: np.ndarray,
                 points: dict[str, np.ndarray], radii: dict[str, float]) -> list[dict]:
    rows = []
    month_numbers = query.clock.dt.month
    for method in METHODS:
        error = truth - points[method]
        daily = pd.DataFrame({"truth": truth, "pred": points[method],
                              "day": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
        centered = daily[["truth", "pred"]] - daily.groupby("day")[["truth", "pred"]].transform("mean")
        radius = radii[method]
        for month in sorted(month_numbers.unique()):
            mask = month_numbers.eq(month).to_numpy()
            e = np.abs(error[mask])
            rows.append({"target_country": target.upper(), "method": method,
                         "month": int(month), "hours": int(mask.sum()),
                         "mae": float(e.mean()),
                         "shape_mae": float(np.abs(centered.truth.to_numpy()[mask] -
                                                    centered.pred.to_numpy()[mask]).mean()),
                         "coverage90": float((e <= radius).mean()), "width90": 2 * radius,
                         "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(e-radius, 0.)))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(nine.__file__).resolve(),
                  Path(fuel_model.__file__).resolve(),
                  Path(nine.six_group.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    input_paths = []
    for country in nine.COUNTRIES:
        folder, manifest, _ = nine.six_group.data_locations(country)
        input_paths.extend([manifest, folder / f"{country}_2022_2024.json"])
        if country in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    (out / "startup_hashes.json").write_text(json.dumps(
        {"code_sha256": code_hashes, "input_sha256": input_hashes}, indent=2) + "\n")

    prepared, audits = prepare_all()
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")
    monthly_rows, selections, calibrations, seals = [], [], [], []

    for target in nine.COUNTRIES:
        sources = [c for c in nine.COUNTRIES if c != target]
        h1_table = oof_source_tables(prepared, sources, target)
        h1_by_holdout = {
            pseudo: oof_source_tables(prepared, sources, target, pseudo)
            for pseudo in sources
        }
        selected = {}
        selection_scores = {}
        for family in FAMILIES:
            alpha, cv = tune_family(h1_table, family)
            selected[family], selection_scores[family] = alpha, cv
        season_alpha = selected["season_only"]
        cn5_alpha = selected["cn5_monthly"]
        season_cv, season_nested = nested_validate_family(h1_by_holdout, "season_only")
        cn5_cv, cn5_nested = nested_validate_family(h1_by_holdout, "cn5_monthly")
        season_country_cv = [row["month_residual_mae"] for row in season_nested]
        cn5_country_cv = [row["month_residual_mae"] for row in cn5_nested]
        zero_cv = float(np.mean([
            np.mean(np.abs(h1_table.loc[h1_table.country.eq(country), "residual"].to_numpy()))
            for country in sorted(h1_table.country.unique())]))
        folds_nonworse = int(np.sum(np.asarray(cn5_country_cv) <=
                                    1.05 * np.asarray(season_country_cv)))
        source_gate_pass = cn5_cv <= 0.95 * season_cv and folds_nonworse >= 6

        radii, cal_rows = calibrate_radii(prepared, h1_by_holdout, sources)
        calibrations.append({"target": target.upper(), "radii90": radii,
                             "pseudo_target_calibration": cal_rows})
        selections.append({"target": target.upper(), "selected_family": "cn5_monthly",
                           "selected_alpha": cn5_alpha,
                           "nested_h1_cv": selection_scores,
                           "outer_nested_validation": {"season_only": season_nested,
                                                       "cn5_monthly": cn5_nested},
                           "zero_correction_h1_cv_mae": zero_cv,
                           "season_only_h1_cv_month_residual_mae": season_cv,
                           "cn5_monthly_h1_cv_month_residual_mae": cn5_cv,
                           "cn5_relative_gain_vs_season_pct": 100 * (season_cv-cn5_cv)/season_cv,
                           "cn5_nonworse_country_folds_at_5pct": folds_nonworse,
                           "source_promotion_gate_pass": bool(source_gate_pass)})

        raw, query = prepared[target]
        query = query.loc[query.year.eq(2024)].copy()
        fitted_base = source_fit(prepared, sources)
        base = source_predict(fitted_base, query)
        points = {m: base[m] for m in ("anchor700", "ratio_recomposition", "fuelshare_hgb")}
        month_frame = make_month_features(query).reset_index()
        for family, name in (("season_only", "season_level_correction"),
                             ("cn5_monthly", "cn5_level_correction")):
            alpha, _ = tune_family(h1_table, family)
            vals = fit_predict_months(h1_table, month_frame, family, alpha)
            points[name] = base["fuelshare_hgb"] + expand_to_hours(query, vals)
        radius_path = out / f"{target.upper()}_source_radii.json"
        radius_path.write_text(json.dumps(radii, indent=2) + "\n")
        radius_digest = sha(radius_path)
        prediction_path = out / f"{target.upper()}_2024_predictions.csv.gz"
        times = raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()
        pd.DataFrame({"time": times, **points}).to_csv(
            prediction_path, index=False, compression="gzip")
        seal = {"target": target.upper(), "sources": [c.upper() for c in sources],
                "prediction_file": prediction_path.name, "prediction_sha256": sha(prediction_path),
                "radius_file": radius_path.name, "radius_sha256": radius_digest,
                "rows": len(query), "selected_family": "cn5_monthly", "selected_alpha": cn5_alpha}
        seals.append(seal)
        (out / "prediction_seals.json").write_text(json.dumps(seals, indent=2) + "\n")
        (out / "source_calibration.json").write_text(json.dumps(calibrations, indent=2) + "\n")
        if (sha(prediction_path) != seal["prediction_sha256"] or
                sha(radius_path) != seal["radius_sha256"]):
            raise RuntimeError("outer prediction/calibration seal mismatch")
        sealed_points = pd.read_csv(prediction_path)
        if not np.array_equal(sealed_points.time.to_numpy(), times):
            raise RuntimeError("sealed prediction timestamps changed")
        points = {method: sealed_points[method].to_numpy(float) for method in METHODS}
        radii = json.loads(radius_path.read_text())

        # Outer 2024 labels are opened only after all points and source radii are sealed.
        truth_fuel = nine.fuel_for(raw, query)
        truth = truth_fuel @ model.FACTORS / truth_fuel.sum(axis=1)
        monthly_rows.extend(score_months(target, query, truth, points, radii))
        print(f"scored {target.upper()} after sealing {len(query)} predictions", flush=True)

    if any(sha(Path(p)) != digest for p, digest in code_hashes.items()):
        raise RuntimeError("frozen code/protocol changed during run")
    if any(sha(Path(p)) != digest for p, digest in input_hashes.items()):
        raise RuntimeError("source inputs changed during run")
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    country = monthly.groupby(["target_country", "method"])[
        ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]].mean().reset_index()
    country.to_csv(out / "country_metrics.csv", index=False)
    country.groupby("method")[
        ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]].mean().to_csv(
            out / "macro_metrics.csv")
    (out / "source_selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    (out / "source_calibration.json").write_text(json.dumps(calibrations, indent=2) + "\n")
    (out / "prediction_seals.json").write_text(json.dumps(seals, indent=2) + "\n")
    (out / "run_manifest.json").write_text(json.dumps({
        "status": "DEVELOPMENT_ONLY_PREVIOUSLY_VIEWED_2024_TARGETS",
        "protocol_sha256": sha(PROTOCOL), "code_sha256": code_hashes,
        "input_sha256": input_hashes, "outer_labels_used_before_prediction": False,
        "target_year": 2024, "train_year": 2022, "source_h1_month_model": "nested pseudo-target cross-fit",
        "calibration": "country-cross-fitted pseudo-target 2023H2 q90",
        "budget": "strict five-group t-2 through t-7 public monthly mix plus target month",
        "target": "fixed-factor generation-side proxy, not measured CIF or consumption CIF",
    }, indent=2) + "\n")
    print(pd.read_csv(out / "macro_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
