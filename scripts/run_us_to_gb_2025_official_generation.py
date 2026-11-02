#!/usr/bin/env python3
"""EIA-trained, CN5 t-2 budget forecast of NESO GB 2025 generation intensity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BAS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
FUELS = ("coal", "gas", "oil", "other", "nuclear", "hydro", "wind", "solar")
COARSE = ("thermal", "hydro", "nuclear", "wind", "solar")
METHODS = ("anchor700", "source_thermal_anchor", "calendar_ridge", "mix_ridge",
           "mix_hgb", "source_fuel_recomposition")
ZONES = {"BPAT": "America/Los_Angeles", "CISO": "America/Los_Angeles",
         "ERCO": "America/Chicago", "FPL": "America/New_York",
         "ISNE": "America/New_York", "MISO": "America/Chicago",
         "NYIS": "America/New_York"}
LB_TO_G = 453.59237
TARGET_FUELS = ROOT / "data/neso-gb/df_fuel_ckan.csv"
SOURCE_DIR = ROOT / "data/eia930-hourly-co2-r1"
TARGET_LABELS = ROOT / "docs/evidence/neso-national-intensity-2025-r1/neso_national_intensity_history.csv"
PROTOCOL = ROOT / "docs/CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_GENERATION_PROTOCOL_20261102.md"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def monthly_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach only exact t-2 five-group month shares and local calendar terms."""
    out = frame[["time", "clock", "region"]].copy()
    period = out.clock.dt.to_period("M")
    fuel = frame[list(FUELS)].copy()
    fuel["period"] = period.to_numpy()
    months = fuel.groupby("period")[list(FUELS)].sum()
    counts = fuel.groupby("period").size()
    expected = frame.assign(period=period.to_numpy()).groupby("period").expected_hours.first()
    months["coverage"] = counts / expected
    months = months.loc[months.coverage.ge(.99)].copy()
    months["thermal"] = months[["coal", "gas", "oil", "other"]].sum(axis=1)
    total = months[list(FUELS)].sum(axis=1)
    shares = months[list(COARSE)].div(total, axis=0)
    shares["observed_month"] = shares.index.astype(str)
    shares.index = shares.index + 2
    out["period"] = period.to_numpy()
    out = out.join(shares, on="period")
    clock = pd.to_datetime(out.clock)
    for name, values, n in [("hour", clock.dt.hour, 24), ("dow", clock.dt.dayofweek, 7),
                            ("month", clock.dt.month, 12)]:
        out[f"{name}_sin"] = np.sin(2*np.pi*values/n)
        out[f"{name}_cos"] = np.cos(2*np.pi*values/n)
    for fuel_name in COARSE:
        for suffix in ("sin", "cos"):
            out[f"{fuel_name}_hour_{suffix}"] = out[fuel_name] * out[f"hour_{suffix}"]
    return out


def features(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    names = [f"{part}_{suffix}" for part in ("hour", "dow", "month")
             for suffix in ("sin", "cos")]
    x = frame[names].copy()
    mix_cols = [f"m_{fuel}" for fuel in COARSE]
    for fuel in COARSE:
        x[f"m_{fuel}"] = frame[fuel].to_numpy()
    for fuel in COARSE:
        for suffix in ("sin", "cos"):
            name = f"{fuel}_hour_{suffix}"
            x[name] = frame[fuel].to_numpy() * frame[f"hour_{suffix}"].to_numpy()
    return x, tuple([*names, *mix_cols, *(f"{fuel}_hour_{s}" for fuel in COARSE for s in ("sin", "cos"))])


def load_sources() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    raw, labels, provenance = {}, {}, {}
    for ba in BAS:
        feature_path = SOURCE_DIR / f"feature_inputs/{ba}_generation_2022_2024.csv.gz"
        label_path = SOURCE_DIR / f"{ba}_official_intensity_2022_2024.csv.gz"
        fuel = pd.read_csv(feature_path)
        fuel["time"] = pd.to_datetime(fuel.time_utc, utc=True)
        values = fuel[[f"generation_{f}" for f in FUELS]].to_numpy(float)
        valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1) & (values.sum(axis=1) > 0)
        fuel = fuel.loc[valid].copy().reset_index(drop=True)
        fuel["time"] = pd.to_datetime(fuel.time_utc, utc=True)
        fuel["clock"] = fuel.time.dt.tz_convert(ZONES[ba]).dt.tz_localize(None)
        fuel["expected_hours"] = fuel.clock.map(
            lambda t: ((t.to_period("M") + 1).start_time.tz_localize(ZONES[ba])
                       - t.to_period("M").start_time.tz_localize(ZONES[ba])).total_seconds()/3600)
        fuel["region"] = ba
        fuel["country"] = "US"
        for name in FUELS:
            fuel[name] = fuel[f"generation_{name}"].astype(float)
        ci = pd.read_csv(label_path)
        ci["time"] = pd.to_datetime(ci["UTC time"], utc=True)
        ci["y"] = pd.to_numeric(ci["CO2 Emissions Intensity for Generated Electricity"], errors="coerce") * LB_TO_G
        ci = ci.loc[ci.y.notna() & ci.y.ge(0), ["time", "y"]]
        joined = fuel.merge(ci, on="time", how="inner", validate="one_to_one")
        if joined.empty:
            raise ValueError(f"no paired EIA fuel/official intensity rows for {ba}")
        raw[ba] = joined.drop(columns="y")
        labels[ba] = joined[["time", "y"]].copy()
        provenance[str(feature_path.resolve())] = sha(feature_path)
        provenance[str(label_path.resolve())] = sha(label_path)
    return raw, labels, provenance


def load_gb_inputs() -> tuple[pd.DataFrame, dict]:
    columns = ["DATETIME", "GAS", "COAL", "NUCLEAR", "WIND", "WIND_EMB", "HYDRO",
               "BIOMASS", "OTHER", "SOLAR"]
    raw = pd.read_csv(TARGET_FUELS, usecols=columns)
    # The archive's naive timestamps align to the UTC API intervals used by the
    # NESO-to-US official runner. Attach UTC before converting to the GB clock.
    stamp = pd.to_datetime(raw.DATETIME, errors="raise").dt.tz_localize("UTC")
    raw["time_utc"] = stamp
    raw["clock"] = stamp.dt.tz_convert("Europe/London").dt.tz_localize(None)
    raw["time"] = stamp
    raw["coal"], raw["gas"], raw["oil"] = raw.COAL, raw.GAS, 0.
    raw["other"] = raw.BIOMASS + raw.OTHER
    raw["nuclear"], raw["hydro"] = raw.NUCLEAR, raw.HYDRO
    raw["wind"], raw["solar"] = raw.WIND + raw.WIND_EMB, raw.SOLAR
    values = raw[list(FUELS)].to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    raw = raw.loc[valid].copy()
    raw["hour_utc"] = raw.time.dt.floor("h")
    count = raw.groupby("hour_utc").size()
    hourly = raw.groupby("hour_utc")[list(FUELS)].mean()
    hourly = hourly.loc[count.eq(2)].reset_index().rename(columns={"hour_utc": "time"})
    hourly["clock"] = hourly.time.dt.tz_convert("Europe/London").dt.tz_localize(None)
    hourly["region"], hourly["country"] = "GB", "GB"
    hourly["expected_hours"] = hourly.clock.map(lambda t: t.days_in_month * 24)
    hourly = hourly.sort_values("time").reset_index(drop=True)
    gb2025 = hourly.loc[hourly.clock.dt.year.eq(2025)].copy()
    months = sorted(gb2025.clock.dt.to_period("M").unique().astype(str).tolist())
    if len(gb2025) != 8760 or gb2025.time.duplicated().any() or len(months) != 12:
        raise ValueError(f"GB target generation support invalid: {len(gb2025)} hours, {len(months)} months")
    return hourly, {"fuel_file": str(TARGET_FUELS.resolve()), "fuel_sha256": sha(TARGET_FUELS),
                   "gb_hours": len(gb2025), "gb_target_months": months,
                   "two_halfhour_support_required": True,
                   "naive_archive_timestamp_interpretation": "UTC, empirically matched by the existing NESO source loader"}


def fit_models(source_raw: dict[str, pd.DataFrame], source_labels: dict[str, pd.DataFrame]) -> tuple[dict, dict, dict]:
    features_by_ba = {}
    train_rows, train_y, train_weights = [], [], []
    cal_parts: dict[str, dict[str, np.ndarray]] = {m: {} for m in METHODS}
    models = {}
    for ba in BAS:
        raw = source_raw[ba]
        inp = monthly_features(raw)
        inp["raw_row"] = inp.index
        usable = inp.dropna(subset=list(COARSE)).copy()
        usable = usable.join(raw.loc[usable.index, list(FUELS)].rename(
            columns={fuel: f"fuel_{fuel}" for fuel in FUELS}))
        x, mix_columns = features(usable)
        response = source_labels[ba].set_index("time").reindex(usable.time).y.to_numpy(float)
        train_mask = usable.clock.dt.year.le(2023).to_numpy()
        cal_mask = usable.clock.dt.year.eq(2024).to_numpy()
        if train_mask.sum() == 0 or cal_mask.sum() == 0 or not np.isfinite(response).all():
            raise ValueError(f"invalid training/calibration support for {ba}")
        train_rows.append((ba, usable.loc[train_mask].copy(), x.loc[train_mask].copy(), response[train_mask]))
        # Store BA-indexed source rows; all models are fit jointly below.
        features_by_ba[ba] = {"usable": usable, "x": x, "response": response,
                               "cal_mask": cal_mask, "train_mask": train_mask,
                               "mix_columns": mix_columns}

    # Preserve equal BA contribution in direct regression and fuel response fits.
    xtr = pd.concat([part[2].loc[part[1].clock.dt.year.le(2023)] for part in train_rows], ignore_index=True)
    ytr = np.concatenate([part[3] for part in train_rows])
    ba_ids = np.concatenate([[part[0]] * len(part[3]) for part in train_rows])
    counts = pd.Series(ba_ids).value_counts()
    w = np.array([1. / counts[ba] for ba in ba_ids], dtype=float)
    w *= len(w) / w.sum()
    calendar_cols = [f"{n}_{s}" for n in ("hour", "dow", "month") for s in ("sin", "cos")]
    mix_columns = features_by_ba[BAS[0]]["mix_columns"]
    models["calendar_ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=10.))
    models["calendar_ridge"].fit(xtr[calendar_cols], ytr, ridge__sample_weight=w)
    models["mix_ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=10.))
    models["mix_ridge"].fit(xtr[list(mix_columns)], ytr, ridge__sample_weight=w)
    models["mix_hgb"] = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=10., random_state=412)
    models["mix_hgb"].fit(xtr[list(mix_columns)], ytr, sample_weight=w)

    # Equal-BA source thermal factor and fuel response/profile.
    weighted_totals = 0.
    weighted_thermal = 0.
    coarse_parts, profile_parts = [], []
    for ba, train_frame, _, y in train_rows:
        fuels = train_frame[[f"fuel_{fuel}" for fuel in FUELS]].to_numpy(float)
        yy = y
        w_region = np.repeat(1. / len(yy), len(yy))
        total = fuels.sum(axis=1)
        thermal = fuels[:, :4].sum(axis=1)
        weighted_totals += float(np.sum(w_region * yy * total))
        weighted_thermal += float(np.sum(w_region * thermal))
        coarse = np.column_stack([thermal, fuels[:, 5], fuels[:, 4], fuels[:, 6], fuels[:, 7]])
        coarse_parts.append(coarse)
        hour = train_frame.clock.dt.hour.to_numpy()
        g = pd.DataFrame(coarse, columns=COARSE)
        g["hour"] = hour
        p = g.groupby("hour")[list(COARSE)].mean().reindex(range(24)).fillna(0.)
        profile_parts.append(p.div(p.mean().replace(0., np.nan)).fillna(1.).to_numpy())
    models["thermal_ef"] = weighted_totals / weighted_thermal
    models["profile"] = np.mean(profile_parts, axis=0)
    coarse_train = np.concatenate(coarse_parts)
    models["fuel_response"] = LinearRegression(positive=True, fit_intercept=False)
    models["fuel_response"].fit(coarse_train, ytr, sample_weight=w)

    # Source-only calibration. Bias is equal-BA; the absolute residual quantile
    # uses an equal-BA weighted empirical distribution.
    calibration = {}
    residuals = {method: {} for method in METHODS}
    for ba in BAS:
        entry = features_by_ba[ba]
        cmask = entry["cal_mask"]
        cal = entry["usable"].loc[cmask]
        cx = entry["x"].loc[cmask]
        y = entry["response"][cmask]
        preds = predict(models, cal, cx, mix_columns)
        for method in METHODS:
            residuals[method][ba] = y - preds[method]
    for method in METHODS:
        bias = float(np.mean([residuals[method][ba].mean() for ba in BAS]))
        vals, weights = [], []
        for ba in BAS:
            values = np.abs(residuals[method][ba] - bias)
            vals.extend(values.tolist())
            weights.extend([1. / len(BAS) / len(values)] * len(values))
        order = np.argsort(vals)
        cumulative = np.cumsum(np.asarray(weights)[order])
        radius = float(np.asarray(vals)[order[np.searchsorted(cumulative, .90, side="left")]])
        calibration[method] = {"bias": bias, "radius90": radius,
                               "calibration_hours_by_ba": {ba: len(residuals[method][ba]) for ba in BAS},
                               "calibration_year": 2024}
    audit = {"source_training_years": [2022, 2023], "source_calibration_year": 2024,
             "source_training_rows_by_ba": {ba: int(len(features_by_ba[ba]["x"].loc[
                 features_by_ba[ba]["usable"].clock.dt.year.le(2023)])) for ba in BAS},
             "source_thermal_ef_g_per_kwh": float(models["thermal_ef"]),
             "fuel_response_g_per_kwh": dict(zip(COARSE, models["fuel_response"].coef_.tolist())),
             "training_weights": "equal total weight per BA", "source_ba_count": len(BAS)}
    return models, calibration, audit


def predict(models: dict, frame: pd.DataFrame, x: pd.DataFrame,
            mix_columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    result = {"anchor700": 700. * frame.thermal.to_numpy(),
              "source_thermal_anchor": models["thermal_ef"] * frame.thermal.to_numpy()}
    calendar_cols = [f"{n}_{s}" for n in ("hour", "dow", "month") for s in ("sin", "cos")]
    for name, cols in [("calendar_ridge", calendar_cols), ("mix_ridge", mix_columns),
                       ("mix_hgb", mix_columns)]:
        result[name] = models[name].predict(x[list(cols)])
    q = frame[list(COARSE)].to_numpy(float) * models["profile"][frame.clock.dt.hour.to_numpy()]
    qsum = q.sum(axis=1, keepdims=True)
    q = np.divide(q, qsum, out=np.zeros_like(q), where=qsum > 0)
    result["source_fuel_recomposition"] = models["fuel_response"].predict(q)
    return {name: np.clip(value, 0., None) for name, value in result.items()}


def score(labels: pd.DataFrame, predictions: pd.DataFrame,
          calibration: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    joined = predictions.merge(labels, on="time", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("no official target label hours joined to sealed forecasts")
    for method in METHODS:
        bias = calibration[method]["bias"]
        radius = calibration[method]["radius90"]
        point = joined[method].to_numpy(float) + bias
        y = joined.y.to_numpy(float)
        error = y - point
        daily = pd.DataFrame({"y": y, "p": point,
                              "day": joined.clock.dt.strftime("%Y-%m-%d").to_numpy()})
        centered = daily[["y", "p"]] - daily.groupby("day")[["y", "p"]].transform("mean")
        rows.append({"method": method, "hours": len(joined), "mae_g_per_kwh": float(np.mean(np.abs(error))),
            "bias_g_per_kwh": float(np.mean(error)),
            "shape_mae_g_per_kwh": float(np.mean(np.abs(centered.y-centered.p))),
            "coverage90": float(np.mean(np.abs(error) <= radius)),
            "width90_g_per_kwh": float(2*radius),
            "interval_score90": float(np.mean(2*radius + 20*np.maximum(np.abs(error)-radius, 0.))),
            "target_mean_g_per_kwh": float(np.mean(y)), "target_median_g_per_kwh": float(np.median(y))})
    monthly = []
    for month, part in joined.groupby(joined.clock.dt.month):
        for method in METHODS:
            bias = calibration[method]["bias"]
            radius = calibration[method]["radius90"]
            error = part.y.to_numpy(float) - (part[method].to_numpy(float) + bias)
            monthly.append({"month": int(month), "method": method, "hours": len(part),
                "mae_g_per_kwh": float(np.mean(np.abs(error))),
                "bias_g_per_kwh": float(np.mean(error)),
                "coverage90": float(np.mean(np.abs(error) <= radius)),
                "width90_g_per_kwh": float(2*radius),
                "interval_score90": float(np.mean(2*radius + 20*np.maximum(np.abs(error)-radius, 0.)))})
    return pd.DataFrame(rows), pd.DataFrame(monthly)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-labels", type=Path, default=TARGET_LABELS)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    source_raw, source_y, source_hashes = load_sources()
    gb_raw, target_audit = load_gb_inputs()
    gb_inputs = monthly_features(gb_raw)
    target_mask = gb_inputs.clock.dt.year.eq(2025).to_numpy()
    target = gb_inputs.loc[target_mask].dropna(subset=list(COARSE)).copy()
    if len(target) != 8760 or target.observed_month.isna().any():
        raise ValueError(f"invalid GB t-2 feature cohort: {len(target)} rows")
    models, calibration, audit = fit_models(source_raw, source_y)
    target_x, mix_columns = features(target)
    raw_predictions = predict(models, target, target_x, mix_columns)
    prediction_frame = pd.DataFrame({"time": target.time, "clock": target.clock,
        "observed_month": target.observed_month, **{f"input_{f}": target[f].to_numpy() for f in COARSE},
        **raw_predictions})
    prediction_path = out / "sealed_predictions.csv.gz"
    prediction_frame.to_csv(prediction_path, index=False, compression="gzip")
    (out / "source_calibration.json").write_text(json.dumps(calibration, indent=2)+"\n")
    code_hashes = {str(p.resolve()): sha(p) for p in [Path(__file__), PROTOCOL]}
    input_hashes = {**source_hashes, str(TARGET_FUELS.resolve()): sha(TARGET_FUELS),
                    **{str(path.resolve()): sha(path) for path in [
                        SOURCE_DIR/"acquisition_manifest.json",
                        SOURCE_DIR/"official_intensity_panel_manifest.json"] if path.exists()}}
    seal = {"prediction_file": prediction_path.name, "prediction_sha256": sha(prediction_path),
            "prediction_rows": len(prediction_frame),
            "source_calibration_sha256": sha(out/"source_calibration.json"),
            "protocol_runner_sha256": code_hashes,
            "source_target_feature_sha256": input_hashes,
            "target_label_path": str(args.target_labels.resolve()),
            "target_labels_read_before_prediction_seal": False,
            "target_features": "exact t-2 CN5 monthly generation shares and calendar"}
    (out / "prediction_seal.json").write_text(json.dumps(seal, indent=2)+"\n")
    if sha(prediction_path) != seal["prediction_sha256"]:
        raise RuntimeError("prediction seal check failed")
    if any(sha(Path(path)) != value for path, value in {**input_hashes, **code_hashes}.items()):
        raise RuntimeError("an input or code file changed before target scoring")

    # First read of the 2025 target label occurs after the sealed forecast.
    labels = pd.read_csv(args.target_labels, parse_dates=["from"])
    labels["time"] = pd.to_datetime(labels["from"], utc=True).dt.floor("h")
    labels["y"] = pd.to_numeric(labels.actual_gco2_per_kwh, errors="coerce")
    label_hour = labels.groupby("time").y.agg(["mean", "count"]).reset_index()
    label_hour = label_hour.loc[label_hour["count"].eq(2) & label_hour["mean"].notna()].copy()
    label_hour["y"] = label_hour["mean"]
    label_hour = label_hour[["time", "y"]]
    target_labels = label_hour.merge(target[["time", "clock"]], on="time", how="left", validate="one_to_one")
    target_labels = target_labels.loc[target_labels.clock.dt.year.eq(2025)].copy()
    preds = prediction_frame[["time", *METHODS]].copy()
    overall, monthly = score(target_labels, preds, calibration)
    overall.to_csv(out/"overall_metrics.csv", index=False)
    monthly.to_csv(out/"monthly_metrics.csv", index=False)
    model_audit = audit
    model_audit["training_response"] = "EIA-930 official generated CO2 intensity; lb/kWh x 453.59237"
    model_audit["target_response"] = "NESO national intensity.actual, mean of two available half-hours"
    model_audit["target_hours_with_two_label_halfhours"] = int(len(target_labels))
    model_audit["target_missing_or_incomplete_hourly_labels"] = int(8760-len(target_labels))
    model_audit["target_label_sha256"] = sha(args.target_labels)
    (out/"model_audit.json").write_text(json.dumps(model_audit, indent=2)+"\n")
    (out/"scoring_manifest.json").write_text(json.dumps({
        "target_label_path": str(args.target_labels.resolve()),
        "target_label_sha256": sha(args.target_labels),
        "prediction_sha256_verified": seal["prediction_sha256"],
        "scored_hours": int(overall.hours.iloc[0]),
        "complete_label_hour_count": int(len(target_labels)),
        "target_label_first_opened_after_seal": True,
    }, indent=2)+"\n")
    (out/"audit.json").write_text(json.dumps({"status": "DEVELOPMENT_TIME_HOLDOUT_OFFICIAL_GENERATION_ESTIMATES",
        "direction": "US_EIA_to_GB_NESO", "train_years": [2022, 2023], "calibration_year": 2024,
        "target_year": 2025, "target_features": "t-2 CN5 monthly generation shares and local calendar",
        "target_months": sorted(target.clock.dt.month.unique().astype(int).tolist()),
        "target_label": "NESO official estimated intensity.actual; not measured emissions",
        "target_inputs_as_issued": False, "blind_geographic_confirmation": False,
        "consumption_track": False, "training_rows_by_ba": model_audit["source_training_rows_by_ba"],
        "target_feature_audit": target_audit}, indent=2)+"\n")
    print(overall.round(4).to_string(index=False))
    print(f"wrote US-to-GB official generation transfer: {out}", flush=True)


if __name__ == "__main__":
    main()
