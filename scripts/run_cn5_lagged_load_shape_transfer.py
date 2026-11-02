"""Nested strict-CN5 LOCO test of a one-year-lagged load-shape feature."""
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_nine_country_loco as six  # noqa: E402
import run_energy_charts_nine_country_cn5_loco as cn5  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = tuple(code.upper() for code in cn5.COUNTRIES)
PROTOCOL = ROOT / "docs/CN5_LAGGED_LOAD_SHAPE_TRANSFER_PROTOCOL_20261102.md"
LOAD_FEATURE = "lag_load_shape"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hourly_load(code: str) -> tuple[pd.DataFrame, dict]:
    folder, manifest_path, manifest = six.data_locations(code.lower())
    raw_path = folder / f"{code.lower()}_2022_2024.json"
    expected = next(row["sha256"] for row in manifest["countries"]
                    if row["country"] == code.lower())
    if sha(raw_path) != expected:
        raise ValueError(f"raw input hash mismatch for {code}")
    payload = json.loads(raw_path.read_text())
    zone = payload["timezone"]
    records = []
    local_stamps = []
    for row in payload["data"]:
        values = row.get("values", {})
        value = pd.to_numeric(values.get("load"), errors="coerce")
        original_stamp = pd.Timestamp(row["timestamp"])
        local_stamps.append(original_stamp)
        if pd.notna(value) and np.isfinite(float(value)) and float(value) > 0:
            stamp = original_stamp.tz_convert("UTC")
            records.append((stamp.floor("h"), original_stamp.year, float(value)))
    if not records:
        raise ValueError(f"no finite hourly load for {code}")
    timestamp_frame = pd.DataFrame({
        "stamp": pd.to_datetime([stamp.tz_convert("UTC") for stamp in local_stamps], utc=True),
        "local_year": [stamp.year for stamp in local_stamps],
    })
    resolution_by_year = {}
    for year, block in timestamp_frame.groupby("local_year"):
        deltas = block.stamp.sort_values().diff().dt.total_seconds().dropna()
        candidates = deltas[deltas.isin([900., 3600.])]
        if candidates.empty:
            raise ValueError(f"cannot infer native load resolution for {code}/{year}")
        resolution_by_year[int(year)] = int(candidates.mode().iloc[0] / 60.)
    samples = pd.DataFrame(records, columns=["time", "year", "sample"])
    grouped = samples.groupby(["year", "time"]).agg(load=("sample", "mean"), count=("sample", "size"))
    keep = [int(row["count"]) == max(1, 60 // resolution_by_year[int(year)])
            for (year, _), row in grouped.iterrows()]
    grouped = grouped.loc[np.asarray(keep, dtype=bool), ["load"]].reset_index()
    grouped["clock"] = grouped.time.dt.tz_convert(zone).dt.tz_localize(None)
    grouped["period"] = grouped.clock.dt.to_period("M")
    grouped["weekend"] = grouped.clock.dt.dayofweek.ge(5).astype(int)
    grouped["hour"] = grouped.clock.dt.hour.astype(int)
    coverage = float(grouped.load.notna().mean())
    audit = {"country": code, "timezone": zone,
             "resolution_minutes_by_year": resolution_by_year,
             "hourly_finite_rows": int(len(grouped)), "hourly_load_mean_mw": float(grouped.load.mean()),
             "raw_sha256": sha(raw_path), "manifest_sha256": sha(manifest_path)}
    return grouped, audit


def add_lagged_load_feature(features: pd.DataFrame, load: pd.DataFrame) -> pd.DataFrame:
    """Map the previous-year same-month 48-bin load shape onto hourly rows."""
    out = features.copy()
    period = out.clock.dt.to_period("M")
    load_profiles: dict[pd.Period, np.ndarray] = {}
    load_by_period = {}
    for month, block in load.groupby("period", sort=True):
        profile = block.groupby(["weekend", "hour"]).load.mean().reindex(
            pd.MultiIndex.from_product([[0, 1], range(24)], names=["weekend", "hour"]))
        if profile.isna().any() or not np.isfinite(profile.to_numpy(float)).all():
            continue
        mean = float(block.load.mean())
        if mean <= 0:
            continue
        load_profiles[month] = profile.to_numpy(float) / mean
        load_by_period[month] = int(len(block))
    raw_clock = out.clock
    wknd = raw_clock.dt.dayofweek.ge(5).astype(int).to_numpy()
    hour = raw_clock.dt.hour.to_numpy(int)
    values = np.full(len(out), np.nan, dtype=float)
    for idx, current_month in enumerate(period):
        prior = current_month - 12
        profile = load_profiles.get(prior)
        if profile is not None:
            values[idx] = profile[wknd[idx] * 24 + hour[idx]]
    out[LOAD_FEATURE] = values
    out["lag_load_source_hours"] = [load_by_period.get(month - 12, 0) for month in period]
    return out


def month_score(target: str, method: str, month: int, mask: np.ndarray,
                query: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                radius: float) -> dict:
    error = truth[mask] - prediction[mask]
    daily = pd.DataFrame({"truth": truth[mask], "prediction": prediction[mask],
                          "day": query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy()})
    centered = daily[["truth", "prediction"]] - daily.groupby("day")[
        ["truth", "prediction"]].transform("mean")
    abs_error = np.abs(error)
    return {"target_country": target, "method": method, "month": int(month),
            "hours": int(mask.sum()), "mae": float(abs_error.mean()),
            "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
            "coverage90": float(np.mean(abs_error <= radius)), "radius90": radius,
            "width90": float(2 * radius),
            "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(abs_error - radius, 0.)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    prepared, audits, load_audits, input_paths = {}, {}, {}, [PROTOCOL]
    for country in COUNTRIES:
        code = country.lower()
        raw, features, audit = cn5.prepare_country(code)
        load, load_audit = load_hourly_load(code)
        features = add_lagged_load_feature(features, load)
        # Exact common-support construction: same 2023 fitting/calibration and
        # 2024 target rows for both M and M+L12.
        if features.loc[features.year.isin([2023, 2024]), LOAD_FEATURE].isna().any():
            missing = features.loc[features.year.isin([2023, 2024]) & features[LOAD_FEATURE].isna()]
            raise ValueError(f"{country}: missing lagged load profile on {len(missing)} required rows")
        prepared[country] = raw, features
        audits[country] = audit
        load_audits[country] = load_audit
        folder, manifest_path, _ = six.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in six.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")

    input_hashes = {str(path.resolve()): sha(path) for path in sorted(set(input_paths))}
    code_paths = [Path(__file__).resolve(), PROTOCOL, cn5.PROTOCOL, Path(cn5.__file__).resolve(),
                  Path(six.__file__).resolve(), Path(six.base.__file__).resolve(),
                  Path(model.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    (output / "startup_hashes.json").write_text(json.dumps(
        {"inputs": input_hashes, "code": code_hashes}, indent=2) + "\n")
    (output / "load_input_audit.json").write_text(json.dumps(
        {"generation_input_audit": audits, "lagged_load_audit": load_audits},
        ensure_ascii=False, indent=2) + "\n")

    seals, radii = [], {}
    for target in COUNTRIES:
        sources = [country for country in COUNTRIES if country != target]
        train_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2023)
                       & prepared[c][1].clock.dt.month.le(6)] for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([cn5.fuel_for(prepared[c][0], part)
                                     for c, part in zip(sources, train_parts)])
        train_coarse = np.concatenate([model.coarse_values(prepared[c][0].iloc[
            part.raw_row.to_numpy()]) for c, part in zip(sources, train_parts)])
        if len(train) != len(train_fuel) or train.empty or train[LOAD_FEATURE].isna().any():
            raise ValueError(f"{target}: invalid source training support")
        fitted = model.fit(train, train_fuel, train_coarse)

        cal_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2023)
                     & prepared[c][1].clock.dt.month.ge(7)] for c in sources]
        cal = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([cn5.fuel_for(prepared[c][0], part)
                                   for c, part in zip(sources, cal_parts)])
        if len(cal) != len(cal_fuel) or cal.empty or cal[LOAD_FEATURE].isna().any():
            raise ValueError(f"{target}: invalid source calibration support")
        cal_truth = cal_fuel @ model.FACTORS / cal_fuel.sum(axis=1)
        cal_weights = model.weights(cal)
        cal_pred = model.predict(fitted, cal)

        feature_cols = [*model.HISTORY, LOAD_FEATURE]
        w = model.weights(train)
        thermal = train_fuel[:, :4].sum(axis=1)
        factor_y = np.divide(train_fuel @ model.FACTORS, thermal,
                             out=np.full(len(train_fuel), 400.), where=thermal > 0)
        active = thermal > 0
        load_factor = HistGradientBoostingRegressor(
            max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=10., random_state=416, early_stopping=False)
        load_factor.fit(train.loc[active, feature_cols], factor_y[active],
                        sample_weight=w[active] * len(w))

        def plus_load_predict(frame: pd.DataFrame) -> np.ndarray:
            q = frame[list(model.GROUPS)].to_numpy(float) * fitted["profile"][
                frame.clock.dt.hour.to_numpy(int)]
            thermal_share = q[:, 0] / q.sum(axis=1)
            estimated_factor = np.clip(load_factor.predict(frame[feature_cols]), 400., 1000.)
            return estimated_factor * thermal_share

        cal_pred["factor_history_lag_load_shape"] = plus_load_predict(cal)
        radii[target] = {name: float(weighted_quantile(
            np.abs(cal_truth - pred), cal_weights, .9)) for name, pred in cal_pred.items()
            if name in ("anchor700", "ratio_recomposition", "factor_history",
                        "factor_history_lag_load_shape")}

        query = prepared[target][1].loc[prepared[target][1].year.eq(2024)].copy()
        if query.empty or query[LOAD_FEATURE].isna().any():
            raise ValueError(f"{target}: invalid 2024 target feature support")
        target_predictions = model.predict(fitted, query)
        target_predictions["factor_history_lag_load_shape"] = plus_load_predict(query)
        raw = prepared[target][0]
        forecast_path = output / f"{target}_heldout_predictions.csv.gz"
        timestamp = raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()
        pd.DataFrame({"time": timestamp, **target_predictions}).to_csv(
            forecast_path, index=False, compression="gzip")
        radius_path = output / f"{target}_source_radii.json"
        radius_path.write_text(json.dumps(radii[target], indent=2) + "\n")
        seals.append({"target": target, "prediction_file": forecast_path.name,
                      "prediction_sha256": sha(forecast_path), "radius_file": radius_path.name,
                      "radius_sha256": sha(radius_path), "hours": len(query),
                      "sources": sources})
        print(f"sealed {target}: {len(query)} hourly predictions", flush=True)

    seal_path = output / "prediction_seal.json"
    seal_path.write_text(json.dumps(seals, indent=2) + "\n")
    if any(sha(Path(path)) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("an input changed during the run")
    if any(sha(Path(path)) != digest for path, digest in code_hashes.items()):
        raise RuntimeError("code or protocol changed during the run")

    rows = []
    for target in COUNTRIES:
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        truth_fuel = cn5.fuel_for(raw, query)
        truth = truth_fuel @ model.FACTORS / truth_fuel.sum(axis=1)
        seal = next(item for item in seals if item["target"] == target)
        forecast_path = output / seal["prediction_file"]
        radius_path = output / seal["radius_file"]
        if sha(forecast_path) != seal["prediction_sha256"] or sha(radius_path) != seal["radius_sha256"]:
            raise RuntimeError(f"sealed artifacts changed before scoring: {target}")
        forecast = pd.read_csv(forecast_path)
        expected_times = raw.iloc[query.raw_row.to_numpy()].time.astype(str).reset_index(drop=True)
        if not forecast.time.astype(str).reset_index(drop=True).equals(expected_times):
            raise ValueError(f"{target}: prediction/label timestamp misalignment")
        target_radii = json.loads(radius_path.read_text())
        methods = ("anchor700", "ratio_recomposition", "factor_history",
                   "factor_history_lag_load_shape")
        months = query.clock.dt.month.to_numpy(int)
        for method in methods:
            pred = forecast[method].to_numpy(float)
            for month in range(1, 13):
                mask = months == month
                rows.append(month_score(target, method, month, mask, query, truth, pred,
                                        float(target_radii[method])))
    details = pd.DataFrame(rows)
    details.to_csv(output / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    country = details.groupby(["target_country", "method"])[metrics].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    summary = country.groupby("method")[metrics].mean().reset_index()
    baseline = summary.set_index("method")
    m, ml = baseline.loc["factor_history"], baseline.loc["factor_history_lag_load_shape"]
    summary["mae_delta_vs_factor_history"] = summary.method.map(
        lambda method: baseline.loc[method, "mae"] - m.mae)
    summary["coverage_delta_vs_factor_history"] = summary.method.map(
        lambda method: baseline.loc[method, "coverage90"] - m.coverage90)
    summary["score_delta_vs_factor_history"] = summary.method.map(
        lambda method: baseline.loc[method, "interval_score90"] - m.interval_score90)
    summary.to_csv(output / "macro_metrics.csv", index=False)
    (output / "verification.json").write_text(json.dumps({
        "countries": list(COUNTRIES), "target_year": 2024, "train_year": "2023H1",
        "calibration_period": "2023H2", "target_rows": int(len(details)),
        "target_hours_by_country": details[["target_country", "month", "hours"]]
            .drop_duplicates().groupby("target_country").hours.sum().to_dict(),
        "all_predictions_sealed_before_target_scoring": True,
        "m_plus_l_input": "previous-calendar-year same-month local weekday/weekend x hour load profile, normalized by its month mean",
        "labels": "fixed-factor generation-side hourly proxy, not measured CIF",
        "countries_are_development_domains": True,
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
