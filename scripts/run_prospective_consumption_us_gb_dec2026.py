"""Seal paired US↔GB December 2026 official consumption-intensity forecasts."""
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

import audit_live_eia923_ba_vintage as vintage  # noqa: E402
import run_official_intensity_transfer as official  # noqa: E402
import run_prospective_us_to_gb_nov2026 as us_gb  # noqa: E402
from transcif.research import capacity_transfer  # noqa: E402
from transcif.research.cross_country import COARSE, FUELS, ZONES, fit_models  # noqa: E402

OUT = ROOT / "results/prospective-consumption-us-gb-dec2026-r1"
PROTOCOL = ROOT / "docs/PROSPECTIVE_CONSUMPTION_US_GB_DEC2026_PROTOCOL.md"
GB_PANEL = ROOT / "data/neso-gb/df_fuel_ckan.csv"
EIA923_ZIP = ROOT / "docs/evidence/eia923-live-monthly-2026-08-r1/f923_2026.zip"
CAPACITY_TABLE = official.CAPACITY_TABLE_PATH
FORECAST_ORIGIN = pd.Timestamp("2026-11-01T15:33:33Z")
TARGET_MONTH = pd.Period("2026-12", freq="M")
OBSERVED_EIA_MONTH = "2026-08"
OBSERVED_NESO_MONTH = pd.Period("2026-09", freq="M")
TARGET_START_UTC = pd.Timestamp("2026-12-01T00:00:00Z")
TARGET_STOP_UTC = pd.Timestamp("2027-01-01T00:00:00Z")
MIN_EIA_ROW_COVERAGE = 0.90


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def neso_monthly_shares() -> tuple[dict[str, float], dict]:
    columns = ["DATETIME", "GAS", "COAL", "NUCLEAR", "WIND", "WIND_EMB",
               "HYDRO", "BIOMASS", "OTHER", "SOLAR"]
    frame = pd.read_csv(GB_PANEL, usecols=columns)
    stamp = pd.to_datetime(frame.DATETIME, errors="raise")
    if stamp.dt.tz is not None:
        stamp = stamp.dt.tz_convert("UTC").dt.tz_localize(None)
    frame["stamp"] = stamp
    fuel = pd.DataFrame(index=frame.index)
    fuel["coal"], fuel["gas"], fuel["oil"] = frame.COAL, frame.GAS, 0.0
    fuel["other"] = frame.BIOMASS + frame.OTHER
    fuel["nuclear"], fuel["hydro"] = frame.NUCLEAR, frame.HYDRO
    fuel["wind"], fuel["solar"] = frame.WIND + frame.WIND_EMB, frame.SOLAR
    values = fuel.to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    fuel, stamp = fuel.loc[valid], frame.loc[valid, "stamp"]
    hour = stamp.dt.floor("h")
    hourly = fuel.groupby(hour).mean()
    counts = hour.groupby(hour).size()
    hourly = hourly.loc[counts.eq(2)]
    local = hourly.index.tz_localize("UTC").tz_convert("Europe/London").tz_localize(None)
    periods = local.to_period("M")
    month = hourly.loc[periods == OBSERVED_NESO_MONTH].sum()
    expected = int(((OBSERVED_NESO_MONTH + 1).start_time.tz_localize("Europe/London")
                    - OBSERVED_NESO_MONTH.start_time.tz_localize("Europe/London"))
                   .total_seconds() / 3600)
    observed = int((periods == OBSERVED_NESO_MONTH).sum())
    if observed != expected or month.sum() <= 0:
        raise ValueError(f"NESO target month incomplete: {observed}/{expected}")
    total = float(month.sum())
    shares = {
        "thermal": float(month[["coal", "gas", "oil", "other"]].sum() / total),
        "hydro": float(month.hydro / total), "nuclear": float(month.nuclear / total),
        "wind": float(month.wind / total), "solar": float(month.solar / total),
    }
    if abs(sum(shares.values()) - 1.0) > 1e-10:
        raise ValueError("NESO monthly five-group vector does not close")
    return shares, {"source": str(GB_PANEL.relative_to(ROOT)), "columns_read": columns,
                    "target_response_column_read": False,
                    "target_response_column_excluded": "CARBON_INTENSITY",
                    "observed_month": str(OBSERVED_NESO_MONTH),
                    "observed_hours": observed, "expected_hours": expected,
                    "target_month_lag": "t-3", "shares": shares,
                    "vintage": "current revised snapshot; no complete as-issued chain"}


def eia923_monthly_shares() -> tuple[dict[str, dict[str, float]], dict]:
    rows, map_audit = vintage.parse_monthly_generation(
        EIA923_ZIP, 2026, range(1, 9), vintage.DEFAULT_BAS)
    selected = rows.loc[rows.month.eq(OBSERVED_EIA_MONTH)].copy()
    selected = selected.loc[selected.eia923_numeric_row_fraction.ge(MIN_EIA_ROW_COVERAGE)]
    if selected.ba.nunique() < 5:
        raise ValueError("fewer than five BAs pass the frozen 90% EIA-923 numeric-row gate")
    output = {}
    for row in selected.itertuples(index=False):
        fuel = {name: float(getattr(row, f"eia923_{name}_mwh")) for name in FUELS}
        if any(not np.isfinite(value) for value in fuel.values()):
            raise ValueError(f"nonfinite reported EIA-923 vector for {row.ba}")
        total = float(sum(fuel.values()))
        if total <= 0:
            raise ValueError(f"nonpositive reported EIA-923 total for {row.ba}")
        shares = {
            "thermal": sum(fuel[name] for name in ("coal", "gas", "oil", "other")) / total,
            "hydro": fuel["hydro"] / total, "nuclear": fuel["nuclear"] / total,
            "wind": fuel["wind"] / total, "solar": fuel["solar"] / total,
        }
        if abs(sum(shares.values()) - 1.0) > 1e-10:
            raise ValueError(f"EIA-923 shares fail closure for {row.ba}")
        output[str(row.ba)] = {
            "shares": shares,
            "numeric_row_fraction": float(row.eia923_numeric_row_fraction),
            "numeric_rows": int(row.eia923_numeric_rows),
            "reported_rows": int(row.eia923_plant_fuel_rows),
            "reported_subset_mwh": total,
            "signed_hydro_mwh": fuel["hydro"],
        }
    return {ba: record["shares"] for ba, record in output.items()}, {
        "source": str(EIA923_ZIP.relative_to(ROOT)),
        "issue_release_date": "2026-10-23", "observed_month": OBSERVED_EIA_MONTH,
        "target_month_lag": "t-4", "coverage_threshold": MIN_EIA_ROW_COVERAGE,
        "mapping_audit": map_audit, "target_ba_inputs": output,
        "input_scope": "numeric reported EIA-923 plant-fuel subset; not certified complete BA generation",
    }


def target_frame(region: str, shares: dict[str, float], zone: str,
                 capacity: dict) -> pd.DataFrame:
    local_start = TARGET_MONTH.start_time.tz_localize(zone)
    local_stop = (TARGET_MONTH + 1).start_time.tz_localize(zone)
    utc = pd.date_range(local_start.tz_convert("UTC"), local_stop.tz_convert("UTC"),
                        inclusive="left", freq="h")
    local = utc.tz_convert(zone).tz_localize(None)
    frame = pd.DataFrame({"time": utc, "clock": local, "year": local.year,
                          "region": region})
    for name, values, cycle in (("hour", local.hour, 24),
                                ("dow", local.dayofweek, 7),
                                ("month", local.month, 12)):
        frame[f"{name}_sin"] = np.sin(2 * np.pi * values / cycle)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * values / cycle)
    for category in COARSE:
        frame[category] = float(shares[category])
        frame[f"{category}_hour_sin"] = frame[category] * frame.hour_sin
        frame[f"{category}_hour_cos"] = frame[category] * frame.hour_cos
    frame["observed_month"] = (str(OBSERVED_NESO_MONTH) if region == "GB"
                               else OBSERVED_EIA_MONTH)
    frame = capacity_transfer.add_capacity(frame, capacity)
    frame["log_capacity_coal_gas_mw"] = np.log1p(
        frame.capacity_coal_mw + frame.capacity_gas_mw)
    return frame


def fit_gb_consumption_source(capacity: dict) -> tuple[dict, dict, dict]:
    raw, prepared, labels, source_audit = official.prepare(ROOT, capacity)
    query = prepared["GB"]
    train = query.loc[query.year.eq(2022)].copy()
    y_train = official.aligned_labels(train, labels["GB"])
    good = np.isfinite(y_train)
    train, y_train = train.loc[good].copy(), y_train[good]
    official_models = official.train_official(train, y_train)
    train_fuel = raw["GB"].iloc[train.raw_row.to_numpy()][list(FUELS)].to_numpy(float)
    proxy_models = fit_models(train.reset_index(drop=True), train_fuel)

    calibration = query.loc[query.year.eq(2023)].copy()
    y_cal = official.aligned_labels(calibration, labels["GB"])
    good = np.isfinite(y_cal)
    calibration, y_cal = calibration.loc[good].copy(), y_cal[good]
    predictions = official.point_predictions(official_models, proxy_models, calibration)
    predictions["anchor700"] = np.clip(700 * calibration.thermal.to_numpy(float), 0., 1000.)
    residuals = {method: [y_cal - predictions[method]] for method in us_gb.METHODS}
    calibration_stats = official.calibrate(residuals)
    if "anchor700" not in calibration_stats:
        values = residuals["anchor700"]
        bias = float(np.mean([part.mean() for part in values]))
        sigma = float(np.sqrt(np.mean([np.mean((part - bias) ** 2)
                                       for part in values])))
        calibration_stats["anchor700"] = {
            "bias": bias, "sigma": max(sigma, 1.),
            "regions": len(values),
            "calibration_hours": int(sum(len(part) for part in values)),
        }
    source_audit = {"source_country": "GB", "train_regions": 1,
                    "train_year": 2022, "train_hours": int(len(train)),
                    "calibration_year": 2023, "calibration_hours": int(len(calibration)),
                    "labels": "NESO official estimated consumption intensity",
                    "source_data_audit": source_audit}
    return official_models, proxy_models, calibration_stats, source_audit


def seal_prediction(path: Path, frame: pd.DataFrame, direction: str,
                    endpoint: str) -> dict:
    frame.to_csv(path, index=False, compression="gzip")
    return {"path": path.name, "sha256": sha(path), "rows": len(frame),
            "direction": direction, "endpoint": endpoint,
            "first_time_utc": frame.time_utc.iloc[0],
            "last_time_utc": frame.time_utc.iloc[-1],
            "target_response_read_before_seal": False}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if pd.Timestamp.now(tz="UTC") >= TARGET_START_UTC:
        raise RuntimeError("December response period has started; refuse retrospective forecast")
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)

    capacity = capacity_transfer.load_capacity(CAPACITY_TABLE)
    neso_shares, neso_audit = neso_monthly_shares()
    eia_shares, eia_audit = eia923_monthly_shares()
    if len(eia_shares) < 5:
        raise ValueError("insufficient BA input coverage for the GB→US direction")

    files = [Path(__file__).resolve(), PROTOCOL, Path(official.__file__).resolve(),
             Path(us_gb.__file__).resolve(), Path(vintage.__file__).resolve(),
             Path(capacity_transfer.__file__).resolve(),
             ROOT / "scripts/run_cross_country.py",
             ROOT / "src/transcif/research/cross_country.py",
             ROOT / "scripts/prepare_official_intensity_panel.py",
             ROOT / "docs/OFFICIAL_CONSUMPTION_INTENSITY_TRANSFER_PROTOCOL_20261101.md",
             ROOT / "docs/OFFICIAL_ANNUAL_COAL_GAS_CAPACITY_PROTOCOL_20261101.md",
             GB_PANEL, EIA923_ZIP, CAPACITY_TABLE,
             ROOT / "data/eia930-hourly-co2-r1/acquisition_manifest.json",
             ROOT / "data/eia930-hourly-co2-r1/coverage_audit_2022_2024.json",
             ROOT / "data/eia930-hourly-co2-r1/feature_inputs/GB_generation_2022_2024.csv.gz",
             official.SOURCE_LABEL_ROOT / "GB_consumption_2022_2023_halfhour.csv.gz"]
    files += [official.SOURCE_LABEL_ROOT / f"{ba}_consumption_2022_2023.csv.gz"
              for ba in official.US_REGIONS]
    files += [ROOT / f"results/research/data/{ba}_hourly.csv"
              for ba in official.US_REGIONS]
    files += [ROOT / name for name in capacity["input_hashes"]]
    files = list(dict.fromkeys(files))
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("one or more training/input/protocol assets are missing")
    startup_hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}

    # US-source -> GB target, using the same frozen official-intensity pipeline
    # as the November prospective run with only month and input-lag changed.
    us_official, us_proxy, us_cal, us_audit = us_gb.fit_and_calibrate(capacity)
    us_query = target_frame("GB", neso_shares, "Europe/London", capacity)
    us_points = official.point_predictions(us_official, us_proxy, us_query)
    us_points["anchor700"] = np.clip(700 * us_query.thermal.to_numpy(float), 0., 1000.)
    us_frame = pd.DataFrame({
        "time_utc": us_query.time.astype(str),
        "local_time": us_query.clock.astype(str),
        "observed_month": us_query.observed_month,
        **{column: us_query[column] for column in COARSE},
    })
    for method in us_gb.METHODS:
        point = np.asarray(us_points[method], dtype=float)
        bias, sigma = us_cal[method]["bias"], us_cal[method]["sigma"]
        radius = 1.6448536269514722 * sigma
        us_frame[f"{method}_raw"] = point
        us_frame[f"{method}_source_calibrated"] = point + bias
        us_frame[f"{method}_lower90_source"] = point + bias - radius
        us_frame[f"{method}_upper90_source"] = point + bias + radius

    # GB-source -> US targets. The direction-specific source model uses only
    # GB 2022 response for training and GB 2023 response for calibration.
    gb_models, gb_proxy, gb_cal, gb_audit = fit_gb_consumption_source(capacity)
    us_frames = {}
    clipped_frames = {}
    for ba, shares in eia_shares.items():
        zone = ZONES[ba]
        query = target_frame(ba, shares, zone, capacity)
        points = official.point_predictions(gb_models, gb_proxy, query)
        points["anchor700"] = np.clip(700 * query.thermal.to_numpy(float), 0., 1000.)
        frame = pd.DataFrame({
            "time_utc": query.time.astype(str), "local_time": query.clock.astype(str),
            "observed_month": query.observed_month,
            **{column: query[column] for column in COARSE},
        })
        for method in us_gb.METHODS:
            point = np.asarray(points[method], dtype=float)
            bias, sigma = gb_cal[method]["bias"], gb_cal[method]["sigma"]
            radius = 1.6448536269514722 * sigma
            frame[f"{method}_raw"] = point
            frame[f"{method}_source_calibrated"] = point + bias
            frame[f"{method}_lower90_source"] = point + bias - radius
            frame[f"{method}_upper90_source"] = point + bias + radius
        us_frames[ba] = frame

        nonnegative = {name: max(float(value), 0.0) for name, value in shares.items()}
        total_nonnegative = sum(nonnegative.values())
        if total_nonnegative <= 0:
            raise ValueError(f"zero-clipped EIA-923 composition is degenerate for {ba}")
        clipped_shares = {name: value / total_nonnegative
                          for name, value in nonnegative.items()}
        if any(abs(clipped_shares[name] - shares[name]) > 1e-12 for name in shares):
            clipped_query = target_frame(ba, clipped_shares, zone, capacity)
            clipped_points = official.point_predictions(gb_models, gb_proxy, clipped_query)
            clipped_points["anchor700"] = np.clip(
                700 * clipped_query.thermal.to_numpy(float), 0., 1000.)
            clipped_frame = pd.DataFrame({
                "time_utc": clipped_query.time.astype(str),
                "local_time": clipped_query.clock.astype(str),
                "observed_month": clipped_query.observed_month,
                **{column: clipped_query[column] for column in COARSE},
            })
            for method in us_gb.METHODS:
                point = np.asarray(clipped_points[method], dtype=float)
                bias, sigma = gb_cal[method]["bias"], gb_cal[method]["sigma"]
                radius = 1.6448536269514722 * sigma
                clipped_frame[f"{method}_raw"] = point
                clipped_frame[f"{method}_source_calibrated"] = point + bias
                clipped_frame[f"{method}_lower90_source"] = point + bias - radius
                clipped_frame[f"{method}_upper90_source"] = point + bias + radius
            clipped_frames[ba] = clipped_frame

    if len(us_frame) != 744 or any(len(frame) != 744 for frame in us_frames.values()):
        raise ValueError("every December endpoint must have 744 local hours")
    if any(not np.isfinite(frame.filter(regex="_raw$").to_numpy(float)).all()
           for frame in [us_frame, *us_frames.values()]):
        raise ValueError("nonfinite raw predictions")

    OUT.mkdir(parents=True)
    seals = [seal_prediction(OUT / "US_to_GB_NESO_consumption_Dec2026.csv.gz",
                             us_frame, "US_to_GB", "GB")]
    for ba, frame in us_frames.items():
        seals.append(seal_prediction(OUT / f"GB_to_US_{ba}_consumption_Dec2026.csv.gz",
                                     frame, "GB_to_US", ba))
    for ba, frame in clipped_frames.items():
        seals.append(seal_prediction(
            OUT / f"GB_to_US_{ba}_consumption_Dec2026_negative_clipped.csv.gz",
            frame, "GB_to_US_zero_clipped_sensitivity", ba))
    (OUT / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    (OUT / "source_calibration.json").write_text(json.dumps({
        "US_to_GB": us_cal, "GB_to_US": gb_cal,
    }, indent=2) + "\n")
    (OUT / "source_audit.json").write_text(json.dumps({
        "US_to_GB": us_audit, "GB_to_US": gb_audit,
    }, indent=2) + "\n")
    (OUT / "input_audit.json").write_text(json.dumps({
        "forecast_origin_utc": FORECAST_ORIGIN.isoformat(),
        "target_month": str(TARGET_MONTH),
        "target_start_utc": TARGET_START_UTC.isoformat(),
        "target_stop_utc_exclusive": TARGET_STOP_UTC.isoformat(),
        "hours_per_gb": len(us_frame), "hours_per_us_ba": 744,
        "neso_input": neso_audit, "eia923_input": eia_audit,
        "target_response_loaded": False,
        "target_response_sources": {"GB": "NESO CARBON_INTENSITY",
                                     "US": "EIA official BA consumption intensity"},
        "source_capacity_budget": "2020 coal/gas nameplate capacity (separate M+C feature arm)",
    }, indent=2) + "\n")
    (OUT / "startup_hashes.json").write_text(json.dumps(startup_hashes, indent=2) + "\n")
    if any(sha(ROOT / name) != digest for name, digest in startup_hashes.items()):
        raise RuntimeError("code or inputs changed during forecast construction")
    if any(sha(OUT / seal["path"]) != seal["sha256"] for seal in seals):
        raise RuntimeError("prediction file hash verification failed")
    status = {
        "status": "SEALED_PROSPECTIVE_CONSUMPTION_US_GB_DEC2026",
        "tracks": ["official_consumption_intensity"],
        "target_month": str(TARGET_MONTH),
        "forecast_origin_utc": FORECAST_ORIGIN.isoformat(),
        "target_hours_gb": len(us_frame),
        "target_hours_per_us_ba": 744,
        "us_to_gb_source_regions": list(official.US_REGIONS),
        "gb_to_us_source_regions": ["GB"],
        "gb_to_us_target_bas": sorted(us_frames),
        "target_response_loaded": False, "target_scored": False,
        "label_semantics": "official operator-estimated consumption-side intensity; not measured emissions",
        "target_inputs": {"GB": {"observed_month": "2026-09", "lag": "t-3",
                                  "vintage": "current revised NESO snapshot"},
                          "US": {"observed_month": "2026-08", "lag": "t-4",
                                  "vintage": "EIA-923 issue released 2026-10-23"}},
        "annual_capacity_information_budget_separate": True,
        "prospective_time_holdout_only": True,
        "unseen_geography_confirmation": False,
        "coverage_guarantee": False,
        "score_eligible_after_utc": "2027-02-03T00:00:00Z",
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"status": status["status"], "seals": seals,
                      "target_bas": sorted(us_frames)}, indent=2))


if __name__ == "__main__":
    main()
