"""Test the information value of monthly within-thermal fuel composition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_live_eia923_ba_vintage as vintage
import run_actual_issue_cross_country as source_model
import run_eia923_actual_ba_information_value as matched
from transcif.research.cross_country import COARSE, EF, FUELS, ZONES, labels, metrics, predict_models

OUT = ROOT / "results/eia923-fine-fuel-composition-increment-r1"
EIA923 = ROOT / "docs/evidence/eia923-live-monthly-2026-08-r1/f923_2026.zip"
PRIOR_ALIGNMENT = ROOT / "docs/evidence/eia923-live-monthly-2026-08-r1/results/ba_month_alignment.csv"
PROTOCOL = ROOT / "docs/EIA923_FINE_FUEL_COMPOSITION_INCREMENT_PROTOCOL_20261101.md"
BASES = tuple(vintage.DEFAULT_BAS)
MONTHS = range(1, 9)
NATIVE = [ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jan_Jun.csv",
          ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jul_Dec.csv",
          ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jan_Jun.csv",
          ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"]
SOURCE_INPUTS = [ROOT / f"results/research/data/{region}_hourly.csv" for region in ZONES]
SOURCE_INPUTS += [ROOT / "data/neso-gb/df_fuel_ckan.csv"]
CODE_INPUTS = [Path(__file__).resolve(), Path(vintage.__file__).resolve(),
               Path(source_model.__file__).resolve(), Path(matched.__file__).resolve(),
               PROTOCOL, ROOT / "src/transcif/data/eia930.py",
               ROOT / "src/transcif/research/cross_country.py",
               ROOT / "src/transcif/research/eia_proxy.py"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_profiles(source: pd.DataFrame) -> np.ndarray:
    train = source.loc[pd.to_datetime(source.clock).dt.year.eq(2022)].copy()
    by_hour = (train.assign(hour=pd.to_datetime(train.clock).dt.hour)
               .groupby("hour")[list(FUELS)].mean().reindex(range(24)).fillna(0.))
    means = by_hour.mean(axis=0).replace(0., np.nan)
    return by_hour.div(means, axis=1).fillna(1.).to_numpy(float)


def source_fuel_support(source: pd.DataFrame) -> dict[str, dict[str, int | bool]]:
    """Record which fuel profiles are learned versus flat fallback assumptions."""
    train = source.loc[pd.to_datetime(source.clock).dt.year.eq(2022)].copy()
    support = {}
    for fuel in FUELS:
        values = train[fuel].fillna(0.).astype(float)
        support[fuel] = {
            "active_hours": int(values.gt(0).sum()),
            "active_months": int(pd.DataFrame({
                "month": pd.to_datetime(train.clock).dt.to_period("M"),
                "value": values,
            }).groupby("month").value.sum().gt(0).sum()),
            "flat_fallback_used": bool(values.sum() <= 0),
        }
    return support


def fine_monthly_source_inputs(source: pd.DataFrame, lag: int = 2) -> pd.DataFrame:
    frame = source[["time", "clock", "region"]].copy()
    clock = pd.to_datetime(frame.clock)
    frame["period"] = clock.dt.to_period("M")
    monthly = source[list(FUELS)].copy()
    monthly["period"] = frame.period
    sums = monthly.groupby("period")[list(FUELS)].sum()
    counts = monthly.groupby("period").size()
    expected = source.assign(period=frame.period).groupby("period").expected_hours.first()
    complete = counts / expected >= .99
    sums = sums.loc[complete]
    shares = sums.div(sums[list(FUELS)].sum(axis=1), axis=0)
    shares.index += lag
    frame = frame.join(shares, on="period", rsuffix="_share")
    frame["hour"] = clock.dt.hour
    return frame


def fine_prediction(profile: np.ndarray, frame: pd.DataFrame,
                    shares: dict[str, float]) -> np.ndarray:
    by_hour = profile[pd.to_datetime(frame.clock).dt.hour.to_numpy()]
    share_vector = np.asarray([shares[fuel] for fuel in FUELS], dtype=float)
    fuel_activity = by_hour * share_vector[None, :]
    total = fuel_activity.sum(axis=1)
    if np.any(total <= 0) or not np.isfinite(fuel_activity).all():
        raise ValueError("fine fuel profile creates nonpositive/nonfinite total activity")
    return fuel_activity @ EF / total


def coarse_shares(row: pd.Series, prefix: str) -> dict[str, float]:
    values = {fuel: float(row[f"{prefix}_{fuel}_mwh"]) for fuel in FUELS}
    if any(not np.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError(f"negative/nonfinite fine fuel share input: {values}")
    values["thermal"] = sum(values[fuel] for fuel in ("coal", "gas", "oil", "other"))
    total = sum(values[fuel] for fuel in FUELS)
    return {name: values[name] / total for name in COARSE}


def fine_shares(row: pd.Series, prefix: str) -> dict[str, float]:
    values = {fuel: float(row[f"{prefix}_{fuel}_mwh"]) for fuel in FUELS}
    if any(not np.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError(f"negative/nonfinite fine fuel share input: {values}")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("fine fuel shares need positive total")
    return {fuel: value / total for fuel, value in values.items()}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if not all(path.is_file() for path in [EIA923, PRIOR_ALIGNMENT, PROTOCOL, *NATIVE,
                                           *SOURCE_INPUTS, *CODE_INPUTS]):
        raise FileNotFoundError("one or more protocol/data/code provenance assets are missing")

    monthly, _ = vintage.parse_monthly_generation(EIA923, 2026, MONTHS, BASES)
    hourly_summary = vintage.parse_hourly(NATIVE, 2026, MONTHS, BASES)
    eligible = monthly.merge(hourly_summary, on=["ba", "month"], validate="one_to_one")
    eligible = eligible.loc[eligible.eia923_complete & eligible.eia930_complete
                            & eligible.eia923_total_mwh.gt(0)
                            & eligible.eia930_total_mwh.gt(0)].copy()
    for fuel in FUELS:
        eligible[f"report_share_{fuel}"] = eligible[f"eia923_{fuel}_mwh"] / eligible.eia923_total_mwh
        eligible[f"oracle_share_{fuel}"] = eligible[f"eia930_{fuel}_mwh"] / eligible.eia930_total_mwh
    eligible["total_ratio_eia923_to_eia930"] = eligible.eia923_total_mwh / eligible.eia930_total_mwh
    eligible["mix_share_l1"] = sum(
        (eligible[f"report_share_{fuel}"] - eligible[f"oracle_share_{fuel}"]).abs()
        for fuel in FUELS)
    eligible = eligible.sort_values(["ba", "month"]).reset_index(drop=True)
    if len(eligible) != 20 or eligible.groupby("ba").size().to_dict() != {
            "BPAT": 6, "CISO": 1, "ERCO": 2, "FPL": 7, "NYIS": 4}:
        raise ValueError("fine-fuel diagnostic requires the frozen raw-recomputed 20-cell queue")

    raw, source_audit, coarse_models, coarse_cal = source_model.fit_source()
    source = raw["GB"]
    profile = normalized_profiles(source)
    fuel_support = source_fuel_support(source)
    source_cal = fine_monthly_source_inputs(source)
    # Keep only 2023 source rows with complete, legal t-2 fuel shares.
    share_columns = [f"{fuel}_share" for fuel in FUELS]
    source_features = source_cal.loc[source_cal.region.eq("GB")
                                     & pd.to_datetime(source_cal.clock).dt.year.eq(2023)].copy()
    # `fine_monthly_source_inputs` preserves original fuel names in the join;
    # use those as target-month shares and remove incomplete lag rows.
    share_columns = list(FUELS)
    good = np.isfinite(source_features[share_columns].to_numpy(float)).all(axis=1)
    source_features = source_features.loc[good].copy()
    source_by_index = source.loc[source_features.index]
    y_cal = labels(source_by_index)
    # Vectorized t-2 fuel shares differ by source month; apply each row's own vector.
    activity = profile[source_features.hour.to_numpy(int)] * source_features[share_columns].to_numpy(float)
    total_activity = activity.sum(axis=1)
    pred_cal = activity @ EF / total_activity
    residual = y_cal - pred_cal
    fine_calibration = {"bias": float(residual.mean()),
                        "sigma": max(float(np.sqrt(np.mean((residual-residual.mean())**2))), 1.),
                        "calibration_hours": int(len(residual)),
                        "calibration_year": 2023, "monthly_feature_lag": 2}

    target_hourly = matched.load_hourly(NATIVE, BASES)
    target_hourly["month"] = target_hourly.time.dt.strftime("%Y-%m")
    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    for record in eligible.itertuples(index=False):
        ba, month = record.ba, record.month
        block = target_hourly.loc[target_hourly.region.eq(ba)
                                  & target_hourly.month.eq(month)].copy()
        if len(block) != int(record.eia930_expected_hours):
            raise ValueError(f"hourly label count changed for {ba}/{month}")
        row = pd.Series(record._asdict())
        actual_fine = fine_shares(row, "eia923")
        oracle_fine = fine_shares(row, "eia930")
        actual_coarse = coarse_shares(row, "eia923")
        features = matched.feature_rows(block, actual_coarse, ba)
        y = labels(block.rename(columns={f"generation_{fuel}": fuel for fuel in FUELS}))
        coarse_point = predict_models(coarse_models, features)["coarse_recomposition"]
        for arm, fuel_mix in (("M5_EIA923_report", actual_fine),
                              ("MplusF_EIA923_internal_mix", actual_fine),
                              ("MplusF_EIA930_same_month_oracle", oracle_fine)):
            if arm == "M5_EIA923_report":
                method, point, bias, sigma = ("M5_coarse_recomposition", coarse_point,
                                               *coarse_cal["coarse_recomposition"])
            else:
                method, point = "MplusF_fuel_profile_recomposition", fine_prediction(profile, features, fuel_mix)
                bias, sigma = fine_calibration["bias"], fine_calibration["sigma"]
            score = metrics(features, y, point, bias, sigma)
            rows.append({"ba": ba, "month": month, "arm": arm, "method": method,
                         **score, "eia923_to_eia930_total_ratio": float(record.total_ratio_eia923_to_eia930),
                         "fine_share_l1": float(record.mix_share_l1)})
            pred_rows.append(pd.DataFrame({"ba": ba, "month": month, "arm": arm,
                                           "method": method, "time_utc": block.time.astype(str),
                                           "local_time": features.clock.astype(str),
                                           "y_fixed_ef_proxy": y, "prediction_raw": point,
                                           "prediction_bias_corrected": point+bias,
                                           "thermal": features.thermal, "hydro": features.hydro,
                                           "nuclear": features.nuclear, "wind": features.wind,
                                           "solar": features.solar}))

    detail = pd.DataFrame(rows)
    macro = (detail.groupby(["arm", "method"], as_index=False)
             .agg(ba_months=("ba", "size"), hours=("hours", "sum"),
                  macro_mae=("mae", "mean"), macro_shape_mae=("shape_mae", "mean"),
                  macro_crps=("crps_gaussian", "mean"), macro_coverage90=("coverage90", "mean"),
                  mean_width90=("width90", "mean"), macro_interval_score90=("interval_score90", "mean")))
    predictions = pd.concat(pred_rows, ignore_index=True)
    pooled_rows = []
    for (arm, method), group in predictions.groupby(["arm", "method"], sort=True):
        bias, sigma = ((coarse_cal["coarse_recomposition"] if method == "M5_coarse_recomposition"
                        else (fine_calibration["bias"], fine_calibration["sigma"])))
        # Both calibration paths are pairs of (bias, sigma).
        if method != "M5_coarse_recomposition":
            bias, sigma = fine_calibration["bias"], fine_calibration["sigma"]
        features = pd.DataFrame({"clock": pd.to_datetime(group.local_time)})
        score = metrics(features, group.y_fixed_ef_proxy.to_numpy(float),
                        group.prediction_raw.to_numpy(float), float(bias), float(sigma))
        daily = pd.DataFrame({"ba": group.ba.to_numpy(),
                              "day": features.clock.dt.strftime("%Y-%m-%d"),
                              "y": group.y_fixed_ef_proxy.to_numpy(float),
                              "p": group.prediction_bias_corrected.to_numpy(float)})
        centered = daily[["y", "p"]] - daily.groupby(["ba", "day"])[["y", "p"]].transform("mean")
        score["shape_mae"] = float(np.abs(centered.y-centered.p).mean())
        pooled_rows.append({"arm": arm, "method": method, **score})

    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "ba_month_method_metrics.csv", index=False)
    macro.to_csv(OUT / "macro_metrics.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(OUT / "pooled_metrics.csv", index=False)
    predictions.to_csv(OUT / "hourly_predictions_and_labels.csv.gz", index=False, compression="gzip")
    source_calibration = {"fine_fuel_profile": fine_calibration,
                          "source_fuel_hourly_support_2022": fuel_support,
                          "M5_coarse_recomposition": {"bias": float(coarse_cal["coarse_recomposition"][0]),
                                                       "sigma": float(coarse_cal["coarse_recomposition"][1]),
                                                       "calibration_year": 2023}}
    (OUT / "source_calibration.json").write_text(json.dumps(source_calibration, indent=2) + "\n")
    provenance = {
        "status": "DEVELOPMENT_FINE_FUEL_INFORMATION_INCREMENT_DIAGNOSTIC",
        "paired_ba_months": int(len(eligible)), "paired_cells_by_ba": {
            str(k): int(v) for k,v in eligible.groupby("ba").size().items()},
        "target_months": sorted(eligible.month.unique().tolist()),
        "source_model": "GB 2022 fit; GB 2023 source-only calibration",
        "MplusF_source_feature_lag": 2,
        "MplusF_source_fuel_profile_support_2022": fuel_support,
        "MplusF_target_budget": "EIA-923 monthly fine fuel composition; extra information unavailable in China five-group M",
        "oracle_arm": "same BA-month EIA-930 fuel shares; nondeployable target-period oracle",
        "label": "current EIA-930 fixed-EF generation-intensity proxy",
        "old_cached_alignment_used_as_input": False,
        "input_hashes": {str(path.relative_to(ROOT)): sha(path)
                         for path in [EIA923, PRIOR_ALIGNMENT, *NATIVE, *SOURCE_INPUTS]},
        "code_hashes": {str(path.relative_to(ROOT)): sha(path) for path in CODE_INPUTS},
        "runtime_versions": {"python": platform.python_version(), "pandas": pd.__version__,
                             "numpy": np.__version__, "scipy": scipy.__version__,
                             "scikit_learn": sklearn.__version__},
        "limitations": ["20 BA-months from five BAs only", "EIA-923 issue postdates evaluated months",
                        "EIA-923 covers reporting plants, not certified BA totals",
                        "MplusF is an upper information budget, not China-available monthly input",
                        "GB 2022 has zero oil-generation hours; oil hourly shape is a flat fallback, not a learned profile",
                        "target intervals are source-calibrated and have no target coverage guarantee"],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(macro.to_string(index=False))
    print("\n", json.dumps({k: provenance[k] for k in
                          ("paired_ba_months", "paired_cells_by_ba", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
