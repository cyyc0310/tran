"""Test annual and trailing local thermal-EF anchors under legal monthly timing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path.cwd()
OUTPUT = ROOT / "results/v51-annual-thermal-ef-probe"
OUTPUT.mkdir(parents=True, exist_ok=True)
REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
TIMEZONES = {
    "BPAT": "America/Los_Angeles",
    "CISO": "America/Los_Angeles",
    "ERCO": "America/Chicago",
    "FPL": "America/New_York",
    "ISNE": "America/New_York",
    "MISO": "America/Chicago",
    "NYIS": "America/New_York",
}
FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
THERMAL = ("coal", "gas", "oil", "other")
EF = np.array([1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 400.0])
ANNUAL_RELEASE_DELAYS = (30, 60, 90, 120)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fuel_totals(row: pd.Series) -> np.ndarray:
    return float(row.total_mwh) * np.array([float(row[f"fuelshare_{fuel}"]) for fuel in FUELS])


def thermal_ef(values: np.ndarray) -> float:
    thermal_energy = sum(values[FUELS.index(fuel)] for fuel in THERMAL)
    if thermal_energy <= 0:
        return float("nan")
    return float(sum(values[FUELS.index(fuel)] * EF[FUELS.index(fuel)] for fuel in THERMAL) / thermal_energy)


def coarse_totals(values: np.ndarray) -> np.ndarray:
    return np.array(
        [
            sum(values[FUELS.index(fuel)] for fuel in THERMAL),
            values[FUELS.index("nuclear")],
            values[FUELS.index("hydro")],
            values[FUELS.index("wind")],
            values[FUELS.index("solar")],
        ]
    )


monthly_path = ROOT / "results/heldout-2025-data/monthly.csv"
monthly = pd.read_csv(monthly_path)
for column in ("period_start", "period_end", "available_at"):
    monthly[column] = pd.to_datetime(monthly[column], utc=True)

hourly: dict[str, pd.DataFrame] = {}
hourly_hashes: dict[str, str] = {}
for region in REGIONS:
    path = ROOT / f"results/heldout-2025-data/{region}_hourly.csv"
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    hourly[region] = frame
    hourly_hashes[str(path)] = sha256(path)


def monthly_row_asof(region: str, target_month: pd.Timestamp) -> pd.Series | None:
    rows = monthly.loc[
        (monthly.region == region)
        & (monthly.period_start.dt.month == target_month.month)
        & (monthly.period_start < target_month)
        & (monthly.period_end <= target_month)
        & (monthly.available_at <= target_month)
    ].sort_values(["period_start", "available_at"])
    return None if rows.empty else rows.iloc[-1]


def local_lag_same_month_ef(region: str, target_month: pd.Timestamp) -> float:
    row = monthly_row_asof(region, target_month)
    return thermal_ef(fuel_totals(row)) if row is not None else float("nan")


def local_trailing12_ef(region: str, target_month: pd.Timestamp) -> float:
    rows = monthly.loc[
        (monthly.region == region)
        & (monthly.period_end <= target_month)
        & (monthly.available_at <= target_month)
    ].sort_values("period_start").tail(12)
    if len(rows) != 12:
        return float("nan")
    periods = [str(value) for value in rows.period_start.dt.strftime("%Y-%m")]
    expected = [str(value) for value in pd.period_range(end=pd.Period(periods[-1], freq="M"), periods=12, freq="M")]
    if periods != expected:
        return float("nan")
    totals = np.sum(np.stack([fuel_totals(row) for _, row in rows.iterrows()]), axis=0)
    return thermal_ef(totals)


def local_latest_annual_ef(region: str, target_month: pd.Timestamp, release_delay_days: int) -> tuple[float, int] | None:
    candidate_years = sorted(monthly.loc[monthly.region == region, "period_start"].dt.year.unique(), reverse=True)
    for year in candidate_years:
        rows = monthly.loc[(monthly.region == region) & (monthly.period_start.dt.year == year)].sort_values("period_start")
        if len(rows) != 12 or rows.period_start.dt.month.tolist() != list(range(1, 13)):
            continue
        assumed_release = pd.Timestamp(f"{year + 1}-01-01", tz="UTC") + pd.Timedelta(days=release_delay_days)
        if assumed_release > target_month:
            continue
        annual_fuel_totals = np.sum(np.stack([fuel_totals(row) for _, row in rows.iterrows()]), axis=0)
        return thermal_ef(annual_fuel_totals), int(year)
    return None


def target_month_oracle_ef(region: str, target_month: pd.Timestamp) -> float:
    rows = monthly.loc[(monthly.region == region) & (monthly.period_start == target_month)]
    return thermal_ef(fuel_totals(rows.iloc[0])) if len(rows) == 1 else float("nan")


def asof_coarse_totals(region: str, target_month: pd.Timestamp) -> np.ndarray | None:
    row = monthly_row_asof(region, target_month)
    return coarse_totals(fuel_totals(row)) if row is not None else None


def donor_same_month_ef(donors: tuple[str, ...], target_month: pd.Timestamp, forecast_year: int) -> float:
    estimates = []
    for donor in donors:
        rows = monthly.loc[
            (monthly.region == donor)
            & (monthly.period_start.dt.month == target_month.month)
            & (monthly.period_start.dt.year == forecast_year - 1)
        ]
        if len(rows) == 1:
            estimate = thermal_ef(fuel_totals(rows.iloc[0]))
            if np.isfinite(estimate):
                estimates.append(estimate)
    return float(np.mean(estimates)) if estimates else float(np.mean(EF[[0, 1, 2, 7]]))


def donor_hourly_shape(donors: tuple[str, ...], target_month: pd.Timestamp, training_year: int) -> np.ndarray:
    lower = pd.Timestamp(f"{training_year}-01-01", tz="UTC")
    upper = pd.Timestamp(f"{training_year + 1}-01-01", tz="UTC")
    source_shapes = []
    for donor in donors:
        frame = hourly[donor].loc[
            (hourly[donor].time >= lower)
            & (hourly[donor].time < upper)
            & (hourly[donor].time.dt.month == target_month.month)
            & hourly[donor].valid.astype(str).str.lower().isin(("true", "1"))
        ].copy()
        local = frame.time.dt.tz_convert(ZoneInfo(TIMEZONES[donor]))
        frame["local_hour"] = local.dt.hour
        values = np.stack(
            [sum(frame[f"generation_{fuel}"].to_numpy(float) for fuel in THERMAL)]
            + [frame[f"generation_{fuel}"].to_numpy(float) for fuel in ("nuclear", "hydro", "wind", "solar")],
            axis=1,
        )
        shape = np.zeros((24, 5))
        for hour in range(24):
            selected = frame.local_hour.to_numpy() == hour
            if selected.any():
                shape[hour] = values[selected].mean(axis=0)
        sums = shape.sum(axis=0)
        source_shapes.append(np.divide(shape, sums[None, :], out=np.ones_like(shape) / 24, where=sums[None, :] > 0))
    average = np.mean(source_shapes, axis=0)
    sums = average.sum(axis=0, keepdims=True)
    return np.divide(average, sums, out=np.ones_like(average) / 24, where=sums > 0)


def predict(region: str, target_month: pd.Timestamp) -> list[dict]:
    donors = tuple(other for other in REGIONS if other != region)
    totals = asof_coarse_totals(region, target_month)
    if totals is None:
        return []
    target = hourly[region].loc[
        (hourly[region].time >= target_month)
        & (hourly[region].time < target_month + pd.offsets.MonthBegin(1))
        & hourly[region].valid.astype(str).str.lower().isin(("true", "1"))
    ].copy()
    if target.empty:
        return []
    actual = np.stack([target[f"generation_{fuel}"].to_numpy(float) for fuel in FUELS], axis=1)
    valid = np.isfinite(actual).all(axis=1) & (actual >= 0).all(axis=1) & (actual.sum(axis=1) > 0)
    target = target.loc[valid]
    actual = actual[valid]
    if len(target) < 480:
        return []

    shape = donor_hourly_shape(donors, target_month, target_month.year - 1)
    local_hour = target.time.dt.tz_convert(ZoneInfo(TIMEZONES[region])).dt.hour.to_numpy()
    raw_groups = shape[local_hour]
    denominator = raw_groups.sum(axis=0)
    predicted_groups = np.divide(
        raw_groups * totals[None, :],
        denominator[None, :],
        out=np.zeros_like(raw_groups),
        where=denominator[None, :] > 0,
    )
    truth = actual @ EF / actual.sum(axis=1)
    actual_thermal = sum(actual[:, FUELS.index(fuel)] for fuel in THERMAL)
    actual_total = actual.sum(axis=1)
    actual_thermal_share = actual_thermal / np.maximum(actual_total, 1e-12)
    actual_thermal_ef = np.divide(
        truth * actual_total,
        actual_thermal,
        out=np.zeros_like(truth),
        where=actual_thermal > 0,
    )

    ef_candidates: dict[str, float] = {
        "donor_same_month": donor_same_month_ef(donors, target_month, target_month.year),
        "local_lag_same_month": local_lag_same_month_ef(region, target_month),
        "local_trailing12_asof": local_trailing12_ef(region, target_month),
        "target_month_oracle": target_month_oracle_ef(region, target_month),
    }
    annual_years: dict[str, int] = {}
    for delay in ANNUAL_RELEASE_DELAYS:
        annual = local_latest_annual_ef(region, target_month, delay)
        name = f"local_latest_annual_d{delay}d"
        if annual is not None:
            ef_candidates[name], annual_years[name] = annual

    output = []
    for method, factor in ef_candidates.items():
        if not np.isfinite(factor):
            continue
        prediction = predicted_groups[:, 0] * factor / np.maximum(predicted_groups.sum(axis=1), 1e-12)
        error = truth - prediction
        predicted_thermal_share = predicted_groups[:, 0] / np.maximum(predicted_groups.sum(axis=1), 1e-12)
        shape_error = factor * (actual_thermal_share - predicted_thermal_share)
        thermal_ef_error = actual_thermal_share * (actual_thermal_ef - factor)
        # Diagnostic ceiling only: replace transferred hourly thermal share with
        # its realized local value while keeping the same legally available EF.
        # This is an oracle, not a deployable predictor.
        oracle_shape_error = truth - actual_thermal_share * factor
        reconstruction_error = error - shape_error - thermal_ef_error
        output.append(
            {
                "region": region,
                "month": str(target_month.date()),
                "method": method,
                "annual_data_year": annual_years.get(method),
                "hours": len(truth),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "shape_error_mae": float(np.mean(np.abs(shape_error))),
                "shape_error_rmse": float(np.sqrt(np.mean(shape_error**2))),
                "shape_error_bias": float(np.mean(shape_error)),
                "thermal_ef_error_mae": float(np.mean(np.abs(thermal_ef_error))),
                "thermal_ef_error_rmse": float(np.sqrt(np.mean(thermal_ef_error**2))),
                "thermal_ef_error_bias": float(np.mean(thermal_ef_error)),
                "oracle_shape_mae": float(np.mean(np.abs(oracle_shape_error))),
                "oracle_shape_rmse": float(np.sqrt(np.mean(oracle_shape_error**2))),
                "oracle_shape_bias": float(np.mean(oracle_shape_error)),
                "component_cross_term": float(2.0 * np.mean(shape_error * thermal_ef_error)),
                "reconstruction_max_abs": float(np.max(np.abs(reconstruction_error))),
            }
        )
    return output


records = []
for half, first_month, last_month in (("H1", 1, 6), ("H2", 7, 12)):
    target_months = pd.date_range(f"2025-{first_month:02d}-01", f"2025-{last_month:02d}-01", freq="MS", tz="UTC")
    for region in REGIONS:
        for target_month in target_months:
            for record in predict(region, target_month):
                records.append({**record, "half": half})

monthly_scores = pd.DataFrame(records)
monthly_scores.to_csv(OUTPUT / "source_month.csv", index=False)
source_summary = (
    monthly_scores.groupby(["half", "region", "method"])
    .agg(
        months=("month", "size"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        bias=("bias", "mean"),
        shape_error_mae=("shape_error_mae", "mean"),
        shape_error_rmse=("shape_error_rmse", "mean"),
        thermal_ef_error_mae=("thermal_ef_error_mae", "mean"),
        thermal_ef_error_rmse=("thermal_ef_error_rmse", "mean"),
        oracle_shape_mae=("oracle_shape_mae", "mean"),
        oracle_shape_rmse=("oracle_shape_rmse", "mean"),
        component_cross_term=("component_cross_term", "mean"),
        reconstruction_max_abs=("reconstruction_max_abs", "max"),
    )
    .reset_index()
)
source_summary.to_csv(OUTPUT / "source_macro.csv", index=False)
macro = (
    source_summary.groupby(["half", "method"])
    .agg(
        sources=("region", "nunique"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        bias=("bias", "mean"),
        shape_error_mae=("shape_error_mae", "mean"),
        shape_error_rmse=("shape_error_rmse", "mean"),
        thermal_ef_error_mae=("thermal_ef_error_mae", "mean"),
        thermal_ef_error_rmse=("thermal_ef_error_rmse", "mean"),
        oracle_shape_mae=("oracle_shape_mae", "mean"),
        oracle_shape_rmse=("oracle_shape_rmse", "mean"),
        component_cross_term=("component_cross_term", "mean"),
        reconstruction_max_abs=("reconstruction_max_abs", "max"),
    )
    .reset_index()
)
macro.to_csv(OUTPUT / "macro.csv", index=False)
output_files = ("source_month.csv", "source_macro.csv", "macro.csv")
summary = {
    "status": "EXPLORATORY_ANNUAL_THERMAL_EF_INFORMATION_PROBE",
    "prediction_contract": "Target monthly five-group totals as-of at monthly forecast origin; source-excluded 2024 local-clock hourly group shapes; vary only thermal EF anchor.",
    "annual_release_sensitivity_days": ANNUAL_RELEASE_DELAYS,
    "annual_release_assumption": "calendar-year annual EF is available Jan 1 of the next year plus the configured 30/60/90/120 day delay; requires 12 monthly fuel records; no true historical annual vintage is present.",
    "target_timezone": TIMEZONES,
    "support": "2025H1: 37 region-months/7 previously inspected regions; 2025H2: 30 region-months/6 previously inspected regions.",
    "input_sha256": {"monthly": sha256(monthly_path), "hourly": hourly_hashes},
    "macro": macro.to_dict("records"),
    "outputs_sha256": {name: sha256(OUTPUT / name) for name in output_files},
    "interpretation": "Exploratory fixed-EF proxy point comparison; not unseen-region confirmation, Chinese validity, interval calibration, or algorithm novelty.",
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(macro.to_string(index=False))
print("\nBY SOURCE\n", source_summary.to_string(index=False))
