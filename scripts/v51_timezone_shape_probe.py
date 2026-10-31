"""Compare UTC-hour and regional-clock donor shapes under a monthly-only target budget."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path.cwd()
OUTPUT = ROOT / "results/v51-timezone-profile-probe"
OUTPUT.mkdir(parents=True, exist_ok=True)
REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
TIMEZONES = {
    "BPAT": "America/Los_Angeles",
    "CISO": "America/Los_Angeles",
    "ERCO": "America/Chicago",
    "FPL": "America/New_York",
    "ISNE": "America/New_York",
    "MISO": "America/Chicago",  # MISO spans time zones; Central is its representative convention.
    "NYIS": "America/New_York",
}
FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
THERMAL = ("coal", "gas", "oil", "other")
EF = np.array([1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 400.0])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fuel_totals(row: pd.Series) -> np.ndarray:
    return float(row.total_mwh) * np.array([float(row[f"fuelshare_{fuel}"]) for fuel in FUELS])


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


def thermal_ef(values: np.ndarray) -> float:
    total = sum(values[FUELS.index(fuel)] for fuel in THERMAL)
    if total <= 0:
        return 0.0
    return float(sum(values[FUELS.index(fuel)] * EF[FUELS.index(fuel)] for fuel in THERMAL) / total)


monthly_path = ROOT / "results/heldout-2025-data/monthly.csv"
monthly = pd.read_csv(monthly_path)
for column in ("period_start", "period_end", "available_at"):
    monthly[column] = pd.to_datetime(monthly[column], utc=True)

hourly: dict[str, pd.DataFrame] = {}
input_hashes: dict[str, str] = {}
for region in REGIONS:
    path = ROOT / f"results/heldout-2025-data/{region}_hourly.csv"
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    hourly[region] = frame
    input_hashes[str(path)] = sha256(path)


def asof_totals(region: str, target_month: pd.Timestamp) -> np.ndarray | None:
    available = monthly.loc[
        (monthly.region == region)
        & (monthly.period_start.dt.month == target_month.month)
        & (monthly.period_start < target_month)
        & (monthly.period_end <= target_month)
        & (monthly.available_at <= target_month)
    ].sort_values(["period_start", "available_at"])
    if available.empty:
        return None
    return coarse_totals(fuel_totals(available.iloc[-1]))


def profile_clock_hours(region: str, timestamps: pd.Series, mode: str) -> pd.Series:
    if mode == "utc":
        return timestamps.dt.hour
    return timestamps.dt.tz_convert(ZoneInfo(TIMEZONES[region])).dt.hour


def donor_shape(donors: tuple[str, ...], month_no: int, training_year: int, mode: str) -> np.ndarray:
    source_profiles = []
    lower = pd.Timestamp(f"{training_year}-01-01", tz="UTC")
    upper = pd.Timestamp(f"{training_year + 1}-01-01", tz="UTC")
    for donor in donors:
        frame = hourly[donor].loc[
            (hourly[donor].time >= lower)
            & (hourly[donor].time < upper)
            & hourly[donor].valid.astype(str).str.lower().isin(("true", "1"))
        ].copy()
        frame["month_no"] = frame.time.dt.month
        frame["profile_hour"] = profile_clock_hours(donor, frame.time, mode)
        frame = frame.loc[frame.month_no == month_no]
        matrix = np.zeros((24, 5))
        for hour in range(24):
            rows = frame.loc[frame.profile_hour == hour]
            if rows.empty:
                continue
            matrix[hour, 0] = sum(rows[f"generation_{fuel}"].mean() for fuel in THERMAL)
            for index, fuel in enumerate(("nuclear", "hydro", "wind", "solar"), start=1):
                matrix[hour, index] = rows[f"generation_{fuel}"].mean()
        sums = matrix.sum(axis=0)
        source_profiles.append(np.divide(matrix, sums[None, :], out=np.ones_like(matrix) / 24, where=sums[None, :] > 0))
    average = np.mean(source_profiles, axis=0)
    sums = average.sum(axis=0, keepdims=True)
    return np.divide(average, sums, out=np.ones_like(average) / 24, where=sums > 0)


def donor_ef(donors: tuple[str, ...], target_month: pd.Timestamp, forecast_year: int) -> float:
    values = []
    for donor in donors:
        rows = monthly.loc[
            (monthly.region == donor)
            & (monthly.period_start.dt.month == target_month.month)
            & (monthly.period_start.dt.year == forecast_year - 1)
        ]
        if len(rows) == 1:
            value = thermal_ef(fuel_totals(rows.iloc[0]))
            if value > 0:
                values.append(value)
    return float(np.mean(values)) if values else float(np.mean(EF[[0, 1, 2, 7]]))


def predict(region: str, donors: tuple[str, ...], target_month: pd.Timestamp, forecast_year: int, mode: str) -> dict | None:
    totals = asof_totals(region, target_month)
    if totals is None:
        return None
    target = hourly[region].loc[
        (hourly[region].time >= target_month)
        & (hourly[region].time < target_month + pd.offsets.MonthBegin(1))
        & hourly[region].valid.astype(str).str.lower().isin(("true", "1"))
    ].copy()
    if target.empty:
        return None
    actual = np.stack([target[f"generation_{fuel}"].to_numpy(float) for fuel in FUELS], axis=1)
    valid = np.isfinite(actual).all(axis=1) & (actual >= 0).all(axis=1) & (actual.sum(axis=1) > 0)
    target = target.loc[valid]
    actual = actual[valid]
    if len(target) < 480:
        return None

    shape = donor_shape(donors, target_month.month, forecast_year - 1, mode)
    local_hour = profile_clock_hours(region, target.time, mode).to_numpy()
    raw_groups = np.stack([shape[local_hour, index] for index in range(5)], axis=1)
    denominator = raw_groups.sum(axis=0)
    predicted_groups = np.divide(
        raw_groups * totals[None, :],
        denominator[None, :],
        out=np.zeros_like(raw_groups),
        where=denominator[None, :] > 0,
    )
    ef = donor_ef(donors, target_month, forecast_year)
    prediction = predicted_groups[:, 0] * ef / np.maximum(predicted_groups.sum(axis=1), 1e-12)
    truth = actual @ EF / actual.sum(axis=1)
    residual = truth - prediction
    return {
        "region": region,
        "month": str(target_month.date()),
        "n_hours": len(target),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
    }


records = []
for year in (2025,):
    for half, first_month, last_month in (("H1", 1, 6), ("H2", 7, 12)):
        for region in REGIONS:
            donors = tuple(other for other in REGIONS if other != region)
            months = pd.date_range(f"{year}-{first_month:02d}-01", f"{year}-{last_month:02d}-01", freq="MS", tz="UTC")
            for target_month in months:
                for mode in ("utc", "regional_clock"):
                    row = predict(region, donors, target_month, year, mode)
                    if row is not None:
                        records.append(
                            {
                                "heldout_region": row["region"],
                                **{key: value for key, value in row.items() if key != "region"},
                                "half": half,
                                "clock_alignment": mode,
                            }
                        )

monthly_scores = pd.DataFrame(records)
monthly_scores.to_csv(OUTPUT / "source_month.csv", index=False)
source_summary = (
    monthly_scores.groupby(["half", "heldout_region", "clock_alignment"])
    .agg(months=("month", "size"), mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"))
    .reset_index()
)
source_summary.to_csv(OUTPUT / "source_macro.csv", index=False)
macro = (
    source_summary.groupby(["half", "clock_alignment"])
    .agg(sources=("heldout_region", "nunique"), mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"))
    .reset_index()
)
macro.to_csv(OUTPUT / "macro.csv", index=False)
output_files = ("source_month.csv", "source_macro.csv", "macro.csv")
summary = {
    "status": "EXPLORATORY_TIMEZONE_SHAPE_BASELINE_PROBE",
    "task": "same strict-monthly-only donor-EF point forecast, varying only whether donor and target hourly profiles align by UTC or regional local clock",
    "regional_timezone": TIMEZONES,
    "MISO_note": "MISO spans multiple time zones; America/Chicago is a representative primary convention and requires sensitivity analysis before operational interpretation.",
    "support": "Previously inspected US EIA proxy regions; 2025H1 and 2025H2; not unseen-region confirmation.",
    "input_sha256": {"monthly": sha256(monthly_path), "hourly": input_hashes},
    "macro": macro.to_dict("records"),
    "outputs_sha256": {name: sha256(OUTPUT / name) for name in output_files},
    "interpretation": "Calendar alignment is a standard baseline choice, not an algorithm novelty claim.",
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(macro.to_string(index=False))
print("\nBY SOURCE\n", source_summary.to_string(index=False))
