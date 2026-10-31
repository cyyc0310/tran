"""Compare local-clock 24h, weekday/weekend, and weekday-hour donor shapes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path.cwd()
OUTPUT = ROOT / "results/v51-daytype-shape-probe"
OUTPUT.mkdir(parents=True, exist_ok=True)
REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
TIMEZONES = {
    "BPAT": "America/Los_Angeles",
    "CISO": "America/Los_Angeles",
    "ERCO": "America/Chicago",
    "FPL": "America/New_York",
    "ISNE": "America/New_York",
    "MISO": "America/Chicago",  # MISO spans time zones; Central is a representative convention.
    "NYIS": "America/New_York",
}
FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
THERMAL = ("coal", "gas", "oil", "other")
EF = np.array([1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 400.0])
METHODS = ("hour24_local", "daytype_hour48", "weekday_hour168")


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


def coarse_asof(region: str, target_month: pd.Timestamp) -> np.ndarray | None:
    available = monthly.loc[
        (monthly.region == region)
        & (monthly.period_start.dt.month == target_month.month)
        & (monthly.period_start < target_month)
        & (monthly.period_end <= target_month)
        & (monthly.available_at <= target_month)
    ].sort_values(["period_start", "available_at"])
    return coarse_totals(fuel_totals(available.iloc[-1])) if not available.empty else None


def thermal_ef_donor(donors: tuple[str, ...], target_month: pd.Timestamp, forecast_year: int) -> float:
    estimates = []
    for donor in donors:
        rows = monthly.loc[
            (monthly.region == donor)
            & (monthly.period_start.dt.month == target_month.month)
            & (monthly.period_start.dt.year == forecast_year - 1)
        ]
        if len(rows) == 1:
            estimate = thermal_ef(fuel_totals(rows.iloc[0]))
            if estimate > 0:
                estimates.append(estimate)
    return float(np.mean(estimates)) if estimates else float(np.mean(EF[[0, 1, 2, 7]]))


def local_fields(region: str, timestamps: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    local = timestamps.dt.tz_convert(ZoneInfo(TIMEZONES[region]))
    hour = local.dt.hour.to_numpy()
    weekday = local.dt.weekday.to_numpy()
    return hour, weekday


def slot_ids(region: str, timestamps: pd.Series, method: str) -> np.ndarray:
    hour, weekday = local_fields(region, timestamps)
    if method == "hour24_local":
        return hour
    if method == "daytype_hour48":
        return (weekday >= 5).astype(int) * 24 + hour
    return weekday * 24 + hour


def slot_count(method: str) -> int:
    return {"hour24_local": 24, "daytype_hour48": 48, "weekday_hour168": 168}[method]


def donor_shape(donors: tuple[str, ...], target_month: pd.Timestamp, training_year: int, method: str) -> np.ndarray:
    lower = pd.Timestamp(f"{training_year}-01-01", tz="UTC")
    upper = pd.Timestamp(f"{training_year + 1}-01-01", tz="UTC")
    n_slots = slot_count(method)
    donor_profiles = []
    for donor in donors:
        frame = hourly[donor].loc[
            (hourly[donor].time >= lower)
            & (hourly[donor].time < upper)
            & (hourly[donor].time.dt.month == target_month.month)
            & hourly[donor].valid.astype(str).str.lower().isin(("true", "1"))
        ].copy()
        frame["slot"] = slot_ids(donor, frame.time, method)
        data = np.stack(
            [sum(frame[f"generation_{fuel}"].to_numpy(float) for fuel in THERMAL)]
            + [frame[f"generation_{fuel}"].to_numpy(float) for fuel in ("nuclear", "hydro", "wind", "solar")],
            axis=1,
        )
        matrix = np.zeros((n_slots, 5))
        for slot in range(n_slots):
            rows = frame.slot.to_numpy() == slot
            if rows.any():
                matrix[slot] = data[rows].mean(axis=0)
        if method != "hour24_local":
            fallback = donor_shape((donor,), target_month, training_year, "hour24_local")
            hours = np.arange(n_slots) % 24
            for slot in range(n_slots):
                if not np.any(matrix[slot]):
                    matrix[slot] = fallback[hours[slot]]
        sums = matrix.sum(axis=0)
        donor_profiles.append(np.divide(matrix, sums[None, :], out=np.ones_like(matrix) / n_slots, where=sums[None, :] > 0))
    average = np.mean(donor_profiles, axis=0)
    sums = average.sum(axis=0, keepdims=True)
    return np.divide(average, sums, out=np.ones_like(average) / n_slots, where=sums > 0)


def predict(region: str, donors: tuple[str, ...], target_month: pd.Timestamp, method: str) -> dict | None:
    totals = coarse_asof(region, target_month)
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

    shape = donor_shape(donors, target_month, target_month.year - 1, method)
    slots = slot_ids(region, target.time, method)
    raw_groups = shape[slots]
    denominator = raw_groups.sum(axis=0)
    predicted_groups = np.divide(
        raw_groups * totals[None, :],
        denominator[None, :],
        out=np.zeros_like(raw_groups),
        where=denominator[None, :] > 0,
    )
    ef = thermal_ef_donor(donors, target_month, target_month.year)
    point = predicted_groups[:, 0] * ef / np.maximum(predicted_groups.sum(axis=1), 1e-12)
    truth = actual @ EF / actual.sum(axis=1)
    error = truth - point
    return {
        "heldout_region": region,
        "month": str(target_month.date()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


records = []
for half, first_month, last_month in (("H1", 1, 6), ("H2", 7, 12)):
    target_months = pd.date_range(f"2025-{first_month:02d}-01", f"2025-{last_month:02d}-01", freq="MS", tz="UTC")
    for heldout in REGIONS:
        donors = tuple(region for region in REGIONS if region != heldout)
        for target_month in target_months:
            for method in METHODS:
                result = predict(heldout, donors, target_month, method)
                if result is not None:
                    records.append({**result, "half": half, "shape_method": method})

monthly_scores = pd.DataFrame(records)
monthly_scores.to_csv(OUTPUT / "source_month.csv", index=False)
source_summary = (
    monthly_scores.groupby(["half", "heldout_region", "shape_method"])
    .agg(months=("month", "size"), mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"))
    .reset_index()
)
source_summary.to_csv(OUTPUT / "source_macro.csv", index=False)
macro = (
    source_summary.groupby(["half", "shape_method"])
    .agg(sources=("heldout_region", "nunique"), mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"))
    .reset_index()
)
macro.to_csv(OUTPUT / "macro.csv", index=False)
output_files = ("source_month.csv", "source_macro.csv", "macro.csv")
summary = {
    "status": "EXPLORATORY_DAYTYPE_SHAPE_BASELINE_PROBE",
    "input_contract": "target region uses legally available lagged five-group monthly totals only; source-excluded prior-year source hourly shapes and donor thermal EF; no target hourly inputs",
    "methods": METHODS,
    "regional_timezone": TIMEZONES,
    "MISO_note": "MISO spans multiple time zones; America/Chicago is a representative convention.",
    "support": "2025H1: 37 region-months/7 regions; 2025H2: 30 region-months/6 regions; all regions previously inspected.",
    "input_sha256": {"monthly": sha256(monthly_path), "hourly": input_hashes},
    "macro": macro.to_dict("records"),
    "outputs_sha256": {name: sha256(OUTPUT / name) for name in output_files},
    "interpretation": "Calendar-conditional profiles are standard forecasting baselines, not an algorithm novelty claim.",
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(macro.to_string(index=False))
print("\nBY SOURCE\n", source_summary.to_string(index=False))
