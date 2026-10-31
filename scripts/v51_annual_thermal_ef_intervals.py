"""Forward interval comparison for local annual versus monthly thermal-EF anchors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()
OUTPUT = ROOT / "results/v51-annual-thermal-ef-intervals"
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
METHODS = ("donor_same_month", "local_lag_same_month", "local_latest_annual_d90d")
HALVES = {"H1": (1, 6), "H2": (7, 12)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fuel_totals(row: pd.Series) -> np.ndarray:
    return float(row.total_mwh) * np.array([float(row[f"fuelshare_{fuel}"]) for fuel in FUELS])


def thermal_ef(values: np.ndarray) -> float:
    total = sum(values[FUELS.index(fuel)] for fuel in THERMAL)
    return float(sum(values[FUELS.index(fuel)] * EF[FUELS.index(fuel)] for fuel in THERMAL) / total) if total > 0 else float("nan")


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


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(ordered[min(np.searchsorted(cumulative, quantile * cumulative[-1]), len(order) - 1)])


def source_balanced_weights(rows: list[dict]) -> np.ndarray:
    counts = {region: sum(row["region"] == region for row in rows) for region in {row["region"] for row in rows}}
    return np.array([1.0 / counts[row["region"]] for row in rows])


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


def asof_month(region: str, target_month: pd.Timestamp) -> pd.Series | None:
    rows = monthly.loc[
        (monthly.region == region)
        & (monthly.period_start.dt.month == target_month.month)
        & (monthly.period_start < target_month)
        & (monthly.period_end <= target_month)
        & (monthly.available_at <= target_month)
    ].sort_values(["period_start", "available_at"])
    return None if rows.empty else rows.iloc[-1]


def asof_coarse_totals(region: str, target_month: pd.Timestamp) -> np.ndarray | None:
    row = asof_month(region, target_month)
    return coarse_totals(fuel_totals(row)) if row is not None else None


def annual_ef_asof(region: str, target_month: pd.Timestamp) -> float:
    years = sorted(monthly.loc[monthly.region == region, "period_start"].dt.year.unique(), reverse=True)
    for year in years:
        rows = monthly.loc[(monthly.region == region) & (monthly.period_start.dt.year == year)].sort_values("period_start")
        if len(rows) != 12 or rows.period_start.dt.month.tolist() != list(range(1, 13)):
            continue
        release = pd.Timestamp(f"{year + 1}-01-01", tz="UTC") + pd.Timedelta(days=90)
        if release > target_month:
            continue
        total = np.sum(np.stack([fuel_totals(row) for _, row in rows.iterrows()]), axis=0)
        return thermal_ef(total)
    return float("nan")


def local_same_month_ef(region: str, target_month: pd.Timestamp) -> float:
    row = asof_month(region, target_month)
    return thermal_ef(fuel_totals(row)) if row is not None else float("nan")


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
            if np.isfinite(value):
                values.append(value)
    return float(np.mean(values)) if values else float(np.mean(EF[[0, 1, 2, 7]]))


def hourly_shape(donors: tuple[str, ...], target_month: pd.Timestamp, training_year: int) -> np.ndarray:
    lower, upper = pd.Timestamp(f"{training_year}-01-01", tz="UTC"), pd.Timestamp(f"{training_year + 1}-01-01", tz="UTC")
    source_shapes = []
    for donor in donors:
        frame = hourly[donor].loc[
            (hourly[donor].time >= lower)
            & (hourly[donor].time < upper)
            & (hourly[donor].time.dt.month == target_month.month)
            & hourly[donor].valid.astype(str).str.lower().isin(("true", "1"))
        ].copy()
        frame["local_hour"] = frame.time.dt.tz_convert(ZoneInfo(TIMEZONES[donor])).dt.hour
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


def forecast(region: str, target_month: pd.Timestamp, method: str) -> dict | None:
    donors = tuple(other for other in REGIONS if other != region)
    totals = asof_coarse_totals(region, target_month)
    if totals is None:
        return None
    if method == "donor_same_month":
        factor = donor_ef(donors, target_month, target_month.year)
    elif method == "local_lag_same_month":
        factor = local_same_month_ef(region, target_month)
    else:
        factor = annual_ef_asof(region, target_month)
    if not np.isfinite(factor):
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
    target, actual = target.loc[valid], actual[valid]
    if len(target) < 480:
        return None

    shape = hourly_shape(donors, target_month, target_month.year - 1)
    local_hour = target.time.dt.tz_convert(ZoneInfo(TIMEZONES[region])).dt.hour.to_numpy()
    raw = shape[local_hour]
    denominators = raw.sum(axis=0)
    predicted_groups = np.divide(raw * totals[None, :], denominators[None, :], out=np.zeros_like(raw), where=denominators[None, :] > 0)
    predicted = predicted_groups[:, 0] * factor / np.maximum(predicted_groups.sum(axis=1), 1e-12)
    truth = actual @ EF / actual.sum(axis=1)
    error = truth - predicted
    total = max(float(totals.sum()), 1e-12)
    return {
        "region": region,
        "month": str(target_month.date()),
        "features": np.array([totals[0] / total, (totals[3] + totals[4]) / total]),
        "truth": truth,
        "prediction": predicted,
        "residual": error,
        "q90": float(np.quantile(np.abs(error), 0.90)),
    }


def months_for(year: int, half: str) -> list[pd.Timestamp]:
    first, last = HALVES[half]
    return list(pd.date_range(f"{year}-{first:02d}-01", f"{year}-{last:02d}-01", freq="MS", tz="UTC"))


cache: dict[tuple[str, int, str, str, str], dict | None] = {}
for method in METHODS:
    for year in (2024, 2025):
        for half in HALVES:
            for region in REGIONS:
                for target_month in months_for(year, half):
                    cache[(method, year, half, region, str(target_month.date()))] = forecast(region, target_month, method)


records = []
for method in METHODS:
    for half in HALVES:
        for heldout in REGIONS:
            calibration_regions = [region for region in REGIONS if region != heldout]
            calibration_by_region = {
                region: [
                    cache[(method, 2024, half, region, str(month.date()))]
                    for month in months_for(2024, half)
                    if cache[(method, 2024, half, region, str(month.date()))] is not None
                ]
                for region in calibration_regions
            }
            calibration_by_region = {region: rows for region, rows in calibration_by_region.items() if rows}
            target_rows = [
                cache[(method, 2025, half, heldout, str(month.date()))]
                for month in months_for(2025, half)
                if cache[(method, 2025, half, heldout, str(month.date()))] is not None
            ]
            if not calibration_by_region or not target_rows:
                continue

            values, weights = [], []
            for rows in calibration_by_region.values():
                residuals = np.concatenate([np.abs(row["residual"]) for row in rows])
                values.extend(residuals.tolist())
                weights.extend(np.full(len(residuals), 1 / (len(calibration_by_region) * len(residuals))).tolist())
            global_q90 = weighted_quantile(np.asarray(values), np.asarray(weights), 0.90)

            normalized_by_region: dict[str, list[float]] = {region: [] for region in calibration_by_region}
            for calibration_heldout, rows in calibration_by_region.items():
                training_rows = [
                    row
                    for region, source_rows in calibration_by_region.items()
                    if region != calibration_heldout
                    for row in source_rows
                ]
                if len(training_rows) < 8:
                    continue
                x_train = np.stack([row["features"] for row in training_rows])
                y_train = np.log(np.array([row["q90"] for row in training_rows]) + 1.0)
                scale_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                scale_model.fit(x_train, y_train, ridge__sample_weight=source_balanced_weights(training_rows))
                x_calibration = np.stack([row["features"] for row in rows])
                scales = np.exp(scale_model.predict(x_calibration))
                for row, scale in zip(rows, scales):
                    normalized_by_region[calibration_heldout].extend((np.abs(row["residual"]) / max(float(scale), 1e-6)).tolist())

            normalized_values, normalized_weights = [], []
            for source_values in normalized_by_region.values():
                if source_values:
                    array = np.asarray(source_values)
                    normalized_values.extend(array.tolist())
                    normalized_weights.extend(np.full(len(array), 1 / (len(normalized_by_region) * len(array))).tolist())
            normalized_q90 = weighted_quantile(np.asarray(normalized_values), np.asarray(normalized_weights), 0.90)

            training_rows = [row for source_rows in calibration_by_region.values() for row in source_rows]
            x_train = np.stack([row["features"] for row in training_rows])
            y_train = np.log(np.array([row["q90"] for row in training_rows]) + 1.0)
            final_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            final_model.fit(x_train, y_train, ridge__sample_weight=source_balanced_weights(training_rows))

            for row in target_rows:
                conditional_scale = float(np.exp(final_model.predict(row["features"].reshape(1, -1))[0]))
                for risk_method, radius in (
                    ("global_equal_source_q90", global_q90),
                    ("conditional_mix_scale", conditional_scale * normalized_q90),
                ):
                    truth, predicted = row["truth"], row["prediction"]
                    lower, upper = predicted - radius, predicted + radius
                    covered = (truth >= lower) & (truth <= upper)
                    interval_score = (upper - lower) + 20 * (lower - truth) * (truth < lower) + 20 * (truth - upper) * (truth > upper)
                    records.append(
                        {
                            "half": half,
                            "heldout_region": heldout,
                            "month": row["month"],
                            "ef_method": method,
                            "risk_method": risk_method,
                            "n_hours": len(truth),
                            "radius": radius,
                            "coverage": float(covered.mean()),
                            "width": float((upper - lower).mean()),
                            "interval_score": float(interval_score.mean()),
                            "lower_miss": float((truth < lower).mean()),
                            "upper_miss": float((truth > upper).mean()),
                            "point_mae": float(np.mean(np.abs(truth - predicted))),
                        }
                    )

monthly_results = pd.DataFrame(records)
monthly_results.to_csv(OUTPUT / "outer_source_month_intervals.csv", index=False)
source_summary = (
    monthly_results.groupby(["half", "heldout_region", "ef_method", "risk_method"])
    .agg(
        months=("month", "size"),
        coverage=("coverage", "mean"),
        width=("width", "mean"),
        interval_score=("interval_score", "mean"),
        lower_miss=("lower_miss", "mean"),
        upper_miss=("upper_miss", "mean"),
        point_mae=("point_mae", "mean"),
    )
    .reset_index()
)
source_summary.to_csv(OUTPUT / "source_macro.csv", index=False)
macro = (
    source_summary.groupby(["half", "ef_method", "risk_method"])
    .agg(
        sources=("heldout_region", "nunique"),
        coverage=("coverage", "mean"),
        width=("width", "mean"),
        interval_score=("interval_score", "mean"),
        lower_miss=("lower_miss", "mean"),
        upper_miss=("upper_miss", "mean"),
        point_mae=("point_mae", "mean"),
    )
    .reset_index()
)
macro.to_csv(OUTPUT / "macro.csv", index=False)
outputs = ("outer_source_month_intervals.csv", "source_macro.csv", "macro.csv")
summary = {
    "status": "EXPLORATORY_ANNUAL_EF_FORWARD_INTERVAL_COMPARISON",
    "point_information": "strict as-of five-group monthly target totals; EF variants donor same-month, local same-month, or local latest annual with 90-day assumed release delay; source-excluded prior-year local-clock hourly shapes",
    "risk_methods": ["source-balanced pooled q90", "two-feature mix-only log-q90 Ridge with source-held-out normalized q90 calibration"],
    "temporal_protocol": "2024H1/H2 calibration to 2025H1/H2 scoring; each heldout region excluded from point donors and risk calibration",
    "annual_release_assumption": "annual EF release is Jan 1 of following year plus 90 days; no historical annual vintage exists",
    "support": "Previously inspected US proxy regions; 2025H1 has 37 region-months/7 sources, 2025H2 has 30 region-months/6 sources.",
    "input_sha256": {"monthly": sha256(monthly_path), "hourly": hourly_hashes},
    "macro": macro.to_dict("records"),
    "outputs_sha256": {name: sha256(OUTPUT / name) for name in outputs},
    "interpretation": "Exploratory only; repeated sources, synthetic annual release timing, fixed factors; no Chinese validity, arbitrary domain-shift coverage guarantee, or novelty claim.",
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(macro.to_string(index=False))
print("\nWORST SOURCE COVERAGE\n", source_summary.loc[source_summary.groupby(["half", "ef_method", "risk_method"]).coverage.idxmin()].to_string(index=False))
