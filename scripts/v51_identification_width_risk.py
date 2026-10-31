"""Strict monthly-category-only CIF interval diagnostic for the V51 proxy data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()
REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
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
    thermal_total = sum(values[FUELS.index(fuel)] for fuel in THERMAL)
    if thermal_total <= 0:
        return 0.0
    return float(sum(values[FUELS.index(fuel)] * EF[FUELS.index(fuel)] for fuel in THERMAL) / thermal_total)


def forecast_ambiguity_features(totals: np.ndarray, predicted_groups: np.ndarray, ef: float) -> tuple[float, float]:
    """Mean and p90 proxy width from monthly thermal mass and predicted hourly G.

    This is an as-of feature, not a true identified bound: lagged monthly
    totals are assumed to proxy the target month and thermal EF is donor-only.
    """
    hourly_total = predicted_groups.sum(axis=1)
    month_total = float(totals.sum())
    thermal_total = float(totals[0])
    lower_thermal = np.maximum(0., thermal_total - (month_total - hourly_total))
    upper_thermal = np.minimum(hourly_total, thermal_total)
    width = ef * (upper_thermal - lower_thermal) / np.maximum(hourly_total, 1e-12)
    return float(width.mean()), float(np.quantile(width, .90))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(ordered_values[min(np.searchsorted(cumulative, quantile * cumulative[-1]), len(order) - 1)])


def source_balanced_weights(rows: list[dict]) -> np.ndarray:
    counts = {region: sum(row["region"] == region for row in rows) for region in {r["region"] for r in rows}}
    return np.array([1.0 / counts[row["region"]] for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half", choices=("h1", "h2"), required=True)
    args = parser.parse_args()

    months = (1, 6) if args.half == "h1" else (7, 12)
    half = args.half.upper()
    output_dir = ROOT / f"results/v51-identification-width-risk-r3-{args.half}"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    def asof_coarse_totals(region: str, target_month: pd.Timestamp) -> tuple[np.ndarray, str] | None:
        available = monthly.loc[
            (monthly.region == region)
            & (monthly.period_start.dt.month == target_month.month)
            & (monthly.period_start < target_month)
            & (monthly.period_end <= target_month)
            & (monthly.available_at <= target_month)
        ].sort_values(["period_start", "available_at"])
        if available.empty:
            return None
        row = available.iloc[-1]
        return coarse_totals(fuel_totals(row)), str(row.period_start.date())

    def donor_thermal_ef(donors: tuple[str, ...], target_month: pd.Timestamp, forecast_year: int) -> float:
        values = []
        for donor in donors:
            rows = monthly.loc[
                (monthly.region == donor)
                & (monthly.period_start.dt.month == target_month.month)
                & (monthly.period_start.dt.year == forecast_year - 1)
            ]
            if len(rows) == 1:
                ef = thermal_ef(fuel_totals(rows.iloc[0]))
                if ef > 0:
                    values.append(ef)
        if values:
            return float(np.mean(values))
        return float(np.mean(EF[[0, 1, 2, 7]]))

    def donor_shape(donors: tuple[str, ...], month_no: int, training_year: int) -> np.ndarray:
        profiles = []
        lower = pd.Timestamp(f"{training_year}-01-01", tz="UTC")
        upper = pd.Timestamp(f"{training_year + 1}-01-01", tz="UTC")
        for donor in donors:
            frame = hourly[donor].loc[
                (hourly[donor].time >= lower)
                & (hourly[donor].time < upper)
                & hourly[donor].valid.astype(str).str.lower().isin(("true", "1"))
            ].copy()
            frame["month_no"] = frame.time.dt.month
            frame["hour_no"] = frame.time.dt.hour
            frame = frame.loc[frame.month_no == month_no]
            matrix = np.zeros((24, 5))
            for hour in range(24):
                group = frame.loc[frame.hour_no == hour]
                if group.empty:
                    continue
                matrix[hour, 0] = sum(group[f"generation_{fuel}"].mean() for fuel in THERMAL)
                for index, fuel in enumerate(("nuclear", "hydro", "wind", "solar"), start=1):
                    matrix[hour, index] = group[f"generation_{fuel}"].mean()
            sums = matrix.sum(axis=0)
            matrix = np.divide(matrix, sums[None, :], out=np.ones_like(matrix) / 24, where=sums[None, :] > 0)
            profiles.append(matrix)
        average = np.mean(profiles, axis=0)
        sums = average.sum(axis=0, keepdims=True)
        return np.divide(average, sums, out=np.ones_like(average) / 24, where=sums > 0)

    def point_forecast(region: str, donors: tuple[str, ...], target_month: pd.Timestamp, forecast_year: int) -> dict | None:
        coarse = asof_coarse_totals(region, target_month)
        if coarse is None:
            return None
        totals, reference_month = coarse
        shape = donor_shape(donors, target_month.month, forecast_year - 1)
        frame = hourly[region].loc[
            (hourly[region].time >= target_month)
            & (hourly[region].time < target_month + pd.offsets.MonthBegin(1))
            & hourly[region].valid.astype(str).str.lower().isin(("true", "1"))
        ].copy()
        if frame.empty:
            return None
        actual_groups = np.stack([frame[f"generation_{fuel}"].to_numpy(float) for fuel in FUELS], axis=1)
        valid = np.isfinite(actual_groups).all(axis=1) & (actual_groups >= 0).all(axis=1) & (actual_groups.sum(axis=1) > 0)
        frame = frame.loc[valid]
        actual_groups = actual_groups[valid]
        if len(frame) < 480:
            return None
        hour = frame.time.dt.hour.to_numpy()
        predicted_groups = np.stack(
            [totals[index] * shape[hour, index] / target_month.days_in_month for index in range(5)], axis=1
        )
        ef = donor_thermal_ef(donors, target_month, forecast_year)
        prediction = predicted_groups[:, 0] * ef / np.maximum(predicted_groups.sum(axis=1), 1e-12)
        truth = actual_groups @ EF / actual_groups.sum(axis=1)
        residual = truth - prediction
        total = max(float(totals.sum()), 1e-12)
        features = np.array([totals[0] / total, (totals[3] + totals[4]) / total])
        ambiguity_mean, ambiguity_p90 = forecast_ambiguity_features(totals, predicted_groups, ef)
        features = np.r_[features, ambiguity_mean / 1000., ambiguity_p90 / 1000.]
        return {
            "region": region,
            "month": str(target_month.date()),
            "features": features,
            "prediction": prediction,
            "truth": truth,
            "residual": residual,
            "point_mae": float(np.mean(np.abs(residual))),
            "reference_month": reference_month,
        }

    def half_months(year: int) -> list[pd.Timestamp]:
        return list(pd.date_range(f"{year}-{months[0]:02d}-01", f"{year}-{months[1]:02d}-01", freq="MS", tz="UTC"))

    cache: dict[tuple[int, str, str], dict | None] = {}
    for year in (2024, 2025):
        for region in REGIONS:
            donors = tuple(candidate for candidate in REGIONS if candidate != region)
            for target_month in half_months(year):
                row = point_forecast(region, donors, target_month, year)
                if row is not None:
                    row["q90"] = float(np.quantile(np.abs(row["residual"]), 0.90))
                cache[(year, region, str(target_month.date()))] = row

    results = []
    for heldout in REGIONS:
        donor_regions = tuple(region for region in REGIONS if region != heldout)
        calibration_by_region = {
            region: [
                cache[(2024, region, str(month.date()))]
                for month in half_months(2024)
                if cache[(2024, region, str(month.date()))] is not None
            ]
            for region in donor_regions
        }
        calibration_by_region = {region: rows for region, rows in calibration_by_region.items() if rows}
        target_rows = [
            cache[(2025, heldout, str(month.date()))]
            for month in half_months(2025)
            if cache[(2025, heldout, str(month.date()))] is not None
        ]
        all_abs = {region: np.concatenate([np.abs(row["residual"]) for row in rows]) for region, rows in calibration_by_region.items()}
        global_values, global_weights = [], []
        for values in all_abs.values():
            global_values.extend(values.tolist())
            global_weights.extend(np.full(len(values), 1 / (len(all_abs) * len(values))).tolist())
        global_q90 = weighted_quantile(np.array(global_values), np.array(global_weights), 0.90)

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
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(x_train, y_train, ridge__sample_weight=source_balanced_weights(training_rows))
            x_calibration = np.stack([row["features"] for row in rows])
            scales = np.exp(model.predict(x_calibration))
            for row, scale in zip(rows, scales):
                normalized_by_region[calibration_heldout].extend((np.abs(row["residual"]) / max(float(scale), 1e-6)).tolist())

        normalized_values, normalized_weights = [], []
        for values in normalized_by_region.values():
            if values:
                array = np.asarray(values)
                normalized_values.extend(array.tolist())
                normalized_weights.extend(np.full(len(array), 1 / (len(normalized_by_region) * len(array))).tolist())
        normalized_q90 = weighted_quantile(np.array(normalized_values), np.array(normalized_weights), 0.90)
        source_normalized_q90 = [
            float(np.quantile(values, .90))
            for values in normalized_by_region.values()
            if values
        ]
        max_source_normalized_q90 = max(source_normalized_q90)

        training_rows = [row for source_rows in calibration_by_region.values() for row in source_rows]
        x_train = np.stack([row["features"] for row in training_rows])
        y_train = np.log(np.array([row["q90"] for row in training_rows]) + 1.0)
        final_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        final_model.fit(x_train, y_train, ridge__sample_weight=source_balanced_weights(training_rows))
        scaler = final_model.named_steps["standardscaler"]
        z_train = scaler.transform(x_train)
        z_min, z_max = z_train.min(axis=0), z_train.max(axis=0)
        for row in target_rows:
            conditional_scale = float(np.exp(final_model.predict(row["features"].reshape(1, -1))[0]))
            z_target = scaler.transform(row["features"].reshape(1, -1))[0]
            feature_zmax = float(np.max(np.abs(z_target)))
            outside_train_bbox = bool(np.any(z_target < z_min) or np.any(z_target > z_max))
            for method, radius in (
                ("global_equal_source_q90", global_q90),
            ("conditional_identification_width", conditional_scale * normalized_q90),
                ("conditional_identification_width_max_source", conditional_scale * max_source_normalized_q90),
            ):
                truth, prediction = row["truth"], row["prediction"]
                lower, upper = prediction - radius, prediction + radius
                covered = (truth >= lower) & (truth <= upper)
                score = (upper - lower) + 20 * (lower - truth) * (truth < lower) + 20 * (truth - upper) * (truth > upper)
                results.append(
                    {
                        "heldout_region": heldout,
                        "month": row["month"],
                        "method": method,
                        "n_hours": len(truth),
                        "radius": radius,
                        "coverage": float(covered.mean()),
                        "width": float((upper - lower).mean()),
                        "interval_score": float(score.mean()),
                        "lower_miss": float((truth < lower).mean()),
                        "upper_miss": float((truth > upper).mean()),
                        "point_mae": row["point_mae"],
                        "feature_zmax": feature_zmax,
                        "outside_train_bbox": outside_train_bbox,
                    }
                )

    monthly_results = pd.DataFrame(results)
    monthly_results.to_csv(output_dir / "outer_source_month_intervals.csv", index=False)
    source_summary = (
        monthly_results.groupby(["heldout_region", "method"])
        .agg(
            months=("month", "size"),
            coverage=("coverage", "mean"),
            width=("width", "mean"),
            interval_score=("interval_score", "mean"),
            lower_miss=("lower_miss", "mean"),
            upper_miss=("upper_miss", "mean"),
            point_mae=("point_mae", "mean"),
            feature_zmax_mean=("feature_zmax", "mean"),
            feature_zmax_max=("feature_zmax", "max"),
            outside_train_bbox_fraction=("outside_train_bbox", "mean"),
        )
        .reset_index()
    )
    source_summary.to_csv(output_dir / "source_macro.csv", index=False)
    macro = (
        source_summary.groupby("method")
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
    macro.to_csv(output_dir / "macro.csv", index=False)

    output_files = ("outer_source_month_intervals.csv", "source_macro.csv", "macro.csv")
    summary = {
        "status": "EXPLORATORY_STRICT_MONTHLY_ONLY_MIX_RISK",
        "information_contract": "target supplies only legally published lagged five-group monthly generation totals; point thermal EF and hourly profile are donor-only; risk scale uses local monthly mix plus donor-implied target-month thermal reallocation ambiguity",
        "point_model": "target lagged five-group monthly totals combined with source-excluded prior-year five-group hourly shapes and source-equal prior-year thermal EF",
        "risk_model": "source-balanced Ridge(alpha=10) predicts log monthly q90 absolute residual from local thermal share, wind-plus-solar share, and donor-implied monthly-to-hourly thermal ambiguity mean/p90; source-held-out OOF scale predictions calibrate normalized q90",
        "comparison": "same strict-monthly-only point forecasts, ambiguity-augmented pooled normalized-q90 calibration and max-source normalized-q90 calibration versus global pooled q90",
        "support": f"{monthly_results[['heldout_region', 'month']].drop_duplicates().shape[0]} available heldout region-months across {source_summary.heldout_region.nunique()} previously inspected regions; 2024{half} calibration and 2025{half} evaluation; not unseen-region confirmation",
        "input_sha256": {"monthly": sha256(monthly_path), "hourly": hourly_hashes},
        "macro": macro.to_dict("records"),
        "outputs_sha256": {name: sha256(output_dir / name) for name in output_files},
        "interpretation": "Exploratory repeated-source forward diagnostics only; no arbitrary-domain-shift coverage guarantee or novelty claim.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print("\nBY SOURCE\n", source_summary.to_string(index=False))


if __name__ == "__main__":
    main()
