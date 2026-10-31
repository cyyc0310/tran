"""Value-test local monthly thermal-emissions-factor anchors on seen EIA regions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("AVA", "BANC", "BPAT", "FPL", "JEA", "MISO", "NWMT", "NYIS", "SC", "SCEG", "SEC")
FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
THERMAL = ("coal", "gas", "oil", "other")
EF = np.array([1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 400.0])
ROOT_BY_REGION = {
    "AVA": "results/external-v4-data", "BANC": "results/external-v4-data",
    "BPAT": "results/heldout-2025-data", "FPL": "results/heldout-2025-data",
    "JEA": "results/external-v4-data", "MISO": "results/heldout-2025-data",
    "NWMT": "results/external-v4-data", "NYIS": "results/heldout-2025-data",
    "SC": "results/external-v4-data", "SCEG": "results/external-v5-data",
    "SEC": "results/external-v5-data",
}
METHODS = ("donor_same_month", "local_same_month_lag", "local_latest_asof_month",
           "local_latest_complete_year_asof")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fuel_totals(row: pd.Series) -> np.ndarray:
    return float(row.total_mwh) * np.array([float(row[f"fuelshare_{fuel}"]) for fuel in FUELS])


def thermal_ef(values: np.ndarray) -> float:
    indexes = [FUELS.index(fuel) for fuel in THERMAL]
    denominator = float(values[indexes].sum())
    return float(values[indexes] @ EF[indexes] / denominator) if denominator > 0 else float("nan")


def coarse_totals(values: np.ndarray) -> np.ndarray:
    return np.array([sum(values[FUELS.index(f)] for f in THERMAL), values[3], values[4], values[5], values[6]])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    vals, wts = values[order], weights[order]
    return float(vals[min(np.searchsorted(np.cumsum(wts), quantile * wts.sum()), len(vals) - 1)])


def main() -> None:
    monthly_parts, hourly, input_hashes = [], {}, {}
    for region, relative in ROOT_BY_REGION.items():
        folder = ROOT / relative
        monthly_path, hourly_path = folder / "monthly.csv", folder / f"{region}_hourly.csv"
        monthly = pd.read_csv(monthly_path)
        monthly = monthly.loc[monthly.region.eq(region)].copy()
        for column in ("period_start", "period_end", "available_at"):
            monthly[column] = pd.to_datetime(monthly[column], utc=True)
        monthly_parts.append(monthly)
        frame = pd.read_csv(hourly_path)
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["_year"], frame["_month"], frame["_hour"] = frame.time.dt.year, frame.time.dt.month, frame.time.dt.hour
        frame["_valid"] = frame.valid.astype(str).str.lower().isin(("true", "1"))
        hourly[region] = frame
        input_hashes[str(monthly_path.resolve())] = sha256(monthly_path)
        input_hashes[str(hourly_path.resolve())] = sha256(hourly_path)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    if monthly.duplicated(["region", "period_start"]).any():
        raise ValueError("Duplicate region-month records")

    def row_for(region: str, period: pd.Timestamp, origin: pd.Timestamp) -> pd.Series | None:
        rows = monthly.loc[monthly.region.eq(region) & monthly.period_start.eq(period)
                           & monthly.period_end.le(origin) & monthly.available_at.le(origin)]
        return rows.iloc[-1] if len(rows) else None

    def asof_totals(region: str, origin: pd.Timestamp) -> tuple[np.ndarray, str] | None:
        candidates = monthly.loc[monthly.region.eq(region)
                                 & monthly.period_start.dt.month.eq(origin.month)
                                 & monthly.period_start.lt(origin)
                                 & monthly.period_end.le(origin) & monthly.available_at.le(origin)]
        if candidates.empty:
            return None
        row = candidates.sort_values(["period_start", "available_at"]).iloc[-1]
        return coarse_totals(fuel_totals(row)), str(row.period_start.date())

    def estimate_ef(method: str, region: str, donors: tuple[str, ...], origin: pd.Timestamp,
                    forecast_year: int) -> tuple[float, str] | None:
        if method == "donor_same_month":
            values = []
            period = pd.Timestamp(f"{forecast_year - 1}-{origin.month:02d}-01", tz="UTC")
            for donor in donors:
                row = row_for(donor, period, origin)
                if row is not None:
                    value = thermal_ef(fuel_totals(row))
                    if np.isfinite(value):
                        values.append(value)
            return (float(np.mean(values)), f"{len(values)}_donors") if values else None
        available = monthly.loc[monthly.region.eq(region) & monthly.period_end.le(origin)
                                & monthly.available_at.le(origin)]
        if method == "local_same_month_lag":
            period = pd.Timestamp(f"{forecast_year - 1}-{origin.month:02d}-01", tz="UTC")
            row = row_for(region, period, origin)
            if row is None:
                return None
            return thermal_ef(fuel_totals(row)), str(period.date())
        if method == "local_latest_asof_month":
            if available.empty:
                return None
            row = available.sort_values(["period_end", "available_at"]).iloc[-1]
            return thermal_ef(fuel_totals(row)), str(row.period_start.date())
        if method == "local_latest_complete_year_asof":
            years = sorted(available.period_start.dt.year.unique(), reverse=True)
            for year in years:
                if year >= forecast_year:
                    continue
                rows = available.loc[available.period_start.dt.year.eq(year)]
                if rows.period_start.dt.month.nunique() == 12:
                    totals = np.sum([fuel_totals(row) for _, row in rows.iterrows()], axis=0)
                    return thermal_ef(totals), str(year)
            return None
        raise ValueError(method)

    shape_cache: dict[tuple[str, int, int], np.ndarray] = {}

    def shape(region: str, month: int, year: int) -> np.ndarray:
        key = (region, month, year)
        if key in shape_cache:
            return shape_cache[key]
        frame = hourly[region].loc[hourly[region]._year.eq(year) & hourly[region]._month.eq(month)
                                   & hourly[region]._valid]
        fuel_cols = [f"generation_{fuel}" for fuel in FUELS]
        matrix = np.zeros((24, 5))
        if not frame.empty:
            means = frame.groupby("_hour")[fuel_cols].mean()
            for hour, values in means.iterrows():
                matrix[int(hour), 0] = sum(float(values[f"generation_{fuel}"]) for fuel in THERMAL)
                for idx, fuel in enumerate(("nuclear", "hydro", "wind", "solar"), start=1):
                    matrix[int(hour), idx] = float(values[f"generation_{fuel}"])
        sums = matrix.sum(axis=0)
        matrix = np.divide(matrix, sums[None, :], out=np.ones_like(matrix) / 24, where=sums[None, :] > 0)
        shape_cache[key] = matrix
        return matrix

    def predict(region: str, donors: tuple[str, ...], origin: pd.Timestamp, year: int,
                method: str) -> dict | None:
        coarse = asof_totals(region, origin)
        factor = estimate_ef(method, region, donors, origin, year)
        if coarse is None or factor is None or not np.isfinite(factor[0]):
            return None
        totals, reference = coarse
        frame = hourly[region].loc[hourly[region]._year.eq(origin.year)
                                   & hourly[region]._month.eq(origin.month) & hourly[region]._valid].copy()
        if frame.empty:
            return None
        actual = np.stack([frame[f"generation_{f}"].to_numpy(float) for f in FUELS], axis=1)
        valid = np.isfinite(actual).all(axis=1) & (actual >= 0).all(axis=1) & (actual.sum(axis=1) > 0)
        frame, actual = frame.loc[valid], actual[valid]
        if len(frame) < 480:
            return None
        profiles = np.mean([shape(donor, origin.month, year - 1) for donor in donors], axis=0)
        profiles /= np.maximum(profiles.sum(axis=0, keepdims=True), 1e-12)
        predicted_groups = np.stack([totals[j] * profiles[frame._hour.to_numpy(), j] / origin.days_in_month
                                     for j in range(5)], axis=1)
        prediction = predicted_groups[:, 0] * factor[0] / np.maximum(predicted_groups.sum(axis=1), 1e-12)
        truth = actual @ EF / actual.sum(axis=1)
        residual = truth - prediction
        return {"region": region, "month": str(origin.date()), "truth": truth,
                "prediction": prediction, "residual": residual, "factor": factor[0],
                "factor_reference": factor[1], "coarse_reference": reference}

    def months(year: int, half: str) -> list[pd.Timestamp]:
        start, stop = (1, 7) if half == "H1" else (7, 13)
        return [pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC") for month in range(start, stop)]

    predictions = {}
    for outer in REGIONS:
        for half in ("H1", "H2"):
            for cal_region in REGIONS:
                if cal_region == outer:
                    continue
                donors = tuple(region for region in REGIONS if region not in (outer, cal_region))
                for month in months(2024, half):
                    for method in METHODS:
                        predictions[(outer, half, "cal", cal_region, str(month.date()), method)] = predict(
                            cal_region, donors, month, 2024, method)
            donors = tuple(region for region in REGIONS if region != outer)
            for month in months(2025, half):
                for method in METHODS:
                    predictions[(outer, half, "target", outer, str(month.date()), method)] = predict(
                        outer, donors, month, 2025, method)

    scores = []
    for outer in REGIONS:
        for half in ("H1", "H2"):
            for method in METHODS:
                cal_by_region = {}
                for cal_region in REGIONS:
                    if cal_region == outer:
                        continue
                    rows = [predictions[(outer, half, "cal", cal_region, str(month.date()), method)]
                            for month in months(2024, half)]
                    rows = [row for row in rows if row is not None]
                    if rows:
                        cal_by_region[cal_region] = rows
                if len(cal_by_region) < 9:
                    raise ValueError(f"Insufficient calibration regions for {outer}/{half}/{method}: {len(cal_by_region)}")
                vals, weights = [], []
                q_by_region = {}
                for region, rows in cal_by_region.items():
                    residual = np.concatenate([np.abs(row["residual"]) for row in rows])
                    q_by_region[region] = float(np.quantile(residual, .90))
                    vals.extend(residual.tolist())
                    weights.extend(np.full(len(residual), 1 / (len(cal_by_region) * len(residual))).tolist())
                pooled = weighted_quantile(np.asarray(vals), np.asarray(weights), .90)
                group = float(np.max(list(q_by_region.values())))
                for month in months(2025, half):
                    target = predictions[(outer, half, "target", outer, str(month.date()), method)]
                    if target is None:
                        continue
                    for calibration, radius in (("pooled_q90", pooled), ("max_region_q90", group)):
                        y, pred = target["truth"], target["prediction"]
                        lo, hi = pred - radius, pred + radius
                        interval_score = 2 * radius + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
                        scores.append({"heldout_region": outer, "half": half, "target_month": target["month"],
                                       "ef_method": method, "calibration": calibration,
                                       "calibration_regions": len(cal_by_region), "n_hours": len(y),
                                       "point_mae": float(np.mean(np.abs(target["residual"]))),
                                       "point_bias": float(np.mean(target["residual"])),
                                       "coverage": float(np.mean((y >= lo) & (y <= hi))),
                                       "mean_width": 2 * radius, "interval_score": float(np.mean(interval_score)),
                                       "factor": target["factor"], "factor_reference": target["factor_reference"],
                                       "coarse_reference": target["coarse_reference"], "radius": radius,
                                       "calibration_max_region_q90": max(q_by_region.values())})

    # Train a cross-region EF blend on the 2024 source folds, then score 2025.
    # The held-out target region is excluded from both alpha selection and calibration.
    for outer in REGIONS:
        for half in ("H1", "H2"):
            for local_method in ("local_latest_asof_month", "local_latest_complete_year_asof"):
                cal_pairs = {}
                for cal_region in REGIONS:
                    if cal_region == outer:
                        continue
                    pairs = []
                    for month in months(2024, half):
                        donor = predictions[(outer, half, "cal", cal_region, str(month.date()), "donor_same_month")]
                        local = predictions[(outer, half, "cal", cal_region, str(month.date()), local_method)]
                        if donor is not None and local is not None:
                            pairs.append((donor, local))
                    if pairs:
                        cal_pairs[cal_region] = pairs
                if len(cal_pairs) < 9:
                    raise ValueError(f"Insufficient paired blend calibration regions: {outer}/{half}/{local_method}")

                candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
                alpha_losses = {}
                for alpha in candidates:
                    regional_mae = []
                    for pairs in cal_pairs.values():
                        residuals = [d["truth"] - ((1 - alpha) * d["prediction"] + alpha * l["prediction"])
                                     for d, l in pairs]
                        regional_mae.append(float(np.mean(np.abs(np.concatenate(residuals)))))
                    alpha_losses[alpha] = float(np.mean(regional_mae))
                alpha = min(candidates, key=lambda value: (alpha_losses[value], value))

                cal_abs_by_region = {}
                for region, pairs in cal_pairs.items():
                    residuals = [d["truth"] - ((1 - alpha) * d["prediction"] + alpha * l["prediction"])
                                 for d, l in pairs]
                    cal_abs_by_region[region] = np.abs(np.concatenate(residuals))
                pooled_values, pooled_weights = [], []
                for values in cal_abs_by_region.values():
                    pooled_values.extend(values.tolist())
                    pooled_weights.extend(np.full(len(values), 1 / (len(cal_abs_by_region) * len(values))).tolist())
                pooled_radius = weighted_quantile(np.asarray(pooled_values), np.asarray(pooled_weights), .90)
                region_q = [float(np.quantile(values, .90)) for values in cal_abs_by_region.values()]
                group_radius = float(np.max(region_q))

                for month in months(2025, half):
                    donor = predictions[(outer, half, "target", outer, str(month.date()), "donor_same_month")]
                    local = predictions[(outer, half, "target", outer, str(month.date()), local_method)]
                    if donor is None or local is None:
                        continue
                    y = donor["truth"]
                    pred = (1 - alpha) * donor["prediction"] + alpha * local["prediction"]
                    residual = y - pred
                    for calibration, radius in (("pooled_q90", pooled_radius), ("max_region_q90", group_radius)):
                        lo, hi = pred - radius, pred + radius
                        interval_score = 2 * radius + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
                        scores.append({"heldout_region": outer, "half": half, "target_month": str(month.date()),
                                       "ef_method": f"cross_region_blend_{local_method}", "selected_alpha": alpha,
                                       "calibration": calibration, "calibration_regions": len(cal_pairs),
                                       "n_hours": len(y), "point_mae": float(np.mean(np.abs(residual))),
                                       "point_bias": float(np.mean(residual)),
                                       "coverage": float(np.mean((y >= lo) & (y <= hi))),
                                       "mean_width": 2 * radius, "interval_score": float(np.mean(interval_score)),
                                       "factor": (1 - alpha) * donor["factor"] + alpha * local["factor"],
                                       "factor_reference": f"alpha={alpha:g};{local_method}",
                                       "coarse_reference": donor["coarse_reference"], "radius": radius,
                                       "calibration_max_region_q90": group_radius})

    score_frame = pd.DataFrame(scores)
    output = ROOT / "results/v53-local-thermal-ef-anchor-dev-r4"
    output.mkdir(parents=True, exist_ok=False)
    score_frame.to_csv(output / "region_month_scores.csv", index=False)
    if "selected_alpha" not in score_frame:
        score_frame["selected_alpha"] = np.nan
    region_scores = (score_frame.groupby(["half", "ef_method", "calibration", "heldout_region"])
                     .apply(lambda frame: pd.Series({
                         "coverage": np.average(frame.coverage, weights=frame.n_hours),
                         "width": np.average(frame.mean_width, weights=frame.n_hours),
                         "interval_score": np.average(frame.interval_score, weights=frame.n_hours),
                         "point_mae": np.average(frame.point_mae, weights=frame.n_hours),
                         "bias": np.average(frame.point_bias, weights=frame.n_hours),
                         "selected_alpha": frame.selected_alpha.mean(),
                         "hours": frame.n_hours.sum()}), include_groups=False).reset_index())
    region_scores.to_csv(output / "region_scores.csv", index=False)
    macro = (score_frame.groupby(["half", "ef_method", "calibration"])
             .agg(regions=("heldout_region", "nunique"), source_equal_coverage=("coverage", "mean"),
                  source_equal_width=("mean_width", "mean"), source_equal_interval_score=("interval_score", "mean"),
                  source_equal_point_mae=("point_mae", "mean"), source_equal_bias=("point_bias", "mean"),
                  mean_calibration_regions=("calibration_regions", "mean"),
                  mean_selected_alpha=("selected_alpha", "mean")).reset_index())
    worst = region_scores.groupby(["half", "ef_method", "calibration"]).coverage.min().rename("worst_region_coverage").reset_index()
    macro = macro.merge(worst, on=["half", "ef_method", "calibration"])
    macro.to_csv(output / "macro.csv", index=False)
    script = Path(__file__).resolve()
    protocol = ROOT / "docs/V53_LOCAL_THERMAL_EF_ANCHOR_DEV_PROTOCOL.md"
    summary = {"status": "DEVELOPMENT_ONLY_PREVIOUSLY_EVALUATED_REGIONS", "regions": list(REGIONS),
               "n_regions": len(REGIONS), "methods": list(METHODS), "never_scored_regions_read": False,
               "target_label_period": "2025 H1/H2; all 11 regions had been evaluated before this experiment",
               "information_contract": "Same five coarse monthly totals and donor hourly shapes; only thermal EF information source changes.",
               "factor_proxy": "EIA monthly fuel shares and fixed factors; a proxy information-value experiment, not China-specific EF validation.",
               "calibration": "2024 same-month residual q90; nested exclusion of outer target and calibration region; source-equal pooled or maximum-region radius.",
               "script_sha256": sha256(script), "protocol_sha256": sha256(protocol), "input_sha256": input_hashes,
               "macro": macro.astype(object).where(pd.notna(macro), None).to_dict("records")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
