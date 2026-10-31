"""Develop region-level interval baselines on previously evaluated EIA zones."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("AVA", "BANC", "BPAT", "FPL", "JEA", "MISO", "NWMT", "NYIS", "SC", "SCEG", "SEC")
FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
THERMAL = ("coal", "gas", "oil", "other")
EF = np.array([1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 400.0])
ROOT_BY_REGION = {
    "AVA": "results/external-v4-data",
    "BANC": "results/external-v4-data",
    "BPAT": "results/heldout-2025-data",
    "FPL": "results/heldout-2025-data",
    "JEA": "results/external-v4-data",
    "MISO": "results/heldout-2025-data",
    "NWMT": "results/external-v4-data",
    "NYIS": "results/heldout-2025-data",
    "SC": "results/external-v4-data",
    "SCEG": "results/external-v5-data",
    "SEC": "results/external-v5-data",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    return float(v[min(np.searchsorted(np.cumsum(w), q * w.sum()), len(v) - 1)])


def source_weights(rows: list[dict]) -> np.ndarray:
    counts = {r: sum(x["region"] == r for x in rows) for r in {x["region"] for x in rows}}
    return np.array([1.0 / counts[x["region"]] for x in rows])


def fuel_totals(row: pd.Series) -> np.ndarray:
    return float(row.total_mwh) * np.array([float(row[f"fuelshare_{f}"]) for f in FUELS])


def coarse_totals(values: np.ndarray) -> np.ndarray:
    return np.array([
        sum(values[FUELS.index(f)] for f in THERMAL),
        values[FUELS.index("nuclear")],
        values[FUELS.index("hydro")],
        values[FUELS.index("wind")],
        values[FUELS.index("solar")],
    ])


def thermal_ef(values: np.ndarray) -> float:
    denominator = sum(values[FUELS.index(f)] for f in THERMAL)
    return float(sum(values[FUELS.index(f)] * EF[FUELS.index(f)] for f in THERMAL) / denominator) if denominator > 0 else 0.0


def main() -> None:
    monthly_parts, hourly, input_hashes = [], {}, {}
    for region, relative in ROOT_BY_REGION.items():
        folder = ROOT / relative
        monthly_path, hourly_path = folder / "monthly.csv", folder / f"{region}_hourly.csv"
        m = pd.read_csv(monthly_path)
        m = m.loc[m.region.eq(region)].copy()
        for c in ("period_start", "period_end", "available_at"):
            m[c] = pd.to_datetime(m[c], utc=True)
        monthly_parts.append(m)
        h = pd.read_csv(hourly_path)
        h["time"] = pd.to_datetime(h.time, utc=True)
        h["_year"], h["_month"], h["_hour"] = h.time.dt.year, h.time.dt.month, h.time.dt.hour
        h["_valid"] = h.valid.astype(str).str.lower().isin(("true", "1"))
        hourly[region] = h
        input_hashes[str(monthly_path.resolve())] = sha256(monthly_path)
        input_hashes[str(hourly_path.resolve())] = sha256(hourly_path)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    if monthly.duplicated(["region", "period_start"]).any():
        raise ValueError("Duplicate region-month input rows")

    def asof_totals(region: str, target: pd.Timestamp) -> tuple[np.ndarray, str] | None:
        rows = monthly.loc[
            monthly.region.eq(region)
            & monthly.period_start.dt.month.eq(target.month)
            & monthly.period_start.lt(target)
            & monthly.period_end.le(target)
            & monthly.available_at.le(target)
        ].sort_values(["period_start", "available_at"])
        if rows.empty:
            return None
        row = rows.iloc[-1]
        return coarse_totals(fuel_totals(row)), str(row.period_start.date())

    def donor_ef(donors: tuple[str, ...], target: pd.Timestamp, year: int) -> float:
        values = []
        for donor in donors:
            rows = monthly.loc[
                monthly.region.eq(donor)
                & monthly.period_start.dt.month.eq(target.month)
                & monthly.period_start.dt.year.eq(year - 1)
            ]
            if len(rows) == 1:
                value = thermal_ef(fuel_totals(rows.iloc[0]))
                if value > 0:
                    values.append(value)
        return float(np.mean(values)) if values else float(np.mean(EF[[0, 1, 2, 7]]))

    profile_cache = {}

    def region_shape(region: str, month: int, year: int) -> np.ndarray:
        key = (region, month, year)
        if key in profile_cache:
            return profile_cache[key]
        frame = hourly[region]
        frame = frame.loc[frame._year.eq(year) & frame._month.eq(month) & frame._valid]
        matrix = np.zeros((24, 5))
        if not frame.empty:
            fuel_columns = [f"generation_{fuel}" for fuel in FUELS]
            means = frame.groupby("_hour")[fuel_columns].mean()
            for hour, row in means.iterrows():
                matrix[int(hour), 0] = sum(float(row[f"generation_{fuel}"]) for fuel in THERMAL)
                for j, fuel in enumerate(("nuclear", "hydro", "wind", "solar"), start=1):
                    matrix[int(hour), j] = float(row[f"generation_{fuel}"])
        sums = matrix.sum(axis=0)
        matrix = np.divide(matrix, sums[None, :], out=np.ones_like(matrix) / 24,
                           where=sums[None, :] > 0)
        profile_cache[key] = matrix
        return matrix

    def donor_shape(donors: tuple[str, ...], month: int, year: int) -> np.ndarray:
        profiles = [region_shape(donor, month, year) for donor in donors]
        mean = np.mean(profiles, axis=0)
        sums = mean.sum(axis=0, keepdims=True)
        return np.divide(mean, sums, out=np.ones_like(mean) / 24, where=sums > 0)

    def predict_month(region: str, donors: tuple[str, ...], target: pd.Timestamp, year: int) -> dict | None:
        coarse = asof_totals(region, target)
        if coarse is None:
            return None
        totals, reference_month = coarse
        frame = hourly[region]
        frame = frame.loc[
            frame._year.eq(target.year) & frame._month.eq(target.month) & frame._valid
        ].copy()
        if frame.empty:
            return None
        actual = np.stack([frame[f"generation_{f}"].to_numpy(float) for f in FUELS], axis=1)
        valid = np.isfinite(actual).all(axis=1) & (actual >= 0).all(axis=1) & (actual.sum(axis=1) > 0)
        frame, actual = frame.loc[valid], actual[valid]
        if len(frame) < 480:
            return None
        shape = donor_shape(donors, target.month, year - 1)
        predicted_groups = np.stack([
            totals[j] * shape[frame._hour.to_numpy(), j] / target.days_in_month
            for j in range(5)
        ], axis=1)
        ef = donor_ef(donors, target, year)
        prediction = predicted_groups[:, 0] * ef / np.maximum(predicted_groups.sum(axis=1), 1e-12)
        truth = actual @ EF / actual.sum(axis=1)
        residual = truth - prediction
        total = max(float(totals.sum()), 1e-12)
        features = np.array([totals[0] / total, (totals[3] + totals[4]) / total])
        full_mix = totals / total
        return {"region": region, "month": str(target.date()), "features": features,
                "full_mix_features": np.concatenate([
                    full_mix,
                    [np.sin(2 * np.pi * target.month / 12), np.cos(2 * np.pi * target.month / 12)],
                ]),
                "truth": truth, "prediction": prediction, "residual": residual,
                "point_mae": float(np.mean(np.abs(residual))), "reference_month": reference_month}

    def target_months(year: int, half: str) -> list[pd.Timestamp]:
        start = 1 if half == "H1" else 7
        stop = pd.Timestamp(f"{year + 1}-01-01", tz="UTC") if half == "H2" else pd.Timestamp(f"{year}-07-01", tz="UTC")
        return list(pd.date_range(pd.Timestamp(f"{year}-{start:02d}-01", tz="UTC"), stop,
                                  freq="MS", inclusive="left"))

    # Cache historical calibration predictions. For each outer target region,
    # calibration-region forecasts use donors excluding both regions.
    predictions = {}
    for outer in REGIONS:
        for half in ("H1", "H2"):
            for cal_region in REGIONS:
                if cal_region == outer:
                    continue
                cal_donors = tuple(r for r in REGIONS if r not in (outer, cal_region))
                for month in target_months(2024, half):
                    predictions[(outer, half, "cal", cal_region, str(month.date()))] = predict_month(
                        cal_region, cal_donors, month, 2024)
            target_donors = tuple(r for r in REGIONS if r != outer)
            for month in target_months(2025, half):
                predictions[(outer, half, "target", outer, str(month.date()))] = predict_month(
                    outer, target_donors, month, 2025)

    out_rows = []
    for outer in REGIONS:
        for half in ("H1", "H2"):
            cal_by_region = {}
            for cal_region in REGIONS:
                if cal_region == outer:
                    continue
                rows = [predictions[(outer, half, "cal", cal_region, str(month.date()))]
                        for month in target_months(2024, half)]
                rows = [row for row in rows if row is not None]
                if rows:
                    for row in rows:
                        row["q90"] = float(np.quantile(np.abs(row["residual"]), 0.90))
                        row["center"] = float(np.mean(row["residual"]))
                    cal_by_region[cal_region] = rows
            if len(cal_by_region) < 9:
                raise ValueError(f"Need >=9 calibration regions for 90% finite rank; got {len(cal_by_region)}")

            # Pooled residual calibration with equal total weight per source.
            pooled_values, pooled_weights = [], []
            for rows in cal_by_region.values():
                values = np.concatenate([np.abs(row["residual"]) for row in rows])
                pooled_values.extend(values.tolist())
                pooled_weights.extend(np.full(len(values), 1 / (len(cal_by_region) * len(values))).tolist())
            pooled_q90 = weighted_quantile(np.asarray(pooled_values), np.asarray(pooled_weights), 0.90)

            per_region_q90 = {
                region: float(np.quantile(np.concatenate([np.abs(row["residual"]) for row in rows]), 0.90))
                for region, rows in cal_by_region.items()
            }
            # K=10, alpha=.1 -> order ceil((K+1)*.9)=K, i.e. maximum group score.
            k = int(np.ceil((len(cal_by_region) + 1) * 0.90))
            group_q90 = float(np.sort(list(per_region_q90.values()))[min(k - 1, len(per_region_q90) - 1)])

            # Source-held-out calibration scales for the monthly-mix candidate.
            normalized_by_region = {region: [] for region in cal_by_region}
            for held_cal, cal_rows in cal_by_region.items():
                train_rows = [row for region, rows in cal_by_region.items() if region != held_cal for row in rows]
                x_train = np.stack([row["features"] for row in train_rows])
                y_train = np.log(np.array([row["q90"] for row in train_rows]) + 1.0)
                model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                model.fit(x_train, y_train, ridge__sample_weight=source_weights(train_rows))
                x_cal = np.stack([row["features"] for row in cal_rows])
                scales = np.exp(model.predict(x_cal))
                for row, scale in zip(cal_rows, scales):
                    normalized_by_region[held_cal].extend((np.abs(row["residual"]) / max(float(scale), 1e-6)).tolist())
            norm_values, norm_weights = [], []
            for values in normalized_by_region.values():
                arr = np.asarray(values)
                if len(arr):
                    norm_values.extend(arr.tolist())
                    norm_weights.extend(np.full(len(arr), 1 / (len(normalized_by_region) * len(arr))).tolist())
            normalized_q90 = weighted_quantile(np.asarray(norm_values), np.asarray(norm_weights), 0.90)

            train_rows = [row for rows in cal_by_region.values() for row in rows]
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(np.stack([row["features"] for row in train_rows]),
                      np.log(np.array([row["q90"] for row in train_rows]) + 1.0),
                      ridge__sample_weight=source_weights(train_rows))

            # Cross-fitted source-month seasonal center correction.
            seasonal_centered = {region: [] for region in cal_by_region}
            mix_centered = {region: [] for region in cal_by_region}
            full_mix_centered = {region: [] for region in cal_by_region}
            for held_cal, cal_rows in cal_by_region.items():
                for row in cal_rows:
                    month_no = pd.Timestamp(row["month"]).month
                    other_centers = [other["center"]
                                     for region, rows in cal_by_region.items() if region != held_cal
                                     for other in rows if pd.Timestamp(other["month"]).month == month_no]
                    seasonal_center = float(np.mean(other_centers)) if other_centers else 0.0
                    seasonal_centered[held_cal].extend((row["residual"] - seasonal_center).tolist())

                center_train = [row for region, rows in cal_by_region.items() if region != held_cal for row in rows]
                center_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                center_model.fit(np.stack([row["features"] for row in center_train]),
                                 np.array([row["center"] for row in center_train]),
                                 ridge__sample_weight=source_weights(center_train))
                for row in cal_rows:
                    shift = float(center_model.predict(row["features"].reshape(1, -1))[0])
                    mix_centered[held_cal].extend((row["residual"] - shift).tolist())

                full_mix_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                full_mix_model.fit(np.stack([row["full_mix_features"] for row in center_train]),
                                   np.array([row["center"] for row in center_train]),
                                   ridge__sample_weight=source_weights(center_train))
                for row in cal_rows:
                    shift = float(full_mix_model.predict(row["full_mix_features"].reshape(1, -1))[0])
                    full_mix_centered[held_cal].extend((row["residual"] - shift).tolist())

            def pooled_centered_q90(values_by_region: dict[str, list[float]]) -> float:
                values, weights = [], []
                for region_values in values_by_region.values():
                    array = np.asarray(region_values)
                    if len(array):
                        values.extend(np.abs(array).tolist())
                        weights.extend(np.full(len(array), 1 / (len(values_by_region) * len(array))).tolist())
                return weighted_quantile(np.asarray(values), np.asarray(weights), 0.90)

            seasonal_q90 = pooled_centered_q90(seasonal_centered)
            mix_center_q90 = pooled_centered_q90(mix_centered)
            full_mix_center_q90 = pooled_centered_q90(full_mix_centered)
            center_rows = [row for rows in cal_by_region.values() for row in rows]
            final_center_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            final_center_model.fit(np.stack([row["features"] for row in center_rows]),
                                   np.array([row["center"] for row in center_rows]),
                                   ridge__sample_weight=source_weights(center_rows))
            final_full_mix_center_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            final_full_mix_center_model.fit(np.stack([row["full_mix_features"] for row in center_rows]),
                                            np.array([row["center"] for row in center_rows]),
                                            ridge__sample_weight=source_weights(center_rows))

            for month in target_months(2025, half):
                target = predictions[(outer, half, "target", outer, str(month.date()))]
                if target is None:
                    continue
                conditional_scale = float(np.exp(model.predict(target["features"].reshape(1, -1))[0]))
                month_no = pd.Timestamp(target["month"]).month
                seasonal_target_centers = [row["center"] for rows in cal_by_region.values()
                                            for row in rows if pd.Timestamp(row["month"]).month == month_no]
                seasonal_shift = float(np.mean(seasonal_target_centers)) if seasonal_target_centers else 0.0
                mix_center_shift = float(final_center_model.predict(target["features"].reshape(1, -1))[0])
                full_mix_center_shift = float(final_full_mix_center_model.predict(
                    target["full_mix_features"].reshape(1, -1))[0])
                for method, radius, center_shift in (
                    ("pooled_source_equal_q90", pooled_q90, 0.0),
                    ("max_region_q90_conformal_rank", group_q90, 0.0),
                    ("monthly_mix_conditional_scale", conditional_scale * normalized_q90, 0.0),
                    ("seasonal_source_center_plus_pooled_q90", seasonal_q90, seasonal_shift),
                    ("monthly_mix_ridge_center_plus_pooled_q90", mix_center_q90, mix_center_shift),
                    ("full_monthly_mix_seasonal_ridge_center", full_mix_center_q90, full_mix_center_shift),
                ):
                    y, raw_pred = target["truth"], target["prediction"]
                    pred = raw_pred + center_shift
                    lo, hi = pred - radius, pred + radius
                    below, above = y < lo, y > hi
                    alpha = 0.10
                    interval_score = 2 * radius + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
                    out_rows.append({
                        "heldout_region": outer, "half": half, "target_month": target["month"],
                        "method": method, "calibration_regions": len(cal_by_region),
                        "conformal_rank": k, "n_hours": len(y), "coverage": float(np.mean((y >= lo) & (y <= hi))),
                        "lower_miss": float(np.mean(below)), "upper_miss": float(np.mean(above)),
                        "mean_width": float(2 * radius), "interval_score": float(np.mean(interval_score)),
                        "point_mae": float(np.mean(np.abs(target["residual"] - center_shift))),
                        "raw_point_mae": target["point_mae"], "radius": radius,
                        "center_shift": center_shift,
                        "point_bias": float(np.mean(target["residual"] - center_shift)),
                        "target_abs_residual_q90": float(np.quantile(np.abs(target["residual"] - center_shift), 0.90)),
                        "calibration_max_region_q90": max(per_region_q90.values()),
                        "pooled_q90": pooled_q90, "group_q90": group_q90,
                        "conditional_scale": conditional_scale, "normalized_q90": normalized_q90,
                    })

    scores = pd.DataFrame(out_rows)
    output = ROOT / "results/v52-residual-center-dev-r2"
    output.mkdir(parents=True, exist_ok=False)
    scores.to_csv(output / "region_month_scores.csv", index=False)
    macro = (scores.groupby(["half", "method"])
             .agg(regions=("heldout_region", "nunique"), region_months=("target_month", "count"),
                  source_equal_coverage=("coverage", "mean"), source_equal_lower_miss=("lower_miss", "mean"),
                  source_equal_upper_miss=("upper_miss", "mean"), source_equal_width=("mean_width", "mean"),
                  source_equal_interval_score=("interval_score", "mean"), source_equal_point_mae=("point_mae", "mean"),
                  mean_calibration_regions=("calibration_regions", "mean"))
             .reset_index())
    macro.to_csv(output / "macro.csv", index=False)
    region_scores = (scores.groupby(["half", "method", "heldout_region"])
                     .apply(lambda x: pd.Series({
                         "coverage": float(np.average(x.coverage, weights=x.n_hours)),
                         "mean_width": float(np.average(x.mean_width, weights=x.n_hours)),
                         "interval_score": float(np.average(x.interval_score, weights=x.n_hours)),
                         "point_mae": float(np.average(x.point_mae, weights=x.n_hours)),
                         "hours": int(x.n_hours.sum()),
                     }), include_groups=False).reset_index())
    region_scores.to_csv(output / "region_scores.csv", index=False)
    worst = region_scores.groupby(["half", "method"]).coverage.min().rename("worst_region_coverage").reset_index()
    macro = macro.merge(worst, on=["half", "method"])
    macro.to_csv(output / "macro.csv", index=False)
    summary = {
        "status": "DEVELOPMENT_ONLY_PREVIOUSLY_EVALUATED_REGIONS",
        "regions": list(REGIONS), "n_regions": len(REGIONS),
        "calibration_regions_per_outer_fold": len(REGIONS) - 1,
        "nominal_region_q90_rank": int(np.ceil(len(REGIONS) * 0.90)),
        "holdout_regions_scored": list(REGIONS),
        "never_scored_regions_read": False,
        "target_label_period": "2025 H1/H2 only; all included regions were already development/evaluated targets",
        "information_contract": "lagged same-month previous-year five-category totals; source-only hourly shape and thermal EF; no target hourly CIF labels in prediction",
        "calibration": "2024 same-month point residuals; nested source region exclusion; source-level empirical q90 scores; residual-center correction is source-held-out before calibration",
        "script_sha256": sha256(Path(__file__).resolve()),
        "protocol_sha256": sha256(ROOT / "docs/V52_REGION_POOL_RISK_DEV_PROTOCOL.md"),
        "center_amendment_sha256": sha256(ROOT / "docs/V52_RESIDUAL_CENTER_DEV_AMENDMENT.md"),
        "full_mix_center_amendment_sha256": sha256(ROOT / "docs/V52_FULL_MIX_CENTER_DEV_AMENDMENT.md"),
        "input_sha256": input_hashes,
        "macro": macro.to_dict("records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
