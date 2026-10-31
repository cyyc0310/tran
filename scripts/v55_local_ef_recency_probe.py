"""Development probe of local thermal-EF recency windows, 2025 to 2026 H1."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v54_unseen_local_ef_confirmation as data  # noqa: E402

SOURCE = data.SOURCE_REGIONS
TARGETS = SOURCE + data.TARGET_REGIONS
METHODS = ("donor_same_month", "local_latest_complete_year", "local_latest_asof_month",
           "local_trailing3", "local_trailing6", "local_trailing12")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = ROOT / "docs/V55_LOCAL_EF_RECENCY_PROBE_PROTOCOL.md"
    protocol_hash = sha256(protocol)
    hourly, monthly, release_audits, input_hashes = data.prepare_regions()

    def eligible(region: str, origin: pd.Timestamp) -> pd.DataFrame:
        return monthly.loc[monthly.region.eq(region) & monthly.period_end.le(origin)
                           & monthly.available_at.le(origin)].sort_values("period_start")

    def coarse(region: str, origin: pd.Timestamp) -> np.ndarray | None:
        rows = monthly.loc[monthly.region.eq(region) & monthly.period_start.dt.month.eq(origin.month)
                           & monthly.period_start.lt(origin) & monthly.period_end.le(origin)
                           & monthly.available_at.le(origin)]
        if rows.empty:
            return None
        return data.coarse_totals(data.fuel_totals(rows.iloc[-1]))

    def local_factor(region: str, origin: pd.Timestamp, method: str) -> float | None:
        rows = eligible(region, origin)
        if method == "local_latest_complete_year":
            for year in sorted(rows.period_start.dt.year.unique(), reverse=True):
                if year >= origin.year:
                    continue
                annual = rows.loc[rows.period_start.dt.year.eq(year)]
                if annual.period_start.dt.month.nunique() == 12:
                    value = data.thermal_ef(np.sum([data.fuel_totals(row) for _, row in annual.iterrows()], axis=0))
                    if np.isfinite(value):
                        return value
            return None
        window = {"local_latest_asof_month": 1, "local_trailing3": 3,
                  "local_trailing6": 6, "local_trailing12": 12}.get(method)
        if window is None:
            return None
        rows = rows.tail(window)
        if len(rows) != window:
            return None
        periods = rows.period_start.sort_values().reset_index(drop=True)
        expected = pd.date_range(periods.iloc[0], periods.iloc[0] + pd.offsets.MonthBegin(window),
                                 freq="MS", inclusive="left", tz="UTC")
        if periods.tolist() != expected.tolist():
            return None
        totals = np.sum([data.fuel_totals(row) for _, row in rows.iterrows()], axis=0)
        value = data.thermal_ef(totals)
        return float(value) if np.isfinite(value) else None

    def donor_factor(donors: tuple[str, ...], origin: pd.Timestamp, forecast_year: int) -> float | None:
        period = pd.Timestamp(f"{forecast_year - 1}-{origin.month:02d}-01", tz="UTC")
        values = []
        for donor in donors:
            row = monthly.loc[monthly.region.eq(donor) & monthly.period_start.eq(period)
                              & monthly.period_end.le(origin) & monthly.available_at.le(origin)]
            if len(row):
                factor = data.thermal_ef(data.fuel_totals(row.iloc[-1]))
                if np.isfinite(factor):
                    values.append(factor)
        return float(np.mean(values)) if values else None

    profile_cache = {}

    def source_shape(region: str, month_no: int, year: int) -> np.ndarray:
        key = (region, month_no, year)
        if key in profile_cache:
            return profile_cache[key]
        frame = hourly[region]
        times = pd.to_datetime(frame.time, utc=True)
        selected = frame.loc[times.dt.year.eq(year) & times.dt.month.eq(month_no) & frame.valid]
        matrix = np.zeros((24, 5))
        if not selected.empty:
            work = selected.copy()
            work["hour"] = pd.to_datetime(work.time, utc=True).dt.hour
            cols = [f"generation_{fuel}" for fuel in data.FUELS]
            means = work.groupby("hour")[cols].mean()
            for hour, row in means.iterrows():
                matrix[int(hour), 0] = sum(float(row[f"generation_{fuel}"]) for fuel in data.THERMAL)
                for idx, fuel in enumerate(("nuclear", "hydro", "wind", "solar"), start=1):
                    matrix[int(hour), idx] = float(row[f"generation_{fuel}"])
        sums = matrix.sum(axis=0)
        matrix = np.divide(matrix, sums[None, :], out=np.ones_like(matrix) / 24, where=sums[None, :] > 0)
        profile_cache[key] = matrix
        return matrix

    def predict(region: str, donors: tuple[str, ...], origin: pd.Timestamp, forecast_year: int,
                method: str) -> dict | None:
        totals = coarse(region, origin)
        if totals is None:
            return None
        factor = (donor_factor(donors, origin, forecast_year) if method == "donor_same_month"
                  else local_factor(region, origin, method))
        if factor is None:
            return None
        frame = hourly[region]
        times = pd.to_datetime(frame.time, utc=True)
        current = frame.loc[times.dt.year.eq(origin.year) & times.dt.month.eq(origin.month)
                            & frame.valid].copy()
        if current.empty:
            return None
        actual = np.stack([current[f"generation_{fuel}"].to_numpy(float) for fuel in data.FUELS], axis=1)
        valid = np.isfinite(actual).all(axis=1) & (actual >= 0).all(axis=1) & (actual.sum(axis=1) > 0)
        current, actual = current.loc[valid], actual[valid]
        if len(current) < 480:
            return None
        profile = np.mean([source_shape(source, origin.month, forecast_year - 1) for source in donors], axis=0)
        profile /= np.maximum(profile.sum(axis=0, keepdims=True), 1e-12)
        times = pd.to_datetime(current.time, utc=True)
        estimates = np.stack([totals[j] * profile[times.dt.hour.to_numpy(), j] / origin.days_in_month
                              for j in range(5)], axis=1)
        point = estimates[:, 0] * factor / np.maximum(estimates.sum(axis=1), 1e-12)
        truth = actual @ data.EF / actual.sum(axis=1)
        return {"truth": truth, "prediction": point, "residual": truth - point}

    def half_months(year: int) -> list[pd.Timestamp]:
        return [pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC") for month in range(1, 7)]

    # Cache calibration predictions per outer target: calibration source excludes
    # both itself and the outer source target when they overlap.
    calibration = {}
    for outer in TARGETS:
        for source in SOURCE:
            if outer in SOURCE and source == outer:
                continue
            donors = tuple(item for item in SOURCE if item not in (outer, source))
            for method in METHODS:
                calibration[(outer, source, method)] = [predict(source, donors, month, 2025, method)
                                                        for month in half_months(2025)]

    scores = []
    for outer in TARGETS:
        target_donors = tuple(item for item in SOURCE if item != outer)
        source_regions = tuple(item for item in SOURCE if item != outer)
        for method in METHODS:
            cal_by_source = {}
            for source in source_regions:
                rows = [item for item in calibration[(outer, source, method)] if item is not None]
                if rows:
                    cal_by_source[source] = rows
            if len(cal_by_source) < 9:
                continue
            values, weights, source_q = [], [], []
            for rows in cal_by_source.values():
                residual = np.abs(np.concatenate([row["residual"] for row in rows]))
                values.extend(residual.tolist())
                weights.extend(np.full(len(residual), 1 / (len(cal_by_source) * len(residual))).tolist())
                source_q.append(float(np.quantile(residual, .90)))
            pooled_radius = data.weighted_quantile(np.asarray(values), np.asarray(weights), .90)
            max_radius = max(source_q)
            for month in half_months(2026):
                target = predict(outer, target_donors, month, 2026, method)
                if target is None:
                    continue
                for calibration_name, radius in (("pooled_q90", pooled_radius), ("max_region_q90", max_radius)):
                    truth, point = target["truth"], target["prediction"]
                    low, high = point - radius, point + radius
                    winkler = 2 * radius + 20 * np.maximum(low - truth, 0) + 20 * np.maximum(truth - high, 0)
                    scores.append({"target_region": outer, "month": str(month.date()), "method": method,
                                   "calibration": calibration_name, "calibration_regions": len(cal_by_source),
                                   "hours": len(truth), "point_mae": float(np.mean(np.abs(target["residual"]))),
                                   "bias": float(np.mean(target["residual"])),
                                   "coverage": float(np.mean((truth >= low) & (truth <= high))),
                                   "width": 2 * radius, "winkler": float(np.mean(winkler))})

    frame = pd.DataFrame(scores)
    output = ROOT / "results/v55-local-ef-recency-probe-r1"
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "region_month_scores.csv", index=False)
    region_scores = (frame.groupby(["calibration", "method", "target_region"])
                     .apply(lambda part: pd.Series({
                         "coverage": np.average(part.coverage, weights=part.hours),
                         "point_mae": np.average(part.point_mae, weights=part.hours),
                         "bias": np.average(part.bias, weights=part.hours),
                         "width": np.average(part.width, weights=part.hours),
                         "winkler": np.average(part.winkler, weights=part.hours),
                         "months": part.month.nunique()}), include_groups=False).reset_index())
    region_scores.to_csv(output / "region_scores.csv", index=False)
    macro = (region_scores.groupby(["calibration", "method"])
             .agg(regions=("target_region", "nunique"), macro_coverage=("coverage", "mean"),
                  worst_coverage=("coverage", "min"), point_mae=("point_mae", "mean"),
                  bias=("bias", "mean"), width=("width", "mean"), winkler=("winkler", "mean"),
                  mean_months=("months", "mean")).reset_index())
    macro.to_csv(output / "macro.csv", index=False)
    summary = {"status": "DEVELOPMENT_PROBE_NOT_INDEPENDENT_CONFIRMATION",
               "target_period": "2026 H1", "calibration_period": "2025 H1",
               "target_regions": list(TARGETS), "source_regions": list(SOURCE),
               "methods": list(METHODS), "never_scored_regions_read_before_v54": False,
               "script_sha256": sha256(Path(__file__).resolve()), "protocol_sha256": protocol_hash,
               "input_sha256": input_hashes, "label_boundary": "fixed-EF EIA positive reported generation proxy",
               "monthly_preparation": "earliest hourly snapshot, per-snapshot native schema, month-end+30d availability",
               "macro": macro.astype(object).where(pd.notna(macro), None).to_dict("records")}
    if sha256(protocol) != protocol_hash:
        raise RuntimeError("Protocol changed during probe")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
