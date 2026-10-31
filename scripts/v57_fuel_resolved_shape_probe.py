"""Test target-local monthly fuel constraints with source hourly fuel shapes."""

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
ARMS = ("donor_coarse", "local_latest_coarse", "local_annual_coarse", "local_fuel_resolved")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = ROOT / "docs/V57_FUEL_RESOLVED_SHAPE_PROBE_PROTOCOL.md"
    protocol_hash = sha256(protocol)
    hourly, monthly, _, input_hashes = data.prepare_regions()

    def monthly_row(region: str, origin: pd.Timestamp) -> pd.Series | None:
        rows = monthly.loc[monthly.region.eq(region) & monthly.period_start.dt.month.eq(origin.month)
                           & monthly.period_start.lt(origin) & monthly.period_end.le(origin)
                           & monthly.available_at.le(origin)].sort_values("period_start")
        return rows.iloc[-1] if not rows.empty else None

    def annual_ef(region: str, origin: pd.Timestamp) -> float | None:
        rows = monthly.loc[monthly.region.eq(region) & monthly.period_end.le(origin)
                           & monthly.available_at.le(origin)]
        for year in sorted(rows.period_start.dt.year.unique(), reverse=True):
            if year >= origin.year:
                continue
            full = rows.loc[rows.period_start.dt.year.eq(year)]
            if full.period_start.dt.month.nunique() == 12:
                value = data.thermal_ef(np.sum([data.fuel_totals(row) for _, row in full.iterrows()], axis=0))
                if np.isfinite(value):
                    return value
        return None

    def latest_local_ef(region: str, origin: pd.Timestamp) -> float | None:
        rows = monthly.loc[monthly.region.eq(region) & monthly.period_end.le(origin)
                           & monthly.available_at.le(origin)].sort_values(["period_end", "available_at"])
        if rows.empty:
            return None
        value = data.thermal_ef(data.fuel_totals(rows.iloc[-1]))
        return float(value) if np.isfinite(value) else None

    def donor_ef(donors: tuple[str, ...], origin: pd.Timestamp, forecast_year: int) -> float | None:
        period = pd.Timestamp(f"{forecast_year - 1}-{origin.month:02d}-01", tz="UTC")
        values = []
        for region in donors:
            rows = monthly.loc[monthly.region.eq(region) & monthly.period_start.eq(period)
                               & monthly.period_end.le(origin) & monthly.available_at.le(origin)]
            if not rows.empty:
                value = data.thermal_ef(data.fuel_totals(rows.iloc[-1]))
                if np.isfinite(value):
                    values.append(value)
        return float(np.mean(values)) if values else None

    profiles: dict[tuple[str, int, int, str], np.ndarray | None] = {}

    def fuel_profile(region: str, month_no: int, year: int, fuel: str) -> np.ndarray | None:
        key = (region, month_no, year, fuel)
        if key in profiles:
            return profiles[key]
        frame = hourly[region]
        times = pd.to_datetime(frame.time, utc=True)
        selected = frame.loc[times.dt.year.eq(year) & times.dt.month.eq(month_no) & frame.valid].copy()
        col = f"generation_{fuel}"
        if selected.empty:
            profiles[key] = None
            return None
        selected["hour"] = pd.to_datetime(selected.time, utc=True).dt.hour
        hourly_means = selected.groupby("hour")[col].mean().reindex(range(24), fill_value=0).to_numpy(float)
        total = float(hourly_means.sum())
        profiles[key] = hourly_means / total if total > 0 else None
        return profiles[key]

    shape_cache: dict[tuple[tuple[str, ...], int, int, str], np.ndarray] = {}

    def group_profile(donors: tuple[str, ...], month_no: int, year: int, group: str) -> np.ndarray:
        key = (donors, month_no, year, group)
        if key in shape_cache:
            return shape_cache[key]
        active = [fuel_profile(region, month_no, year, fuel) for region in donors
                  for fuel in (data.THERMAL if group == "thermal" else (group,))]
        active = [p for p in active if p is not None]
        if group == "thermal":
            per_region = []
            for region in donors:
                frame = hourly[region]
                times = pd.to_datetime(frame.time, utc=True)
                selected = frame.loc[times.dt.year.eq(year) & times.dt.month.eq(month_no) & frame.valid].copy()
                if selected.empty:
                    continue
                selected["hour"] = pd.to_datetime(selected.time, utc=True).dt.hour
                columns = [f"generation_{fuel}" for fuel in data.THERMAL]
                by_hour = selected.groupby("hour")[columns].mean().sum(axis=1).reindex(range(24), fill_value=0)
                values = by_hour.to_numpy(float)
                if values.sum() > 0:
                    per_region.append(values / values.sum())
            profile = np.mean(per_region, axis=0) if per_region else np.ones(24) / 24
        else:
            profile = np.mean(active, axis=0) if active else np.ones(24) / 24
        profile = profile / profile.sum()
        shape_cache[key] = profile
        return profile

    fuel_profile_cache: dict[tuple[tuple[str, ...], int, int, str], tuple[np.ndarray, bool]] = {}

    def pooled_fuel_profile(donors: tuple[str, ...], month_no: int, year: int,
                            fuel: str) -> tuple[np.ndarray, bool]:
        key = (donors, month_no, year, fuel)
        if key in fuel_profile_cache:
            return fuel_profile_cache[key]
        active = [fuel_profile(region, month_no, year, fuel) for region in donors]
        active = [profile for profile in active if profile is not None]
        fallback = not active
        profile = np.mean(active, axis=0) if active else np.ones(24) / 24
        profile /= profile.sum()
        fuel_profile_cache[key] = (profile, fallback)
        return profile, fallback

    def predict(region: str, donors: tuple[str, ...], origin: pd.Timestamp, year: int,
                arm: str) -> dict | None:
        row = monthly_row(region, origin)
        if row is None:
            return None
        totals_fuel = data.fuel_totals(row)
        group_totals = data.coarse_totals(totals_fuel)
        if arm == "local_fuel_resolved":
            ef = data.EF.copy()
        elif arm in ("local_latest_coarse", "local_annual_coarse"):
            value = (latest_local_ef(region, origin) if arm == "local_latest_coarse"
                     else annual_ef(region, origin))
            if value is None:
                value = donor_ef(donors, origin, year)
            if value is None:
                return None
            ef = np.array([value, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        elif arm == "donor_coarse":
            value = donor_ef(donors, origin, year)
            if value is None:
                return None
            ef = np.array([value, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        else:
            raise ValueError(arm)

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
        hours = pd.to_datetime(current.time, utc=True).dt.hour.to_numpy()
        predicted_fuels = np.zeros((len(current), len(data.FUELS)))
        fallbacks = []
        if arm == "local_fuel_resolved":
            for idx, fuel in enumerate(data.FUELS):
                profile, fallback = pooled_fuel_profile(donors, origin.month, year - 1, fuel)
                predicted_fuels[:, idx] = totals_fuel[idx] * profile[hours] / origin.days_in_month
                if fallback:
                    fallbacks.append(fuel)
        else:
            profiles_by_group = [group_profile(donors, origin.month, year - 1, group)
                                 for group in ("thermal", "nuclear", "hydro", "wind", "solar")]
            predicted_groups = np.stack([group_totals[j] * profiles_by_group[j][hours] / origin.days_in_month
                                         for j in range(5)], axis=1)
            # Collapse grouped intensity to a thermal-equivalent 8-fuel vector.
            predicted_fuels[:, 0] = predicted_groups[:, 0]
            predicted_fuels[:, 3:7] = predicted_groups[:, 1:]
        predicted_total = predicted_fuels.sum(axis=1)
        estimate = predicted_fuels @ ef / np.maximum(predicted_total, 1e-12)
        truth = actual @ data.EF / actual.sum(axis=1)
        residual = truth - estimate
        return {"truth": truth, "prediction": estimate, "residual": residual,
                "fine_fallbacks": fallbacks, "monthly_fuel_totals": totals_fuel.tolist()}

    def half_months(year: int) -> list[pd.Timestamp]:
        return [pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC") for month in range(1, 7)]

    calibration = {}
    for outer in TARGETS:
        for source in SOURCE:
            if outer in SOURCE and source == outer:
                continue
            donors = tuple(item for item in SOURCE if item not in (outer, source))
            for arm in ARMS:
                calibration[(outer, source, arm)] = [predict(source, donors, month, 2025, arm)
                                                     for month in half_months(2025)]

    scores = []
    for outer in TARGETS:
        donors = tuple(item for item in SOURCE if item != outer)
        for arm in ARMS:
            cal_by_source = {}
            for source in SOURCE:
                if outer in SOURCE and source == outer:
                    continue
                rows = [row for row in calibration[(outer, source, arm)] if row is not None]
                if rows:
                    cal_by_source[source] = rows
            if len(cal_by_source) < 9:
                continue
            pooled_values, pooled_weights, source_q = [], [], []
            for rows in cal_by_source.values():
                absolute = np.abs(np.concatenate([row["residual"] for row in rows]))
                pooled_values.extend(absolute.tolist())
                pooled_weights.extend(np.full(len(absolute), 1 / (len(cal_by_source) * len(absolute))).tolist())
                source_q.append(float(np.quantile(absolute, .90)))
            pooled_radius = data.weighted_quantile(np.asarray(pooled_values), np.asarray(pooled_weights), .90)
            max_radius = max(source_q)
            for month in half_months(2026):
                target = predict(outer, donors, month, 2026, arm)
                if target is None:
                    continue
                for cal_name, radius in (("pooled_q90", pooled_radius), ("max_region_q90", max_radius)):
                    truth, point = target["truth"], target["prediction"]
                    low, high = point - radius, point + radius
                    winkler = 2 * radius + 20 * np.maximum(low - truth, 0) + 20 * np.maximum(truth - high, 0)
                    scores.append({"target_region": outer, "month": str(month.date()), "arm": arm,
                                   "calibration": cal_name, "calibration_regions": len(cal_by_source),
                                   "hours": len(truth), "point_mae": float(np.mean(np.abs(target["residual"]))),
                                   "bias": float(np.mean(target["residual"])),
                                   "coverage": float(np.mean((truth >= low) & (truth <= high))),
                                   "width": 2 * radius, "winkler": float(np.mean(winkler)),
                                   "uniform_fuel_fallbacks": ";".join(target["fine_fallbacks"])})

    score_frame = pd.DataFrame(scores)
    output = ROOT / "results/v57-fuel-resolved-shape-probe-r1"
    output.mkdir(parents=True, exist_ok=False)
    score_frame.to_csv(output / "region_month_scores.csv", index=False)
    region_scores = (score_frame.groupby(["calibration", "arm", "target_region"])
                     .apply(lambda part: pd.Series({
                         "coverage": np.average(part.coverage, weights=part.hours),
                         "point_mae": np.average(part.point_mae, weights=part.hours),
                         "bias": np.average(part.bias, weights=part.hours),
                         "width": np.average(part.width, weights=part.hours),
                         "winkler": np.average(part.winkler, weights=part.hours),
                         "months": part.month.nunique()}), include_groups=False).reset_index())
    region_scores.to_csv(output / "region_scores.csv", index=False)
    macro = (region_scores.groupby(["calibration", "arm"])
             .agg(regions=("target_region", "nunique"), macro_coverage=("coverage", "mean"),
                  worst_coverage=("coverage", "min"), point_mae=("point_mae", "mean"),
                  bias=("bias", "mean"), width=("width", "mean"), winkler=("winkler", "mean"),
                  mean_months=("months", "mean")).reset_index())
    macro.to_csv(output / "macro.csv", index=False)
    summary = {"status": "DEVELOPMENT_PROBE_NOT_INDEPENDENT_CONFIRMATION",
               "target_period": "2026 H1", "calibration_period": "2025 H1",
               "target_regions": list(TARGETS), "source_regions": list(SOURCE), "arms": list(ARMS),
               "never_scored_regions_read_before_v54": False,
               "script_sha256": sha256(Path(__file__).resolve()), "protocol_sha256": protocol_hash,
               "raw_input_sha256": input_hashes,
               "label_boundary": "EIA fixed-factor positive reported-generation proxy",
               "monthly_availability": "month-end plus 30 days, strict as-of filtering",
               "macro": macro.astype(object).where(pd.notna(macro), None).to_dict("records")}
    if sha256(protocol) != protocol_hash:
        raise RuntimeError("Frozen protocol changed during run")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
