"""Test whether recent target-local fuel mix improves transfer of hourly fuel shapes."""

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
TARGETS = data.ALL_REGIONS
FUELS = data.FUELS
EF = data.EF
ARMS = ("local_latest_fullmix_shape", "local_yoy_fullmix_shape",
        "local_latest_thermal_ef_coarse", "flat_latest_fullmix")
OUTPUT = ROOT / "results/v62-recent-full-mix-shape-transfer-r1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_row(monthly: pd.DataFrame, region: str, origin: pd.Timestamp) -> pd.Series | None:
    rows = monthly.loc[monthly.region.eq(region) & monthly.period_end.le(origin)
                       & monthly.available_at.le(origin) & monthly.period_start.lt(origin)]
    if rows.empty:
        return None
    return rows.sort_values(["period_start", "available_at"]).iloc[-1]


def yoy_row(monthly: pd.DataFrame, region: str, origin: pd.Timestamp,
            forecast_year: int) -> pd.Series | None:
    period = pd.Timestamp(f"{forecast_year - 1}-{origin.month:02d}-01", tz="UTC")
    rows = monthly.loc[monthly.region.eq(region) & monthly.period_start.eq(period)
                       & monthly.period_end.le(origin) & monthly.available_at.le(origin)]
    return rows.iloc[-1] if not rows.empty else None


def mix(row: pd.Series) -> np.ndarray:
    values = np.array([float(row[f"fuelshare_{fuel}"]) for fuel in FUELS], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Invalid monthly fuel shares")
    return values / values.sum()


def group_mix(row: pd.Series) -> np.ndarray:
    values = mix(row)
    thermal = sum(values[FUELS.index(fuel)] for fuel in data.THERMAL)
    return np.array([thermal, values[FUELS.index("nuclear")], values[FUELS.index("hydro")],
                     values[FUELS.index("wind")], values[FUELS.index("solar")]])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    idx = min(np.searchsorted(np.cumsum(weights), quantile * weights.sum()), len(values) - 1)
    return float(values[idx])


def main() -> None:
    protocol = ROOT / "docs/V62_RECENT_FULL_MIX_SHAPE_TRANSFER_PROTOCOL.md"
    protocol_hash = sha256(protocol)
    hourly, monthly, release_audits, input_hashes = data.prepare_regions()
    if sha256(protocol) != protocol_hash:
        raise RuntimeError("Protocol changed during data preparation")

    profile_cache: dict[tuple[str, int, int, str], np.ndarray | None] = {}

    def donor_profile(region: str, month_no: int, year: int, fuel: str) -> np.ndarray | None:
        key = (region, month_no, year, fuel)
        if key in profile_cache:
            return profile_cache[key]
        frame = hourly[region]
        times = pd.to_datetime(frame.time, utc=True)
        selected = frame.loc[times.dt.year.eq(year) & times.dt.month.eq(month_no)
                             & frame.valid].copy()
        if selected.empty:
            profile_cache[key] = None
            return None
        selected["hour"] = pd.to_datetime(selected.time, utc=True).dt.hour
        values = selected.groupby("hour")[f"generation_{fuel}"].mean().reindex(range(24), fill_value=0).to_numpy(float)
        total = float(values.sum())
        profile_cache[key] = values / total if total > 0 else None
        return profile_cache[key]

    def pooled_profiles(donors: tuple[str, ...], month_no: int, year: int) -> tuple[np.ndarray, list[str]]:
        profiles, fallback = [], []
        for fuel in FUELS:
            available = [donor_profile(region, month_no, year, fuel) for region in donors]
            available = [p for p in available if p is not None]
            if not available:
                profiles.append(np.full(24, 1 / 24))
                fallback.append(fuel)
            else:
                avg = np.mean(available, axis=0)
                profiles.append(avg / avg.sum())
        return np.stack(profiles, axis=1), fallback

    group_cache: dict[tuple[str, int, int], np.ndarray] = {}

    def pooled_group_profiles(donors: tuple[str, ...], month_no: int, year: int) -> np.ndarray:
        key = ("|".join(donors), month_no, year)
        if key in group_cache:
            return group_cache[key]
        output = []
        for group in ("thermal", "nuclear", "hydro", "wind", "solar"):
            if group == "thermal":
                rows = []
                for region in donors:
                    frame = hourly[region]
                    times = pd.to_datetime(frame.time, utc=True)
                    selected = frame.loc[times.dt.year.eq(year) & times.dt.month.eq(month_no)
                                         & frame.valid].copy()
                    if selected.empty:
                        continue
                    selected["hour"] = pd.to_datetime(selected.time, utc=True).dt.hour
                    cols = [f"generation_{fuel}" for fuel in data.THERMAL]
                    row = selected.groupby("hour")[cols].mean().sum(axis=1).reindex(range(24), fill_value=0).to_numpy(float)
                    if row.sum() > 0:
                        rows.append(row / row.sum())
            else:
                fuel = group
                rows = [profile for region in donors
                        if (profile := donor_profile(region, month_no, year, fuel)) is not None]
            profile = np.mean(rows, axis=0) if rows else np.full(24, 1 / 24)
            output.append(profile / profile.sum())
        group_cache[key] = np.stack(output, axis=1)
        return group_cache[key]

    def predict(region: str, donors: tuple[str, ...], origin: pd.Timestamp,
                forecast_year: int, arm: str) -> dict | None:
        recent = latest_row(monthly, region, origin)
        prior_year = yoy_row(monthly, region, origin, forecast_year)
        if recent is None or prior_year is None:
            return None
        frame = hourly[region]
        times = pd.to_datetime(frame.time, utc=True)
        current = frame.loc[times.dt.year.eq(origin.year) & times.dt.month.eq(origin.month)
                            & frame.valid].copy()
        if current.empty:
            return None
        actual = np.stack([current[f"generation_{fuel}"].to_numpy(float) for fuel in FUELS], axis=1)
        valid = np.isfinite(actual).all(axis=1) & (actual >= 0).all(axis=1) & (actual.sum(axis=1) > 0)
        current, actual = current.loc[valid], actual[valid]
        if len(current) < 480:
            return None
        hours = pd.to_datetime(current.time, utc=True).dt.hour.to_numpy()
        latest_mix = mix(recent)
        profiles, fallbacks = pooled_profiles(donors, origin.month, forecast_year - 1)
        if arm == "local_latest_fullmix_shape":
            prediction = (profiles[hours] * latest_mix[None, :]) @ EF
            prediction /= np.maximum((profiles[hours] * latest_mix[None, :]).sum(axis=1), 1e-12)
        elif arm == "local_yoy_fullmix_shape":
            yoy_mix = mix(prior_year)
            prediction = (profiles[hours] * yoy_mix[None, :]) @ EF
            prediction /= np.maximum((profiles[hours] * yoy_mix[None, :]).sum(axis=1), 1e-12)
        elif arm == "local_latest_thermal_ef_coarse":
            latest_thermal = sum(latest_mix[FUELS.index(f)] for f in data.THERMAL)
            thermal_ef = (latest_mix[[FUELS.index(f) for f in data.THERMAL]]
                          @ EF[[FUELS.index(f) for f in data.THERMAL]] / latest_thermal
                          if latest_thermal > 0 else float("nan"))
            if not np.isfinite(thermal_ef):
                return None
            coarse = group_mix(prior_year)
            grouped = pooled_group_profiles(donors, origin.month, forecast_year - 1)
            by_hour = grouped[hours] * coarse[None, :]
            prediction = by_hour[:, 0] * thermal_ef / np.maximum(by_hour.sum(axis=1), 1e-12)
        elif arm == "flat_latest_fullmix":
            prediction = np.full(len(current), float(latest_mix @ EF))
            fallbacks = []
        else:
            raise ValueError(arm)
        truth = actual @ EF / actual.sum(axis=1)
        residual = truth - prediction
        return {"truth": truth, "prediction": prediction, "residual": residual,
                "fallbacks": fallbacks, "latest_period": str(recent.period_start.date()),
                "yoy_period": str(prior_year.period_start.date())}

    def half_months(year: int) -> list[pd.Timestamp]:
        return [pd.Timestamp(f"{year}-{m:02d}-01", tz="UTC") for m in range(1, 7)]

    calibrations: dict[tuple[str, str, str], list[dict | None]] = {}
    for outer in TARGETS:
        for cal_region in SOURCE:
            if outer in SOURCE and outer == cal_region:
                continue
            donors = tuple(r for r in SOURCE if r not in (outer, cal_region))
            for arm in ARMS:
                calibrations[(outer, cal_region, arm)] = [
                    predict(cal_region, donors, month, 2025, arm) for month in half_months(2025)]

    scores = []
    for outer in TARGETS:
        donors = tuple(r for r in SOURCE if r != outer)
        target_predictions = {arm: [predict(outer, donors, month, 2026, arm)
                                   for month in half_months(2026)] for arm in ARMS}
        for arm in ARMS:
            cal_by_region = {}
            for cal_region in SOURCE:
                if outer in SOURCE and outer == cal_region:
                    continue
                rows = [r for r in calibrations[(outer, cal_region, arm)] if r is not None]
                if rows:
                    cal_by_region[cal_region] = rows
            if len(cal_by_region) < 9:
                raise ValueError(f"Too few calibration regions for {outer}/{arm}: {len(cal_by_region)}")
            residuals = {region: np.concatenate([row["residual"] for row in rows])
                         for region, rows in cal_by_region.items()}
            values, weights, region_q = [], [], []
            for residual in residuals.values():
                absolute = np.abs(residual)
                values.extend(absolute.tolist())
                weights.extend(np.full(len(absolute), 1 / (len(residuals) * len(absolute))).tolist())
                region_q.append(float(np.quantile(absolute, .90)))
            pooled = weighted_quantile(np.asarray(values), np.asarray(weights), .90)
            max_q = max(region_q)
            for month, target in zip(half_months(2026), target_predictions[arm]):
                if target is None:
                    continue
                for calibration, radius in (("source_equal_pooled_q90", pooled), ("max_source_q90", max_q)):
                    y, pred = target["truth"], target["prediction"]
                    low, high = pred - radius, pred + radius
                    score = 2 * radius + 20 * np.maximum(low - y, 0) + 20 * np.maximum(y - high, 0)
                    scores.append({"target_region": outer, "month": str(month.date()), "arm": arm,
                                   "calibration": calibration, "n_hours": len(y),
                                   "point_mae": float(np.mean(np.abs(y - pred))),
                                   "bias": float(np.mean(y - pred)),
                                   "coverage": float(np.mean((y >= low) & (y <= high))),
                                   "width": 2 * radius, "winkler": float(np.mean(score)),
                                   "fallback_fuels": ";".join(target["fallbacks"]),
                                   "latest_mix_period": target["latest_period"],
                                   "yoy_mix_period": target["yoy_period"],
                                   "calibration_sources": len(cal_by_region)})

    frame = pd.DataFrame(scores)
    if frame.empty:
        raise ValueError("No eligible V62 target predictions")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    frame.to_csv(OUTPUT / "region_month_scores.csv", index=False)
    region = (frame.groupby(["arm", "calibration", "target_region"])
              .apply(lambda part: pd.Series({
                  "coverage": np.average(part.coverage, weights=part.n_hours),
                  "point_mae": np.average(part.point_mae, weights=part.n_hours),
                  "bias": np.average(part.bias, weights=part.n_hours),
                  "width": np.average(part.width, weights=part.n_hours),
                  "winkler": np.average(part.winkler, weights=part.n_hours),
                  "months": part.month.nunique(),
                  "fallback_months": int(part.fallback_fuels.ne("").sum())}), include_groups=False).reset_index())
    region.to_csv(OUTPUT / "region_scores.csv", index=False)
    macro = (region.groupby(["arm", "calibration"])
             .agg(regions=("target_region", "nunique"), macro_coverage=("coverage", "mean"),
                  worst_coverage=("coverage", "min"), point_mae=("point_mae", "mean"),
                  bias=("bias", "mean"), mean_width=("width", "mean"),
                  winkler=("winkler", "mean"), mean_months=("months", "mean"),
                  fallback_regions=("fallback_months", lambda x: int((x > 0).sum())))
             .reset_index())
    macro.to_csv(OUTPUT / "macro.csv", index=False)
    if sha256(protocol) != protocol_hash:
        raise RuntimeError("Frozen protocol changed during scoring")
    summary = {"status": "DEVELOPMENT_ONLY_ALL_REGIONS_PREVIOUSLY_OPENED",
               "target_regions": list(TARGETS), "source_regions": list(SOURCE),
               "calibration_period": "2025 H1", "target_period": "2026 H1",
               "arms": list(ARMS), "release_assumption": "source-prepared month_end_plus_30d; strict period_end and available_at",
               "label_definition": "EIA fixed-factor positive reported-generation proxy",
               "script_sha256": sha256(Path(__file__).resolve()), "protocol_sha256": protocol_hash,
               "monthly_release_audit": release_audits, "input_sha256": input_hashes,
               "macro": macro.to_dict("records")}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
