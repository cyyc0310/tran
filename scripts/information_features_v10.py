"""As-of monthly and lagged total-generation features for V10.

The raw EIA total is signed whole-BA net generation. It is not the sum of the
eight positive fuel-proxy buckets. Its assumed lag is an experimental budget,
not evidence of an original EIA publication vintage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.coarse_dynamics_v9 import _ridge
from transcif.research.coarse_forecast_v7 import COLUMNS

SOURCES = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
TARGETS = ("AZPS", "DUK", "LGEE", "PACE", "PSCO", "SRP")
REGIONS = SOURCES + TARGETS
RAW_COLUMNS = ("Balancing Authority", "UTC Time at End of Hour", "Net Generation (MW)")
BASE_NAMES = (*[f"harmonic_{x}" for x in COLUMNS],
              *[f"released_{x}" for x in COLUMNS],
              "released_log_mean_mw", "released_age_days", "released_count",
              "annual_sin", "annual_cos", "weekly_sin", "weekly_cos")
RANGE_NAMES = ("log_mean_28", "log_q10_28", "log_q90_28",
               "log_q90_q10_28", "cv_28", "log_mean_to_released_28",
               "log_mean_to_previous_28", "support_28", "support_previous_28")
SHAPE_NAMES = (*[f"hour_{h:02d}" for h in range(24)],
               "same_weekday_profiles", "fallback_all_profiles", "complete_profiles")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_total_telemetry(native_paths: Iterable[Path], output: Path,
                            regions: Iterable[str] = REGIONS,
                            chunksize: int = 100_000) -> dict:
    """Persist only BA, UTC hour-end and signed G, plus raw-file hash audit.

    String chunks select registered BAs before any timestamp or numeric parsing;
    no fuel, demand, CIF, or retrospective valid-mask column is read.
    """
    paths = tuple(Path(path) for path in native_paths)
    wanted = tuple(regions)
    if (not paths or len(set(paths)) != len(paths) or not wanted or
            len(set(wanted)) != len(wanted) or not set(wanted) <= set(REGIONS)
            or chunksize < 1):
        raise ValueError("unique native files and registered regions required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pieces: list[pd.DataFrame] = []
    sources: list[dict] = []
    for path in paths:
        count = 0
        for chunk in pd.read_csv(path, usecols=list(RAW_COLUMNS), dtype=str,
                                 chunksize=chunksize):
            chunk = chunk.loc[chunk["Balancing Authority"].isin(wanted), list(RAW_COLUMNS)]
            if chunk.empty:
                continue
            end = pd.to_datetime(chunk["UTC Time at End of Hour"], utc=True,
                                 format="mixed", errors="raise")
            value = pd.to_numeric(chunk["Net Generation (MW)"], errors="raise")
            if end.isna().any() or not end.eq(end.dt.floor("h")).all():
                raise ValueError("registered BA must have finite hour-aligned UTC end")
            if np.isinf(value.to_numpy(float)).any():
                raise ValueError("infinite G")
            pieces.append(pd.DataFrame({"region": chunk["Balancing Authority"].to_numpy(),
                                        "hour_end": end.to_numpy(),
                                        "g_mw": value.to_numpy(float)}))
            count += len(chunk)
        sources.append({"path": str(path.resolve()), "sha256": _sha(path),
                        "registered_rows": count})
    if not pieces:
        raise ValueError("no registered BA telemetry")
    table = pd.concat(pieces, ignore_index=True).sort_values(["region", "hour_end"])
    if table.duplicated(["region", "hour_end"]).any():
        raise ValueError("duplicate BA/hour-end across native files")
    audit = {"schema": ["region", "hour_end", "g_mw"],
             "meaning": "signed native Net Generation (MW), not positive fuel-proxy sum",
             "publication_vintage_known": False, "raw_sources": sources,
             "regions": {region: {"rows": int((table.region == region).sum()),
                         "missing_g": int(table.loc[table.region.eq(region), "g_mw"].isna().sum())}
                         for region in wanted}}
    output.mkdir(parents=True)
    table.to_csv(output / "total_generation.csv", index=False)
    audit["telemetry_sha256"] = _sha(output / "total_generation.csv")
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return audit


def _positive_window(series: pd.Series, start: pd.Timestamp,
                     stop: pd.Timestamp) -> np.ndarray:
    values = series.loc[(series.index > start) & (series.index <= stop)].to_numpy(float)
    return values[np.isfinite(values) & (values > 0)]


def _shape(series: pd.Series, origin: pd.Timestamp,
           cutoff: pd.Timestamp) -> tuple[np.ndarray, float, float, float]:
    """Complete UTC days in preceding 28 days; each profile has mean one."""
    profiles, same = [], []
    for day in pd.date_range(cutoff.normalize() - pd.Timedelta(days=28),
                             cutoff.normalize(), freq="D", tz="UTC"):
        end = day + pd.Timedelta(days=1)
        if end > cutoff:
            continue
        hours = pd.date_range(day + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC")
        values = series.reindex(hours).to_numpy(float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            continue
        profile = values / values.mean()
        profiles.append(profile)
        if day.dayofweek == origin.dayofweek:
            same.append(profile)
    selected = same if same else profiles
    if not selected:
        return np.full(24, np.nan), 0., 1., 0.
    point = np.median(np.stack(selected), axis=0)
    point /= point.mean()
    return point, float(len(same)), float(not same), float(len(profiles))


def build_features(monthly: pd.DataFrame, telemetry: pd.DataFrame, region: str,
                   origins: Iterable, lag_days: int = 7) -> dict:
    """Return named B/R/S arrays and a per-origin availability audit.

    No target hourly CIF, fuel trajectory, or retrospective quality mask enters.
    Missing numeric predictors remain NaN for source-only imputation downstream.
    """
    if region not in REGIONS or not isinstance(lag_days, int) or lag_days < 0:
        raise ValueError("registered region and nonnegative integer lag required")
    dates = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    if dates.hasnans or dates.duplicated().any():
        raise ValueError("finite distinct origins required")
    n = len(dates)
    base = np.full((n, len(BASE_NAMES)), np.nan)
    ranges = np.full((n, len(RANGE_NAMES)), np.nan)
    shape = np.full((n, len(SHAPE_NAMES)), np.nan)
    if n == 0:
        return {"base": base, "range": ranges, "shape": shape,
                "names": {"base": BASE_NAMES, "range": RANGE_NAMES, "shape": SHAPE_NAMES},
                "audit": {"region": region, "lag_days": lag_days, "rows": []}}
    cutoffs = dates - pd.Timedelta(days=lag_days)
    # Time and region gates precede conversion/validation of numeric values.
    m = monthly.loc[monthly.region.eq(region)].copy()
    for key in ("period_start", "period_end", "available_at"):
        m[key] = pd.to_datetime(m[key], utc=True, errors="raise")
    m = m.loc[m.period_end.le(dates.max()) & m.available_at.le(dates.max())].copy()
    m = m.sort_values(["period_end", "available_at"]).reset_index(drop=True)
    if (m.available_at < m.period_end).any():
        raise ValueError("released month cannot precede its period end")
    if m.period_start.duplicated().any():
        raise ValueError("one released monthly vintage per period required")
    if len(m):
        values = m[list(COLUMNS)].to_numpy(float)
        if (not np.isfinite(values).all() or np.any(values < 0) or
                not np.allclose(values.sum(axis=1), 1)):
            raise ValueError("invalid released monthly composition")
    t = telemetry.loc[telemetry.region.eq(region)].copy()
    t["hour_end"] = pd.to_datetime(t.hour_end, utc=True, errors="raise")
    t = t.loc[t.hour_end.le(cutoffs.max())].copy()
    t["g_mw"] = pd.to_numeric(t.g_mw, errors="raise")
    if np.isinf(t.g_mw.to_numpy(float)).any() or t.hour_end.isna().any():
        raise ValueError("nonfinite timestamp or infinite registered G")
    if t.hour_end.duplicated().any() or not t.hour_end.eq(t.hour_end.dt.floor("h")).all():
        raise ValueError("unique UTC hour ends required")
    series = t.set_index("hour_end").g_mw.sort_index()
    # The exact V9 harmonic point needs only a released prefix. Avoid building
    # five other methods, pseudo-origin residuals, and random draws per day.
    harmonic_cache: dict[tuple, np.ndarray] = {}
    rows: list[dict] = []
    for i, (origin, cutoff) in enumerate(zip(dates, cutoffs)):
        prefix = m.loc[m.period_end.le(origin) & m.available_at.le(origin)]
        last = prefix.iloc[-1] if len(prefix) else None
        cache_key = (tuple(prefix.index), origin.year, origin.month)
        if cache_key not in harmonic_cache:
            harmonic_cache[cache_key] = _ridge(prefix, origin)
        base[i, :8] = harmonic_cache[cache_key]
        base[i, 18] = len(prefix)
        annual = 2 * np.pi * (origin.dayofyear - 1) / 365.25
        weekly = 2 * np.pi * origin.dayofweek / 7
        base[i, 19:] = [np.sin(annual), np.cos(annual), np.sin(weekly), np.cos(weekly)]
        released_mean = np.nan
        if last is not None:
            mix = last[list(COLUMNS)].to_numpy(float)
            total, hours = float(last.total_mwh), float(last.observed_hours)
            if (not np.isfinite(mix).all() or np.any(mix < 0) or
                    not np.isclose(mix.sum(), 1) or not np.isfinite(total)
                    or total <= 0 or not np.isfinite(hours) or hours <= 0):
                raise ValueError("invalid released monthly observation")
            released_mean = total / hours
            base[i, 8:16] = mix
            base[i, 16] = np.log(released_mean)
            base[i, 17] = (origin - last.period_end).total_seconds() / 86400
        past = cutoff - pd.Timedelta(days=28)
        prev = past - pd.Timedelta(days=28)
        current = _positive_window(series, past, cutoff)
        previous = _positive_window(series, prev, past)
        ranges[i, 7] = len(current) / 672
        ranges[i, 8] = len(previous) / 672
        if len(current):
            mean = float(current.mean())
            q10, q90 = np.quantile(current, [.1, .9])
            ranges[i, :5] = [np.log(mean), np.log(q10), np.log(q90),
                             np.log(q90 / q10), current.std() / mean]
            if np.isfinite(released_mean) and released_mean > 0:
                ranges[i, 5] = np.log(mean / released_mean)
            if len(previous):
                ranges[i, 6] = np.log(mean / previous.mean())
        profile, same_count, fallback, complete_count = _shape(series, origin, cutoff)
        shape[i, :24] = profile
        shape[i, 24:] = [same_count, fallback, complete_count]
        used = series.loc[(series.index > prev) & (series.index <= cutoff)]
        used = used.loc[np.isfinite(used) & used.gt(0)]
        rows.append({"origin": origin.isoformat(),
                     "max_hour_end_allowed": cutoff.isoformat(),
                     "max_hour_end_used": used.index.max().isoformat() if len(used) else None,
                     "released_period_end": last.period_end.isoformat() if last is not None else None,
                     "released_available_at": last.available_at.isoformat() if last is not None else None,
                     "positive_g_28d": int(len(current)),
                     "positive_g_previous_28d": int(len(previous)),
                     "same_weekday_profiles": int(same_count),
                     "all_complete_profiles": int(complete_count)})
    return {"base": base, "range": ranges, "shape": shape,
            "names": {"base": BASE_NAMES, "range": RANGE_NAMES, "shape": SHAPE_NAMES},
            "audit": {"region": region, "lag_days": lag_days, "rows": rows,
                      "total_mean_definition": "released positive proxy MWh / observed hours",
                      "telemetry_definition": "signed native BA net generation MW"}}
