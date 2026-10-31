"""V286 frozen temporal holdout for prior-year local EF support."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import v224_eia_sparse_daily_maxrisk as dayrisk
import v279_asymmetric_ef_sensor_frontier as sensor
import v280_eia_official_label_pace_lgee as v280

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V286_PRIOR_YEAR_EF_TEMPORAL_HOLDOUT_PROTOCOL.md"
DATA = ROOT / "results/v280-eia-official-label-pace-lgee"
DONOR_DIR = ROOT / "results/v272-eia-monitor"
SUPPORT = ROOT / "results/v269-asymmetric-empirical-ef-support-r2/cems_asymmetric_support_intervals.csv"
OUT = ROOT / "results/v286-prior-year-ef-temporal-holdout-r1"
TARGETS = ("PACE", "LGEE")
DONORS = ("BANC", "BPAT", "CISO", "ERCO", "FPL", "MISO", "NYIS")
TARGET_YEARS = (2020, 2021, 2022, 2023)
PRIOR_YEARS = (2019, 2020, 2021, 2022)
FUEL_ORDER = v280.FUEL_ORDER
FUEL_COLUMNS = v280.MONTHLY_FUELS
FACTOR_COLUMNS = {"coal": ("CO2 Factor: COL", "NG: COL"),
                  "gas": ("CO2 Factor: NG", "NG: NG"),
                  "oil": ("CO2 Factor: OIL", "NG: OIL")}
FUEL_INDEX = {"coal": 0, "gas": 1, "oil": 2}
CONVERSION = 453.59237
REPEATS = 100
CAL_DAYS = 9
METHODS = ("cems_support", "prior_year_local_ef", "source_loo")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def read_ba(ba: str, path: Path, target: bool) -> tuple[dict, dict, dict]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Published Hourly Data"]
    rows = ws.iter_rows(values_only=True)
    headers = next(rows)
    ix = {str(value): i for i, value in enumerate(headers) if value is not None}
    required = ["Local date", "Hour", v280.GEN_COL, v280.CI_COL,
                *(c for group in FUEL_COLUMNS.values() for c in group)]
    if target:
        required.extend(c for pair in FACTOR_COLUMNS.values() for c in pair)
    missing = [c for c in required if c not in ix]
    if missing:
        raise ValueError(f"{ba} missing columns: {missing}")
    by_month: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    factor_values: dict[int, dict[str, list[float]]] = {
        y: {fuel: [] for fuel in FACTOR_COLUMNS} for y in PRIOR_YEARS
    }
    raw_rows = defaultdict(int)
    for row in rows:
        day = parse_date(row[ix["Local date"]])
        if day is None:
            continue
        year = day.year
        if year not in TARGET_YEARS and (not target or year not in PRIOR_YEARS):
            continue
        if target and year in PRIOR_YEARS:
            for fuel, (factor_col, generation_col) in FACTOR_COLUMNS.items():
                try:
                    gen = float(row[ix[generation_col]] or 0.)
                    factor = float(row[ix[factor_col]]) * CONVERSION
                except (TypeError, ValueError):
                    continue
                if gen > 0 and np.isfinite(factor):
                    factor_values[year][fuel].append(factor)
        if year not in TARGET_YEARS:
            continue
        hour = v280.finite(row[ix["Hour"]])
        total = v280.finite(row[ix[v280.GEN_COL]])
        ci = v280.finite(row[ix[v280.CI_COL]])
        if hour is None or total is None or ci is None or total <= 0:
            continue
        fuels = [sum(max(0., v280.finite(row[ix[c]]) or 0.) for c in FUEL_COLUMNS[fuel])
                 for fuel in FUEL_ORDER]
        if sum(fuels) <= 0:
            continue
        key = (year, day.month)
        by_month[key].append((day, int(hour), fuels, ci * CONVERSION, total))
        raw_rows[key] += 1
    wb.close()
    months, quality = {}, {}
    for (year, month), data in sorted(by_month.items()):
        day_rows = defaultdict(list)
        for record in data:
            day_rows[record[0]].append(record)
        complete = []
        for day, records in day_rows.items():
            slots = [r[1] for r in records]
            if len(records) == 24 and len(set(slots)) == 24 and set(slots) == set(range(1, 25)):
                complete.extend(sorted(records, key=lambda r: r[1]))
        if len(complete) < 24 * 20:
            quality[f"{year}-{month:02d}"] = {"complete_days": len(complete) // 24, "status": "too_few_days"}
            continue
        fuels = np.asarray([r[2] for r in complete], float)
        y = np.asarray([r[3] for r in complete], float)
        gen = fuels.sum(axis=1)
        published = np.asarray([r[4] for r in complete], float)
        ratio = float(gen.sum() / published.sum())
        if not .95 <= ratio <= 1.05:
            quality[f"{year}-{month:02d}"] = {"complete_days": len(complete) // 24,
                "fuel_to_positive_generation": ratio, "status": "fuel_total_mismatch"}
            continue
        margins = fuels.sum(axis=0)
        center = float(np.average(y, weights=gen))
        times = pd.DatetimeIndex([pd.Timestamp(r[0]) + pd.Timedelta(hours=r[1] - 1)
                                  for r in complete])
        months[(year, month)] = {"time": times, "cif": y, "fuel_totals": margins,
            "fuel_profile": fuels / np.maximum(margins[None, :], 1e-12), "monthly_y": center}
        quality[f"{year}-{month:02d}"] = {"complete_days": int(len(complete) // 24),
            "fuel_to_positive_generation": ratio, "monthly_center_g_kwh": center, "status": "included"}
    return months, factor_values, {"ba": ba, "file": path.name, "sha256": sha256(path),
                                   "months": quality, "raw_rows_by_month": {
                                       f"{y}-{m:02d}": n for (y, m), n in raw_rows.items()}}


def local_prior_bounds(ba: str, target_year: int, month: int,
                       factor_values: dict, support: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    quarter = (month - 1) // 3 + 1
    low, high = sensor.factor_support(quarter, support)
    year = target_year - 1
    used = {}
    for fuel, idx in FUEL_INDEX.items():
        values = np.asarray(factor_values[year][fuel], float)
        if len(values):
            low[idx], high[idx] = float(values.min()), float(values.max())
            used[fuel] = {"n_previous_year_active_hours": len(values),
                          "low_g_kwh": float(low[idx]), "high_g_kwh": float(high[idx]),
                          "source": str(year)}
        else:
            used[fuel] = {"source": "quarterly_cems_fallback"}
    return low, high, {"ba": ba, "target_year": target_year, "factor_support": used}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    support = pd.read_csv(SUPPORT)
    target_series, target_factors, target_meta = {}, {}, {}
    for ba in TARGETS:
        path = DATA / f"{ba}.xlsx"
        target_series[ba], target_factors[ba], target_meta[ba] = read_ba(ba, path, target=True)
    donor_series, donor_meta = {}, {}
    for ba in DONORS:
        path = DONOR_DIR / f"{ba}.xlsx"
        donor_series[ba], _, donor_meta[ba] = read_ba(ba, path, target=False)
    prepared, prior_meta = {}, {}
    for ba in TARGETS:
        for key, d in target_series[ba].items():
            year, month = key
            quarter = (month - 1) // 3 + 1
            c_low, c_high = sensor.factor_support(quarter, support)
            p_low, p_high, p_meta = local_prior_bounds(ba, year, month, target_factors[ba], support)
            score_map = {}
            for method, low, high in (("cems_support", c_low, c_high),
                                      ("prior_year_local_ef", p_low, p_high)):
                lo, hi = sensor.robust_hour_ranges(d, "support_minimum", low, high)
                score_map[method] = (lo, hi, sensor.daily_scores(d, lo, hi))
            source_scores = []
            for donor in DONORS:
                source_d = donor_series[donor].get(key)
                if source_d is not None:
                    source_scores.extend(sensor.nominal_source_scores(source_d).tolist())
            if len(source_scores) < 100:
                raise RuntimeError(f"insufficient donor scores for {ba} {year}-{month:02d}")
            prepared[(ba, year, month)] = {"data": d, "scores": score_map,
                                          "source": np.asarray(source_scores, float)}
            prior_meta[f"{ba}-{year}-{month:02d}"] = p_meta
    expected = {(ba, year, month) for ba in TARGETS for year in TARGET_YEARS for month in range(1, 13)}
    if set(prepared) != expected:
        absent = sorted(expected - set(prepared))
        extra = sorted(set(prepared) - expected)
        raise RuntimeError(f"frozen 48-month target roster incomplete: missing={absent}; extra={extra}")
    rng = np.random.default_rng(286)
    records = []
    for key, item in sorted(prepared.items()):
        ba, year, month = key
        d = item["data"]
        day_ids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
        days = np.asarray(sorted(item["scores"]["cems_support"][2]), dtype=np.int64)
        if len(days) <= CAL_DAYS + 5:
            raise RuntimeError(f"too few days for {key}")
        for rep in range(REPEATS):
            sampled = rng.choice(days, size=CAL_DAYS, replace=False)
            test_days = np.setdiff1d(days, sampled)
            test_mask = np.isin(day_ids, test_days)
            for method in METHODS:
                if method == "source_loo":
                    q = dayrisk.radius(item["source"])
                    vals = sensor.score_source_baseline(d, q, test_mask)
                    set_cov = np.nan
                else:
                    lo, hi, daily = item["scores"][method]
                    q = dayrisk.radius(np.asarray([daily[int(day)] for day in sampled]))
                    vals = sensor.score_test(d, lo, hi, q, test_mask)
                    set_cov = vals[4]
                records.append({"ba": ba, "year": year, "month": month, "repeat": rep,
                    "method": method, "hour_coverage": vals[0], "width": vals[1],
                    "winkler90": vals[2], "simultaneous_day_coverage": vals[3],
                    "ex_post_robust_day_set_containment": set_cov})
    reps = pd.DataFrame(records)
    month = reps.groupby(["ba", "year", "month", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"))
    by_year = month.groupby(["ba", "year", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_months=("month", "count"))
    by_ba = month.groupby(["ba", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_months=("month", "count"))
    macro = by_year.groupby("method", as_index=False).agg(
        macro_hour_coverage=("hour_coverage", "mean"), macro_width=("width", "mean"),
        macro_winkler90=("winkler90", "mean"),
        macro_simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        macro_ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_ba_years=("year", "count"))
    OUT.mkdir(parents=True)
    reps.to_csv(OUT / "replicate_metrics.csv", index=False)
    month.to_csv(OUT / "ba_month_metrics.csv", index=False)
    by_year.to_csv(OUT / "ba_year_metrics.csv", index=False)
    by_ba.to_csv(OUT / "ba_metrics.csv", index=False)
    macro.to_csv(OUT / "macro_metrics.csv", index=False)
    metadata = {"protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
        "target_files": target_meta, "donor_files": donor_meta,
        "support_sha256": sha256(SUPPORT), "target_roster": sorted(f"{b}-{y}-{m:02d}" for b,y,m in prepared),
        "previous_year_factor_support": prior_meta, "target_years": list(TARGET_YEARS),
        "calibration_days": CAL_DAYS, "repeats": REPEATS, "seed": 286,
        "methods": list(METHODS),
        "target_center_and_margins": "computed from target-year complete days; oracle hindsight condition",
        "label_caveat": "EIA BA-generated hourly CO2 estimate, not direct CEMS full-grid truth",
        "status": "temporal holdout on known BA; not geographic blind validation"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(macro.to_string(index=False))
    print(by_ba.to_string(index=False))
    print(by_year.to_string(index=False))


if __name__ == "__main__":
    main()
