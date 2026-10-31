"""V282 exploratory paired experiment: prior-year local EF support vs CEMS support."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import v224_eia_sparse_daily_maxrisk as dayrisk
import v279_asymmetric_ef_sensor_frontier as sensor
import v280_eia_official_label_pace_lgee as v280

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V282_PRIOR_YEAR_LOCAL_EF_SUPPORT_PROTOCOL.md"
DATA = ROOT / "results/v280-eia-official-label-pace-lgee"
OUT = ROOT / "results/v282-prior-year-local-ef-support-r1"
TARGETS = ("PACE", "LGEE")
FACTOR_COLUMNS = {"coal": ("CO2 Factor: COL", "NG: COL"),
                  "gas": ("CO2 Factor: NG", "NG: NG"),
                  "oil": ("CO2 Factor: OIL", "NG: OIL")}
FUEL_INDEX = {"coal": 0, "gas": 1, "oil": 2}
CONVERSION = 453.59237
REPEATS = 100
N_DAYS = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_day(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def previous_year_support(ba: str, cems: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    path = DATA / f"{ba}.xlsx"
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Published Hourly Data"]
    rows = ws.iter_rows(values_only=True)
    headers = next(rows)
    ix = {str(value): i for i, value in enumerate(headers) if value is not None}
    supports = {}
    counts = {}
    for fuel, (factor_col, generation_col) in FACTOR_COLUMNS.items():
        if factor_col not in ix or generation_col not in ix:
            raise ValueError(f"{ba} missing {factor_col}/{generation_col}")
        supports[fuel] = []
        counts[fuel] = 0
    for row in rows:
        day = parse_day(row[ix["Local date"]])
        if day is None or day.year != 2024:
            continue
        for fuel, (factor_col, generation_col) in FACTOR_COLUMNS.items():
            try:
                gen = float(row[ix[generation_col]] or 0.)
                ef = float(row[ix[factor_col]]) * CONVERSION
            except (TypeError, ValueError):
                continue
            if gen > 0 and np.isfinite(ef):
                supports[fuel].append(ef)
                counts[fuel] += 1
    wb.close()
    low, high = sensor.ALL_EF.copy(), sensor.ALL_EF.copy()
    used = {}
    for fuel, idx in FUEL_INDEX.items():
        values = np.asarray(supports[fuel], float)
        if len(values):
            low[idx], high[idx] = float(values.min()), float(values.max())
            used[fuel] = {"source": "2024_local_active_hours", "n": len(values),
                          "low": float(low[idx]), "high": float(high[idx])}
        else:
            used[fuel] = {"source": "quarterly_CEMS_fallback"}
    return low, high, {"ba": ba, "factor_support": used}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    cems = pd.read_csv(v280.SUPPORT)
    prior = {ba: previous_year_support(ba, cems) for ba in TARGETS}
    targets = {ba: v280.read_workbook(ba, DATA / f"{ba}.xlsx")[0] for ba in TARGETS}
    rng = np.random.default_rng(280)
    records = []
    for ba in TARGETS:
        for (year, month), d in sorted(targets[ba].items()):
            quarter = (month - 1) // 3 + 1
            c_low, c_high = sensor.factor_support(quarter, cems)
            p_low, p_high, _ = prior[ba]
            range_map = {}
            for method, low, high in (("cems_support", c_low, c_high),
                                      ("prior_year_local_ef", p_low, p_high)):
                lo, hi = sensor.robust_hour_ranges(d, "support_minimum", low, high)
                range_map[method] = (lo, hi, sensor.daily_scores(d, lo, hi))
            day_ids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
            days = np.asarray(sorted(range_map["cems_support"][2]), dtype=np.int64)
            for rep in range(REPEATS):
                sampled = rng.choice(days, size=N_DAYS, replace=False)
                test_days = np.setdiff1d(days, sampled)
                test_mask = np.isin(day_ids, test_days)
                for method, (lo, hi, scores) in range_map.items():
                    radius = dayrisk.radius(np.asarray([scores[int(day)] for day in sampled]))
                    vals = sensor.score_test(d, lo, hi, radius, test_mask)
                    records.append({"ba": ba, "year": year, "month": month,
                        "repeat": rep, "method": method, "hour_coverage": vals[0],
                        "width": vals[1], "winkler90": vals[2],
                        "simultaneous_day_coverage": vals[3],
                        "ex_post_robust_day_set_containment": vals[4]})
    reps = pd.DataFrame(records)
    month = reps.groupby(["ba", "month", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"))
    ba = month.groupby(["ba", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_months=("month", "count"))
    macro = ba.groupby("method", as_index=False).agg(
        macro_hour_coverage=("hour_coverage", "mean"), macro_width=("width", "mean"),
        macro_winkler90=("winkler90", "mean"),
        macro_simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        macro_ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_bas=("ba", "nunique"))
    OUT.mkdir(parents=True)
    reps.to_csv(OUT / "replicate_metrics.csv", index=False)
    month.to_csv(OUT / "ba_month_metrics.csv", index=False)
    ba.to_csv(OUT / "ba_metrics.csv", index=False)
    macro.to_csv(OUT / "macro_metrics.csv", index=False)
    source = pd.read_csv(DATA / "run-r1/ba_metrics.csv")
    source = source[source.method.eq("source_loo")]
    metadata = {"protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
        "target_workbooks": {x: sha256(DATA / f"{x}.xlsx") for x in TARGETS},
        "cems_support_sha256": sha256(v280.SUPPORT), "prior_year_factor_support": {k: v[2] for k, v in prior.items()},
        "methods": ["cems_support", "prior_year_local_ef", "source_loo (V280 paired baseline)"],
        "repeats": REPEATS, "local_calibration_days": N_DAYS,
        "caveat": "post-hoc exploratory; 2025 EIA target labels had been opened in V280",
        "source_loo_baseline": source.to_dict(orient="records")}
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(macro.to_string(index=False))
    print(ba.to_string(index=False))


if __name__ == "__main__":
    main()
