"""V280: test V279 channels on EIA's official PACE/LGEE CO2 workbooks."""
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

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V280_EIA_OFFICIAL_LABEL_PACE_LGEE_PROTOCOL.md"
DATA = ROOT / "results/v280-eia-official-label-pace-lgee"
OUT = DATA / "run-r1"
SUPPORT = ROOT / "results/v269-asymmetric-empirical-ef-support-r2/cems_asymmetric_support_intervals.csv"
TARGETS = ("PACE", "LGEE")
DONORS = ("BANC", "BPAT", "CISO", "ERCO", "FPL", "MISO", "NYIS")
MONTHLY_FUELS = {
    "coal": ("NG: COL",),
    "gas": ("NG: NG",),
    "oil": ("NG: OIL",),
    "nuclear": ("NG: NUC",),
    "hydro": ("NG: WAT", "NG: PS"),
    "wind": ("NG: WND", "NG: WNB"),
    "solar": ("NG: SUN", "NG: SNB"),
    "other": ("NG: GEO", "NG: OES", "NG: UES", "NG: OTH", "NG: UNK", "NG: BAT"),
}
FUEL_ORDER = ("coal", "gas", "oil", "other", "nuclear", "hydro", "wind", "solar")
GEN_COL = "Positive Generation"
CI_COL = "CO2 Emissions Intensity for Generated Electricity"
REPEATS = 100
N_LOCAL_DAYS = 12
METHODS = ("support_minimum", "fixed_four", "full8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def parse_day(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def read_workbook(ba: str, path: Path) -> tuple[dict[tuple[int, int], dict], dict]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Published Hourly Data"]
    rows = ws.iter_rows(values_only=True)
    headers = next(rows)
    ix = {str(v): j for j, v in enumerate(headers) if v is not None}
    required = ("Local date", "Hour", GEN_COL, CI_COL,
                *(col for cols in MONTHLY_FUELS.values() for col in cols))
    missing = [col for col in required if col not in ix]
    if missing:
        raise ValueError(f"{ba} missing workbook fields {missing}")

    by_month: dict[tuple[int, int], list[tuple]] = {}
    row_counts: dict[tuple[int, int], int] = {}
    for row in rows:
        day = parse_day(row[ix["Local date"]])
        if day is None or day.year != 2025:
            continue
        hour, total, ci = finite(row[ix["Hour"]]), finite(row[ix[GEN_COL]]), finite(row[ix[CI_COL]])
        if hour is None or total is None or ci is None or total <= 0:
            continue
        fuel_values = []
        for fuel in FUEL_ORDER:
            # Blank source-specific columns denote zero output in EIA's hourly table.
            fuel_values.append(sum(max(0., finite(row[ix[col]]) or 0.) for col in MONTHLY_FUELS[fuel]))
        fuel_total = sum(fuel_values)
        if fuel_total <= 0:
            continue
        key = (day.year, day.month)
        by_month.setdefault(key, []).append((day, int(hour), fuel_values, ci * 453.59237, total, fuel_total))
        row_counts[key] = row_counts.get(key, 0) + 1
    wb.close()

    result = {}
    month_quality = {}
    for (year, month), data in sorted(by_month.items()):
        # Keep only local dates with 24 distinct hourly slots; DST transition
        # days are excluded instead of coercing 23/25-hour dates to 24 hours.
        dates: dict[date, list[tuple]] = {}
        for rec in data:
            dates.setdefault(rec[0], []).append(rec)
        complete = []
        for day, day_rows in dates.items():
            slots = [r[1] for r in day_rows]
            if len(day_rows) == 24 and len(set(slots)) == 24 and set(slots) == set(range(1, 25)):
                complete.extend(sorted(day_rows, key=lambda r: r[1]))
        if len(complete) < 24 * 20:
            continue
        fuels = np.asarray([r[2] for r in complete], float)
        ci = np.asarray([r[3] for r in complete], float)
        published_gen = np.asarray([r[4] for r in complete], float)
        gen = fuels.sum(axis=1)
        coverage = float(gen.sum() / published_gen.sum())
        if not 0.95 <= coverage <= 1.05:
            month_quality[f"{year}-{month:02d}"] = {"fuel_to_positive_generation": coverage, "status": "excluded"}
            continue
        margins = fuels.sum(axis=0)
        center = float(np.average(ci, weights=gen))
        times = pd.DatetimeIndex([
            pd.Timestamp(r[0]) + pd.Timedelta(hours=r[1] - 1) for r in complete
        ])
        d = {"time": times, "cif": ci, "fuel_totals": margins,
             "fuel_profile": fuels / np.maximum(margins[None, :], 1e-12),
             "monthly_y": center}
        result[(year, month)] = d
        month_quality[f"{year}-{month:02d}"] = {
            "complete_days": int(len(complete) // 24),
            "fuel_to_positive_generation": coverage,
            "published_intensity_center_g_kwh": center,
            "status": "included",
        }
    return result, {"ba": ba, "file": path.name, "sha256": sha256(path),
                    "bytes": path.stat().st_size, "months": month_quality,
                    "raw_2025_rows_after_positive_finite_filter": {
                        f"{year}-{month:02d}": count
                        for (year, month), count in row_counts.items()
                    }}


def target_rows(ba: str, months: dict, donor_series: dict, support: pd.DataFrame,
                rng: np.random.Generator) -> tuple[list[dict], list[dict], dict]:
    records, budgets = [], []
    for (year, month), d in sorted(months.items()):
        quarter = (month - 1) // 3 + 1
        ef_low, ef_high = sensor.factor_support(quarter, support)
        score_data = {}
        for method in METHODS:
            lo, hi = sensor.robust_hour_ranges(d, method, ef_low, ef_high)
            score_data[method] = (lo, hi, sensor.daily_scores(d, lo, hi))
        source_scores = []
        donor_counts = {}
        for donor in DONORS:
            source_d = donor_series.get(donor, {}).get((year, month))
            if source_d is None:
                donor_counts[donor] = 0
                continue
            donor_day_scores = sensor.nominal_source_scores(source_d).tolist()
            donor_counts[donor] = len(donor_day_scores)
            source_scores.extend(donor_day_scores)
        if len(source_scores) < 100:
            raise RuntimeError(f"insufficient same-month donor days for {ba} {year}-{month:02d}: {donor_counts}")
        dayids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
        days = np.asarray(sorted(score_data["full8"][2]), dtype=np.int64)
        if len(days) < N_LOCAL_DAYS + 10:
            raise RuntimeError(f"too few target days for {ba} {year}-{month:02d}")
        support_channels = sensor.method_channels("support_minimum", d)
        budgets.append({"ba": ba, "year": year, "month": month, "method": "support_minimum",
                        "numeric_fields_including_total": 1 + len(support_channels),
                        "channels": "+".join(support_channels)})
        for rep in range(REPEATS):
            sampled = rng.choice(days, size=N_LOCAL_DAYS, replace=False)
            test_days = np.setdiff1d(days, sampled)
            test_mask = np.isin(dayids, test_days)
            for method in METHODS:
                lo, hi, scores = score_data[method]
                q = dayrisk.radius(np.asarray([scores[int(day)] for day in sampled]))
                vals = sensor.score_test(d, lo, hi, q, test_mask)
                records.append({"ba": ba, "year": year, "month": month, "repeat": rep,
                    "method": method, "hour_coverage": vals[0], "width": vals[1],
                    "winkler90": vals[2], "simultaneous_day_coverage": vals[3],
                    "ex_post_robust_day_set_containment": vals[4], "n_days": len(days)})
            vals = sensor.score_source_baseline(d, dayrisk.radius(np.asarray(source_scores)), test_mask)
            records.append({"ba": ba, "year": year, "month": month, "repeat": rep,
                "method": "source_loo", "hour_coverage": vals[0], "width": vals[1],
                "winkler90": vals[2], "simultaneous_day_coverage": vals[3],
                "ex_post_robust_day_set_containment": np.nan, "n_days": len(days)})
    return records, budgets, {"ba": ba, "included_months": len(months)}


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    input_dir = DATA
    support = pd.read_csv(SUPPORT)
    raw_target, target_metadata, raw_donors, donor_metadata = {}, {}, {}, {}
    for ba in TARGETS:
        raw_target[ba], target_metadata[ba] = read_workbook(ba, input_dir / f"{ba}.xlsx")
    for ba in DONORS:
        raw_donors[ba], donor_metadata[ba] = read_workbook(ba, ROOT / "results" / "v272-eia-monitor" / f"{ba}.xlsx")
    rng = np.random.default_rng(280)
    records, budgets, roster = [], [], {}
    for ba in TARGETS:
        rows, field_rows, item = target_rows(ba, raw_target[ba], raw_donors, support, rng)
        records.extend(rows); budgets.extend(field_rows); roster[ba] = item
    reps = pd.DataFrame(records)
    rm = reps.groupby(["ba", "month", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_repeats=("repeat", "count"))
    by_ba = rm.groupby(["ba", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_months=("month", "count"))
    macro = by_ba.groupby("method", as_index=False).agg(
        macro_hour_coverage=("hour_coverage", "mean"), macro_width=("width", "mean"),
        macro_winkler90=("winkler90", "mean"),
        macro_simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        macro_ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_bas=("ba", "nunique"))
    OUT.mkdir(parents=True, exist_ok=True)
    reps.to_csv(OUT / "replicate_metrics.csv", index=False)
    rm.to_csv(OUT / "ba_month_metrics.csv", index=False)
    by_ba.to_csv(OUT / "ba_metrics.csv", index=False)
    macro.to_csv(OUT / "macro_metrics.csv", index=False)
    pd.DataFrame(budgets).to_csv(OUT / "field_budget.csv", index=False)
    metadata = {"protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
        "support_sha256": sha256(SUPPORT), "target_files": target_metadata,
        "donor_files": donor_metadata, "target_roster": roster,
        "donor_roster": list(DONORS), "repeats": REPEATS, "local_days": N_LOCAL_DAYS,
        "methods": list(METHODS) + ["source_loo"],
        "label_caveat": "EIA official BA-generated hourly CO2 estimate, not CEMS direct full-grid emissions; target monthly center is oracle",
        "status": "new EIA label-source regional pressure test; target regions are not new to all project work"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(macro.to_string(index=False))
    print(by_ba.to_string(index=False))
    print(f"PACE months={roster['PACE']['included_months']}; LGEE months={roster['LGEE']['included_months']}; output={OUT}")


if __name__ == "__main__":
    main()
