"""V284 exploratory nested calibration-day budget frontier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v224_eia_sparse_daily_maxrisk as dayrisk
import v279_asymmetric_ef_sensor_frontier as sensor
import v280_eia_official_label_pace_lgee as v280
import v282_prior_year_local_ef_support as v282

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V284_LOCAL_DAY_BUDGET_PRIOR_EF_PROTOCOL.md"
DATA = ROOT / "results/v280-eia-official-label-pace-lgee"
OUT = ROOT / "results/v284-local-day-budget-prior-ef-r1"
TARGETS = ("PACE", "LGEE")
DONORS = v280.DONORS
BUDGETS = (9, 12, 15)
MAX_DAYS = max(BUDGETS)
REPEATS = 100
METHODS = ("cems_support", "prior_year_local_ef", "source_loo")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    support = pd.read_csv(v280.SUPPORT)
    targets = {ba: v280.read_workbook(ba, DATA / f"{ba}.xlsx")[0] for ba in TARGETS}
    local_ef = {ba: v282.previous_year_support(ba, support)[:2] for ba in TARGETS}
    donors = {ba: v280.read_workbook(ba, ROOT / "results/v272-eia-monitor" / f"{ba}.xlsx")[0]
              for ba in DONORS}
    rng = np.random.default_rng(284)
    records = []
    for ba in TARGETS:
        for (year, month), d in sorted(targets[ba].items()):
            quarter = (month - 1) // 3 + 1
            c_low, c_high = sensor.factor_support(quarter, support)
            p_low, p_high = local_ef[ba]
            range_scores = {}
            for method, low, high in (("cems_support", c_low, c_high),
                                      ("prior_year_local_ef", p_low, p_high)):
                lo, hi = sensor.robust_hour_ranges(d, "support_minimum", low, high)
                range_scores[method] = (lo, hi, sensor.daily_scores(d, lo, hi))
            source_scores = []
            for donor in DONORS:
                source_d = donors[donor].get((year, month))
                if source_d is not None:
                    source_scores.extend(sensor.nominal_source_scores(source_d).tolist())
            if len(source_scores) < 100:
                raise RuntimeError(f"not enough donor scores for {ba} {year}-{month:02d}")
            day_ids = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
            days = np.asarray(sorted(range_scores["cems_support"][2]), dtype=np.int64)
            if len(days) <= MAX_DAYS + 5:
                raise RuntimeError(f"too few days for {ba} {year}-{month:02d}")
            for rep in range(REPEATS):
                ordered = rng.choice(days, size=MAX_DAYS, replace=False)
                for budget in BUDGETS:
                    sampled = ordered[:budget]
                    test_days = np.setdiff1d(days, sampled)
                    test_mask = np.isin(day_ids, test_days)
                    for method in METHODS:
                        if method == "source_loo":
                            q = dayrisk.radius(np.asarray(source_scores))
                            vals = sensor.score_source_baseline(d, q, test_mask)
                            set_cover = np.nan
                        else:
                            lo, hi, scores = range_scores[method]
                            q = dayrisk.radius(np.asarray([scores[int(day)] for day in sampled]))
                            vals = sensor.score_test(d, lo, hi, q, test_mask)
                            set_cover = vals[4]
                        records.append({"ba": ba, "year": year, "month": month,
                            "repeat": rep, "budget_days": budget, "method": method,
                            "hour_coverage": vals[0], "width": vals[1],
                            "winkler90": vals[2], "simultaneous_day_coverage": vals[3],
                            "ex_post_robust_day_set_containment": set_cover})
    reps = pd.DataFrame(records)
    month = reps.groupby(["ba", "month", "budget_days", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"))
    ba = month.groupby(["ba", "budget_days", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_months=("month", "count"))
    macro = ba.groupby(["budget_days", "method"], as_index=False).agg(
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
    metadata = {"protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
        "target_files": {ba: v280.sha256(DATA / f"{ba}.xlsx") for ba in TARGETS},
        "donor_files": {ba: v280.sha256(ROOT / "results/v272-eia-monitor" / f"{ba}.xlsx") for ba in DONORS},
        "support_sha256": v280.sha256(v280.SUPPORT), "budgets": list(BUDGETS),
        "repeats": REPEATS, "seed": 284, "methods": list(METHODS),
        "status": "exploratory development on already opened V280 targets"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(macro.to_string(index=False))
    print(ba.to_string(index=False))


if __name__ == "__main__":
    main()
