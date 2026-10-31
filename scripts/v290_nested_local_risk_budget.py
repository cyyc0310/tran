"""Exploratory nested local-risk calibration budget on the V286 cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v224_eia_sparse_daily_maxrisk as dayrisk
import v279_asymmetric_ef_sensor_frontier as sensor
import v286_prior_year_ef_temporal_holdout as v286
import v289_lagged_current_ef_update as v289

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V290_NESTED_LOCAL_RISK_BUDGET_PROTOCOL.md"
OUT = ROOT / "results/v290-nested-local-risk-budget-r2"
BUDGETS = (9, 12, 15, 18)
METHODS = ("cems_support", "prior_year_local_ef", "lagged_current_year_ef", "source_loo")
REPEATS = 100
BASE_SEED = 286
EXTENSION_SEED = 290


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha(PROTOCOL)
    support = pd.read_csv(v286.SUPPORT)
    targets, prior, month_factors, prepared = {}, {}, {}, {}
    for ba in v286.TARGETS:
        targets[ba], prior[ba], _ = v286.read_ba(ba, v286.DATA / f"{ba}.xlsx", target=True)
        month_factors[ba] = v289.read_monthly_factor_history(ba)

    donors = {}
    for ba in v286.DONORS:
        donors[ba], _, _ = v286.read_ba(ba, v286.DONOR_DIR / f"{ba}.xlsx", target=False)

    for ba in v286.TARGETS:
        for (year, month), d in targets[ba].items():
            quarter = (month - 1) // 3 + 1
            c_low, c_high = sensor.factor_support(quarter, support)
            p_low, p_high, _ = v286.local_prior_bounds(ba, year, month, prior[ba], support)
            l_low, l_high = v289.update_bounds(
                ba, year, month, c_low, c_high, prior[ba], month_factors[ba]
            )
            ranges = {}
            for method, loef, hief in (
                ("cems_support", c_low, c_high),
                ("prior_year_local_ef", p_low, p_high),
                ("lagged_current_year_ef", l_low, l_high),
            ):
                lo, hi = sensor.robust_hour_ranges(d, "support_minimum", loef, hief)
                ranges[method] = (lo, hi, sensor.daily_scores(d, lo, hi))
            source = []
            for donor in v286.DONORS:
                sd = donors[donor].get((year, month))
                if sd is not None:
                    source.extend(sensor.nominal_source_scores(sd).tolist())
            if len(source) < 100:
                raise RuntimeError(f"insufficient donor scores: {ba}/{year}-{month}")
            prepared[(ba, year, month)] = {"d": d, "ranges": ranges, "source": np.asarray(source, float)}

    expected = {(ba, y, m) for ba in v286.TARGETS for y in v286.TARGET_YEARS for m in range(1, 13)}
    if set(prepared) != expected:
        raise RuntimeError(f"target roster incomplete: {sorted(expected - set(prepared))}")

    # Reproduce V289's exact first-nine-day sample stream before generating extensions.
    base_rng = np.random.default_rng(BASE_SEED)
    extension_rng = np.random.default_rng(EXTENSION_SEED)
    records = []
    for key, item in sorted(prepared.items()):
        ba, year, month = key
        d = item["d"]
        did = dayrisk.day_ids(pd.DatetimeIndex(d["time"]))
        days = np.asarray(sorted(item["ranges"]["cems_support"][2]), dtype=np.int64)
        for rep in range(REPEATS):
            base9 = base_rng.choice(days, size=9, replace=False)
            remaining = np.setdiff1d(days, base9)
            extra = extension_rng.choice(remaining, size=9, replace=False)
            for budget in BUDGETS:
                cal = np.concatenate((base9, extra[: budget - 9]))
                test_days = np.setdiff1d(days, cal)
                mask = np.isin(did, test_days)
                for method in METHODS:
                    if method == "source_loo":
                        q = dayrisk.radius(item["source"])
                        value = sensor.score_source_baseline(d, q, mask)
                        setcov = np.nan
                    else:
                        _, _, scores = item["ranges"][method]
                        q = dayrisk.radius(np.asarray([scores[int(day)] for day in cal]))
                        lo, hi, _ = item["ranges"][method]
                        value = sensor.score_test(d, lo, hi, q, mask)
                        setcov = value[4]
                    records.append(
                        {
                            "ba": ba,
                            "year": year,
                            "month": month,
                            "repeat": rep,
                            "calibration_days": budget,
                            "method": method,
                            "hour_coverage": value[0],
                            "width": value[1],
                            "winkler90": value[2],
                            "simultaneous_day_coverage": value[3],
                            "ex_post_robust_day_set_containment": setcov,
                        }
                    )

    reps = pd.DataFrame(records)
    month = reps.groupby(["calibration_days", "ba", "year", "month", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"),
        width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
    )
    year = month.groupby(["calibration_days", "ba", "year", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"),
        width=("width", "mean"),
        winkler90=("winkler90", "mean"),
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_months=("month", "count"),
    )
    macro = year.groupby(["calibration_days", "method"], as_index=False).agg(
        macro_hour_coverage=("hour_coverage", "mean"),
        macro_width=("width", "mean"),
        macro_winkler90=("winkler90", "mean"),
        macro_simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        macro_ex_post_robust_day_set_containment=("ex_post_robust_day_set_containment", "mean"),
        n_ba_years=("year", "count"),
    )
    month_counts = month.groupby(["calibration_days", "method"]).agg(
        months_below90=("simultaneous_day_coverage", lambda x: int((x < 0.9).sum())),
        min_month_coverage=("simultaneous_day_coverage", "min"),
    ).reset_index()
    macro = macro.merge(month_counts, on=["calibration_days", "method"], how="left")

    # Structural check: V289's 9-day branch is identical because its RNG stream is replayed.
    old = pd.read_csv(ROOT / "results/v289-lagged-current-ef-update-r1/replicate_metrics.csv")
    old9 = old[old.method.isin(METHODS)].copy()
    new9 = reps[reps.calibration_days == 9].copy()
    join = ["ba", "year", "month", "repeat", "method"]
    check = old9.merge(new9, on=join, suffixes=("_old", "_new"), validate="one_to_one")
    for col in ("hour_coverage", "width", "winkler90", "simultaneous_day_coverage"):
        delta = np.nanmax(np.abs(check[f"{col}_old"].to_numpy() - check[f"{col}_new"].to_numpy()))
        if delta > 1e-10:
            raise RuntimeError(f"V289 nine-day reproduction failed for {col}: {delta}")

    OUT.mkdir(parents=True)
    reps.to_csv(OUT / "replicate_metrics.csv", index=False)
    month.to_csv(OUT / "ba_month_metrics.csv", index=False)
    year.to_csv(OUT / "ba_year_metrics.csv", index=False)
    macro.to_csv(OUT / "macro_metrics.csv", index=False)
    metadata = {
        "protocol_sha256": protocol_hash,
        "script_sha256": sha(Path(__file__)),
        "v289_results_sha256": sha(ROOT / "results/v289-lagged-current-ef-update-r1/replicate_metrics.csv"),
        "base_seed": BASE_SEED,
        "extension_seed": EXTENSION_SEED,
        "repeats": REPEATS,
        "budgets": list(BUDGETS),
        "methods": list(METHODS),
        "nine_day_max_abs_diff": {
            col: float(np.nanmax(np.abs(check[f"{col}_old"].to_numpy() - check[f"{col}_new"].to_numpy())))
            for col in ("hour_coverage", "width", "winkler90", "simultaneous_day_coverage")
        },
        "status": "exploratory post-hoc repeated temporal diagnosis; not China deployment or geographic confirmation",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(macro.to_string(index=False))
    print(year.to_string(index=False))


if __name__ == "__main__":
    main()
