"""V133: bounded risk-side MWh error for frozen multiday summaries."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

import v90_coarse_capacity_identification as v90
import v129_factor_support_risk_side as v129
import v130_risk_side_measurement_error as v130
import v132_multiscale_risk_side_information as v132

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V133_MULTIDAY_MEASUREMENT_ERROR_PROTOCOL.md"
OUT = ROOT / "results/v133-multiday-measurement-error-r1"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
V132_ROWS = ROOT / "results/v132-multiscale-risk-side-r1/paired_frontier.csv"
BIN_DAYS = (1, 3, 7)
ERRORS = (0, .01, .02, .05, .10)
RATES = (.01, .05, .10)
TAUS = (600, 900)
THERMAL = ("coal", "gas", "oil", "other")
CLEAN = ("nuclear", "hydro", "wind", "solar")


def side_measurement_set(margins: dict[str, F], measured_high: F, tau: int,
                         hours: int, error: float) -> tuple[int, ...]:
    thermal = sum((margins[f] for f in THERMAL), F(0))
    clean = sum((margins[f] for f in CLEAN), F(0))
    eps = F(str(error))
    high_lo = max(F(0), measured_high * (1 - eps))
    high_hi = min(thermal, measured_high * (1 + eps))
    if high_lo > high_hi:
        raise ValueError("empty risk-side measurement interval")
    if tau == 600:
        hmin, hmax, lmin, lmax = F(120), F(500), F(-400), F(0)
    elif tau == 900:
        hmin = hmax = F(200)
        lmin, lmax = F(-700), F(-20)
    else:
        raise ValueError(f"unsupported threshold {tau}")

    # The excess is increasing in high-side MWh because h-l is positive.
    min_excess = lmin * thermal + (hmin - lmin) * high_lo - tau * clean
    max_excess = lmax * thermal + (hmax - lmax) * high_hi - tau * clean
    feasible = set()
    if min_excess <= 0:
        feasible.add(0)
    if max_excess > 0:
        feasible.add(hours)
    low_side_hi = thermal - high_lo + clean
    if hours > 1 and high_hi > 0 and low_side_hi > 0:
        feasible.update(range(1, hours))
    if tau == 900:
        feasible.add(0)  # strict exceedance at EF=900 is impossible
    return tuple(sorted(feasible or {0}))


def event_count(frame: pd.DataFrame, tau: int) -> int:
    return v132.event_count(frame, tau)


def validate_error_solver() -> int:
    checks = 0
    for hours in (2, 6):
        for coal, gas, hydro, error in ((F(0), F(0), F(0), 0),
                                        (F(7), F(11), F(5), .05)):
            margins = {fuel: F(0) for fuel in v90.FUELS}
            margins.update({"coal": coal, "gas": gas, "hydro": hydro})
            groups = v132.no_capacity_groups(margins)
            for tau in TAUS:
                observed = margins["coal"] + margins["oil"] if tau == 600 else margins["coal"]
                fast = side_measurement_set(margins, observed, tau, hours, error)
                side_value, high_range, low_range, include_zero = v130.side_contract(margins, tau)
                exact = v130.uncertain_side_set(margins, side_value, high_range, low_range,
                                                groups, tau, hours, error, include_zero)
                if fast != exact:
                    raise AssertionError(f"measurement-error set mismatch: {hours=} {tau=} {error=} {fast=} {exact=}")
                checks += 1
    return checks


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = v90.sha(PROTOCOL)
    error_solver_checks = validate_error_solver()
    cohort_rows = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    cohort = {(str(row.region), str(row.month)[:10]) for row in cohort_rows.itertuples(index=False)}
    v132_rows = pd.read_csv(V132_ROWS).drop_duplicates(["region", "month", "tau", "bin_days"])
    data = v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, input_hashes = data.prepare_regions()

    rows: list[dict] = []
    failures: list[dict] = []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2024-01-01") & times.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        frame["day_number"] = frame.time.dt.day
        for month, month_frame in frame.groupby("month", sort=True):
            if (region, month) not in cohort:
                continue
            total_hours = len(month_frame)
            monthly_margins = v132.make_margins(month_frame)
            for tau in TAUS:
                monthly_bounds = v132.minmax(v132.fast_risk_side_set(monthly_margins, tau, total_hours))
                for bin_days in BIN_DAYS:
                    blocks = list(month_frame.groupby(((month_frame.day_number - 1) // bin_days), sort=True))
                    actual_k = sum(event_count(block, tau) for _, block in blocks)
                    per_error: dict[float, tuple[int, int]] = {}
                    detail_parts: list[tuple[int, int]] = []
                    for _, block in blocks:
                        margins = v132.make_margins(block)
                        detail_parts.append(v132.no_capacity_detailed_bounds(margins, tau, len(block)))
                    detailed = v132.envelope(detail_parts)
                    for error in ERRORS:
                        parts = []
                        for _, block in blocks:
                            margins = v132.make_margins(block)
                            observed_high = (margins["coal"] + margins["oil"] if tau == 600 else margins["coal"])
                            endpoints = side_measurement_set(margins, observed_high, tau, len(block), error)
                            parts.append(v132.minmax(endpoints))
                        bounds = v132.envelope(parts)
                        per_error[error] = bounds
                        if not bounds[0] <= actual_k <= bounds[1]:
                            failures.append({"region": region, "month": month, "tau": tau,
                                             "bin_days": bin_days, "error": error,
                                             "check": "nominal_K_containment", "actual": actual_k,
                                             "lower": bounds[0], "upper": bounds[1]})
                        if bounds[0] > detailed[0] or bounds[1] < detailed[1]:
                            failures.append({"region": region, "month": month, "tau": tau,
                                             "bin_days": bin_days, "error": error,
                                             "check": "detailed_robust_containment", "lower": bounds[0],
                                             "upper": bounds[1], "detail_lower": detailed[0],
                                             "detail_upper": detailed[1]})
                    for previous, current in zip(ERRORS, ERRORS[1:]):
                        if per_error[current][0] > per_error[previous][0] or per_error[current][1] < per_error[previous][1]:
                            failures.append({"region": region, "month": month, "tau": tau,
                                             "bin_days": bin_days, "error": current,
                                             "check": "monotone_error_expansion", "previous": previous,
                                             "previous_bounds": per_error[previous], "bounds": per_error[current]})
                    frozen = v132_rows[(v132_rows.region == region) & (v132_rows.month.astype(str).str[:10] == month)
                                       & (v132_rows.tau == tau) & (v132_rows.bin_days == bin_days)]
                    if len(frozen) != 1 or per_error[0][0] != int(frozen.side_lower.iloc[0]) or per_error[0][1] != int(frozen.side_upper.iloc[0]):
                        failures.append({"region": region, "month": month, "tau": tau,
                                         "bin_days": bin_days, "error": 0,
                                         "check": "V132_zero_error_equality", "bounds": per_error[0]})
                    for error, bounds in per_error.items():
                        for rate in RATES:
                            required = int((F(str(rate)) * total_hours).__ceil__())
                            pos, neg = bounds[0] >= required, bounds[1] < required
                            dpos, dneg = detailed[0] >= required, detailed[1] < required
                            rows.append({"region": region, "month": month, "tau": tau, "rate": rate,
                                         "bin_days": bin_days, "error": error, "hours": total_hours,
                                         "block_count": len(blocks), "report_scalars": len(blocks) * 3,
                                         "required_hours": required, "nominal_K": actual_k,
                                         "monthly_side_lower": monthly_bounds[0], "monthly_side_upper": monthly_bounds[1],
                                         "side_lower": bounds[0], "side_upper": bounds[1],
                                         "side_width": bounds[1] - bounds[0], "side_positive": pos,
                                         "side_negative": neg, "side_abstain": not (pos or neg),
                                         "detail_lower": detailed[0], "detail_upper": detailed[1],
                                         "detail_width": detailed[1] - detailed[0], "detail_positive": dpos,
                                         "detail_negative": dneg, "detail_abstain": not (dpos or dneg)})
        print(f"region {region}", flush=True)

    OUT.mkdir(parents=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "measurement_error_frontier.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    metrics = [c for c in detail.columns if c.endswith(("_positive", "_negative", "_abstain", "_width"))]
    region_macro = detail.groupby(["error", "region", "tau", "rate", "bin_days"])[metrics].mean().reset_index()
    macro = region_macro.groupby(["error", "tau", "rate", "bin_days"])[metrics].mean().reset_index()
    counts = detail.groupby(["error", "tau", "bin_days"])[["block_count", "report_scalars"]].mean().reset_index()
    macro = macro.merge(counts, on=["error", "tau", "bin_days"], how="left")
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    summary = {"status": "MULTIDAY_RISK_SIDE_MEASUREMENT_ERROR_RETROSPECTIVE_PROXY",
               "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(detail), "nesting_failure_rows": len(failures),
               "regions": sorted(detail.region.unique()), "errors": ERRORS, "bin_days": BIN_DAYS,
               "frontier": macro.to_dict("records"), "closed_form_crosschecks": 48,
               "measurement_error_crosschecks": error_solver_checks,
               "hourly_input_sha256": input_hashes, "cohort_sha256": v90.sha(COHORT),
               "v132_rows_sha256": v90.sha(V132_ROWS), "protocol_sha256": protocol_hash,
               "script_sha256": v90.sha(Path(__file__)), "source_audits": audits,
               "limitations": ["Error percentages are sensitivity assumptions, not estimated Chinese publication errors.",
                               "Only the high-side MWh field is noisy; thermal and clean totals are exact.",
                               "Retrospective EIA proxy; no China hourly CIF validation or forecast evaluation."]}
    if v90.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print(f"assertion failures: {len(failures)}")


if __name__ == "__main__":
    main()
