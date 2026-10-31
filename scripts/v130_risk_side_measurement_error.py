"""V130: stress V129 risk-side MWh with bounded measurement error."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

import v90_coarse_capacity_identification as v90
import v93_china_budget_coarsening as v93
from scripts import v129_factor_support_risk_side as v129

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V130_RISK_SIDE_MEASUREMENT_ERROR_PROTOCOL.md"
OUT = ROOT / "results/v130-risk-side-measurement-error-r2"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
V129_ROWS = ROOT / "results/v129-factor-support-risk-side-r1/paired_frontier.csv"
ERRORS = (0, .01, .02, .05, .10)


def uncertain_side_set(margins, observed_high, high_range, low_range, caps, tau, hours, error,
                       include_zero=False):
    thermal = sum((margins[f] for f in v129.THERMAL), F(0))
    clean = sum((margins[f] for f in v129.CLEAN), F(0))
    observed = F(observed_high); eps = F(str(error))
    lower = max(F(0), observed * (1 - eps))
    upper = min(thermal, observed * (1 + eps))
    high_min = F(high_range[0] - tau); low_min = F(low_range[0] - tau)
    high_max = F(high_range[1] - tau); low_max = F(low_range[1] - tau)
    t = F(tau)
    if lower > upper:
        raise ValueError("empty risk-side measurement interval")

    result = set(v129.threshold_side_set(margins, lower, high_range, low_range, caps, tau, hours))
    total = thermal + clean
    # When event high-side energy exceeds the minimum true monthly total,
    # choose the latent total H=z; the complement then has no high-side energy.
    for k in range(1, hours):
        bounds = v93.group_bounds(margins, caps, hours, k)
        event_lo, event_hi = bounds["thermal"]
        clean_lo, clean_hi = bounds["clean"]
        event_lo = max(event_lo, lower)
        complement_limit = total - clean_lo
        if event_lo > event_hi or event_lo >= complement_limit:
            continue
        event_hi = min(event_hi, complement_limit)
        candidates = {event_lo, event_hi}
        if event_lo <= upper <= event_hi:
            candidates.add(upper)
        max_excess = max(low_max * x + (high_max - low_max) * min(upper, x) - t * clean_lo
                         for x in candidates)
        if max_excess > 0:
            result.add(k)
    if high_max * upper + low_max * (thermal - upper) - t * clean > 0:
        result.add(hours)
    if include_zero:
        result.add(0)
    return tuple(sorted(result))


def side_contract(margins, tau):
    if tau == 600:
        return margins["coal"] + margins["oil"], (720, 1100), (200, 600), False
    return margins["coal"], (1100, 1100), (200, 880), True


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = v90.sha(PROTOCOL)
    frozen = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    cohort = {(str(row.region), str(row.month)[:10]) for row in frozen.itertuples(index=False)}
    v129_rows = pd.read_csv(V129_ROWS)
    data = v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, input_hashes = data.prepare_regions()
    capacities, capacity_hashes = {}, {}
    for year in (2023, 2024):
        capacities[year], capacity_hashes[str(year)] = v90.capacities(year)

    rows, nesting_failures = [], []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2024-01-01") & times.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month, group in frame.groupby("month", sort=True):
            if (region, month) not in cohort:
                continue
            hours = len(group); vintage = int(month[:4]) - 1
            margins = {fuel: sum((F(str(float(value))) for value in group[f"generation_{fuel}"]), F(0))
                       for fuel in v90.FUELS}
            caprow = capacities[vintage].loc[region] if region in capacities[vintage].index else pd.Series(dtype=float)
            _, caps = v129.make_capacity_groups(caprow)
            cap_categories = {"thermal": caps["thermal"], **{fuel: caps[fuel] for fuel in v129.CLEAN}}
            if not all(sum((margins[f] for f in members), F(0)) <= hours * cap_categories[g]
                       for g, members in v90.GROUPS.items()):
                continue
            for tau in v129.TAUS:
                side_value, high_range, low_range, include_zero = side_contract(margins, tau)
                _, detailed = v129.robust_detailed_sets(margins, caps, tau, hours)
                coarse = v129.broad_coarse_set(margins, cap_categories, tau, hours)
                actual = 0
                for _, obs in group.iterrows():
                    hourly_values = {fuel: F(str(float(obs[f"generation_{fuel}"]))) for fuel in v90.FUELS}
                    total = sum(hourly_values.values(), F(0))
                    if total > 0:
                        carbon = sum((hourly_values[f] * v90.EF[f] for f in v90.FUELS), F(0))
                        actual += carbon > F(tau) * total
                v129_row = v129_rows[(v129_rows.region.eq(region)) & v129_rows.month.str[:10].eq(month)
                                     & v129_rows.tau.eq(tau) & v129_rows.rate.eq(.01)]
                if len(v129_row) != 1:
                    raise ValueError(f"missing V129 frozen row {region}/{month}/tau={tau}")
                frozen_lower = int(v129_row.risk_side_capacity_lower.iloc[0])
                frozen_upper = int(v129_row.risk_side_capacity_upper.iloc[0])
                last_set = None
                for error in ERRORS:
                    side = uncertain_side_set(margins, side_value, high_range, low_range,
                                              cap_categories, tau, hours, error, include_zero)
                    lower, upper = min(side), max(side)
                    if actual < lower or actual > upper:
                        nesting_failures.append({"region": region, "month": month, "tau": tau,
                                                 "error": error, "check": "nominal K containment",
                                                 "lower": lower, "upper": upper, "actual": actual})
                    if lower > detailed[0] or upper < detailed[1]:
                        nesting_failures.append({"region": region, "month": month, "tau": tau,
                                                 "error": error, "check": "detailed robust containment",
                                                 "lower": lower, "upper": upper,
                                                 "detailed_lower": detailed[0], "detailed_upper": detailed[1]})
                    if lower < min(coarse) or upper > max(coarse):
                        nesting_failures.append({"region": region, "month": month, "tau": tau,
                                                 "error": error, "check": "coarse outer nesting",
                                                 "lower": lower, "upper": upper,
                                                 "coarse_lower": min(coarse), "coarse_upper": max(coarse)})
                    if error == 0 and (lower, upper) != (frozen_lower, frozen_upper):
                        nesting_failures.append({"region": region, "month": month, "tau": tau,
                                                 "error": error, "check": "zero-error V129 equality",
                                                 "lower": lower, "upper": upper,
                                                 "v129_lower": frozen_lower, "v129_upper": frozen_upper})
                    if last_set is not None and (lower > last_set[0] or upper < last_set[1]):
                        nesting_failures.append({"region": region, "month": month, "tau": tau,
                                                 "error": error, "check": "monotone expansion",
                                                 "lower": lower, "upper": upper,
                                                 "previous_lower": last_set[0], "previous_upper": last_set[1]})
                    last_set = (lower, upper)
                    for rate in v129.RATES:
                        required = int((F(str(rate)) * hours).__ceil__())
                        positive, negative, abstain, width = v129.decide_bounds(lower, upper, required)
                        dpos, dneg, dabst, dwidth = v129.decide_bounds(detailed[0], detailed[1], required)
                        rows.append({"region": region, "month": month, "tau": tau, "rate": rate,
                                     "relative_error": error, "hours": hours, "actual_nominal_K": actual,
                                     "side_lower": lower, "side_upper": upper,
                                     "side_positive": positive, "side_negative": negative,
                                     "side_abstain": abstain, "side_width": width,
                                     "detailed_lower": detailed[0], "detailed_upper": detailed[1],
                                     "detailed_positive": dpos, "detailed_negative": dneg,
                                     "detailed_abstain": dabst, "detailed_width": dwidth,
                                     "coarse_lower": min(coarse), "coarse_upper": max(coarse)})
        print(f"region {region}", flush=True)

    OUT.mkdir(parents=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "measurement_error_frontier.csv", index=False)
    pd.DataFrame(nesting_failures, columns=["region", "month", "tau", "error", "check", "lower", "upper",
                                           "actual", "detailed_lower", "detailed_upper", "coarse_lower", "coarse_upper",
                                           "previous_lower", "previous_upper", "v129_lower", "v129_upper"]).to_csv(
        OUT / "assertion_failures.csv", index=False)
    metrics = [c for c in detail.columns if c.endswith(("_positive", "_negative", "_abstain", "_width"))]
    rm = detail.groupby(["relative_error", "region", "tau", "rate", "month"])[metrics].mean().reset_index()
    region_macro = rm.groupby(["relative_error", "region", "tau", "rate"])[metrics].mean().reset_index()
    macro = region_macro.groupby(["relative_error", "tau", "rate"])[metrics].mean().reset_index()
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    summary = {"status": "RISK_SIDE_MWH_MEASUREMENT_ERROR_WITH_EF_SUPPORT_RETROSPECTIVE_PROXY",
               "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(detail), "nesting_failure_rows": len(nesting_failures),
               "errors": list(ERRORS), "frontier": macro.to_dict("records"),
               "capacity_hashes": capacity_hashes, "hourly_source_hashes": input_hashes,
               "v129_rows_sha256": v90.sha(V129_ROWS), "cohort_sha256": v90.sha(COHORT),
               "protocol_sha256": protocol_hash, "script_sha256": v90.sha(Path(__file__)),
               "source_audits": audits,
               "limitations": ["Measurement errors are sensitivity assumptions, not estimated Chinese publication errors.",
                               "Thermal total and coarse capacities are treated as exact.",
                               "Oracle EIA coarsening, retrospective capacity-screened cohort, no China hourly validation."]}
    if v90.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
