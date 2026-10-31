"""V129: risk-side monthly summaries with fuel-specific EF uncertainty."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

import v90_coarse_capacity_identification as v90
import v93_china_budget_coarsening as v93
from scripts.v92_sharp_coarse_capacity_identification import sharp_group_set

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V129_FACTOR_SUPPORT_RISK_SIDE_PROTOCOL.md"
OUT = ROOT / "results/v129-factor-support-risk-side-r1"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
TAUS = (600, 900)
RATES = (.01, .05, .10)
EF_BOUNDS = {
    "coal": (900, 1100), "gas": (360, 440), "oil": (720, 880), "other": (200, 600),
    "nuclear": (0, 0), "hydro": (0, 0), "wind": (0, 0), "solar": (0, 0),
}
THERMAL = ("coal", "gas", "oil", "other")
CLEAN = ("nuclear", "hydro", "wind", "solar")


def threshold_side_set(margins, high_amount, high_range, low_range, caps, tau, hours):
    """Exact set with high-side EF > tau and low-side EF <= tau."""
    thermal_total = sum((margins[f] for f in THERMAL), F(0))
    clean_total = sum((margins[f] for f in CLEAN), F(0))
    total = thermal_total + clean_total
    high_amount = F(high_amount)
    low_amount = thermal_total - high_amount
    if high_amount < 0 or low_amount < 0:
        raise ValueError("invalid threshold-side monthly margin")
    hmin, hmax = (F(value - tau) for value in high_range)
    lmin, lmax = (F(value - tau) for value in low_range)
    t = F(tau)
    if not (hmin > 0 and lmax <= 0):
        raise ValueError("EF ranges must put high group strictly above and low group at/below tau")
    if total <= 0:
        return (0,)
    out = []
    monthly_excess_min = hmin * high_amount + lmin * low_amount - t * clean_total
    monthly_excess_max = hmax * high_amount + lmax * low_amount - t * clean_total
    if monthly_excess_min <= 0:
        out.append(0)
    if monthly_excess_max > 0:
        out.append(hours)
    negative_low = -lmin
    spread = hmin - lmin
    for k in range(1, hours):
        bounds = v93.group_bounds(margins, caps, hours, k)
        xlo, xhi = bounds["thermal"]
        ylo, yhi = bounds["clean"]
        if xlo > xhi or ylo > yhi:
            continue
        base = monthly_excess_min
        rhs = -base - t * ylo
        xlo = max(xlo, rhs / (negative_low - spread))
        xhi = min(xhi, (rhs + spread * high_amount) / negative_low)
        complement_limit = total - ylo
        if xlo > xhi or xlo >= complement_limit:
            continue
        xhi = min(xhi, complement_limit)
        candidates = {xlo, xhi}
        if xlo <= high_amount <= xhi:
            candidates.add(high_amount)
        qmax = max(lmax * x + (hmax - lmax) * min(high_amount, x) - t * ylo
                   for x in candidates)
        if qmax > 0:
            out.append(k)
    return tuple(sorted(set(out)))


def robust_detailed_sets(margins, caps, tau, hours):
    low = dict(v90.EF)
    high = dict(v90.EF)
    for fuel, (lower, upper) in EF_BOUNDS.items():
        low[fuel], high[fuel] = lower, upper
    original = dict(v90.EF)
    try:
        v90.EF.clear(); v90.EF.update(low)
        low_capacity = sharp_group_set(margins, caps, tau, hours)
        low_month = v90.v66_set(margins, tau, hours)
        v90.EF.clear(); v90.EF.update(high)
        high_capacity = sharp_group_set(margins, caps, tau, hours)
        high_month = v90.v66_set(margins, tau, hours)
    finally:
        v90.EF.clear(); v90.EF.update(original)
    return (min(low_month), max(high_month)), (min(low_capacity), max(high_capacity))


def broad_coarse_set(margins, caps, tau, hours):
    """Monthly thermal/clean outer set under EF support [200,1100]."""
    thermal = sum((margins[f] for f in THERMAL), F(0))
    clean = sum((margins[f] for f in CLEAN), F(0))
    total = thermal + clean
    lo_q, hi_q, t = F(200 - tau), F(1100 - tau), F(tau)
    out = []
    if lo_q * thermal - t * clean <= 0:
        out.append(0)
    if hi_q * thermal - t * clean > 0:
        out.append(hours)
    for k in range(1, hours):
        bounds = v93.group_bounds(margins, caps, hours, k)
        xlo, xhi = bounds["thermal"]
        ylo, yhi = bounds["clean"]
        rhs = t * clean - lo_q * thermal
        vertices = v93.polygon_vertices(xlo, xhi, ylo, yhi, lo_q, t, rhs)
        if not vertices:
            continue
        maximum_event = max(hi_q * x - t * y for x, y in vertices)
        minimum_event_total = min(x + y for x, y in vertices)
        if maximum_event > 0 and minimum_event_total < total:
            out.append(k)
    return tuple(sorted(set(out)))


def decide_bounds(lo, hi, required):
    return lo >= required, hi < required, not (lo >= required or hi < required), hi - lo


def side_sets(margins, no_cap, capacity, tau, hours):
    if tau == 600:
        high = margins["coal"] + margins["oil"]
        ranges = ((720, 1100), (200, 600))
        return (threshold_side_set(margins, high, *ranges, no_cap, tau, hours),
                threshold_side_set(margins, high, *ranges, capacity, tau, hours))
    if tau == 900:
        # EF=900 makes every fuel non-event under strict >; EF=1100 gives
        # the largest possible coal-side risk set, with other thermal EF <= 880.
        high = margins["coal"]
        ranges = ((1100, 1100), (200, 880))
        no_capacity = threshold_side_set(margins, high, *ranges, no_cap, tau, hours)
        with_capacity = threshold_side_set(margins, high, *ranges, capacity, tau, hours)
        return tuple(sorted(set((0, *no_capacity)))), tuple(sorted(set((0, *with_capacity))))
    raise ValueError(f"unsupported threshold {tau}")


def make_capacity_groups(caprow):
    fine = {fuel: F(str(float(caprow.get(fuel, 0)))) for fuel in THERMAL + CLEAN}
    coarse = {group: sum((fine[fuel] for fuel in members), F(0))
              for group, members in v90.GROUPS.items()}
    return fine, coarse


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = v90.sha(PROTOCOL)
    frozen = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    cohort = {(str(row.region), str(row.month)[:10]) for row in frozen.itertuples(index=False)}
    data = v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, input_hashes = data.prepare_regions()
    capacities, capacity_hashes = {}, {}
    for year in (2023, 2024):
        capacities[year], capacity_hashes[str(year)] = v90.capacities(year)

    rows, nesting_failures, capacity_failures = [], [], []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2024-01-01") & times.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month, group in frame.groupby("month", sort=True):
            if (region, month) not in cohort:
                continue
            hours = len(group); year = int(month[:4]); vintage = year - 1
            margins = {fuel: sum((F(str(float(value))) for value in group[f"generation_{fuel}"]), F(0))
                       for fuel in v90.FUELS}
            caprow = capacities[vintage].loc[region] if region in capacities[vintage].index else pd.Series(dtype=float)
            _, caps = make_capacity_groups(caprow)
            no_cap = {"thermal": sum((margins[f] for f in THERMAL), F(0)),
                      **{fuel: margins[fuel] for fuel in CLEAN}}
            cap_categories = {"thermal": caps["thermal"], **{fuel: caps[fuel] for fuel in CLEAN}}
            energy_ok = all(sum((margins[f] for f in members), F(0)) <= hours * cap_categories[g]
                            for g, members in v90.GROUPS.items())
            if not energy_ok:
                capacity_failures.append({"region": region, "month": month, "reason": "monthly group energy exceeds lagged capacity"})
                continue
            nominal = dict(v90.EF)
            for tau in TAUS:
                actual_k = 0
                for _, obs in group.iterrows():
                    generation = {fuel: F(str(float(obs[f"generation_{fuel}"]))) for fuel in v90.FUELS}
                    total = sum(generation.values(), F(0))
                    if total > 0:
                        co2 = sum((generation[f] * nominal[f] for f in v90.FUELS), F(0))
                        actual_k += co2 > F(tau) * total
                detail_no, detail_cap = robust_detailed_sets(margins, caps, tau, hours)
                side_no, side_cap = side_sets(margins, no_cap, cap_categories, tau, hours)
                coarse_no = broad_coarse_set(margins, no_cap, tau, hours)
                coarse_cap = broad_coarse_set(margins, cap_categories, tau, hours)
                if actual_k not in range(detail_no[0], detail_no[1] + 1):
                    raise ValueError(f"nominal proxy K outside robust detailed set: {region}/{month}/tau={tau}")
                for name, endpoints in (("side_month_nominal", side_no), ("side_capacity_nominal", side_cap),
                                        ("coarse_month_nominal", coarse_no), ("coarse_capacity_nominal", coarse_cap)):
                    if not min(endpoints) <= actual_k <= max(endpoints):
                        nesting_failures.append({"region": region, "month": month, "tau": tau,
                                                 "check": name, "actual_nominal_K": actual_k,
                                                 "outer_lower": min(endpoints), "outer_upper": max(endpoints)})
                for name, inner, outer in (("side_month", detail_no, side_no),
                                           ("side_capacity", detail_cap, side_cap)):
                    if min(outer) > inner[0] or max(outer) < inner[1]:
                        nesting_failures.append({"region": region, "month": month, "tau": tau,
                                                 "check": name, "inner_lower": inner[0], "inner_upper": inner[1],
                                                 "outer_lower": min(outer), "outer_upper": max(outer)})
                for name, inner, outer in (("coarse_month", side_no, coarse_no),
                                           ("coarse_capacity", side_cap, coarse_cap)):
                    if min(outer) > min(inner) or max(outer) < max(inner):
                        nesting_failures.append({"region": region, "month": month, "tau": tau,
                                                 "check": name, "inner_lower": min(inner), "inner_upper": max(inner),
                                                 "outer_lower": min(outer), "outer_upper": max(outer)})
                for rate in RATES:
                    required = int((F(str(rate)) * hours).__ceil__())
                    record = {"region": region, "month": month, "tau": tau, "rate": rate,
                              "hours": hours, "actual_nominal_K": actual_k, "required_hours": required,
                              "robust_detailed_month_lower": detail_no[0], "robust_detailed_month_upper": detail_no[1],
                              "robust_detailed_capacity_lower": detail_cap[0], "robust_detailed_capacity_upper": detail_cap[1],
                              "risk_side_month_lower": min(side_no), "risk_side_month_upper": max(side_no),
                              "risk_side_capacity_lower": min(side_cap), "risk_side_capacity_upper": max(side_cap),
                              "coarse_month_lower": min(coarse_no), "coarse_month_upper": max(coarse_no),
                              "coarse_capacity_lower": min(coarse_cap), "coarse_capacity_upper": max(coarse_cap)}
                    sets = {"robust_detailed_month": detail_no, "risk_side_month": (min(side_no), max(side_no)),
                            "robust_detailed_capacity": detail_cap, "risk_side_capacity": (min(side_cap), max(side_cap)),
                            "coarse_month": (min(coarse_no), max(coarse_no)),
                            "coarse_capacity": (min(coarse_cap), max(coarse_cap))}
                    for name, endpoints in sets.items():
                        lo, hi = endpoints
                        pos, neg, abst, width = decide_bounds(lo, hi, required)
                        record.update({f"{name}_positive": pos, f"{name}_negative": neg,
                                       f"{name}_abstain": abst, f"{name}_width": width})
                    rows.append(record)
        print(f"region {region}", flush=True)

    OUT.mkdir(parents=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "paired_frontier.csv", index=False)
    pd.DataFrame(nesting_failures).to_csv(OUT / "nesting_failures.csv", index=False)
    pd.DataFrame(capacity_failures).to_csv(OUT / "capacity_failures.csv", index=False)
    metrics = [column for column in detail.columns if column.endswith(("_positive", "_negative", "_abstain", "_width"))]
    region_month = detail.groupby(["region", "month", "tau", "rate"])[metrics].mean().reset_index()
    region_macro = region_month.groupby(["region", "tau", "rate"])[metrics].mean().reset_index()
    macro = region_macro.groupby(["tau", "rate"])[metrics].mean().reset_index()
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    summary = {"status": "FACTOR_SUPPORT_RISK_SIDE_MONTHLY_INFO_RETROSPECTIVE_PROXY",
               "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(detail), "regions": sorted(detail.region.unique()),
               "capacity_failure_rows": len(capacity_failures), "nesting_failure_rows": len(nesting_failures),
               "factor_intervals_gco2_kwh": EF_BOUNDS, "frontier": macro.to_dict("records"),
               "capacity_sha256": capacity_hashes, "cohort_sha256": v90.sha(COHORT),
               "hourly_input_sha256": input_hashes, "source_audits": audits,
               "protocol_sha256": protocol_hash, "script_sha256": v90.sha(Path(__file__)),
               "limitations": ["Oracle coarsening of EIA hourly fuel records; not a real public Chinese field experiment.",
                               "EF intervals are stress assumptions, not empirical Chinese factor confidence sets.",
                               "The 358-month cohort was selected using retrospective capacity feasibility.",
                               "No independent Chinese hourly CIF labels or forecast-vintage inputs are evaluated."]}
    if v90.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
