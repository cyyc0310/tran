"""V136: combine frozen three-day risk-side information with coarse capacity."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

import v90_coarse_capacity_identification as v90
import v129_factor_support_risk_side as v129
import v132_multiscale_risk_side_information as v132

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V136_THREE_DAY_CAPACITY_COMBINATION_PROTOCOL.md"
OUT = ROOT / "results/v136-three-day-capacity-combination-r1"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
V129_ROWS = ROOT / "results/v129-factor-support-risk-side-r1/paired_frontier.csv"
TAUS = (600, 900)
RATES = (.01, .05, .10)
THERMAL = ("coal", "gas", "oil", "other")
CLEAN = ("nuclear", "hydro", "wind", "solar")


def event_count(frame: pd.DataFrame, tau: int) -> int:
    return v132.event_count(frame, tau)


def cap_feasible(margins: dict[str, F], caps: dict[str, F], hours: int) -> bool:
    return all(sum((margins[f] for f in members), F(0)) <= hours * caps[group]
               for group, members in v90.GROUPS.items())


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = v90.sha(PROTOCOL)
    cohort = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    eligible = {(str(r.region), str(r.month)[:10]) for r in cohort.itertuples(index=False)}
    prior = pd.read_csv(V129_ROWS)
    data = v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, hourly_hashes = data.prepare_regions()
    capacities, capacity_hashes = {}, {}
    for year in (2023, 2024):
        capacities[year], capacity_hashes[str(year)] = v90.capacities(year)

    rows: list[dict] = []
    capacity_audit: list[dict] = []
    failures: list[dict] = []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2024-01-01") & times.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        frame["day_number"] = frame.time.dt.day
        for month, month_frame in frame.groupby("month", sort=True):
            if (region, month) not in eligible:
                continue
            hours = len(month_frame)
            year = int(month[:4])
            vintage = year - 1
            capacity_row = capacities[vintage].loc[region] if region in capacities[vintage].index else pd.Series(dtype=float)
            _, coarse_caps = v129.make_capacity_groups(capacity_row)
            cap_categories = {"thermal": coarse_caps["thermal"], **{fuel: coarse_caps[fuel] for fuel in CLEAN}}
            month_margins = v132.make_margins(month_frame)
            no_cap_month = v132.no_capacity_groups(month_margins)
            blocks = list(month_frame.groupby(((month_frame.day_number - 1) // 3), sort=True))
            month_capacity_failures = 0
            block_records = []
            for block_id, block in blocks:
                block_margins = v132.make_margins(block)
                no_cap = v132.no_capacity_groups(block_margins)
                block_hours = len(block)
                feasible = cap_feasible(block_margins, coarse_caps, block_hours)
                active_caps = cap_categories if feasible else no_cap
                if not feasible:
                    month_capacity_failures += 1
                block_records.append((block_id, block, block_margins, no_cap, active_caps, feasible))
            capacity_audit.append({"region": region, "month": month, "capacity_vintage": vintage,
                                   "blocks": len(blocks), "capacity_fallback_blocks": month_capacity_failures,
                                   "fallback_fraction": month_capacity_failures / len(blocks) if blocks else 0.0})

            for tau in TAUS:
                actual = event_count(month_frame, tau)
                # Same month-level EIA technology capacity, used as a direct baseline.
                prior_row = prior[(prior.region == region) & (prior.month.astype(str).str[:10] == month)
                                  & (prior.tau == tau) & (prior.rate == .01)]
                if len(prior_row) != 1:
                    raise ValueError(f"missing V129 baseline {region}/{month}/tau={tau}")
                monthly_detail = (int(prior_row.robust_detailed_capacity_lower.iloc[0]),
                                  int(prior_row.robust_detailed_capacity_upper.iloc[0]))
                monthly_side = (int(prior_row.risk_side_capacity_lower.iloc[0]),
                                int(prior_row.risk_side_capacity_upper.iloc[0]))
                no_cap_side_parts, side_parts, detail_parts = [], [], []
                for _, block, margins, no_cap, active_caps, feasible in block_records:
                    block_hours = len(block)
                    no_cap_side = v132.fast_risk_side_set(margins, tau, block_hours)
                    no_cap_side_parts.append(v132.minmax(no_cap_side))
                    side_no, side_cap = v129.side_sets(margins, no_cap, active_caps, tau, block_hours)
                    side_parts.append(v132.minmax(side_cap if feasible else side_no))
                    detail_parts.append(v129.robust_detailed_sets(margins, active_caps, tau, block_hours)[1])
                no_cap_daily = v132.envelope(no_cap_side_parts)
                daily_side = v132.envelope(side_parts)
                daily_detail = v132.envelope(detail_parts)
                for name, bounds in (("monthly_detail", monthly_detail), ("monthly_side", monthly_side),
                                     ("three_day_no_capacity", no_cap_daily),
                                     ("three_day_capacity", daily_side), ("three_day_detail", daily_detail)):
                    if not bounds[0] <= actual <= bounds[1]:
                        failures.append({"region": region, "month": month, "tau": tau,
                                         "check": f"{name}_contains_nominal_K", "K": actual,
                                         "lower": bounds[0], "upper": bounds[1]})
                if daily_side[0] > daily_detail[0] or daily_side[1] < daily_detail[1]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "three_day_capacity_contains_detail",
                                     "side": daily_side, "detail": daily_detail})
                for rate in RATES:
                    required = int((F(str(rate)) * hours).__ceil__())
                    row = {"region": region, "month": month, "tau": tau, "rate": rate,
                           "hours": hours, "capacity_vintage": vintage, "blocks": len(blocks),
                           "capacity_fallback_blocks": month_capacity_failures,
                           "capacity_fallback_fraction": month_capacity_failures / len(blocks),
                           "nominal_K": actual, "required_hours": required}
                    for name, bounds in (("monthly_detail", monthly_detail), ("monthly_side", monthly_side),
                                         ("three_day_no_capacity", no_cap_daily),
                                         ("three_day_capacity", daily_side), ("three_day_detail", daily_detail)):
                        pos, neg = bounds[0] >= required, bounds[1] < required
                        row.update({f"{name}_lower": bounds[0], f"{name}_upper": bounds[1],
                                    f"{name}_width": bounds[1] - bounds[0],
                                    f"{name}_positive": pos, f"{name}_negative": neg,
                                    f"{name}_abstain": not (pos or neg)})
                    rows.append(row)
        print(f"region {region}", flush=True)

    OUT.mkdir(parents=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "paired_frontier.csv", index=False)
    pd.DataFrame(capacity_audit).to_csv(OUT / "capacity_availability.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    metrics = [c for c in detail.columns if c.endswith(("_positive", "_negative", "_abstain", "_width"))]
    region_macro = detail.groupby(["region", "tau", "rate"])[metrics].mean().reset_index()
    macro = region_macro.groupby(["tau", "rate"])[metrics].mean().reset_index()
    fallback_macro = pd.DataFrame(capacity_audit).groupby("region")["fallback_fraction"].mean().mean()
    macro["mean_region_capacity_fallback_fraction"] = float(fallback_macro)
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    summary = {"status": "THREE_DAY_RISK_SIDE_PLUS_COARSE_CAPACITY_RETROSPECTIVE_PROXY",
               "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(detail), "regions": sorted(detail.region.unique()),
               "assertion_failure_rows": len(failures), "mean_region_capacity_fallback_fraction": float(fallback_macro),
               "frontier": macro.to_dict("records"), "capacity_hashes": capacity_hashes,
               "hourly_input_sha256": hourly_hashes, "cohort_sha256": v90.sha(COHORT),
               "v129_rows_sha256": v90.sha(V129_ROWS), "protocol_sha256": protocol_hash,
               "script_sha256": v90.sha(Path(__file__)), "source_audits": audits,
               "limitations": ["Coarse capacities are EIA-860 proxies, not validated Chinese provincial capacities.",
                               "Capacity-infeasible three-day blocks fall back to no-capacity bounds.",
                               "EF intervals are stress assumptions; labels are fixed-factor EIA proxies.",
                               "No Chinese hourly CIF validation or actual reporting-time audit."]}
    if v90.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print(f"assertion failures: {len(failures)}")


if __name__ == "__main__":
    main()
