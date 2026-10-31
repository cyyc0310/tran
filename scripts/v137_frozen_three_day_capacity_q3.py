"""V137: frozen V136 three-day risk-side+capacity transfer to 2026Q3."""
from __future__ import annotations

import json
import sys
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v54_unseen_local_ef_confirmation as data  # noqa: E402
import v90_coarse_capacity_identification as v90  # noqa: E402
import v129_factor_support_risk_side as v129  # noqa: E402
import v132_multiscale_risk_side_information as v132  # noqa: E402

PROTOCOL = ROOT / "docs/V137_FROZEN_THREE_DAY_CAPACITY_Q3_PROTOCOL.md"
OUT = ROOT / "results/v137-frozen-three-day-capacity-q3-r1"
NATIVE = ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"
AVAILABILITY = ROOT / "results/v113-2026q3-future-time-confirmation-r1/availability.csv"
V113_ROWS = ROOT / "results/v113-2026q3-future-time-confirmation-r1/target_rows.csv"
REGIONS = ("CPLE", "FPC", "IPCO", "PNM", "SOCO")
MONTHS = ("2026-07-01", "2026-08-01", "2026-09-01")
TAUS = (600, 900)
RATES = (.01, .05, .10)
CLEAN = ("nuclear", "hydro", "wind", "solar")


def cap_feasible(margins: dict[str, F], caps: dict[str, F], hours: int) -> bool:
    return all(sum((margins[f] for f in members), F(0)) <= hours * caps[group]
               for group, members in v90.GROUPS.items())


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if not NATIVE.exists():
        raise FileNotFoundError(NATIVE)
    protocol_hash = v90.sha(PROTOCOL)
    previous_paths = list(data.default_native_paths())
    data.default_native_paths = lambda: previous_paths + [NATIVE]
    data.STOP = pd.Timestamp("2026-10-01T00:00:00Z")
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + REGIONS))
    v90.REGIONS = REGIONS
    v90.source_data.ALL_REGIONS = data.ALL_REGIONS
    hourly, _, audits, hourly_hashes = data.prepare_regions()
    capacities, capacity_hash = v90.capacities(2025)

    availability = pd.read_csv(AVAILABILITY)
    availability = availability.loc[availability.coverage_eligible.astype(bool)].copy()
    expected_rows = {(str(r.region), str(r.month)[:10]): r for r in availability.itertuples(index=False)}
    expected_roster = {(r, m) for r in REGIONS for m in MONTHS}
    if set(expected_rows) != expected_roster:
        raise ValueError("V113 eligible roster mismatch")
    v113 = pd.read_csv(V113_ROWS).drop_duplicates(["region", "month", "tau"])

    rows: list[dict] = []
    block_audit: list[dict] = []
    failures: list[dict] = []
    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2026-07-01") & times.lt("2026-10-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        frame["day_number"] = frame.time.dt.day
        for month in MONTHS:
            month_frame = frame.loc[frame.month.eq(month)]
            hours = len(month_frame)
            expected_hours = int(expected_rows[(region, month)].valid_hours)
            if hours != expected_hours:
                failures.append({"region": region, "month": month, "check": "source_hours_match_v113",
                                 "actual": hours, "expected": expected_hours})
            caprow = capacities.loc[region] if region in capacities.index else pd.Series(dtype=float)
            _, cap_groups = v129.make_capacity_groups(caprow)
            cap_categories = {"thermal": cap_groups["thermal"], **{fuel: cap_groups[fuel] for fuel in CLEAN}}
            margins = v132.make_margins(month_frame)
            no_cap = v132.no_capacity_groups(margins)
            month_capacity_feasible = cap_feasible(margins, cap_groups, hours)
            if not month_capacity_feasible:
                failures.append({"region": region, "month": month,
                                 "check": "monthly_capacity_feasibility", "hours": hours})
            blocks = list(month_frame.groupby(((month_frame.day_number - 1) // 3), sort=True))
            block_info = []
            for block_id, block in blocks:
                block_margins = v132.make_margins(block)
                block_no_cap = v132.no_capacity_groups(block_margins)
                feasible = cap_feasible(block_margins, cap_groups, len(block))
                active_caps = cap_categories if feasible else block_no_cap
                block_info.append((block_id, block, block_margins, block_no_cap, active_caps, feasible))
                block_audit.append({"region": region, "month": month, "block_id": int(block_id),
                                    "hours": len(block), "capacity_feasible": bool(feasible),
                                    "capacity_fallback": not bool(feasible)})
            for tau in TAUS:
                actual = v132.event_count(month_frame, tau)
                monthly_caps = cap_categories if month_capacity_feasible else no_cap
                monthly_side_sets = v129.side_sets(margins, no_cap, monthly_caps, tau, hours)
                monthly_side = monthly_side_sets[1] if month_capacity_feasible else monthly_side_sets[0]
                monthly_detail = v129.robust_detailed_sets(
                    margins, cap_groups if month_capacity_feasible else no_cap, tau, hours)[1]
                monthly_side = (min(monthly_side), max(monthly_side))
                monthly_detail = (min(monthly_detail), max(monthly_detail))
                side_parts, detail_parts = [], []
                block_actual = 0
                for _, block, block_margins, block_no_cap, active_caps, feasible in block_info:
                    block_hours = len(block)
                    side_no, side_cap = v129.side_sets(block_margins, block_no_cap, active_caps,
                                                       tau, block_hours)
                    side_parts.append(v132.minmax(side_cap if feasible else side_no))
                    detail_parts.append(v129.robust_detailed_sets(block_margins, active_caps,
                                                                  tau, block_hours)[1])
                    block_actual += v132.event_count(block, tau)
                three_day_side = v132.envelope(side_parts)
                three_day_detail = v132.envelope(detail_parts)
                if block_actual != actual:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "block_actual_K_additivity", "month_K": actual,
                                     "blocks_K": block_actual})
                for name, bounds in (("monthly_side", monthly_side), ("monthly_detail", monthly_detail),
                                     ("three_day_side", three_day_side), ("three_day_detail", three_day_detail)):
                    if not bounds[0] <= actual <= bounds[1]:
                        failures.append({"region": region, "month": month, "tau": tau,
                                         "check": f"{name}_contains_nominal_K", "K": actual,
                                         "lower": bounds[0], "upper": bounds[1]})
                if three_day_side[0] > three_day_detail[0] or three_day_side[1] < three_day_detail[1]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "three_day_side_contains_detail",
                                     "side": three_day_side, "detail": three_day_detail})
                prior = v113.loc[(v113.region == region) & (v113.month.astype(str).str[:10] == month)
                                 & (v113.tau == tau)]
                if len(prior) != 1 or actual != int(prior.actual_nominal_K.iloc[0]):
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "V113_K_match", "actual": actual,
                                     "V113": int(prior.actual_nominal_K.iloc[0]) if len(prior) == 1 else None})
                for rate in RATES:
                    required = int((F(str(rate)) * hours).__ceil__())
                    record = {"region": region, "month": month, "tau": tau, "rate": rate,
                              "hours": hours, "coverage": hours / (pd.Period(month, freq="M").days_in_month * 24),
                              "blocks": len(blocks), "fallback_blocks": sum(not x[-1] for x in block_info),
                              "required_hours": required, "nominal_K": actual}
                    for name, bounds in (("monthly_detail", monthly_detail), ("monthly_side", monthly_side),
                                         ("three_day_side", three_day_side), ("three_day_detail", three_day_detail)):
                        pos, neg = bounds[0] >= required, bounds[1] < required
                        record.update({f"{name}_lower": bounds[0], f"{name}_upper": bounds[1],
                                       f"{name}_width": bounds[1] - bounds[0],
                                       f"{name}_positive": pos, f"{name}_negative": neg,
                                       f"{name}_abstain": not (pos or neg)})
                    rows.append(record)
        print(f"region {region}", flush=True)

    OUT.mkdir(parents=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "paired_frontier.csv", index=False)
    pd.DataFrame(block_audit).to_csv(OUT / "capacity_blocks.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    metrics = [c for c in detail.columns if c.endswith(("_positive", "_negative", "_abstain", "_width"))]
    region_macro = detail.groupby(["region", "tau", "rate"])[metrics].mean().reset_index()
    macro = region_macro.groupby(["tau", "rate"])[metrics].mean().reset_index()
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    b = pd.DataFrame(block_audit)
    fallback_fraction = float(b.groupby("region").capacity_fallback.mean().mean())
    summary = {"status": "FROZEN_THREE_DAY_CAPACITY_Q3_METHOD_TRANSFER_POSTHOC",
               "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(detail), "regions": list(REGIONS), "months": list(MONTHS),
               "capacity_blocks": len(b), "capacity_fallback_blocks": int(b.capacity_fallback.sum()),
               "region_equal_fallback_fraction": fallback_fraction,
               "assertion_failure_rows": len(failures), "frontier": macro.to_dict("records"),
               "factor_intervals_gco2_kwh": v129.EF_BOUNDS,
               "capacity_hash": capacity_hash, "native_q3_sha256": v90.sha(NATIVE),
               "v113_availability_sha256": v90.sha(AVAILABILITY), "v113_rows_sha256": v90.sha(V113_ROWS),
               "hourly_source_hashes": hourly_hashes, "protocol_sha256": protocol_hash,
               "script_sha256": v90.sha(Path(__file__)), "source_audits": audits,
               "limitations": ["2026Q3 outcomes were previously opened for other TransCIF methods; not blind confirmation.",
                               "Only five regions and three months; small transfer probe.",
                               "EIA capacity and hourly fixed-factor proxies; no Chinese data or actual CIF labels."]}
    if v90.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print(f"capacity fallback fraction: {fallback_fraction:.4f}; assertion failures: {len(failures)}")


if __name__ == "__main__":
    main()
