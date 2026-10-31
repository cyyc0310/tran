"""V135: preserve shared EF uncertainty across multiday risk-side bounds."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

import v90_coarse_capacity_identification as v90
import v129_factor_support_risk_side as v129
import v132_multiscale_risk_side_information as v132

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V135_COHERENT_FACTOR_MULTIDAY_PROTOCOL.md"
OUT = ROOT / "results/v135-coherent-factor-multiday-r1"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
V132_ROWS = ROOT / "results/v132-multiscale-risk-side-r1/paired_frontier.csv"
TAUS = (600, 900)
RATES = (.01, .05, .10)
HIGH_SIDE = {600: ("coal", "oil"), 900: ("coal",)}
LOW_SIDE = {600: ("gas", "other"), 900: ("gas", "oil", "other")}


def endpoint_ef(which: str) -> dict[str, int]:
    index = 0 if which == "low" else 1
    return {fuel: bounds[index] for fuel, bounds in v129.EF_BOUNDS.items()}


def side_support(tau: int, which: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    ef = endpoint_ef(which)
    high_values = [ef[fuel] for fuel in HIGH_SIDE[tau]]
    low_values = [ef[fuel] for fuel in LOW_SIDE[tau]]
    if tau == 900 and which == "low":
        return None  # all fuels are at or below the strict threshold
    return (min(high_values), max(high_values)), (min(low_values), max(low_values))


def side_set_at_endpoint(margins: dict[str, F], tau: int, hours: int,
                         which: str) -> tuple[int, ...]:
    supports = side_support(tau, which)
    if supports is None:
        return (0,)
    high_range, low_range = supports
    high = sum((margins[f] for f in HIGH_SIDE[tau]), F(0))
    low = sum((margins[f] for f in LOW_SIDE[tau]), F(0))
    clean = sum((margins[f] for f in v132.CLEAN), F(0))
    hmin, hmax = F(high_range[0] - tau), F(high_range[1] - tau)
    lmin, lmax = F(low_range[0] - tau), F(low_range[1] - tau)
    min_excess = hmin * high + lmin * low - tau * clean
    max_excess = hmax * high + lmax * low - tau * clean
    feasible = set()
    if min_excess <= 0:
        feasible.add(0)
    if max_excess > 0:
        feasible.add(hours)
    if hours > 1 and high > 0 and low + clean > 0:
        feasible.update(range(1, hours))
    return tuple(sorted(feasible or {0}))


def detail_set_at_endpoint(margins: dict[str, F], tau: int, hours: int,
                           which: str) -> tuple[int, int]:
    original = dict(v90.EF)
    try:
        v90.EF.clear()
        v90.EF.update(endpoint_ef(which))
        counts = v90.v66_set(margins, tau, hours)
    finally:
        v90.EF.clear()
        v90.EF.update(original)
    return min(counts), max(counts)


def sum_bounds(parts: list[tuple[int, int]]) -> tuple[int, int]:
    return sum(p[0] for p in parts), sum(p[1] for p in parts)


def event_count(frame: pd.DataFrame, tau: int) -> int:
    return v132.event_count(frame, tau)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = v90.sha(PROTOCOL)
    cohort_rows = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    cohort = {(str(r.region), str(r.month)[:10]) for r in cohort_rows.itertuples(index=False)}
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
            month_hours = len(month_frame)
            month_margins = v132.make_margins(month_frame)
            blocks = [block for _, block in month_frame.groupby(((month_frame.day_number - 1) // 3), sort=True)]
            for tau in TAUS:
                nominal_k = event_count(month_frame, tau)
                broad_month = v132.minmax(v132.fast_risk_side_set(month_margins, tau, month_hours))
                scenario_side: dict[str, tuple[int, int]] = {}
                scenario_detail: dict[str, tuple[int, int]] = {}
                for which in ("low", "high"):
                    side_parts, detail_parts = [], []
                    for block in blocks:
                        margins = v132.make_margins(block)
                        side_parts.append(v132.minmax(side_set_at_endpoint(margins, tau, len(block), which)))
                        detail_parts.append(detail_set_at_endpoint(margins, tau, len(block), which))
                    scenario_side[which] = sum_bounds(side_parts)
                    scenario_detail[which] = sum_bounds(detail_parts)
                coherent_side = (min(scenario_side[x][0] for x in scenario_side),
                                 max(scenario_side[x][1] for x in scenario_side))
                coherent_detail = (min(scenario_detail[x][0] for x in scenario_detail),
                                   max(scenario_detail[x][1] for x in scenario_detail))
                frozen = v132_rows[(v132_rows.region == region)
                                   & (v132_rows.month.astype(str).str[:10] == month)
                                   & (v132_rows.tau == tau) & (v132_rows.bin_days == 3)]
                broad_blockwise = (int(frozen.side_lower.iloc[0]), int(frozen.side_upper.iloc[0])) if len(frozen) == 1 else None
                if not broad_month[0] <= nominal_k <= broad_month[1]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "monthly_side_contains_nominal_K", "bounds": broad_month,
                                     "K": nominal_k})
                if not coherent_side[0] <= nominal_k <= coherent_side[1]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "coherent_side_contains_nominal_K", "bounds": coherent_side,
                                     "K": nominal_k})
                if coherent_side[0] > coherent_detail[0] or coherent_side[1] < coherent_detail[1]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "coherent_side_contains_detail", "side": coherent_side,
                                     "detail": coherent_detail})
                if broad_blockwise is None or coherent_side[0] < broad_blockwise[0] or coherent_side[1] > broad_blockwise[1]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "coherent_within_blockwise_hull", "coherent": coherent_side,
                                     "blockwise": broad_blockwise})
                for rate in RATES:
                    required = int((F(str(rate)) * month_hours).__ceil__())
                    record = {"region": region, "month": month, "tau": tau, "rate": rate,
                              "hours": month_hours, "blocks": len(blocks), "nominal_K": nominal_k,
                              "monthly_lower": broad_month[0], "monthly_upper": broad_month[1],
                              "blockwise_lower": broad_blockwise[0] if broad_blockwise else None,
                              "blockwise_upper": broad_blockwise[1] if broad_blockwise else None,
                              "coherent_lower": coherent_side[0], "coherent_upper": coherent_side[1],
                              "detail_lower": coherent_detail[0], "detail_upper": coherent_detail[1],
                              "low_scenario_lower": scenario_side["low"][0],
                              "low_scenario_upper": scenario_side["low"][1],
                              "high_scenario_lower": scenario_side["high"][0],
                              "high_scenario_upper": scenario_side["high"][1]}
                    for name, bounds in (("monthly", broad_month), ("blockwise", broad_blockwise),
                                         ("coherent", coherent_side), ("detail", coherent_detail)):
                        if bounds is None:
                            continue
                        pos, neg = bounds[0] >= required, bounds[1] < required
                        record.update({f"{name}_positive": pos, f"{name}_negative": neg,
                                       f"{name}_abstain": not (pos or neg), f"{name}_width": bounds[1] - bounds[0]})
                    rows.append(record)
        print(f"region {region}", flush=True)

    OUT.mkdir(parents=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "paired_frontier.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    metrics = [c for c in detail.columns if c.endswith(("_positive", "_negative", "_abstain", "_width"))]
    region_macro = detail.groupby(["region", "tau", "rate"])[metrics].mean().reset_index()
    macro = region_macro.groupby(["tau", "rate"])[metrics].mean().reset_index()
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    summary = {"status": "COHERENT_FACTOR_MULTIDAY_RETROSPECTIVE_PROXY",
               "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(detail), "regions": sorted(detail.region.unique()),
               "assertion_failure_rows": len(failures), "frontier": macro.to_dict("records"),
               "ef_endpoints": {"low": endpoint_ef("low"), "high": endpoint_ef("high")},
               "hourly_input_sha256": input_hashes, "cohort_sha256": v90.sha(COHORT),
               "v132_rows_sha256": v90.sha(V132_ROWS), "protocol_sha256": protocol_hash,
               "script_sha256": v90.sha(Path(__file__)), "source_audits": audits,
               "limitations": ["Shared EF box is a sensitivity model, not a Chinese measured EF distribution.",
                               "Blockwise hidden high-side fuel mixes remain partially identified.",
                               "Retrospective EIA proxy and capacity-free; no actual Chinese hourly CIF labels."]}
    if v90.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print(f"assertion failures: {len(failures)}")


if __name__ == "__main__":
    main()
