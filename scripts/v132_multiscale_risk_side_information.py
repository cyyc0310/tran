"""V132: compare threshold-side information at multiple temporal resolutions."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

import v90_coarse_capacity_identification as v90
import v129_factor_support_risk_side as v129

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V132_MULTISCALE_RISK_SIDE_PROTOCOL.md"
OUT = ROOT / "results/v132-multiscale-risk-side-r1"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
THERMAL = ("coal", "gas", "oil", "other")
CLEAN = ("nuclear", "hydro", "wind", "solar")
TAUS = (600, 900)
RATES = (.01, .05, .10)
BIN_DAYS = (1, 3, 7, 14)


def make_margins(frame: pd.DataFrame) -> dict[str, F]:
    return {fuel: sum((F(str(float(x))) for x in frame[f"generation_{fuel}"]), F(0))
            for fuel in v90.FUELS}


def no_capacity_groups(margins: dict[str, F]) -> dict[str, F]:
    return {"thermal": sum((margins[f] for f in THERMAL), F(0)),
            **{fuel: margins[fuel] for fuel in CLEAN}}


def envelope(parts: list[tuple[int, int]]) -> tuple[int, int]:
    return sum(x[0] for x in parts), sum(x[1] for x in parts)


def fast_risk_side_set(margins: dict[str, F], tau: int, hours: int) -> tuple[int, ...]:
    """Exact no-cap set: the only interior feasibility conditions are high-side
    energy and positive non-event-side energy; endpoint feasibility follows
    from the low/high EF support totals.
    """
    if tau == 600:
        high = margins["coal"] + margins["oil"]
        low = margins["gas"] + margins["other"]
        hmin, hmax, lmin, lmax = F(720 - tau), F(1100 - tau), F(200 - tau), F(600 - tau)
    elif tau == 900:
        high = margins["coal"]
        low = margins["gas"] + margins["oil"] + margins["other"]
        hmin = hmax = F(1100 - tau)
        lmin, lmax = F(200 - tau), F(880 - tau)
    else:
        raise ValueError(f"unsupported threshold {tau}")
    clean = sum((margins[f] for f in CLEAN), F(0))
    if high + low + clean <= 0:
        return (0,)
    min_excess = hmin * high + lmin * low - tau * clean
    max_excess = hmax * high + lmax * low - tau * clean
    feasible = set()
    if min_excess <= 0:
        feasible.add(0)
    if max_excess > 0:
        feasible.add(hours)
    if hours > 1 and high > 0 and low + clean > 0:
        feasible.update(range(1, hours))
    if tau == 900:
        feasible.add(0)  # coal at the lower EF endpoint equals the strict threshold
    return tuple(sorted(feasible or {0}))


def no_capacity_detailed_bounds(margins: dict[str, F], tau: int, hours: int) -> tuple[int, int]:
    original = dict(v90.EF)
    endpoint_sets = []
    try:
        for endpoint in ("low", "high"):
            for fuel, bounds in v129.EF_BOUNDS.items():
                v90.EF[fuel] = bounds[0 if endpoint == "low" else 1]
            endpoint_sets.append(v90.v66_set(margins, tau, hours))
    finally:
        v90.EF.clear()
        v90.EF.update(original)
    return min(min(values) for values in endpoint_sets), max(max(values) for values in endpoint_sets)


def event_count(frame: pd.DataFrame, tau: int) -> int:
    count = 0
    for row in frame.itertuples(index=False):
        values = {fuel: F(str(float(getattr(row, f"generation_{fuel}")))) for fuel in v90.FUELS}
        total = sum(values.values(), F(0))
        if total > 0:
            co2 = sum((values[fuel] * F(str(v90.EF[fuel])) for fuel in v90.FUELS), F(0))
            count += co2 > F(tau) * total
    return count


def validate_fast_paths() -> None:
    """Cross-check exact closed forms against the slower general solvers."""
    checks = 0
    for hours in (2, 6, 24):
        for high in (F(0), F(7)):
            for low in (F(0), F(11)):
                for clean in (F(0), F(5)):
                    margins = {fuel: F(0) for fuel in v90.FUELS}
                    margins.update({"coal": high, "gas": low, "hydro": clean})
                    groups = no_capacity_groups(margins)
                    for tau in TAUS:
                        fast = fast_risk_side_set(margins, tau, hours)
                        exact = v129.side_sets(margins, groups, groups, tau, hours)[0]
                        if fast != exact:
                            raise AssertionError(f"fast risk set mismatch: {hours=} {high=} {low=} {clean=} {tau=} {fast=} {exact=}")
                        checks += 1
    return checks


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = v90.sha(PROTOCOL)
    closed_form_checks = validate_fast_paths()
    frozen = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    cohort = {(str(row.region), str(row.month)[:10]) for row in frozen.itertuples(index=False)}
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
            month_margins = make_margins(month_frame)
            month_no_cap = no_capacity_groups(month_margins)
            for tau in TAUS:
                monthly_side = minmax(fast_risk_side_set(month_margins, tau, total_hours))
                for bin_days in BIN_DAYS:
                    side_parts: list[tuple[int, int]] = []
                    detail_parts: list[tuple[int, int]] = []
                    observed_k = 0
                    block_count = 0
                    for block_id, block in month_frame.groupby(((month_frame.day_number - 1) // bin_days), sort=True):
                        block_hours = len(block)
                        margins = make_margins(block)
                        groups = no_capacity_groups(margins)
                        side = fast_risk_side_set(margins, tau, block_hours)
                        detail = no_capacity_detailed_bounds(margins, tau, block_hours)
                        side_parts.append(minmax(side))
                        detail_parts.append(detail)
                        observed_k += event_count(block, tau)
                        block_count += 1
                    side_bounds = envelope(side_parts)
                    detail_bounds = envelope(detail_parts)
                    if not (side_bounds[0] <= observed_k <= side_bounds[1]):
                        failures.append({"region": region, "month": month, "tau": tau, "bin_days": bin_days,
                                         "check": "side_contains_observed_K", "K": observed_k,
                                         "lower": side_bounds[0], "upper": side_bounds[1]})
                    if not (side_bounds[0] <= detail_bounds[0] and side_bounds[1] >= detail_bounds[1]):
                        failures.append({"region": region, "month": month, "tau": tau, "bin_days": bin_days,
                                         "check": "side_contains_detailed", "side": side_bounds,
                                         "detail": detail_bounds})
                    for rate in RATES:
                        required = int((F(str(rate)) * total_hours).__ceil__())
                        def metrics(bounds: tuple[int, int]) -> dict[str, object]:
                            pos = bounds[0] >= required
                            neg = bounds[1] < required
                            return {"positive": pos, "negative": neg, "abstain": not (pos or neg),
                                    "width": bounds[1] - bounds[0]}
                        row = {"region": region, "month": month, "tau": tau, "rate": rate,
                               "bin_days": bin_days, "block_count": block_count, "hours": total_hours,
                               "report_scalars": block_count * 3, "required_hours": required,
                               "nominal_K": observed_k,
                               "monthly_side_lower": monthly_side[0], "monthly_side_upper": monthly_side[1],
                               "side_lower": side_bounds[0], "side_upper": side_bounds[1],
                               "detail_lower": detail_bounds[0], "detail_upper": detail_bounds[1]}
                        for name, bounds in (("side", side_bounds), ("detail", detail_bounds),
                                             ("monthly_side", monthly_side)):
                            for key, value in metrics(bounds).items():
                                row[f"{name}_{key}"] = value
                        rows.append(row)
        print(f"region {region}", flush=True)

    OUT.mkdir(parents=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "paired_frontier.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "nesting_failures.csv", index=False)
    metrics = [c for c in detail.columns if c.endswith(("_positive", "_negative", "_abstain", "_width"))]
    region_macro = detail.groupby(["region", "tau", "rate", "bin_days"])[metrics].mean().reset_index()
    macro = region_macro.groupby(["tau", "rate", "bin_days"])[metrics].mean().reset_index()
    counts = detail.groupby(["tau", "bin_days"])[["block_count", "report_scalars"]].mean().reset_index()
    macro = macro.merge(counts, on=["tau", "bin_days"], how="left")
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    summary = {"status": "MULTISCALE_RISK_SIDE_INFORMATION_RETROSPECTIVE_PROXY",
               "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(detail), "regions": sorted(detail.region.unique()),
               "nesting_failure_rows": len(failures), "closed_form_crosschecks": closed_form_checks,
               "frontier": macro.to_dict("records"),
               "bin_days": BIN_DAYS, "thresholds": TAUS, "rates": RATES,
               "information_contract": "per block: total thermal, threshold high-side, and total clean generation",
               "factor_intervals_gco2_kwh": v129.EF_BOUNDS,
               "hourly_input_sha256": input_hashes, "cohort_sha256": v90.sha(COHORT),
               "protocol_sha256": protocol_hash, "script_sha256": v90.sha(Path(__file__)),
               "source_audits": audits,
               "limitations": ["Retrospective oracle coarsening of EIA hourly fuel generation; not China or actual CIF labels.",
                               "Factor intervals are stress assumptions, not validated factor confidence sets.",
                               "Blockwise interval addition is conservative under shared cross-block factors.",
                               "Reporting availability, vintage, and measurement error are unvalidated."]}
    if v90.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print(f"nesting failures: {len(failures)}")


def minmax(values: tuple[int, ...]) -> tuple[int, int]:
    return min(values), max(values)


if __name__ == "__main__":
    main()
