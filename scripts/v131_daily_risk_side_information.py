"""V131: value of a minimal daily risk-side summary over monthly information."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

import v90_coarse_capacity_identification as v90
import v129_factor_support_risk_side as v129

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V131_DAILY_RISK_SIDE_INFORMATION_PROTOCOL.md"
OUT = ROOT / "results/v131-daily-risk-side-information-r1"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
THERMAL = ("coal", "gas", "oil", "other")
CLEAN = ("nuclear", "hydro", "wind", "solar")
TAUS = (600, 900)
RATES = (.01, .05, .10)


def make_margins(frame: pd.DataFrame) -> dict[str, F]:
    return {fuel: sum((F(str(float(x))) for x in frame[f"generation_{fuel}"]), F(0))
            for fuel in v90.FUELS}


def no_capacity_groups(margins: dict[str, F]) -> dict[str, F]:
    return {"thermal": sum((margins[f] for f in THERMAL), F(0)),
            **{fuel: margins[fuel] for fuel in CLEAN}}


def interval(bounds: tuple[int, ...]) -> tuple[int, int]:
    return min(bounds), max(bounds)


def sum_intervals(bounds: list[tuple[int, int]]) -> tuple[int, int]:
    return sum(x[0] for x in bounds), sum(x[1] for x in bounds)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = v90.sha(PROTOCOL)
    cohort_rows = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    cohort = {(str(row.region), str(row.month)[:10]) for row in cohort_rows.itertuples(index=False)}
    data = v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, input_hashes = data.prepare_regions()

    rows: list[dict] = []
    nesting_failures: list[dict] = []
    coverage_rows: list[dict] = []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2024-01-01") & times.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        frame["day"] = frame.time.dt.strftime("%Y-%m-%d")
        for month, month_frame in frame.groupby("month", sort=True):
            if (region, month) not in cohort:
                continue
            monthly_margins = make_margins(month_frame)
            monthly_no_cap = no_capacity_groups(monthly_margins)
            hours = len(month_frame)
            for tau in TAUS:
                # The single monthly scalar plus monthly group totals.
                monthly_side = interval(v129.side_sets(monthly_margins, monthly_no_cap,
                                                         monthly_no_cap, tau, hours)[0])
                monthly_detail = interval(v129.robust_detailed_sets(
                    monthly_margins, monthly_no_cap, tau, hours)[0])

                daily_side_parts: list[tuple[int, int]] = []
                daily_detail_parts: list[tuple[int, int]] = []
                daily_observed_k = 0
                days_used = 0
                for _, day_frame in month_frame.groupby("day", sort=True):
                    day_hours = len(day_frame)
                    day_margins = make_margins(day_frame)
                    day_no_cap = no_capacity_groups(day_margins)
                    daily_side = interval(v129.side_sets(day_margins, day_no_cap, day_no_cap,
                                                          tau, day_hours)[0])
                    daily_detail = interval(v129.robust_detailed_sets(
                        day_margins, day_no_cap, tau, day_hours)[0])
                    daily_side_parts.append(daily_side)
                    daily_detail_parts.append(daily_detail)
                    days_used += 1
                    for record in day_frame.itertuples(index=False):
                        generation = {fuel: F(str(float(getattr(record, f"generation_{fuel}")))) for fuel in v90.FUELS}
                        total = sum(generation.values(), F(0))
                        if total > 0:
                            co2 = sum((generation[fuel] * F(str(v90.EF[fuel])) for fuel in v90.FUELS), F(0))
                            daily_observed_k += co2 > F(tau) * total
                daily_side_total = sum_intervals(daily_side_parts)
                daily_detail_total = sum_intervals(daily_detail_parts)
                if not (monthly_side[0] <= daily_observed_k <= monthly_side[1]):
                    nesting_failures.append({"region": region, "month": month, "tau": tau,
                                             "check": "monthly_side_contains_nominal_K",
                                             "K": daily_observed_k, "lo": monthly_side[0], "hi": monthly_side[1]})
                if not (daily_side_total[0] <= daily_observed_k <= daily_side_total[1]):
                    nesting_failures.append({"region": region, "month": month, "tau": tau,
                                             "check": "daily_side_contains_nominal_K",
                                             "K": daily_observed_k, "lo": daily_side_total[0], "hi": daily_side_total[1]})
                if daily_side_total[0] > daily_detail_total[0] or daily_side_total[1] < daily_detail_total[1]:
                    nesting_failures.append({"region": region, "month": month, "tau": tau,
                                             "check": "daily_side_contains_daily_detail",
                                             "side": daily_side_total, "detail": daily_detail_total})
                for rate in RATES:
                    required = int((F(str(rate)) * hours).__ceil__())
                    def decision(bounds: tuple[int, int]) -> tuple[bool, bool, bool]:
                        positive = bounds[0] >= required
                        negative = bounds[1] < required
                        return positive, negative, not (positive or negative)
                    row = {"region": region, "month": month, "tau": tau, "rate": rate,
                           "hours": hours, "days_used": days_used, "required_hours": required,
                           "nominal_K": daily_observed_k,
                           "monthly_side_lower": monthly_side[0], "monthly_side_upper": monthly_side[1],
                           "daily_side_lower": daily_side_total[0], "daily_side_upper": daily_side_total[1],
                           "daily_detail_lower": daily_detail_total[0], "daily_detail_upper": daily_detail_total[1],
                           "monthly_detail_lower": monthly_detail[0], "monthly_detail_upper": monthly_detail[1]}
                    for name, bounds in (("monthly_side", monthly_side), ("daily_side", daily_side_total),
                                         ("daily_detail", daily_detail_total), ("monthly_detail", monthly_detail)):
                        pos, neg, abstain = decision(bounds)
                        row.update({f"{name}_positive": pos, f"{name}_negative": neg,
                                    f"{name}_abstain": abstain, f"{name}_width": bounds[1] - bounds[0]})
                    rows.append(row)
            coverage_rows.append({"region": region, "month": month, "observed_hours": hours,
                                  "calendar_hours": pd.Period(month, freq="M").days_in_month * 24,
                                  "valid_day_bins": month_frame.day.nunique()})
        print(f"region {region}", flush=True)

    OUT.mkdir(parents=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "paired_frontier.csv", index=False)
    pd.DataFrame(nesting_failures).to_csv(OUT / "nesting_failures.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT / "availability.csv", index=False)
    metrics = [c for c in detail.columns if c.endswith(("_positive", "_negative", "_abstain", "_width"))]
    region_macro = detail.groupby(["region", "tau", "rate"])[metrics].mean().reset_index()
    macro = region_macro.groupby(["tau", "rate"])[metrics].mean().reset_index()
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    summary = {"status": "DAILY_RISK_SIDE_INFORMATION_RETROSPECTIVE_PROXY",
               "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(detail), "regions": sorted(detail.region.unique()),
               "nesting_failure_rows": len(nesting_failures), "frontier": macro.to_dict("records"),
               "information_contract": {"monthly": "monthly risk-side energy + monthly thermal/clean totals",
                                        "daily": "daily risk-side energy + daily thermal total + daily clean total",
                                        "daily_detail_oracle": "daily detailed fuel totals",
                                        "daily_bins": "UTC calendar days; valid observed hours define each bin"},
               "factor_intervals_gco2_kwh": v129.EF_BOUNDS,
               "hourly_input_sha256": input_hashes, "cohort_sha256": v90.sha(COHORT),
               "protocol_sha256": protocol_hash, "script_sha256": v90.sha(Path(__file__)),
               "source_audits": audits,
               "limitations": ["Retrospective oracle coarsening of EIA hourly fuel records; not Chinese public data or CIF labels.",
                               "Factor intervals are sensitivity assumptions, not validated Chinese emission-factor confidence sets.",
                               "Daily interval addition is conservative and ignores cross-day common-factor dependence.",
                               "Daily information publication vintage and actual reporting error are not validated."]}
    if v90.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print(f"nesting failures: {len(nesting_failures)}")


if __name__ == "__main__":
    main()
