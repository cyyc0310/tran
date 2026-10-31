"""V138: risk-side measurement-error sensitivity for the V136 contract."""
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
PROTOCOL = ROOT / "docs/V138_THREE_DAY_CAPACITY_FIELD_ERROR_PROTOCOL.md"
OUT = ROOT / "results/v138-three-day-capacity-field-error-r1"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
V129_ROWS = ROOT / "results/v129-factor-support-risk-side-r1/paired_frontier.csv"
V136_ROWS = ROOT / "results/v136-three-day-capacity-combination-r1/paired_frontier.csv"
ERRORS = (0, .01, .02, .05, .10)
RATES = (.01, .05, .10)
TAUS = (600, 900)
CLEAN = ("nuclear", "hydro", "wind", "solar")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    protocol_hash = v90.sha(PROTOCOL)
    cohort_rows = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    cohort = {(str(r.region), str(r.month)[:10]) for r in cohort_rows.itertuples(index=False)}
    prior = pd.read_csv(V129_ROWS)
    v136 = pd.read_csv(V136_ROWS).drop_duplicates(["region", "month", "tau", "rate"])
    v136 = v136.loc[v136.rate.eq(.01)].copy()
    data = v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, input_hashes = data.prepare_regions()
    capacities, capacity_hashes = {}, {}
    for year in (2023, 2024):
        capacities[year], capacity_hashes[str(year)] = v90.capacities(year)

    rows: list[dict] = []
    failures: list[dict] = []
    availability: list[dict] = []
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
            hours = len(month_frame)
            vintage = int(month[:4]) - 1
            caprow = capacities[vintage].loc[region] if region in capacities[vintage].index else pd.Series(dtype=float)
            _, cap_groups = v129.make_capacity_groups(caprow)
            cap_categories = {"thermal": cap_groups["thermal"], **{fuel: cap_groups[fuel] for fuel in CLEAN}}
            blocks = list(month_frame.groupby(((month_frame.day_number - 1) // 3), sort=True))
            block_info = []
            fallback_blocks = 0
            for block_id, block in blocks:
                margins = v132.make_margins(block)
                no_cap = v132.no_capacity_groups(margins)
                feasible = all(sum((margins[f] for f in members), F(0)) <= len(block) * cap_groups[group]
                               for group, members in v90.GROUPS.items())
                active_caps = cap_categories if feasible else no_cap
                fallback_blocks += not feasible
                block_info.append((block_id, block, margins, no_cap, active_caps, feasible))
            availability.append({"region": region, "month": month, "blocks": len(blocks),
                                 "capacity_fallback_blocks": fallback_blocks,
                                 "fallback_fraction": fallback_blocks / len(blocks) if blocks else 0.0})
            for tau in TAUS:
                actual = v132.event_count(month_frame, tau)
                detailed_parts = []
                for _, block, margins, _, active_caps, _ in block_info:
                    detailed_parts.append(v129.robust_detailed_sets(margins, active_caps, tau, len(block))[1])
                detailed = v132.envelope(detailed_parts)
                error_bounds: dict[float, tuple[int, int]] = {}
                for error in ERRORS:
                    parts = []
                    for _, block, margins, no_cap, active_caps, _ in block_info:
                        high_value = (margins["coal"] + margins["oil"] if tau == 600 else margins["coal"])
                        high_range, low_range = ((720, 1100), (200, 600)) if tau == 600 else ((1100, 1100), (200, 880))
                        side = v130.uncertain_side_set(margins, high_value, high_range, low_range,
                                                       active_caps, tau, len(block), error,
                                                       include_zero=(tau == 900))
                        parts.append(v132.minmax(side))
                    bounds = v132.envelope(parts)
                    error_bounds[error] = bounds
                    if not bounds[0] <= actual <= bounds[1]:
                        failures.append({"region": region, "month": month, "tau": tau, "error": error,
                                         "check": "nominal_K_containment", "bounds": bounds, "K": actual})
                    if bounds[0] > detailed[0] or bounds[1] < detailed[1]:
                        failures.append({"region": region, "month": month, "tau": tau, "error": error,
                                         "check": "detailed_envelope_containment", "bounds": bounds,
                                         "detail": detailed})
                for previous_error, current_error in zip(ERRORS, ERRORS[1:]):
                    if error_bounds[current_error][0] > error_bounds[previous_error][0] or error_bounds[current_error][1] < error_bounds[previous_error][1]:
                        failures.append({"region": region, "month": month, "tau": tau,
                                         "error": current_error, "check": "monotone_error_expansion",
                                         "previous_error": previous_error,
                                         "previous_bounds": error_bounds[previous_error],
                                         "bounds": error_bounds[current_error]})
                frozen = v136[(v136.region == region) & (v136.month.astype(str).str[:10] == month)
                              & (v136.tau == tau)]
                if len(frozen) != 1 or error_bounds[0][0] != int(frozen.three_day_capacity_lower.iloc[0]) or error_bounds[0][1] != int(frozen.three_day_capacity_upper.iloc[0]):
                    failures.append({"region": region, "month": month, "tau": tau, "error": 0,
                                     "check": "V136_zero_error_equality", "bounds": error_bounds[0]})
                monthly_row = prior[(prior.region == region) & (prior.month.astype(str).str[:10] == month)
                                    & (prior.tau == tau) & (prior.rate == .01)]
                monthly_detail = (int(monthly_row.robust_detailed_capacity_lower.iloc[0]),
                                  int(monthly_row.robust_detailed_capacity_upper.iloc[0]))
                if not monthly_detail[0] <= actual <= monthly_detail[1]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "monthly_detailed_capacity_contains_K",
                                     "bounds": monthly_detail, "K": actual})
                for error, bounds in error_bounds.items():
                    for rate in RATES:
                        required = int((F(str(rate)) * hours).__ceil__())
                        pos, neg = bounds[0] >= required, bounds[1] < required
                        dpos, dneg = detailed[0] >= required, detailed[1] < required
                        mpos, mneg = monthly_detail[0] >= required, monthly_detail[1] < required
                        rows.append({"region": region, "month": month, "tau": tau, "rate": rate,
                                     "error": error, "hours": hours, "blocks": len(blocks),
                                     "capacity_fallback_blocks": fallback_blocks,
                                     "required_hours": required, "nominal_K": actual,
                                     "side_lower": bounds[0], "side_upper": bounds[1],
                                     "side_width": bounds[1] - bounds[0], "side_positive": pos,
                                     "side_negative": neg, "side_abstain": not (pos or neg),
                                     "detail_lower": detailed[0], "detail_upper": detailed[1],
                                     "detail_width": detailed[1] - detailed[0], "detail_positive": dpos,
                                     "detail_negative": dneg, "detail_abstain": not (dpos or dneg),
                                     "monthly_detail_lower": monthly_detail[0],
                                     "monthly_detail_upper": monthly_detail[1],
                                     "monthly_detail_width": monthly_detail[1] - monthly_detail[0],
                                     "monthly_detail_positive": mpos,
                                     "monthly_detail_negative": mneg,
                                     "monthly_detail_abstain": not (mpos or mneg)})
        print(f"region {region}", flush=True)

    OUT.mkdir(parents=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "measurement_error_frontier.csv", index=False)
    pd.DataFrame(availability).to_csv(OUT / "capacity_availability.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    metrics = [c for c in detail.columns if c.endswith(("_positive", "_negative", "_abstain", "_width"))]
    region_macro = detail.groupby(["error", "region", "tau", "rate"])[metrics].mean().reset_index()
    macro = region_macro.groupby(["error", "tau", "rate"])[metrics].mean().reset_index()
    fallback_fraction = float(pd.DataFrame(availability).groupby("region").fallback_fraction.mean().mean())
    macro["mean_region_capacity_fallback_fraction"] = fallback_fraction
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    summary = {"status": "THREE_DAY_RISK_SIDE_CAPACITY_MEASUREMENT_ERROR_RETROSPECTIVE_PROXY",
               "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(detail), "nesting_failure_rows": len(failures),
               "capacity_fallback_fraction": fallback_fraction, "errors": ERRORS,
               "frontier": macro.to_dict("records"), "capacity_hashes": capacity_hashes,
               "hourly_input_sha256": input_hashes, "cohort_sha256": v90.sha(COHORT),
               "v129_rows_sha256": v90.sha(V129_ROWS), "v136_rows_sha256": v90.sha(V136_ROWS),
               "protocol_sha256": protocol_hash, "script_sha256": v90.sha(Path(__file__)),
               "source_audits": audits,
               "limitations": ["Measurement error is a sensitivity assumption, not measured Chinese reporting error.",
                               "Thermal total, clean total, and capacity are exact; only risk-side MWh is noisy.",
                               "EIA fixed-factor proxy and retrospective cohort; no Chinese CIF validation."]}
    if v90.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print(f"assertion failures: {len(failures)}")


if __name__ == "__main__":
    main()
