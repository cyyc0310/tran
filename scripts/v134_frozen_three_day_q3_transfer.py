"""V134: apply frozen V132 three-day risk-side contract to 2026Q3 regions."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

import v90_coarse_capacity_identification as v90
import v129_factor_support_risk_side as v129
import v132_multiscale_risk_side_information as v132

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V134_FROZEN_THREE_DAY_Q3_TRANSFER_PROTOCOL.md"
OUT = ROOT / "results/v134-frozen-three-day-q3-transfer-r1"
NATIVE = ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"
AVAILABILITY = ROOT / "results/v113-2026q3-future-time-confirmation-r1/availability.csv"
V113_ROWS = ROOT / "results/v113-2026q3-future-time-confirmation-r1/target_rows.csv"
REGIONS = ("CPLE", "FPC", "IPCO", "PNM", "SOCO")
MONTHS = ("2026-07-01", "2026-08-01", "2026-09-01")
TAUS = (600, 900)
RATES = (.01, .05, .10)


def event_count(frame: pd.DataFrame, tau: int) -> int:
    return v132.event_count(frame, tau)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if not NATIVE.exists():
        raise FileNotFoundError(NATIVE)
    protocol_hash = v90.sha(PROTOCOL)
    data = v90.source_data
    previous_paths = list(data.default_native_paths())
    data.default_native_paths = lambda: previous_paths + [NATIVE]
    data.STOP = pd.Timestamp("2026-10-01T00:00:00Z")
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + REGIONS))
    v90.REGIONS = REGIONS
    v90.source_data.ALL_REGIONS = data.ALL_REGIONS
    hourly, _, audits, source_hashes = data.prepare_regions()

    availability = pd.read_csv(AVAILABILITY)
    availability = availability.loc[availability.coverage_eligible.astype(bool)].copy()
    eligible = {(str(r.region), str(r.month)[:10]): r for r in availability.itertuples(index=False)}
    expected = {(region, month) for region in REGIONS for month in MONTHS}
    if set(eligible) != expected:
        raise ValueError(f"V113 eligible roster mismatch: missing={expected-set(eligible)}, extra={set(eligible)-expected}")
    v113 = pd.read_csv(V113_ROWS)
    v113 = v113.drop_duplicates(["region", "month", "tau"])
    rows: list[dict] = []
    failures: list[dict] = []
    actual_availability: list[dict] = []
    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2026-07-01") & times.lt("2026-10-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        frame["day_number"] = frame.time.dt.day
        for month in MONTHS:
            month_frame = frame.loc[frame.month.eq(month)]
            expected_row = eligible[(region, month)]
            hours = len(month_frame)
            if hours != int(expected_row.valid_hours):
                failures.append({"region": region, "month": month, "check": "V113_hour_count_match",
                                 "actual_hours": hours, "expected_hours": int(expected_row.valid_hours)})
            monthly_margins = v132.make_margins(month_frame)
            month_actual: dict[int, int] = {}
            month_bounds: dict[int, tuple[int, int]] = {}
            daily_side: dict[int, tuple[int, int]] = {}
            daily_detail: dict[int, tuple[int, int]] = {}
            for tau in TAUS:
                month_actual[tau] = event_count(month_frame, tau)
                month_bounds[tau] = v132.minmax(v132.fast_risk_side_set(monthly_margins, tau, hours))
                blocks = list(month_frame.groupby(((month_frame.day_number - 1) // 3), sort=True))
                side_parts, detail_parts = [], []
                block_actual = 0
                for _, block in blocks:
                    margins = v132.make_margins(block)
                    side_parts.append(v132.minmax(v132.fast_risk_side_set(margins, tau, len(block))))
                    detail_parts.append(v132.no_capacity_detailed_bounds(margins, tau, len(block)))
                    block_actual += event_count(block, tau)
                daily_side[tau] = v132.envelope(side_parts)
                daily_detail[tau] = v132.envelope(detail_parts)
                if block_actual != month_actual[tau]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "monthly_daily_actual_K_additivity",
                                     "monthly_K": month_actual[tau], "block_K": block_actual})
                if not month_bounds[tau][0] <= month_actual[tau] <= month_bounds[tau][1]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "monthly_side_contains_K", "K": month_actual[tau],
                                     "lower": month_bounds[tau][0], "upper": month_bounds[tau][1]})
                if not daily_side[tau][0] <= month_actual[tau] <= daily_side[tau][1]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "three_day_side_contains_K", "K": month_actual[tau],
                                     "lower": daily_side[tau][0], "upper": daily_side[tau][1]})
                if daily_side[tau][0] > daily_detail[tau][0] or daily_side[tau][1] < daily_detail[tau][1]:
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "three_day_detail_containment",
                                     "side": daily_side[tau], "detail": daily_detail[tau]})
                baseline = v113.loc[(v113.region == region) & (v113.month.astype(str).str[:10] == month)
                                    & (v113.tau == tau)]
                if len(baseline) != 1 or month_actual[tau] != int(baseline.actual_nominal_K.iloc[0]):
                    failures.append({"region": region, "month": month, "tau": tau,
                                     "check": "V113_nominal_K_match", "actual": month_actual[tau],
                                     "V113": int(baseline.actual_nominal_K.iloc[0]) if len(baseline) == 1 else None})
                for rate in RATES:
                    required = int((F(str(rate)) * hours).__ceil__())
                    record = {"region": region, "month": month, "tau": tau, "rate": rate,
                              "hours": hours, "coverage": hours / (pd.Period(month, freq="M").days_in_month * 24),
                              "block_count": len(blocks), "report_scalars": len(blocks) * 3,
                              "required_hours": required, "nominal_K": month_actual[tau],
                              "monthly_side_lower": month_bounds[tau][0], "monthly_side_upper": month_bounds[tau][1],
                              "three_day_side_lower": daily_side[tau][0], "three_day_side_upper": daily_side[tau][1],
                              "three_day_detail_lower": daily_detail[tau][0], "three_day_detail_upper": daily_detail[tau][1]}
                    for name, bounds in (("monthly_side", month_bounds[tau]),
                                         ("three_day_side", daily_side[tau]),
                                         ("three_day_detail", daily_detail[tau])):
                        pos, neg = bounds[0] >= required, bounds[1] < required
                        record.update({f"{name}_positive": pos, f"{name}_negative": neg,
                                       f"{name}_abstain": not (pos or neg),
                                       f"{name}_width": bounds[1] - bounds[0]})
                    rows.append(record)
            actual_availability.append({"region": region, "month": month, "hours": hours,
                                        "v113_hours": int(expected_row.valid_hours),
                                        "coverage": hours / (pd.Period(month, freq="M").days_in_month * 24)})
        print(f"region {region}", flush=True)

    OUT.mkdir(parents=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "paired_frontier.csv", index=False)
    pd.DataFrame(actual_availability).to_csv(OUT / "availability.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    metrics = [c for c in detail.columns if c.endswith(("_positive", "_negative", "_abstain", "_width"))]
    region_macro = detail.groupby(["region", "tau", "rate"])[metrics].mean().reset_index()
    macro = region_macro.groupby(["tau", "rate"])[metrics].mean().reset_index()
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    summary = {"status": "FROZEN_THREE_DAY_Q3_METHOD_TRANSFER_POSTHOC",
               "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(detail), "regions": list(REGIONS), "months": list(MONTHS),
               "assertion_failure_rows": len(failures), "frontier": macro.to_dict("records"),
               "factor_intervals_gco2_kwh": v129.EF_BOUNDS,
               "native_2026q3_sha256": v90.sha(NATIVE), "v113_availability_sha256": v90.sha(AVAILABILITY),
               "v113_rows_sha256": v90.sha(V113_ROWS), "hourly_source_hashes": source_hashes,
               "protocol_sha256": protocol_hash, "script_sha256": v90.sha(Path(__file__)),
               "source_audits": audits,
               "limitations": ["V113 Q3 outcomes were previously opened for other methods; this is not blind confirmation.",
                               "Only five regions and three months; no capacity information is used.",
                               "EF intervals are sensitivity assumptions and target labels are fixed-factor EIA proxies.",
                               "No Chinese source table or actual hourly CIF labels are used."]}
    if v90.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print(f"assertion failures: {len(failures)}")


if __name__ == "__main__":
    main()
