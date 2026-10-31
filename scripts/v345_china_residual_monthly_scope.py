"""V345: monthly allocation ambiguity of the annual all-plant scope remainder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
ANNUAL = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_annual_generation_by_tech.csv"
CAPACITY = ROOT / "data/china/cec_province_installed_capacity_2022_2023.csv"
V320_SUMMARY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/summary.json"
V338_PROTOCOL = ROOT / "docs/V338_CHINA_LAGGED_CAPACITY_FEASIBILITY_PROTOCOL.md"
PROTOCOL = ROOT / "docs/V345_CHINA_RESIDUAL_MONTHLY_SCOPE_PROTOCOL.md"
OUT = ROOT / "results/v345-china-residual-monthly-scope-r1"
PERIOD_HOURS = {2: 59 * 24, 3: 31 * 24, 4: 30 * 24, 5: 31 * 24,
                6: 30 * 24, 7: 31 * 24, 8: 31 * 24, 9: 30 * 24,
                10: 31 * 24, 11: 30 * 24, 12: 31 * 24}
VINTAGES = (2022, 2023)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    monthly = pd.read_csv(MONTHLY)
    annual = pd.read_csv(ANNUAL)
    capacity = pd.read_csv(CAPACITY)
    m = monthly.loc[(monthly.year == 2023) & (monthly.technology == "total")].copy()
    a = annual.loc[(annual.year == 2023) & (annual.technology == "total")].copy()
    if m.duplicated(["province", "month"]).any() or a.province.duplicated().any():
        raise ValueError("duplicate 2023 total-generation source keys")
    if set(m.month.unique()) != set(PERIOD_HOURS) or m.province.nunique() != 31:
        raise ValueError("unexpected 2023 monthly reporting schema")
    if set(m.province) != set(a.province):
        raise ValueError("monthly and annual province rosters differ")

    annual_map = a.set_index("province").generation_gwh.astype(float).to_dict()
    capacity = capacity.loc[capacity.year.isin(VINTAGES)].copy()
    if capacity.duplicated(["province", "year"]).any():
        raise ValueError("duplicate province-capacity vintage")
    cap_map = capacity.set_index(["province", "year"])["total"].astype(float).to_dict()
    rows = []
    for province, group in m.groupby("province", sort=True):
        observed_annual = float(group.generation_gwh.sum())
        annual_all = annual_map[province]
        remainder = annual_all - observed_annual
        if remainder < 0:
            raise ValueError(f"negative annual scope remainder for {province}: {remainder}")
        periods = group.sort_values("month").to_dict("records")
        for vintage in VINTAGES:
            # The source capacity unit is 10,000 kW = 10 MW. A common annual
            # capacity is used as a permissive period cap in this feasibility audit.
            total_capacity_mw = cap_map[(province, vintage)] * 10.0
            period_residual_caps = []
            for rec in periods:
                h = PERIOD_HOURS[int(rec["month"])]
                observed_mwh = float(rec["generation_gwh"]) * 1000.0
                period_residual_caps.append(max(0.0, total_capacity_mw * h - observed_mwh) / 1000.0)
            total_residual_room = sum(period_residual_caps)
            if total_residual_room + 1e-9 < remainder:
                raise ValueError(f"annual remainder exceeds capacity room: {province} {vintage}")
            for rec, room in zip(periods, period_residual_caps):
                other_room = total_residual_room - room
                low = max(0.0, remainder - other_room)
                high = min(remainder, room)
                if low > high + 1e-8:
                    raise ValueError(f"empty monthly residual range: {province} {vintage} month={rec['month']}")
                month = int(rec["month"])
                rows.append({
                    "province": province,
                    "period": "Jan-Feb" if month == 2 else f"{month:02d}",
                    "month_code": month,
                    "hours": PERIOD_HOURS[month],
                    "capacity_vintage": vintage,
                    "monthly_reported_total_gwh": float(rec["generation_gwh"]),
                    "annual_all_plant_total_gwh": annual_all,
                    "annual_unobserved_scope_remainder_gwh": remainder,
                    "total_capacity_mw": total_capacity_mw,
                    "period_residual_capacity_room_gwh": room,
                    "annual_residual_room_gwh": total_residual_room,
                    "monthly_residual_min_gwh": low,
                    "monthly_residual_max_gwh": high,
                    "monthly_residual_range_width_gwh": high - low,
                    "max_is_full_annual_remainder": abs(high - remainder) <= 1e-8,
                    "min_is_zero": low <= 1e-8,
                })
    df = pd.DataFrame(rows)
    if len(df) != 31 * 11 * len(VINTAGES) or df.duplicated(["province", "period", "capacity_vintage"]).any():
        raise ValueError(f"row/key validation failed: {len(df)}")
    if not df.min_is_zero.all():
        # Do not silently turn an unexpectedly informative capacity restriction into a pass.
        raise ValueError("some province-periods have a positive minimum remainder")

    frontier = []
    for vintage, frame in df.groupby("capacity_vintage"):
        frontier.append({
            "capacity_vintage": int(vintage),
            "province_periods": int(len(frame)),
            "mean_annual_remainder_gwh": float(frame.groupby("province").annual_unobserved_scope_remainder_gwh.first().mean()),
            "median_monthly_residual_width_gwh": float(frame.monthly_residual_range_width_gwh.median()),
            "mean_monthly_residual_width_gwh": float(frame.monthly_residual_range_width_gwh.mean()),
            "median_share_of_annual_remainder_allocable_to_any_one_period": float((frame.monthly_residual_max_gwh / frame.annual_unobserved_scope_remainder_gwh).median()),
            "mean_share_of_annual_remainder_allocable_to_any_one_period": float((frame.monthly_residual_max_gwh / frame.annual_unobserved_scope_remainder_gwh).mean()),
            "periods_where_full_annual_remainder_fits": int(frame.max_is_full_annual_remainder.sum()),
            "period_count": int(len(frame)),
            "annual_remainder_capacity_feasible_provinces": int(frame.province.nunique()),
            "minimum_capacity_room_slack_gwh": float((frame.groupby("province").annual_residual_room_gwh.first() - frame.groupby("province").annual_unobserved_scope_remainder_gwh.first()).min()),
        })
    OUT.mkdir(parents=True)
    df.to_csv(OUT / "province_period_residual_ranges.csv", index=False)
    summary = {
        "status": "CONDITIONAL_SCOPE_REMAINDER_ALLOCATION_FEASIBILITY_NOT_HOURLY_CIF",
        "year": 2023,
        "n_provinces": int(df.province.nunique()),
        "reporting_periods": ["Jan-Feb", *[f"{i:02d}" for i in range(3, 13)]],
        "source_schema": "NBS-derived above-size monthly total; same-source annual all-plant total; CEC province total installed capacity",
        "annual_unobserved_remainder_gwh_sum": float(df.loc[df.capacity_vintage.eq(2023)].groupby("province").annual_unobserved_scope_remainder_gwh.first().sum()),
        "frontier": frontier,
        "assumptions": [
            "February is the cumulative Jan-Feb reporting bin; it is counted once and assigned 59 days.",
            "Annual all-plant minus summed above-size monthly total is treated as a nonnegative annual remainder, without assigning it to technology groups.",
            "The province total installed capacity is a permissive upper bound on total monthly generation, applied as a constant across reporting periods.",
            "CEC 2023 capacity is retrospective; 2022 capacity is only a sensitivity arm and is not assumed to be a proven 2023 upper bound.",
            "This computes the feasible monthly allocation range of the annual remainder. It does not calculate CIF, K, coverage, or prediction performance."
        ],
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (MONTHLY, ANNUAL, CAPACITY, V320_SUMMARY, V338_PROTOCOL, PROTOCOL, Path(__file__))},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(frontier).to_string(index=False))
    print(f"province-period rows={len(df)}; annual remainder={summary['annual_unobserved_remainder_gwh_sum']:.3f} GWh")


if __name__ == "__main__":
    main()
