"""V338: audit Chinese lagged installed-capacity upper bounds vs monthly generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
CAPACITY = ROOT / "data/china/cec_province_installed_capacity_2022_2023.csv"
PROTOCOL = ROOT / "docs/V338_CHINA_LAGGED_CAPACITY_FEASIBILITY_PROTOCOL.md"
OUT = ROOT / "results/v338-china-lagged-capacity-feasibility-r2"
GROUPS = ("thermal", "hydro", "nuclear", "wind", "solar")
CAPACITY_COLUMN = {"thermal": "thermal", "hydro": "hydro", "nuclear": "total",
                   "wind": "wind", "solar": "solar"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    monthly = pd.read_csv(MONTHLY)
    monthly = monthly.loc[monthly.year.eq(2023)].copy()
    capacities = pd.read_csv(CAPACITY)
    by_cap = capacities.set_index(["province", "year"])
    rows: list[dict[str, object]] = []
    for province in sorted(monthly.province.unique()):
        for month in (2, *range(3, 13)):
            period = monthly.loc[monthly.province.eq(province) & monthly.month.eq(month)]
            total = period.loc[period.technology.eq("total"), "generation_gwh"]
            if total.empty or not np.isfinite(float(total.iloc[0])):
                raise ValueError(f"missing total {province} month={month}")
            hours = (59 if month == 2 else int(pd.Period(f"2023-{month:02d}").days_in_month)) * 24
            for group in GROUPS:
                generation = period.loc[period.technology.eq(group), "generation_gwh"]
                reported = not generation.empty and np.isfinite(float(generation.iloc[0]))
                for vintage in (2022, 2023):
                    caprow = by_cap.loc[(province, vintage)] if (province, vintage) in by_cap.index else None
                    cap_10k_kw = np.nan if caprow is None else float(caprow[CAPACITY_COLUMN[group]])
                    cap_mw = cap_10k_kw * 10.0
                    gen_gwh = float(generation.iloc[0]) if reported else np.nan
                    if reported and cap_mw > 0:
                        cf = gen_gwh * 1000 / (cap_mw * hours)
                    elif reported and gen_gwh == 0:
                        cf = 0.0
                    else:
                        cf = np.nan
                    rows.append({"province": province, "month": month, "period_label": "Jan-Feb" if month == 2 else f"{month:02d}",
                                 "period_hours": hours, "technology": group, "capacity_vintage": vintage,
                                 "capacity_10k_kw": cap_10k_kw, "generation_gwh": gen_gwh,
                                 "generation_reported": reported, "average_capacity_factor": cf,
                                 "monthly_energy_within_capacity": bool(cf <= 1 + 1e-10) if np.isfinite(cf) else None})
    detail = pd.DataFrame(rows)
    summary = (detail.loc[detail.generation_reported]
               .groupby(["capacity_vintage", "technology"], as_index=False)
               .agg(n_province_periods=("average_capacity_factor", "size"),
                    n_capacity_unavailable=("average_capacity_factor", lambda x: int(x.isna().sum())),
                    n_above_cap=("monthly_energy_within_capacity", lambda x: int(x.eq(False).sum())),
                    max_monthly_capacity_factor=("average_capacity_factor", "max"),
                    median_monthly_capacity_factor=("average_capacity_factor", "median")))
    by_vintage = (detail.loc[detail.generation_reported]
                  .groupby("capacity_vintage", as_index=False)
                  .agg(n_observed_groups=("monthly_energy_within_capacity", "size"),
                       n_capacity_unavailable=("average_capacity_factor", lambda x: int(x.isna().sum())),
                       n_above_cap=("monthly_energy_within_capacity", lambda x: int(x.eq(False).sum())),
                       province_periods=("province", lambda x: x.nunique() * 11)))
    cap_balance = capacities.copy()
    cec_summary = cap_balance.groupby("year")[
        ["thermal", "hydro", "wind", "solar", "total", "category_sum_minus_total_signed"]
    ].sum().reset_index()
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "capacity_feasibility_detail.csv", index=False)
    summary.to_csv(OUT / "technology_summary.csv", index=False)
    by_vintage.to_csv(OUT / "vintage_summary.csv", index=False)
    cap_balance.to_csv(OUT / "cec_capacity_balance_audit.csv", index=False)
    metadata = {
        "status": "NECESSARY_CONDITION_CAPACITY_AUDIT_NOT_HOURLY_CIF_VALIDATION",
        "n_provinces": int(monthly.province.nunique()),
        "periods_per_province": 11,
        "technology_summary": summary.to_dict("records"),
        "vintage_summary": by_vintage.to_dict("records"),
        "cec_national_sums_10k_kw": cec_summary.to_dict("records"),
        "missing_monthly_category_rows": int((~detail.generation_reported).sum()),
        "limitations": ["Monthly mean energy feasibility only; no hourly peaks or CIF labels.",
                        "NBS generation is above-designated-size; CEC capacity is province-wide.",
                        "Yearbook values are retrospective; 2023 capacity is not ex-ante at 2023 origin.",
                        "Passing a monthly mean capacity test is necessary, not sufficient, for hourly capacity validity.",
                        "CEC does not report provincial nuclear capacity as a separate table in this yearbook; total provincial capacity is used as a conservative nuclear upper bound.",
                        "Cross-table category sums do not exactly equal total capacity (rounding/scope differences); see signed audit column."],
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (MONTHLY, CAPACITY, PROTOCOL, Path(__file__))},
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("CEC provincial capacity source sums:")
    print(cec_summary.to_string(index=False))
    print("\nCapacity feasibility by technology and vintage:")
    print(summary.to_string(index=False))
    print("\nVintage summary:")
    print(by_vintage.to_string(index=False))


if __name__ == "__main__":
    main()
