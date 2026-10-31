"""V350: factorial ablation of estimated fuel mix and installed capacity."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import GROUPS, ROOT, sharp_group_set
from v348_ember_share_error_frontier import FUEL_NAMES, robust_ef_bounds

V341 = ROOT / "results/v341-china-complete-category-hull-r2"
V348 = ROOT / "results/v348-ember-share-error-frontier-r1"
V349 = ROOT / "results/v349-ember-total-vs-mix-ablation-r2"
EMBER = ROOT / "results/v294-china-ember-lineage-r1/ember_china_monthly_generation.csv"
PROTOCOL = ROOT / "docs/V350_FUEL_MIX_CAPACITY_FACTORIAL_PROTOCOL.md"
OUT = ROOT / "results/v350-fuel-mix-capacity-factorial-r1"
TAU = 300
FACTOR_OUTER = (0, 1100)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = pd.read_csv(V341 / "monthly_margin_capacity_audit.csv")
    source = source.loc[source.included_in_complete_case.astype(bool)].copy()
    ember = pd.read_csv(EMBER)
    ember = ember.loc[
        ember.Province.notna()
        & ember.Province.ne("China (total)")
        & ember["Is aggregated source"].eq(False)
        & ember["Electricity source"].isin(
            ["Coal", "Gas", "Other fossil", "Bioenergy", "Hydro", "Nuclear", "Wind", "Solar"]
        )
    ].copy()
    ember["Province"] = ember.Province.replace({"Tibet (Xizang)": "Tibet"})
    ember["month"] = pd.to_datetime(ember.Date).dt.month
    ember["group"] = ember["Electricity source"].map(
        {"Coal": "coal", "Gas": "gas", "Other fossil": "other_fossil", "Bioenergy": "bioenergy",
         "Hydro": "hydro", "Nuclear": "nuclear", "Wind": "wind", "Solar": "solar"}
    )
    ember = ember.loc[pd.to_datetime(ember.Date).dt.year.eq(2023)]
    wide = ember.pivot(index=["Province", "month"], columns="group", values="Generation (TWh)").reset_index()

    mix_rows = []
    groups_cols = list(wide.columns.drop(["Province", "month"]))
    for province in sorted(wide.Province.unique()):
        sub = wide.loc[wide.Province.eq(province)]
        for month in range(1, 13):
            period = "Jan-Feb" if month == 1 else f"{month:02d}"
            block = sub.loc[sub.month.isin([1, 2]), groups_cols] if month == 1 else sub.loc[sub.month.eq(month), groups_cols]
            row = block.sum() if month == 1 else block.iloc[0]
            fuel_twh = {g: float(row[g]) for g in FUEL_NAMES}
            thermal_twh = sum(fuel_twh.values())
            shares = [fuel_twh[g] / thermal_twh for g in FUEL_NAMES] if thermal_twh else [0, 0, 0, 1]
            mix_rows.append({"province": province, "period": period, "shares": shares,
                             "thermal": F(str(thermal_twh)) * 1_000_000,
                             **{g: F(str(float(row[g]))) * 1_000_000 for g in ("hydro", "nuclear", "wind", "solar")}})
    mix = pd.DataFrame(mix_rows).set_index(["province", "period"])

    v348 = pd.read_csv(V348 / "province_period_share_error_frontier.csv")
    v348 = v348.loc[v348.relative_fuel_share_error.eq(1.0)]
    v348_map = v348.set_index(["province", "period", "arm"])["estimated_mix_width_h"]
    v349 = pd.read_csv(V349 / "paired_province_period_ablation.csv")
    v349_map = v349.set_index(["province", "period"])["ember_5tech_broad_ef_width_h"]

    records = []
    infeasible = []
    for rec in source.to_dict("records"):
        p, period, hours = rec["province"], rec["period"], int(rec["hours"])
        if (p, period) not in v349_map.index or (p, period, "lagged_2022_capacity") not in v348_map.index:
            continue
        mr = mix.loc[(p, period)]
        margins = {"thermal": mr.thermal, "hydro": mr.hydro, "nuclear": mr.nuclear,
                   "wind": mr.wind, "solar": mr.solar, "unresolved": F(0)}
        capacities_none = {g: max(F(1), margins[g]) for g in GROUPS}
        capacities_lagged = {g: F(str(rec[f"2022_{g}_cap_mw"])) for g in GROUPS}
        ef_low = {g: F(0) for g in GROUPS}
        ef_high = {g: F(0) for g in GROUPS}
        ef_low["thermal"] = F(FACTOR_OUTER[0])
        ef_high["thermal"] = F(FACTOR_OUTER[1])

        no_cap_lo = sharp_group_set(margins, capacities_none, ef_low, TAU, hours)
        no_cap_hi = sharp_group_set(margins, capacities_none, ef_high, TAU, hours)
        cap_lo = sharp_group_set(margins, capacities_lagged, ef_low, TAU, hours)
        cap_hi = sharp_group_set(margins, capacities_lagged, ef_high, TAU, hours)
        if any(x is None for x in (no_cap_lo, no_cap_hi, cap_lo, cap_hi)):
            infeasible.append({"province": p, "period": period})
            continue

        broad_no_cap = max(no_cap_hi) - min(no_cap_lo)
        broad_cap = max(cap_hi) - min(cap_lo)
        mix_no_cap = float(v348_map.loc[(p, period, "monthly_only")])
        mix_cap = float(v348_map.loc[(p, period, "lagged_2022_capacity")])
        if broad_no_cap + 1e-9 < mix_no_cap or broad_cap + 1e-9 < mix_cap:
            raise ValueError(f"fuel-mix K hull escaped its common outer EF support: {p} {period}")
        records.append({"province": p, "period": period, "hours": hours,
                        "broad_no_capacity_width_h": broad_no_cap,
                        "broad_with_capacity_width_h": broad_cap,
                        "mix_pm100_no_capacity_width_h": mix_no_cap,
                        "mix_pm100_with_capacity_width_h": mix_cap,
                        "mix_gain_no_capacity_h": broad_no_cap-mix_no_cap,
                        "mix_gain_with_capacity_h": broad_cap-mix_cap,
                        "capacity_gain_broad_h": broad_no_cap-broad_cap,
                        "capacity_gain_mix_h": mix_no_cap-mix_cap,
                        "mix_capacity_complementarity_h": (broad_cap-mix_cap)-(broad_no_cap-mix_no_cap)})

    out = pd.DataFrame(records)
    if len(out) != 328 or out.duplicated(["province", "period"]).any():
        raise ValueError(f"unexpected matched cohort: {len(out)}")
    frontier = {"province_periods": len(out),
                **{f"mean_{c}": float(out[c].mean()) for c in (
                    "broad_no_capacity_width_h", "broad_with_capacity_width_h",
                    "mix_pm100_no_capacity_width_h", "mix_pm100_with_capacity_width_h",
                    "mix_gain_no_capacity_h", "mix_gain_with_capacity_h",
                    "capacity_gain_broad_h", "capacity_gain_mix_h",
                    "mix_capacity_complementarity_h")},
                "median_mix_capacity_complementarity_h": float(out.mix_capacity_complementarity_h.median()),
                "cells_mix_gain_no_capacity_positive": int(out.mix_gain_no_capacity_h.gt(0).sum()),
                "cells_mix_gain_with_capacity_positive": int(out.mix_gain_with_capacity_h.gt(0).sum()),
                "cells_positive_complementarity": int(out.mix_capacity_complementarity_h.gt(0).sum()),
                "cells_negative_complementarity": int(out.mix_capacity_complementarity_h.lt(0).sum())}
    OUT.mkdir(parents=True)
    out.to_csv(OUT / "paired_factorial_province_period.csv", index=False)
    pd.DataFrame(infeasible).to_csv(OUT / "capacity_infeasible_cells.csv", index=False)
    summary = {"status": "COMPLETED_FACTORIAL_INFORMATION_VALUE_NOT_EMPIRICAL_VALIDATION",
               "tau_g_kwh": TAU, "fuel_share_relative_error": 1.0,
               "thermal_ef_outer_support": FACTOR_OUTER, "frontier": frontier,
               "limits": ["All fuel margins are based on estimated Ember provincial monthly generation.",
                          "The ±100% share error band is a stress scenario, not an error distribution estimate.",
                          "No China hourly CIF labels or empirical interval coverage are used.",
                          "The with-capacity arm uses lagged 2022 capacity as a retrospective proxy; Tibet infeasible cells are excluded."],
               "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                   V341 / "monthly_margin_capacity_audit.csv", V348 / "province_period_share_error_frontier.csv",
                   V349 / "paired_province_period_ablation.csv", EMBER, PROTOCOL, Path(__file__))}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(frontier, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
