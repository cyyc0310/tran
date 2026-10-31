"""V348: stress V347's information value against Ember fuel-share error."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd
from scipy.optimize import linprog

from v342_china_sharp_group_hull import GROUPS, ROOT, sharp_group_set

V341 = ROOT / "results/v341-china-complete-category-hull-r2"
V346 = ROOT / "results/v346-scope-residual-k-hulls-r2"
EMBER = ROOT / "results/v294-china-ember-lineage-r1/ember_china_monthly_generation.csv"
PROTOCOL = ROOT / "docs/V348_EMBER_SHARE_ERROR_FRONTIER_PROTOCOL.md"
OUT = ROOT / "results/v348-ember-share-error-frontier-r1"
TAU = 300
ARMS = ("monthly_only", "lagged_2022_capacity")
ERROR_BANDS = (0.0, 0.10, 0.25, 0.50, 1.0)
FUEL_NAMES = ("coal", "gas", "other_fossil", "bioenergy")
SOURCE_TO_GROUP = {"Coal": "coal", "Gas": "gas", "Other fossil": "other_fossil", "Bioenergy": "bioenergy"}
EF_LOW = {"coal": 900.0, "gas": 360.0, "other_fossil": 200.0, "bioenergy": 0.0}
EF_HIGH = {"coal": 1100.0, "gas": 440.0, "other_fossil": 600.0, "bioenergy": 0.0}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def robust_ef_bounds(shares: list[float], relative_error: float) -> tuple[float, float]:
    lower = [max(0.0, s * (1 - relative_error)) for s in shares]
    upper = [min(1.0, s * (1 + relative_error)) for s in shares]
    if sum(lower) > 1 + 1e-10 or sum(upper) < 1 - 1e-10:
        raise ValueError("share uncertainty box does not intersect the simplex")
    A_eq, b_eq = [[1.0] * len(shares)], [1.0]
    lo_res = linprog([EF_LOW[g] for g in FUEL_NAMES], A_eq=A_eq, b_eq=b_eq,
                     bounds=list(zip(lower, upper)), method="highs")
    hi_res = linprog([-EF_HIGH[g] for g in FUEL_NAMES], A_eq=A_eq, b_eq=b_eq,
                     bounds=list(zip(lower, upper)), method="highs")
    if not lo_res.success or not hi_res.success:
        raise ValueError("fuel-share robust EF LP failed")
    return float(lo_res.fun), float(-hi_res.fun)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = pd.read_csv(V341 / "monthly_margin_capacity_audit.csv")
    source = source.loc[source.included_in_complete_case.astype(bool)].copy()
    ember = pd.read_csv(EMBER)
    ember = ember.loc[ember.Province.notna() & ember.Province.ne("China (total)") &
                      (ember["Is aggregated source"] == False) &
                      ember["Electricity source"].isin(["Coal", "Gas", "Other fossil", "Bioenergy", "Hydro", "Nuclear", "Wind", "Solar"])].copy()
    ember["Province"] = ember["Province"].replace({"Tibet (Xizang)": "Tibet"})
    ember["month"] = pd.to_datetime(ember.Date).dt.month
    ember["group"] = ember["Electricity source"].map({"Coal":"coal","Gas":"gas","Other fossil":"other_fossil","Bioenergy":"bioenergy",
                                                       "Hydro":"hydro","Nuclear":"nuclear","Wind":"wind","Solar":"solar"})
    ember = ember.loc[pd.to_datetime(ember.Date).dt.year.eq(2023)]
    wide = ember.pivot(index=["Province", "month"], columns="group", values="Generation (TWh)").reset_index()
    records = []
    for province in sorted(wide.Province.unique()):
        sub = wide.loc[wide.Province.eq(province)]
        for month in range(1, 13):
            period = "Jan-Feb" if month == 1 else f"{month:02d}"
            row = sub.loc[sub.month.isin([1, 2]), list(wide.columns.drop(["Province", "month"]))].sum() if month == 1 else sub.loc[sub.month.eq(month), list(wide.columns.drop(["Province", "month"]))].iloc[0]
            totals = {g: float(row[g]) for g in FUEL_NAMES}
            thermal = sum(totals.values())
            shares = [totals[g] / thermal for g in FUEL_NAMES] if thermal > 0 else [0.0, 0.0, 0.0, 1.0]
            tech = {g: float(row[g]) for g in ("hydro", "nuclear", "wind", "solar")}
            records.append({"province": province, "period": period, "fuel_shares": shares,
                            "thermal_twh": thermal, "hydro_twh": tech["hydro"], "nuclear_twh": tech["nuclear"],
                            "wind_twh": tech["wind"], "solar_twh": tech["solar"]})
    mix = pd.DataFrame(records).set_index(["province", "period"])
    base_hulls = pd.read_csv(V346 / "province_period_k_hulls.csv")
    rows, infeasible = [], []
    for rec in source.to_dict("records"):
        p, period, hours = rec["province"], rec["period"], int(rec["hours"])
        mr = mix.loc[(p, period)]
        margins = {"thermal": F(str(mr.thermal_twh)) * 1_000_000,
                   "hydro": F(str(mr.hydro_twh)) * 1_000_000,
                   "nuclear": F(str(mr.nuclear_twh)) * 1_000_000,
                   "wind": F(str(mr.wind_twh)) * 1_000_000,
                   "solar": F(str(mr.solar_twh)) * 1_000_000, "unresolved": F(0)}
        for arm in ARMS:
            vintage = 2022 if arm == "lagged_2022_capacity" else None
            if arm == "monthly_only":
                caps = {g: max(F(1), margins[g]) for g in GROUPS}
            else:
                caps = {g: F(str(rec[f"2022_{g}_cap_mw"])) for g in GROUPS}
            for relerr in ERROR_BANDS:
                eflo, efhi = robust_ef_bounds(list(mr.fuel_shares), relerr)
                lowef = {g: 0 for g in GROUPS}; lowef["thermal"] = F(str(eflo))
                highef = {g: 0 for g in GROUPS}; highef["thermal"] = F(str(efhi))
                lowset = sharp_group_set(margins, caps, lowef, TAU, hours)
                highset = sharp_group_set(margins, caps, highef, TAU, hours)
                if lowset is None or highset is None:
                    infeasible.append({"province": p, "period": period, "arm": arm, "relative_share_error": relerr})
                    continue
                lower, upper = min(lowset), max(highset)
                base = base_hulls.loc[base_hulls.province.eq(p) & base_hulls.period.eq(period) &
                                      base_hulls.tau_g_kwh.eq(TAU) & base_hulls.arm.eq(arm)]
                if len(base) != 1:
                    raise ValueError("missing matched V346 comparison")
                bw = int(base.iloc[0].expanded_width)
                if lower < 0 or upper > hours:
                    raise ValueError("K hull outside period")
                rows.append({"province": p, "period": period, "hours": hours, "arm": arm,
                             "relative_fuel_share_error": relerr, "thermal_ef_low": eflo, "thermal_ef_high": efhi,
                             "nbs_residual_width_h": bw, "estimated_mix_width_h": upper-lower,
                             "width_reduction_h": bw-(upper-lower), "lower_k": lower, "upper_k": upper})
    out = pd.DataFrame(rows)
    if out.duplicated(["province", "period", "arm", "relative_fuel_share_error"]).any():
        raise ValueError("duplicate V348 result keys")
    frontier = []
    for (arm, relerr), frame in out.groupby(["arm", "relative_fuel_share_error"]):
        frontier.append({"arm": arm, "relative_fuel_share_error": float(relerr), "province_periods": len(frame),
                         "mean_nbs_residual_width_h": float(frame.nbs_residual_width_h.mean()),
                         "mean_estimated_mix_width_h": float(frame.estimated_mix_width_h.mean()),
                         "mean_reduction_h": float(frame.width_reduction_h.mean()),
                         "median_reduction_h": float(frame.width_reduction_h.median()),
                         "cells_narrower": int(frame.width_reduction_h.gt(0).sum())})
    OUT.mkdir(parents=True)
    out.to_csv(OUT / "province_period_share_error_frontier.csv", index=False)
    pd.DataFrame(infeasible).to_csv(OUT / "capacity_infeasible_cells.csv", index=False)
    summary = {"status": "SENSITIVITY_FRONTIER_FOR_ESTIMATED_FUEL_SHARE_ERROR_NOT_EMPIRICAL_ERROR_CALIBRATION",
               "tau_g_kwh": TAU, "relative_share_error_bands": ERROR_BANDS,
               "frontier": frontier, "capacity_infeasible_rows": len(infeasible),
               "limits": ["Share error bands are analyst-selected perturbation scenarios, not measured Ember error distributions.",
                          "Ember source mix remains estimated; no local hourly label validates the K hull.",
                          "NBS residual comparator includes broader scope uncertainty; Ember uses a different estimated monthly boundary.",
                          "2022 capacity arm is retrospective proxy information and some Ember allocations can be capacity-infeasible."],
               "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (V341 / "monthly_margin_capacity_audit.csv",
                          V346 / "province_period_k_hulls.csv", EMBER, PROTOCOL, Path(__file__))}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(frontier).to_string(index=False))
    print(f"rows={len(out)}; capacity-infeasible rows={len(infeasible)}")


if __name__ == "__main__":
    main()
