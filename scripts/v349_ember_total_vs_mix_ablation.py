"""V349: separate complete monthly-total value from fuel-mix value."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import GROUPS, ROOT, sharp_group_set

V341 = ROOT / "results/v341-china-complete-category-hull-r2"
V346 = ROOT / "results/v346-scope-residual-k-hulls-r2"
V348 = ROOT / "results/v348-ember-share-error-frontier-r1"
EMBER = ROOT / "results/v294-china-ember-lineage-r1/ember_china_monthly_generation.csv"
PROTOCOL = ROOT / "docs/V349_EMBER_TOTAL_VS_MIX_ABLATION_PROTOCOL.md"
OUT = ROOT / "results/v349-ember-total-vs-mix-ablation-r2"
TAU = 300
# V348's fuel-level EF bands span [0, 1100] g/kWh after fuel-share mixing.
# Use the same outer support in the technology-collapsed control so fuel mix
# does not gain an unfair advantage merely from a wider EF interval.
FACTORS = (0, 1100)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = pd.read_csv(V341 / "monthly_margin_capacity_audit.csv")
    source = source.loc[source.included_in_complete_case.astype(bool)].copy()
    emb = pd.read_csv(EMBER)
    emb = emb.loc[emb.Province.notna() & emb.Province.ne("China (total)") &
                  (emb["Is aggregated source"] == False) &
                  emb["Electricity source"].isin(["Coal", "Gas", "Other fossil", "Bioenergy", "Hydro", "Nuclear", "Wind", "Solar"])].copy()
    emb["Province"] = emb["Province"].replace({"Tibet (Xizang)": "Tibet"})
    emb["month"] = pd.to_datetime(emb.Date).dt.month
    emb["group"] = emb["Electricity source"].map({"Coal":"coal","Gas":"gas","Other fossil":"other_fossil","Bioenergy":"bioenergy",
                                                  "Hydro":"hydro","Nuclear":"nuclear","Wind":"wind","Solar":"solar"})
    emb = emb.loc[pd.to_datetime(emb.Date).dt.year.eq(2023)]
    wide = emb.pivot(index=["Province", "month"], columns="group", values="Generation (TWh)").reset_index()
    mix_rows = []
    for province in sorted(wide.Province.unique()):
        sub = wide.loc[wide.Province.eq(province)]
        for month in range(1, 13):
            period = "Jan-Feb" if month == 1 else f"{month:02d}"
            row = sub.loc[sub.month.isin([1, 2]), list(wide.columns.drop(["Province", "month"]))].sum() if month == 1 else sub.loc[sub.month.eq(month), list(wide.columns.drop(["Province", "month"]))].iloc[0]
            thermal = sum(float(row[g]) for g in ("coal", "gas", "other_fossil", "bioenergy"))
            mix_rows.append({"province": province, "period": period,
                             "thermal": F(str(thermal))*1_000_000,
                             **{g: F(str(float(row[g])))*1_000_000 for g in ("hydro", "nuclear", "wind", "solar")}})
    mix = pd.DataFrame(mix_rows).set_index(["province", "period"])
    residual_compare = pd.read_csv(V346 / "province_period_k_hulls.csv")
    share_compare = pd.read_csv(V348 / "province_period_share_error_frontier.csv")
    share_compare = share_compare.loc[share_compare.relative_fuel_share_error.eq(1.0) & share_compare.arm.eq("lagged_2022_capacity")]
    share_map = share_compare.set_index(["province", "period"]).estimated_mix_width_h
    rows, infeasible = [], []
    for rec in source.to_dict("records"):
        p, period, hours = rec["province"], rec["period"], int(rec["hours"])
        er = mix.loc[(p, period)]
        margins = {"thermal": er.thermal, "hydro": er.hydro, "nuclear": er.nuclear,
                   "wind": er.wind, "solar": er.solar, "unresolved": F(0)}
        caps = {g: F(str(rec[f"2022_{g}_cap_mw"])) for g in GROUPS}
        eflo = {g: 0 for g in GROUPS}; eflo["thermal"] = FACTORS[0]
        efhi = {g: 0 for g in GROUPS}; efhi["thermal"] = FACTORS[1]
        lower_set = sharp_group_set(margins, caps, eflo, TAU, hours)
        upper_set = sharp_group_set(margins, caps, efhi, TAU, hours)
        if lower_set is None or upper_set is None:
            infeasible.append({"province": p, "period": period})
            continue
        base = residual_compare.loc[residual_compare.province.eq(p) & residual_compare.period.eq(period) &
                                    residual_compare.tau_g_kwh.eq(TAU) & residual_compare.arm.eq("lagged_2022_capacity")]
        if len(base) != 1:
            raise ValueError("missing V346 baseline")
        if (p, period) not in share_map.index:
            raise ValueError("missing V348 robust mix comparison")
        width5 = max(upper_set)-min(lower_set)
        rows.append({"province": p, "period": period, "hours": hours,
                     "nbs_plus_residual_width_h": int(base.iloc[0].expanded_width),
                     "ember_5tech_broad_ef_width_h": width5,
                     "ember_fuel_mix_pm100_width_h": float(share_map.loc[(p, period)]),
                     "residual_to_5tech_reduction_h": int(base.iloc[0].expanded_width)-width5,
                     "mix_gain_beyond_5tech_h": width5-float(share_map.loc[(p, period)])})
    out = pd.DataFrame(rows)
    if len(out) != 328 or out.duplicated(["province", "period"]).any():
        raise ValueError(f"paired V349 cohort mismatch: {len(out)}")
    frontier = {"province_periods": len(out),
                "mean_nbs_residual_width_h": float(out.nbs_plus_residual_width_h.mean()),
                "mean_ember_5tech_broad_ef_width_h": float(out.ember_5tech_broad_ef_width_h.mean()),
                "mean_ember_fuel_mix_pm100_width_h": float(out.ember_fuel_mix_pm100_width_h.mean()),
                "mean_gain_from_complete_estimated_total_h": float(out.residual_to_5tech_reduction_h.mean()),
                "mean_increment_from_fuel_mix_beyond_5tech_h": float(out.mix_gain_beyond_5tech_h.mean()),
                "median_increment_from_fuel_mix_beyond_5tech_h": float(out.mix_gain_beyond_5tech_h.median()),
                "cells_mix_narrower_than_5tech": int(out.mix_gain_beyond_5tech_h.gt(0).sum()),
                "capacity_infeasible_cells": len(infeasible)}
    OUT.mkdir(parents=True)
    out.to_csv(OUT / "paired_province_period_ablation.csv", index=False)
    pd.DataFrame(infeasible).to_csv(OUT / "capacity_infeasible_cells.csv", index=False)
    summary = {"status": "COMPLETE_TOTAL_VS_FUEL_MIX_VALUE_ABLATION_NOT_VALIDATION", "tau_g_kwh": TAU,
               "factor_support_5tech": {"thermal": FACTORS, "other_technologies": 0},
               "frontier": frontier,
               "limits": ["Ember complete monthly generation and fuel composition are estimated and differ from NBS above-size scope.",
                          "This ablation separates total/scope information from added fuel-share information only within the declared K model.",
                          "The technology-collapsed thermal EF support [0,1100] is the outer envelope of V348 fuel-level EF scenarios, not a measured local distribution.",
                          "No hourly China labels or empirical K coverage are used."],
               "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (V341 / "monthly_margin_capacity_audit.csv",
                          V346 / "province_period_k_hulls.csv", V348 / "province_period_share_error_frontier.csv",
                          EMBER, PROTOCOL, Path(__file__))}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(frontier, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
