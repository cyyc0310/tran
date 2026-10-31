"""V347: information-value sensitivity for an estimated complete monthly fuel mix."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import GROUPS, ROOT, sharp_group_set

V341 = ROOT / "results/v341-china-complete-category-hull-r2"
V346 = ROOT / "results/v346-scope-residual-k-hulls-r2"
EMBER = ROOT / "results/v294-china-ember-lineage-r1/ember_china_monthly_generation.csv"
CAPACITY = ROOT / "data/china/cec_province_installed_capacity_2022_2023.csv"
PROTOCOL = ROOT / "docs/V347_EMBER_COMPLETE_MIX_VALUE_PROTOCOL.md"
OUT = ROOT / "results/v347-ember-complete-mix-value-r1"
TAUS = (100, 300)
SOURCE_TO_GROUP = {"Coal": "coal", "Gas": "gas", "Other fossil": "other_fossil", "Bioenergy": "bioenergy",
                   "Hydro": "hydro", "Nuclear": "nuclear", "Wind": "wind", "Solar": "solar"}
FACTOR_BANDS = {"coal": (900, 1100), "gas": (360, 440), "other_fossil": (200, 600),
                "bioenergy": (0, 0), "hydro": (0, 0), "nuclear": (0, 0), "wind": (0, 0), "solar": (0, 0)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = pd.read_csv(V341 / "monthly_margin_capacity_audit.csv")
    source = source.loc[source.included_in_complete_case.astype(bool)].copy()
    ember = pd.read_csv(EMBER)
    ember = ember.loc[ember["Province"].notna() & ember["Province"].ne("China (total)") &
                      (ember["Is aggregated source"] == False) &
                      ember["Electricity source"].isin(SOURCE_TO_GROUP)].copy()
    ember["month"] = pd.to_datetime(ember["Date"]).dt.month
    ember["Province"] = ember["Province"].replace({"Tibet (Xizang)": "Tibet"})
    ember["group"] = ember["Electricity source"].map(SOURCE_TO_GROUP)
    ember = ember.loc[pd.to_datetime(ember["Date"]).dt.year.eq(2023)]
    if ember.duplicated(["Province", "month", "group"]).any():
        raise ValueError("duplicate Ember province-month-group rows")
    wide = ember.pivot(index=["Province", "month"], columns="group", values="Generation (TWh)").reset_index()
    required = set(SOURCE_TO_GROUP.values())
    if not required.issubset(wide.columns) or len(wide) != 31 * 12 or wide[list(required)].isna().any().any():
        raise ValueError("Ember 2023 atomic fuel panel is incomplete")
    # Match NBS's Jan-Feb cumulative bin by summing both Ember months.
    records = []
    for province in sorted(wide.Province.unique()):
        sub = wide.loc[wide.Province.eq(province)]
        for month in range(1, 13):
            period = "Jan-Feb" if month == 1 else f"{month:02d}"
            if month == 1:
                row = sub.loc[sub.month.isin([1, 2]), list(required)].sum()
            else:
                row = sub.loc[sub.month.eq(month), list(required)].iloc[0]
            vals = {g: F(str(float(row[g]))) * 1_000_000 for g in required}  # TWh -> MWh
            thermal = sum((vals[g] for g in ("coal", "gas", "other_fossil", "bioenergy")), F(0))
            if thermal <= 0:
                ef_low = ef_high = F(0)
            else:
                ef_low = sum((vals[g] * FACTOR_BANDS[g][0] for g in ("coal", "gas", "other_fossil", "bioenergy")), F(0)) / thermal
                ef_high = sum((vals[g] * FACTOR_BANDS[g][1] for g in ("coal", "gas", "other_fossil", "bioenergy")), F(0)) / thermal
            records.append({"province": province, "period": period, **{f"{g}_mwh": vals[g] for g in required},
                            "total_mwh": sum(vals.values(), F(0)), "thermal_ef_low": ef_low, "thermal_ef_high": ef_high})
    em = pd.DataFrame(records)
    if em.duplicated(["province", "period"]).any():
        raise ValueError("duplicate Ember matched bins")
    em_map = em.set_index(["province", "period"])
    source_keys = set(zip(source.province, source.period))
    if not source_keys.issubset(set(em_map.index)):
        raise ValueError("V341 cohort not covered by Ember")
    baseline = pd.read_csv(V346 / "province_period_k_hulls.csv")
    if len(baseline) != 330 * len(TAUS) * 3:
        raise ValueError("V346 baseline cohort mismatch")

    rows, infeasible_rows = [], []
    for rec in source.to_dict("records"):
        province, period, hours = rec["province"], rec["period"], int(rec["hours"])
        er = em_map.loc[(province, period)]
        atomic = {g: F(str(er[f"{g}_mwh"])) for g in SOURCE_TO_GROUP.values()}
        # Collapse Ember's estimated fossil mix into NBS's thermal group. This
        # keeps the same V342 solver and preserves fuel-factor uncertainty as a
        # monthly composition-weighted interval, not a learned probability.
        tech_margins = {"thermal": sum((atomic[g] for g in ("coal", "gas", "other_fossil", "bioenergy")), F(0)),
                        "hydro": atomic["hydro"], "nuclear": atomic["nuclear"],
                        "wind": atomic["wind"], "solar": atomic["solar"], "unresolved": F(0)}
        tech_caps = {"monthly_only": {g: max(F(1), tech_margins[g]) for g in GROUPS},
                     "lagged_2022_capacity": {g: F(str(rec[f"2022_{g}_cap_mw"])) for g in GROUPS},
                     "retrospective_2023_capacity": {g: F(str(rec[f"2023_{g}_cap_mw"])) for g in GROUPS}}
        for tau in TAUS:
            for arm in tech_caps:
                low_ef = {g: 0 for g in GROUPS}; low_ef["thermal"] = er.thermal_ef_low
                high_ef = {g: 0 for g in GROUPS}; high_ef["thermal"] = er.thermal_ef_high
                lowset = sharp_group_set(tech_margins, tech_caps[arm], low_ef, tau, hours)
                highset = sharp_group_set(tech_margins, tech_caps[arm], high_ef, tau, hours)
                if lowset is None or highset is None:
                    infeasible_rows.append({"province": province, "period": period, "tau_g_kwh": tau,
                                            "arm": arm, "reason": "estimated Ember monthly group energy exceeds vintage capacity room"})
                    continue
                lower, upper = min(lowset), max(highset)
                base = baseline.loc[baseline.province.eq(province) & baseline.period.eq(period) &
                                    baseline.tau_g_kwh.eq(tau) & baseline.arm.eq(arm)]
                if len(base) != 1:
                    raise ValueError("missing V346 baseline row")
                b = base.iloc[0]
                rows.append({"province": province, "period": period, "hours": hours, "tau_g_kwh": tau,
                             "arm": arm, "ember_total_mwh": float(er.total_mwh),
                             "nbs_reported_total_mwh": float(rec["total_mwh"]),
                             "ember_vs_nbs_total_ratio": float(er.total_mwh / F(str(rec["total_mwh"]))),
                             "thermal_ef_low": float(er.thermal_ef_low), "thermal_ef_high": float(er.thermal_ef_high),
                             "nbs_plus_residual_width_h": int(b.expanded_width),
                             "ember_estimated_mix_width_h": upper-lower,
                             "width_reduction_h": int(b.expanded_width)-(upper-lower),
                             "ember_lower": lower, "ember_upper": upper})
    out = pd.DataFrame(rows)
    if out.empty or out.duplicated(["province", "period", "tau_g_kwh", "arm"]).any():
        raise ValueError("V347 result key mismatch")
    frontier = []
    for (tau, arm), frame in out.groupby(["tau_g_kwh", "arm"]):
        frontier.append({"tau_g_kwh": int(tau), "arm": arm, "province_periods": len(frame),
                         "unique_capacity_feasible_province_periods": int(frame[["province", "period"]].drop_duplicates().shape[0]),
                         "mean_nbs_residual_width_h": float(frame.nbs_plus_residual_width_h.mean()),
                         "mean_ember_estimated_mix_width_h": float(frame.ember_estimated_mix_width_h.mean()),
                         "mean_width_reduction_h": float(frame.width_reduction_h.mean()),
                         "median_width_reduction_h": float(frame.width_reduction_h.median()),
                         "cells_narrower": int(frame.width_reduction_h.gt(0).sum()),
                         "mean_ember_to_nbs_total_ratio": float(frame.ember_vs_nbs_total_ratio.mean())})
    OUT.mkdir(parents=True)
    out.to_csv(OUT / "province_period_information_value.csv", index=False)
    pd.DataFrame(infeasible_rows).to_csv(OUT / "capacity_infeasible_cells.csv", index=False)
    summary = {"status": "ESTIMATED_MONTHLY_FUEL_MIX_VALUE_SENSITIVITY_NOT_INDEPENDENT_VALIDATION",
               "cohort": "V341 complete 30-province × 11-period panel",
               "factor_bands_g_kwh": FACTOR_BANDS,
               "frontier": frontier,
               "capacity_infeasible_rows": len(infeasible_rows),
               "capacity_infeasible_unique_province_periods_by_arm": {arm: int(pd.DataFrame(infeasible_rows).loc[pd.DataFrame(infeasible_rows).arm.eq(arm), ["province", "period"]].drop_duplicates().shape[0]) if infeasible_rows else 0
                                                                        for arm in ("monthly_only", "lagged_2022_capacity", "retrospective_2023_capacity")},
               "limits": ["Ember monthly fuel categories are estimated from multiple sources and are not independent local fuel meter observations.",
                          "This is information-value in a conditional sharp-set model, not prediction skill or empirical coverage.",
                          "Ember total generation may not match NBS above-size reporting scope; the ratio is reported to expose this boundary difference.",
                          "Thermal EF is composition-weighted from broad scenario bands; no local factor is measured.",
                          "CEC capacity vintages and hourly coverage are retrospective/proxy constraints; no hourly China labels are used."],
               "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (V341 / "monthly_margin_capacity_audit.csv",
                          V346 / "province_period_k_hulls.csv", EMBER, CAPACITY, PROTOCOL, Path(__file__))}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(frontier).to_string(index=False))
    print(f"rows={len(out)}")


if __name__ == "__main__":
    main()
