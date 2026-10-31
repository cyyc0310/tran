"""V352: threshold robustness of fuel-mix × capacity complementarity."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import GROUPS, ROOT, sharp_group_set
from v348_ember_share_error_frontier import FUEL_NAMES, robust_ef_bounds

V341 = ROOT / "results/v341-china-complete-category-hull-r2"
V349 = ROOT / "results/v349-ember-total-vs-mix-ablation-r2"
EMBER = ROOT / "results/v294-china-ember-lineage-r1/ember_china_monthly_generation.csv"
PROTOCOL = ROOT / "docs/V352_THRESHOLD_FACTORIAL_SWEEP_PROTOCOL.md"
OUT = ROOT / "results/v352-threshold-factorial-sweep-r1"
TAUS = (100, 300, 500)
SHARE_ERROR = 1.0
OUTER_EF = (0, 1100)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = pd.read_csv(V341 / "monthly_margin_capacity_audit.csv")
    source = source.loc[source.included_in_complete_case.astype(bool)].copy()
    paired_keys = set(map(tuple, pd.read_csv(V349 / "paired_province_period_ablation.csv")[["province", "period"]].itertuples(index=False, name=None)))
    ember = pd.read_csv(EMBER)
    ember = ember.loc[
        ember.Province.notna() & ember.Province.ne("China (total)")
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
    fuel_data = []
    columns = list(wide.columns.drop(["Province", "month"]))
    for province in sorted(wide.Province.unique()):
        sub = wide.loc[wide.Province.eq(province)]
        for month in range(1, 13):
            period = "Jan-Feb" if month == 1 else f"{month:02d}"
            block = sub.loc[sub.month.isin([1, 2]), columns] if month == 1 else sub.loc[sub.month.eq(month), columns]
            row = block.sum() if month == 1 else block.iloc[0]
            fuels = {g: float(row[g]) for g in FUEL_NAMES}
            thermal = sum(fuels.values())
            shares = [fuels[g] / thermal for g in FUEL_NAMES] if thermal else [0, 0, 0, 1]
            fuel_data.append({"province": province, "period": period, "shares": shares,
                              "thermal": F(str(thermal))*1_000_000,
                              **{g: F(str(float(row[g])))*1_000_000 for g in ("hydro", "nuclear", "wind", "solar")}})
    mix = pd.DataFrame(fuel_data).set_index(["province", "period"])
    records, infeasible = [], []
    for rec in source.to_dict("records"):
        p, period, hours = rec["province"], rec["period"], int(rec["hours"])
        if (p, period) not in paired_keys:
            continue
        mr = mix.loc[(p, period)]
        margins = {"thermal": mr.thermal, "hydro": mr.hydro, "nuclear": mr.nuclear,
                   "wind": mr.wind, "solar": mr.solar, "unresolved": F(0)}
        caps_no = {g: max(F(1), margins[g]) for g in GROUPS}
        caps_lag = {g: F(str(rec[f"2022_{g}_cap_mw"])) for g in GROUPS}
        ef_low_mix, ef_high_mix = robust_ef_bounds(list(mr.shares), SHARE_ERROR)
        ef_bounds = [(F(OUTER_EF[0]), F(OUTER_EF[1])), (F(str(ef_low_mix)), F(str(ef_high_mix)))]
        for tau in TAUS:
            widths: dict[tuple[str, str], int] = {}
            for info, (lo_ef, hi_ef) in zip(("broad", "mix"), ef_bounds):
                elo = {g: F(0) for g in GROUPS}; ehi = {g: F(0) for g in GROUPS}
                elo["thermal"], ehi["thermal"] = lo_ef, hi_ef
                for cap_name, caps in (("no_capacity", caps_no), ("with_capacity", caps_lag)):
                    lo_set = sharp_group_set(margins, caps, elo, tau, hours)
                    hi_set = sharp_group_set(margins, caps, ehi, tau, hours)
                    if lo_set is None or hi_set is None:
                        widths[(info, cap_name)] = -1
                    else:
                        widths[(info, cap_name)] = max(hi_set)-min(lo_set)
            if any(value < 0 for value in widths.values()):
                infeasible.append({"province": p, "period": period, "tau_g_kwh": tau})
                continue
            broad_no, broad_cap = widths[("broad", "no_capacity")], widths[("broad", "with_capacity")]
            mix_no, mix_cap = widths[("mix", "no_capacity")], widths[("mix", "with_capacity")]
            if mix_no > broad_no or mix_cap > broad_cap:
                raise ValueError(f"mix set escaped common EF support: {p} {period} tau={tau}")
            records.append({"province": p, "period": period, "hours": hours, "tau_g_kwh": tau,
                            "broad_no_capacity_width_h": broad_no, "broad_with_capacity_width_h": broad_cap,
                            "mix_pm100_no_capacity_width_h": mix_no, "mix_pm100_with_capacity_width_h": mix_cap,
                            "mix_gain_no_capacity_h": broad_no-mix_no,
                            "mix_gain_with_capacity_h": broad_cap-mix_cap,
                            "complementarity_h": (broad_cap-mix_cap)-(broad_no-mix_no)})
    out = pd.DataFrame(records)
    if out.duplicated(["province", "period", "tau_g_kwh"]).any():
        raise ValueError("duplicate paired threshold records")
    summaries = []
    for tau, frame in out.groupby("tau_g_kwh"):
        summaries.append({"tau_g_kwh": int(tau), "province_periods": len(frame),
                          **{f"mean_{c}": float(frame[c].mean()) for c in (
                              "broad_no_capacity_width_h", "broad_with_capacity_width_h",
                              "mix_pm100_no_capacity_width_h", "mix_pm100_with_capacity_width_h",
                              "mix_gain_no_capacity_h", "mix_gain_with_capacity_h", "complementarity_h")},
                          "median_complementarity_h": float(frame.complementarity_h.median()),
                          "positive_complementarity_cells": int(frame.complementarity_h.gt(0).sum()),
                          "negative_complementarity_cells": int(frame.complementarity_h.lt(0).sum())})
    OUT.mkdir(parents=True)
    out.to_csv(OUT / "paired_threshold_factorial.csv", index=False)
    pd.DataFrame(infeasible).to_csv(OUT / "infeasible_cells.csv", index=False)
    summary = {"status": "COMPLETED_THRESHOLD_ROBUSTNESS_NOT_EMPIRICAL_VALIDATION",
               "share_error_relative": SHARE_ERROR, "outer_thermal_ef": OUTER_EF,
               "frontier": summaries, "infeasible_rows": len(infeasible),
               "limits": ["Ember estimated margins and analyst-defined ±100% share stress band.",
                          "Lagged 2022 capacity is a retrospective proxy.",
                          "No hourly labels or empirical coverage are used."],
               "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                   V341 / "monthly_margin_capacity_audit.csv", V349 / "paired_province_period_ablation.csv",
                   EMBER, PROTOCOL, Path(__file__))}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"infeasible_rows={len(infeasible)}")


if __name__ == "__main__":
    main()
