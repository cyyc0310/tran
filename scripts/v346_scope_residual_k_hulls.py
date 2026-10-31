"""V346: K-hull sensitivity to the annual scope remainder at monthly resolution."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import GROUPS, ROOT, sharp_group_set

V341 = ROOT / "results/v341-china-complete-category-hull-r2"
V345 = ROOT / "results/v345-china-residual-monthly-scope-r1"
CAP = ROOT / "data/china/cec_province_installed_capacity_2022_2023.csv"
SCRIPT342 = ROOT / "scripts/v342_china_sharp_group_hull.py"
PROTOCOL = ROOT / "docs/V346_SCOPE_RESIDUAL_K_HULLS_PROTOCOL.md"
OUT = ROOT / "results/v346-scope-residual-k-hulls-r2"
TECHS = ("thermal", "hydro", "nuclear", "wind", "solar", "unresolved")
HOURS = {"Jan-Feb": 59 * 24, **{f"{m:02d}": (31 if m in (3, 5, 7, 8, 10, 12) else 30) * 24 for m in range(3, 13)}}
TAUS = (100, 300)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = pd.read_csv(V341 / "monthly_margin_capacity_audit.csv")
    source = source.loc[source.included_in_complete_case.astype(bool)].copy()
    residual = pd.read_csv(V345 / "province_period_residual_ranges.csv")
    if len(source) != 330:
        raise ValueError("source cohort mismatch")
    source_keys = set(zip(source.province, source.period))
    residual = residual.loc[[key in source_keys for key in zip(residual.province, residual.period)]].copy()
    if residual.duplicated(["province", "period"]).any():
        if residual.duplicated(["province", "period", "capacity_vintage"]).any():
            raise ValueError("duplicate V345 source keys")
    residual_map = residual.set_index(["province", "period", "capacity_vintage"])[["monthly_residual_min_gwh", "monthly_residual_max_gwh"]]
    if set(zip(source.province, source.period)) != set((p, month) for p, month, _ in residual_map.index):
        raise ValueError("V341/V345 cohort keys differ")

    baseline_rows, rows = [], []
    for rec in source.to_dict("records"):
        p, period, h = rec["province"], rec["period"], int(rec["hours"])
        if HOURS[period] != h:
            raise ValueError(f"hour count mismatch: {period}")
        margins = {g: F(str(rec[f"{g}_mwh"])) for g in ("thermal", "hydro", "nuclear", "wind", "solar")}
        cap_arms = {
            "monthly_only": {g: margins[g] for g in margins},
            "lagged_2022_capacity": {g: F(str(rec[f"2022_{g}_cap_mw"])) for g in margins},
            "retrospective_2023_capacity": {g: F(str(rec[f"2023_{g}_cap_mw"])) for g in margins},
        }
        for tau in TAUS:
            # V342 baseline capacity-conditioned K set, independent replication check.
            eflo = {g: 0 for g in TECHS}; eflo["thermal"] = 400
            efhi = {g: 0 for g in TECHS}; efhi["thermal"] = 1000
            for arm, caps in cap_arms.items():
                vintage = 2022 if arm == "lagged_2022_capacity" else 2023
                province_cap = F(str(rec[f"{vintage}_unresolved_cap_mw"]))
                rv = residual_map.loc[(p, period, vintage)]
                rlo, rhi = F(str(rv.monthly_residual_min_gwh)) * 1000, F(str(rv.monthly_residual_max_gwh)) * 1000
                fixed_margin = {**margins, "unresolved": F(0)}
                baseline_caps = {**caps, "unresolved": max(F(1), province_cap)}
                lowset = sharp_group_set(fixed_margin, baseline_caps, eflo, tau, h)
                highset = sharp_group_set(fixed_margin, baseline_caps, efhi, tau, h)
                if lowset is None or highset is None:
                    raise ValueError(f"baseline capacity infeasible {p} {period} {arm}")
                baseline_rows.append({"province": p, "period": period, "tau": tau, "arm": arm,
                                      "lower": min(lowset), "upper": max(highset), "width": max(highset)-min(lowset)})

            for arm, caps in cap_arms.items():
                # Evaluate the no-residual and maximum-residual endpoints. The
                # extremal tail count is monotone in residual mass at fixed EF:
                # zero-EF output can only dilute; high-EF output can only add
                # above-threshold hours. EF extrema are likewise monotone.
                endpoint_sets = []
                for residual_mwh in {rlo, rhi}:
                    for thermal_ef, residual_ef in ((400, 0), (1000, 1000)):
                        ef = {g: 0 for g in TECHS}
                        ef["thermal"] = thermal_ef
                        ef["unresolved"] = residual_ef
                        group_margins = {**margins, "unresolved": residual_mwh}
                        group_caps = {**caps, "unresolved": max(F(1), province_cap if arm != "monthly_only" else residual_mwh)}
                        feasible = sharp_group_set(group_margins, group_caps, ef, tau, h)
                        if feasible is None:
                            raise ValueError(f"endpoint infeasible: {p} {period} {arm} tau={tau} residual={residual_mwh}")
                        endpoint_sets.append(feasible)
                expanded_values = set().union(*(set(s) for s in endpoint_sets))
                lower, upper = min(expanded_values), max(expanded_values)
                base = next(x for x in baseline_rows if (x["province"], x["period"], x["tau"], x["arm"]) == (p, period, tau, arm))
                if lower > base["lower"] or upper < base["upper"]:
                    raise ValueError(f"expanded hull fails nesting: {p} {period} {arm} tau={tau}")
                rows.append({"province": p, "period": period, "hours": h, "tau_g_kwh": tau,
                             "arm": arm, "residual_min_gwh": float(rlo/1000), "residual_max_gwh": float(rhi/1000),
                             "baseline_lower": base["lower"], "baseline_upper": base["upper"], "baseline_width": base["width"],
                             "expanded_lower": lower, "expanded_upper": upper, "expanded_width": upper-lower,
                             "width_increase_hours": upper-lower-base["width"],
                             "endpoint_union_K_count": len(expanded_values)})
    out = pd.DataFrame(rows)
    if len(out) != 330 * len(TAUS) * 3 or out.duplicated(["province", "period", "tau_g_kwh", "arm"]).any():
        raise ValueError("output key/row count mismatch")
    summary_rows = []
    for (tau, arm), frame in out.groupby(["tau_g_kwh", "arm"]):
        summary_rows.append({"tau_g_kwh": int(tau), "arm": arm, "province_periods": len(frame),
                             "mean_baseline_width_h": float(frame.baseline_width.mean()),
                             "mean_expanded_width_h": float(frame.expanded_width.mean()),
                             "mean_increase_h": float(frame.width_increase_hours.mean()),
                             "median_increase_h": float(frame.width_increase_hours.median()),
                             "cells_with_positive_increase": int(frame.width_increase_hours.gt(0).sum()),
                             "cells_with_full_month_hull": int(frame.expanded_width.eq(frame.hours).sum())})
    OUT.mkdir(parents=True)
    out.to_csv(OUT / "province_period_k_hulls.csv", index=False)
    summary = {"status": "SCOPE_RESIDUAL_ROBUST_K_OUTER_HULL_NO_HOURLY_LABELS",
               "cohort": "V341 complete 30-province × 11-period panel (330 cells)",
               "thresholds_g_kwh": TAUS,
               "residual_EF_support_g_kwh": [0, 1000],
               "thermal_EF_support_g_kwh": [400, 1000],
               "frontier": summary_rows,
               "relaxations": ["independent technology caps retain V342 convention; unresolved residual is given the full province total capacity as its hourly cap, so simultaneous capacity use can be over-allocated.",
                               "monthly residual mass ranges come from V345 and are treated cellwise; the annual cross-month sum is not jointly optimized.",
                               "residual technology/EF is represented by endpoint factors 0 and 1000; monotonicity gives safe factor-support extrema.",
                               "K ranges are conservative outer hulls under these relaxations, not empirical coverage or forecast intervals.",
                               "The model assumes the V345 annual remainder is measured without revision error and may be assigned within its feasible capacity range."],
               "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                   V341 / "monthly_margin_capacity_audit.csv", V345 / "province_period_residual_ranges.csv",
                   V341 / "summary.json", V345 / "summary.json", CAP, SCRIPT342, PROTOCOL, Path(__file__))}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"rows={len(out)}")


if __name__ == "__main__":
    main()
