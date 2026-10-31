"""V340: corrected vintage mapping for Chinese capacity-constrained K sets.

V339 is preserved as an invalid diagnostic: it crossed the capacity vintages
between arm labels and counted each province-period twice.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v339_china_capacity_tail_identification import (
    CAP_COL, CAPACITY, EF_SUPPORT, GROUPS, MONTHLY, NBS_GROUPS, RATES, TAUS,
    capacity_outer_set, decision, margin_only_set, sha,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V340_CHINA_CAPACITY_TAIL_IDENTIFICATION_CORRECTION.md"
OUT = ROOT / "results/v340-china-capacity-tail-identification-corrected-r1"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    raw = pd.read_csv(MONTHLY)
    raw = raw.loc[raw.year.eq(2023)].copy()
    cap = pd.read_csv(CAPACITY).set_index(["province", "year"])
    rows: list[dict] = []
    audits: list[dict] = []
    for province in sorted(raw.province.unique()):
        for month in (2, *range(3, 13)):
            sub = raw.loc[raw.province.eq(province) & raw.month.eq(month)]
            total = sub.loc[sub.technology.eq("total"), "generation_gwh"]
            if total.empty or pd.isna(total.iloc[0]):
                raise ValueError(f"missing total: {province}, {month}")
            observed: dict[str, float] = {}
            present: dict[str, bool] = {}
            for group in NBS_GROUPS:
                value = sub.loc[sub.technology.eq(group), "generation_gwh"]
                present[group] = bool(not value.empty and pd.notna(value.iloc[0]))
                observed[group] = float(value.iloc[0]) * 1000 if present[group] else np.nan
            total_mwh = float(total.iloc[0]) * 1000
            known_sum = sum(x for x in observed.values() if np.isfinite(x))
            missing = [g for g in NBS_GROUPS if not present[g]]
            if len(missing) > 1:
                raise ValueError(f"multiple missing groups: {province}, {month}: {missing}")
            if missing:
                observed[missing[0]] = 0.0
                observed["unresolved"] = max(0.0, total_mwh - known_sum)
            else:
                observed["unresolved"] = 0.0
            category_total = sum(observed.values())
            mismatch = category_total - total_mwh
            hours = (59 if month == 2 else int(pd.Period(f"2023-{month:02d}").days_in_month)) * 24
            margins = {g: float(observed[g]) for g in GROUPS}
            capacities = {}
            for vintage in (2022, 2023):
                c = cap.loc[(province, vintage)]
                capacities[vintage] = {g: float(c[CAP_COL[g]]) * 10 for g in GROUPS}
            sets: dict[str, dict[tuple[float, float], tuple[int, ...]]] = {
                "monthly_only": {}, "monthly_plus_lagged_capacity": {},
                "monthly_plus_2023_capacity": {},
            }
            for tau in TAUS:
                for arm in sets:
                    endpoint_sets = []
                    for thermal_ef in EF_SUPPORT:
                        for unresolved_ef in (0.0, 1000.0):
                            ef = {g: 0.0 for g in GROUPS}
                            ef["thermal"] = thermal_ef
                            ef["unresolved"] = unresolved_ef
                            if arm == "monthly_only":
                                endpoint_sets.append(margin_only_set(margins, ef, tau, hours))
                            else:
                                vintage = 2022 if arm.endswith("lagged_capacity") else 2023
                                endpoint_sets.append(capacity_outer_set(margins, capacities[vintage], ef, tau, hours))
                    sets[arm][(tau, 0.0)] = tuple(sorted(set().union(*(set(s) for s in endpoint_sets))))
            for tau in TAUS:
                for rate in RATES:
                    required = int(np.ceil(rate * hours))
                    row = {"province": province, "period": "Jan-Feb" if month == 2 else f"{month:02d}",
                           "month_index": month, "hours": hours, "tau_g_kwh": tau,
                           "tail_rate": rate, "required_hours": required,
                           "thermal_ef_support": str(EF_SUPPORT),
                           "nbs_monthly_category_missing": bool(missing),
                           "reported_category_sum_minus_total_mwh": mismatch}
                    for arm, arm_sets in sets.items():
                        bounds = arm_sets[(tau, 0.0)]
                        label, width = decision(bounds, required)
                        row[f"{arm}_lower"] = min(bounds)
                        row[f"{arm}_upper"] = max(bounds)
                        row[f"{arm}_width"] = width
                        row[f"{arm}_decision"] = label
                    rows.append(row)
            audits.append({"province": province, "period": "Jan-Feb" if month == 2 else f"{month:02d}",
                           "hours": hours, "total_mwh": total_mwh, "category_sum_mwh": category_total,
                           "missing_nbs_categories": ";".join(missing),
                           "category_sum_minus_total_mwh": mismatch,
                           **{f"{g}_mwh": margins[g] for g in GROUPS},
                           **{f"{vintage}_{g}_cap_mw": capacities[vintage][g]
                              for vintage in capacities for g in GROUPS}})
    result = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    metrics = []
    for (tau, rate), frame in result.groupby(["tau_g_kwh", "tail_rate"]):
        for arm in ("monthly_only", "monthly_plus_lagged_capacity", "monthly_plus_2023_capacity"):
            metrics.append({"tau_g_kwh": tau, "tail_rate": rate, "arm": arm,
                            "province_periods": len(frame),
                            "mean_K_width_hours": float(frame[f"{arm}_width"].mean()),
                            "median_K_width_hours": float(frame[f"{arm}_width"].median()),
                            "positive_certificate_pct": float(frame[f"{arm}_decision"].eq("positive").mean() * 100),
                            "negative_certificate_pct": float(frame[f"{arm}_decision"].eq("negative").mean() * 100),
                            "abstain_pct": float(frame[f"{arm}_decision"].eq("abstain").mean() * 100)})
    OUT.mkdir(parents=True)
    result.to_csv(OUT / "province_period_tail_sets.csv", index=False)
    audit.to_csv(OUT / "monthly_margin_capacity_audit.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "information_frontier.csv", index=False)
    same = ((result.monthly_plus_lagged_capacity_lower == result.monthly_plus_2023_capacity_lower) &
            (result.monthly_plus_lagged_capacity_upper == result.monthly_plus_2023_capacity_upper))
    payload = {"status": "CORRECTED_SCENARIO_IDENTIFIED_SET_NO_HOURLY_LABEL_VALIDATION",
               "correction_of": "V339_INVALID_CROSSED_VINTAGE_ARMS_AND_DOUBLE_COUNTED_ROWS",
               "n_provinces": int(result.province.nunique()), "n_province_periods": int(audit.shape[0]),
               "n_evaluation_rows": int(result.shape[0]), "expected_evaluation_rows": 31 * 11 * len(TAUS) * len(RATES),
               "same_lagged_and_2023_K_sets_n": int(same.sum()), "compared_K_sets_n": int(len(result)),
               "thermal_ef_support": EF_SUPPORT, "frontier": metrics,
               "n_missing_monthly_category_periods": int(audit.missing_nbs_categories.ne("").sum()),
               "limitations": ["No Chinese hourly generation/CIF labels; these are conditional outer identified sets, not empirical coverage.",
                               "The target is generation-side direct CIF in the NBS-reported pool, not consumption-side MEE EF.",
                               "Thermal EF [400,1000] and unresolved-group EF [0,1000] are scenarios, not calibrated China-wide factors.",
                               "CEC capacities are retrospective; lagged 2022 installed capacity can miss capacity commissioned during 2023.",
                               "V338 only checks monthly-average feasibility, not hourly nameplate validity.",
                               "Nuclear and unresolved groups use total provincial capacity as a loose upper bound.",
                               "The outer-set routine does not prove sharpness of each feasible K endpoint."]}
    payload["sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in (MONTHLY, CAPACITY, PROTOCOL, Path(__file__))}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"rows={len(result)}; lag/current identical={same.sum()}/{len(result)}")


if __name__ == "__main__":
    main()
