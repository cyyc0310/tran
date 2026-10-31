"""V339: identified sets for hourly tail counts using Chinese monthly mix/capacity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
CAPACITY = ROOT / "data/china/cec_province_installed_capacity_2022_2023.csv"
PROTOCOL = ROOT / "docs/V339_CHINA_CAPACITY_TAIL_IDENTIFICATION_PROTOCOL.md"
OUT = ROOT / "results/v339-china-capacity-tail-identification-r1"
GROUPS = ("thermal", "hydro", "nuclear", "wind", "solar", "unresolved")
NBS_GROUPS = GROUPS[:-1]
CAP_COL = {"thermal": "thermal", "hydro": "hydro", "nuclear": "total", "wind": "wind", "solar": "solar", "unresolved": "total"}
EF_SUPPORT = (400.0, 1000.0)
TAUS = (100.0, 300.0)
RATES = (0.01, 0.05, 0.10)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def margin_only_set(margins: dict[str, float], ef: dict[str, float], tau: float, hours: int) -> tuple[int, ...]:
    positive = any(margins[f] > 0 and ef[f] > tau for f in GROUPS)
    negative = any(margins[f] > 0 and ef[f] < tau for f in GROUPS)
    neutral = any(margins[f] > 0 and ef[f] == tau for f in GROUPS)
    total_excess = sum((ef[f] - tau) * margins[f] for f in GROUPS)
    if not positive:
        return (0,)
    if negative:
        return tuple(range(1, hours + 1)) if total_excess > 0 else tuple(range(hours))
    if neutral:
        return tuple(range(1, hours + 1))
    return (hours,)


def capacity_outer_set(margins: dict[str, float], caps_mw: dict[str, float], ef: dict[str, float], tau: float, hours: int) -> tuple[int, ...]:
    total_excess = sum((ef[f] - tau) * margins[f] for f in GROUPS)
    for f in GROUPS:
        if margins[f] > hours * caps_mw[f] + 1e-5:
            raise ValueError(f"monthly energy exceeds capacity: {f}, {margins[f]} > {hours*caps_mw[f]}")
    result = []
    q = {f: ef[f] - tau for f in GROUPS}
    for k in range(hours + 1):
        if k == 0:
            if total_excess <= 0:
                result.append(k)
            continue
        if k == hours:
            if total_excess > 0:
                result.append(k)
            continue
        q_event_max = 0.0
        for f in GROUPS:
            margin = margins[f]
            cap = caps_mw[f]
            lower = max(0.0, margin - (hours - k) * cap)
            upper = min(margin, k * cap)
            if lower > upper + 1e-7:
                break
            q_event_max += q[f] * (upper if q[f] > 0 else lower)
        else:
            eps = 1e-9 * max(1.0, abs(total_excess), sum(abs(q[f] * margins[f]) for f in GROUPS))
            if q_event_max > 0 and q_event_max >= total_excess - eps:
                result.append(k)
    return tuple(result)


def union_sets(sets: list[tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(sorted(set().union(*(set(s) for s in sets))))


def decision(counts: tuple[int, ...], required: int) -> tuple[str, int]:
    lo, hi = min(counts), max(counts)
    return ("positive" if lo >= required else "negative" if hi < required else "abstain", hi - lo)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    raw = pd.read_csv(MONTHLY)
    raw = raw.loc[raw.year.eq(2023)].copy()
    cap = pd.read_csv(CAPACITY).set_index(["province", "year"])
    rows = []
    audit = []
    for province in sorted(raw.province.unique()):
        for month in (2, *range(3, 13)):
            sub = raw.loc[raw.province.eq(province) & raw.month.eq(month)]
            total_row = sub.loc[sub.technology.eq("total"), "generation_gwh"]
            if total_row.empty or pd.isna(total_row.iloc[0]):
                raise ValueError(f"missing published total: {province}, {month}")
            observed = {}
            present = {}
            for group in NBS_GROUPS:
                val = sub.loc[sub.technology.eq(group), "generation_gwh"]
                present[group] = bool(not val.empty and pd.notna(val.iloc[0]))
                observed[group] = float(val.iloc[0]) * 1000 if present[group] else np.nan  # MWh
            total_mwh = float(total_row.iloc[0]) * 1000
            known_sum = sum(v for v in observed.values() if np.isfinite(v))
            # A missing category is not zero. Keep the remainder in an explicit
            # unresolved group instead of silently assigning it to hydro.
            missing = [g for g in NBS_GROUPS if not present[g]]
            if missing:
                if len(missing) != 1:
                    raise ValueError(f"multiple missing category rows for {province}, {month}: {missing}")
                residual = total_mwh - known_sum
                observed[missing[0]] = 0.0
                observed["unresolved"] = max(0.0, residual)
            else:
                observed["unresolved"] = 0.0
            # Some NBS category values differ slightly from the reported total.
            # Define this estimand over the sum of reported categories, and retain
            # the discrepancy as an audit instead of silently forcing closure.
            category_total_mwh = sum(observed.values())
            category_minus_total = category_total_mwh - total_mwh
            hours = (59 if month == 2 else int(pd.Period(f"2023-{month:02d}").days_in_month)) * 24
            margins = {g: float(observed[g]) for g in GROUPS}
            for vintage in (2022, 2023):
                c = cap.loc[(province, vintage)]
                caps = {g: float(c[CAP_COL[g]]) * 10.0 for g in GROUPS}  # 10,000 kW = 10 MW
                for tau in TAUS:
                    all_bounds = {}
                    for arm in ("monthly_only", "monthly_plus_lagged_capacity", "monthly_plus_2023_capacity"):
                        cap_arm = None if arm == "monthly_only" else caps
                        endpoint_sets = []
                        for thermal_ef in EF_SUPPORT:
                          for unresolved_ef in (0.0, 1000.0):
                            ef = {g: 0.0 for g in GROUPS}
                            ef["thermal"] = thermal_ef
                            ef["unresolved"] = unresolved_ef
                            if cap_arm is None:
                                endpoint_sets.append(margin_only_set(margins, ef, tau, hours))
                            else:
                                endpoint_sets.append(capacity_outer_set(margins, cap_arm, ef, tau, hours))
                        all_bounds[arm] = union_sets(endpoint_sets)
                    # The EF support was ordered; bounds must widen monotonically to the union.
                    if not set(all_bounds["monthly_plus_lagged_capacity"]).issubset(all_bounds["monthly_only"]):
                        raise ValueError(f"capacity set is not nested: {province}, {month}, tau={tau}")
                    if not set(all_bounds["monthly_plus_2023_capacity"]).issubset(all_bounds["monthly_only"]):
                        raise ValueError(f"same-year capacity set is not nested: {province}, {month}, tau={tau}")
                    for rate in RATES:
                        required = int(np.ceil(rate * hours))
                        decisions = {arm: decision(bounds, required) for arm, bounds in all_bounds.items()}
                        rows.append({"province": province, "period": "Jan-Feb" if month == 2 else f"{month:02d}",
                                     "month_index": month, "hours": hours, "tau_g_kwh": tau,
                                     "tail_rate": rate, "required_hours": required,
                                     **{f"{arm}_lower": min(bounds) for arm, bounds in all_bounds.items()},
                                     **{f"{arm}_upper": max(bounds) for arm, bounds in all_bounds.items()},
                                     **{f"{arm}_width": decisions[arm][1] for arm in all_bounds},
                                     **{f"{arm}_decision": decisions[arm][0] for arm in all_bounds},
                                     "thermal_ef_support": list(EF_SUPPORT).__repr__(),
                                     "nbs_monthly_category_missing": bool(missing),
                                     "reported_category_sum_minus_total_mwh": category_minus_total})
            audit.append({"province": province, "period": "Jan-Feb" if month == 2 else f"{month:02d}",
                          "hours": hours, "total_mwh": total_mwh, "category_sum_mwh": category_total_mwh,
                          "missing_nbs_categories": ";".join(missing), "category_sum_minus_total_mwh": category_minus_total})
    result = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audit)
    metrics = []
    for keys, frame in result.groupby(["tau_g_kwh", "tail_rate"]):
        for arm in ("monthly_only", "monthly_plus_lagged_capacity", "monthly_plus_2023_capacity"):
            metrics.append({"tau_g_kwh": keys[0], "tail_rate": keys[1], "arm": arm,
                            "province_periods": int(frame.province.nunique() * 11),
                            "mean_K_width_hours": float(frame[f"{arm}_width"].mean()),
                            "median_K_width_hours": float(frame[f"{arm}_width"].median()),
                            "positive_certificate_pct": float(frame[f"{arm}_decision"].eq("positive").mean() * 100),
                            "negative_certificate_pct": float(frame[f"{arm}_decision"].eq("negative").mean() * 100),
                            "abstain_pct": float(frame[f"{arm}_decision"].eq("abstain").mean() * 100)})
    OUT.mkdir(parents=True)
    result.to_csv(OUT / "province_period_tail_sets.csv", index=False)
    audit_df.to_csv(OUT / "monthly_margin_closure_audit.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "information_frontier.csv", index=False)
    payload = {"status": "SCENARIO_IDENTIFIED_SET_NO_HOURLY_LABEL_VALIDATION",
               "n_provinces": int(result.province.nunique()), "n_province_periods": int(audit_df.shape[0]),
               "n_evaluation_rows": int(result.shape[0]), "thermal_ef_support": EF_SUPPORT,
               "frontier": metrics,
               "n_missing_monthly_category_periods": int(audit_df.missing_nbs_categories.ne("").sum()),
               "limitations": ["No Chinese hourly generation/CIF labels; identified sets are not empirical coverage.",
                               "Target is direct generation CIF within the NBS above-designated-size pool, not MEE consumption-side EF.",
                               "Thermal EF support [400,1000] and zero direct EF for nonthermal technology are declared scenarios, not calibrated China-wide factors.",
                               "CEC capacities are retrospective; 2023-vintage arm is ex-post.",
                               "V338 tested only monthly-average capacity; hourly upper-bound validity remains unverified.",
                               "Nuclear cap uses total province-wide capacity because CEC yearbook lacks a separate province nuclear capacity table."]}
    payload["sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in (MONTHLY, CAPACITY, PROTOCOL, Path(__file__))}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
