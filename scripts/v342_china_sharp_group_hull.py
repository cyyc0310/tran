"""V342: exact fixed-factor K sets and tightness versus V341 outer hull."""
from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V342_CHINA_SHARP_GROUP_HULL_PROTOCOL.md"
V341 = ROOT / "results/v341-china-complete-category-hull-r2"
V341_SCRIPT = ROOT / "scripts/v341_china_complete_category_hull.py"
V339_IMPL = ROOT / "scripts/v339_china_capacity_tail_identification.py"
MONTHLY_INPUT = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
CAPACITY_INPUT = ROOT / "data/china/cec_province_installed_capacity_2022_2023.csv"
OUT = ROOT / "results/v342-china-sharp-group-hull-r4"
GROUPS = ("thermal", "hydro", "nuclear", "wind", "solar", "unresolved")
TAUS = (100, 300)
RATES = (0.01, 0.05, 0.10)
THERMAL_EFS = (400, 1000)


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def margin_only_base(margins: dict[str, F], ef: dict[str, int], tau: int, hours: int) -> tuple[int, ...]:
    positive = any(margins[g] > 0 and ef[g] > tau for g in GROUPS)
    negative = any(margins[g] > 0 and ef[g] < tau for g in GROUPS)
    neutral = any(margins[g] > 0 and ef[g] == tau for g in GROUPS)
    excess = sum((F(ef[g] - tau) * margins[g] for g in GROUPS), F(0))
    if not positive:
        return (0,)
    if negative:
        return tuple(range(1, hours + 1)) if excess > 0 else tuple(range(hours))
    if neutral:
        return tuple(range(1, hours + 1))
    return (hours,)


def sharp_group_set(margins: dict[str, F], caps: dict[str, F], ef: dict[str, int], tau: int,
                    hours: int) -> tuple[int, ...] | None:
    """Exact K set under fixed group factors, shared hourly caps, and positive output."""
    if hours < 1 or any(x < 0 for x in margins.values()) or any(x < 0 for x in caps.values()):
        raise ValueError("hours, margins and capacity caps must be nonnegative")
    for group in GROUPS:
        if margins[group] > hours * caps[group]:
            return None
    q = {g: F(ef[g] - tau) for g in GROUPS}
    total_excess = sum((q[g] * margins[g] for g in GROUPS), F(0))
    total_generation = sum(margins.values(), F(0))
    base = margin_only_base(margins, ef, tau, hours)
    out = []
    for k in range(hours + 1):
        if k == 0:
            if k in base:
                out.append(k)
            continue
        if k == hours:
            if k in base:
                out.append(k)
            continue
        q_event_max = F(0)
        min_event_generation = F(0)
        feasible = True
        for group in GROUPS:
            m = margins[group]
            cap = caps[group]
            lo = max(F(0), m - (hours - k) * cap)
            hi = min(m, k * cap)
            if lo > hi:
                feasible = False
                break
            event_mass = hi if q[group] > 0 else lo
            q_event_max += q[group] * event_mass
            min_event_generation += event_mass
        if not feasible:
            continue
        # Strict event excess; complement excess cannot be positive. At the
        # equality boundary, the complement must still carry positive output.
        if q_event_max > 0 and q_event_max >= total_excess:
            if q_event_max == total_excess and min_event_generation >= total_generation:
                continue
            if k in base:
                out.append(k)
    return tuple(out)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = pd.read_csv(V341 / "monthly_margin_capacity_audit.csv")
    source = source.loc[source.included_in_complete_case.astype(bool)].copy()
    hull = pd.read_csv(V341 / "province_period_tail_sets.csv")
    if len(source) != 330 or source.duplicated(["province", "period"]).any():
        raise ValueError("V341 source cohort mismatch")
    source_period_keys = set(zip(source.province, source.period))
    expected_hull_keys = {(p, period, tau) for p, period in source_period_keys for tau in TAUS}
    actual_hull_keys = set(hull[["province", "period", "tau_g_kwh"]].drop_duplicates()
                           .itertuples(index=False, name=None))
    if actual_hull_keys != expected_hull_keys:
        raise ValueError(f"V341 source/hull keys differ: missing={len(expected_hull_keys-actual_hull_keys)}, "
                         f"extra={len(actual_hull_keys-expected_hull_keys)}")
    rows = []
    for rec in source.to_dict("records"):
        province, period = rec["province"], rec["period"]
        hours = int(rec["hours"])
        margins = {g: F(str(rec[f"{g}_mwh"])) for g in GROUPS}
        caps_by_vintage = {}
        for vintage in (2022, 2023):
            caps_by_vintage[vintage] = {
                g: F(str(rec[f"{vintage}_{g}_cap_mw"])) for g in GROUPS
            }
        arms = {"monthly_only": {g: margins[g] for g in GROUPS},
                "monthly_plus_lagged_capacity": caps_by_vintage[2022],
                "monthly_plus_2023_capacity": caps_by_vintage[2023]}
        for tau in TAUS:
            endpoint_sets = {}
            for arm, caps in arms.items():
                vals = []
                for thermal_ef in THERMAL_EFS:
                    ef = {g: 0 for g in GROUPS}
                    ef["thermal"] = thermal_ef
                    vals.append(sharp_group_set(margins, caps, ef, tau, hours))
                if any(v is None for v in vals):
                    raise ValueError(f"capacity infeasible: {province} {period} {arm}")
                endpoint_sets[arm] = vals
            hsub = hull.loc[hull.province.eq(province) & hull.period.eq(period) & hull.tau_g_kwh.eq(tau)]
            if len(hsub) != len(RATES) or set(hsub.tail_rate) != set(RATES):
                raise ValueError(f"V341 hull keys mismatch: {province} {period} tau={tau}")
            if hsub.duplicated(["province", "period", "tau_g_kwh", "tail_rate"]).any():
                raise ValueError(f"duplicate V341 rate rows: {province} {period} tau={tau}")
            for arm in endpoint_sets:
                for field in ("lower", "upper", "width"):
                    if hsub[f"{arm}_{field}"].nunique() != 1:
                        raise ValueError(f"V341 hull unexpectedly varies by tail rate: {province} {period} {tau} {arm}")
            outer_row = hsub.iloc[0]
            row = {"province": province, "period": period, "hours": hours, "tau_g_kwh": tau}
            for arm, (low_factor, high_factor) in endpoint_sets.items():
                endpoint_union = set(low_factor) | set(high_factor)
                lo, hi = min(low_factor), max(high_factor)
                row[f"{arm}_fixed400_k_values"] = ";".join(map(str, low_factor))
                row[f"{arm}_fixed1000_k_values"] = ";".join(map(str, high_factor))
                row[f"{arm}_fixed400_k_count"] = len(low_factor)
                row[f"{arm}_fixed1000_k_count"] = len(high_factor)
                row[f"{arm}_support_endpoint_union_count"] = len(endpoint_union)
                row[f"{arm}_support_endpoint_union_hull_gap_count"] = hi - lo + 1 - len(endpoint_union)
                row[f"{arm}_sharp_hull_lower"] = lo
                row[f"{arm}_sharp_hull_upper"] = hi
                row[f"{arm}_sharp_hull_width"] = hi - lo
                row[f"{arm}_outer_hull_lower"] = int(outer_row[f"{arm}_lower"])
                row[f"{arm}_outer_hull_upper"] = int(outer_row[f"{arm}_upper"])
                row[f"{arm}_outer_hull_width"] = int(outer_row[f"{arm}_width"])
                if lo < row[f"{arm}_outer_hull_lower"] or hi > row[f"{arm}_outer_hull_upper"]:
                    raise ValueError(f"sharp set escapes outer hull: {province} {period} {tau} {arm}")
            rows.append(row)
    result = pd.DataFrame(rows)
    expected = 330 * len(TAUS)
    if len(result) != expected or result.duplicated(["province", "period", "tau_g_kwh"]).any():
        raise ValueError(f"row/key count mismatch {len(result)} != {expected}")
    metrics = []
    for tau, frame in result.groupby("tau_g_kwh"):
        for arm in ("monthly_only", "monthly_plus_lagged_capacity", "monthly_plus_2023_capacity"):
            gap_count = frame[f"{arm}_sharp_hull_width"] + 1 - frame[f"{arm}_support_endpoint_union_count"]
            metrics.append({"tau_g_kwh": tau, "arm": arm,
                            "province_periods": len(frame),
                            "mean_outer_hull_width": float(frame[f"{arm}_outer_hull_width"].mean()),
                            "mean_sharp_hull_width": float(frame[f"{arm}_sharp_hull_width"].mean()),
                            "mean_hull_width_reduction_vs_outer_hours": float((frame[f"{arm}_outer_hull_width"] - frame[f"{arm}_sharp_hull_width"]).mean()),
                            "same_hull_endpoints_n": int(((frame[f"{arm}_outer_hull_lower"] == frame[f"{arm}_sharp_hull_lower"]) &
                                                           (frame[f"{arm}_outer_hull_upper"] == frame[f"{arm}_sharp_hull_upper"])).sum()),
                            "mean_endpoint_union_hull_gap_count": float(frame[f"{arm}_support_endpoint_union_hull_gap_count"].mean()),
                            "full_integer_interval_endpoint_union_n": int(gap_count.eq(0).sum()),
                            "checked_province_periods_n": int(len(frame)),
                            "mean_fixed400_set_count": float(frame[f"{arm}_fixed400_k_count"].mean()),
                            "mean_fixed1000_set_count": float(frame[f"{arm}_fixed1000_k_count"].mean())})
            if not gap_count.eq(0).all():
                raise ValueError(f"support endpoint union has an integer gap: tau={tau} arm={arm}")
    OUT.mkdir(parents=True)
    result.to_csv(OUT / "province_period_sharp_and_outer_hulls.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "sharp_vs_outer_frontier.csv", index=False)
    summary = {"status": "EXACT_FIXED_FACTOR_SET_TIGHTNESS_AUDIT_NO_CHINA_HOURLY_LABELS",
               "n_provinces": int(result.province.nunique()), "n_province_periods": int(source.shape[0]),
               "n_rows": int(result.shape[0]), "expected_rows": expected,
               "thermal_ef_endpoints_g_kwh": THERMAL_EFS, "frontier": metrics,
               "all_endpoint_unions_fill_hulls": True,
               "endpoint_union_proof_scope": "empirical finite-cohort check for each fixed-factor endpoint set; continuous support hull follows from monotonic K in thermal EF",
               "limitations": ["Exactness applies only at each fixed-factor endpoint and under shared group hourly caps, exact monthly margins, nonnegative generation, and positive output in both event and complement hours.",
                               "Support-endpoint extrema give a safe hull by monotonicity; endpoint-set union is not asserted to be the full feasible K set over continuous EF support.",
                               "The output remains scenario identification, not empirical coverage, forecast skill, or measured Chinese hourly CIF.",
                               "The group-capacity sharp construction is already present in project V92/V102, so this China transfer alone is not claimed as a new algorithm."]}
    summary["sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in
                         (V341 / "monthly_margin_capacity_audit.csv", V341 / "province_period_tail_sets.csv",
                          V341 / "summary.json", MONTHLY_INPUT, CAPACITY_INPUT, V341_SCRIPT,
                          V339_IMPL, PROTOCOL, Path(__file__))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"rows={len(result)} expected={expected}")


if __name__ == "__main__":
    main()
