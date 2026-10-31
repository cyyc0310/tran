"""V376: exact K-hull sensitivity with a shared total system capacity."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import GROUPS, sharp_group_set


ROOT = Path(__file__).resolve().parents[1]
V320 = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
V341 = ROOT / "results/v341-china-complete-category-hull-r2/monthly_margin_capacity_audit.csv"
V345 = ROOT / "results/v345-china-residual-monthly-scope-r1/province_period_residual_ranges.csv"
V373 = ROOT / "results/v373-national-vs-local-mix-sharp-k-r1/province_period_comparison.csv"
V374 = ROOT / "results/v374-local-mix-with-scope-residual-r1/province_period_comparison.csv"
PROTOCOL = ROOT / "docs/V376_SHARED_SYSTEM_CAPACITY_PROTOCOL.md"
OUT = ROOT / "results/v376-shared-system-capacity-r1"
TECHS = ("thermal", "hydro", "nuclear", "wind", "solar")
PERIOD_MONTH = {"Jan-Feb": 2, **{f"{m:02d}": m for m in range(3, 13)}}
TAU = 300
VINTAGE = 2022
EF_CONFIGS = ((400, 0), (1000, 1000))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qmax_with_sum_bounds(q, lower, upper, sum_lo, sum_hi):
    """Return exact objective max and the minimum sum among maximizers."""
    lower_sum = sum(lower, F(0))
    if sum_lo > sum_hi or lower_sum > sum_hi or sum(upper, F(0)) < sum_lo:
        return None
    d_min = max(F(0), sum_lo - lower_sum)
    d_max = min(sum_hi - lower_sum, sum((u - l for l, u in zip(lower, upper)), F(0)))
    if d_min > d_max:
        return None
    positive_capacity = sum((u - l for qi, l, u in zip(q, lower, upper) if qi > 0), F(0))
    extra_target = max(d_min, min(d_max, positive_capacity))
    remaining = extra_target
    extras = [F(0)] * len(q)
    for i in sorted(range(len(q)), key=lambda idx: (q[idx], -idx), reverse=True):
        take = min(upper[i] - lower[i], remaining)
        extras[i] = take
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        return None
    objective = sum((qi * (li + xi) for qi, li, xi in zip(q, lower, extras)), F(0))
    min_sum = lower_sum + extra_target
    return objective, min_sum


def shared_capacity_set(margins, group_caps, system_cap, ef, tau, hours):
    """Exact K set with per-group and shared total hourly capacity limits."""
    if hours < 1 or system_cap < 0 or any(m < 0 for m in margins.values()) or any(c < 0 for c in group_caps.values()):
        raise ValueError("invalid hours, margins or capacity")
    if any(margins[g] > hours * group_caps[g] for g in GROUPS):
        return None
    total_mass = sum((margins[g] for g in GROUPS), F(0))
    if total_mass > hours * system_cap:
        return None
    q = {g: F(ef[g] - tau) for g in GROUPS}
    total_excess = sum((q[g] * margins[g] for g in GROUPS), F(0))
    result = []
    if total_mass > 0 and total_excess <= 0:
        result.append(0)
    if total_mass > 0 and total_excess > 0:
        result.append(hours)
    for k in range(1, hours):
        lower = [max(F(0), margins[g] - (hours - k) * group_caps[g]) for g in GROUPS]
        upper = [min(margins[g], k * group_caps[g]) for g in GROUPS]
        sum_lo = max(sum(lower, F(0)), total_mass - (hours - k) * system_cap)
        sum_hi = min(sum(upper, F(0)), k * system_cap, total_mass)
        q_vec = [q[g] for g in GROUPS]
        optimum = _qmax_with_sum_bounds(q_vec, lower, upper, sum_lo, sum_hi)
        if optimum is None:
            continue
        q_event_max, min_event_mass = optimum
        if q_event_max <= 0 or q_event_max < total_excess:
            continue
        if q_event_max == total_excess:
            complement_can_be_positive = min_event_mass < total_mass
        else:
            complement_can_be_positive = sum_lo < total_mass
        if complement_can_be_positive:
            result.append(k)
    return tuple(result)


def outer_hull(margins, caps, system_cap, residual_hi, hours):
    candidates = []
    for residual in {F(0), residual_hi}:
        for thermal_ef, residual_ef in EF_CONFIGS:
            factors = {g: 0 for g in GROUPS}
            factors["thermal"] = thermal_ef
            factors["unresolved"] = residual_ef
            full_margins = {**margins, "unresolved": residual}
            group_caps = {**caps, "unresolved": max(F(1), system_cap)}
            values = shared_capacity_set(full_margins, group_caps, system_cap, factors, TAU, hours)
            if values is None:
                return None
            candidates.extend(values)
    if not candidates:
        return None
    return min(candidates), max(candidates)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(V320)
    raw = raw.loc[raw["year"].eq(2023) & raw["technology"].isin((*TECHS, "total"))]
    pivot = raw.pivot_table(index=["province", "month"], columns="technology", values="generation_gwh", aggfunc="first")
    pivot["classified_gwh"] = pivot[list(TECHS)].sum(axis=1, min_count=len(TECHS))
    complete = pivot.loc[pivot["classified_gwh"].notna()]
    pooled = complete.groupby("month")[[*TECHS, "classified_gwh"]].sum(min_count=30)
    shares = {int(m): F(str(r.thermal)) / F(str(r.classified_gwh)) for m, r in pooled.iterrows()}

    source = pd.read_csv(V341)
    source = source.loc[source["included_in_complete_case"].astype(bool)].copy()
    residual = pd.read_csv(V345)
    residual = residual.loc[residual["capacity_vintage"].eq(VINTAGE)].copy()
    keys = set(zip(source["province"], source["period"]))
    residual = residual.loc[[
        (p, period) in keys for p, period in zip(residual["province"], residual["period"])
    ]].copy()
    residual_map = residual.set_index(["province", "period"])
    common = pd.read_csv(V373)
    common = common.loc[
        common["capacity_vintage"].eq(VINTAGE)
        & common["threshold_g_kwh"].eq(TAU)
        & common["national_mix_feasible"].astype(bool)
    ]
    common_keys = set(zip(common["province"], common["period"]))
    if len(common_keys) != 262:
        raise ValueError(f"V373 paired-feasible cohort mismatch: {len(common_keys)}")

    prior = pd.read_csv(V374)
    prior = prior.loc[prior["capacity_vintage"].eq(VINTAGE) & prior["threshold_g_kwh"].eq(TAU)]
    prior_map = prior.set_index(["province", "period"])

    rows = []
    for rec in source.to_dict("records"):
        province, period = rec["province"], rec["period"]
        if (province, period) not in common_keys:
            continue
        month, hours = PERIOD_MONTH[period], int(rec["hours"])
        local = {g: F(str(rec[f"{g}_mwh"])) for g in TECHS}
        total = sum(local.values(), F(0))
        national = dict(local)
        national["thermal"] = total * shares[month]
        nonthermal = total - local["thermal"]
        if nonthermal == 0 and national["thermal"] != total:
            raise ValueError(f"nonthermal redistribution undefined for {province} {period}")
        if nonthermal:
            for g in TECHS:
                if g != "thermal":
                    national[g] = local[g] * (total - national["thermal"]) / nonthermal

        caps = {g: F(str(rec[f"{VINTAGE}_{g}_cap_mw"])) for g in TECHS}
        system_cap = F(str(rec[f"{VINTAGE}_unresolved_cap_mw"]))
        residual_hi = F(str(residual_map.loc[(province, period), "monthly_residual_max_gwh"])) * 1000
        for mix_name, margins in (("local", local), ("national", national)):
            bounds = outer_hull(margins, caps, system_cap, residual_hi, hours)
            if bounds is None:
                raise ValueError(f"shared-capacity case infeasible: {province} {period} {mix_name}")
            prior_row = prior_map.loc[(province, period)]
            independent_low = int(prior_row[f"{mix_name}_lower"])
            independent_high = int(prior_row[f"{mix_name}_upper"])
            if bounds[0] < independent_low or bounds[1] > independent_high:
                raise ValueError(f"shared hull escaped independent hull: {province} {period} {mix_name}")
            rows.append({
                "province": province, "period": period, "hours": hours, "mix": mix_name,
                "shared_lower": bounds[0], "shared_upper": bounds[1], "shared_width_h": bounds[1]-bounds[0],
                "independent_lower": independent_low, "independent_upper": independent_high,
                "independent_width_h": independent_high-independent_low,
                "shrink_vs_independent_h": (independent_high-independent_low)-(bounds[1]-bounds[0]),
            })

        # Redundant shared-capacity limit must recover the independent solver exactly.
        check_caps = {**caps, "unresolved": system_cap}
        factors = {g: 0 for g in GROUPS}; factors["thermal"] = 400
        margins_for_check = {**local, "unresolved": F(0)}
        redundant = shared_capacity_set(
            margins_for_check, check_caps, sum(check_caps.values(), F(0)), factors, TAU, hours
        )
        reference = sharp_group_set(margins_for_check, check_caps, factors, TAU, hours)
        if tuple(sorted(redundant or ())) != tuple(sorted(reference or ())):
            raise ValueError(f"redundant-capacity consistency mismatch: {province} {period}")

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "province_period_shared_capacity_hulls.csv", index=False)
    summary_rows = []
    for mix, frame in detail.groupby("mix"):
        summary_rows.append({
            "mix": mix, "province_periods": int(len(frame)),
            "shared_mean_width_h": float(frame["shared_width_h"].mean()),
            "independent_mean_width_h": float(frame["independent_width_h"].mean()),
            "mean_shrink_from_shared_capacity_h": float(frame["shrink_vs_independent_h"].mean()),
            "full_month_shared_hulls_n": int((frame["shared_width_h"] == frame["hours"]).sum()),
        })
    local = detail.loc[detail["mix"].eq("local")].set_index(["province", "period"])
    national = detail.loc[detail["mix"].eq("national")].set_index(["province", "period"])
    delta = national["shared_width_h"] - local["shared_width_h"]
    comparison = {
        "paired_n": int(len(delta)), "mean_local_width_reduction_h": float(delta.mean()),
        "local_narrower_n": int((delta > 0).sum()), "local_wider_n": int((delta < 0).sum()),
        "equal_n": int((delta == 0).sum()),
        "local_full_month_n": int((local["shared_width_h"] == local["hours"]).sum()),
        "national_full_month_n": int((national["shared_width_h"] == national["hours"]).sum()),
    }
    pd.DataFrame(summary_rows).to_csv(OUT / "summary_by_mix.csv", index=False)
    summary = {
        "experiment": "V376 shared total installed-capacity sensitivity for V373/V374 mix comparison",
        "status": "SHARP_FIXED_MARGINAL_K_HULL_WITH_SHARED_SYSTEM_CAPACITY",
        "n_province_periods": int(local.shape[0]), "n_provinces": int(local.index.get_level_values(0).nunique()),
        "capacity_vintage": VINTAGE, "threshold_g_kwh": TAU,
        "thermal_ef_support_g_kwh": [400, 1000], "scope_residual_ef_support_g_kwh": [0, 1000],
        "comparison": comparison,
        "limitations": [
            "CEC total installed capacity is a retrospective aggregate upper bound, not hourly availability.",
            "This enforces total output <= total installed capacity per hour while retaining loose per-group capacities.",
            "Scope residual ranges are V345 annual gap plus monthly capacity room and are relaxed cellwise, not jointly across months.",
            "V320 is an NBS-derived transcription, not an official portal response; no hourly Chinese target labels or coverage metric are available.",
        ],
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            V320, V341, V345, V373, V374, ROOT / "scripts/v342_china_sharp_group_hull.py", PROTOCOL, Path(__file__),
        )},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
