"""V377: isolate local-mix and shared-capacity effects with zero scope residual."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import GROUPS
from v376_shared_system_capacity import shared_capacity_set


ROOT = Path(__file__).resolve().parents[1]
V320 = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
V341 = ROOT / "results/v341-china-complete-category-hull-r2/monthly_margin_capacity_audit.csv"
V373 = ROOT / "results/v373-national-vs-local-mix-sharp-k-r1/province_period_comparison.csv"
V376 = ROOT / "results/v376-shared-system-capacity-r1/province_period_shared_capacity_hulls.csv"
PROTOCOL = ROOT / "docs/V377_SHARED_CAPACITY_NO_SCOPE_RESIDUAL_PROTOCOL.md"
OUT = ROOT / "results/v377-shared-capacity-no-scope-residual-r1"
TECHS = ("thermal", "hydro", "nuclear", "wind", "solar")
PERIOD_MONTH = {"Jan-Feb": 2, **{f"{m:02d}": m for m in range(3, 13)}}
TAU = 300
VINTAGE = 2022


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_margin_hull(margins, group_caps, system_cap, hours):
    candidates = []
    for thermal_ef in (400, 1000):
        factors = {g: 0 for g in GROUPS}
        factors["thermal"] = thermal_ef
        factors["unresolved"] = 0
        fixed = {**margins, "unresolved": F(0)}
        caps = {**group_caps, "unresolved": max(F(1), system_cap)}
        values = shared_capacity_set(fixed, caps, system_cap, factors, TAU, hours)
        if values is None:
            return None
        candidates.extend(values)
    return min(candidates), max(candidates)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(V341)
    source = source.loc[source["included_in_complete_case"].astype(bool)].copy()
    cohort = pd.read_csv(V373)
    cohort = cohort.loc[
        cohort["capacity_vintage"].eq(VINTAGE)
        & cohort["threshold_g_kwh"].eq(TAU)
        & cohort["national_mix_feasible"].astype(bool)
    ]
    keys = set(zip(cohort["province"], cohort["period"]))
    source = source.loc[[
        (p, period) in keys for p, period in zip(source["province"], source["period"])
    ]].copy()
    if len(source) != 262:
        raise ValueError(f"paired cohort mismatch: {len(source)}")
    v373 = pd.read_csv(V373).query("capacity_vintage == 2022 and threshold_g_kwh == 300")
    v373_map = v373.set_index(["province", "period"])
    v376 = pd.read_csv(V376).set_index(["province", "period", "mix"])
    raw = pd.read_csv(V320)
    raw = raw.loc[raw["year"].eq(2023) & raw["technology"].isin(TECHS)]
    pivot = raw.pivot_table(index=["province", "month"], columns="technology", values="generation_gwh", aggfunc="first")
    pivot["classified_gwh"] = pivot[list(TECHS)].sum(axis=1, min_count=len(TECHS))
    pooled = pivot.loc[pivot["classified_gwh"].notna()].groupby("month")[["thermal", "classified_gwh"]].sum(min_count=30)
    national_share_by_month = {
        int(month): F(str(rec.thermal)) / F(str(rec.classified_gwh))
        for month, rec in pooled.iterrows()
    }

    rows = []
    for rec in source.to_dict("records"):
        province, period = rec["province"], rec["period"]
        local = {g: F(str(rec[f"{g}_mwh"])) for g in TECHS}
        total = sum(local.values(), F(0))
        national_share = national_share_by_month[PERIOD_MONTH[period]]
        national = dict(local)
        national["thermal"] = total * national_share
        nonthermal = total - local["thermal"]
        if nonthermal == 0 and national["thermal"] != total:
            raise ValueError(f"nonthermal redistribution undefined: {province} {period}")
        if nonthermal:
            for g in TECHS:
                if g != "thermal":
                    national[g] = local[g] * (total - national["thermal"]) / nonthermal
        group_caps = {g: F(str(rec[f"{VINTAGE}_{g}_cap_mw"])) for g in TECHS}
        system_cap = F(str(rec[f"{VINTAGE}_unresolved_cap_mw"]))
        hours = int(rec["hours"])
        for mix, margins in (("local", local), ("national", national)):
            bounds = fixed_margin_hull(margins, group_caps, system_cap, hours)
            if bounds is None:
                raise ValueError(f"zero-residual case infeasible: {province} {period} {mix}")
            shared_full = v376.loc[(province, period, mix)]
            independent = v373_map.loc[(province, period)]
            independent_col = "local_sharp_hull_width_hours" if mix == "local" else "national_mix_sharp_hull_width_hours"
            rows.append({
                "province": province, "period": period, "hours": hours, "mix": mix,
                "zero_residual_lower": bounds[0], "zero_residual_upper": bounds[1],
                "zero_residual_width_h": bounds[1]-bounds[0],
                "shared_full_residual_width_h": int(shared_full["shared_width_h"]),
                "independent_zero_residual_width_h": int(independent[independent_col]),
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "province_period_comparison.csv", index=False)
    local = detail.loc[detail["mix"].eq("local")].set_index(["province", "period"])
    national = detail.loc[detail["mix"].eq("national")].set_index(["province", "period"])
    reduction = national["zero_residual_width_h"] - local["zero_residual_width_h"]
    summary = {
        "experiment": "V377 shared-capacity K hull with scope residual fixed to zero",
        "status": "OPTIMISTIC_BOUNDARY_SENSITIVITY_NOT_FORECAST_VALIDATION",
        "paired_province_periods": int(len(local)), "provinces": int(local.index.get_level_values(0).nunique()),
        "capacity_vintage": VINTAGE, "threshold_g_kwh": TAU,
        "local_mean_width_h": float(local["zero_residual_width_h"].mean()),
        "national_mean_width_h": float(national["zero_residual_width_h"].mean()),
        "mean_local_width_reduction_h": float(reduction.mean()),
        "local_narrower_n": int((reduction > 0).sum()), "local_wider_n": int((reduction < 0).sum()),
        "equal_n": int((reduction == 0).sum()),
        "local_full_month_hulls_n": int((local["zero_residual_width_h"] == local["hours"]).sum()),
        "national_full_month_hulls_n": int((national["zero_residual_width_h"] == national["hours"]).sum()),
        "shared_capacity_shrink_local_vs_v373_h": float((local["independent_zero_residual_width_h"]-local["zero_residual_width_h"]).mean()),
        "limitations": [
            "Scope residual fixed to zero is an optimistic boundary, not a valid assumption for NBS above-designated-size data.",
            "CEC total capacity is retrospective installed capacity, not hourly available output.",
            "No Chinese hourly CIF labels or empirical coverage/efficiency measures are used.",
        ],
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            V320, V341, V373, V376, ROOT / "scripts/v376_shared_system_capacity.py", PROTOCOL, Path(__file__),
        )},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
