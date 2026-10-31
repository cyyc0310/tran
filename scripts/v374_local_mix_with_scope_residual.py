"""V374: test local-mix sharp K gains after adding NBS scope residual ranges."""

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
OUT = ROOT / "results/v374-local-mix-with-scope-residual-r1"
TECHS = ("thermal", "hydro", "nuclear", "wind", "solar")
PERIOD_MONTH = {"Jan-Feb": 2, **{f"{m:02d}": m for m in range(3, 13)}}
TAUS = (100, 300)
EF_CONFIGS = ((400, 0), (1000, 1000))
VINTAGES = (2022, 2023)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outer_hull(margins: dict[str, F], caps: dict[str, F], residual_lo: F, residual_hi: F, tau: int, hours: int):
    candidates = []
    residual_cap = max(F(1), caps["unresolved"])
    for residual in {residual_lo, residual_hi}:
        for thermal_ef, residual_ef in EF_CONFIGS:
            factors = {g: 0 for g in GROUPS}
            factors["thermal"] = thermal_ef
            factors["unresolved"] = residual_ef
            full_margins = {**margins, "unresolved": residual}
            full_caps = {**caps, "unresolved": residual_cap}
            values = sharp_group_set(full_margins, full_caps, factors, tau, hours)
            if values is None:
                return None
            candidates.extend(values)
    return (min(candidates), max(candidates))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(V320)
    raw = raw.loc[raw["year"].eq(2023) & raw["technology"].isin((*TECHS, "total"))]
    pivot = raw.pivot_table(index=["province", "month"], columns="technology", values="generation_gwh", aggfunc="first")
    pivot["classified_gwh"] = pivot[list(TECHS)].sum(axis=1, min_count=len(TECHS))
    complete = pivot.loc[pivot["classified_gwh"].notna()]
    if complete.index.get_level_values("province").nunique() != 30:
        raise ValueError("Expected 30 provinces with complete five-technology categories")
    pooled = complete.groupby("month")[[*TECHS, "classified_gwh"]].sum(min_count=30)
    share = {
        int(month): F(str(rec.thermal)) / F(str(rec.classified_gwh))
        for month, rec in pooled.iterrows()
    }

    source = pd.read_csv(V341)
    source = source.loc[source["included_in_complete_case"].astype(bool)].copy()
    residual = pd.read_csv(V345)
    residual = residual.loc[residual["capacity_vintage"].isin(VINTAGES)].copy()
    source_keys = set(zip(source["province"], source["period"]))
    residual = residual.loc[[
        (p, period) in source_keys for p, period in zip(residual["province"], residual["period"])
    ]].copy()
    residual_map = residual.set_index(["province", "period", "capacity_vintage"])
    if len(source) != 330 or source["province"].nunique() != 30:
        raise ValueError("V341 complete-case cohort mismatch")
    expected_keys = {(r.province, r.period, vintage) for r in source.itertuples() for vintage in VINTAGES}
    if expected_keys != set(residual_map.index):
        raise ValueError("V341 and V345 cohort keys differ")

    rows = []
    for rec in source.to_dict("records"):
        province, period = rec["province"], rec["period"]
        month = PERIOD_MONTH[period]
        hours = int(rec["hours"])
        local = {g: F(str(rec[f"{g}_mwh"])) for g in TECHS}
        local_total = sum(local.values(), F(0))
        local_thermal_share = local["thermal"] / local_total
        pooled_share = share[month]
        national = dict(local)
        national["thermal"] = local_total * pooled_share
        local_nonthermal = local_total - local["thermal"]
        if local_nonthermal == 0 and national["thermal"] != local_total:
            raise ValueError(f"nonthermal redistribution undefined for {province} {period}")
        if local_nonthermal:
            for tech in TECHS:
                if tech != "thermal":
                    national[tech] = local[tech] * (local_total - national["thermal"]) / local_nonthermal

        for vintage in VINTAGES:
            caps = {g: F(str(rec[f"{vintage}_{g}_cap_mw"])) for g in TECHS}
            province_cap = F(str(rec[f"{vintage}_unresolved_cap_mw"]))
            caps["unresolved"] = province_cap
            rr = residual_map.loc[(province, period, vintage)]
            residual_lo = F(str(rr["monthly_residual_min_gwh"])) * 1000
            residual_hi = F(str(rr["monthly_residual_max_gwh"])) * 1000
            for tau in TAUS:
                local_bounds = outer_hull(local, caps, residual_lo, residual_hi, tau, hours)
                national_bounds = outer_hull(national, caps, residual_lo, residual_hi, tau, hours)
                local_width = None if local_bounds is None else local_bounds[1] - local_bounds[0]
                national_width = None if national_bounds is None else national_bounds[1] - national_bounds[0]
                rows.append({
                    "province": province, "period": period, "hours": hours,
                    "capacity_vintage": vintage, "threshold_g_kwh": tau,
                    "local_thermal_share": float(local_thermal_share), "national_thermal_share": float(pooled_share),
                    "residual_min_gwh": float(residual_lo / 1000), "residual_max_gwh": float(residual_hi / 1000),
                    "local_feasible": local_bounds is not None, "national_mix_feasible": national_bounds is not None,
                    "local_lower": None if local_bounds is None else local_bounds[0],
                    "local_upper": None if local_bounds is None else local_bounds[1],
                    "local_width_h": local_width,
                    "national_lower": None if national_bounds is None else national_bounds[0],
                    "national_upper": None if national_bounds is None else national_bounds[1],
                    "national_width_h": national_width,
                    "local_width_reduction_h": None if local_width is None or national_width is None else national_width - local_width,
                })

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "province_period_comparison.csv", index=False)
    summaries = []
    for (vintage, tau), frame in detail.groupby(["capacity_vintage", "threshold_g_kwh"]):
        paired = frame.loc[frame["local_feasible"] & frame["national_mix_feasible"]]
        summaries.append({
            "capacity_vintage": int(vintage), "threshold_g_kwh": int(tau), "province_periods": int(len(frame)),
            "local_feasible_n": int(frame["local_feasible"].sum()), "national_mix_feasible_n": int(frame["national_mix_feasible"].sum()),
            "paired_feasible_n": int(len(paired)), "national_infeasible_n": int((~frame["national_mix_feasible"]).sum()),
            "local_mean_width_paired_h": float(paired["local_width_h"].mean()) if len(paired) else None,
            "national_mean_width_paired_h": float(paired["national_width_h"].mean()) if len(paired) else None,
            "mean_local_reduction_h": float(paired["local_width_reduction_h"].mean()) if len(paired) else None,
            "local_narrower_n": int((paired["local_width_reduction_h"] > 0).sum()),
            "local_wider_n": int((paired["local_width_reduction_h"] < 0).sum()),
            "equal_n": int((paired["local_width_reduction_h"] == 0).sum()),
            "local_full_month_hull_n": int((paired["local_width_h"] == paired["hours"]).sum()),
            "national_full_month_hull_n": int((paired["national_width_h"] == paired["hours"]).sum()),
        })
    pd.DataFrame(summaries).to_csv(OUT / "summary_by_capacity_threshold.csv", index=False)
    summary = {
        "experiment": "V374 local-versus-national mix K hulls with monthly statistical-scope residual",
        "status": "CONSERVATIVE_IDENTIFICATION_HULL_NOT_FORECAST_VALIDATION",
        "n_rows": int(len(detail)), "n_provinces": int(detail["province"].nunique()),
        "thresholds_g_kwh": list(TAUS), "capacity_vintages": list(VINTAGES),
        "thermal_ef_support_g_kwh": [400, 1000], "residual_ef_support_g_kwh": [0, 1000],
        "summary": summaries,
        "scope": "V345 annual full-plant minus monthly above-designated-size feasible monthly residual; cellwise relaxation",
        "limitations": [
            "Residual can vary independently in each period; annual cross-period mass constraint is not jointly solved.",
            "Unresolved residual uses the province total installed capacity as an independent hourly cap; this is conservative and may over-allocate total capacity.",
            "The two EF/residual endpoint combinations and residual mass endpoints follow V346 monotonicity construction; widths are conditional outer hulls.",
            "V320 is NBS-derived OREO transcription, not raw portal response; there is no Chinese hourly CIF label or empirical coverage score.",
        ],
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            V320, V341, V345, ROOT / "scripts/v342_china_sharp_group_hull.py",
            ROOT / "docs/V374_LOCAL_MIX_WITH_SCOPE_RESIDUAL_PROTOCOL.md", Path(__file__),
        )},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"province-period-capacity-threshold rows={len(detail)}")


if __name__ == "__main__":
    main()
