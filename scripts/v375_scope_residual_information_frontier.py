"""V375: map local-mix value against the allowed monthly scope-remainder range."""

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
OUT = ROOT / "results/v375-scope-residual-information-frontier-r1"
TECHS = ("thermal", "hydro", "nuclear", "wind", "solar")
PERIOD_MONTH = {"Jan-Feb": 2, **{f"{m:02d}": m for m in range(3, 13)}}
ALPHAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
TAU = 300
VINTAGE = 2022


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residual_hull(margins, caps, residual_hi, hours):
    candidates = []
    for residual in {F(0), residual_hi}:
        for thermal_ef, residual_ef in ((400, 0), (1000, 1000)):
            factors = {g: 0 for g in GROUPS}
            factors["thermal"] = thermal_ef
            factors["unresolved"] = residual_ef
            full_margins = {**margins, "unresolved": residual}
            full_caps = {**caps, "unresolved": max(F(1), caps["unresolved"])}
            values = sharp_group_set(full_margins, full_caps, factors, TAU, hours)
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
    pooled = complete.groupby("month")[[*TECHS, "classified_gwh"]].sum(min_count=30)
    share = {int(m): F(str(r.thermal)) / F(str(r.classified_gwh)) for m, r in pooled.iterrows()}

    source = pd.read_csv(V341)
    source = source.loc[source["included_in_complete_case"].astype(bool)].copy()
    residual = pd.read_csv(V345)
    residual = residual.loc[residual["capacity_vintage"].eq(VINTAGE)].copy()
    source_keys = set(zip(source["province"], source["period"]))
    residual = residual.loc[[
        (p, period) in source_keys for p, period in zip(residual["province"], residual["period"])
    ]].copy()
    residual_map = residual.set_index(["province", "period"])

    common = pd.read_csv(V373)
    common = common.loc[
        common["capacity_vintage"].eq(VINTAGE)
        & common["threshold_g_kwh"].eq(TAU)
        & common["national_mix_feasible"].astype(bool)
    ].copy()
    common_keys = set(zip(common["province"], common["period"]))
    if len(common_keys) != 262:
        raise ValueError(f"Expected 262 V373 paired-feasible province-periods; found {len(common_keys)}")

    rows = []
    subset = source.loc[[
        (p, period) in common_keys for p, period in zip(source["province"], source["period"])
    ]]
    for rec in subset.to_dict("records"):
        province, period = rec["province"], rec["period"]
        month, hours = PERIOD_MONTH[period], int(rec["hours"])
        local = {g: F(str(rec[f"{g}_mwh"])) for g in TECHS}
        total = sum(local.values(), F(0))
        national = dict(local)
        national["thermal"] = total * share[month]
        nonthermal = total - local["thermal"]
        if nonthermal == 0 and national["thermal"] != total:
            raise ValueError(f"nonthermal redistribution undefined for {province} {period}")
        if nonthermal:
            for tech in TECHS:
                if tech != "thermal":
                    national[tech] = local[tech] * (total - national["thermal"]) / nonthermal
        caps = {g: F(str(rec[f"{VINTAGE}_{g}_cap_mw"])) for g in TECHS}
        caps["unresolved"] = F(str(rec[f"{VINTAGE}_unresolved_cap_mw"]))
        residual_max = F(str(residual_map.loc[(province, period), "monthly_residual_max_gwh"])) * 1000
        for alpha in ALPHAS:
            high = residual_max * F(str(alpha))
            local_bounds = residual_hull(local, caps, high, hours)
            national_bounds = residual_hull(national, caps, high, hours)
            rows.append({
                "province": province, "period": period, "hours": hours, "alpha": alpha,
                "residual_upper_gwh": float(high / 1000),
                "local_feasible": local_bounds is not None,
                "national_feasible": national_bounds is not None,
                "local_width_h": None if local_bounds is None else local_bounds[1] - local_bounds[0],
                "national_width_h": None if national_bounds is None else national_bounds[1] - national_bounds[0],
                "local_width_reduction_h": None if local_bounds is None or national_bounds is None else national_bounds[1] - national_bounds[0] - (local_bounds[1] - local_bounds[0]),
            })

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "province_period_frontier.csv", index=False)
    summaries = []
    for alpha, frame in detail.groupby("alpha"):
        paired = frame.loc[frame["local_feasible"] & frame["national_feasible"]]
        summaries.append({
            "alpha": float(alpha), "paired_feasible_n": int(len(paired)),
            "local_infeasible_n": int((~frame["local_feasible"]).sum()),
            "national_infeasible_n": int((~frame["national_feasible"]).sum()),
            "local_mean_width_h": float(paired["local_width_h"].mean()) if len(paired) else None,
            "national_mean_width_h": float(paired["national_width_h"].mean()) if len(paired) else None,
            "mean_local_reduction_h": float(paired["local_width_reduction_h"].mean()) if len(paired) else None,
            "local_narrower_n": int((paired["local_width_reduction_h"] > 0).sum()),
            "local_wider_n": int((paired["local_width_reduction_h"] < 0).sum()),
            "equal_n": int((paired["local_width_reduction_h"] == 0).sum()),
            "local_full_month_hull_n": int((paired["local_width_h"] == paired["hours"]).sum()),
            "national_full_month_hull_n": int((paired["national_width_h"] == paired["hours"]).sum()),
        })
    pd.DataFrame(summaries).to_csv(OUT / "summary_by_residual_fraction.csv", index=False)
    summary = {
        "experiment": "V375 monthly residual constraint versus local-mix K-width information frontier",
        "status": "CELLWISE_OUTER_HULL_SENSITIVITY_NOT_EMPIRICAL_PREDICTION",
        "cohort": "V373 2023 paired-feasible 262 province-periods; 2022 lagged capacity",
        "threshold_g_kwh": TAU, "alpha_grid": list(ALPHAS),
        "thermal_ef_support_g_kwh": [400, 1000], "residual_ef_support_g_kwh": [0, 1000],
        "summary": summaries,
        "limitations": [
            "Alpha is a sensitivity fraction of the V345 annual scope gap, not an observed accuracy level or a public data field.",
            "Residual mass can be assigned independently in each month; annual cross-month residual conservation is not solved.",
            "Scope residual uses the V346 independent province-total capacity relaxation.",
            "V320 is an NBS-derived transcription, not raw response from the user-provided portal; no hourly China label or coverage metric is available.",
        ],
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            V320, V341, V345, V373, ROOT / "scripts/v342_china_sharp_group_hull.py",
            ROOT / "docs/V375_SCOPE_RESIDUAL_INFORMATION_FRONTIER_PROTOCOL.md", Path(__file__),
        )},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"province-period-alpha rows={len(detail)}")


if __name__ == "__main__":
    main()
