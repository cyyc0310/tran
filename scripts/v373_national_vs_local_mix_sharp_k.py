"""V373: compare sharp K hulls under local and pooled national monthly mix."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import GROUPS, sharp_group_set


ROOT = Path(__file__).resolve().parents[1]
V320_MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
V341_SOURCE = ROOT / "results/v341-china-complete-category-hull-r2/monthly_margin_capacity_audit.csv"
OUT = ROOT / "results/v373-national-vs-local-mix-sharp-k-r1"
TECHS = ("thermal", "hydro", "nuclear", "wind", "solar")
TAUS = (100, 300)
EF_ENDPOINTS = (400, 1000)
CAPACITY_VINTAGES = (2022, 2023)
PERIOD_MONTH = {"Jan-Feb": 2, **{f"{m:02d}": m for m in range(3, 13)}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hull_width(margins: dict[str, F], caps: dict[str, F], tau: int, hours: int) -> int | None:
    endpoint_sets = []
    for thermal_ef in EF_ENDPOINTS:
        factors = {g: 0 for g in GROUPS}
        factors["thermal"] = thermal_ef
        feasible = sharp_group_set(margins, caps, factors, tau, hours)
        if feasible is None:
            return None
        endpoint_sets.append(feasible)
    return max(max(values) for values in endpoint_sets) - min(min(values) for values in endpoint_sets)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(V320_MONTHLY)
    raw = raw.loc[raw["year"].eq(2023) & raw["technology"].isin((*TECHS, "total"))]
    pivot = raw.pivot_table(index=["province", "month"], columns="technology", values="generation_gwh", aggfunc="first")
    if not set(TECHS + ("total",)).issubset(pivot.columns):
        raise ValueError("V320 monthly table lacks expected technology columns")
    pivot["classified_gwh"] = pivot[list(TECHS)].sum(axis=1, min_count=len(TECHS))
    complete_pivot = pivot.loc[pivot["classified_gwh"].notna()]
    common_provinces = set(complete_pivot.index.get_level_values("province"))
    if len(common_provinces) != 30:
        raise ValueError(f"Expected 30 provinces with complete five-technology mix, found {len(common_provinces)}")
    pooled = complete_pivot.groupby("month")[[*TECHS, "classified_gwh"]].sum(min_count=30)
    if pooled[["thermal", "classified_gwh"]].isna().any().any():
        raise ValueError("At least one 2023 report bin lacks complete national five-technology data")
    national_share_by_month = {
        int(month): F(str(rec["thermal"])) / F(str(rec["classified_gwh"]))
        for month, rec in pooled.iterrows()
    }

    source = pd.read_csv(V341_SOURCE)
    source = source.loc[source["included_in_complete_case"].astype(bool)].copy()
    if source["province"].nunique() != 30 or len(source) != 330:
        raise ValueError("V341 complete-case cohort mismatch")

    rows: list[dict[str, object]] = []
    for rec in source.to_dict("records"):
        province, period = rec["province"], rec["period"]
        month = PERIOD_MONTH[period]
        national_share = national_share_by_month[month]
        hours = int(rec["hours"])
        local = {g: F(str(rec[f"{g}_mwh"])) for g in GROUPS}
        local_five_total = sum((local[g] for g in TECHS), F(0))
        local_thermal_share = local["thermal"] / local_five_total
        national = dict(local)
        national["thermal"] = local_five_total * national_share
        nonthermal_local = local_five_total - local["thermal"]
        if nonthermal_local:
            for g in TECHS:
                if g != "thermal":
                    national[g] = local[g] * (local_five_total - national["thermal"]) / nonthermal_local
        elif national["thermal"] != local_five_total:
            raise ValueError(f"Cannot redistribute zero local nonthermal mix: {province} {period}")

        for vintage in CAPACITY_VINTAGES:
            caps = {g: F(str(rec[f"{vintage}_{g}_cap_mw"])) for g in GROUPS}
            for tau in TAUS:
                local_width = hull_width(local, caps, tau, hours)
                national_width = hull_width(national, caps, tau, hours)
                rows.append({
                    "province": province,
                    "period": period,
                    "hours": hours,
                    "capacity_vintage": vintage,
                    "threshold_g_kwh": tau,
                    "national_thermal_share": float(national_share),
                    "local_thermal_share": float(local_thermal_share),
                    "local_feasible": local_width is not None,
                    "national_mix_feasible": national_width is not None,
                    "local_sharp_hull_width_hours": local_width,
                    "national_mix_sharp_hull_width_hours": national_width,
                    "paired_local_width_reduction_hours": (
                        national_width - local_width if local_width is not None and national_width is not None else None
                    ),
                })

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "province_period_comparison.csv", index=False)
    summary_rows = []
    for (vintage, tau), frame in detail.groupby(["capacity_vintage", "threshold_g_kwh"]):
        paired = frame.loc[frame["local_feasible"] & frame["national_mix_feasible"]]
        summary_rows.append({
            "capacity_vintage": int(vintage),
            "threshold_g_kwh": int(tau),
            "province_periods": int(len(frame)),
            "local_feasible_n": int(frame["local_feasible"].sum()),
            "national_mix_feasible_n": int(frame["national_mix_feasible"].sum()),
            "both_feasible_n": int(len(paired)),
            "national_mix_infeasible_n": int((~frame["national_mix_feasible"]).sum()),
            "local_mean_width_paired_h": float(paired["local_sharp_hull_width_hours"].mean()) if len(paired) else None,
            "national_mean_width_paired_h": float(paired["national_mix_sharp_hull_width_hours"].mean()) if len(paired) else None,
            "mean_width_reduction_from_local_mix_h": float(paired["paired_local_width_reduction_hours"].mean()) if len(paired) else None,
            "local_narrower_n": int((paired["paired_local_width_reduction_hours"] > 0).sum()),
            "local_wider_n": int((paired["paired_local_width_reduction_hours"] < 0).sum()),
            "equal_width_n": int((paired["paired_local_width_reduction_hours"] == 0).sum()),
        })
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(OUT / "summary_by_capacity_threshold.csv", index=False)
    summary = {
        "experiment": "V373 sharp K hull comparison: provincial monthly mix versus national pooled mix",
        "status": "IDENTIFICATION_SENSITIVITY_NOT_FORECAST_VALIDATION",
        "n_rows": int(len(detail)),
        "n_provinces": int(detail["province"].nunique()),
        "capacity_vintages": list(CAPACITY_VINTAGES),
        "thresholds_g_kwh": list(TAUS),
        "thermal_ef_endpoints_g_kwh": list(EF_ENDPOINTS),
        "group_factors": "thermal EF varies over endpoint hull [400,1000] g/kWh; hydro/nuclear/wind/solar/unresolved fixed at 0 direct g/kWh",
        "summary": summary_rows,
        "limitations": [
            "V320 OREO/NBS transcription, not raw response from the user-provided NBS portal; above-designated-size scope.",
            "National-mix arm preserves each province's five-category total and annual unexplained remainder while replacing its within-five-category shares with pooled 2023 shares.",
            "Annual installed capacity is a retrospective group-capacity proxy, not hourly available capacity.",
            "Sharp hulls are conditional identification sets; no Chinese hourly labels, empirical interval coverage, or forecast skill are evaluated.",
        ],
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            V320_MONTHLY, V341_SOURCE, ROOT / "scripts/v342_china_sharp_group_hull.py",
            ROOT / "docs/V373_NATIONAL_VS_LOCAL_MIX_SHARP_K_PROTOCOL.md", Path(__file__),
        )},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(summary_frame.to_string(index=False))
    print(f"province-period-capacity-threshold rows={len(detail)}")


if __name__ == "__main__":
    main()
