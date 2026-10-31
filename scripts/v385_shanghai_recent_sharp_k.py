"""V385: exact recent Shanghai sharp-K comparison using user NBS exports."""

from __future__ import annotations

import hashlib
import json
import sys
import csv
from calendar import monthrange
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v342_china_sharp_group_hull import GROUPS, sharp_group_set  # noqa: E402

OUT = ROOT / "results/v385-shanghai-recent-sharp-k-r2"
DATA = ROOT / "data/nbs_user_exports/shanghai"
V320 = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
V341 = ROOT / "results/v341-china-complete-category-hull-r2/monthly_margin_capacity_audit.csv"
PROTOCOL = ROOT / "docs/V385_SHANGHAI_RECENT_SHARP_K_PROTOCOL.md"
SOURCE_FILES = {"thermal": DATA / "thermal.csv", "solar": DATA / "solar.csv", "wind": DATA / "wind.csv"}
TECHS = ("thermal", "hydro", "nuclear", "wind", "solar")
TAUS = (100, 300)
EF_ENDPOINTS = (400, 1000)
MONTHS = [(2025, m) for m in (9, 10, 11, 12)] + [(2026, m) for m in range(3, 10)]
PERIOD = lambda year, month: f"{month:02d}"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_current_values(path: Path, metric: str) -> dict[tuple[int, int], F]:
    # The downloaded CSV is comma-delimited; cells retain trailing tabs.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        table = list(csv.reader(handle))
    header = table[2]
    matches = [r for r in table if r and r[0].strip().startswith(metric + "发电量当期值")]
    if len(matches) != 1:
        raise ValueError(f"Expected one current-value row in {path.name}")
    values = matches[0]
    out: dict[tuple[int, int], F] = {}
    for label, raw in zip(header[1:], values[1:]):
        label = str(label).strip().replace("\t", "")
        if not label or "年" not in label:
            continue
        year_s, month_s = label.split("年", 1)
        month_s = month_s.replace("月", "").strip()
        value = str(raw).strip()
        if not value or value.lower() == "nan":
            continue
        out[(int(year_s), int(month_s))] = F(value)
    return out


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {OUT}")
    OUT.mkdir(parents=True)
    raw_sh = {
        "thermal": load_current_values(SOURCE_FILES["thermal"], "火力"),
        "solar": load_current_values(SOURCE_FILES["solar"], "太阳能"),
        "wind": load_current_values(SOURCE_FILES["wind"], "风力"),
    }
    rows = []
    for year, month in MONTHS:
        if any((year, month) not in vals for vals in raw_sh.values()):
            raise ValueError(f"Missing a direct monthly value for {year}-{month:02d}")
        # CSV value is 0.1 billion kWh; 1 unit = 100 GWh = 100,000 MWh.
        local = {g: F(0) for g in GROUPS}
        for g in ("thermal", "wind", "solar"):
            local[g] = raw_sh[g][(year, month)] * 100_000

        sub = pd.read_csv(V320)
        sub = sub.loc[sub.year.eq(year) & sub.month.eq(month) & sub.technology.isin(TECHS)]
        pv = sub.pivot(index="province", columns="technology", values="generation_gwh")
        complete = pv.dropna(subset=list(TECHS))
        if "Shanghai" not in complete.index:
            raise ValueError(f"No complete source row for Shanghai {year}-{month:02d}")
        # Cross-check direct user export against the matching V320 values, including explicit zeros.
        for g in TECHS:
            expected_gwh = float(local[g] / 1000)
            observed = float(complete.loc["Shanghai", g])
            if abs(expected_gwh - observed) > 1e-8:
                raise ValueError(f"NBS export/V320 mismatch: {year}-{month:02d} {g} {expected_gwh} != {observed}")

        national_totals = complete[list(TECHS)].sum(axis=0)
        national_share = F(str(national_totals["thermal"])) / F(str(national_totals.sum()))
        local_total = sum((local[g] for g in TECHS), F(0))
        national = {g: F(0) for g in GROUPS}
        national["thermal"] = local_total * national_share
        local_nonthermal = local_total - local["thermal"]
        nonthermal_mix = {g: local[g] for g in TECHS if g != "thermal"}
        nonthermal_sum = sum(nonthermal_mix.values(), F(0))
        if nonthermal_sum:
            for g, value in nonthermal_mix.items():
                national[g] = (local_total - national["thermal"]) * value / nonthermal_sum
        elif national["thermal"] != local_total:
            raise ValueError("Cannot reallocate national nonthermal share when local nonthermal mix is zero")

        cap_row = pd.read_csv(V341)
        cap_row = cap_row.loc[cap_row.province.eq("Shanghai") & cap_row.period.eq(PERIOD(year, month))]
        if len(cap_row) != 1:
            raise ValueError(f"Expected one Shanghai 2023-capacity row for period {year}-{month:02d}")
        cap_rec = cap_row.iloc[0]
        cec_caps = {g: F(str(cap_rec[f"2023_{g}_cap_mw"])) for g in GROUPS}
        monthly_only_caps_local = {g: max(local[g], F(1)) for g in GROUPS}
        monthly_only_caps_national = {g: max(national[g], F(1)) for g in GROUPS}
        hours = monthrange(year, month)[1] * 24
        for tau in TAUS:
            for capacity_contract, capacity_by_arm in (
                ("cec_2023", {"local": cec_caps, "national_mix": cec_caps}),
                ("monthly_only_nonbinding", {"local": monthly_only_caps_local, "national_mix": monthly_only_caps_national}),
            ):
                endpoint_sets = {}
                start = len(rows)
                for arm_name, margins in (("local", local), ("national_mix", national)):
                    caps = capacity_by_arm[arm_name]
                    vals = []
                    for thermal_ef in EF_ENDPOINTS:
                        factors = {g: 0 for g in GROUPS}
                        factors["thermal"] = thermal_ef
                        vals.append(sharp_group_set(margins, caps, factors, tau, hours))
                    endpoint_sets[arm_name] = vals
                    feasible = all(v is not None for v in vals)
                    width = max(max(v) for v in vals) - min(min(v) for v in vals) if feasible else None
                    rows.append({
                        "year": year, "month": month, "province": "Shanghai", "hours": hours,
                        "complete_national_provinces": int(len(complete)), "threshold_g_kwh": tau,
                        "capacity_contract": capacity_contract, "arm": arm_name, "feasible": feasible,
                        "k_hull_width_hours": width,
                        "k_hull_lower": min(min(v) for v in vals) if feasible else None,
                        "k_hull_upper": max(max(v) for v in vals) if feasible else None,
                        "national_thermal_share": float(national_share),
                        "local_thermal_share": float(local["thermal"] / local_total),
                        "thermal_ef_low": EF_ENDPOINTS[0], "thermal_ef_high": EF_ENDPOINTS[1],
                    })
                l, n = endpoint_sets["local"], endpoint_sets["national_mix"]
                paired = all(v is not None for v in l + n)
                delta = (max(max(v) for v in n) - min(min(v) for v in n)) - (max(max(v) for v in l) - min(min(v) for v in l)) if paired else None
                for row in rows[start:]:
                    row["paired_national_minus_local_width_hours"] = delta

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "shanghai_recent_exact_k_by_month.csv", index=False)
    summaries = []
    for (capacity_contract, tau), frame in detail.groupby(["capacity_contract", "threshold_g_kwh"]):
        wide = frame.pivot(index=["year", "month"], columns="arm", values="k_hull_width_hours")
        paired = wide.dropna()
        diffs = paired.national_mix - paired.local
        summaries.append({
            "threshold_g_kwh": int(tau),
            "capacity_contract": capacity_contract,
            "month_count": int(len(wide)),
            "local_feasible_n": int(frame.loc[frame.arm.eq("local"), "feasible"].sum()),
            "national_mix_feasible_n": int(frame.loc[frame.arm.eq("national_mix"), "feasible"].sum()),
            "paired_n": int(len(paired)),
            "local_mean_width_h_paired": float(paired.local.mean()) if len(paired) else None,
            "national_mean_width_h_paired": float(paired.national_mix.mean()) if len(paired) else None,
            "mean_shrinkage_local_vs_national_h": float(diffs.mean()) if len(diffs) else None,
            "local_narrower_n": int((diffs > 0).sum()),
            "local_wider_n": int((diffs < 0).sum()),
            "equal_n": int((diffs == 0).sum()),
        })
    pd.DataFrame(summaries).to_csv(OUT / "summary_by_threshold.csv", index=False)
    files = [*SOURCE_FILES.values(), V320, V341, ROOT / "scripts/v342_china_sharp_group_hull.py", PROTOCOL, Path(__file__)]
    summary = {
        "experiment": "V385 exact sharp-K information comparison on directly exported recent Shanghai monthly mix",
        "status": "CONDITIONAL_IDENTIFICATION_NOT_FORECAST_VALIDATION",
        "months": [f"{y}-{m:02d}" for y, m in MONTHS],
        "hydro_nuclear": "set to zero per user's statement; verified against V320 for matching months",
        "unit_conversion": "NBS export 1e8 kWh -> GWh multiply by 100; GWh -> MWh multiply by 1000",
        "national_mix_denominator": "same-month pooled five-technology totals among provinces with complete five-category rows",
        "thermal_factor_endpoints_g_kwh": list(EF_ENDPOINTS),
        "other_direct_factors_g_kwh": 0,
        "capacity_contracts": ["2023 CEC group-capacity proxy", "nonbinding monthly-only control; not a physical capacity estimate"],
        "summary": summaries,
        "limitations": [
            "Single province, 11 months, no Chinese hourly target labels or empirical interval coverage.",
            "Shanghai direct export values match V320 transcription; V320 is still not a raw official portal response snapshot.",
            "2023 installed capacity is a retrospective proxy for 2025-2026 and may be stale.",
            "Scope residual, imports, within-thermal fuel composition, hourly availability, and hourly dispatch are unobserved.",
            "Any width change is a conditional information-value comparison, not a forecast skill or uncertainty coverage result.",
        ],
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
