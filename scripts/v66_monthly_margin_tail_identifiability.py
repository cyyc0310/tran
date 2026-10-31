"""Sharp hourly threshold-count bounds from monthly fuel margins alone."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
PROTOCOL = ROOT / "docs/V66_MONTHLY_MARGIN_TAIL_IDENTIFIABILITY_PROTOCOL.md"
OUT = ROOT / "results/v66-monthly-margin-tail-identifiability-r2"
FUELS = ("Coal", "Gas", "Other fossil", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy")
EF = {"Coal": 1000, "Gas": 400, "Other fossil": 400, "Nuclear": 0,
      "Hydro": 0, "Wind": 0, "Solar": 0, "Bioenergy": 0}
TAUS = (100, 300, 600, 900)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sharp_k_bounds(margins: dict[str, Fraction], tau: int, hours: int) -> tuple[int, int, str, str]:
    positive = sum((mass for fuel, mass in margins.items() if EF[fuel] > tau), Fraction(0))
    negative = sum((mass for fuel, mass in margins.items() if EF[fuel] < tau), Fraction(0))
    neutral = sum((mass for fuel, mass in margins.items() if EF[fuel] == tau), Fraction(0))
    if positive == 0:
        return 0, 0, "no_positive_excess_fuel", "K=0"
    weighted_excess = sum((mass * (EF[fuel] - tau) for fuel, mass in margins.items()), Fraction(0))
    if negative > 0:
        if weighted_excess > 0:
            return 1, hours, "mixed_support_positive_month_excess", "K=1..T"
        return 0, hours - 1, "mixed_support_nonpositive_month_excess", "K=0..T-1"
    if neutral > 0:
        return 1, hours, "positive_and_neutral_support", "K=1..T"
    return hours, hours, "strictly_positive_support_only", "K=T"


def main() -> None:
    protocol_hash = sha256(PROTOCOL)
    frame = pd.read_csv(INPUT, dtype={"Generation (TWh)": str}, low_memory=False)
    frame = frame.loc[~frame["Is aggregated source"].astype(str).str.lower().eq("true")]
    frame = frame.loc[frame["Electricity source"].isin(FUELS)]
    frame = frame.loc[frame["Date"].between("2016-01-01", "2024-12-31")]
    rows = []
    for (province, date), group in frame.groupby(["Province", "Date"], sort=True):
        if province == "China (total)":
            continue
        by_fuel = group.set_index("Electricity source")
        if set(FUELS).difference(by_fuel.index) or by_fuel.index.duplicated().any():
            continue
        margins = {fuel: Fraction(str(by_fuel.loc[fuel, "Generation (TWh)"])) for fuel in FUELS}
        if any(v < 0 for v in margins.values()) or sum(margins.values()) <= 0:
            continue
        total = sum(margins.values())
        carbon = sum((margins[fuel] * EF[fuel] for fuel in FUELS), Fraction(0))
        mean = carbon / total
        month = pd.Timestamp(date)
        hours = int(month.days_in_month * 24)
        support = "+".join(f for f in FUELS if margins[f] > 0)
        for tau in TAUS:
            lo, hi, case, set_label = sharp_k_bounds(margins, tau, hours)
            rows.append({"province": province, "month": month.strftime("%Y-%m"), "hours": hours,
                         "tau_g_kwh": tau, "weighted_mean_g_kwh": float(mean),
                         "monthly_excess_sign": "positive" if mean > tau else "zero" if mean == tau else "negative",
                         "positive_ef_support": int(sum(margins[f] > 0 and EF[f] > tau for f in FUELS)),
                         "negative_ef_support": int(sum(margins[f] > 0 and EF[f] < tau for f in FUELS)),
                         "neutral_ef_support": int(sum(margins[f] > 0 and EF[f] == tau for f in FUELS)),
                         "active_fuels": int(sum(margins[f] > 0 for f in FUELS)),
                         "fuel_support": support, "sharp_k_min": lo, "sharp_k_max": hi,
                         "sharp_k_width": hi - lo, "fraction_min": lo / hours, "fraction_max": hi / hours,
                         "certifies_no_exceedance": lo == hi == 0,
                         "certifies_at_least_one": lo > 0,
                         "case": case, "sharp_set": set_label})
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No complete province-month fuel margins")
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "province_month_tail_bounds.csv", index=False)
    summary = (detail.groupby("tau_g_kwh").agg(province_months=("month", "count"), provinces=("province", "nunique"),
               mean_weighted_cif=("weighted_mean_g_kwh", "mean"), mean_k_width=("sharp_k_width", "mean"),
               median_k_width=("sharp_k_width", "median"),
               certified_zero_months=("certifies_no_exceedance", "sum"),
               certified_positive_months=("certifies_at_least_one", "sum"),
               positive_excess_mean_months=("monthly_excess_sign", lambda x: (x == "positive").sum()),
               negative_or_zero_excess_months=("monthly_excess_sign", lambda x: (x != "positive").sum()))
               .reset_index())
    # Width as an hourly fraction is calculated per row because month lengths differ.
    fraction_width = detail.assign(fraction_width=detail.fraction_max - detail.fraction_min).groupby(
        "tau_g_kwh").fraction_width.mean().rename("mean_fraction_width")
    summary = summary.merge(fraction_width, on="tau_g_kwh")
    summary.to_csv(OUT / "threshold_summary.csv", index=False)
    case_summary = (detail.groupby(["tau_g_kwh", "case"]).agg(months=("month", "count"),
                    mean_k_width=("sharp_k_width", "mean"),
                    certified_zero_months=("certifies_no_exceedance", "sum"),
                    certified_positive_months=("certifies_at_least_one", "sum"))
                    .reset_index())
    case_summary.to_csv(OUT / "case_summary.csv", index=False)
    payload = {"status": "EXACT_POSTHOC_SHARP_BOUNDS_FROM_MONTHLY_MARGINS_ONLY_NOT_FORECAST",
               "province_month_threshold_rows": len(detail), "province_months": int(detail[["province", "month"]].drop_duplicates().shape[0]),
               "provinces": sorted(detail.province.unique().tolist()), "period": [detail.month.min(), detail.month.max()],
               "threshold_summary": summary.to_dict("records"), "case_summary": case_summary.to_dict("records"),
               "ef_gco2e_kwh": EF, "input_sha256": sha256(INPUT), "script_sha256": sha256(Path(__file__)),
               "protocol_sha256": protocol_hash,
               "limitations": ["Monthly Ember margins are retrospective estimates; no release-vintage forecast claim.",
                              "Assumes only nonnegative hourly fuel allocations and positive hourly total generation; no capacity, ramp, daily margin, or minimum-output constraints.",
                              "The exact set is conditional on fixed illustrative fuel emission factors.",
                              "This is an identifiability result, not empirical hourly tail coverage or forecast skill."]}
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("Frozen protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(summary.to_string(index=False))
    print(case_summary.to_string(index=False))
    print(f"rows={len(detail)} outputs={OUT}")


if __name__ == "__main__":
    main()
