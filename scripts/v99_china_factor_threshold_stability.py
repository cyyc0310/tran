"""V99: quantify how factor uncertainty changes threshold-side fuel support in China monthly data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
PROTOCOL = ROOT / "docs/V99_CHINA_FACTOR_THRESHOLD_STABILITY_PROTOCOL.md"
OUT = ROOT / "results/v99-china-factor-threshold-stability-r1"
THRESHOLDS = (300, 400, 500, 600, 800, 900, 1000)
FACTORS = {
    "Coal": (900.0, 1100.0),
    "Gas": (360.0, 440.0),
    "Other fossil": (200.0, 600.0),
    "Bioenergy": (0.0, 0.0),
    "Hydro": (0.0, 0.0),
    "Nuclear": (0.0, 0.0),
    "Wind": (0.0, 0.0),
    "Solar": (0.0, 0.0),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    raw = pd.read_csv(INPUT)
    provinces = raw.loc[raw["Province code"].notna()].copy()
    atomic = list(FACTORS)
    required = atomic + ["Total generation"]
    pivot = provinces.pivot(index=["Province code", "Province", "Date"], columns="Electricity source", values="Generation (TWh)")
    missing = sorted(set(required) - set(pivot.columns))
    if missing or pivot[required].isna().any().any():
        raise ValueError(f"incomplete atomic province-month coverage: {missing}")
    monthly = pivot[required].copy()
    monthly["atomic_sum"] = monthly[atomic].sum(axis=1)
    monthly["generation_abs_gap_TWh"] = (monthly["atomic_sum"] - monthly["Total generation"]).abs()
    if monthly["generation_abs_gap_TWh"].max() > 0.0030001:
        raise ValueError("atomic categories fail V60 monthly generation reconciliation")
    monthly["year"] = pd.to_datetime(monthly.index.get_level_values("Date")).year
    rows = []
    for tau in THRESHOLDS:
        above = [fuel for fuel, (lo, hi) in FACTORS.items() if lo > tau]
        below = [fuel for fuel, (lo, hi) in FACTORS.items() if hi <= tau]
        ambiguous = [fuel for fuel, (lo, hi) in FACTORS.items() if lo <= tau < hi]
        if set(above) | set(below) | set(ambiguous) != set(atomic):
            raise AssertionError(f"factor classes do not partition categories at tau={tau}")
        for key, row in monthly.iterrows():
            total = float(row["Total generation"])
            vals = {fuel: float(row[fuel]) for fuel in atomic}
            if total <= 0:
                continue
            shares = {name: sum(vals[f] for f in fuels) / total for name, fuels in (("definitely_above", above), ("definitely_below", below), ("ambiguous", ambiguous))}
            if abs(sum(shares.values()) - 1) > 0.003 / total + 1e-8:
                raise AssertionError("generation shares do not reconcile")
            rows.append({"province_code": key[0], "province": key[1], "date": key[2], "year": int(row["year"]), "tau": tau,
                         "definitely_above_fuels": ",".join(above), "definitely_below_fuels": ",".join(below), "ambiguous_fuels": ",".join(ambiguous),
                         "definitely_above_TWh": sum(vals[f] for f in above), "definitely_below_TWh": sum(vals[f] for f in below), "ambiguous_TWh": sum(vals[f] for f in ambiguous),
                         "total_TWh": total, "definitely_above_share": shares["definitely_above"], "definitely_below_share": shares["definitely_below"], "ambiguous_share": shares["ambiguous"],
                         "coal_TWh": vals["Coal"], "gas_TWh": vals["Gas"], "other_fossil_TWh": vals["Other fossil"]})
    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "province_month_threshold_support.csv", index=False)
    # Energy-weighted summaries prevent small provinces receiving the same weight as large systems.
    summary_rows = []
    for tau, group in detail.groupby("tau"):
        for period, subset in [("all", group), *[(str(y), group[group.year.eq(y)]) for y in sorted(group.year.unique())]]:
            den = subset.total_TWh.sum()
            summary_rows.append({"tau": int(tau), "period": period, "province_months": int(subset.shape[0]),
                                "generation_TWh": float(den), "definitely_above_share": float(subset.definitely_above_TWh.sum() / den),
                                "definitely_below_share": float(subset.definitely_below_TWh.sum() / den),
                                "ambiguous_share": float(subset.ambiguous_TWh.sum() / den),
                                "mean_province_month_ambiguous_share": float(subset.ambiguous_share.mean()),
                                "ambiguous_fuels": str(subset.ambiguous_fuels.iloc[0])})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "factor_threshold_summary.csv", index=False)
    metadata = {"status": "CHINA_MONTHLY_FACTOR_SUPPORT_SENSITIVITY_NOT_HOURLY_VALIDATION", "province_months": int(monthly.shape[0]),
                "thresholds": list(THRESHOLDS), "factor_sensitivity_gCO2_kWh": FACTORS, "input_sha256": sha(INPUT), "protocol_sha256": sha(PROTOCOL),
                "script_sha256": sha(Path(__file__)), "max_monthly_generation_gap_TWh": float(monthly.generation_abs_gap_TWh.max()),
                "limitations": ["The factor bounds are transparent sensitivity assumptions, not official factor measurements.",
                                "The public Ember monthly source is estimated/revised and has no historical vintage chain.",
                                "This measures category stability only; monthly data do not identify hourly dispatch, CIF, or exceedance counts."]}
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(summary[summary.period.eq("all")].to_string(index=False))


if __name__ == "__main__":
    main()
