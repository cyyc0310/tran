"""Descriptive cross-estimand comparison of Ember 2022 generation EF and MEE EF."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
MEE_SCRIPT = ROOT / "scripts/china_projection_annual_anchor_probe.py"
OUT = ROOT / "results/china-monthly-ember-audit-r2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    spec = importlib.util.spec_from_file_location("annual_anchor", MEE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen official-factor values")
    annual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(annual)
    frame = pd.read_csv(SOURCE_CSV, low_memory=False)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame = frame.loc[(frame.Date.dt.year == 2022)
                      & ~frame["Is aggregated source"].astype(str).str.lower().eq("true")]
    totals = frame.groupby("Province", as_index=True).agg(
        generation_twh=("Generation (TWh)", "sum"), emissions_mtco2e=("Emissions (MtCO2e)", "sum"))
    totals["ember_generation_ef_g_per_kwh"] = totals.emissions_mtco2e / totals.generation_twh * 1000.0
    totals["mee_2022_consumption_ef_g_per_kwh"] = pd.Series(annual.MEE_2022_KG_PER_KWH) * 1000.0
    totals = totals.drop(index="China (total)", errors="ignore").dropna()
    totals["generation_minus_consumption_ef"] = (
        totals.ember_generation_ef_g_per_kwh - totals.mee_2022_consumption_ef_g_per_kwh)
    totals["absolute_difference"] = totals.generation_minus_consumption_ef.abs()
    totals = totals.sort_values("Province")
    if len(totals) != 30:
        raise ValueError(f"Expected 30 matching provinces, found {len(totals)}")
    OUT.mkdir(parents=True, exist_ok=False)
    totals.to_csv(OUT / "province_comparison.csv")
    corr = float(totals[["ember_generation_ef_g_per_kwh", "mee_2022_consumption_ef_g_per_kwh"]]
                 .corr().iloc[0, 1])
    diff = totals.generation_minus_consumption_ef
    summary = {
        "status": "DESCRIPTIVE_CROSS_ESTIMAND_COMPARISON_NOT_VALIDATION",
        "comparison": "Ember 2022 monthly province fuel generation/emissions aggregated to generation EF vs MEE/NBS 2022 official province consumption EF",
        "n_provinces": int(len(totals)),
        "pearson_correlation": corr,
        "mean_absolute_difference_g_per_kwh": float(totals.absolute_difference.mean()),
        "median_absolute_difference_g_per_kwh": float(totals.absolute_difference.median()),
        "mean_signed_difference_g_per_kwh": float(diff.mean()),
        "minimum_signed_difference_g_per_kwh": float(diff.min()),
        "maximum_signed_difference_g_per_kwh": float(diff.max()),
        "source_csv_sha256": sha256(SOURCE_CSV),
        "mee_values_and_protocol_source_sha256": sha256(MEE_SCRIPT),
        "limitations": [
            "The estimands differ: Ember is generation-side and MEE is consumption-side with electricity-flow treatment.",
            "This is not a forecast validation or a calibration mapping; high correlation does not make the levels interchangeable.",
            "Ember monthly province values are constructed estimates; this uses the downloaded current vintage, not historical as-of versions.",
            "No hourly CIF labels are evaluated."
        ]
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
