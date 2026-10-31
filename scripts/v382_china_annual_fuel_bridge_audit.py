"""V382: audit compatibility of a published 2020 Shanghai fuel split with NBS monthly totals."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
OUT = ROOT / "results/v382-china-annual-fuel-bridge-audit-r1"

# Transcribed from the 2020 Shanghai row of Table 3 in the public 1mi1 report.
# Table unit: 100 million kWh. Values represent reported supply quantities, not
# independently observed gross generation by fuel.
SOURCE_TABLE_100M_KWH = {
    "total_supply": Decimal("821.62"),
    "coal_supply": Decimal("611.33"),
    "gas_supply": Decimal("149.91"),
    "oil_supply": Decimal("1.52"),
}
SOURCE_URL = "https://study.1mi1.org/articleAttachments/upload_9e0a7bd8b24583aa4daa4eefd5defe11.pdf"


def main() -> None:
    panel = pd.read_csv(PANEL)
    panel = panel.loc[
        panel["province"].eq("Shanghai")
        & panel["year"].eq(2020)
        & panel["technology"].isin(["thermal", "total"])
    ]
    thermal_rows = panel.loc[panel["technology"].eq("thermal")]
    total_rows = panel.loc[panel["technology"].eq("total")]
    if thermal_rows.empty or total_rows.empty:
        raise ValueError("Missing Shanghai 2020 thermal or total series in V320 panel")

    # NBS Jan-Feb is one cumulative reporting bin; it is included once as month 2.
    nbs_thermal_gwh = Decimal(str(thermal_rows["generation_gwh"].sum()))
    nbs_total_gwh = Decimal(str(total_rows["generation_gwh"].sum()))
    nbs_thermal_bins = int(thermal_rows["month"].nunique())
    if nbs_thermal_bins != 11:
        raise ValueError(f"Expected 11 NBS reporting bins including Jan-Feb, got {nbs_thermal_bins}")

    source = SOURCE_TABLE_100M_KWH
    source_gwh = {key: value * Decimal(100) for key, value in source.items()}
    subtype_sum = sum(source_gwh[key] for key in ("coal_supply", "gas_supply", "oil_supply"))
    residual = source_gwh["total_supply"] - subtype_sum
    comparison_rows = [
        {"source": "NBS-derived V320 panel", "measure": "thermal_generation", "gwh": float(nbs_thermal_gwh), "semantics": "gross generation, Jan-Feb cumulative bin plus Mar-Dec"},
        {"source": "NBS-derived V320 panel", "measure": "total_generation", "gwh": float(nbs_total_gwh), "semantics": "all listed generation, same reporting bins"},
        {"source": "1mi1 Table 3", "measure": "total_supply", "gwh": float(source_gwh["total_supply"]), "semantics": "reported supply; derived from yearbook inputs"},
        {"source": "1mi1 Table 3", "measure": "coal_supply", "gwh": float(source_gwh["coal_supply"]), "semantics": "derived fuel-specific supply"},
        {"source": "1mi1 Table 3", "measure": "gas_supply", "gwh": float(source_gwh["gas_supply"]), "semantics": "derived fuel-specific supply"},
        {"source": "1mi1 Table 3", "measure": "oil_supply", "gwh": float(source_gwh["oil_supply"]), "semantics": "derived fuel-specific supply"},
        {"source": "1mi1 Table 3", "measure": "unallocated_table_residual", "gwh": float(residual), "semantics": "table total minus listed coal/gas/oil; category not identified here"},
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(OUT / "shanghai_2020_bridge.csv", index=False)

    result = {
        "experiment": "V382 annual fuel-split source-to-monthly-NBS compatibility audit",
        "source_url": SOURCE_URL,
        "source_year": 2020,
        "province": "Shanghai",
        "nbs_panel": str(PANEL.relative_to(ROOT)),
        "nbs_reporting_bins": nbs_thermal_bins,
        "nbs_jan_feb_semantics": "single cumulative reporting bin; counted once, not split into January and February",
        "nbs_thermal_generation_gwh": float(nbs_thermal_gwh),
        "published_table_total_supply_gwh": float(source_gwh["total_supply"]),
        "published_table_coal_gas_oil_subtotal_gwh": float(subtype_sum),
        "published_table_unallocated_residual_gwh": float(residual),
        "table_fossil_subtype_shares_of_listed_subtotal": {
            "coal": float(source_gwh["coal_supply"] / subtype_sum),
            "gas": float(source_gwh["gas_supply"] / subtype_sum),
            "oil": float(source_gwh["oil_supply"] / subtype_sum),
        },
        "table_total_supply_minus_nbs_thermal_gwh": float(source_gwh["total_supply"] - nbs_thermal_gwh),
        "table_total_supply_relative_gap_vs_nbs_thermal": float((source_gwh["total_supply"] - nbs_thermal_gwh) / nbs_thermal_gwh),
        "table_coal_gas_oil_subtotal_minus_nbs_thermal_gwh": float(subtype_sum - nbs_thermal_gwh),
        "interpretation": "The published annual fuel split is a derived supply estimate with explicit allocation assumptions. Its total, listed fossil subtotal, and NBS gross thermal generation do not close. Treat only as a sensitivity prior until the scope difference and residual are resolved; do not hard-anchor monthly fuel generation to it.",
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
