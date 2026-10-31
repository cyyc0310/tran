"""V383: quantify what annual fuel margins identify after monthly thermal totals."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
OUT = ROOT / "results/v383-annual-monthly-fuel-identification-r1"

# Table 3 values in 100 million kWh, carried from V382.
ANNUAL_LISTED_SUPPLY_GWH = {
    "coal": Decimal("61133"),
    "gas": Decimal("14991"),
    "oil": Decimal("152"),
}


def main() -> None:
    panel = pd.read_csv(PANEL)
    rows = panel.loc[
        panel["province"].eq("Shanghai")
        & panel["year"].eq(2020)
        & panel["technology"].eq("thermal")
    ].sort_values("month")
    if len(rows) != 11:
        raise ValueError(f"Expected 11 NBS reporting bins including Jan-Feb cumulative, got {len(rows)}")

    bins = [Decimal(str(value)) for value in rows["generation_gwh"]]
    annual_nbs_thermal = sum(bins)
    listed_total = sum(ANNUAL_LISTED_SUPPLY_GWH.values())
    # Residual only closes the published coal/gas/oil *supply* estimates to the
    # NBS thermal gross-generation annual margin; its actual technology and EF
    # are unknown, so keep it explicitly unclassified.
    margins = dict(ANNUAL_LISTED_SUPPLY_GWH)
    margins["unclassified_scope_residual"] = annual_nbs_thermal - listed_total
    if margins["unclassified_scope_residual"] < 0:
        raise ValueError("Fuel-specific margins exceed the NBS thermal annual total")
    if sum(margins.values()) != annual_nbs_thermal:
        raise AssertionError("Annual columns do not close after explicit residual allocation")

    month_rows = []
    for period, bucket in zip(rows["month"].astype(int), bins):
        for fuel, annual_margin in margins.items():
            # Sharp marginal bounds from x_mf>=0, row sum G_m, column sum A_f.
            lower_gwh = max(Decimal(0), annual_margin - (annual_nbs_thermal - bucket))
            upper_gwh = min(bucket, annual_margin)
            month_rows.append(
                {
                    "reporting_bin": period,
                    "nbs_thermal_gwh": float(bucket),
                    "fuel_group": fuel,
                    "annual_margin_gwh": float(annual_margin),
                    "monthly_lower_gwh": float(lower_gwh),
                    "monthly_upper_gwh": float(upper_gwh),
                    "share_lower": float(lower_gwh / bucket),
                    "share_upper": float(upper_gwh / bucket),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    bounds = pd.DataFrame(month_rows)
    bounds.to_csv(OUT / "monthly_fuel_share_bounds.csv", index=False)
    n_bins = len(bins)
    n_groups = len(margins)
    polytope_dimension = (n_bins - 1) * (n_groups - 1)
    by_fuel = {}
    for fuel, group in bounds.groupby("fuel_group", sort=False):
        by_fuel[fuel] = {
            "bins_with_share_lower_0_and_upper_1": int(((group.share_lower < 1e-12) & (group.share_upper > 1 - 1e-12)).sum()),
            "mean_monthly_share_bound_width": float((group.share_upper - group.share_lower).mean()),
            "max_monthly_share_upper": float(group.share_upper.max()),
            "min_monthly_share_lower": float(group.share_lower.min()),
        }

    summary = {
        "experiment": "V383 annual fuel margins plus monthly thermal margins identification audit",
        "province": "Shanghai",
        "year": 2020,
        "monthly_margin_source": "V320 OREO-transcribed NBS panel; Jan-Feb cumulative is one reporting bin",
        "annual_fuel_margin_source": "V382 Table 3 derived coal/gas/oil supply estimates",
        "n_reporting_bins": n_bins,
        "fuel_groups_in_transportation_table": n_groups,
        "nbs_annual_thermal_gwh": float(annual_nbs_thermal),
        "annual_fuel_margins_gwh": {key: float(value) for key, value in margins.items()},
        "transportation_polytope_dimension_if_all_margins_positive": polytope_dimension,
        "dimension_formula": "M*K - (M+K-1) = (M-1)*(K-1); row and column totals are known, internal monthly allocation remains free",
        "fuel_share_bounds": by_fuel,
        "monthly_coal_and_gas_each_allow_full_0_to_1_share_in_all_bins": bool(
            (bounds.loc[bounds.fuel_group.isin(["coal", "gas"])]
             .groupby("fuel_group")
             .apply(lambda group: ((group.share_lower < 1e-12) & (group.share_upper > 1 - 1e-12)).all(), include_groups=False)
             .all())
        ),
        "interpretation": "Annual fuel totals constrain yearly totals but do not identify monthly fuel composition: coal and gas each individually retain [0,1] marginal share in all 11 reporting bins. At least 30 independent additional month-by-fuel constraints are needed to identify the 11x4 allocation table under the idealized fixed margins; an annual proportion copied to each month is a structural assumption, not an implication of the data.",
        "caveat": "The 4,074 GWh unclassified group is an explicit cross-source closure residual, not a confirmed fourth technology. This calculation is a sensitivity audit, not a validated Chinese monthly fuel allocation.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
