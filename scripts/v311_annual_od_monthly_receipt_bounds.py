"""Sharp month-level gross-import bounds from annual OD and monthly generation.

The monthly China source is the audited Ember estimate in V294. This does not
estimate hourly CIF and intentionally does not use Ember's emissions/intensity
columns. The LP asks only which monthly receipt volumes remain feasible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "v310_zenodo_interprovincial_audit_r1"
EMBER_FILE = ROOT / "results" / "v294-china-ember-lineage-r1" / "ember_china_monthly_generation.csv"
OUT = ROOT / "results" / "v311-annual-od-monthly-receipt-bounds-r1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    od_full = pd.read_excel(DATA / "5_interprovincial transfer matrix.xlsx", index_col=0)
    od_full = od_full.iloc[:31, :31].fillna(0.0)

    monthly = pd.read_csv(EMBER_FILE)
    monthly = monthly.loc[
        monthly["Date"].astype(str).str.startswith("2023")
        & monthly["Is aggregated source"].eq(False)
        & monthly["Province"].ne("China (total)")
    ]
    monthly["Province"] = monthly["Province"].replace({"Tibet (Xizang)": "Tibet"})
    months = sorted(monthly["Date"].unique())
    generation = monthly.pivot_table(
        index="Province", columns="Date", values="Generation (TWh)", aggfunc="sum"
    ).reindex(columns=months)
    if generation.isna().any().any() or len(months) != 12:
        raise ValueError("Expected a complete 12-month panel for every included province")

    # Keep exactly the provinces with all 12 monthly generation values. Tibet
    # is not present in the V294 monthly proxy and is explicitly excluded.
    provinces = generation.index.tolist()
    od = od_full.reindex(index=provinces, columns=provinces).fillna(0.0)
    edges = [(origin, destination, float(od.loc[origin, destination]))
             for origin in provinces for destination in provinces
             if od.loc[origin, destination] > 0]
    n_edges, n_months = len(edges), len(months)
    n_vars = n_edges * n_months
    if n_edges == 0:
        raise ValueError("No OD edges overlap the monthly generation panel")

    # Every observed annual route total must be allocated across 12 months.
    a_eq = np.zeros((n_edges, n_vars))
    b_eq = np.zeros(n_edges)
    for edge_idx, (origin, destination, annual_flow) in enumerate(edges):
        a_eq[edge_idx, edge_idx * n_months:(edge_idx + 1) * n_months] = 1.0
        b_eq[edge_idx] = annual_flow

    # Monthly outgoing electricity from an origin cannot exceed that month's
    # total generation. No monthly line capacities or destination demand are
    # available, so this is a deliberately weak, auditable physical constraint.
    a_ub = np.zeros((len(provinces) * n_months, n_vars))
    b_ub = np.zeros(len(provinces) * n_months)
    for province_idx, origin in enumerate(provinces):
        for month_idx, month in enumerate(months):
            row_idx = province_idx * n_months + month_idx
            for edge_idx, (edge_origin, _, _) in enumerate(edges):
                if edge_origin == origin:
                    a_ub[row_idx, edge_idx * n_months + month_idx] = 1.0
            b_ub[row_idx] = float(generation.loc[origin, month])

    feasible = linprog(
        np.zeros(n_vars), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
        bounds=(0, None), method="highs"
    )
    if not feasible.success:
        raise RuntimeError(f"Annual OD cannot be allocated under monthly generation caps: {feasible.message}")

    annual_imports = od.sum(axis=0).reindex(provinces)
    records: list[dict[str, float | str]] = []
    for destination in provinces:
        for month_idx, month in enumerate(months):
            objective = np.zeros(n_vars)
            for edge_idx, (_, edge_destination, _) in enumerate(edges):
                if edge_destination == destination:
                    objective[edge_idx * n_months + month_idx] = 1.0
            lower = linprog(
                objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                bounds=(0, None), method="highs"
            )
            upper = linprog(
                -objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                bounds=(0, None), method="highs"
            )
            if not lower.success or not upper.success:
                raise RuntimeError(f"Receipt LP failed for {destination} {month}")
            local_generation = float(generation.loc[destination, month])
            minimum, maximum = float(lower.fun), float(-upper.fun)
            uniform_monthly_import = float(annual_imports.loc[destination] / n_months)
            denominator_low = local_generation + minimum
            denominator_high = local_generation + maximum
            records.append({
                "province": destination,
                "month": month,
                "local_generation_twh": local_generation,
                "annual_imports_twh": float(annual_imports.loc[destination]),
                "monthly_import_min_twh": minimum,
                "monthly_import_max_twh": maximum,
                "monthly_import_range_twh": maximum - minimum,
                "gross_import_share_min": minimum / denominator_low if denominator_low > 0 else np.nan,
                "gross_import_share_max": maximum / denominator_high if denominator_high > 0 else np.nan,
                "uniform_annual_split_import_twh": uniform_monthly_import,
                "uniform_annual_split_share": uniform_monthly_import / (local_generation + uniform_monthly_import),
            })

    result = pd.DataFrame(records)
    result.to_csv(OUT / "province_month_receipt_bounds.csv", index=False)
    share_width = result["gross_import_share_max"] - result["gross_import_share_min"]
    all_imports = od_full.to_numpy().sum()
    included_imports = od.to_numpy().sum()
    zenodo_generation = pd.read_excel(DATA / "3_provincial electricity generation dataset.xlsx").set_index("Province")
    zenodo_generation_total = float(zenodo_generation.loc[provinces].sum(axis=1).sum())
    monthly_generation_total = float(generation.to_numpy().sum())
    uniform_outgoing = np.zeros((len(provinces), n_months))
    for origin_idx, origin in enumerate(provinces):
        uniform_outgoing[origin_idx, :] = float(od.loc[origin].sum()) / n_months
    uniform_feasible = bool((uniform_outgoing <= generation.to_numpy() + 1e-10).all())

    summary = {
        "experiment": "V311 sharp monthly receipt bounds under annual OD and monthly generation caps",
        "n_provinces": len(provinces),
        "n_months": n_months,
        "province_months": len(result),
        "n_annual_routes": n_edges,
        "annual_od_twh_full_matrix": float(all_imports),
        "annual_od_twh_included_provinces": float(included_imports),
        "excluded_provinces": sorted(set(od_full.index) - set(provinces)),
        "excluded_od_volume_twh": float(all_imports - included_imports),
        "monthly_proxy_generation_twh": monthly_generation_total,
        "zenodo_generation_same_provinces_twh": zenodo_generation_total,
        "monthly_proxy_vs_zenodo_generation_relative_difference": float(
            monthly_generation_total / zenodo_generation_total - 1.0
        ),
        "all_route_annual_totals_feasible": bool(feasible.success),
        "uniform_annual_flow_split_respects_source_monthly_generation": uniform_feasible,
        "zero_lower_bound_months": int((result["monthly_import_min_twh"] <= 1e-8).sum()),
        "all_monthly_import_share_bounds_span_zero": bool((result["gross_import_share_min"] <= 1e-8).all()),
        "mean_gross_import_share_bound_width": float(share_width.mean()),
        "median_gross_import_share_bound_width": float(share_width.median()),
        "mean_uniform_split_gross_import_share": float(result["uniform_annual_split_share"].mean()),
        "max_gross_import_share_upper_bound": float(result["gross_import_share_max"].max()),
        "interpretation": (
            "With only annual OD totals and monthly total generation caps, no province-month has a positive guaranteed gross-import share. "
            "The intervals are loose because monthly demand, flow schedule, line capacities, and transmission losses are absent. "
            "They are bounds on gross direct receipts, not CIF prediction intervals or hourly estimates."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
