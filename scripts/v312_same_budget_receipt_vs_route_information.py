"""Compare robust value of monthly receipt totals and OD route observations.

For each source-weighted import exposure, a two-copy LP computes the largest
remaining interval width among any two feasible flow schedules that agree on
the measured fields. This is a worst-case posterior diameter, independent of
an assumed monthly schedule. The target is an annual-factor import proxy, not
hourly CIF.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import block_diag, coo_matrix, csr_matrix, hstack, vstack


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "v310_zenodo_interprovincial_audit_r1"
V311 = ROOT / "results" / "v311-annual-od-monthly-receipt-bounds-r1"
OUT = ROOT / "results" / "v312-same-budget-import-information-r1"
N_ROUTE_FIELDS = 31


def measurement_matrix(
    fields: list[list[tuple[int, float]]], n_vars: int
) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for row_idx, items in enumerate(fields):
        for col_idx, value in items:
            rows.append(row_idx)
            cols.append(col_idx)
            vals.append(value)
    return coo_matrix((vals, (rows, cols)), shape=(len(fields), n_vars)).tocsr()


def build_measurements(
    provinces: list[str], months: list[str], edges: list[tuple[str, str, float]], n_vars: int
) -> dict[str, tuple[csr_matrix, int, str]]:
    n_months = len(months)
    observed_months = range(n_months - 1)  # December is implied by annual totals.

    national_fields = []
    province_fields = []
    for month_idx in observed_months:
        national_fields.append(
            [(edge_idx * n_months + month_idx, 1.0) for edge_idx in range(len(edges))]
        )
    for destination in provinces:
        for month_idx in observed_months:
            province_fields.append(
                [
                    (edge_idx * n_months + month_idx, 1.0)
                    for edge_idx, (_, edge_destination, _) in enumerate(edges)
                    if edge_destination == destination
                ]
            )

    annual_flows = np.asarray([v for _, _, v in edges])
    volume_order = np.argsort(-annual_flows)
    source_factors = pd.read_csv(DATA / "v310_audit" / "province_comparison.csv").set_index("province")
    ef = source_factors["local_generation_factor"]
    carbon_order = np.argsort(-np.asarray([v * ef.loc[i] for i, _, v in edges]))

    def route_fields(order: np.ndarray) -> list[list[tuple[int, float]]]:
        chosen = order[:N_ROUTE_FIELDS]
        fields = []
        for edge_idx in chosen:
            for month_idx in observed_months:
                fields.append([(int(edge_idx) * n_months + month_idx, 1.0)])
        return fields

    global_matrix = measurement_matrix(national_fields, n_vars)
    province_matrix = measurement_matrix(province_fields, n_vars)
    volume_route_matrix = measurement_matrix(route_fields(volume_order), n_vars)
    carbon_route_matrix = measurement_matrix(route_fields(carbon_order), n_vars)
    return {
        "annual_od_only": (csr_matrix((0, n_vars)), 0, "0 additional scalar fields"),
        "national_monthly_receipts": (global_matrix, global_matrix.shape[0], "11 independent fields"),
        "province_monthly_receipts": (province_matrix, province_matrix.shape[0], "341 independent fields"),
        "top31_volume_routes_monthly": (
            volume_route_matrix,
            volume_route_matrix.shape[0],
            "341 independent fields",
        ),
        "top31_carbon_mass_routes_monthly": (
            carbon_route_matrix,
            carbon_route_matrix.shape[0],
            "341 independent fields",
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bound_table = pd.read_csv(V311 / "province_month_receipt_bounds.csv")
    generation = pd.read_csv(
        ROOT / "results" / "v294-china-ember-lineage-r1" / "ember_china_monthly_generation.csv"
    )
    generation = generation.loc[
        generation["Date"].astype(str).str.startswith("2023")
        & generation["Is aggregated source"].eq(False)
        & generation["Province"].ne("China (total)")
    ].copy()
    generation["Province"] = generation["Province"].replace({"Tibet (Xizang)": "Tibet"})
    monthly_generation = generation.pivot_table(
        index="Province", columns="Date", values="Generation (TWh)", aggfunc="sum"
    )
    provinces = monthly_generation.index.tolist()
    months = sorted(monthly_generation.columns.tolist())
    monthly_generation = monthly_generation.reindex(index=provinces, columns=months)

    od_all = pd.read_excel(DATA / "5_interprovincial transfer matrix.xlsx", index_col=0).iloc[:31, :31]
    od = od_all.reindex(index=provinces, columns=provinces).fillna(0.0)
    edges = [(i, j, float(od.loc[i, j])) for i in provinces for j in provinces if od.loc[i, j] > 0]
    n_edges, n_months = len(edges), len(months)
    n_vars = n_edges * n_months

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_values: list[float] = []
    b_eq = np.zeros(n_edges)
    for edge_idx, (_, _, annual_flow) in enumerate(edges):
        b_eq[edge_idx] = annual_flow
        for month_idx in range(n_months):
            eq_rows.append(edge_idx)
            eq_cols.append(edge_idx * n_months + month_idx)
            eq_values.append(1.0)
    a_eq = coo_matrix((eq_values, (eq_rows, eq_cols)), shape=(n_edges, n_vars)).tocsr()

    cap_rows: list[int] = []
    cap_cols: list[int] = []
    cap_values: list[float] = []
    b_ub = np.zeros(len(provinces) * n_months)
    pindex = {p: idx for idx, p in enumerate(provinces)}
    for edge_idx, (origin, _, _) in enumerate(edges):
        for month_idx in range(n_months):
            row_idx = pindex[origin] * n_months + month_idx
            cap_rows.append(row_idx)
            cap_cols.append(edge_idx * n_months + month_idx)
            cap_values.append(1.0)
    for province_idx, province in enumerate(provinces):
        for month_idx, month in enumerate(months):
            b_ub[province_idx * n_months + month_idx] = monthly_generation.loc[province, month]
    a_ub = coo_matrix(
        (cap_values, (cap_rows, cap_cols)), shape=(len(b_ub), n_vars)
    ).tocsr()

    paired_eq_base = block_diag((a_eq, a_eq), format="csr")
    paired_ub = block_diag((a_ub, a_ub), format="csr")
    source_factors = pd.read_csv(DATA / "v310_audit" / "province_comparison.csv").set_index("province")
    source_ef = source_factors["local_generation_factor"]
    measurements = build_measurements(provinces, months, edges, n_vars)

    records: list[dict[str, float | int | str]] = []
    baseline_widths: dict[tuple[str, str], float] = {}
    for destination in provinces:
        for month_idx, month in enumerate(months):
            local_generation = float(monthly_generation.loc[destination, month])
            objective = np.zeros(n_vars)
            for edge_idx, (origin, edge_destination, _) in enumerate(edges):
                if edge_destination == destination:
                    objective[edge_idx * n_months + month_idx] = float(source_ef.loc[origin]) / local_generation
            pair_objective = np.concatenate([objective, -objective])

            for info_name, (m_matrix, field_count, budget_note) in measurements.items():
                if m_matrix.shape[0]:
                    paired_measurement = hstack([m_matrix, -m_matrix], format="csr")
                    a_eq_pair = vstack([paired_eq_base, paired_measurement], format="csr")
                    b_eq_pair = np.concatenate([b_eq, b_eq, np.zeros(m_matrix.shape[0])])
                else:
                    a_eq_pair = paired_eq_base
                    b_eq_pair = np.concatenate([b_eq, b_eq])
                result = linprog(
                    -pair_objective,
                    A_ub=paired_ub,
                    b_ub=np.concatenate([b_ub, b_ub]),
                    A_eq=a_eq_pair,
                    b_eq=b_eq_pair,
                    bounds=(0, None),
                    method="highs",
                )
                if not result.success:
                    raise RuntimeError(f"Diameter LP failed for {destination} {month} {info_name}: {result.message}")
                width = float(-result.fun)
                key = (destination, month)
                if info_name == "annual_od_only":
                    baseline_widths[key] = width
                records.append({
                    "province": destination,
                    "month": month,
                    "information_set": info_name,
                    "additional_independent_fields": field_count,
                    "field_budget_note": budget_note,
                    "worst_case_remaining_import_carbon_exposure_width_kgco2e_kwh": width,
                    "baseline_width_kgco2e_kwh": np.nan,
                    "guaranteed_relative_width_reduction": np.nan,
                })

    result_table = pd.DataFrame(records)
    result_table["baseline_width_kgco2e_kwh"] = result_table.apply(
        lambda row: baseline_widths[(row["province"], row["month"])], axis=1
    )
    result_table["guaranteed_relative_width_reduction"] = 1.0 - (
        result_table["worst_case_remaining_import_carbon_exposure_width_kgco2e_kwh"]
        / result_table["baseline_width_kgco2e_kwh"]
    )
    result_table.to_csv(OUT / "province_month_information_diameter.csv", index=False)
    summary_rows = []
    for info_name, group in result_table.groupby("information_set", sort=False):
        summary_rows.append({
            "information_set": info_name,
            "additional_independent_fields": int(group["additional_independent_fields"].iloc[0]),
            "mean_baseline_width_kgco2e_kwh": float(group["baseline_width_kgco2e_kwh"].mean()),
            "mean_remaining_width_kgco2e_kwh": float(group["worst_case_remaining_import_carbon_exposure_width_kgco2e_kwh"].mean()),
            "median_remaining_width_kgco2e_kwh": float(group["worst_case_remaining_import_carbon_exposure_width_kgco2e_kwh"].median()),
            "mean_guaranteed_relative_width_reduction": float(group["guaranteed_relative_width_reduction"].mean()),
            "median_guaranteed_relative_width_reduction": float(group["guaranteed_relative_width_reduction"].median()),
            "province_months_with_zero_remaining_width": int(
                (group["worst_case_remaining_import_carbon_exposure_width_kgco2e_kwh"] <= 1e-10).sum()
            ),
        })
    summary = {
        "experiment": "V312 robust same-budget information value for monthly imported carbon exposure",
        "n_provinces": len(provinces),
        "n_months": n_months,
        "province_month_outcomes": len(provinces) * n_months,
        "n_annual_routes": n_edges,
        "route_measurement_design": "31 largest annual routes by volume or by volume times origin annual factor; 11 months per route, December inferred from annual total",
        "risk_target": "imported source-factor-weighted electricity carbon exposure divided by local monthly generation",
        "risk_target_is_hourly_cif": False,
        "source_factor_vintage": "V310 annual technology-weighted generation factors from the same Zenodo record",
        "monthly_generation_input": "V294 Ember estimate, used only for total-generation capacity constraints",
        "method": "two-copy LP; maximum objective difference among feasible schedules that agree on observed fields",
        "summary_by_information_set": summary_rows,
        "interpretation": "Worst-case conditional diameter over all possible measured values; no synthetic monthly schedule is treated as truth.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
