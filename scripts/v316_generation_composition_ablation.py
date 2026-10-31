"""Quantify how monthly technology-specific generation constrains import risk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import block_diag, coo_matrix, csr_matrix, hstack, vstack


ROOT = Path(__file__).resolve().parents[1]
V310 = ROOT / "results" / "v310_zenodo_interprovincial_audit_r1"
EMBER = ROOT / "results" / "v294-china-ember-lineage-r1" / "ember_china_monthly_generation.csv"
OUT = ROOT / "results" / "v316-generation-composition-ablation-r1"
def sparse_matrix(fields: list[list[int]], n_vars: int) -> csr_matrix:
    rows, cols = [], []
    for row, indices in enumerate(fields):
        rows.extend([row] * len(indices))
        cols.extend(indices)
    return coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(fields), n_vars)
    ).tocsr()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generation = pd.read_csv(EMBER)
    generation = generation.loc[
        generation["Date"].astype(str).str.startswith("2023")
        & generation["Is aggregated source"].eq(False)
        & generation["Province"].ne("China (total)")
    ].copy()
    generation["Province"] = generation["Province"].replace({"Tibet (Xizang)": "Tibet"})
    provinces = sorted(generation["Province"].unique().tolist())
    months = sorted(generation["Date"].astype(str).unique().tolist())
    technologies = sorted(generation["Electricity source"].unique().tolist())
    pidx = {name: i for i, name in enumerate(provinces)}
    tidx = {name: i for i, name in enumerate(technologies)}
    midx = {name: i for i, name in enumerate(months)}

    gen = generation.pivot_table(
        index="Province", columns=["Electricity source", "Date"],
        values="Generation (TWh)", aggfunc="sum"
    )
    emissions = generation.pivot_table(
        index="Province", columns=["Electricity source", "Date"],
        values="Emissions (MtCO2e)", aggfunc="sum"
    )
    gen = gen.reindex(index=provinces)
    emissions = emissions.reindex(index=provinces)
    tech_gen = np.zeros((len(provinces), len(technologies), len(months)))
    tech_ef = np.zeros_like(tech_gen)
    for province in provinces:
        for technology in technologies:
            for month in months:
                key = (technology, month)
                g = float(gen.loc[province, key])
                e = float(emissions.loc[province, key])
                tech_gen[pidx[province], tidx[technology], midx[month]] = g
                tech_ef[pidx[province], tidx[technology], midx[month]] = e / g if g > 0 else 0.0

    monthly_total = generation.pivot_table(
        index="Province", columns="Date", values="Generation (TWh)", aggfunc="sum"
    ).reindex(index=provinces, columns=months)

    od = pd.read_excel(V310 / "5_interprovincial transfer matrix.xlsx", index_col=0).iloc[:31, :31]
    od = od.reindex(index=provinces, columns=provinces).fillna(0.0)
    edges = [(i, j, float(od.loc[i, j])) for i in provinces for j in provinces if od.loc[i, j] > 0]
    n_edges, n_tech, n_months = len(edges), len(technologies), len(months)
    n_vars = n_edges * n_tech * n_months
    vi = lambda edge, tech, month: edge * n_tech * n_months + tech * n_months + month

    eq_fields, b_eq = [], []
    for edge_idx, (_, _, annual_flow) in enumerate(edges):
        eq_fields.append([vi(edge_idx, t, m) for t in range(n_tech) for m in range(n_months)])
        b_eq.append(annual_flow)
    a_eq = sparse_matrix(eq_fields, n_vars)
    b_eq = np.asarray(b_eq)

    cap_fields, b_ub = [], []
    for province in provinces:
        for tech in technologies:
            for month in months:
                cells = [
                    vi(edge_idx, tidx[tech], midx[month])
                    for edge_idx, (origin, _, _) in enumerate(edges)
                    if origin == province
                ]
                cap_fields.append(cells)
                b_ub.append(tech_gen[pidx[province], tidx[tech], midx[month]])
    a_ub = sparse_matrix(cap_fields, n_vars)
    b_ub = np.asarray(b_ub)
    total_cap_fields, b_total = [], []
    for province in provinces:
        for month in months:
            total_cap_fields.append([
                vi(edge_idx, t, midx[month])
                for edge_idx, (origin, _, _) in enumerate(edges)
                if origin == province for t in range(n_tech)
            ])
            b_total.append(float(monthly_total.loc[province, month]))
    a_total = sparse_matrix(total_cap_fields, n_vars)
    b_total = np.asarray(b_total)

    def province_receipt_fields() -> list[list[int]]:
        observed_months = range(n_months - 1)
        fields: list[list[int]] = []
        for dest in provinces:
            for m in observed_months:
                fields.append([
                    vi(e, t, m) for e, (_, edge_dest, _) in enumerate(edges)
                    if edge_dest == dest for t in range(n_tech)
                ])
        return fields

    # Isolate the information in fuel-specific monthly generation caps.
    # The objective keeps the same technology-resolved export mix uncertainty;
    # only the feasible supply constraints change.
    information_sets = [
        "aggregate_caps_annual_od_only", "technology_caps_annual_od_only",
        "aggregate_caps_plus_receipts", "technology_caps_plus_receipts",
    ]
    measurement_matrices = {
        name: sparse_matrix(province_receipt_fields(), n_vars)
        for name in information_sets if name.endswith("plus_receipts")
    }
    paired_eq_base = block_diag((a_eq, a_eq), format="csr")
    paired_ub_by_model = {
        "aggregate": block_diag((a_total, a_total), format="csr"),
        "technology": block_diag((a_ub, a_ub), format="csr"),
    }
    total_gen = monthly_total.reindex(index=provinces, columns=months)
    rows = []
    widths: dict[tuple[str, str, str], float] = {}

    for destination in provinces:
        d = pidx[destination]
        for month_idx, month in enumerate(months):
            denom = float(total_gen.loc[destination, month])
            objective = np.zeros(n_vars)
            for edge_idx, (_, edge_dest, _) in enumerate(edges):
                if edge_dest == destination:
                    for t in range(n_tech):
                        objective[vi(edge_idx, t, month_idx)] = tech_ef[d, t, month_idx] / denom
            pair_obj = np.r_[objective, -objective]
            for name in information_sets:
                model = "technology" if name.startswith("technology") else "aggregate"
                mat = measurement_matrices.get(name, csr_matrix((0, n_vars)))
                if mat.shape[0]:
                    aeq = vstack([paired_eq_base, hstack([mat, -mat], format="csr")], format="csr")
                    beq = np.r_[b_eq, b_eq, np.zeros(mat.shape[0])]
                else:
                    aeq, beq = paired_eq_base, np.r_[b_eq, b_eq]
                result = linprog(
                    -pair_obj, A_ub=paired_ub_by_model[model],
                    b_ub=np.r_[b_ub, b_ub] if model == "technology" else np.r_[b_total, b_total],
                    A_eq=aeq, b_eq=beq, bounds=(0, None), method="highs"
                )
                if not result.success:
                    raise RuntimeError(f"LP failed {destination} {month} {name}: {result.message}")
                width = float(-result.fun)
                if name.endswith("annual_od_only"):
                    widths[(model, destination, month)] = width
                rows.append({
                    "province": destination, "month": month,
                    "capacity_model": model, "information_set": name,
                    "additional_independent_fields": 0 if name.endswith("annual_od_only") else 341,
                    "worst_case_remaining_width_kgco2e_kwh": width,
                    "guaranteed_relative_width_reduction": np.nan,
                })
        print(f"completed {destination}", flush=True)

    result_table = pd.DataFrame(rows)
    result_table["baseline_width_kgco2e_kwh"] = result_table.apply(
        lambda r: widths[(r["capacity_model"], r["province"], r["month"])], axis=1
    )
    result_table["guaranteed_relative_width_reduction"] = 1 - (
        result_table["worst_case_remaining_width_kgco2e_kwh"] /
        result_table["baseline_width_kgco2e_kwh"]
    )
    result_table.to_csv(OUT / "province_month_information_diameter.csv", index=False)
    summaries = []
    for name, group in result_table.groupby("information_set", sort=False):
        summaries.append({
            "information_set": name,
            "additional_independent_fields": int(group["additional_independent_fields"].iloc[0]),
            "mean_remaining_width_kgco2e_kwh": float(group["worst_case_remaining_width_kgco2e_kwh"].mean()),
            "median_remaining_width_kgco2e_kwh": float(group["worst_case_remaining_width_kgco2e_kwh"].median()),
            "mean_guaranteed_relative_width_reduction": float(group["guaranteed_relative_width_reduction"].mean()),
            "median_guaranteed_relative_width_reduction": float(group["guaranteed_relative_width_reduction"].median()),
            "zero_width_province_months": int((group["worst_case_remaining_width_kgco2e_kwh"] < 1e-10).sum()),
        })
    summary = {
        "experiment": "V316 ablation of total versus technology-specific monthly generation caps",
        "n_provinces": len(provinces), "n_months": len(months), "n_routes": n_edges,
        "n_technologies": len(technologies), "technologies": technologies,
        "risk_target_is_hourly_cif": False,
        "data": "V310 annual OD plus V294 province-month generation and emissions by eight technology categories",
        "method": "route × technology × month nonnegative LP; annual route totals fixed; compares aggregate origin-month caps against origin-technology-month caps, with and without 341 province-month receipt fields; two-copy worst-case conditional diameter",
        "limitations": [
            "V294 generation and emissions are estimates and not independent labels",
            "technology output is an availability cap, not observed exported fuel composition",
            "no monthly recipient load, line capacity, losses, or physical-flow validation",
            "outcome is a monthly direct-import carbon exposure proxy, not standard CIF or hourly risk",
        ],
        "summary_by_information_set": summaries,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
