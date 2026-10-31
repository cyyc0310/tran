"""Origin-corrected partial-ID audit of coarse monthly generation categories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import block_diag, coo_matrix, csr_matrix, hstack, vstack


ROOT = Path(__file__).resolve().parents[1]
V310 = ROOT / "results" / "v310_zenodo_interprovincial_audit_r1"
EMBER = ROOT / "results" / "v294-china-ember-lineage-r1" / "ember_china_monthly_generation.csv"
OUT = ROOT / "results" / "v319-china-monthly-category-information-r1"
GROUPS = {
    "thermal": {"Bioenergy", "Coal", "Gas", "Other fossil"},
    "hydro": {"Hydro"},
    "nuclear": {"Nuclear"},
    "wind": {"Wind"},
    "solar": {"Solar"},
}


def matrix(fields: list[list[int]], n_vars: int) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for r, indices in enumerate(fields):
        rows.extend([r] * len(indices))
        cols.extend(indices)
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(fields), n_vars)).tocsr()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    data = pd.read_csv(EMBER)
    data = data.loc[
        data["Date"].astype(str).str.startswith("2023")
        & data["Is aggregated source"].eq(False)
        & data["Province"].ne("China (total)")
    ].copy()
    data["Province"] = data["Province"].replace({"Tibet (Xizang)": "Tibet"})
    provinces = sorted(data["Province"].unique())
    months = sorted(data["Date"].astype(str).unique())
    techs = sorted(data["Electricity source"].unique())
    if set(techs) != set.union(*GROUPS.values()):
        raise ValueError(f"Unexpected nonaggregated source schema: {techs}")
    pidx, midx, tidx = ({v: i for i, v in enumerate(vals)} for vals in (provinces, months, techs))

    gen = data.pivot_table(
        index="Province", columns=["Electricity source", "Date"],
        values="Generation (TWh)", aggfunc="sum",
    ).reindex(index=provinces)
    emis = data.pivot_table(
        index="Province", columns=["Electricity source", "Date"],
        values="Emissions (MtCO2e)", aggfunc="sum",
    ).reindex(index=provinces)
    g = np.zeros((len(provinces), len(techs), len(months)))
    e = np.zeros_like(g)
    for p in provinces:
        for t in techs:
            for m in months:
                key = (t, m)
                g[pidx[p], tidx[t], midx[m]] = float(gen.loc[p, key])
                value = float(emis.loc[p, key])
                amount = float(gen.loc[p, key])
                e[pidx[p], tidx[t], midx[m]] = value / amount if amount > 0 else 0.0

    total = data.pivot_table(
        index="Province", columns="Date", values="Generation (TWh)", aggfunc="sum"
    ).reindex(index=provinces, columns=months)
    od = pd.read_excel(V310 / "5_interprovincial transfer matrix.xlsx", index_col=0).iloc[:31, :31]
    od = od.reindex(index=provinces, columns=provinces).fillna(0.0)
    edges = [(i, j, float(od.loc[i, j])) for i in provinces for j in provinces if od.loc[i, j] > 0]
    ne, nt, nm = len(edges), len(techs), len(months)
    nvars = ne * nt * nm

    def vi(edge: int, tech: int, month: int) -> int:
        return edge * nt * nm + tech * nm + month

    eq_fields, b_eq = [], []
    for ei, (_, _, flow) in enumerate(edges):
        eq_fields.append([vi(ei, t, m) for t in range(nt) for m in range(nm)])
        b_eq.append(flow)
    a_eq, b_eq_arr = matrix(eq_fields, nvars), np.asarray(b_eq)

    cap_mats: dict[str, tuple[csr_matrix, np.ndarray]] = {}
    group_specs: dict[str, list[set[str]]] = {
        "aggregate": [set(techs)],
        "five_categories": list(GROUPS.values()),
        "eight_categories": [{t} for t in techs],
    }
    for name, groups in group_specs.items():
        fields, rhs = [], []
        for p in provinces:
            for group in groups:
                for m in range(nm):
                    tis = [tidx[t] for t in group]
                    fields.append([
                        vi(ei, ti, m)
                        for ei, (origin, _, _) in enumerate(edges) if origin == p
                        for ti in tis
                    ])
                    rhs.append(sum(g[pidx[p], ti, m] for ti in tis))
        cap_mats[name] = matrix(fields, nvars), np.asarray(rhs)

    receipt_fields = []
    for dest in provinces:
        for m in range(nm - 1):
            receipt_fields.append([
                vi(ei, t, m) for ei, (_, destination, _) in enumerate(edges) if destination == dest
                for t in range(nt)
            ])
    receipt_mat = matrix(receipt_fields, nvars)

    arms = [(cap, False) for cap in group_specs] + [(cap, True) for cap in group_specs]
    paired_eq_base = block_diag((a_eq, a_eq), format="csr")
    paired_ub = {name: block_diag((a, a), format="csr") for name, (a, _) in cap_mats.items()}
    baseline: dict[tuple[str, str, str], float] = {}
    records: list[dict[str, object]] = []
    for dest in provinces:
        di = pidx[dest]
        for mi, month in enumerate(months):
            denom = float(total.loc[dest, month])
            objective = np.zeros(nvars)
            for ei, (origin, destination, _) in enumerate(edges):
                if destination == dest:
                    oi = pidx[origin]
                    for ti in range(nt):
                        objective[vi(ei, ti, mi)] = e[oi, ti, mi] / denom
            pair_obj = np.r_[objective, -objective]
            for cap_name, with_receipts in arms:
                cap, rhs = cap_mats[cap_name]
                if with_receipts:
                    aeq = vstack([
                        paired_eq_base,
                        hstack([receipt_mat, -receipt_mat], format="csr"),
                    ], format="csr")
                    beq = np.r_[b_eq_arr, b_eq_arr, np.zeros(receipt_mat.shape[0])]
                else:
                    aeq, beq = paired_eq_base, np.r_[b_eq_arr, b_eq_arr]
                result = linprog(
                    -pair_obj,
                    A_ub=paired_ub[cap_name], b_ub=np.r_[rhs, rhs],
                    A_eq=aeq, b_eq=beq, bounds=(0, None), method="highs",
                )
                if not result.success:
                    raise RuntimeError(f"LP failed for {dest} {month} {cap_name} receipts={with_receipts}: {result.message}")
                width = float(-result.fun)
                if not with_receipts:
                    baseline[(cap_name, dest, month)] = width
                base_width = baseline[(cap_name, dest, month)]
                records.append({
                    "province": dest,
                    "month": month,
                    "supply_cap_schema": cap_name,
                    "province_month_receipts_observed": with_receipts,
                    "additional_independent_receipt_fields": 341 if with_receipts else 0,
                    "worst_case_remaining_width_kgco2e_kwh": width,
                    "baseline_width_same_schema": base_width,
                    "relative_width_reduction_vs_same_schema": 1 - width / base_width if base_width else 0.0,
                })
        print(f"completed {dest}", flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "province_month_information_diameter.csv", index=False)
    summaries = []
    for (schema, receipts), group in frame.groupby(
        ["supply_cap_schema", "province_month_receipts_observed"], sort=False
    ):
        summaries.append({
            "supply_cap_schema": schema,
            "province_month_receipts_observed": bool(receipts),
            "additional_independent_receipt_fields": int(group["additional_independent_receipt_fields"].iloc[0]),
            "mean_width": float(group["worst_case_remaining_width_kgco2e_kwh"].mean()),
            "median_width": float(group["worst_case_remaining_width_kgco2e_kwh"].median()),
            "mean_paired_width_reduction": float(group["relative_width_reduction_vs_same_schema"].mean()),
            "median_paired_width_reduction": float(group["relative_width_reduction_vs_same_schema"].median()),
            "zero_width_cells": int((group["worst_case_remaining_width_kgco2e_kwh"] < 1e-10).sum()),
        })
    summary = {
        "experiment": "V319 origin-corrected monthly generation category information",
        "n_provinces": len(provinces), "n_months": len(months), "n_routes": ne,
        "n_detailed_sources": nt,
        "detailed_sources": techs,
        "coarse_schema": {k: sorted(v) for k, v in GROUPS.items()},
        "additional_receipt_fields": 341,
        "target": "incoming source-weighted emissions divided by receiver monthly total generation; direct-import exposure proxy, not standard CIF or hourly risk",
        "factor_assignment": "origin province i, technology k, month m",
        "limitations": [
            "Ember generation and emissions are estimates, not independent labels or the user's original table",
            "annual OD flows and monthly supply may use incompatible accounting scopes",
            "generation is a supply cap rather than observed export composition",
            "no monthly demand, physical flow, line losses, hourly observations, or CIF validation",
            "five categories are a schema proxy; actual report coverage and definitions require source-specific audit",
        ],
        "summary_by_information_set": summaries,
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [EMBER, V310 / "5_interprovincial transfer matrix.xlsx"]},
        "protocol_sha256": sha256(ROOT / "docs/V319_CHINA_MONTHLY_CATEGORY_INFORMATION_PROTOCOL.md"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
