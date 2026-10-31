"""Recompute Chinese monthly source-risk diameters with missing generation explicit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import block_diag, coo_matrix, csr_matrix, hstack, vstack


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1"
MONTHLY = BASE / "cn_prov_monthly_generation_by_tech.csv"
ANNUAL = BASE / "cn_prov_annual_generation_by_tech.csv"
OD_PATH = ROOT / "results/v310_zenodo_interprovincial_audit_r1/5_interprovincial transfer matrix.xlsx"
EMBER = ROOT / "results/v294-china-ember-lineage-r1/ember_china_monthly_generation.csv"
OUT = ROOT / "results/v321-unobserved-generation-remainder-r1"
CATEGORIES = ["thermal", "hydro", "nuclear", "wind", "solar", "unobserved"]
OBSERVED = CATEGORIES[:-1]
UNOBSERVED_EF_MAX = 1.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sparse(fields: list[list[int]], nvars: int) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for r, values in enumerate(fields):
        rows.extend([r] * len(values))
        cols.extend(values)
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(fields), nvars)).tocsr()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(MONTHLY)
    raw = raw.loc[raw["year"].eq(2023)].copy()
    provinces = sorted(raw["province"].unique())
    pidx = {p: i for i, p in enumerate(provinces)}
    periods = ["Jan-Feb", *[f"{m:02d}" for m in range(3, 13)]]
    nprov, nperiod, ncat = len(provinces), len(periods), len(CATEGORIES)
    tech_raw = raw.pivot_table(
        index="province", columns=["technology", "month"], values="generation_gwh", aggfunc="sum"
    ).reindex(index=provinces)
    caps = np.zeros((nprov, ncat - 1, nperiod))
    totals = np.zeros((nprov, nperiod))
    cat_present = np.zeros((nprov, ncat - 1, nperiod), dtype=bool)
    for pi, province in enumerate(provinces):
        for period_idx, month in enumerate([2, *range(3, 13)]):
            totals[pi, period_idx] = float(tech_raw.loc[province, ("total", month)]) / 1000.0
            for ti, category in enumerate(OBSERVED):
                value = tech_raw.loc[province, (category, month)]
                if pd.notna(value):
                    caps[pi, ti, period_idx] = float(value) / 1000.0
                    cat_present[pi, ti, period_idx] = True

    annual_raw = pd.read_csv(ANNUAL)
    annual_total = annual_raw.loc[
        annual_raw["year"].eq(2023) & annual_raw["technology"].eq("total"), ["province", "generation_gwh"]
    ].set_index("province")["generation_gwh"].reindex(provinces).to_numpy(float) / 1000.0
    observed_annual = totals.sum(axis=1)
    remainder = annual_total - observed_annual
    if np.any(remainder < -1e-9):
        bad = {provinces[i]: float(remainder[i]) for i in np.where(remainder < -1e-9)[0]}
        raise ValueError(f"Monthly observed generation exceeds annual all-plant total: {bad}")
    remainder = np.maximum(remainder, 0.0)

    ember = pd.read_csv(EMBER)
    ember = ember.loc[
        ember["Date"].astype(str).str.startswith("2023")
        & ember["Is aggregated source"].eq(False)
        & ember["Province"].ne("China (total)")
    ].copy()
    ember["Province"] = ember["Province"].replace({"Tibet (Xizang)": "Tibet"})
    source_map = {
        "thermal": {"Bioenergy", "Coal", "Gas", "Other fossil"},
        "hydro": {"Hydro"}, "nuclear": {"Nuclear"}, "wind": {"Wind"}, "solar": {"Solar"},
    }
    ember["category"] = ember["Electricity source"].map(
        {source: category for category, sources in source_map.items() for source in sources}
    )
    ember = ember.loc[ember["category"].notna()]
    ember["Date"] = pd.to_datetime(ember["Date"])
    ember["month"] = ember["Date"].dt.month
    fac_rows = ember.groupby(["Province", "category", "month"], as_index=False).agg(
        gen=("Generation (TWh)", "sum"), emis=("Emissions (MtCO2e)", "sum")
    )
    efs = np.zeros((nprov, len(OBSERVED), nperiod))
    for pi, province in enumerate(provinces):
        for ti, category in enumerate(OBSERVED):
            for period_idx, months in enumerate([[1, 2], *[[m] for m in range(3, 13)]]):
                rows = fac_rows.loc[
                    fac_rows["Province"].eq(province)
                    & fac_rows["category"].eq(category)
                    & fac_rows["month"].isin(months)
                ]
                generation = float(rows["gen"].sum())
                emissions = float(rows["emis"].sum())
                efs[pi, ti, period_idx] = emissions / generation if generation > 0 else 0.0

    od = pd.read_excel(OD_PATH, index_col=0).iloc[:31, :31].fillna(0.0)
    od.index = od.index.astype(str).str.strip()
    od.columns = od.columns.astype(str).str.strip()
    od = od.reindex(index=provinces, columns=provinces).fillna(0.0).astype(float)
    edges = [(i, j, float(od.loc[i, j])) for i in provinces for j in provinces if od.loc[i, j] > 0]
    nedge = len(edges)
    n_energy = nedge * ncat * nperiod
    n_q = nedge * nperiod
    nvars = n_energy + n_q

    def xi(edge: int, category: int, period: int) -> int:
        return edge * ncat * nperiod + category * nperiod + period

    def qi(edge: int, period: int) -> int:
        return n_energy + edge * nperiod + period

    # Route-year conservation across all five observed groups and the annual residual pool.
    route_fields = []
    route_rhs = []
    for edge_idx, (_, _, flow) in enumerate(edges):
        route_fields.append([xi(edge_idx, c, per) for c in range(ncat) for per in range(nperiod)])
        route_rhs.append(flow)
    a_route = sparse(route_fields, nvars)
    b_route = np.asarray(route_rhs)

    # Every reported above-size monthly total is a hard cap on all observed-category exports.
    total_fields, total_rhs = [], []
    for pi, province in enumerate(provinces):
        for per in range(nperiod):
            total_fields.append([
                xi(edge_idx, c, per)
                for edge_idx, (origin, _, _) in enumerate(edges) if origin == province
                for c in range(ncat - 1)
            ])
            total_rhs.append(totals[pi, per])

    # The fifth category fields are added only in the composition arm. Missing values
    # receive the full observed total as a non-informative cap, never a zero cap.
    category_fields, category_rhs = [], []
    for pi, province in enumerate(provinces):
        for ti in range(ncat - 1):
            for per in range(nperiod):
                category_fields.append([
                    xi(edge_idx, ti, per)
                    for edge_idx, (origin, _, _) in enumerate(edges) if origin == province
                ])
                category_rhs.append(caps[pi, ti, per] if cat_present[pi, ti, per] else totals[pi, per])

    # Annual all-plant generation beyond the above-size monthly series is available
    # only as an annual source pool; its within-year placement is unrestricted.
    residual_fields, residual_rhs = [], []
    for pi, province in enumerate(provinces):
        residual_fields.append([
            xi(edge_idx, ncat - 1, per)
            for edge_idx, (origin, _, _) in enumerate(edges) if origin == province
            for per in range(nperiod)
        ])
        residual_rhs.append(remainder[pi])

    # Unknown residual emissions are interval-valued: 0 <= q <= 1.0 * residual energy.
    q_fields, q_rhs = [], []
    for edge_idx in range(nedge):
        for per in range(nperiod):
            q_fields.append([qi(edge_idx, per), xi(edge_idx, ncat - 1, per)])
            q_rhs.append(0.0)
    q_rows, q_cols, q_vals = [], [], []
    for r, (q_var, x_var) in enumerate(q_fields):
        q_rows.extend([r, r])
        q_cols.extend([q_var, x_var])
        q_vals.extend([1.0, -UNOBSERVED_EF_MAX])
    a_q = coo_matrix((q_vals, (q_rows, q_cols)), shape=(len(q_rhs), nvars)).tocsr()

    a_total = sparse(total_fields, nvars)
    a_categories = sparse(category_fields, nvars)
    a_remainder = sparse(residual_fields, nvars)
    a_ub_common = vstack([a_total, a_remainder, a_q], format="csr")
    b_ub_common = np.r_[np.asarray(total_rhs), np.asarray(residual_rhs), np.asarray(q_rhs)]
    a_ub_category = vstack([a_ub_common, a_categories], format="csr")
    b_ub_category = np.r_[b_ub_common, np.asarray(category_rhs)]

    # Monthly receiver totals are hypothetical measurements. Last-bin receipt follows
    # from annual OD, leaving 31 * 10 = 310 independent new values.
    receipt_fields = []
    for dest in provinces:
        for per in range(nperiod - 1):
            receipt_fields.append([
                xi(edge_idx, c, per)
                for edge_idx, (_, destination, _) in enumerate(edges) if destination == dest
                for c in range(ncat)
            ])
    a_receipt = sparse(receipt_fields, nvars)
    paired_route = block_diag((a_route, a_route), format="csr")
    paired_ub = {
        "total_only": block_diag((a_ub_common, a_ub_common), format="csr"),
        "five_categories": block_diag((a_ub_category, a_ub_category), format="csr"),
    }
    info_arms = [(cap, receipts) for cap in ("total_only", "five_categories") for receipts in (False, True)]
    rows: list[dict[str, object]] = []
    baseline: dict[tuple[str, str, str], float] = {}

    for dest in provinces:
        di = pidx[dest]
        for per, period in enumerate(periods):
            denominator = totals[di, per]
            if denominator <= 0:
                raise ValueError(f"Nonpositive local generation denominator: {dest} {period}")
            objective = np.zeros(nvars)
            for edge_idx, (origin, destination, _) in enumerate(edges):
                if destination == dest:
                    oi = pidx[origin]
                    for ti in range(ncat - 1):
                        objective[xi(edge_idx, ti, per)] = efs[oi, ti, per] / denominator
                    objective[xi(edge_idx, ncat - 1, per)] = 0.0
                    objective[qi(edge_idx, per)] = 1.0 / denominator
            pair_objective = np.r_[objective, -objective]
            for cap_name, with_receipts in info_arms:
                if with_receipts:
                    aeq = vstack([
                        paired_route,
                        hstack([a_receipt, -a_receipt], format="csr"),
                    ], format="csr")
                    beq = np.r_[b_route, b_route, np.zeros(a_receipt.shape[0])]
                else:
                    aeq, beq = paired_route, np.r_[b_route, b_route]
                result = linprog(
                    -pair_objective,
                    A_ub=paired_ub[cap_name],
                    b_ub=np.r_[b_ub_common, b_ub_common] if cap_name == "total_only" else np.r_[b_ub_category, b_ub_category],
                    A_eq=aeq, b_eq=beq, bounds=(0, None), method="highs",
                )
                if not result.success:
                    raise RuntimeError(f"LP failed {dest} {period} {cap_name} receipts={with_receipts}: {result.message}")
                width = float(-result.fun)
                key = (cap_name, dest, period)
                if not with_receipts:
                    baseline[key] = width
                rows.append({
                    "province": dest,
                    "period": period,
                    "supply_information": cap_name,
                    "monthly_receipts_observed": with_receipts,
                    "additional_independent_receipt_fields": 310 if with_receipts else 0,
                    "worst_case_diameter_kgco2e_kwh_proxy": width,
                    "diameter_without_receipts_same_supply_info": baseline[key],
                    "paired_shrink_from_receipts": 1 - width / baseline[key] if baseline[key] else 0.0,
                })
        print(f"completed {dest}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "province_period_diameters.csv", index=False)
    summaries = []
    for (cap, receipts), group in results.groupby(["supply_information", "monthly_receipts_observed"], sort=False):
        summaries.append({
            "supply_information": cap,
            "monthly_receipts_observed": bool(receipts),
            "additional_independent_receipt_fields": int(group["additional_independent_receipt_fields"].iloc[0]),
            "mean_diameter": float(group["worst_case_diameter_kgco2e_kwh_proxy"].mean()),
            "median_diameter": float(group["worst_case_diameter_kgco2e_kwh_proxy"].median()),
            "mean_paired_receipt_shrink": float(group["paired_shrink_from_receipts"].mean()),
            "median_paired_receipt_shrink": float(group["paired_shrink_from_receipts"].median()),
            "zero_width_cells": int((group["worst_case_diameter_kgco2e_kwh_proxy"] < 1e-10).sum()),
        })
    paired = results.pivot(
        index=["province", "period"], columns=["supply_information", "monthly_receipts_observed"],
        values="worst_case_diameter_kgco2e_kwh_proxy",
    )
    cap_gain_no_receipts = 1 - paired[("five_categories", False)] / paired[("total_only", False)]
    cap_gain_receipts = 1 - paired[("five_categories", True)] / paired[("total_only", True)]
    summary = {
        "experiment": "V321 bounds with explicit below-size annual generation remainder",
        "n_provinces": nprov,
        "n_reporting_periods": nperiod,
        "periods": periods,
        "n_annual_od_routes": nedge,
        "annual_generation_remainder_TWh_sum": float(remainder.sum()),
        "annual_generation_remainder_TWh_min": float(remainder.min()),
        "annual_generation_remainder_TWh_max": float(remainder.max()),
        "unobserved_ef_support_kgco2e_kwh": [0.0, UNOBSERVED_EF_MAX],
        "unobserved_ef_support_note": "Upper endpoint equals the maximum 2023 Ember source-month factor (1.0); this is a declared proxy support, not a validated engineering limit.",
        "missing_technology_rows": int((~cat_present).sum()),
        "missing_category_handling": "For a missing category, no zero cap is imposed; its upper cap is the province-period reported total, also subject to the combined reported total cap.",
        "monthly_receipt_budget": 310,
        "target": "source-weighted imports divided by NBS above-size local generation within each reported period; Jan-Feb is a two-month aggregate; not consumption CIF or hourly risk",
        "category_cap_value_no_receipt": {
            "mean_paired_shrink_vs_total_only": float(cap_gain_no_receipts.mean()),
            "median_paired_shrink_vs_total_only": float(cap_gain_no_receipts.median()),
            "positive_cells": int((cap_gain_no_receipts > 1e-9).sum()),
            "n_cells": len(cap_gain_no_receipts),
        },
        "category_cap_value_with_receipt": {
            "mean_paired_shrink_vs_total_only": float(cap_gain_receipts.mean()),
            "median_paired_shrink_vs_total_only": float(cap_gain_receipts.median()),
            "positive_cells": int((cap_gain_receipts > 1e-9).sum()),
            "n_cells": len(cap_gain_receipts),
        },
        "summary_by_information_set": summaries,
        "limitations": [
            "NBS monthly generation is above-designated-size coverage, while the annual remainder is not monthly resolved",
            "V310 annual OD is an external accounting source not reconciled with the NBS tables",
            "Ember category factors and the unobserved EF support are proxy assumptions",
            "the denominator is local reported generation, not electricity consumption",
            "this is a structural partial-identification experiment, not hourly prediction validation",
        ],
        "sha256": {
            str(MONTHLY.relative_to(ROOT)): sha256(MONTHLY),
            str(ANNUAL.relative_to(ROOT)): sha256(ANNUAL),
            str(OD_PATH.relative_to(ROOT)): sha256(OD_PATH),
            str(EMBER.relative_to(ROOT)): sha256(EMBER),
            "docs/V321_UNOBSERVED_GENERATION_REMAINDER_PROTOCOL.md": sha256(ROOT / "docs/V321_UNOBSERVED_GENERATION_REMAINDER_PROTOCOL.md"),
            "scripts/v321_unobserved_generation_remainder_bounds.py": sha256(Path(__file__)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
