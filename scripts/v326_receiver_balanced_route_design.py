"""Compare receiver-balanced sparse route-month and receiver-month information."""

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
OUT = ROOT / "results/v326-receiver-balanced-route-design-r1"
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
    # Global top-volume baseline and destination-stratified design: one largest incoming
    # annual route for every receiver. The V310 matrix has at least one incoming route
    # for all provinces, so both designs use exactly 31 routes x 10 independent bins.
    selected_routes_volume = sorted(
        range(nedge), key=lambda e: (-edges[e][2], edges[e][0], edges[e][1])
    )[:31]
    selected_routes_balanced = []
    for dest in provinces:
        incoming = [e for e, (_, destination, _) in enumerate(edges) if destination == dest]
        if not incoming:
            raise ValueError(f"No annual OD route into receiver {dest}")
        selected_routes_balanced.append(
            min(incoming, key=lambda e: (-edges[e][2], edges[e][0], edges[e][1]))
        )
    if len(set(selected_routes_balanced)) != 31:
        raise ValueError("Receiver-balanced design did not produce 31 unique routes")

    def route_measurement_matrix(selected: list[int]) -> csr_matrix:
        fields = [
            [xi(edge_idx, category, period) for category in range(ncat)]
            for edge_idx in selected
            for period in range(nperiod - 1)
        ]
        return sparse(fields, nvars)

    a_route_volume = route_measurement_matrix(selected_routes_volume)
    a_route_balanced = route_measurement_matrix(selected_routes_balanced)
    paired_route_base = block_diag((a_route, a_route), format="csr")
    paired_ub_base = {
        "total_only": block_diag((a_ub_common, a_ub_common), format="csr"),
        "five_categories": block_diag((a_ub_category, a_ub_category), format="csr"),
    }

    info_arms = [
        (cap, field_set)
        for cap in ("total_only", "five_categories")
        for field_set in ("none", "receipts", "top31_volume", "receiver_balanced", "receipts_and_volume", "receipts_and_balanced")
    ]
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
            pair_objective_base = np.r_[objective, -objective]
            for cap_name, field_set in info_arms:
                aeq, beq = paired_route_base, np.r_[b_route, b_route]
                field_matrices = []
                if field_set in ("receipts", "receipts_and_volume", "receipts_and_balanced"):
                    field_matrices.append(a_receipt)
                if field_set in ("top31_volume", "receipts_and_volume"):
                    field_matrices.append(a_route_volume)
                if field_set in ("receiver_balanced", "receipts_and_balanced"):
                    field_matrices.append(a_route_balanced)
                if field_matrices:
                    observed_fields = vstack(field_matrices, format="csr")
                    field_pair = hstack([observed_fields, -observed_fields], format="csr")
                    aeq = vstack([aeq, field_pair], format="csr")
                    beq = np.r_[beq, np.zeros(observed_fields.shape[0])]
                a_ub_current = paired_ub_base[cap_name]
                b_ub_current = np.r_[b_ub_common, b_ub_common] if cap_name == "total_only" else np.r_[b_ub_category, b_ub_category]
                result = linprog(
                    -pair_objective_base,
                    A_ub=a_ub_current,
                    b_ub=b_ub_current,
                    A_eq=aeq, b_eq=beq, bounds=(0, None), method="highs",
                )
                if not result.success:
                    raise RuntimeError(f"LP failed {dest} {period} {cap_name} fields={field_set}: {result.message}")
                width = float(-result.fun)
                key = (cap_name, dest, period)
                if field_set == "none":
                    baseline[key] = width
                rows.append({
                    "province": dest,
                    "period": period,
                    "supply_information": cap_name,
                    "observed_field_set": field_set,
                    "additional_independent_receipt_fields": 310 if field_set in ("receipts", "receipts_and_volume", "receipts_and_balanced") else 0,
                    "additional_independent_route_month_fields": 310 if field_set in ("top31_volume", "receiver_balanced", "receipts_and_volume", "receipts_and_balanced") else 0,
                    "additional_independent_fields_total": 310 * (int(field_set in ("receipts", "receipts_and_volume", "receipts_and_balanced")) + int(field_set in ("top31_volume", "receiver_balanced", "receipts_and_volume", "receipts_and_balanced"))),
                    "worst_case_diameter_kgco2e_kwh_proxy": width,
                    "diameter_without_extra_fields_same_supply_info": baseline[key],
                    "paired_shrink_vs_no_extra_fields": 1 - width / baseline[key] if baseline[key] else 0.0,
                })
        print(f"completed {dest}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "province_period_diameters.csv", index=False)
    summaries = []
    group_columns = ["supply_information", "observed_field_set"]
    for (cap, field_set), group in results.groupby(group_columns, sort=False):
        summaries.append({
            "supply_information": cap,
            "observed_field_set": field_set,
            "additional_independent_receipt_fields": int(group["additional_independent_receipt_fields"].iloc[0]),
            "additional_independent_route_month_fields": int(group["additional_independent_route_month_fields"].iloc[0]),
            "additional_independent_fields_total": int(group["additional_independent_fields_total"].iloc[0]),
            "mean_diameter": float(group["worst_case_diameter_kgco2e_kwh_proxy"].mean()),
            "median_diameter": float(group["worst_case_diameter_kgco2e_kwh_proxy"].median()),
            "mean_paired_shrink_vs_no_extra_fields": float(group["paired_shrink_vs_no_extra_fields"].mean()),
            "median_paired_shrink_vs_no_extra_fields": float(group["paired_shrink_vs_no_extra_fields"].median()),
            "zero_width_cells": int((group["worst_case_diameter_kgco2e_kwh_proxy"] < 1e-10).sum()),
        })
    paired = results.pivot(
        index=["province", "period"],
        columns=["supply_information", "observed_field_set"],
        values="worst_case_diameter_kgco2e_kwh_proxy",
    )
    base = paired[("five_categories", "none")]
    gains = {
        "volume": 1 - paired[("five_categories", "top31_volume")] / base,
        "receiver_balanced": 1 - paired[("five_categories", "receiver_balanced")] / base,
        "receipts": 1 - paired[("five_categories", "receipts")] / base,
        "volume_plus_receipts": 1 - paired[("five_categories", "receipts_and_volume")] / base,
        "balanced_plus_receipts": 1 - paired[("five_categories", "receipts_and_balanced")] / base,
    }
    rng = np.random.default_rng(326)
    province_values = pd.DataFrame({"province": paired.index.get_level_values(0), **{k: v.to_numpy() for k, v in gains.items()}})
    province_values = province_values.groupby("province", as_index=False).mean()
    province_means = province_values.set_index("province")
    boot = {name: [] for name in [*gains, "balanced_vs_volume", "balanced_vs_receipts"]}
    province_names = province_values["province"].to_numpy()
    for _ in range(20000):
        sample = rng.choice(province_names, size=len(province_names), replace=True)
        sampled = province_means.loc[sample]
        for name in gains:
            boot[name].append(float(sampled[name].mean()))
        boot["balanced_vs_volume"].append(float((sampled["receiver_balanced"] - sampled["volume"]).mean()))
        boot["balanced_vs_receipts"].append(float((sampled["receiver_balanced"] - sampled["receipts"]).mean()))
    information_value = {}
    for name, values in gains.items():
        information_value[name] = {
            "mean_paired_reduction": float(values.mean()),
            "median_paired_reduction": float(values.median()),
            "positive_cells": int((values > 1e-9).sum()),
            "positive_provinces": int((province_values[name] > 1e-9).sum()),
            "province_cluster_bootstrap_95pct": np.quantile(boot[name], [0.025, 0.975]).tolist(),
        }
    for label, difference_name, left, right in (
        ("receiver_balanced_vs_volume_equal_budget", "balanced_vs_volume", "receiver_balanced", "volume"),
        ("receiver_balanced_vs_receipts_equal_budget", "balanced_vs_receipts", "receiver_balanced", "receipts"),
    ):
        difference = province_values[left] - province_values[right]
        information_value[label] = {
            "mean_paired_reduction_difference": float(difference.mean()),
            "province_cluster_bootstrap_95pct": np.quantile(boot[difference_name], [0.025, 0.975]).tolist(),
            "positive_provinces": int((difference > 1e-9).sum()),
        }
    monotonicity = {
        "all_single_designs_never_widen_vs_base": all(
            (paired[("five_categories", name)] <= base + 1e-8).all()
            for name in ("receipts", "top31_volume", "receiver_balanced")
        ),
        "combined_designs_never_widen_vs_either_component": bool(
            (paired[("five_categories", "receipts_and_volume")] <= paired[("five_categories", "receipts")] + 1e-8).all()
            and (paired[("five_categories", "receipts_and_volume")] <= paired[("five_categories", "top31_volume")] + 1e-8).all()
            and (paired[("five_categories", "receipts_and_balanced")] <= paired[("five_categories", "receipts")] + 1e-8).all()
            and (paired[("five_categories", "receipts_and_balanced")] <= paired[("five_categories", "receiver_balanced")] + 1e-8).all()
        ),
    }
    summary = {
        "experiment": "V326 minimax value of receiver-balanced sparse route selection",
        "n_provinces": nprov,
        "n_reporting_periods": nperiod,
        "periods": periods,
        "n_annual_od_routes": nedge,
        "monthly_receipt_budget": 310,
        "receiver_balanced_rule": "one largest annual incoming route for every receiver, 10 monthly bins per selected route",
        "selected_receiver_balanced_routes": [
            {"origin": edges[e][0], "destination": edges[e][1], "annual_flow_TWh": edges[e][2]}
            for e in selected_routes_balanced
        ],
        "receiver_balanced_annual_flow_share": float(sum(edges[e][2] for e in selected_routes_balanced) / sum(edge[2] for edge in edges)),
        "selected_top31_volume_routes": [
            {"origin": edges[e][0], "destination": edges[e][1], "annual_flow_TWh": edges[e][2]}
            for e in selected_routes_volume
        ],
        "selected_top31_volume_flow_share": float(sum(edges[e][2] for e in selected_routes_volume) / sum(edge[2] for edge in edges)),
        "annual_generation_remainder_TWh_sum": float(remainder.sum()),
        "annual_generation_remainder_TWh_min": float(remainder.min()),
        "annual_generation_remainder_TWh_max": float(remainder.max()),
        "unobserved_ef_support_kgco2e_kwh": [0.0, UNOBSERVED_EF_MAX],
        "unobserved_ef_support_note": "Upper endpoint equals the maximum 2023 Ember source-month factor (1.0); this is a declared proxy support, not a validated engineering limit.",
        "missing_technology_rows": int((~cat_present).sum()),
        "missing_category_handling": "For a missing category, no zero cap is imposed; its upper cap is the province-period reported total, also subject to the combined reported total cap.",
        "target": "source-weighted imports divided by NBS above-size local generation within each reported period; Jan-Feb is a two-month aggregate; not consumption CIF or hourly risk",
        "five_category_arm_information_value": information_value,
        "measurement_constraint_monotonicity": monotonicity,
        "summary_by_information_set": summaries,
        "limitations": [
            "NBS monthly generation is above-designated-size coverage, while the annual remainder is not monthly resolved",
            "V310 annual OD is an external accounting source not reconciled with the NBS tables",
            "Ember category factors and the unobserved EF support are proxy assumptions",
            "the denominator is local reported generation, not electricity consumption",
            "the target is direct-import exposure, not consumption CIF or hourly risk",
            "receiver-balanced selection is based on one annual OD matrix and does not validate transfer across years",
            "hypothetical field values are optimized adversarially; results are not expected benefit under an empirical release distribution",
        ],
        "sha256": {
            str(MONTHLY.relative_to(ROOT)): sha256(MONTHLY),
            str(ANNUAL.relative_to(ROOT)): sha256(ANNUAL),
            str(OD_PATH.relative_to(ROOT)): sha256(OD_PATH),
            str(EMBER.relative_to(ROOT)): sha256(EMBER),
            "docs/V326_RECEIVER_BALANCED_ROUTE_DESIGN_PROTOCOL.md": sha256(ROOT / "docs/V326_RECEIVER_BALANCED_ROUTE_DESIGN_PROTOCOL.md"),
            "scripts/v326_receiver_balanced_route_design.py": sha256(Path(__file__)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
