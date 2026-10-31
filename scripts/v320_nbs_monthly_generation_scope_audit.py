"""Audit NBS-derived monthly generation scope and provincial annual reconciliation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
EMBER = ROOT / "results/v294-china-ember-lineage-r1/ember_china_monthly_generation.csv"
ZENODO_DIR = ROOT / "results/v310_zenodo_interprovincial_audit_r1"
OD_PATH = ZENODO_DIR / "5_interprovincial transfer matrix.xlsx"
ZENODO_GEN_PATH = ZENODO_DIR / "3_provincial electricity generation dataset.xlsx"
LOAD_PATH = ROOT / "results/china-hourly-load-dataset-audit-v64-r1/load_vs_ember_generation.csv"
ANNUAL_GEN_PATH = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_annual_generation_by_tech.csv"
ANNUAL_DEMAND_PATH = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_annual_consumption.csv"
OUT = ROOT / "results/v320-oreo-official-monthly-generation-audit-r2"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_pair(frame: pd.DataFrame, a: str, b: str, name: str) -> dict[str, object]:
    x = frame[[a, b]].dropna()
    err = x[a].to_numpy(float) - x[b].to_numpy(float)
    return {
        "comparison": name,
        "n_provinces": int(len(x)),
        "sum_a_TWh": float(x[a].sum()),
        "sum_b_TWh": float(x[b].sum()),
        "sum_a_minus_b_TWh": float(err.sum()),
        "sum_a_minus_b_pct_of_b": float(err.sum() / x[b].sum()),
        "province_equal_mean_abs_difference_TWh": float(np.mean(np.abs(err))),
        "province_equal_median_abs_difference_TWh": float(np.median(np.abs(err))),
        "province_pearson_r": float(x[a].corr(x[b])),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RAW)
    required = {"technology", "province", "year", "month", "generation_gwh"}
    if set(data.columns) != required:
        raise ValueError(f"Unexpected OREO schema: {data.columns.tolist()}")
    duplicates = int(data.duplicated(["technology", "province", "year", "month"]).sum())
    d23 = data.loc[data["year"].eq(2023)].copy()
    expected_months = list(range(2, 13))
    if duplicates:
        raise ValueError(f"Duplicate technology/province/year/month rows: {duplicates}")
    provinces = sorted(d23["province"].unique())
    techs = sorted(d23["technology"].unique())
    coverage = (
        d23.groupby("technology").agg(
            province_month_rows=("generation_gwh", "size"),
            n_provinces=("province", "nunique"),
            n_months=("month", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
            missing_values=("generation_gwh", lambda s: int(s.isna().sum())),
        ).reset_index()
    )

    # February is the Jan-Feb cumulative figure; summing it once plus Mar-Dec
    # yields the annual amount represented by the published monthly series.
    annual = (
        d23.groupby(["province", "technology"], as_index=False)["generation_gwh"]
        .sum(min_count=11)
        .pivot(index="province", columns="technology", values="generation_gwh")
        .div(1000.0)
    )
    annual.columns = [f"oreo_{col}_TWh" for col in annual.columns]
    annual = annual.reset_index()

    # Category consistency: compare reported total with the five generation
    # classes. Keep a null subtotal wherever any class is unavailable.
    required_categories = ["thermal", "hydro", "nuclear", "wind", "solar"]
    annual["oreo_classified_five_sum_TWh"] = annual[
        [f"oreo_{name}_TWh" for name in required_categories]
    ].sum(axis=1, min_count=5)
    annual["oreo_total_minus_five_categories_TWh"] = (
        annual["oreo_total_TWh"] - annual["oreo_classified_five_sum_TWh"]
    )

    ember = pd.read_csv(EMBER)
    ember23 = ember.loc[
        ember["Date"].astype(str).str.startswith("2023")
        & ember["Is aggregated source"].eq(False)
        & ember["Province"].ne("China (total)")
    ].copy()
    ember23["Province"] = ember23["Province"].replace({"Tibet (Xizang)": "Tibet"})
    ember_total = ember23.groupby("Province", as_index=False)["Generation (TWh)"].sum()
    ember_total = ember_total.rename(columns={"Province": "province", "Generation (TWh)": "ember_total_TWh"})

    zenodo = pd.read_excel(ZENODO_GEN_PATH)
    gen_cols = [col for col in zenodo.columns if col != "Province"]
    zenodo["zenodo_total_TWh"] = zenodo[gen_cols].sum(axis=1)
    zenodo_total = zenodo[["Province", "zenodo_total_TWh"]].rename(columns={"Province": "province"})

    load = pd.read_csv(LOAD_PATH)
    load23 = load.loc[load["Date"].astype(str).str.startswith("2023")]
    load_total = load23.groupby("Province", as_index=False)["load_TWh"].sum().rename(
        columns={"Province": "province", "load_TWh": "modeled_load_proxy_TWh"}
    )

    annual_gen = pd.read_csv(ANNUAL_GEN_PATH)
    annual_gen = annual_gen.loc[annual_gen["year"].eq(2023)]
    annual_gen_wide = annual_gen.pivot(index="province", columns="technology", values="generation_gwh") / 1000.0
    annual_gen_wide = annual_gen_wide.rename(columns={
        "total": "nbs_all_plant_generation_TWh",
        "thermal": "nbs_thermal_generation_TWh",
        "hydro": "nbs_hydro_generation_TWh",
    }).reset_index()
    annual_demand = pd.read_csv(ANNUAL_DEMAND_PATH)
    annual_demand = annual_demand.loc[annual_demand["year"].eq(2023), ["province", "demand_gwh"]].copy()
    annual_demand["nbs_annual_demand_TWh"] = annual_demand.pop("demand_gwh") / 1000.0

    od = pd.read_excel(OD_PATH, index_col=0).iloc[:31, :31].fillna(0.0)
    od.index = od.index.astype(str).str.strip()
    od.columns = od.columns.astype(str).str.strip()
    od = od.astype(float)
    net = (od.sum(axis=0) - od.sum(axis=1)).rename("od_net_import_TWh").rename_axis("province").reset_index()

    province = annual.merge(ember_total, how="outer", on="province").merge(
        zenodo_total, how="outer", on="province"
    ).merge(load_total, how="outer", on="province").merge(net, how="outer", on="province").merge(
        annual_gen_wide, how="outer", on="province"
    ).merge(annual_demand, how="outer", on="province")
    province["oreo_implied_losses_diagnostic_TWh"] = (
        province["oreo_total_TWh"] + province["od_net_import_TWh"] - province["modeled_load_proxy_TWh"]
    )
    province["nbs_annual_implied_losses_diagnostic_TWh"] = (
        province["nbs_all_plant_generation_TWh"] + province["od_net_import_TWh"] - province["nbs_annual_demand_TWh"]
    )
    province["monthly_above_size_share_of_annual_generation"] = (
        province["oreo_total_TWh"] / province["nbs_all_plant_generation_TWh"]
    )
    province["oreo_minus_ember_TWh"] = province["oreo_total_TWh"] - province["ember_total_TWh"]
    province["oreo_minus_zenodo_TWh"] = province["oreo_total_TWh"] - province["zenodo_total_TWh"]
    province.sort_values("province").to_csv(OUT / "province_annual_scope_comparison.csv", index=False)
    coverage.to_csv(OUT / "2023_monthly_coverage.csv", index=False)

    summary = {
        "experiment": "V320 NBS-derived monthly generation scope and annual provincial balance audit",
        "year": 2023,
        "dataset_schema": data.columns.tolist(),
        "technologies": techs,
        "n_provinces": len(provinces),
        "n_2023_rows": len(d23),
        "expected_reporting_months": expected_months,
        "duplicate_cells": duplicates,
        "coverage_by_technology": coverage.to_dict(orient="records"),
        "note_on_february": "February is January-February cumulative; annual amount sums February once plus March-December. January and February are not separately observed.",
        "category_consistency": {
            "n_provinces_with_all_five_categories": int(annual["oreo_classified_five_sum_TWh"].notna().sum()),
            "n_provinces_where_classified_sum_exceeds_total": int((annual["oreo_total_minus_five_categories_TWh"] < -1e-9).sum()),
            "median_total_minus_five_category_sum_TWh": float(annual["oreo_total_minus_five_categories_TWh"].median()),
            "max_abs_total_minus_five_category_sum_TWh": float(annual["oreo_total_minus_five_categories_TWh"].abs().max()),
        },
        "source_comparisons": [
            summarize_pair(province, "oreo_total_TWh", "ember_total_TWh", "OREO monthly annualized total vs Ember estimate"),
            summarize_pair(province, "oreo_total_TWh", "zenodo_total_TWh", "OREO monthly annualized total vs V310 Zenodo annual sum"),
            summarize_pair(province, "oreo_total_TWh", "nbs_all_plant_generation_TWh", "OREO monthly above-size total vs OREO annual all-plant generation"),
        ],
        "balance_diagnostic": {
            "matched_provinces": int(province[["oreo_total_TWh", "od_net_import_TWh", "modeled_load_proxy_TWh"]].dropna().shape[0]),
            "negative_implied_losses_provinces": int((province["oreo_implied_losses_diagnostic_TWh"] < 0).sum()),
            "negative_implied_losses_sum_TWh": float(province.loc[province["oreo_implied_losses_diagnostic_TWh"] < 0, "oreo_implied_losses_diagnostic_TWh"].sum()),
            "mean_absolute_implied_losses_TWh": float(province["oreo_implied_losses_diagnostic_TWh"].abs().mean()),
        },
        "same_source_nbs_annual_balance_diagnostic": {
            "matched_provinces": int(province[["nbs_all_plant_generation_TWh", "od_net_import_TWh", "nbs_annual_demand_TWh"]].dropna().shape[0]),
            "nbs_annual_generation_sum_TWh": float(province["nbs_all_plant_generation_TWh"].sum()),
            "nbs_annual_demand_sum_TWh": float(province["nbs_annual_demand_TWh"].sum()),
            "negative_implied_losses_provinces": int((province["nbs_annual_implied_losses_diagnostic_TWh"] < 0).sum()),
            "negative_implied_losses_sum_TWh": float(province.loc[province["nbs_annual_implied_losses_diagnostic_TWh"] < 0, "nbs_annual_implied_losses_diagnostic_TWh"].sum()),
            "mean_absolute_implied_losses_TWh": float(province["nbs_annual_implied_losses_diagnostic_TWh"].abs().mean()),
        },
        "scope_interpretation": "OREO methodology says monthly generation covers industrial enterprises above designated size; it is slightly lower than all-plant generation. It cannot be used as a hard cap on total provincial exports without a same-scope contract or validated adjustment. Balance values using V64 demand proxy and V310 OD are diagnostic only.",
        "sha256": {
            str(RAW.relative_to(ROOT)): file_hash(RAW),
            str(EMBER.relative_to(ROOT)): file_hash(EMBER),
            str(ZENODO_GEN_PATH.relative_to(ROOT)): file_hash(ZENODO_GEN_PATH),
            str(OD_PATH.relative_to(ROOT)): file_hash(OD_PATH),
            str(LOAD_PATH.relative_to(ROOT)): file_hash(LOAD_PATH),
            str(ANNUAL_GEN_PATH.relative_to(ROOT)): file_hash(ANNUAL_GEN_PATH),
            str(ANNUAL_DEMAND_PATH.relative_to(ROOT)): file_hash(ANNUAL_DEMAND_PATH),
            "scripts/v320_nbs_monthly_generation_scope_audit.py": file_hash(Path(__file__)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
