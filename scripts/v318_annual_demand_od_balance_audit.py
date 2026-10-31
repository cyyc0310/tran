"""Check whether reconstructed provincial demand reconciles with annual OD flows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOAD_INPUT = ROOT / "results/china-hourly-load-dataset-audit-v64-r1/load_vs_ember_generation.csv"
OD_INPUT = ROOT / "results/v310_zenodo_interprovincial_audit_r1/5_interprovincial transfer matrix.xlsx"
OUT = ROOT / "results/v318-annual-demand-od-balance-audit-r1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(LOAD_INPUT)
    monthly = monthly.loc[monthly["Date"].astype(str).str.startswith("2023")].copy()
    province = monthly.groupby("Province", as_index=False).agg(
        demand_proxy_TWh=("load_TWh", "sum"),
        generation_estimate_TWh=("Generation (TWh)", "sum"),
        months_with_data=("Date", "nunique"),
    )

    od = pd.read_excel(OD_INPUT, index_col=0).iloc[:31, :31].fillna(0.0)
    od.index = od.index.astype(str).str.strip()
    od.columns = od.columns.astype(str).str.strip()
    od = od.astype(float)
    inbound = od.sum(axis=0)
    outbound = od.sum(axis=1)
    province = province.set_index("Province").reindex(od.index)
    province.index.name = "Province"
    province["od_gross_import_TWh"] = inbound.reindex(province.index)
    province["od_gross_export_TWh"] = outbound.reindex(province.index)
    province["od_net_import_TWh"] = province["od_gross_import_TWh"] - province["od_gross_export_TWh"]
    province["net_import_implied_by_demand_generation_TWh"] = (
        province["demand_proxy_TWh"] - province["generation_estimate_TWh"]
    )
    # Under an annual energy balance D = G + imports - exports - losses.
    province["implied_losses_TWh"] = (
        province["generation_estimate_TWh"]
        + province["od_net_import_TWh"]
        - province["demand_proxy_TWh"]
    )
    province["balance_gap_TWh"] = -province["implied_losses_TWh"]
    province = province.reset_index().sort_values("Province")
    matched = province.dropna(
        subset=["demand_proxy_TWh", "generation_estimate_TWh", "od_net_import_TWh"]
    )

    gen_total = float(province["generation_estimate_TWh"].sum())
    demand_total = float(province["demand_proxy_TWh"].sum())
    losses = matched["implied_losses_TWh"].to_numpy(float)
    gap = matched["balance_gap_TWh"].to_numpy(float)
    summary = {
        "experiment": "V318 annual mass-balance audit of V64 reconstructed demand against V294 generation and V310 OD",
        "year": 2023,
        "matched_provinces": int(len(matched)),
        "unmatched_provinces": province.loc[
            province[["demand_proxy_TWh", "generation_estimate_TWh"]].isna().any(axis=1)
        ]["Province"].tolist(),
        "matched_months_complete": bool((matched["months_with_data"] == 12).all()),
        "demand_proxy_TWh": demand_total,
        "generation_estimate_TWh": gen_total,
        "demand_minus_generation_TWh": demand_total - gen_total,
        "generation_minus_demand_pct_of_generation": (gen_total - demand_total) / gen_total,
        "od_gross_transfer_TWh_all_31_provinces": float(od.to_numpy().sum()),
        "od_net_import_sum_matched_provinces_TWh": float(matched["od_net_import_TWh"].sum()),
        "implied_losses_sum_matched_provinces_TWh": float(losses.sum()),
        "provinces_with_negative_implied_losses": int((losses < -1e-9).sum()),
        "negative_implied_losses_total_TWh": float(losses[losses < 0].sum()),
        "positive_implied_losses_total_TWh": float(losses[losses > 0].sum()),
        "net_import_upward_correction_needed_for_negative_loss_provinces_TWh": float(-losses[losses < 0].sum()),
        "largest_single_province_upward_net_import_correction_TWh": float(max(0.0, -losses.min())),
        "mean_absolute_net_balance_gap_TWh": float(np.mean(np.abs(gap))),
        "median_absolute_net_balance_gap_TWh": float(np.median(np.abs(gap))),
        "rmse_net_balance_gap_TWh": float(np.sqrt(np.mean(gap ** 2))),
        "maximum_absolute_net_balance_gap_TWh": float(np.max(np.abs(gap))),
        "interpretation": "The national annual totals are close, but province net imports from the annual OD matrix and net exchange implied by reconstructed demand/generation imply negative losses in many provinces. The proxy demand/OD/generation scopes are not jointly reconciled, so this demand proxy cannot yet serve as a strict provincial CIF denominator with these inputs.",
    }
    province.to_csv(OUT / "province_annual_balance.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
