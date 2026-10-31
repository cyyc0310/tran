"""Audit what an annual public China electricity OD matrix adds to CIF estimates.

This is a source-dataset consistency experiment, not an independent prediction
benchmark: all inputs and comparison summaries come from the same Zenodo record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "v310_zenodo_interprovincial_audit_r1"
OUT = DATA / "v310_audit"

GENERATION_FILE = "3_provincial electricity generation dataset.xlsx"
TECH_CF_FILE = "4_provincial technology-specific CFs dataset.xlsx"
TRANSFER_FILE = "5_interprovincial transfer matrix.xlsx"
SUMMARY_FILE = "1_provincial electricity CFs dataset.xlsx"
TD_FILE = "6_provincial transmission and transformation CFs dataset.xlsx"

GENERATION_TO_FACTOR = {
    "circulating fluidized bed": "Coal - circulating fluidized bed",
    "subcritical": "Coal - subcritical",
    "supercritical": "Coal - supercritical",
    "ultra-supercritical": "Coal - ultra-supercritical",
    "Oil": "Oil",
    "Gas": "Gas",
    "Offshore wind": "Wind - Offshore",
    "Onshore wind": "Wind - Onshore",
    "Mono solar": "Solar - Mono",
    "Multi solar": "Solar - Multi",
    "Pumped storage": "Hydro - pumped storage",
    "Reservoir-based hydropower": "Hydro - reservoir-based hydropower",
    "Run-of-river hydropower": "Hydro - run-of-river",
    "Nuclear": "Nuclear",
}


def metric_record(
    name: str,
    predicted: pd.Series,
    observed: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> dict[str, float | str]:
    error = predicted - observed
    return {
        "method": name,
        "n_provinces": int(error.notna().sum()),
        "mae_kgco2e_kwh": float(error.abs().mean()),
        "rmse_kgco2e_kwh": float(np.sqrt(np.mean(np.square(error)))),
        "bias_kgco2e_kwh": float(error.mean()),
        "pearson_r": float(predicted.corr(observed)),
        "spearman_r": float(predicted.corr(observed, method="spearman")),
        "within_published_q025_q975": float(
            ((predicted >= lower) & (predicted <= upper)).mean()
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    generation = pd.read_excel(DATA / GENERATION_FILE).set_index("Province")
    technology_cf = pd.read_excel(DATA / TECH_CF_FILE).set_index("Province")
    summary_all = pd.read_excel(DATA / SUMMARY_FILE).set_index("Province")
    transfers_all = pd.read_excel(DATA / TRANSFER_FILE, index_col=0)
    td_all = pd.read_excel(DATA / TD_FILE).set_index("Region")

    provinces = [p for p in generation.index if p not in {"National", "Unit: TWh"}]
    generation = generation.loc[provinces].rename(columns=GENERATION_TO_FACTOR)
    factors = technology_cf.reindex(index=provinces, columns=generation.columns)
    national_factors = technology_cf.loc["National"].reindex(generation.columns)

    missing_factor_cells = int((factors.isna() & generation.gt(0)).to_numpy().sum())
    # The fallback is transparent and limited to technology cells with no
    # province-specific CF estimate; it must not be mistaken for observation.
    factors = factors.fillna(national_factors)
    if factors.isna().any().any():
        raise ValueError("Technology CF table still has missing factors after national fallback")

    transfers = transfers_all.reindex(index=provinces, columns=provinces).fillna(0.0)
    if transfers.isna().any().any() or (transfers.to_numpy() < 0).any():
        raise ValueError("OD matrix has missing or negative values after blank-to-zero conversion")

    generation_twh = generation.sum(axis=1)
    local_emissions_index = (generation * factors).sum(axis=1)
    local_factor = local_emissions_index / generation_twh
    imports_twh = transfers.sum(axis=0).reindex(provinces)
    exports_twh = transfers.sum(axis=1).reindex(provinces)

    # Proportional-sharing / node-flow balance. Matrix rows are origins and
    # columns are destinations, so incoming embodied intensity enters through M.T.
    balance = np.diag((generation_twh + imports_twh).to_numpy()) - transfers.T.to_numpy()
    if np.linalg.cond(balance) > 1e10:
        raise ValueError("Flow-tracing balance is ill-conditioned")
    flow_factor = pd.Series(
        np.linalg.solve(balance, local_emissions_index.to_numpy()), index=provinces
    )

    summary = summary_all.reindex(provinces)
    td = td_all["Transmission and transformation carbon footprints (kg CO2e/kWh)"].reindex(
        provinces
    )
    flow_plus_td = flow_factor + td

    metrics = [
        metric_record("local_generation_mix", local_factor, summary["Mean"], summary["Q2.5"], summary["Q97.5"]),
        metric_record("annual_od_flow_tracing", flow_factor, summary["Mean"], summary["Q2.5"], summary["Q97.5"]),
        metric_record(
            "annual_od_flow_tracing_plus_published_td",
            flow_plus_td,
            summary["Mean"],
            summary["Q2.5"],
            summary["Q97.5"],
        ),
    ]
    province_table = pd.DataFrame(
        {
            "province": provinces,
            "generation_twh": generation_twh,
            "imports_twh": imports_twh,
            "exports_twh": exports_twh,
            "local_generation_factor": local_factor,
            "od_flow_factor": flow_factor,
            "published_td_factor": td,
            "od_plus_td_factor": flow_plus_td,
            "published_cf_mean": summary["Mean"],
            "published_q025": summary["Q2.5"],
            "published_q975": summary["Q97.5"],
        }
    )
    province_table.to_csv(OUT / "province_comparison.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)

    payload = {
        "experiment": "V310 annual OD matrix source-dataset consistency audit",
        "independent_validation": False,
        "n_provinces": len(provinces),
        "generation_total_twh": float(generation_twh.sum()),
        "transfer_total_twh": float(transfers.to_numpy().sum()),
        "matrix_nonzero_edges": int((transfers.to_numpy() > 0).sum()),
        "matrix_closure_abs_twh": float(abs(transfers.sum(axis=1).sum() - transfers.sum(axis=0).sum())),
        "positive_generation_missing_province_factor_cells_national_imputed": missing_factor_cells,
        "mean_published_td_factor_kgco2e_kwh": float(td.mean()),
        "metrics": metrics,
        "interpretation": (
            "All inputs and the comparison CF distribution are from the same dataset; "
            "the comparison measures internal consistency only. Annual OD and LCA CF "
            "do not establish hourly generation-side CIF accuracy or coverage."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
