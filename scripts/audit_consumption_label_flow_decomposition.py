#!/usr/bin/env python3
"""Post-hoc decomposition of EIA BA consumption-intensity labels and interchange."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_official_intensity_transfer as official
from run_cross_country import ROOT

FIELDS = [
    'UTC time', 'Demand', 'Total interchange', 'Positive Generation',
    'Consumed Electricity', 'CO2 Emissions Generated', 'CO2 Emissions Imported',
    'CO2 Emissions Exported', 'CO2 Emissions Consumed',
    'CO2 Emissions Intensity for Generated Electricity',
    'CO2 Emissions Intensity for Consumed Electricity',
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method='spearman'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    rows, detail, hashes = [], [], {}
    for region in official.US_REGIONS:
        path = official.EIA_ROOT/f'{region}.xlsx'
        hashes[str(path)] = sha(path)
        raw = pd.read_excel(path, sheet_name='Published Hourly Data', usecols=FIELDS)
        time = pd.to_datetime(raw['UTC time'], utc=True, errors='raise')
        d = raw.loc[time.dt.year.eq(2024)].copy().reset_index(drop=True)
        numeric = [c for c in FIELDS if c != 'UTC time']
        d[numeric] = d[numeric].apply(pd.to_numeric, errors='coerce')
        d['time_utc'] = pd.to_datetime(d['UTC time'], utc=True, errors='raise')
        d['net_import_mwh'] = -d['Total interchange']
        d['net_import_share'] = d.net_import_mwh / d['Demand'].replace(0., np.nan)
        d['net_imported_co2'] = d['CO2 Emissions Imported'] - d['CO2 Emissions Exported']
        d['net_imported_co2_share'] = d.net_imported_co2 / d['CO2 Emissions Consumed'].replace(0., np.nan)
        d['consumption_minus_generation_intensity'] = (
            d['CO2 Emissions Intensity for Consumed Electricity']
            - d['CO2 Emissions Intensity for Generated Electricity'])
        d['emissions_accounting_residual'] = (
            d['CO2 Emissions Consumed'] - d['CO2 Emissions Generated']
            - d['CO2 Emissions Imported'] + d['CO2 Emissions Exported'])
        d['energy_accounting_residual'] = (
            d['Consumed Electricity'] - d['Positive Generation'] + d['Total interchange'])
        keep = d[['time_utc','net_import_mwh','net_import_share','net_imported_co2',
                  'net_imported_co2_share','consumption_minus_generation_intensity',
                  'CO2 Emissions Generated','CO2 Emissions Imported','CO2 Emissions Exported',
                  'CO2 Emissions Consumed','Positive Generation','Consumed Electricity',
                  'emissions_accounting_residual','energy_accounting_residual']].copy()
        keep.insert(0, 'region', region)
        detail.append(keep)
        valid = d[['net_import_share','net_imported_co2_share',
                   'consumption_minus_generation_intensity']].replace([np.inf,-np.inf],np.nan).dropna()
        emission_total = d['CO2 Emissions Consumed'].sum(min_count=1)
        rows.append({
            'region': region,
            'rows_2024': len(d),
            'valid_flow_and_label_rows': len(valid),
            'mean_net_import_share': float(valid.net_import_share.mean()),
            'sd_net_import_share': float(valid.net_import_share.std()),
            'mean_net_imported_co2_share': float(valid.net_imported_co2_share.mean()),
            'sd_net_imported_co2_share': float(valid.net_imported_co2_share.std()),
            'mean_consumption_minus_generation_ci_lb_kwh': float(valid.consumption_minus_generation_intensity.mean()),
            'sd_consumption_minus_generation_ci_lb_kwh': float(valid.consumption_minus_generation_intensity.std()),
            'spearman_energy_share_vs_emission_share': safe_corr(valid.net_import_share, valid.net_imported_co2_share),
            'spearman_energy_share_vs_ci_gap': safe_corr(valid.net_import_share, valid.consumption_minus_generation_intensity),
            'spearman_emission_share_vs_ci_gap': safe_corr(valid.net_imported_co2_share, valid.consumption_minus_generation_intensity),
            'emissions_identity_max_abs_residual': float(d.emissions_accounting_residual.abs().max()),
            'energy_identity_max_abs_residual_mwh': float(d.energy_accounting_residual.abs().max()),
            'net_imported_co2_share_of_annual_consumed_emissions': float(
                d.net_imported_co2.sum(min_count=1)/emission_total),
        })
    detail_frame = pd.concat(detail, ignore_index=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'region_decomposition.csv', index=False)
    detail_frame.to_csv(out/'hourly_decomposition.csv.gz', index=False, compression='gzip')
    (out/'hashes.json').write_text(json.dumps({'inputs':hashes,'code':{str(Path(__file__)):sha(Path(__file__))}},indent=2))
    status = {
        'status':'POST_HOC_LABEL_DECOMPOSITION_NO_MODEL_FITTED',
        'year':2024, 'regions':list(official.US_REGIONS),
        'target_hourly_labels_previously_viewed':True,
        'purpose':'Characterize energy-flow and assigned-emissions components in the official consumption-intensity response; not a deployable feature test.',
        'label_units':'Published EIA consumption and generation intensity columns are lb CO2/kWh.',
        'flow_sign':'positive derived net_import_mwh means imports; source Total interchange is negative for net inflow.',
    }
    (out/'status.json').write_text(json.dumps(status,indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
