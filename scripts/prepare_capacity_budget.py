"""Freeze public EIA-860 and WRI stock proxies; target generation is never read."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_cross_country import ROOT, digest


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = ROOT / 'results/fleet-inventory-v33/capacity_op.csv'
    wri = ROOT / 'data/capacity-public/wri-v1-3/global_power_plant_database.csv'
    acquisition = ROOT / 'data/capacity-public/wri-v1-3/acquisition.json'
    eia_acquisition = ROOT / 'results/fleet-schema-v33/acquisition-2020.json'
    files = [inventory, wri, acquisition, eia_acquisition, Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    meta = json.loads(acquisition.read_text())
    if meta['csv_sha256'] != digest(wri):
        raise ValueError('WRI acquisition hash mismatch')
    eia = pd.read_csv(inventory)
    eia = eia.loc[eia.year.eq(2020)]
    if len(eia) != 7 or eia.region.duplicated().any():
        raise ValueError('expected original seven EIA source/target regions')
    regions = {}
    for row in eia.itertuples():
        regions[row.region] = {
            'coal_mw': row.capacity_coal_mw, 'gas_mw': row.capacity_gas_mw,
            'available_by_utc': '2021-09-10T00:00:00Z', 'capacity_year': 2020,
            'source': 'EIA860-2020-OP', 'scope': 'plant registered BA; not contemporaneous operating BA',
        }
    # Whitelist excludes all observed and model-estimated annual generation fields.
    frame = pd.read_csv(wri, usecols=['country', 'primary_fuel', 'capacity_mw',
                                    'year_of_capacity_data', 'other_fuel1', 'other_fuel2', 'other_fuel3'],
                        low_memory=False)
    audit = {}
    for region, country in [('DE', 'DEU'), ('GB', 'GBR')]:
        block = frame.loc[frame.country.eq(country)]
        stock = block.groupby('primary_fuel').capacity_mw.sum()
        regions[region] = {
            'coal_mw': float(stock['Coal']), 'gas_mw': float(stock['Gas']),
            'available_by_utc': '2021-06-03T00:00:00Z', 'capacity_year': None,
            'source': 'WRI-v1.3.0-2021-public-snapshot',
            'scope': 'country plant inventory; UK includes Northern Ireland' if region == 'GB' else 'country plant inventory',
        }
        audit[region] = {
            'plants': len(block), 'all_primary_fuel_capacity_mw': stock.to_dict(),
            'capacity_year_missing': int(block.year_of_capacity_data.isna().sum()),
            'multifuel_plants': int(block[['other_fuel1', 'other_fuel2', 'other_fuel3']].notna().any(axis=1).sum()),
        }
    for p in files:
        if digest(p) != hashes[str(p.relative_to(ROOT))]:
            raise RuntimeError('capacity input changed during preparation')
    table = {
        'status': 'PUBLIC_STATIC_CAPACITY_PROXY_BUDGET_NOT_ANNUAL_TRUTH',
        'regions': regions, 'input_hashes': hashes, 'audit': audit,
        'limits': ['No target generation read', 'Coal/gas primary-fuel stock only; oil/other unresolved',
                   'WRI year missing and incomplete population; no hard physical constraints',
                   'Does not certify provincial China inputs or historical full-coverage annual fleets'],
    }
    (args.output / 'capacity.json').write_text(json.dumps(table, indent=2))
    print(json.dumps(regions, indent=2))


if __name__ == '__main__':
    main()
