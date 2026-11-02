"""Audit 2020 annual source capacities for the four previously viewed donor regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

import fleet_inventory_v33 as inventory
from run_cross_country import ROOT, digest

ADDITIONS = ('PACE', 'LGEE', 'SRP', 'DUK')


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    archive_path = ROOT / 'results/fleet-schema-v33/raw/eia8602020.zip'
    acquisition = ROOT / 'results/fleet-schema-v33/acquisition-2020.json'
    parent = ROOT / 'results/public-capacity-input-r1/capacity.json'
    old_inventory = ROOT / 'results/fleet-inventory-v33/capacity_op.csv'
    paths = [archive_path, acquisition, parent, old_inventory, Path(__file__),
             ROOT / 'scripts/fleet_inventory_v33.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    if digest(archive_path) != json.loads(acquisition.read_text())['sha256']:
        raise ValueError('EIA-860 native archive hash mismatch')
    with zipfile.ZipFile(archive_path) as archive:
        plants = list(inventory.read_workbook_rows(archive.read('2___Plant_Y2020.xlsx'), 'Plant', inventory.PLANT_COLUMNS))
        generators = list(inventory.read_workbook_rows(archive.read('3_1_Generator_Y2020.xlsx'), 'Operable', inventory.GENERATOR_COLUMNS))
    original = inventory.curate_year(2020, plants, generators)
    old = pd.read_csv(old_inventory).query('year == 2020').set_index('region')
    checked = pd.DataFrame(original['op']).set_index('region').reindex(old.index)
    fields = [c for c in old if c.startswith('capacity_')]
    np.testing.assert_allclose(checked[fields], old[fields], atol=1e-8, rtol=0.)
    curated = inventory.curate_year(2020, plants, generators, sources=ADDITIONS)
    table = json.loads(parent.read_text())
    table['status'] = 'PUBLIC_STATIC_CAPACITY_PROXY_WITH_ANNUAL_US_DONORS'
    table['input_hashes'].update(hashes)
    for row in curated['op']:
        region = row['region']
        table['regions'][region] = {
            'coal_mw': row['capacity_coal_mw'], 'gas_mw': row['capacity_gas_mw'],
            'available_by_utc': '2021-09-10T00:00:00Z', 'capacity_year': 2020,
            'source': 'EIA860-2020-OP', 'scope': 'plant registered BA; not contemporaneous operating BA',
        }
    for p in paths:
        if digest(p) != hashes[str(p.relative_to(ROOT))]:
            raise RuntimeError('inventory inputs changed during preparation')
    (args.output / 'capacity.json').write_text(json.dumps(table, indent=2))
    pd.DataFrame(curated['op']).to_csv(args.output / 'capacity_op.csv', index=False)
    (args.output / 'audit.json').write_text(json.dumps({
        'curation': curated['audit'], 'input_hashes': hashes,
        'original_seven_capacity_reproduced_atol': 1e-8,
        'target_hourly_labels_read': False, 'new_confirmatory_regions': False,
        'note': 'All four donors were previously viewed; capacity zeros are not certified physical absence',
    }, indent=2))
    for region in ADDITIONS:
        c = table['regions'][region]
        print(region, c['coal_mw'], c['gas_mw'], 'coal_share', c['coal_mw'] / (c['coal_mw'] + c['gas_mw']))


if __name__ == '__main__':
    main()
