"""Parse a narrow, auditable Eurostat nrg_inf_epc JSON-stat extract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path: Path) -> tuple[list[dict], dict]:
    obj = json.loads(path.read_text())
    dims = obj['id']
    sizes = obj['size']
    if dims != ['freq', 'siec', 'plant_tec', 'operator', 'unit', 'geo', 'time']:
        raise ValueError(f'unexpected Eurostat dimension order in {path.name}: {dims}')
    if obj.get('updated') is None:
        raise ValueError('Eurostat response has no update timestamp')
    category = {d: obj['dimension'][d]['category']['index'] for d in dims}
    labels = obj['dimension']['siec']['category'].get('label', {})
    selected = {
        'TOTAL': 'total_mw', 'CF': 'thermal_mw', 'C0000': 'coal_mw',
        'G3000': 'gas_mw', 'O4000': 'oil_mw', 'RA100': 'hydro_mw',
        'RA300': 'wind_mw', 'RA420': 'solar_pv_mw', 'N9000': 'nuclear_other_mw',
        'R5000_W6000': 'bio_waste_mw',
    }
    value = obj.get('value', {})
    rows = []
    # Eurostat sparse JSON-stat uses row-major flattening, with the last
    # dimension varying fastest. The request fixes every dimension except SIEC
    # and year; retain explicit lookup assertions to guard this contract.
    if (sizes[0], sizes[2:6]) != (1, [1, 1, 1, 1]):
        raise ValueError(f'unexpected filtered dimensions: {sizes}')
    if category['freq'] != {'A': 0} or category['plant_tec'] != {'CAP_NET_ELC': 0}:
        raise ValueError('response filter did not select annual net capacity')
    if category['operator'] != {'TOTAL': 0} or category['unit'] != {'MW': 0}:
        raise ValueError('response filter did not select total MW values')
    country = next(iter(category['geo']))
    years = {int(y): i for y, i in category['time'].items()}
    for year, yi in years.items():
        record = {'country': country, 'stat_year': year}
        for code, column in selected.items():
            si = category['siec'].get(code)
            if si is None:
                record[column] = None
                continue
            flat = si * sizes[-1] + yi
            record[column] = value.get(str(flat))
        if record['total_mw'] is None or record['thermal_mw'] is None:
            raise ValueError(f'missing total/combustible capacity for {country} {year}')
        if record['total_mw'] <= 0 or not 0 <= record['thermal_mw'] <= record['total_mw'] * 1.05:
            raise ValueError(f'invalid total/thermal capacity for {country} {year}')
        record['thermal_capacity_share'] = record['thermal_mw'] / record['total_mw']
        rows.append(record)
    metadata = {
        'dataset': 'Eurostat nrg_inf_epc', 'updated': obj['updated'],
        'country': country, 'raw_sha256': digest(path),
        'siec_labels': {code: labels.get(code) for code in selected},
    }
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, metadata = [], []
    for path in sorted(args.input_dir.glob('[A-Z][A-Z].json')):
        parsed, audit = parse(path)
        rows.extend(parsed)
        metadata.append(audit)
    table = pd.DataFrame(rows).sort_values(['country', 'stat_year'])
    if table.duplicated(['country', 'stat_year']).any():
        raise ValueError('duplicate country/year capacity rows')
    table.to_csv(args.output, index=False)
    args.output.with_suffix('.manifest.json').write_text(json.dumps({
        'schema': 'eurostat-energy-charts-capacity-r1', 'metadata': metadata,
        'output_sha256': digest(args.output), 'output_rows': len(table),
        'availability_note': 'Latest revised Eurostat response; no historic release vintage. Use only as lagged-statistics simulation.',
    }, indent=2))
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
