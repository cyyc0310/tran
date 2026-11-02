"""Build an explicitly retrospective 2020 coal/gas capacity information budget.

This uses annual public records as a sensitivity budget. DE/UK workbook versions
are not authenticated as-issued snapshots, and UK remains a geographic proxy for
GB. Nothing here treats capacity as an hourly dispatch bound.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = ROOT / 'results/public-capacity-input-r1/capacity.json'
    audit = ROOT / 'docs/evidence/public-capacity-annual-audit-r1/audit.json'
    uk = ROOT / 'data/capacity-public/official-2021/UK-DUKES-2021.xlsx'
    de = ROOT / 'data/capacity-public/official-2021/DE-monitoring-2021.pdf'
    output = ROOT / 'data/capacity-public/annual-2021/annual-coal-gas-budget-2020.json'
    table = json.loads(base.read_text())
    audited = json.loads(audit.read_text())['values']
    by_region = {row['region']: row for row in audited}
    table['status'] = 'ANNUAL_2020_CAPACITY_INFORMATION_BUDGET_RETROSPECTIVE_VERSION_LIMITED'
    table['regions']['DE'].update({
        'coal_mw': by_region['DE']['annual_coal_mw'],
        'gas_mw': by_region['DE']['annual_gas_mw'], 'capacity_year': 2020,
        'source': 'German Monitoringbericht 2021, current revised PDF',
        'scope': 'Germany; lignite plus hard coal; report imprint 2022-03-15',
    })
    table['regions']['GB'].update({
        'coal_mw': by_region['GB']['annual_coal_mw'],
        'gas_mw': by_region['GB']['annual_gas_mw'], 'capacity_year': 2020,
        'source': 'DUKES 2021 table 5.7, current workbook',
        'scope': 'UK including Northern Ireland; all generating companies',
    })
    # These are the original publication dates recorded in the source audit.
    # Current workbook/report bytes were later modified, so the data values are
    # only a retrospective approximation of the then-public annual budget.
    table['regions']['DE']['available_by_utc'] = '2021-12-02T00:00:00Z'
    table['regions']['GB']['available_by_utc'] = '2021-07-30T00:00:00Z'
    inputs = [base, audit, uk, de, ROOT/'results/fleet-schema-v33/raw/eia8602020.zip', Path(__file__)]
    table['input_hashes'] = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    table['annual_capacity_budget'] = {
        'statistic_year': 2020,
        'availability_contract': 'retrospective annual budget; current DE/UK vintages are not authenticated as-issued snapshots',
        'US': 'EIA-860 2020 OP inventory by registered balancing authority',
        'GB': 'UK aggregate proxy, includes Northern Ireland',
        'DE': 'German current report version, imprinted 2022-03-15',
        'capacity_is_hard_hourly_bound': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2))
    print(json.dumps({'output': str(output.relative_to(ROOT)),
                      'sha256': sha(output), 'inputs': table['input_hashes']}, indent=2))


if __name__ == '__main__':
    main()
