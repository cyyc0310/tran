"""Pure V33-equivalent inventory curation for fixed viewed development donors."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from scripts import fleet_inventory_v33 as legacy
from scripts.fleet_inventory_v33 import (
    CAPACITY_COLUMNS, ENERGY_SOURCE_TO_FUEL, EXCLUDED_COLUMNS, FUEL_CODES_BY_YEAR,
    FUELS, GENERATOR_COLUMNS, OPERABLE_STATUSES, PLANT_COLUMNS, PRIME_MOVERS,
    RELEASE_DATES, STORAGE_PRIME_MOVERS, SUPPORTED_YEARS, UNIT_COLUMNS,
    _aggregate, _capacity, _plant_code, _text, inventory_for_origin, read_workbook_rows,
)


LEGACY_SOURCES = legacy.SOURCES
DONOR_SOURCES = ('SWPP', 'AVA', 'BANC', 'JEA', 'SC', 'NWMT', 'SCL', 'WACM', 'SEC', 'SCEG')
ALL_SOURCES = LEGACY_SOURCES+DONOR_SOURCES


def curate_year(
    year: int,
    plant_rows: Iterable[Mapping[str, str]],
    generator_rows: Iterable[Mapping[str, str]],
    *, sources: tuple[str, ...] = ALL_SOURCES,
) -> dict[str, Any]:
    """Preserve V33 validation/access/order exactly for either fixed source tuple."""
    if type(sources) is not tuple or sources not in (LEGACY_SOURCES, ALL_SOURCES):
        raise ValueError('sources must equal the ordered legacy-seven or fixed-seventeen tuple')
    if type(year) is not int or year not in SUPPORTED_YEARS:
        raise ValueError('Unsupported inventory year')
    plant_map: dict[str, str] = {}
    plant_blank = plant_outside = 0
    for row in plant_rows:
        plant = _plant_code(_text(row, 'Plant Code'))
        if plant in plant_map:
            raise ValueError(f'Duplicate plant identifier: {plant}')
        ba = _text(row, 'Balancing Authority Code').strip().upper()
        plant_map[plant] = ba
        plant_blank += not ba
        plant_outside += bool(ba) and ba not in sources

    counts = Counter()
    selected = []
    seen = set()
    for row in generator_rows:
        counts['generator_rows'] += 1
        plant = _plant_code(_text(row, 'Plant Code'))
        if plant not in plant_map:
            counts['unmapped_plant_rows'] += 1
            continue
        ba = plant_map[plant]
        if not ba:
            counts['blank_ba_rows'] += 1
            continue
        if ba not in sources:
            counts['outside_source_rows'] += 1
            continue
        gen = _text(row, 'Generator ID').strip()
        if not gen:
            raise ValueError('Selected Generator ID must be nonempty')
        key = (plant, gen)
        if key in seen:
            raise ValueError(f'Duplicate generator identifier: {plant}/{gen}')
        seen.add(key)
        selected.append((ba, plant, gen, row))

    units, excluded = [], []
    selected_status, selected_energy = Counter(), Counter()
    for region, plant, gen, row in selected:
        status = _text(row, 'Status').strip().upper()
        pm = _text(row, 'Prime Mover').strip().upper()
        energy = _text(row, 'Energy Source 1').strip().upper()
        if status not in OPERABLE_STATUSES:
            raise ValueError(f'Unsupported selected generator status at {plant}/{gen}')
        if pm not in PRIME_MOVERS | STORAGE_PRIME_MOVERS:
            raise ValueError(f'Unknown or planning-only prime mover at {plant}/{gen}')
        if energy not in FUEL_CODES_BY_YEAR[year] | {'MWH'}:
            raise ValueError(f'Unknown or year-ineligible primary energy source at {plant}/{gen}')
        selected_status[status] += 1
        selected_energy[energy] += 1
        metadata = dict(year=year, region=region, plant_code=plant, generator_id=gen,
                        status=status, prime_mover=pm, primary_energy_source=energy)
        if energy == 'MWH' or pm in STORAGE_PRIME_MOVERS:
            reason = ('storage_energy_source_and_prime_mover' if energy == 'MWH' and pm in STORAGE_PRIME_MOVERS
                      else 'storage_energy_source' if energy == 'MWH' else 'storage_prime_mover')
            record = {**metadata, 'reason': reason}
            excluded.append({name: record[name] for name in EXCLUDED_COLUMNS})
            continue
        secondary = []
        for i in range(2, 7):
            code = _text(row, f'Energy Source {i}').strip().upper()
            if code and code != energy and code not in secondary:
                secondary.append(code)
        record = {
            **metadata, 'fuel': ENERGY_SOURCE_TO_FUEL[energy], 'capacity_mw': _capacity(row),
            'secondary_energy_sources': ';'.join(secondary),
            'multi_fuels_flag': _text(row, 'Multiple Fuels?'),
            'cofire_flag': _text(row, 'Cofire Fuels?'), 'switch_flag': _text(row, 'Switch Between Oil and Natural Gas?'),
            'in_op': status == 'OP',
        }
        units.append({name: record[name] for name in UNIT_COLUMNS})

    def order(record: Mapping[str, Any]) -> tuple[int, int, str]:
        return sources.index(record['region']), int(record['plant_code']), record['generator_id']

    units.sort(key=order)
    excluded.sort(key=order)
    op, operable = [], []
    for region in sources:
        sensitive = [u for u in units if u['region'] == region]
        operating = [u for u in sensitive if u['in_op']]
        main = _aggregate(year, region, operating)
        if not operating or main['capacity_total_mw'] <= 0:
            raise ValueError(f'{region} requires a positive nonstorage OP inventory in {year}')
        op.append(main)
        operable.append(_aggregate(year, region, sensitive))

    audit = {
        'year': year, 'scope': 'registered_ba_inventory_proxy',
        'capacity_measure': 'nameplate_mw', 'coverage_certified': False,
        'historical_vintage_certified': False, 'observed_zero_is_physical_absence': False,
        'hard_output_bound': False, 'secondary_fuels_duplicate_capacity': False,
        'plant_rows': len(plant_map), 'plant_blank_ba_rows': int(plant_blank),
        'plant_outside_source_rows': int(plant_outside),
        **{key: counts[key] for key in ('generator_rows', 'unmapped_plant_rows', 'blank_ba_rows', 'outside_source_rows')},
        'selected_rows': len(selected), 'excluded_storage_rows': len(excluded), 'curated_rows': len(units),
        'selected_by_status': {s: selected_status[s] for s in OPERABLE_STATUSES},
        'selected_by_primary_energy_source': dict(sorted(selected_energy.items())),
        'excluded_storage_by_status': {s: sum(u['status'] == s for u in excluded) for s in OPERABLE_STATUSES},
        'curated_by_status': {s: sum(u['status'] == s for u in units) for s in OPERABLE_STATUSES},
        'curated_by_fuel': {f: sum(u['fuel'] == f for u in units) for f in FUELS},
        'op_by_fuel': {f: sum(u['fuel'] == f and u['in_op'] for u in units) for f in FUELS},
    }
    return dict(op=op, operable=operable, units=units, excluded=excluded, audit=audit)
