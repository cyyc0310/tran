"""Immutable source-only EIA-860 nameplate inventory curation, without CIF I/O."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts import fleet_inventory_v33 as model
from scripts import run_coarse_context_v30 as common

OUT = ROOT/'results/fleet-inventory-v33'
PARENTS = {
    ROOT/'docs/evidence/history-signature-v32-verification.json': '8e3cbfe354deab91e4f6f42e518f8340677c075339a88a55558791fe1d081a4a',
    ROOT/'docs/evidence/fleet-schema-v33-acquisition.json': 'bfd20570f0b4c58e05108c93a1aaa41b0a95874574fd8e32e116626596ee3398',
}
CODE = ('scripts/fleet_inventory_v33.py', 'scripts/run_fleet_inventory_v33.py',
        'tests/test_fleet_inventory_v33.py', 'tests/test_run_fleet_inventory_v33.py',
        'docs/FLEET_INVENTORY_V33_PROTOCOL.md')
REVIEWS = tuple(ROOT/f'results/fleet-inventory-v33-{kind}-review.md' for kind in ('code', 'python', 'science'))
YEARS = (2020, 2021, 2022, 2023)
RELEASES = {2020: '2021-09-09', 2021: '2022-09-22', 2022: '2023-09-19', 2023: '2024-09-23'}
OUTPUTS = ('capacity_op.csv', 'capacity_operable.csv', 'units.csv', 'excluded.csv', 'audit.json')
sha, read, write, verify, now = common.sha, common.read, common.write, common.verify, common.now


def expected_contract() -> dict:
    inputs = {}
    for path, digest in PARENTS.items():
        if sha(path) != digest:
            raise ValueError('Fixed research and metadata parents required')
        inputs[str(path)] = digest
        evidence = read(path)
        if 'producer_code_hashes' in evidence:
            if evidence['status'] != 'VERIFICATION_COMPLETE' or evidence['independent_actual_audit_status'] != 'PASS':
                raise ValueError('Independent actual V32 PASS required')
            fields = ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes')
        else:
            if (evidence['status'] != 'METADATA_ONLY_VERIFIED' or evidence['artifact_count'] != 12
                    or evidence['numeric_capacity_values_converted'] is not False or evidence['models_trained'] is not False):
                raise ValueError('Fixed metadata-only acquisition required')
            fields = ('artifact_hashes',)
        for field in fields:
            for name, value in evidence[field].items():
                if name in inputs and inputs[name] != value:
                    raise ValueError('Conflicting parent hash mapping')
                inputs[name] = value
    paths = {str(y): str(ROOT/f'results/fleet-schema-v33/raw/eia860{y}.zip') for y in YEARS}
    if any(p not in inputs for p in paths.values()) or model.RELEASE_DATES != RELEASES:
        raise ValueError('Fixed four inventory aliases and release dates required')
    for path in REVIEWS:
        if not path.is_file() or path.is_symlink() or not path.read_text().strip():
            raise ValueError('Three final curation prelaunch reviews required')
        inputs[str(path)] = sha(path)
    if any(not Path(p).is_file() or Path(p).is_symlink() for p in inputs):
        raise ValueError('Regular bound input files required')
    verify(inputs)
    return dict(version=33, stage='SOURCE_CAPACITY_CURATION', sources=list(model.SOURCES), years=list(YEARS),
        raw_archive_paths=paths, release_dates={str(y): RELEASES[y] for y in YEARS},
        plant_columns=list(model.PLANT_COLUMNS), generator_columns=list(model.GENERATOR_COLUMNS),
        capacity_columns=list(model.CAPACITY_COLUMNS), unit_columns=list(model.UNIT_COLUMNS),
        excluded_columns=list(model.EXCLUDED_COLUMNS), inventory_selection='origin.year - 2; release <= origin; no fallback',
        curation_attempt_marker='curation_started.json',
        code_hashes={str(ROOT/p): sha(ROOT/p) for p in CODE}, input_hashes=inputs, environment=common.environment(),
        main_statuses=['OP'], sensitivity_statuses=['OP', 'SB', 'OA', 'OS'], nameplate_only=True,
        registered_BA_scope_only=True, certified_capacity_bounds=False, authenticated_first_release_bytes=False,
        other_BA_capacity_values_converted=False, CIF_or_generation_data_used=False, hourly_values_read=False,
        target_values_opened=False, unseen_values_opened=False, models_trained=False, custom_method=False)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError('Preserve V33 curation attempt')
    contract = expected_contract()
    OUT.mkdir()
    write(OUT/'freeze.json', dict(created_utc=now(), **contract))
    print('Prepared', len(contract['code_hashes']), 'code', len(contract['input_hashes']), 'input hashes', flush=True)


def check() -> dict:
    frozen, expected = read(OUT/'freeze.json'), expected_contract()
    if (set(frozen) != set(expected)|{'created_utc'} or any(frozen[k] != v for k, v in expected.items())
            or any(frozen[k] is not v for k, v in expected.items() if isinstance(v, bool))):
        raise ValueError('Frozen source capacity contract changed')
    return frozen


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    if any(set(row) != set(columns) for row in rows):
        raise ValueError('Exact declared CSV schema required')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def curate() -> None:
    if any((OUT/name).exists() for name in (*OUTPUTS, 'curation_started.json', 'source_completed.json')):
        raise FileExistsError('Preserve V33 curation outputs')
    frozen = check()
    initial = sha(OUT/'freeze.json')
    write(OUT/'curation_started.json', dict(created_utc=now(), freeze_sha256=initial, years=list(YEARS), source_only=True))
    started = sha(OUT/'curation_started.json')
    combined = {key: [] for key in ('op', 'operable', 'units', 'excluded')}
    audits = {}
    for year in YEARS:
        with zipfile.ZipFile(frozen['raw_archive_paths'][str(year)]) as archive:
            plant_data = archive.read(f'2___Plant_Y{year}.xlsx')
            generator_data = archive.read(f'3_1_Generator_Y{year}.xlsx')
        plants = model.read_workbook_rows(plant_data, 'Plant', model.PLANT_COLUMNS)
        generators = model.read_workbook_rows(generator_data, 'Operable', model.GENERATOR_COLUMNS)
        result = model.curate_year(year, plants, generators)
        if set(result) != {'op', 'operable', 'units', 'excluded', 'audit'}:
            raise ValueError('Exact curation result schema required')
        for policy in ('op', 'operable'):
            rows = result[policy]
            if len(rows) != 7 or {r['region'] for r in rows} != set(model.SOURCES) or any(r['year'] != year for r in rows):
                raise ValueError('Complete seven-source annual capacity rows required')
        if any(r['region'] not in model.SOURCES or r['year'] != year for key in ('units', 'excluded') for r in result[key]):
            raise ValueError('Only registered source records may be written')
        for key in combined:
            combined[key].extend(result[key])
        audits[str(year)] = result['audit']
        print('Curated source inventory', year, 'units', len(result['units']), 'storage exclusions', len(result['excluded']), flush=True)
    # No partial numeric table is written until all four source inventories succeed.
    check()
    verify({str(OUT/'freeze.json'): initial, str(OUT/'curation_started.json'): started})
    write_csv(OUT/'capacity_op.csv', combined['op'], model.CAPACITY_COLUMNS)
    write_csv(OUT/'capacity_operable.csv', combined['operable'], model.CAPACITY_COLUMNS)
    write_csv(OUT/'units.csv', combined['units'], model.UNIT_COLUMNS)
    write_csv(OUT/'excluded.csv', combined['excluded'], model.EXCLUDED_COLUMNS)
    write(OUT/'audit.json', dict(version=33, years=audits, raw_archive_paths=frozen['raw_archive_paths'],
        numeric_capacity_values_converted=True, source_only=True, other_BA_capacity_values_converted=False,
        main_statuses=['OP'], sensitivity_statuses=['OP', 'SB', 'OA', 'OS'], registered_BA_scope_only=True,
        authenticated_first_release_bytes=False, certified_capacity_bounds=False, CIF_or_generation_data_used=False,
        hourly_values_read=False, target_values_opened=False, unseen_values_opened=False, models_trained=False))
    check()
    verify({str(OUT/'freeze.json'): initial, str(OUT/'curation_started.json'): started})
    write(OUT/'source_completed.json', dict(version=33, stage='SOURCE_CAPACITY_CURATED', created_utc=now(),
        freeze_sha256=initial, curation_started_sha256=started,
        output_hashes={str(OUT/n): sha(OUT/n) for n in OUTPUTS}, years=list(YEARS),
        source_year_rows_per_policy=28, unit_rows=len(combined['units']), excluded_rows=len(combined['excluded']),
        source_only=True, numeric_capacity_values_converted=True, other_BA_capacity_values_converted=False,
        models_trained=False, CIF_or_generation_data_used=False, hourly_values_read=False,
        target_values_opened=False, unseen_values_opened=False, certified_capacity_bounds=False, custom_method=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'curate'))
    args = parser.parse_args()
    {'prepare': prepare, 'curate': curate}[args.stage]()
