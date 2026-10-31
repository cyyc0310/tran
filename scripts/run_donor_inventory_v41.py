"""Curate the fixed viewed-donor inventory; no generation, emissions or model I/O."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import donor_inventory_v41 as model
from scripts import fleet_inventory_v33 as legacy
from scripts import run_coarse_residual_v40 as parent

sha, read, write, verify, now = parent.sha, parent.read, parent.write, parent.verify, parent.now
OUT = ROOT / 'results/donor-inventory-v41'
PARENT = ROOT / 'docs/evidence/coarse-residual-v40-verification.json'
PARENT_SHA = 'e4684d1c4f749ca3ccb79ab79058aa133be926597ebd66ef26e11335b47f8e7d'
LEGACY = ROOT / 'docs/evidence/fleet-inventory-v33-verification.json'
LEGACY_SHA = 'f3649828c812a19bcc81371c3f1960239fc6c8572962afd61c6bf7be127aebe8'
ROLE = ROOT / 'docs/RISK_TRANSFER_V5_RESULTS.md'
CODE = ('scripts/donor_inventory_v41.py', 'scripts/run_donor_inventory_v41.py',
        'tests/test_donor_inventory_v41.py', 'tests/test_run_donor_inventory_v41.py',
        'docs/DONOR_INVENTORY_V41_PROTOCOL.md')
REVIEWS = tuple(ROOT / f'results/donor-inventory-v41-{kind}-review.md' for kind in ('code', 'math'))
OUTPUTS = ('capacity_op.csv', 'capacity_operable.csv', 'units.csv', 'excluded.csv', 'audit.json', 'legacy_match.json')
CSV_SCHEMAS = dict(op=('capacity_op.csv', legacy.CAPACITY_COLUMNS),
                   operable=('capacity_operable.csv', legacy.CAPACITY_COLUMNS),
                   units=('units.csv', legacy.UNIT_COLUMNS), excluded=('excluded.csv', legacy.EXCLUDED_COLUMNS))


def contract() -> dict:
    verify({str(PARENT): PARENT_SHA, str(LEGACY): LEGACY_SHA})
    evidence = read(PARENT)
    if evidence['status'] != 'VERIFICATION_COMPLETE' or evidence['independent_formula_review_status'] != 'PASS':
        raise ValueError('Verified V40 required')
    closure, seen = {}, set()
    while True:
        for p, digest in parent.parent.mappings(evidence).items():
            if p in closure and closure[p] != digest:
                raise ValueError('Conflicting inherited mapping')
            closure[p] = digest
        if 'parent_manifest' not in evidence:
            break
        p, h = evidence['parent_manifest'], evidence['parent_manifest_sha256']
        if p in seen:
            raise ValueError('Cyclic evidence')
        seen.add(p)
        verify({p: h})
        evidence = read(p)
    verify(closure)
    paths = {str(y): str(ROOT / f'results/fleet-schema-v33/raw/eia860{y}.zip') for y in legacy.SUPPORTED_YEARS}
    old = {kind: str(ROOT / 'results/fleet-inventory-v33' / name) for kind, (name, _) in CSV_SCHEMAS.items()}
    aliases = [*paths.values(), *old.values(), str(ROOT / 'scripts/fleet_inventory_v33.py')]
    if any(p not in closure for p in aliases) or closure.get(str(LEGACY)) != LEGACY_SHA:
        raise ValueError('Bound legacy parser, inventories and archives required')
    inputs = {p: closure[p] for p in aliases} | {str(PARENT): PARENT_SHA, str(LEGACY): LEGACY_SHA, str(ROLE): sha(ROLE)}
    for p in REVIEWS:
        if not p.is_file() or p.is_symlink() or not p.read_text().strip():
            raise ValueError('Two independent prelaunch reviews required')
        inputs[str(p)] = sha(p)
    return dict(version=41, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE}, input_hashes=inputs,
        environment=parent.common.environment(), sources=list(model.ALL_SOURCES), donors=list(model.DONOR_SOURCES),
        legacy_sources=list(model.LEGACY_SOURCES), years=list(legacy.SUPPORTED_YEARS),
        raw_archive_paths=paths, legacy_csv_paths=old, release_dates={str(y): d for y, d in legacy.RELEASE_DATES.items()},
        registered_BA_proxy_only=True, authenticated_first_release_bytes=False, certified_capacity_bounds=False,
        generation_or_CIF_values_read=False, models_trained=False, hourly_values_read=False,
        target_values_opened=False, unseen_values_opened=False, method_innovation_claim=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V41 attempt')
    frozen = contract()
    OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared', len(frozen['input_hashes']), 'direct inputs', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('Frozen V41 inputs/code/environment changed')
    return frozen


def csv_bytes(rows: list[dict], columns: tuple[str, ...]) -> bytes:
    if any(set(row) != set(columns) for row in rows):
        raise ValueError('Exact CSV schema required')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def legacy_match(combined: dict, paths: dict) -> dict:
    matches = {}
    for kind, (name, columns) in CSV_SCHEMAS.items():
        subset = [r for r in combined[kind] if r['region'] in model.LEGACY_SOURCES]
        value = csv_bytes(subset, columns)
        expected = Path(paths[kind]).read_bytes()
        if value != expected:
            raise ValueError('Legacy subset changed: '+kind)
        matches[kind] = dict(path=paths[kind], sha256=sha(Path(paths[kind])), rows=len(subset), bytes_exact=True)
    return matches


def curate() -> None:
    if any((OUT / p).exists() for p in (*OUTPUTS, 'curation_started.json', 'completed.json')):
        raise FileExistsError('Preserve V41 curation attempt')
    frozen = check()
    initial = sha(OUT / 'freeze.json')
    write(OUT / 'curation_started.json', dict(created_utc=now(), freeze_sha256=initial, sources=list(model.ALL_SOURCES)))
    started = sha(OUT / 'curation_started.json')
    combined = {kind: [] for kind in CSV_SCHEMAS}
    audits = {}
    for year in legacy.SUPPORTED_YEARS:
        with zipfile.ZipFile(frozen['raw_archive_paths'][str(year)]) as package:
            plants = legacy.read_workbook_rows(package.read(f'2___Plant_Y{year}.xlsx'), 'Plant', legacy.PLANT_COLUMNS)
            generators = legacy.read_workbook_rows(package.read(f'3_1_Generator_Y{year}.xlsx'), 'Operable', legacy.GENERATOR_COLUMNS)
            result = model.curate_year(year, plants, generators)
        if set(result) != {'op', 'operable', 'units', 'excluded', 'audit'}:
            raise ValueError('Exact curation schema required')
        for policy in ('op', 'operable'):
            if len(result[policy]) != 17 or {r['region'] for r in result[policy]} != set(model.ALL_SOURCES):
                raise ValueError('Complete seventeen-region annual tables required')
        if any(r['year'] != year or r['region'] not in model.ALL_SOURCES for kind in combined for r in result[kind]):
            raise ValueError('Only fixed source-year records allowed')
        for kind in combined:
            combined[kind].extend(result[kind])
        audits[str(year)] = result['audit']
        print('Curated', year, 'units', len(result['units']), 'storage exclusions', len(result['excluded']), flush=True)
    matches = legacy_match(combined, frozen['legacy_csv_paths'])
    check()
    verify({str(OUT / 'freeze.json'): initial, str(OUT / 'curation_started.json'): started})
    # All years and all four exact legacy-subset checks succeed before numeric output.
    for kind, (name, columns) in CSV_SCHEMAS.items():
        with (OUT / name).open('xb') as stream:
            stream.write(csv_bytes(combined[kind], columns))
    write(OUT / 'audit.json', dict(version=41, years=audits, sources=list(model.ALL_SOURCES),
        donors=list(model.DONOR_SOURCES), registered_BA_proxy_only=True, certified_capacity_bounds=False,
        authenticated_first_release_bytes=False, observed_zero_is_physical_absence=False,
        generation_or_CIF_values_read=False, models_trained=False, unseen_values_opened=False))
    write(OUT / 'legacy_match.json', dict(legacy_sources=list(model.LEGACY_SOURCES), files=matches, all_bytes_exact=True))
    check()
    verify({str(OUT / 'freeze.json'): initial, str(OUT / 'curation_started.json'): started})
    write(OUT / 'completed.json', dict(created_utc=now(), freeze_sha256=initial, curation_started_sha256=started,
        output_hashes={str(OUT / p): sha(OUT / p) for p in OUTPUTS}, source_year_rows_per_policy=68,
        unit_rows=len(combined['units']), excluded_rows=len(combined['excluded']),
        legacy_subsets_bytes_exact=True, other_BA_capacity_values_converted=False,
        generation_or_CIF_values_read=False, models_trained=False, hourly_values_read=False,
        target_values_opened=False, unseen_values_opened=False, method_innovation_claim=False))
    print('Inventory complete; legacy subsets exact; no generation or model used', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'curate'), required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'curate': curate}[args.stage]()
