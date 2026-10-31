"""Synthetic immutable curation execution and source-output boundary tests."""
import csv
import json
import zipfile

import pytest
from scripts import run_fleet_inventory_v33 as r


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def toy_rows():
    plants = [dict(zip(r.model.PLANT_COLUMNS, [str(i+1), source])) for i, source in enumerate(r.model.SOURCES)]
    plants += [{'Plant Code': '999', 'Balancing Authority Code': 'OUTSIDE'}]
    def gen(plant, identifier, status='OP', pm='GT', energy='NG', capacity='10'):
        row = {c: '' for c in r.model.GENERATOR_COLUMNS}
        row.update({'Plant Code': plant, 'Generator ID': identifier, 'Status': status,
            'Prime Mover': pm, 'Energy Source 1': energy, 'Nameplate Capacity (MW)': capacity})
        return row
    generators = [gen(str(i+1), '01') for i in range(7)]
    standby = gen('1', '02', status='SB', capacity='2')
    standby['Energy Source 2'] = 'RFO'
    generators += [standby, gen('1', 'battery', pm='BA', energy='MWH', capacity='STORAGE_POISON'),
                   gen('999', 'outside', capacity='OUTSIDE_POISON'), gen('1000', 'unmapped', capacity='UNMAPPED_POISON')]
    return plants, generators


def setup_curation(tmp_path, monkeypatch):
    out = tmp_path/'curation'
    out.mkdir()
    (out/'freeze.json').write_text('{}')
    paths = {}
    for year in r.YEARS:
        path = tmp_path/f'{year}.zip'
        with zipfile.ZipFile(path, 'x') as archive:
            archive.writestr(f'2___Plant_Y{year}.xlsx', b'synthetic-plant')
            archive.writestr(f'3_1_Generator_Y{year}.xlsx', b'synthetic-generator')
        paths[str(year)] = str(path)
    plants, generators = toy_rows()
    monkeypatch.setattr(r, 'OUT', out)
    monkeypatch.setattr(r, 'check', lambda: {'raw_archive_paths': paths})
    monkeypatch.setattr(r.model, 'read_workbook_rows', lambda data, sheet, columns:
                        iter(plants if sheet == 'Plant' else generators))
    return out


def test_complete_curation_keeps_two_policies_and_storage_capacity_unread(tmp_path, monkeypatch):
    out = setup_curation(tmp_path, monkeypatch)
    r.curate()
    assert {p.name for p in out.iterdir()} == set(r.OUTPUTS)|{'freeze.json', 'curation_started.json', 'source_completed.json'}
    op, all_units = read_csv(out/'capacity_op.csv'), read_csv(out/'capacity_operable.csv')
    assert len(op) == len(all_units) == 28
    assert all(float(x['capacity_gas_mw']) == 10. for x in op)
    assert all(float(x['capacity_gas_mw']) == (12. if x['region'] == 'BPAT' else 10.) for x in all_units)
    assert len(read_csv(out/'units.csv')) == 32 and len(read_csv(out/'excluded.csv')) == 4
    assert all('capacity' not in key.lower() for key in read_csv(out/'excluded.csv')[0])
    completed = r.read(out/'source_completed.json')
    assert len(completed['output_hashes']) == 5 and completed['models_trained'] is False
    assert completed['other_BA_capacity_values_converted'] is completed['CIF_or_generation_data_used'] is False
    assert completed['curation_started_sha256'] == r.sha(out/'curation_started.json')
    r.verify(completed['output_hashes'])
    with pytest.raises(FileExistsError):
        r.curate()


def test_late_failure_keeps_attempt_receipt_and_writes_no_partial_numeric_tables(tmp_path, monkeypatch):
    out = setup_curation(tmp_path, monkeypatch)
    real = r.model.curate_year
    def fail_last(year, *args):
        if year == 2023:
            raise ValueError('Synthetic final-year failure')
        return real(year, *args)
    monkeypatch.setattr(r.model, 'curate_year', fail_last)
    with pytest.raises(ValueError, match='final-year failure'):
        r.curate()
    assert {p.name for p in out.iterdir()} == {'freeze.json', 'curation_started.json'}
    monkeypatch.setattr(r, 'check', lambda: pytest.fail('Do not restart a failed attempt'))
    with pytest.raises(FileExistsError):
        r.curate()


@pytest.mark.parametrize('field,value', [('raw_archive_paths', {}), ('main_statuses', ['OP', 'SB']),
    ('nameplate_only', False), ('models_trained', True), ('other_BA_capacity_values_converted', 0)])
def test_contract_cannot_be_reauthorized_by_editing_freeze(monkeypatch, field, value):
    expected = dict(raw_archive_paths={'2020': '/fixed'}, main_statuses=['OP'], nameplate_only=True,
                    models_trained=False, other_BA_capacity_values_converted=False)
    frozen = dict(created_utc='fixed', **expected)
    monkeypatch.setattr(r, 'expected_contract', lambda: expected)
    monkeypatch.setattr(r, 'read', lambda _: frozen)
    assert r.check() == frozen
    frozen[field] = value
    with pytest.raises(ValueError, match='contract changed'):
        r.check()


def test_output_schema_and_empty_excluded_header_are_preserved(tmp_path):
    path = tmp_path/'empty.csv'
    r.write_csv(path, [], r.model.EXCLUDED_COLUMNS)
    with path.open(newline='') as stream:
        assert next(csv.reader(stream)) == list(r.model.EXCLUDED_COLUMNS)
    bad = tmp_path/'bad.csv'
    with pytest.raises(ValueError, match='declared CSV schema'):
        r.write_csv(bad, [{'region': 'BPAT', 'unexpected': 'x'}], r.model.CAPACITY_COLUMNS)
    assert not bad.exists()


def test_existing_prepare_attempt_precedes_parent_access(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'expected_contract', lambda: pytest.fail('Do not overwrite attempt'))
    with pytest.raises(FileExistsError):
        r.prepare()


def test_parent_closure_requires_the_actual_four_consumed_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    archive_paths = [tmp_path/f'results/fleet-schema-v33/raw/eia860{y}.zip' for y in r.YEARS]
    for path in archive_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    extra = [tmp_path/f'note-{i}.txt' for i in range(8)]
    for path in extra:
        path.write_text('metadata')
    metadata = tmp_path/'metadata.json'
    research = tmp_path/'research.json'
    metadata.write_text(json.dumps(dict(status='METADATA_ONLY_VERIFIED', artifact_count=12,
        numeric_capacity_values_converted=False, models_trained=False,
        artifact_hashes={str(p): r.sha(p) for p in [*archive_paths, *extra]})))
    research.write_text(json.dumps(dict(status='VERIFICATION_COMPLETE', independent_actual_audit_status='PASS',
        producer_code_hashes={}, input_hashes={}, auditor_code_hashes={}, artifact_hashes={})))
    monkeypatch.setattr(r, 'PARENTS', {metadata: r.sha(metadata), research: r.sha(research)})
    reviews = tuple(tmp_path/f'review-{i}.md' for i in range(3))
    for path in reviews:
        path.write_text('Synthetic review')
    monkeypatch.setattr(r, 'REVIEWS', reviews)
    for name in r.CODE:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    contract = r.expected_contract()
    assert set(contract['raw_archive_paths'].values()) == set(map(str, archive_paths))
    evidence = json.loads(metadata.read_text())
    evidence['artifact_hashes'].pop(str(archive_paths[-1]))
    metadata.write_text(json.dumps(evidence))
    monkeypatch.setattr(r, 'PARENTS', {metadata: r.sha(metadata), research: r.sha(research)})
    with pytest.raises(ValueError, match='four inventory aliases'):
        r.expected_contract()
