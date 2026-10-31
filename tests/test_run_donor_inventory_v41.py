"""Exclusive attempts, full-year completion and legacy subset byte safeguards."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_donor_inventory_v41 as run


def test_existing_start_receipt_blocks_attempt_before_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    (tmp_path / 'curation_started.json').write_text('{}')
    def forbidden():raise AssertionError('Inputs consumed during prohibited repeat')
    monkeypatch.setattr(run, 'check', forbidden)
    with pytest.raises(FileExistsError, match='attempt'):
        run.curate()


def test_no_numeric_tables_if_fourth_year_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    run.write(tmp_path / 'freeze.json', {})
    monkeypatch.setattr(run, 'check', lambda: {'raw_archive_paths': {str(y): 'synthetic' for y in run.legacy.SUPPORTED_YEARS}})
    class Package:
        def __init__(self, *args):pass
        def __enter__(self):return self
        def __exit__(self, *args):return False
        def read(self, name):return b'synthetic workbook'
    monkeypatch.setattr(run.zipfile, 'ZipFile', Package)
    monkeypatch.setattr(run.legacy, 'read_workbook_rows', lambda *args: [])
    seen = []
    def curate(year, plants, generators):
        assert (tmp_path / 'curation_started.json').exists()
        seen.append(year)
        assert all(not (tmp_path / p).exists() for p in run.OUTPUTS)
        if year == 2023:raise ValueError('fourth year failed')
        rows = [dict(year=year, region=s) for s in run.model.ALL_SOURCES]
        return dict(op=rows, operable=rows, units=[], excluded=[], audit={})
    monkeypatch.setattr(run.model, 'curate_year', curate)
    with pytest.raises(ValueError, match='fourth year'):
        run.curate()
    assert seen == [2020, 2021, 2022, 2023]
    assert {p.name for p in tmp_path.iterdir()} == {'freeze.json', 'curation_started.json'}


def test_legacy_mismatch_is_hard_failure(tmp_path):
    combined = {kind: [] for kind in run.CSV_SCHEMAS}
    paths = {}
    for kind, (name, columns) in run.CSV_SCHEMAS.items():
        p = tmp_path / name
        p.write_bytes(run.csv_bytes([], columns))
        paths[kind] = str(p)
    match = run.legacy_match(combined, paths)
    assert all(item['bytes_exact'] for item in match.values())
    Path(paths['op']).write_bytes(b'changed bytes\n')
    with pytest.raises(ValueError, match='Legacy subset changed'):
        run.legacy_match(combined, paths)


def test_schema_error_does_not_silently_drop_columns():
    with pytest.raises(ValueError, match='Exact CSV schema'):
        run.csv_bytes([{'year': 2020, 'undocumented': 1}], ('year',))


def test_csv_preserves_identifiers_and_legacy_line_endings():
    assert run.csv_bytes([dict(plant='1', generator='001', capacity=1.5, in_op=True)],
        ('plant', 'generator', 'capacity', 'in_op')) == b'plant,generator,capacity,in_op\r\n1,001,1.5,True\r\n'
