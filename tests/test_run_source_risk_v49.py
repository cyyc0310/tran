"""Metadata-only source support and task-specific held-label protection."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts import run_source_risk_v49 as run


def test_availability_never_converts_generation(tmp_path, monkeypatch):
    path = tmp_path / 'BPAT_hourly.csv'
    with path.open('w', newline='') as stream:
        w = csv.writer(stream); w.writerow(['time', 'region', 'valid', 'generation_coal'])
        for day in ('2024-01-25', '2024-07-01'):
            for at in pd.date_range(day, periods=24, freq='h', tz='UTC'):
                w.writerow([at.isoformat(), 'BPAT', 'True', 'forbidden-not-a-number'])
    monkeypatch.setattr(run, 'COUNTS', {'BPAT': 2})
    monkeypatch.setattr(run, 'H1_COUNTS', {'BPAT': 1})
    assert run.availability(str(path), 'BPAT') == ['BPAT/2024-01-25', 'BPAT/2024-07-01']


def test_visible_months_only_access_legal_coarse_fields():
    class Row(dict):
        def __getitem__(self, key):
            if key in {'fuelshare_coal', 'fuelshare_gas', 'fuelshare_oil', 'fuelshare_other'}:
                pytest.fail('local hidden fine accessed')
            return super().__getitem__(key)
    row = Row(region='BPAT', period_start='2023-11-01T00:00:00Z', period_end='2023-12-01T00:00:00Z',
        available_at='2024-01-01T12:00:00Z', fuelshare_nuclear='.1', fuelshare_hydro='.2',
        fuelshare_wind='.1', fuelshare_solar='.1')
    pending = Row(row); pending['available_at'] = '2025-01-01T00:00:00Z'; pending['fuelshare_hydro'] = object()
    visible = run.visible_months([row, pending], 'BPAT')
    assert len(visible) == 1 and visible.fuelshare_POS.iloc[0] == .5


def test_audit_tokens_keep_fine_text_and_parameter_semantics(tmp_path):
    paths = {}
    for key, arm in zip(('size', 'cap'), run.centers.HEADS):
        method = arm.split('-')[1]
        fit = dict(arm=arm, parameters=run.centers.forward.PARAMETERS[method], training_X=[[.12345678901234567]],
                   training_y=[[.25, .25, .25, .25]], **{name: [1234567890123456789] for name in run.centers.DATE_FIELDS})
        path = tmp_path / (key+'.json'); path.write_text(json.dumps(dict(region='BPAT', fits=[fit])))
        paths[key] = str(path)
    fits = run.center_records(paths, ('BPAT', 'CISO'))
    assert isinstance(fits[0]['training_y'][0][0], str)
    assert type(fits[0]['parameters']['max_features']) is float
    assert type(fits[0]['training_period_start_ns'][0]) is int
    payload = json.loads(Path(paths['size']).read_text()); payload['fits'][0]['parameters']['max_features'] = 1
    Path(paths['size']).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='token type'):
        run.center_records(paths, ('BPAT', 'CISO'))


def test_outer_index_rejects_reverse_label_leakage():
    entries = [dict(outer=o, held=h, label_region=h, pair=sorted((o,h)))
               for o in run.SOURCES for h in run.SOURCES if o != h]
    selected = run.tasks_for_outer(entries, 'BPAT')
    assert len(selected) == 6 and all(e['label_region'] != 'BPAT' for e in selected)
    bad = [dict(e) for e in entries]; bad[0]['label_region'] = bad[0]['outer']
    with pytest.raises(ValueError, match='held-only'):
        run.tasks_for_outer(bad, 'BPAT')


def test_bank_pair_metadata_before_fine_fields(monkeypatch):
    class Archive(dict):
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def __getitem__(self, key):
            if key in {'q', 'donor_mix', 'raw_cif', 'monthmean'}: pytest.fail('fine read before bank exclusion')
            return super().__getitem__(key)
    wrong = Archive(excluded=np.array(['BPAT', 'ERCO']), source_regions=np.array([]), regions=np.array([]))
    monkeypatch.setattr(run.np, 'load', lambda *_a, **_kw: wrong)
    with pytest.raises(ValueError, match='provenance'):
        run.bank('/fake/bank.npz', ('BPAT', 'CISO'))


def test_seal_before_risk_label_interface(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUT', tmp_path); monkeypatch.setattr(run, 'check', lambda: {})
    def unsealed(): raise ValueError('unsealed')
    monkeypatch.setattr(run, 'predictions', unsealed)
    monkeypatch.setattr(run.common, 'source_days', lambda *_: pytest.fail('label response read'))
    with pytest.raises(ValueError, match='unsealed'): run.label()
    assert not (tmp_path / 'label_started.json').exists()


def test_preserve_prediction_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUT', tmp_path); monkeypatch.setattr(run, 'check', lambda: {})
    (tmp_path / 'prediction_started.json').write_text('{}')
    with pytest.raises(FileExistsError): run.predict()
