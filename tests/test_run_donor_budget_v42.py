"""Cohort scope, response access and sealed scoring on synthetic records only."""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from scripts import run_donor_budget_v42 as run


class Guard(dict):
    def __getitem__(self, key):
        if key in {'fuelshare_coal', 'fuelshare_gas', 'fuelshare_oil', 'fuelshare_other'}:
            raise AssertionError('Excluded fine label accessed')
        return super().__getitem__(key)


def rows():
    result = []
    for source in run.model.ALL_SOURCES:
        for at in pd.date_range('2022-01-01', periods=9, freq='MS', tz='UTC'):
            end = at+pd.offsets.MonthBegin(1)
            d = dict(region=source, period_start=at.isoformat(), period_end=end.isoformat(),
                available_at=(end+pd.Timedelta(days=30)).isoformat(), total_mwh='74400', coverage='.5',
                fuelshare_coal='.2', fuelshare_gas='.4', fuelshare_oil='0', fuelshare_other='0',
                fuelshare_nuclear='.1', fuelshare_hydro='.1', fuelshare_wind='.1', fuelshare_solar='.1')
            result.append(Guard(d) if source == 'BPAT' else d)
    return result


def test_excluded_outer_fine_is_not_read_and_scope_conflicts_are_diagnostic():
    raw = rows(); cases = run.model.build_cases(raw)
    caps = {(y, s): np.array([0., 10., 0., 0.]) for y in run.inventory.SUPPORTED_YEARS for s in run.model.ALL_SOURCES}
    data = run.training(raw, cases, caps, 'BPAT')
    assert 'BPAT' not in data['regions']
    assert len(set(data['regions'])) == 16
    assert all(d['coverage'] == .5 for d in data['diagnostics'])
    assert all(d['zero_capacity_positive_generation_coal'] for d in data['diagnostics'])
    assert np.all(data['y'][:, 0] > 0)  # No deletion or false rate0 despite C_coal=0.


def test_future_donor_fine_not_accessed():
    raw = rows()
    d = dict(raw[0]);d.update(region='SWPP', period_start='2025-01-01T00:00:00+00:00',
        period_end='2025-02-01T00:00:00+00:00', available_at='2025-03-03T00:00:00+00:00')
    raw.append(Guard(d))
    cases = run.model.build_cases(raw)
    caps = {(y, s): np.ones(4) for y in run.inventory.SUPPORTED_YEARS for s in run.model.ALL_SOURCES}
    assert not any(k.endswith('/2025-01') for k in run.training(raw, cases, caps, 'BPAT')['keys'])


def test_monthly_file_ownership_is_fixed(tmp_path):
    paths = {}
    for group in ('legacy', 'v4', 'v5'):
        p = tmp_path / (group+'.csv');paths[group] = str(p)
        with p.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=('region', 'period_start'))
            writer.writeheader()
            writer.writerows(dict(region=s, period_start='not consumed here') for s in ('BPAT', 'SWPP', 'SCL', 'PACE'))
    assert [r['region'] for r in run.monthly_rows(paths)] == ['BPAT', 'SWPP', 'SCL']


def test_zero_POS_inventory_fails_without_dropping_source(tmp_path):
    p = tmp_path / 'capacity.csv'
    fields = ['year', 'region', *['capacity_'+f+'_mw' for f in run.v30.FUELS], 'capacity_pos_mw']
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields);w.writeheader()
        for y in run.inventory.SUPPORTED_YEARS:
            for s in run.model.ALL_SOURCES:
                d = {k: 0 for k in fields};d.update(year=y, region=s)
                d['capacity_gas_mw'] = d['capacity_pos_mw'] = 1
                if y == 2020 and s == 'SWPP':d['capacity_gas_mw'] = d['capacity_pos_mw'] = 0
                w.writerow(d)
    with pytest.raises(ValueError, match='no source filtering'):
        run.capacity_index(str(p))


def sealed(tmp_path, monkeypatch, bad=False):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    folder = tmp_path / 'predictions';folder.mkdir()
    run.write(tmp_path / 'freeze.json', {})
    freeze = run.sha(tmp_path / 'freeze.json')
    run.write(tmp_path / 'prediction_started.json', dict(freeze_sha256=freeze))
    run.write(tmp_path / 'cohort.json', {})
    (tmp_path / 'training_scope_diagnostics.csv').write_text('outer,key\n')
    entries = []
    for source in run.SOURCES:
        p = folder / (source+'.npz')
        keys = np.array([f'{source}/{d:%Y-%m}' for d in run.DATES])
        if bad and source == run.SOURCES[-1]:keys[0] = 'CLOSED/2025-01'
        mg = np.zeros((6, 128, 5));mg[..., 0] = mg[..., 1] = .5
        np.savez(p, arms=np.array(run.ARMS), keys=keys, origins=run.DATES.asi8,
            EF_draws=np.full((12, 6, 128), 500.), CIF_draws=np.full((12, 6, 128), 250.), MG_draws=mg,
            expanded_raw_EF=np.full((6, 6, 128), 500.), expanded_projection_l1=np.zeros((6, 6, 128)))
        run.write(p.with_suffix('.json'), {})
        entries.append(dict(region=source, path=str(p), sha256=run.sha(p), audit_path=str(p.with_suffix('.json')),
            audit_sha256=run.sha(p.with_suffix('.json'))))
    run.write(tmp_path / 'prediction_seal.json', dict(freeze_sha256=freeze,
        prediction_started_sha256=run.sha(tmp_path / 'prediction_started.json'), cohort_sha256=run.sha(tmp_path / 'cohort.json'),
        training_scope_diagnostics_sha256=run.sha(tmp_path / 'training_scope_diagnostics.csv'), entries=entries))


def test_complete_prediction_validation_precedes_scoring_labels(tmp_path, monkeypatch):
    sealed(tmp_path, monkeypatch, bad=True)
    monkeypatch.setattr(run, 'check', lambda: {})
    def forbidden(*args):raise AssertionError('Scoring labels accessed before grid validation')
    monkeypatch.setattr(run.common, 'monthly_rows', forbidden)
    with pytest.raises(ValueError, match='forward query grid'):
        run.score()


def test_cohort_is_bound_to_prediction_seal(tmp_path, monkeypatch):
    sealed(tmp_path, monkeypatch)
    assert len(run.predictions()['entries']) == 7
    (tmp_path / 'cohort.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='prediction/cohort seal'):
        run.predictions()
