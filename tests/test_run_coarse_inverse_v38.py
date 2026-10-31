"""Information-access and timing regressions for the narrow inverse runner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import run_coarse_inverse_v38 as run


class Forbidden(dict):
    def __getitem__(self, key):
        if key.startswith('fuelshare_') and key not in (
            'fuelshare_nuclear', 'fuelshare_hydro', 'fuelshare_wind', 'fuelshare_solar'
        ):
            raise AssertionError('Hidden fuel field accessed')
        return super().__getitem__(key)


def row(start='2024-01-01', release='2024-03-01', source='BPAT'):
    at = pd.Timestamp(start, tz='UTC')
    return Forbidden(region=source, period_start=at.isoformat(),
        period_end=(at+pd.offsets.MonthBegin(1)).isoformat(), available_at=pd.Timestamp(release, tz='UTC').isoformat(),
        total_mwh='74400', fuelshare_nuclear='.1', fuelshare_hydro='.2', fuelshare_wind='.1', fuelshare_solar='.1')


def test_visible_volume_never_reads_fine_fuels():
    record = next(iter(run.monthly_index([row()]).values()))
    assert run.pos_mw(record) == pytest.approx(50.)


def test_non_source_values_never_accessed():
    class OnlyRegion(dict):
        def __getitem__(self, key):
            if key != 'region':
                raise AssertionError('Non-source access')
            return 'CLOSED'
    assert run.monthly_index([OnlyRegion()]) == {}


def test_history_uses_release_and_preserves_missing_calendar_months():
    records = []
    for at in pd.date_range('2023-01-01', periods=14, freq='MS', tz='UTC'):
        r = row(str(at.date()), str((at+pd.offsets.MonthBegin(2)).date()))
        records.append(r)
    records[-1]['available_at'] = '2026-01-01T00:00:00+00:00'
    index = run.monthly_index(records)
    selected = run.history(index, 'BPAT', pd.Timestamp('2024-04-01', tz='UTC'))
    assert len(selected) == 12
    assert selected[-1][1] == pd.Timestamp('2024-01-01', tz='UTC')
    assert selected[0][1] == pd.Timestamp('2023-02-01', tz='UTC')


def test_insufficient_legal_history_fails():
    index = run.monthly_index([row()])
    with pytest.raises(ValueError, match='twelve'):
        run.history(index, 'BPAT', pd.Timestamp('2024-02-01', tz='UTC'))


def test_duplicate_or_invalid_month_fails():
    with pytest.raises(ValueError, match='Duplicate'):
        run.monthly_index([row(), row()])
    with pytest.raises(ValueError, match='Whole month'):
        run.monthly_index([row('2024-01-02')])


@pytest.mark.parametrize('value', ['nan', '-1', '0', 'inf'])
def test_invalid_total_fails(value):
    r = row()
    r['total_mwh'] = value
    record = next(iter(run.monthly_index([r]).values()))
    with pytest.raises(ValueError, match='monthly total'):
        run.pos_mw(record)


def test_capacity_vintage_follows_each_month_not_query_year():
    table = {(2021, 'BPAT'): np.arange(8.)+1, (2022, 'BPAT'): np.arange(8.)+100}
    a = run.capacity(table, 'BPAT', pd.Timestamp('2023-12-01', tz='UTC'))
    b = run.capacity(table, 'BPAT', pd.Timestamp('2024-01-01', tz='UTC'))
    np.testing.assert_array_equal(a, np.array([1, 2, 3, 8]))
    np.testing.assert_array_equal(b, np.array([100, 101, 102, 107]))


def sealed_fixture(tmp_path, monkeypatch, corrupt=False):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    folder = tmp_path / 'predictions'
    folder.mkdir()
    run.write(tmp_path / 'freeze.json', {})
    freeze = run.sha(tmp_path / 'freeze.json')
    run.write(tmp_path / 'prediction_started.json', {'freeze_sha256': freeze})
    entries = []
    for source in run.SOURCES:
        path = folder / (source+'.npz')
        keys = np.array([f'{source}/{at:%Y-%m}' for at in run.DATES])
        if corrupt and source == run.SOURCES[-1]:
            keys[0] = 'CLOSED/2025-01'
        np.savez(path, arms=np.array(run.ARMS), keys=keys, origins=run.DATES.asi8,
            EF_draws=np.ones((6, 6, 128)), CIF_draws=np.ones((6, 6, 128)),
            ratios=np.full((6, 6, 4), .25), coarse_POS_MW=np.ones((6, 6)))
        run.write(path.with_suffix('.json'), {})
        entries.append(dict(region=source, path=str(path), sha256=run.sha(path),
            audit_path=str(path.with_suffix('.json')), audit_sha256=run.sha(path.with_suffix('.json'))))
    run.write(tmp_path / 'prediction_seal.json', dict(freeze_sha256=freeze,
        prediction_started_sha256=run.sha(tmp_path / 'prediction_started.json'), entries=entries))


def test_full_grid_validation_precedes_label_access(tmp_path, monkeypatch):
    sealed_fixture(tmp_path, monkeypatch, corrupt=True)
    monkeypatch.setattr(run, 'check', lambda: {})
    def forbidden(*args):
        raise AssertionError('Monthly labels accessed before grid validation')
    monkeypatch.setattr(run.common, 'monthly_rows', forbidden)
    with pytest.raises(ValueError, match='query grid'):
        run.score()


def test_seal_binds_start_receipt(tmp_path, monkeypatch):
    sealed_fixture(tmp_path, monkeypatch)
    assert len(run.predictions()['entries']) == 7
    (tmp_path / 'prediction_started.json').write_text('{"freeze_sha256":"wrong"}')
    with pytest.raises(ValueError, match='start receipt'):
        run.predictions()
