"""Metadata, coupling and read-order checks; no real research values."""
import numpy as np
import pandas as pd
import pytest
from scripts import run_recompose_bridge_v48 as run


def support():
    counts = (176, 178, 180, 179, 89, 166, 177)
    table = pd.DataFrame(dict(region=s, origin=at.isoformat(), available=j < n)
        for s, n in zip(run.SOURCES, counts) for j, at in enumerate(run.DAYS))
    keys = [f'{row.region}/{pd.Timestamp(row.origin):%Y-%m-%d}'
            for row in table.itertuples() if row.available]
    return table, keys


def test_complete_support_and_strict_bool():
    table, keys = support()
    run.support_metadata(table, keys)
    with pytest.raises(ValueError):
        run.support_metadata(table.iloc[:-1], keys)
    with pytest.raises(ValueError):
        run.support_metadata(table, keys[:-1])
    with pytest.raises(ValueError):
        run.support_metadata(table.assign(available=table.available.astype(str)), keys)
    changed = table.copy(); changed.loc[0, 'origin'] = changed.loc[1, 'origin']
    with pytest.raises(ValueError):
        run.support_metadata(changed, keys)


def test_coupling_check_precedes_hourly_raw_access(monkeypatch):
    class Archive(dict):
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def __getitem__(self, key):
            if key in ('raw', 'fine_draws', 'query_mean'):
                pytest.fail('raw values read before MG alignment')
            return super().__getitem__(key)
    mg = np.full((181, 128, 5), .2)
    first = Archive(origins=run.DAYS.asi8, arms=np.array(run.model.prior.ARMS),
        level_arms=np.array(run.model.LEVEL_ARMS), keys=np.array([f'BPAT/{at:%Y-%m-%d}' for at in run.DAYS]),
        MG_draws=mg, level_draws=np.zeros((15, 181, 128)))
    second = Archive(origins=run.DAYS.asi8, arms=np.array(run.common.V29_ARMS), MG_draws=mg+.01)
    archives = iter((first, second))
    monkeypatch.setattr(run.np, 'load', lambda *_args, **_kwargs: next(archives))
    with pytest.raises(ValueError, match='MG coupling'):
        run.query({'levels': 'fake1', 'hourly': 'fake2'}, 'BPAT')


def test_truth_matches_frozen_cohort(monkeypatch):
    g = np.ones((24, 8))
    calls = []
    def days(path, start, stop):
        calls.append((str(path), start, stop))
        return [(run.DAYS[0], g)]
    monkeypatch.setattr(run.common, 'source_days', days)
    positions, y = run.truth({'truth': '/fake/BPAT_hourly.csv'}, 'BPAT', ['BPAT/2025-01-01'])
    assert positions.tolist() == [0] and y.shape == (1, 24)
    assert calls == [('/fake/BPAT_hourly.csv', '2025-01-01', '2025-07-01')]
    with pytest.raises(ValueError, match='support mismatch'):
        run.truth({'truth': '/fake/BPAT_hourly.csv'}, 'BPAT', ['BPAT/2025-01-02'])


def test_seal_validation_precedes_truth(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'check', lambda: {})
    def unsealed(): raise ValueError('unsealed')
    monkeypatch.setattr(run, 'predictions', unsealed)
    monkeypatch.setattr(run, 'truth', lambda *_: pytest.fail('truth read'))
    with pytest.raises(ValueError, match='unsealed'):
        run.score()
    assert not (tmp_path / 'score_started.json').exists()


def test_preserve_partial_prediction_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'check', lambda: {})
    (tmp_path / 'prediction_started.json').write_text('{}')
    monkeypatch.setattr(run, 'query', lambda *_: pytest.fail('cached values read'))
    with pytest.raises(FileExistsError):
        run.predict()
