"""Synthetic support, replay and read-order checks; no real research arrays."""
import numpy as np
import pandas as pd
import pytest

from scripts import run_daily_bridge_v47 as run


def metadata():
    return pd.DataFrame(dict(region=s, origin=at.isoformat(), arm=a, factor=f)
        for s in run.SOURCES for at in run.DAYS[:run.COUNTS[s]]
        for a in run.prior.V29_ARMS for f in run.FACTORS)


def test_metadata_locks_all_1145_daily_pairs():
    t = metadata()
    keys = run.support_metadata(t)
    assert len(keys) == 1145 and keys[0] == 'BPAT/2025-01-01'
    with pytest.raises(ValueError):
        run.support_metadata(t.iloc[1:])
    with pytest.raises(ValueError):
        run.support_metadata(t.assign(crps=0.))
    with pytest.raises(ValueError):
        run.support_metadata(pd.concat([t, t.iloc[:1]]))
    t.loc[0, 'origin'] = '2025-07-01T00:00:00+00:00'
    with pytest.raises(ValueError):
        run.support_metadata(t)


def test_inventory_selects_year_and_region_before_fuel_values(tmp_path):
    p = tmp_path / 'capacity.csv'
    cols = ['year', 'region'] + ['capacity_' + f + '_mw' for f in run.model.FUELS]
    rows = [['2023', 'BPAT', *([1.] * 8)], ['2022', 'BPAT', *(['not numeric'] * 8)],
            ['2023', 'CISO', *(['not numeric'] * 8)]]
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    q, c = run.capacity_for(str(p), 'BPAT')
    assert np.array_equal(q, np.full(4, .25)) and c == 4.
    with pytest.raises(ValueError):
        run.capacity_for(str(p), 'NYIS')


def test_truth_support_is_exact_and_scoped(monkeypatch):
    g = np.zeros((24, 8)); g[:, 1] = 1.
    calls = []
    def days(path, start, stop):
        calls.append((str(path), start, stop))
        return [(run.DAYS[0], g), (run.DAYS[2], g)]
    monkeypatch.setattr(run.prior, 'source_days', days)
    entry = {'truth': '/fake/BPAT_hourly.csv'}
    keys = ['BPAT/2025-01-01', 'BPAT/2025-01-03']
    pos, truth = run.scoped_truth(entry, 'BPAT', keys)
    assert calls == [('/fake/BPAT_hourly.csv', '2025-01-01', '2025-07-01')]
    assert np.array_equal(pos, [0, 2]) and np.array_equal(truth, np.full((2, 24), 400.))
    with pytest.raises(ValueError):
        run.scoped_truth(entry, 'BPAT', keys[:-1])


def baseline_table():
    pairs = metadata()[['region', 'origin']].drop_duplicates().to_dict('records')
    return pd.DataFrame(dict(**p, arm=a, factor=f, budget='B0',
                            **{k: float('0.12345678901234567') for k in run.METRICS})
        for p in pairs for a in run.model.ARMS[:2] for f in run.FACTORS)


def test_exact_11450_numeric_copy_and_mismatch_rejection(tmp_path):
    old = baseline_table()
    p = tmp_path / 'old.csv'; old.to_csv(p, index=False)
    parsed = run.prior.original_scores(str(p))
    computed = old.copy(); computed.loc[0, 'crps'] += 1e-11
    result, error = run.restore_baselines(computed, parsed)
    assert 0 < error < 1e-9
    assert np.array_equal(result[list(run.COPY_METRICS)], old[list(run.COPY_METRICS)])
    computed.loc[0, 'crps'] += .01
    with pytest.raises(ValueError):
        run.restore_baselines(computed, parsed)
    with pytest.raises(ValueError):
        run.restore_baselines(old.iloc[:-1], parsed)


def test_prediction_validation_precedes_any_truth(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'check', lambda: {})
    def incomplete():
        raise ValueError('unsealed')
    monkeypatch.setattr(run, 'predictions', incomplete)
    monkeypatch.setattr(run, 'scoped_truth', lambda *_: pytest.fail('truth read'))
    with pytest.raises(ValueError, match='unsealed'):
        run.score()
    assert not (tmp_path / 'score_started.json').exists()


def test_prediction_attempt_cannot_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'check', lambda: {})
    (tmp_path / 'prediction_started.json').write_text('{}')
    monkeypatch.setattr(run, 'hourly_inputs', lambda *_: pytest.fail('loaded cached arrays'))
    with pytest.raises(FileExistsError):
        run.predict()
