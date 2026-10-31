"""Synthetic checks of date support, scoped truth, score copying and region weighting."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_hourly_bridge_v46 as run


def metadata():
    return pd.DataFrame(dict(region=s, origin=at.isoformat(), arm=a, factor=f)
        for s in run.SOURCES for at in run.DATES
        if at.month != 1 and (s != 'ISNE' or at.month in (2, 5))
        for a in run.V29_ARMS for f in run.FACTORS)


def test_support_uses_only_availability_and_locks_missing_days():
    table = metadata()
    keys = run.support_metadata(table)
    assert len(keys) == 32 and 'ISNE/2025-03-01' not in keys
    with pytest.raises(ValueError):
        run.support_metadata(table.iloc[1:])
    with pytest.raises(ValueError):
        run.support_metadata(pd.concat([table, table.iloc[:1]]))
    with pytest.raises(ValueError):
        run.support_metadata(table.assign(crps=0.))
    extra = table.iloc[:50].copy()
    extra['origin'] = run.DATES[0].isoformat()
    with pytest.raises(ValueError):
        run.support_metadata(pd.concat([table, extra]))


def test_date_particle_coupling_is_exact():
    assert np.array_equal(run.indices(run.DAYS.asi8, True), [0, 31, 59, 90, 120, 151])
    assert np.array_equal(run.indices(run.DATES.asi8, False), np.arange(6))
    bad = run.DAYS.asi8.copy()
    bad[31] += 1
    with pytest.raises(ValueError):
        run.indices(bad, True)


def test_truth_loader_calls_only_locked_utc_days(monkeypatch):
    calls = []
    g = np.zeros((24, 8))
    g[:, 0] = 1.

    def source_days(path, start, stop):
        calls.append((path, start, stop))
        return [(pd.Timestamp(start, tz='UTC'), g)]

    monkeypatch.setattr(run, 'source_days', source_days)
    positions, truth = run.scoped_truth({'truth': '/fake/ISNE_hourly.csv'}, 'ISNE',
                                      ['ISNE/2025-02-01', 'ISNE/2025-05-01'])
    assert positions == [1, 4]
    assert calls == [(Path('/fake/ISNE_hourly.csv'), '2025-02-01', '2025-02-02'),
                     (Path('/fake/ISNE_hourly.csv'), '2025-05-01', '2025-05-02')]
    assert np.array_equal(truth, np.full((2, 24), 1000.))
    monkeypatch.setattr(run, 'source_days', lambda *_: [])
    with pytest.raises(ValueError):
        run.scoped_truth({'truth': '/fake/ISNE_hourly.csv'}, 'ISNE', ['ISNE/2025-02-01'])


def score_table():
    pairs = metadata()[['region', 'origin']].drop_duplicates().to_dict('records')
    return pd.DataFrame(dict(**p, arm=a, factor=f, budget='B0',
                            **{k: .123456789012345 for k in run.METRICS})
                        for p in pairs for a in run.model.ARMS[:2] for f in run.FACTORS)


def test_baseline_reconstruction_then_literal_numeric_copy():
    original = score_table()
    computed = original.copy()
    computed.loc[0, 'crps'] += 1e-11
    result, error = run.restore_baselines(computed, original)
    assert 0 < error < 1e-9
    assert np.array_equal(result[list(run.COPY_METRICS)], original[list(run.COPY_METRICS)])
    computed.loc[0, 'crps'] += .1
    with pytest.raises(ValueError):
        run.restore_baselines(computed, original)
    with pytest.raises(ValueError):
        run.restore_baselines(original.iloc[:-1], original)
    with pytest.raises(ValueError):
        run.restore_baselines(original, pd.concat([original, original.iloc[:1]]))


def test_original_csv_preserves_17_digit_double(tmp_path):
    table = score_table()
    table.loc[0, 'crps'] = float('0.12345678901234567')
    path = tmp_path / 'old.csv'
    table.to_csv(path, index=False)
    loaded = run.original_scores(str(path))
    restored, _ = run.restore_baselines(table, loaded)
    assert restored.loc[0, 'crps'].hex() == table.loc[0, 'crps'].hex()


def test_region_weighting_not_day_weighting():
    table = score_table()
    for i, source in enumerate(run.SOURCES):
        table.loc[table.region.eq(source), list(run.METRICS)] = i / 10
    regions, macro, leave = run.aggregate(table)
    assert np.allclose(macro.crps, .3)
    assert not np.isclose(table.crps.mean(), .3)
    assert len(regions) == 70 and len(macro) == 10 and len(leave) == 70
    assert np.allclose(leave[leave.excluded_region.eq('BPAT')].crps, .35)
    assert macro.worst_region_coverage90.eq(0).all()
    assert not macro.nominal_and_efficient_vs_anchor1.any()


def test_score_cannot_open_truth_before_prediction_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'check', lambda: {})
    def incomplete():
        raise ValueError('unsealed')
    monkeypatch.setattr(run, 'predictions', incomplete)
    monkeypatch.setattr(run, 'scoped_truth', lambda *_: pytest.fail('truth opened'))
    with pytest.raises(ValueError, match='unsealed'):
        run.score()
    assert not (tmp_path / 'score_started.json').exists()


def test_predict_cannot_restart_existing_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'check', lambda: {})
    (tmp_path / 'prediction_started.json').write_text('{}')
    monkeypatch.setattr(run, 'load_for_prediction', lambda *_: pytest.fail('loaded arrays'))
    with pytest.raises(FileExistsError):
        run.predict()
