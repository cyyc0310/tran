"""Source exclusion and nested-fit provenance; all examples are synthetic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import run_residual_persistence_v39 as run


def audit(outer='BPAT'):
    keys = []
    for s in run.SOURCES:
        if s == outer:
            continue
        n = 24 if s == 'FPL' else 26
        keys.extend(f'{s}/{at:%Y-%m}' for at in pd.date_range('2022-05-01', periods=n, freq='MS'))
    return dict(region=outer, training_keys=keys,
        training_generation_MW=np.full((len(keys), 4), 10.),
        training_capacity_MW=np.full((len(keys), 4), 100.),
        training_months=np.array([int(k[-2:]) for k in keys]))


def test_outer_guard_precedes_numbers():
    a = audit()
    a['training_keys'][0] = 'BPAT/2022-05'
    class NumbersForbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Numbers converted before source guard')
    a['training_generation_MW'] = NumbersForbidden()
    with pytest.raises(ValueError, match='Outer-excluded'):
        run.source_data(a, 'BPAT')


def test_future_dates_rejected_before_numbers():
    a = audit()
    a['training_keys'][0] = 'CISO/2025-01'
    with pytest.raises(ValueError, match='supervision dates'):
        run.source_data(a, 'BPAT')


def test_unconsumed_future_fields_are_not_accessed():
    class Guard(dict):
        def __getitem__(self, key):
            if key in {'contexts', 'solves', 'CIF_draws'}:
                raise AssertionError('Future field accessed')
            return super().__getitem__(key)
    d = run.source_data(Guard(audit()), 'BPAT')
    assert len(d['keys']) == 154


def test_inner_fit_excludes_both_regions_and_rebalances(monkeypatch):
    data = run.source_data(audit(), 'BPAT')
    calls = []
    original = run.profile.fit_profiles
    def record(g, c, months, weights, keys):
        sources = np.array([k.split('/')[0] for k in keys])
        assert 'BPAT' not in sources
        assert len(set(sources)) == 5
        totals = [weights[sources == s].sum() for s in set(sources)]
        np.testing.assert_allclose(totals, len(keys)/5, atol=1e-12, rtol=0)
        calls.append(set(keys))
        return original(g, c, months, weights, keys)
    monkeypatch.setattr(run.profile, 'fit_profiles', record)
    records, fits = run.nested(data, 'BPAT')
    assert len(records) == 154 and len(fits) == len(calls) == 6
    for fit, called in zip(fits, calls):
        assert set(fit['training_keys']) == called
        assert not (called & set(fit['held_keys']))
        assert all(not k.startswith(fit['held_source']+'/') for k in called)
    np.testing.assert_allclose([r['EF_error_true_minus_pred'] for r in records], 0., atol=1e-9, rtol=0)


@pytest.mark.parametrize('field', ['training_generation_MW', 'training_capacity_MW'])
def test_invalid_numbers_fail(field):
    a = audit()
    a[field][0, 0] = np.nan
    with pytest.raises(ValueError, match='Finite aligned'):
        run.source_data(a, 'BPAT')


def test_exact_fpl_outer_count():
    assert len(run.source_data(audit('FPL'), 'FPL')['keys']) == 156


def test_month_mismatch_fails():
    a = audit()
    a['training_months'][0] = 12
    with pytest.raises(ValueError, match='aligned'):
        run.source_data(a, 'BPAT')
