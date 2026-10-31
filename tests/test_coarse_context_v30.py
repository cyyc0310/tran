"""Synthetic-only V30 information-boundary and ordinary-model checks."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import coarse_context_v30 as m


def rows(months=9):
    result = []
    for r, region in enumerate(m.SOURCES):
        for j, start in enumerate(pd.date_range('2022-01-01', periods=months, freq='MS', tz='UTC')):
            end = start+pd.offsets.MonthBegin(1)
            zero = np.array([.08+.005*j, .09+.002*r, .10+.001*j*r, .04])
            ratios = np.array([.2+.01*r, .5-.01*r, .1+.002*j, .2-.002*j])
            mix = np.empty(8); mix[m.ZERO] = zero; mix[m.POS] = (1-zero.sum())*ratios
            result.append(dict(region=region, period_start=start.isoformat(), period_end=end.isoformat(),
                available_at=end.isoformat(), total_mwh=str((700+20*r+8*j)*((end-start)/pd.Timedelta(hours=1))),
                **dict(zip(m.SHARES, map(str, mix)))))
    return result


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            same(left, right)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def test_visible_cases_never_convert_hidden_fields_and_inputs_immutable():
    raw = rows(); original = deepcopy(raw)
    cases = m.build_cases(raw)
    same(raw, original)
    poison = deepcopy(raw)
    for row in poison:
        row.update({m.SHARES[i]: 'HIDDEN_POISON' for i in m.POS})
    same(cases, m.build_cases(poison))
    assert len(cases['keys']) == 7*6
    for family, size in zip(m.FAMILIES, (5, 7, 28, 30, 31)):
        assert cases['X_by_family'][family].shape == (42, size)
        np.testing.assert_array_equal(cases['X_by_family'][family][:, :5], cases['current_g'])


def test_terminal_current_total_and_future_unreleased_numeric_gate():
    raw = rows(4)
    for row in raw:
        if row['period_start'].startswith('2022-04'):
            row['total_mwh'] = 'CURRENT_POISON'
    before = m.build_cases(raw)
    extra = deepcopy(raw[0]); extra.update(period_start='2026-01-01T00:00:00+00:00',
        period_end='2026-02-01T00:00:00+00:00', available_at='2026-02-01T00:00:00+00:00', total_mwh='FUTURE_POISON')
    extra.update({key: 'FUTURE_POISON' for key in m.SHARES})
    raw.append(extra)
    unreleased = deepcopy(extra); unreleased.update(period_start='2023-01-01T00:00:00+00:00',
        period_end='2023-02-01T00:00:00+00:00', available_at='2026-01-01T00:00:00+00:00')
    raw.append(unreleased)
    same(before, m.build_cases(raw))


def test_inclusive_release_latest_three_gap_ages_and_calendar_volume():
    raw = [row for row in rows(6) if not row['period_start'].startswith('2022-03')]
    cases = m.build_cases(raw)
    i = cases['keys'].index('BPAT/2022-05')
    assert cases['context_keys'][i].tolist() == ['BPAT/2022-04', 'BPAT/2022-02', 'BPAT/2022-01']
    np.testing.assert_array_equal(cases['context_month_age'][i], [1, 3, 4])
    assert cases['context_release_age_days'][i, 0] == 0
    np.testing.assert_allclose(cases['context_MW'][i], [724, 708, 700])
    np.testing.assert_allclose(cases['X_by_family']['Trend'][i, -2:], np.log([724/708, 708/700]))
    delayed = deepcopy(raw)
    for row in delayed:
        if row['region'] == 'BPAT' and row['period_start'].startswith('2022-04'):
            row['available_at'] = '2022-05-01T00:00:00.000000001+00:00'
    assert 'BPAT/2022-05' not in m.build_cases(delayed)['keys']


def test_zero_pos_missing_history_and_invalid_eligible_context():
    raw = rows(4)
    for row in raw:
        if row['region'] == 'BPAT' and row['period_start'].startswith('2022-04'):
            for i in m.ZERO:
                row[m.SHARES[i]] = '.25'
    cases = m.build_cases(raw)
    assert 'BPAT/2022-04' not in cases['keys']
    reasons = {entry['reason'] for entry in cases['audit']['skips']}
    assert reasons == {'zero_POS', 'missing_three_legal_contexts'}
    raw = rows(4); raw[0]['total_mwh'] = '0'
    with pytest.raises(ValueError, match='positive'):
        m.build_cases(raw)


def test_zero_pos_complement_rounding_tolerance():
    raw = rows(4)
    terminal = next(row for row in raw if row['region'] == 'BPAT' and
                    row['period_start'].startswith('2022-04'))
    for index in m.ZERO:
        terminal[m.SHARES[index]] = str(.25)
    terminal[m.SHARES[m.ZERO[0]]] = str(.25+2e-16)
    assert sum(float(terminal[m.SHARES[index]]) for index in m.ZERO) > 1.
    cases = m.build_cases(raw)
    assert dict(key='BPAT/2022-04', reason='zero_POS') in cases['audit']['skips']
    terminal[m.SHARES[m.ZERO[0]]] = str(.25+2e-12)
    with pytest.raises(ValueError, match='simplex'):
        m.build_cases(raw)


def test_fit_requires_all_six_training_regions():
    raw = [row for row in rows() if row['region'] != 'NYIS']
    with pytest.raises(ValueError, match='All six'):
        m.fit_predict(raw, m.build_cases(raw), outer='MISO')


@pytest.mark.parametrize('field,value', [('period_end', '2022-01-30T00:00:00Z'),
    ('available_at', '2021-01-01T00:00:00Z'), ('period_start', '2022-01-02T00:00:00Z')])
def test_invalid_source_metadata_fails(field, value):
    raw = rows(); raw[0][field] = value
    with pytest.raises(ValueError):
        m.build_cases(raw)


def test_duplicate_month_fails_and_unknown_rows_are_not_converted():
    raw = rows(); raw.append(deepcopy(raw[0]))
    with pytest.raises(ValueError, match='Duplicate'):
        m.build_cases(raw)
    raw = rows(); expected = m.build_cases(raw)
    raw.append(dict(region='UNKNOWN', period_start='METADATA_POISON'))
    same(expected, m.build_cases(raw))


def test_outer_hidden_poison_predictions_and_fit_provenance():
    raw = rows(); cases = m.build_cases(raw); before = deepcopy(cases); original = deepcopy(raw)
    result = m.fit_predict(raw, cases, outer='MISO')
    same(raw, original); same(cases, before)
    poison = deepcopy(raw)
    for row in poison:
        if row['region'] == 'MISO':
            row.update({m.SHARES[i]: 'OUTER_HIDDEN_POISON' for i in m.POS})
    same(result, m.fit_predict(poison, cases, outer='MISO'))
    assert result['ratios'].shape == (10, 6, 4)
    assert len(result['fits']) == 10
    for fit in result['fits']:
        assert not any(key.startswith('MISO/') for key in fit['training_keys'])
        assert fit['query_keys'] == result['query_keys']
        assert fit['training_X'].shape[0] == 36
        assert fit['training_y'].shape == (36, 4)
        assert np.all(fit['training_available_ns'] <= m.CUTOFF.value)
    np.testing.assert_allclose(result['ratios'].sum(-1), 1)
    np.testing.assert_allclose(result['EF_point'], result['ratios']@m.EF[m.POS])
    np.testing.assert_allclose(result['projection_l1'], np.abs(result['ratios']-result['ratios_raw']).sum(-1))


def test_scaler_and_region_weights_only_use_training_cohort():
    raw = [row for row in rows() if not (row['region'] == 'BPAT' and row['period_start'].startswith('2022-09'))]
    cases = m.build_cases(raw)
    altered = deepcopy(cases)
    for family in m.FAMILIES:
        altered['X_by_family'][family][cases['regions'] == 'MISO'] += 1000.
    base = m.fit_predict(raw, cases, outer='MISO')
    result = m.fit_predict(raw, altered, outer='MISO')
    for original, fit in zip(base['fits'], result['fits']):
        np.testing.assert_array_equal(fit['scaler_mean'], original['scaler_mean'])
        np.testing.assert_array_equal(fit['scaler_scale'], original['scaler_scale'])
        w, regions = fit['sample_weights'], fit['training_regions']
        totals = [w[regions == region].sum() for region in set(regions)]
        np.testing.assert_allclose(totals, np.full(6, len(w)/6))
        if fit['model'] == 'Ridge':
            mean = np.average(fit['training_X'], axis=0, weights=w)
            scale = np.sqrt(np.average((fit['training_X']-mean)**2, axis=0, weights=w))
            scale[scale <= 1e-12] = 1.
            np.testing.assert_allclose(fit['scaler_mean'], mean)
            np.testing.assert_allclose(fit['scaler_scale'], scale)


def test_manual_projection_and_response():
    raw = np.array([[-1., .2, .3, .5], [2., 0., 0., 0.], [.1, .2, .3, .4]])
    expected = np.array([[0., .2, .3, .5], [1., 0., 0., 0.], [.1, .2, .3, .4]])
    np.testing.assert_allclose(m.project(raw), expected, atol=1e-15)
    np.testing.assert_allclose(m.project(raw)@m.EF[m.POS], [520., 1000., 580.])
    np.testing.assert_allclose(m.response(rows()[0]), [.2, .5, .1, .2])
    with pytest.raises(ValueError):
        m.project([[np.nan, 0., 0., 1.]])
