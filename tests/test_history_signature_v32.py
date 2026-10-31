"""Synthetic monthly history signatures and information-boundary tests."""
from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import pytest

from scripts import history_signature_v32 as m


def rows():
    result = []
    for r, region in enumerate(m.SOURCES):
        for j, at in enumerate(pd.date_range('2022-01-01', '2025-07-01', freq='MS', tz='UTC')):
            end = at+pd.offsets.MonthBegin(1)
            zero = np.array([.08+.003*(j%12), .06+.002*r, .07+.002*(j%7), .03])
            ratio = np.array([.15+.006*r+.001*(j%12), .55-.006*r-.001*(j%12), .1, .2])
            mix = np.empty(8); mix[m.ZERO] = zero; mix[m.POS] = (1-zero.sum())*ratio
            result.append(dict(region=region, period_start=at.isoformat(), period_end=end.isoformat(),
                available_at=end.isoformat(), total_mwh=str((700+20*r+4*j)*((end-at)/pd.Timedelta(hours=1))),
                **dict(zip(m.SHARES, map(str, mix)))))
    return result


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b): same(left, right)
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    else: assert a == b


@pytest.fixture(scope='module')
def fixture():
    raw = rows(); cases = m.build_cases(raw)
    dates = pd.date_range('2025-01-01', periods=2, freq='MS', tz='UTC').as_unit('ns')
    pos = np.linspace(.3, .9, 128)
    draws = np.tile(np.column_stack([pos, np.repeat(((1-pos)/4)[:, None], 4, axis=1)])[None], (2, 1, 1))
    result = m.fit_predict(raw, cases, outer='MISO', origins=dates, mg_draws=draws)
    return raw, cases, dates, draws, result


def predict(fixture, **kw):
    raw, cases, dates, draws, _ = fixture
    return m.fit_predict(kw.pop('raw', raw), kw.pop('cases', cases), outer='MISO',
                         origins=kw.pop('dates', dates), mg_draws=kw.pop('draws', draws), **kw)


def test_complete_common_cohort_prefixes_and_hidden_feature_poison(fixture):
    raw, cases, _, _, _ = fixture
    assert len(cases['keys']) == 7*18
    assert len(cases['audit']['skips']) == 7*12
    assert cases['context_keys'].shape == (126, 12)
    dimensions = [5, 30, 31, 39, 42, 43, 44, 45, 45]
    for family, width in zip(m.FAMILIES, dimensions):
        assert cases['X_by_family'][family].shape == (126, width)
        assert len(m.FEATURE_NAMES[family]) == width
    check_prefixes(cases['X_by_family'])
    poisoned = deepcopy(raw)
    for row in poisoned: row.update({m.SHARES[i]:'HIDDEN_POISON' for i in m.POS})
    same(cases, m.build_cases(poisoned))


def check_prefixes(features):
    pairs = [('Trend','G5'), ('Size','Trend'), ('LongTotal','Size'), ('JointLevel','LongTotal'),
             ('JointTrend','JointLevel'), ('JointAmplitude','JointTrend'),
             ('JointShape','JointTrend'), ('PhaseScramble','JointTrend')]
    for child, ancestor in pairs:
        np.testing.assert_array_equal(features[child][..., :features[ancestor].shape[-1]], features[ancestor])
    np.testing.assert_allclose(np.linalg.norm(features['JointShape'][..., -2:], axis=-1),
                               np.linalg.norm(features['PhaseScramble'][..., -2:], axis=-1), atol=1e-14)
    np.testing.assert_allclose(features['JointAmplitude'][..., -1],
                               np.linalg.norm(features['JointShape'][..., -2:], axis=-1))


def test_summary_linear_trend_normalization_harmonic_and_zero():
    ages = np.arange(1, 13, dtype=float); months = np.arange(12, 0, -1)
    values = 100+2*(-ages+ages.mean())
    result = m.history_summary(values, ages, months)
    expected = [np.log1p(100), np.quantile(values,.1,method='linear')/100, values.std(ddof=0)/100, .02, 0, 0]
    np.testing.assert_allclose(result['summary'], expected, atol=1e-14)
    assert result['harmonic_rank'] == 3
    assert np.isfinite(result['harmonic_condition'])
    zero = m.history_summary(np.zeros(12), ages, months)
    np.testing.assert_array_equal(zero['summary'], np.zeros(6))
    constant = m.history_summary(np.full(12, 50.), ages, months)
    np.testing.assert_allclose(constant['summary'], [np.log1p(50),1,0,0,0,0], atol=1e-14)
    values = 100*(1+.3*np.cos(2*np.pi*(months-1)/12))
    actual = m.history_summary(values, ages, months)
    z = values/values.mean()-1; tc = -ages+ages.mean(); beta = tc@z/(tc@tc)
    h = np.column_stack([np.ones(12),np.cos(2*np.pi*(months-1)/12),np.sin(2*np.pi*(months-1)/12)])
    coef = np.linalg.lstsq(h,z-beta*tc,rcond=None)[0]
    np.testing.assert_allclose(actual['summary'][3:], [beta,*coef[1:]], atol=1e-14)
    deficient = m.history_summary(np.full(12,50.), np.arange(1,145,12), np.ones(12))
    assert deficient['harmonic_rank'] == 1


def test_phase_rotation_matches_locked_hash_and_changes_only_phase(fixture):
    cases = fixture[1]
    for i in (0, 18, 100):
        region, origin = cases['regions'][i], pd.Timestamp(int(cases['origin_ns'][i]),unit='ns',tz='UTC')
        digest = hashlib.sha256(('history-signature-v32-phase/'+region+'/'+origin.strftime('%Y-%m')).encode()).digest()
        theta = 2*np.pi*(int.from_bytes(digest[:8],'big')/2**64)
        assert cases['phase_angle'][i] == theta
        c,s = cases['thermal_summary'][i,4:]
        expected = [c*np.cos(theta)-s*np.sin(theta), c*np.sin(theta)+s*np.cos(theta)]
        np.testing.assert_allclose(cases['X_by_family']['PhaseScramble'][i,-2:], expected)
    assert np.max(np.abs(cases['X_by_family']['JointShape'][:,-2:]-cases['X_by_family']['PhaseScramble'][:,-2:])) > 1e-5


def test_noncontiguous_history_inclusive_release_and_selected_invalid_fails():
    raw = [row for row in rows() if not (row['region']=='BPAT' and row['period_start'].startswith('2022-06'))]
    cases = m.build_cases(raw); i = cases['keys'].index('BPAT/2023-02')
    assert cases['context_keys'][i,0] == 'BPAT/2023-01'
    assert cases['context_keys'][i,-1] == 'BPAT/2022-01'
    assert cases['history_span'][i] == 13 and cases['history_maxgap'][i] == 2
    assert cases['context_release_age_days'][i,0] == 0
    for row in raw:
        if row['region']=='BPAT' and row['period_start'].startswith('2023-01'):
            row['available_at']='2023-02-01T00:00:00.000000001Z'
    assert 'BPAT/2023-02' not in m.build_cases(raw)['keys']
    raw = rows()
    for row in raw:
        if row['region']=='BPAT' and row['period_start'].startswith('2022-12'):
            row['total_mwh']='INVALID_SELECTED'
    with pytest.raises(ValueError): m.build_cases(raw)


def test_zero_thermal_past_legal_current_zero_pos_skips():
    raw = rows()
    for row in raw:
        if row['region']=='BPAT':
            for i in m.ZERO: row[m.SHARES[i]]='.25'
    cases = m.build_cases(raw)
    assert not any(key.startswith('BPAT/') for key in cases['keys'])
    assert any(item['reason']=='zero_POS' for item in cases['audit']['skips'])
    dates = pd.DatetimeIndex(['2025-01-01T00:00:00Z']).as_unit('ns')
    context = m.query_features(raw,region='BPAT',origins=dates,mg_draws=np.full((1,128,5),.2))
    np.testing.assert_array_equal(context['context_thermal_MW'], np.zeros((1,12)))
    np.testing.assert_array_equal(context['thermal_summary'], np.zeros((1,6)))


def test_outer_hidden_current_future_poison_and_immutability(fixture):
    raw,cases,_,draws,expected = fixture; before = deepcopy((raw,cases,draws))
    poisoned = deepcopy(raw)
    for row in poisoned:
        if row['region']=='MISO': row.update({m.SHARES[i]:'OUTER_HIDDEN_POISON' for i in m.POS})
        if row['period_start'] >= '2025-02-01':
            row.update({m.SHARES[i]:'CURRENT_FUTURE_POISON' for i in m.ZERO});row['total_mwh']='CURRENT_TOTAL_POISON'
    same(expected,predict(fixture,raw=poisoned))
    same((raw,cases,draws), before)


def test_fits_common_cohort_train_weights_scaler_and_mean_product(fixture):
    raw,cases,dates,draws,result = fixture
    assert result['ratios_raw'].shape == result['ratios'].shape == (18,2,128,4)
    assert len(result['fits']) == 18
    check_prefixes({family:result['query_feature_'+family] for family in m.FAMILIES})
    for family in m.FAMILIES:
        np.testing.assert_array_equal(result['query_feature_'+family][...,:5],draws)
    reference = result['fits'][0]
    for fit in result['fits']:
        assert fit['training_keys'] == reference['training_keys']
        assert all(not key.startswith('MISO/') for key in fit['training_keys'])
        assert set(fit['training_regions']) == set(m.SOURCES)-{'MISO'}
        w = fit['sample_weights']; regions = fit['training_regions']
        np.testing.assert_allclose([w[regions==r].sum() for r in set(regions)],len(w)/6)
        if fit['model']=='Ridge':
            mean = np.average(fit['training_X'],axis=0,weights=w)
            sd = np.sqrt(np.average((fit['training_X']-mean)**2,axis=0,weights=w));sd[sd<=1e-12]=1
            np.testing.assert_allclose(fit['scaler_mean'],mean)
            np.testing.assert_allclose(fit['scaler_scale'],sd)
    np.testing.assert_allclose(result['ratios'].sum(-1),1)
    np.testing.assert_allclose(result['EF_draws'],result['ratios']@m.EF[m.POS])
    np.testing.assert_allclose(result['CIF_point'],(result['EF_draws']*draws[None,...,0]).mean(-1))
    assert np.max(np.abs(result['CIF_point']-result['EF_point']*draws[...,0].mean(-1))) > 1e-5


def test_ns_us_integer_origins_and_age_values(fixture):
    _,_,dates,_,expected=fixture
    same(expected,predict(fixture,dates=dates.as_unit('us')))
    same(expected,predict(fixture,dates=dates.asi8))
    np.testing.assert_array_equal(expected['origin_ns'],dates.asi8)
    np.testing.assert_array_equal(expected['context_month_age'][:,:3],[[1,2,3],[1,2,3]])
    np.testing.assert_array_equal(expected['context_release_age_days'][:,:3],[[0,31,61],[0,31,62]])
    assert expected['context_keys'][0,10] == 'MISO/2024-02'
    assert expected['context_MW'][0,10] == 900.  # Leap February's 696 calendar hours.


def test_outer_case_features_never_change_training_stats_or_forward_predictions(fixture):
    cases=deepcopy(fixture[1])
    for family in m.FAMILIES:
        cases['X_by_family'][family][cases['regions']=='MISO',-1] += 1000.
    same(fixture[4],predict(fixture,cases=cases))


def test_region_weights_rebalanced_on_smaller_matched_training_cohort(fixture):
    raw=[row for row in fixture[0] if not (row['region']=='BPAT' and row['period_start'].startswith('2024-06'))]
    cases=m.build_cases(raw);result=predict(fixture,raw=raw,cases=cases)
    for fit in result['fits']:
        assert len(fit['training_keys']) == 6*18-1
        regions,w=fit['training_regions'],fit['sample_weights']
        np.testing.assert_allclose([w[regions==region].sum() for region in set(regions)],len(w)/6)
        assert w[regions=='BPAT'][0] > w[regions=='CISO'][0]


def test_future_training_case_rejected_before_hidden_response_conversion(fixture):
    raw=deepcopy(fixture[0]);cases=deepcopy(fixture[1])
    for row in raw:
        row.update({m.SHARES[i]:'HIDDEN_POISON' for i in m.POS})
    cases['origin_ns'][0]=pd.Timestamp('2025-01-01T00:00:00Z').value
    with pytest.raises(ValueError,match='L24'):
        predict(fixture,raw=raw,cases=cases)


def test_unreleased_context_numeric_poison_never_read(fixture):
    raw=deepcopy(fixture[0]);without=[]
    for row in raw:
        if row['region']=='MISO' and row['period_start'].startswith('2024-12'):
            row['available_at']='2026-01-01T00:00:00Z'
            row['total_mwh']='UNRELEASED_POISON'
            row.update({m.SHARES[i]:'UNRELEASED_POISON' for i in m.ZERO})
        else:without.append(row)
    args=dict(region='MISO',origins=fixture[2],mg_draws=fixture[3])
    same(m.query_features(raw,**args),m.query_features(without,**args))


@pytest.mark.parametrize('dates',[['2024-12-01T00:00:00Z'],['2025-07-01T00:00:00Z'],
                                 ['2025-01-02T00:00:00Z'],['2025-01-01']])
def test_date_guard_before_any_rows_or_labels(fixture, dates):
    class Poison:
        def __iter__(self): raise AssertionError('Accessed before origin guard')
    with pytest.raises(ValueError): predict(fixture,raw=Poison(),cases={},dates=dates)


@pytest.mark.parametrize('kind',['shape','negative','sum','nan'])
def test_invalid_forecast_draws_fail(fixture,kind):
    draws=fixture[3].copy()
    if kind=='shape':draws=draws[:,:127]
    elif kind=='negative':draws[0,0,0]=-.1
    elif kind=='sum':draws[0,0,0]+=.1
    else:draws[0,0,0]=np.nan
    with pytest.raises(ValueError):predict(fixture,draws=draws)


def test_missing_history_duplicate_and_bad_volume(fixture):
    with pytest.raises(ValueError):predict(fixture,raw=fixture[0][-5:])
    with pytest.raises(ValueError):m.build_cases([*fixture[0],deepcopy(fixture[0][0])])
    raw=deepcopy(fixture[0])
    for row in raw:
        if row['region']=='MISO' and row['period_start'].startswith('2024-12'):row['total_mwh']='0'
    with pytest.raises(ValueError):predict(fixture,raw=raw)
