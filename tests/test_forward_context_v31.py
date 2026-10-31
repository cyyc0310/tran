"""Synthetic forward-budget and frozen-parent replay tests; no real data I/O."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import coarse_context_v30 as parent
from scripts import forward_context_v31 as m


def synthetic_rows():
    rows = []
    for r, region in enumerate(parent.SOURCES):
        for j, at in enumerate(pd.date_range('2022-01-01', '2025-07-01', freq='MS', tz='UTC')):
            end = at+pd.offsets.MonthBegin(1)
            zero = np.array([.08+.001*(j%12), .06+.002*r, .07+.002*(j%7), .03])
            ratio = np.array([.2+.01*r, .5-.01*r, .1+.001*(j%9), .2-.001*(j%9)])
            mix = np.empty(8); mix[parent.ZERO] = zero; mix[parent.POS] = (1-zero.sum())*ratio
            rows.append(dict(region=region, period_start=at.isoformat(), period_end=end.isoformat(),
                available_at=end.isoformat(), total_mwh=str((700+20*r+4*j)*((end-at)/pd.Timedelta(hours=1))),
                **dict(zip(parent.SHARES, map(str, mix)))))
    return rows


@pytest.fixture(scope='module')
def fixture():
    rows = synthetic_rows(); cases = parent.build_cases(rows)
    trained = parent.fit_predict(rows, cases, outer='MISO')
    audit = dict(region='MISO', outer='MISO', fits=trained['fits'], fit_count=10)
    origins = pd.date_range('2025-01-01', periods=2, freq='MS', tz='UTC')
    pos = np.linspace(.3, .9, 128)
    draws = np.tile(np.column_stack([pos, np.repeat(((1-pos)/4)[:, None], 4, axis=1)])[None], (2, 1, 1))
    return audit, rows, origins, draws


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b): same(left, right)
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    else: assert a == b


def call(fixture, **kw):
    audit, rows, origins, draws = fixture
    return m.fit_predict(kw.pop('audit', audit), kw.pop('rows', rows), region='MISO',
                         origins=kw.pop('origins', origins), mg_draws=kw.pop('draws', draws), **kw)


def test_hidden_poison_current_future_poison_oldquery_unused_immutability(fixture):
    audit, rows, origins, draws = fixture
    before = deepcopy((audit, rows, draws)); expected = call(fixture)
    same((audit, rows, draws), before)
    poisoned = deepcopy(rows)
    for row in poisoned:
        row.update({parent.SHARES[i]: 'HIDDEN_POISON' for i in parent.POS})
        if row['period_start'] >= '2025-02-01':
            row.update({parent.SHARES[i]: 'FUTURE_ZERO_POISON' for i in parent.ZERO})
            row['total_mwh'] = 'CURRENT_TOTAL_POISON'
    oldquery = deepcopy(audit)
    for fit in oldquery['fits']: fit['query_X'] = 'OLD_QUERY_POISON'
    same(expected, call(fixture, rows=poisoned, audit=oldquery))
    for family in m.FAMILIES:
        np.testing.assert_array_equal(expected['query_feature_'+family][..., :5], draws)


def test_shapes_point_product_and_parent_reproduction(fixture):
    result = call(fixture)
    assert result['ratios_raw'].shape == result['ratios'].shape == (6, 2, 128, 4)
    assert result['EF_draws'].shape == result['CIF_draws'].shape == (6, 2, 128)
    np.testing.assert_allclose(result['ratios'].sum(-1), 1)
    np.testing.assert_allclose(result['EF_draws'], result['ratios']@parent.EF[parent.POS])
    np.testing.assert_allclose(result['CIF_draws'], result['EF_draws']*fixture[3][None, ..., 0])
    np.testing.assert_allclose(result['CIF_point'], result['CIF_draws'].mean(-1))
    np.testing.assert_allclose(result['EF_point'], result['EF_draws'].mean(-1))
    assert np.max(np.abs(result['CIF_point']-fixture[3][..., 0].mean(-1)*result['EF_point'])) > 1e-5
    assert len(result['audit']['fits']) == 6
    for fit in result['audit']['fits']:
        assert fit['excluded_outer'] == 'MISO'
        assert set(fit['training_regions']) == set(parent.SOURCES)-{'MISO'}
        assert fit['parent_reproduction_max_abs'] <= (1e-10 if fit['model']=='Ridge' else 1e-12)


def test_release_boundary_gap_leap_and_context_features(fixture):
    rows = deepcopy(fixture[1])
    for row in rows:
        if row['region']=='MISO' and row['period_start'].startswith('2024-12'):
            row['available_at'] = '2025-01-01T00:00:00.000000001Z'
    result = call(fixture, rows=rows)
    assert result['context_keys'][0].tolist() == ['MISO/2024-11', 'MISO/2024-10', 'MISO/2024-09']
    np.testing.assert_array_equal(result['context_month_age'][0], [2, 3, 4])
    ordinary = call(fixture)
    assert ordinary['context_release_age_days'][0, 0] == 0
    assert ordinary['context_keys'][0, 0] == 'MISO/2024-12'
    older = [row for row in fixture[1] if row['period_start'] <= '2024-02-01T00:00:00+00:00']
    leap = call(fixture, rows=older)
    assert leap['context_keys'][0, 0] == 'MISO/2024-02'
    np.testing.assert_allclose(leap['context_MW'][0, 0], 700+20*5+4*25)
    trend = leap['query_feature_Trend']
    np.testing.assert_allclose(trend[0, :, -2], np.log(leap['context_MW'][0, 0]/leap['context_MW'][0, 1]))


@pytest.mark.parametrize('field', ['training_regions', 'training_keys', 'training_period_start_ns',
                                  'training_available_ns', 'parameters', 'scaler_mean', 'scaler_scale'])
def test_corrupt_parent_training_contract_rejected(fixture, field):
    audit = deepcopy(fixture[0]); fit = audit['fits'][0]
    if field=='training_regions': fit[field][0] = 'MISO'
    elif field=='training_keys': fit[field][0] = 'MISO/2022-04'
    elif field=='training_period_start_ns': fit[field][0] = pd.Timestamp('2025-01-01', tz='UTC').value
    elif field=='training_available_ns': fit[field][0] = pd.Timestamp('2025-01-02', tz='UTC').value
    elif field=='parameters': fit[field]['alpha'] = 2.
    else: fit[field][0] += .05
    with pytest.raises(ValueError): call(fixture, audit=audit)


@pytest.mark.parametrize('origins', [['2024-12-01T00:00:00Z'], ['2025-07-01T00:00:00Z'],
    ['2025-01-02T00:00:00Z'], ['2025-01-01'], ['2025-01-01T00:00:00Z']*2])
def test_date_guard_precedes_raw_access_and_fitting(fixture, origins):
    class PoisonRows:
        def __iter__(self): raise AssertionError('Raw rows accessed before date guard')
    with pytest.raises(ValueError):
        call(fixture, origins=origins, rows=PoisonRows(), audit={})


@pytest.mark.parametrize('kind', ['shape', 'negative', 'sum', 'nan'])
def test_invalid_mg_draws(fixture, kind):
    draws = fixture[3].copy()
    if kind=='shape': draws = draws[:, :127]
    elif kind=='negative': draws[0, 0, 0] = -.1
    elif kind=='sum': draws[0, 0, 0] += .1
    else: draws[0, 0, 0] = np.nan
    with pytest.raises(ValueError): call(fixture, draws=draws)


def test_missing_history_duplicate_and_bad_volume(fixture):
    rows = [row for row in fixture[1] if row['period_start'].startswith('2024-12')]
    with pytest.raises(ValueError): call(fixture, rows=rows)
    rows = deepcopy(fixture[1]); rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError): call(fixture, rows=rows)
    rows = deepcopy(fixture[1])
    for row in rows:
        if row['region']=='MISO' and row['period_start'].startswith('2024-12'): row['total_mwh']='0'
    with pytest.raises(ValueError): call(fixture, rows=rows)


@pytest.mark.parametrize('model,field', [('Ridge', 'coefficients'), ('ET', 'feature_importances')])
def test_parent_model_reproduction_rejects_tampering(fixture, model, field):
    audit = deepcopy(fixture[0])
    fit = next(fit for fit in audit['fits'] if fit['arm']=='G5-'+model)
    fit[field].flat[0] += .01
    with pytest.raises(ValueError, match='reproduction'):
        call(fixture, audit=audit)


def test_single_origin_ns_and_no_current_truth_access(fixture):
    raw = deepcopy(fixture[1])
    for row in raw:
        if row['period_start'] >= '2025-01-01':
            row.update({column:'CURRENT_FUTURE_POISON' for column in parent.SHARES})
            row['total_mwh'] = 'CURRENT_FUTURE_POISON'
    origins_ns = fixture[2].as_unit('ns').asi8[:1]
    result = call(fixture, rows=raw, origins=origins_ns, draws=fixture[3][:1])
    assert result['EF_point'].shape == result['CIF_point'].shape == (6, 1)
    np.testing.assert_array_equal(result['origin_ns'], origins_ns)


def test_datetime_resolution_preserves_ns_provenance_and_context_ages(fixture):
    dates_ns = fixture[2].as_unit('ns')
    dates_us = fixture[2].as_unit('us')
    assert dates_ns.dtype.unit == 'ns' and dates_us.dtype.unit == 'us'
    from_ns = call(fixture, origins=dates_ns)
    from_us = call(fixture, origins=dates_us)
    from_integer_ns = call(fixture, origins=dates_ns.asi8)
    same(from_ns, from_us)
    same(from_ns, from_integer_ns)
    np.testing.assert_array_equal(from_ns['origin_ns'], dates_ns.asi8)
    np.testing.assert_array_equal(from_ns['context_month_age'], [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(from_ns['context_release_age_days'], [[0., 31., 61.], [0., 31., 62.]])
    for field, row_field in (('context_start_ns', 'period_start'), ('context_end_ns', 'period_end'),
                             ('context_release_ns', 'available_at')):
        by_key = {parent._key(row['region'], pd.Timestamp(row['period_start'])): row for row in fixture[1]}
        expected = [[pd.Timestamp(by_key[key][row_field]).value for key in keys]
                    for keys in from_ns['context_keys']]
        np.testing.assert_array_equal(from_ns[field], expected)
