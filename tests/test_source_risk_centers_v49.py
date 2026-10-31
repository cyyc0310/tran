"""Pair masking and historical features without real source values."""
import copy
import numpy as np
import pandas as pd
import pytest
from scripts import source_risk_centers_v49 as m


class Poison:
    def __array__(self, *_args, **_kwargs): raise AssertionError('Excluded numeric accessed')
    def __float__(self): raise AssertionError('Excluded numeric accessed')


def records():
    rng = np.random.default_rng(4901)
    regions = np.repeat(m.SOURCES[1:], 2)
    keys = [f'{s}/2023-{month:02d}' for s in m.SOURCES[1:] for month in (5, 6)]
    starts = np.array([pd.Timestamp(k.split('/')[1]+'-01', tz='UTC').value for k in keys], np.int64)
    ends = np.array([(pd.Timestamp(int(v), tz='UTC')+pd.offsets.MonthBegin(1)).value for v in starts], np.int64)
    size = rng.normal(size=(12, 31)); cap = np.c_[size, rng.random((12, 5))]
    y = rng.dirichlet(np.ones(4), size=12)
    answer = []
    for arm, x in zip(m.HEADS, (size, cap)):
        family, method = arm.split('-')
        answer.append(dict(arm=arm, family=family, model=method, excluded_outer='BPAT',
            parameters=m.forward.PARAMETERS[method], feature_names=m.fleet.FEATURE_NAMES[family],
            training_keys=keys, training_regions=regions.tolist(), training_period_start_ns=starts.tolist(),
            training_period_end_ns=ends.tolist(), training_available_ns=(ends+10**9).tolist(),
            training_X=x.tolist(), training_y=y.tolist()))
    return answer


def test_pair_mask_precedes_training_numeric_and_model_shared():
    r = records()
    for fit in r:
        for i, source in enumerate(fit['training_regions']):
            if source == 'CISO':
                fit['training_X'][i] = Poison(); fit['training_y'][i] = Poison()
    result = m.fit_centers(r, ('BPAT', 'CISO'))
    assert result['audit']['fit_count'] == 2
    for fit in result['audit']['fits']:
        assert set(fit['training_regions']) == set(m.SOURCES)-{'BPAT', 'CISO'}
        assert not any(k.startswith('CISO/') for k in fit['training_keys'])
        assert np.isclose(np.sum(fit['sample_weights']), 10)


def test_all_record_metadata_before_any_training_numeric():
    r = records(); r[0]['training_X'] = Poison(); r[1]['excluded_outer'] = 'NYIS'
    with pytest.raises(ValueError, match='identity'):
        m.fit_centers(r, ('BPAT', 'CISO'))


def test_historical_feature_release_and_hidden_field_guard():
    class Visible(dict):
        def __getitem__(self, key):
            if key in {'fuelshare_coal', 'fuelshare_gas', 'fuelshare_oil', 'fuelshare_other'}:
                raise AssertionError('Local fine read')
            return super().__getitem__(key)
    rows = []
    for at in pd.date_range('2023-06-01', '2024-12-01', freq='MS', tz='UTC'):
        end = at+pd.offsets.MonthBegin(1)
        rows.append(Visible(region='BPAT', period_start=at.isoformat(), period_end=end.isoformat(),
            available_at=(end+pd.Timedelta(days=9,hours=12)).isoformat(), total_mwh='72000',
            fuelshare_nuclear='.1', fuelshare_hydro='.2', fuelshare_wind='.1', fuelshare_solar='.1'))
    rows[-1]['total_mwh'] = Poison()
    q = m.features(rows, 'BPAT', np.full((342, 128, 5), .2), np.full(4, .25), 1000.)
    assert q['context_keys'][m.DAYS.get_loc(pd.Timestamp('2024-07-10', tz='UTC')), 0] == 'BPAT/2024-05'
    assert q['context_keys'][m.DAYS.get_loc(pd.Timestamp('2024-07-11', tz='UTC')), 0] == 'BPAT/2024-06'
    assert q['context_release_age_days'][m.DAYS.get_loc(pd.Timestamp('2024-07-11', tz='UTC')), 0] == .5
    assert np.all(q['context_release_ns'] <= q['origin_ns'][:, None])
    fitted = m.fit_centers(records(), ('BPAT', 'CISO'))
    prediction = m.predict_centers(fitted, q)
    assert prediction['CIF_draws'].shape == (2, 342, 128)
    changed = copy.deepcopy(q); changed['region'] = 'ERCO'
    with pytest.raises(ValueError): m.predict_centers(fitted, changed)


def test_error_decomposition_and_alignment():
    rng = np.random.default_rng(4902)
    point = rng.uniform(0, 1000, (8, 3, 24)); truth = rng.uniform(0, 1000, (3, 24))
    a = m.errors(point, truth); b = m.errors(point+20, truth)
    assert a['reconstruction_max_abs'] < 1e-9
    np.testing.assert_allclose(a['shape_error'], b['shape_error'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(b['level_error'], a['level_error']-20, atol=1e-9, rtol=0)


def test_eight_base_registry_and_raw_daily_mean_control():
    rng = np.random.default_rng(4903)
    mg = rng.dirichlet(np.ones(5), size=(342, 128))
    q = rng.dirichlet(np.ones(4), size=(342, 128))
    fine = np.zeros((342, 128, 8)); fine[..., m.POS] = mg[..., :1]*q
    fine[..., m.coarse.ZERO] = mg[..., 1:]
    a, r = (rng.uniform(-20, 800, (342, 128, 24)) for _ in range(2))
    levels = np.stack((mg[..., 0]*500, mg[..., 0]*800))
    paths = m.bases(a, r, fine, q, mg, levels)['raw']
    assert paths.shape == (8, 342, 128, 24)
    assert len(m.ARMS) == 8 and m.BUDGETS.count('B0') == 5
    np.testing.assert_array_equal(paths[:2], np.stack((a, r)))
    for keep, aligned in ((2, 4), (5, 7)):
        np.testing.assert_allclose(paths[keep].mean(-1), paths[aligned].mean(-1), atol=1e-9, rtol=0)
        np.testing.assert_allclose(paths[aligned]-paths[aligned].mean(-1, keepdims=True),
                                   r-r.mean(-1, keepdims=True), atol=1e-9, rtol=0)
