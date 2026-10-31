"""Synthetic fixed-function replay, role guards and budget independence only."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import disjoint_point_v50 as m
from scripts import conditional_law_v29 as old


class Poison:
    def __array__(self, *_args, **_kwargs):
        raise AssertionError('Forbidden numeric conversion')
    def __float__(self):
        raise AssertionError('Forbidden numeric conversion')


class QueryRow(dict):
    def __getitem__(self, key):
        if key in {'fuelshare_coal', 'fuelshare_gas', 'fuelshare_oil', 'fuelshare_other'}:
            raise AssertionError('Local hidden fine accessed')
        return super().__getitem__(key)


class ExcludedRow(dict):
    def __getitem__(self, key):
        if key != 'region':
            raise AssertionError('Excluded training source read')
        return super().__getitem__(key)


def public_rows(region):
    answer = []
    for i, at in enumerate(pd.date_range('2022-01-01', '2025-06-01', freq='MS', tz='UTC')):
        end = at+pd.offsets.MonthBegin(1)
        answer.append(QueryRow(region=region, period_start=at.isoformat(), period_end=end.isoformat(),
            available_at=(end+pd.Timedelta(days=9, hours=12)).isoformat(), total_mwh=str(72000+100*i),
            fuelshare_nuclear=str(.1+.01*np.sin(i)), fuelshare_hydro='.2',
            fuelshare_wind='.1', fuelshare_solar='.1'))
    answer[-1]['total_mwh'] = Poison()
    answer[-1]['fuelshare_hydro'] = Poison()
    return answer


def proper_bank():
    rng = np.random.default_rng(5001)
    sources = m.LEGACY7[1:]
    regions, days, mixes, shapes = [], [], [], []
    for j, source in enumerate(sources):
        for month in (1, 2):
            mix = rng.dirichlet(np.ones(8))
            if j == 0 and month == 1:
                mix[m.POS] = 0.; mix[m.ZERO] = .25
            q = rng.uniform(.1, 2, (24, 8)); q /= q.mean(0); q[:, mix == 0] = 1.
            regions.append(source); mixes.append(mix); shapes.append(q)
            days.append(pd.Timestamp(year=2024, month=month, day=25 if month == 1 else 1, tz='UTC').value)
    mix = np.stack(mixes)
    q = np.ascontiguousarray(np.stack(shapes).transpose(0, 2, 1)).transpose(0, 2, 1)
    return dict(outer='BPAT', source_regions=np.array(sources), regions=np.array(regions),
        days=np.array(days, np.int64), row_ids=np.arange(12), cutoff=np.array(m.BANK_END.value),
        months=np.array([pd.Timestamp(x, tz='UTC').month for x in days]), q=q,
        donor_mix=mix, monthmean=mix*100, support=mix > 0,
        raw_cif=m.recompose_paths(mix, q))


def training_months():
    rng = np.random.default_rng(5002)
    answer = [ExcludedRow(region='BPAT')]
    for source in m.LEGACY7[1:]:
        for at in pd.date_range('2023-01-01', periods=4, freq='MS', tz='UTC'):
            end = at+pd.offsets.MonthBegin(1); mix = rng.dirichlet(np.ones(8))
            answer.append(dict(region=source, period_start=at.isoformat(), period_end=end.isoformat(),
                available_at=(end+pd.Timedelta(days=10)).isoformat(),
                **dict(zip(m.coarse.SHARES, map(str, mix)))))
    return answer


def training_records():
    rng = np.random.default_rng(5003)
    sources = np.repeat(m.LEGACY7[1:], 2)
    keys = [f'{s}/2023-{mo:02d}' for s in m.LEGACY7[1:] for mo in (5, 6)]
    starts = np.array([pd.Timestamp(k.split('/')[1]+'-01', tz='UTC').value for k in keys])
    ends = np.array([(pd.Timestamp(int(x), tz='UTC')+pd.offsets.MonthBegin(1)).value for x in starts])
    size = rng.normal(size=(12, 31)); size[:, :5] = rng.dirichlet(np.ones(5), size=12)
    size[:, 7:22] = rng.dirichlet(np.ones(5), size=(12, 3)).reshape(12, 15)
    cap = m.fleet._features(size, rng.dirichlet(np.ones(4), size=12), np.full(12, 1000.))['SizeCap']
    y = rng.dirichlet(np.ones(4), size=12); w = np.ones(12); records = []
    for arm, x in zip(m.HEADS, (size, cap)):
        family, method = arm.split('-'); mean, sd = np.zeros(x.shape[1]), np.ones(x.shape[1])
        if method == 'Ridge':
            mean = np.average(x, axis=0, weights=w); sd = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)); sd[sd <= 1e-12] = 1.
        records.append(dict(arm=arm, family=family, model=method, excluded_outer='BPAT',
            feature_names=m.fleet.FEATURE_NAMES[family], parameters=m.forward.PARAMETERS[method].copy(),
            training_keys=keys.copy(), training_regions=sources.tolist(), training_X=x.tolist(), training_y=y.tolist(),
            sample_weights=w.tolist(), scaler_mean=mean.tolist(), scaler_scale=sd.tolist(),
            training_period_start_ns=starts.tolist(), training_period_end_ns=ends.tolist(),
            training_available_ns=(ends+86400*10**9).tolist()))
    return records


@pytest.fixture(scope='module')
def fitted():
    return m.fit(proper_bank(), training_months(), training_records(), 'BPAT')


@pytest.fixture(scope='module')
def parity_query():
    return m.query(public_rows('BPAT'), 'BPAT', m.PARITY_DATES, np.full(4, .25), 1000.)


@pytest.fixture(scope='module')
def external_query():
    return m.query(public_rows('AVA'), 'AVA', m.EXTERNAL_DATES, np.full(4, .25), 1000.)


def test_decode_tokens_and_metadata_identity_before_training_numbers():
    records = training_records(); audits = {k: dict(region='BPAT', fits=[f]) for k, f in zip(('size', 'cap'), records)}
    tokenized = json.loads(json.dumps(audits), parse_float=m.FloatToken, parse_int=m.IntToken)
    decoded = m.decode_records(tokenized, 'BPAT')
    assert type(decoded[0]['parameters']['max_features']) is float
    assert isinstance(decoded[0]['training_y'][0][0], str)
    assert type(decoded[0]['training_period_start_ns'][0]) is int
    bad = copy.deepcopy(tokenized); bad['size']['fits'][0]['parameters']['max_features'] = m.IntToken('1')
    with pytest.raises(ValueError, match='token type'):
        m.decode_records(bad, 'BPAT')
    bad = copy.deepcopy(tokenized); bad['cap']['region'] = 'CISO'; bad['size']['fits'][0]['training_X'] = Poison()
    with pytest.raises(ValueError, match='audit'):
        m.decode_records(bad, 'BPAT')


def test_fit_exact_nine_calls_and_original_weights(monkeypatch):
    et, ridge = m.ExtraTreesRegressor.fit, m.Ridge.fit
    calls = []
    def tracked_et(self, *args, **kwargs):
        calls.append(('ET', self.max_features)); return et(self, *args, **kwargs)
    def tracked_ridge(self, *args, **kwargs):
        calls.append(('Ridge', self.alpha)); return ridge(self, *args, **kwargs)
    monkeypatch.setattr(m.ExtraTreesRegressor, 'fit', tracked_et)
    monkeypatch.setattr(m.Ridge, 'fit', tracked_ridge)
    records = training_records(); result = m.fit(proper_bank(), training_months(), records, 'BPAT')
    assert calls == [('ET', 1.)]*8+[('Ridge', 1.)]
    assert all(type(value) is float for _, value in calls)
    assert result['audit']['fit_count'] == 9
    assert result['scale'].audit['fitted_head_models'] == 7
    for actual, expected in zip(result['audit']['center_fits'], records):
        np.testing.assert_array_equal(actual['sample_weights'], expected['sample_weights'])
        np.testing.assert_array_equal(actual['scaler_mean'], expected['scaler_mean'])
        np.testing.assert_array_equal(actual['scaler_scale'], expected['scaler_scale'])
        assert set(actual['training_regions']) == set(m.LEGACY7)-{'BPAT'}


@pytest.mark.parametrize('kind', ['bank_source', 'outer', 'record_outer', 'future', 'weights', 'scaler', 'foreign_training'])
def test_training_guards_before_any_fit(kind, monkeypatch):
    b, records, rows = proper_bank(), training_records(), training_months()
    if kind == 'bank_source':
        b['regions'][0] = 'BPAT'; b['q'] = Poison(); records[0]['training_X'] = Poison()
    elif kind == 'outer': b['outer'] = 'CISO'
    elif kind == 'record_outer': records[0]['training_X'] = Poison(); records[1]['excluded_outer'] = 'CISO'
    elif kind == 'future': records[1]['training_available_ns'][0] = pd.Timestamp('2025-02-01', tz='UTC').value
    elif kind == 'weights': records[0]['sample_weights'][0] = 2.
    elif kind == 'scaler': records[1]['scaler_mean'][0] += 1.
    else: rows.append(ExcludedRow(region='AVA'))
    monkeypatch.setattr(m.ExtraTreesRegressor, 'fit', lambda *_a, **_k: pytest.fail('fit before guard'))
    monkeypatch.setattr(m.Ridge, 'fit', lambda *_a, **_k: pytest.fail('fit before guard'))
    with pytest.raises(ValueError): m.fit(b, rows, records, 'BPAT')


def test_old_source_full_mg_and_context_parity(parity_query):
    rows = public_rows('BPAT')
    records = m._monthly_records(rows, 'BPAT')
    visible = m._visible_frame(records, 'BPAT', m.PARITY_DATES[-1])
    expected = m.monthly.coarse_law(visible, 'BPAT', m.PARITY_DATES, budget='MG')
    for actual, wanted in [('MG_mix', 'mix'), ('MG_draws', 'draws'), ('MG_metadata', 'metadata')]:
        np.testing.assert_array_equal(parity_query[actual], expected[wanted])
    context = m.forward._context(rows, 'BPAT', m.PARITY_DATES)
    for name, value in context.items(): np.testing.assert_array_equal(parity_query[name], value)
    size = m.forward._features(expected['draws'], context, m.PARITY_DATES)['Size']
    np.testing.assert_array_equal(parity_query['features']['Size'], size)
    assert parity_query['audit']['expected_inventory_year'] == 2023
    assert external_query_role_length() == 177
    # The sealed monthly-entry guard remains unchanged.
    with pytest.raises(ValueError): m.forward._dates([m.PARITY_DATES[1]])


def external_query_role_length():
    return len(m.EXTERNAL_DATES)


def test_external_real_release_and_no_hidden_fine(external_query):
    at = pd.Timestamp('2024-07-10', tz='UTC'); j = m.EXTERNAL_DATES.get_loc(at)
    assert external_query['context_keys'][j, 0] == 'AVA/2024-05'
    assert external_query['context_keys'][j+1, 0] == 'AVA/2024-06'
    assert external_query['context_release_age_days'][j+1, 0] == .5
    assert external_query['features']['Size'].shape == (177, 128, 31)
    assert external_query['features']['SizeCap'].shape == (177, 128, 36)
    assert external_query['audit']['expected_inventory_year'] == 2022


@pytest.mark.parametrize('kind', ['blocked', 'wrong_dates', 'naive', 'reversed', 'zero_capacity', 'q_only', 'c_only', 'bad_q'])
def test_role_and_capacity_guards_precede_rows(kind):
    class Forbidden:
        def __iter__(self): raise AssertionError('Query rows read before guard')
    region, dates, q, c = 'AVA', m.EXTERNAL_DATES, np.full(4, .25), 1000.
    if kind == 'blocked': region = 'PACE'
    elif kind == 'wrong_dates': dates = dates[:-1]
    elif kind == 'naive': dates = dates.tz_localize(None)
    elif kind == 'reversed': dates = dates[::-1]
    elif kind == 'zero_capacity': c = 0.
    elif kind == 'q_only': c = None
    elif kind == 'c_only': q = None
    else: q = np.array([-.1, .4, .4, .3])
    with pytest.raises(ValueError): m.query(Forbidden(), region, dates, q, c)


def test_full_sampler_and_five_base_parity(fitted, parity_query):
    query = copy.deepcopy(parity_query)
    query['MG_draws'][:, :4] = [0., .25, .25, .25, .25]
    query['features']['Size'][..., :5] = query['MG_draws']
    query['features']['SizeCap'][..., :31] = query['features']['Size']
    actual = m.predict(fitted, query); ctx = fitted['scale']; b = fitted['bank']; mg = query['MG_draws']
    expected = ctx.final.conditional(mg.reshape(-1, 5)); mu = expected['mean'].reshape(181, 128, 4); std = expected['scale'].reshape(181, 128)
    np.testing.assert_array_equal(actual['query_mean'], mu)
    for t, at in enumerate(m.PARITY_DATES):
        uniforms = m.FuelTrajectoryBank._rng(17023, at).random((128, 8))
        weight = m.monthly.donor_weights(b, query['MG_mix'][t], at, budget='MG', conditional=True)
        ids = m.monthly._sample(weight, uniforms[:, 0])
        original, q, fallback = m.monthly._fine_draws(b, mg[t], ids, weight, uniforms, 'MG')
        selected = np.where(fallback >= 0, fallback, ids); mapping = ctx.table['bank_month_ids'][selected]
        before = mu[t]+(std[t]/ctx.oof_scale[mapping])[:, None]*ctx.oof_residual[mapping]
        fine = original.copy(); fine[:, m.POS] = mg[t, :, :1]*m.ratio_law.project(before)
        a = b['raw_cif'][ids]+((fine-b['donor_mix'][ids])@m.EF)[:, None]
        r = m.recompose_paths(fine, q)
        for name, expected_array in [('donor_ids', ids), ('fallback_ids', fallback), ('fine', fine), ('A', a), ('R', r)]:
            np.testing.assert_array_equal(actual[name][t], expected_array)
    assert np.any(actual['fallback_ids'] >= 0)
    assert actual['raw'].shape == (5, 181, 128, 24)
    np.testing.assert_array_equal(actual['raw'][0], actual['A'])
    for i, j in ((0, 1), (1, 3)):
        level = actual['center_levels'][i]
        np.testing.assert_array_equal(actual['raw'][j], actual['R']-actual['U'][..., None]+level[..., None])
        old_keep = actual['A']-actual['P'][..., None]+level[..., None]
        np.testing.assert_allclose(actual['raw'][j+1].mean(-1), old_keep.mean(-1), atol=1e-9, rtol=0)
        np.testing.assert_allclose(actual['raw'][j+1]-actual['raw'][j+1].mean(-1, keepdims=True),
                                   actual['R']-actual['R'].mean(-1, keepdims=True), atol=1e-9, rtol=0)
    assert np.any(abs(actual['A'][:, :4]) > 1e-9)
    np.testing.assert_array_equal(actual['R'][:, :4], 0.)
    assert actual['audit']['new_fits'] == 0


def test_b0_executable_without_capacity_and_independent_of_b1(fitted, external_query):
    b0 = m.query(public_rows('AVA'), 'AVA', m.EXTERNAL_DATES)
    assert set(b0['features']) == {'Size'} and not b0['audit']['capacity_used']
    original, only = m.predict(fitted, external_query), m.predict(fitted, b0)
    other = m.query(public_rows('AVA'), 'AVA', m.EXTERNAL_DATES, np.array([0., 1., 0., 0.]), 1e6)
    changed = m.predict(fitted, other)
    assert only['raw'].shape == (3, 177, 128, 24)
    assert only['center_levels'].shape == (1, 177, 128)
    np.testing.assert_array_equal(only['raw'], original['raw'][:3])
    np.testing.assert_array_equal(changed['raw'][:3], original['raw'][:3])
    np.testing.assert_array_equal(only['arms'], m.BASES5[:3])
    np.testing.assert_array_equal(only['budgets'], ('B0',)*3)
    class Centers:
        def __iter__(self):
            yield fitted['centers'][0]
            raise AssertionError('B0 read the B1 center')
    guarded = dict(fitted, centers=Centers())
    np.testing.assert_array_equal(m.predict(guarded, b0)['raw'], only['raw'])


def test_predict_role_and_mg_tampering_before_conditional(fitted, parity_query, monkeypatch):
    monkeypatch.setattr(fitted['scale'].final, 'conditional', lambda *_: pytest.fail('conditional before role validation'))
    wrong = copy.deepcopy(parity_query); wrong['region'] = 'CISO'
    with pytest.raises(ValueError, match='role'): m.predict(fitted, wrong)
    wrong = copy.deepcopy(parity_query); wrong['features']['Size'][0, 0, 0] += .01
    with pytest.raises(ValueError, match='coupling'): m.predict(fitted, wrong)


def test_cache_registry_is_literal_and_unique():
    from scripts import recompose_bridge_v48 as cache
    from scripts import daily_bridge_v47 as original
    assert original.ARMS.count(m.CACHE_NAMES[0]) == 1
    for name in m.CACHE_NAMES[1:]: assert cache.NEW_ARMS.count(name) == 1
    assert [cache.NEW_ARMS.index(name) for name in m.CACHE_NAMES[1:]] == [1, 61, 2, 62]
    assert tuple(m.LEGACY7) == tuple(old.SOURCES)


def test_missing_center_never_silently_returns_fewer_bases(fitted, external_query, monkeypatch):
    incomplete = dict(fitted, centers=fitted['centers'][:1])
    monkeypatch.setattr(fitted['scale'].final, 'conditional', lambda *_: pytest.fail('conditional before completeness guard'))
    with pytest.raises(ValueError, match='Both fixed'): m.predict(incomplete, external_query)
