"""Synthetic-only V25 information, donor-pairing and release-time regressions."""
import copy

import numpy as np
import pandas as pd
import pytest

from scripts import monthly_transfer_v25 as m
from scripts.run_weather_recomposition_v11 import _indices
from scripts.weather_coarse_v14 import legal_coarse_law
from transcif.research.fuel_transport_v7 import FuelTrajectoryBank, recompose_paths


def monthly(n=40):
    starts = pd.date_range('2021-01-01', periods=n, freq='MS', tz='UTC')
    frame = pd.DataFrame(dict(region='BPAT', period_start=starts,
        period_end=starts+pd.offsets.MonthBegin(1),
        available_at=starts+pd.offsets.MonthBegin(1)+pd.Timedelta(days=30)))
    frame[list(m.M8_COLUMNS)] = np.random.default_rng(25).dirichlet(np.ones(8), n)
    return frame


def pair(excluded):
    regions = [r for r in m.SOURCES if r not in excluded]
    days = pd.date_range('2024-01-25', periods=2, tz='UTC')
    rng = np.random.default_rng(2501)
    blocks = {}
    for r in m.SOURCES:
        generation = rng.uniform(.1, 2, (2, 24, 8))
        mean = generation.mean((0, 1))
        q = generation/mean[None, None, :]
        mix = mean/mean.sum()
        raw = generation/generation.sum(2, keepdims=True) @ m.EF
        blocks[r] = (q, mix, raw, mean)
    return dict(q=np.concatenate([blocks[r][0] for r in regions]),
        donor_mix=np.concatenate([np.broadcast_to(blocks[r][1], (2, 8)) for r in regions]),
        raw_cif=np.concatenate([blocks[r][2] for r in regions]),
        monthmean=np.concatenate([np.broadcast_to(blocks[r][3], (2, 8)) for r in regions]),
        support=np.ones((10, 8), bool), regions=np.repeat(regions, 2),
        days=np.tile(days.asi8, 5), months=np.ones(10, int), row_ids=np.arange(10),
        excluded=np.array(excluded), source_regions=np.array(regions), cutoff=np.array(m.CUTOFF.value))


def bank():
    regions = [r for r in m.SOURCES if r != 'BPAT']
    query = dict(bank_regions=np.repeat(regions, 2),
        bank_days=np.tile(pd.date_range('2024-01-25', periods=2, tz='UTC').asi8, 6),
        bank_row_ids=np.arange(12), bank_cutoff=np.array(m.CUTOFF.value))
    return m.union_bank(pair(('BPAT', 'CISO')), pair(('BPAT', 'ERCO')), query, outer='BPAT')


def laws(n=2):
    rng = np.random.default_rng(2511)
    return {b: dict(mix=rng.dirichlet(np.ones(k), n),
                    draws=rng.dirichlet(np.ones(k), (n, 128))) for b, k in [('M8', 8), ('MG', 5)]}


def test_m8_law_matches_frozen_v14_all_arrays():
    frame = monthly()
    dates = pd.DatetimeIndex(['2020-12-31', '2021-03-31', '2022-03-01', '2024-04-30'], tz='UTC')
    actual = m.coarse_law(m.group_monthly(frame, 'M8'), 'BPAT', dates, budget='M8')
    expected = legal_coarse_law(frame, 'BPAT', dates)
    for key in ('mix', 'draws', 'metadata'):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=0)


def test_grouped_information_closure_and_empty_prior():
    frame = monthly()
    grouped = m.group_monthly(frame, 'MG')
    assert set(grouped) == {'region', 'period_start', 'period_end', 'available_at', *m.MG_COLUMNS}
    hidden = frame.copy()
    hidden[[m.M8_COLUMNS[i] for i in m.POS]] = frame[[m.M8_COLUMNS[i] for i in m.POS[::-1]]].to_numpy()
    pd.testing.assert_frame_equal(grouped, m.group_monthly(hidden, 'MG'))
    date = pd.DatetimeIndex(['2020-01-01'], tz='UTC')
    result = m.coarse_law(grouped, 'BPAT', date, budget='MG')
    np.testing.assert_array_equal(result['mix'][0], [.5, .125, .125, .125, .125])
    with pytest.raises(ValueError, match='only grouped'):
        m.coarse_law(frame, 'BPAT', date, budget='MG')


def test_release_future_poison_and_origin_order_invariance():
    grouped = m.group_monthly(monthly(), 'MG')
    dates = pd.date_range('2023-06-15', periods=3, tz='UTC')
    a = m.coarse_law(grouped, 'BPAT', dates, budget='MG')
    poison = grouped.copy()
    poison.loc[poison.available_at.gt(dates.max()), list(m.MG_COLUMNS)] = np.nan
    b = m.coarse_law(poison, 'BPAT', dates[::-1], budget='MG')
    for k in ('mix', 'draws', 'metadata'):
        np.testing.assert_array_equal(a[k], b[k][::-1])
    delayed = grouped.copy(); delayed.available_at += pd.Timedelta(days=4000)
    cold = m.coarse_law(delayed, 'BPAT', dates, budget='MG')
    assert np.all(cold['metadata'][:, :2] == 0)


def test_visible_support_and_pseudo_day_clipping():
    frame = monthly()
    frame[list(m.M8_COLUMNS)] = [0, .5, 0, .5, 0, 0, 0, 0]
    days = pd.DatetimeIndex(['2024-01-31', '2024-03-31'], tz='UTC')
    a = m.coarse_law(frame, 'BPAT', days, budget='M8')
    np.testing.assert_array_equal(a['draws'][..., [0, 2, 4, 5, 6, 7]], 0)
    np.testing.assert_allclose(a['draws'].sum(-1), 1, atol=1e-15)


@pytest.mark.parametrize('field', ['q', 'raw_cif', 'donor_mix', 'monthmean', 'support'])
def test_bank_duplicate_changes_fail_exact(field):
    first, second = pair(('BPAT', 'CISO')), pair(('BPAT', 'ERCO'))
    index = np.flatnonzero(second['regions'] == 'FPL')[0]
    if field == 'support':
        second[field][index, 0] = False
    else:
        second[field][index].flat[0] = np.nextafter(second[field][index].flat[0], np.inf)
    query = {k: bank()[v] for k, v in [('bank_regions', 'regions'), ('bank_days', 'days'),
                                      ('bank_row_ids', 'row_ids'), ('bank_cutoff', 'cutoff')]}
    with pytest.raises(ValueError, match='must be exact|normalization inconsistent'):
        m.union_bank(first, second, query, outer='BPAT')


def test_bank_union_alias_provenance_layout_and_no_held_source():
    b = bank()
    assert set(b['regions']) == set(m.SOURCES)-{'BPAT'}
    assert b['q'].strides == (1536, 8, 192)
    assert b['row_mapping'].shape == (12, 2)
    inputs = (pair(('BPAT', 'CISO')), pair(('BPAT', 'ERCO')))
    for row, (which, index) in enumerate(b['row_mapping']):
        for key in ('q', 'donor_mix', 'raw_cif', 'monthmean', 'support'):
            np.testing.assert_array_equal(b[key][row], inputs[which][key][index])


def test_m8_native_ids_and_raw_match_original_sampler():
    b = bank(); law = laws(); days = pd.date_range('2025-01-01', periods=2, tz='UTC')
    output = m.predict(b, law, days)
    old = FuelTrajectoryBank(tuple(b['source_regions']))
    old.q_, old.donor_mix_, old.days_ = b['q'], b['donor_mix'], pd.to_datetime(b['days'], utc=True)
    old.regions_, old.months_ = b['regions'], b['months']
    ids, _ = _indices(old, law['M8']['mix'], days, b['source_regions'])
    expected = np.stack([recompose_paths(draw, old.q_[selected]) for draw, selected in zip(law['M8']['draws'], ids)])
    actual = output['arms']['M8-recompose_conditional']
    np.testing.assert_array_equal(actual['donor_ids'], ids)
    np.testing.assert_array_equal(actual['raw'], expected)


def test_nine_arms_anchor_identity_and_shuffle_multisets():
    b = bank(); law = laws(); days = pd.date_range('2025-02-01', periods=2, tz='UTC')
    output = m.predict(b, law, days)
    assert tuple(output['arms']) == m.ARMS
    for budget in ('M8', 'MG'):
        joint, anchor, shuffled = [output['arms'][budget+'-'+arm] for arm in
                                  ('recompose_conditional', 'anchor_conditional', 'conditional_shuffled')]
        np.testing.assert_array_equal(joint['donor_ids'], anchor['donor_ids'])
        np.testing.assert_array_equal(joint['fine_draws'], anchor['fine_draws'])
        expected = b['raw_cif'][anchor['donor_ids']] + ((anchor['fine_draws']-b['donor_mix'][anchor['donor_ids']])@m.EF)[..., None]
        np.testing.assert_array_equal(anchor['raw'], expected)
        np.testing.assert_array_equal(shuffled['fine_draws'], joint['fine_draws'])
        for t, permutation in enumerate(shuffled['permutation']):
            np.testing.assert_array_equal(np.sort(permutation), np.arange(128))
            np.testing.assert_array_equal(shuffled['sampled_q'][t], joint['sampled_q'][t, permutation])
        assert not np.array_equal(shuffled['sampled_q'], joint['sampled_q'])
    for arm in output['arms'].values():
        assert arm['raw'].shape == (2, 128, 24)
        assert np.isfinite(arm['raw']).all()


def test_mg_group_preservation_and_plugin_mean():
    b = bank(); law = laws(1); day = pd.DatetimeIndex(['2025-03-01'], tz='UTC')
    result = m.predict(b, law, day)['arms']
    for name in ('anchor_conditional', 'recompose_global', 'recompose_conditional', 'conditional_shuffled', 'conditional_plugin_split'):
        arm = result['MG-'+name]
        np.testing.assert_allclose(arm['fine_draws'][..., m.POS].sum(-1), law['MG']['draws'][..., 0], atol=1e-15)
        np.testing.assert_array_equal(arm['fine_draws'][..., m.ZERO], law['MG']['draws'][..., 1:])
    plugin = result['MG-conditional_plugin_split']
    weight = plugin['weights'][0]
    ratio = b['donor_mix'][:, m.POS]/b['donor_mix'][:, m.POS].sum(1)[:, None]
    np.testing.assert_allclose(plugin['fine_draws'][0][:, m.POS]/law['MG']['draws'][0, :, :1],
                               np.broadcast_to(weight@ratio, (128, 4)), atol=1e-15)
    np.testing.assert_array_equal(plugin['sampled_q'], result['MG-recompose_conditional']['sampled_q'])


def test_mg_unsupported_group_borrows_ratio_and_q_jointly():
    b = bank(); bad = 0
    b['donor_mix'][bad] = [0, 0, 0, .25, .25, .25, .25, 0]
    uniforms = np.zeros((128, 8)); uniforms[:, 2] = .37
    weight = np.full(12, 1/12); ids = np.full(128, bad)
    draw = np.tile([.4, .1, .2, .2, .1], (128, 1))
    fine, q, fallback = m._fine_draws(b, draw, ids, weight, uniforms, 'MG')
    assert np.all(fallback >= 0) and np.all(fallback != bad)
    for k, donor in enumerate(fallback):
        np.testing.assert_array_equal(q[k][:, m.POS], b['q'][donor][:, m.POS])
        np.testing.assert_array_equal(q[k][:, m.ZERO], b['q'][bad][:, m.ZERO])
        ratio = b['donor_mix'][donor, m.POS]/b['donor_mix'][donor, m.POS].sum()
        np.testing.assert_allclose(fine[k, m.POS], .4*ratio)
    np.testing.assert_array_equal(recompose_paths(np.tile(b['donor_mix'][bad], (128, 1)), q),
                                  recompose_paths(np.tile(b['donor_mix'][bad], (128, 1)), b['q'][ids]))
    b['donor_mix'][:] = b['donor_mix'][bad]
    with pytest.raises(ValueError, match='no positive-group-supported'):
        m._fine_draws(b, draw, ids, weight, uniforms, 'MG')
    zero_draw = np.tile([0., .25, .25, .25, .25], (128, 1))
    for plugin in (False, True):
        fine, q, fallback = m._fine_draws(b, zero_draw, ids, weight, uniforms, 'MG', plugin)
        np.testing.assert_array_equal(fine[:, m.POS], 0)
        np.testing.assert_array_equal(q, b['q'][ids])
        np.testing.assert_array_equal(fallback, -1)


def test_cdf_never_selects_trailing_zero_mass():
    weight = np.r_[np.full(10, .1), 0.]
    assert m._sample(weight, np.array([np.nextafter(1., 0.)]))[0] == 9


def test_mg_conditional_kernel_cannot_see_hidden_split():
    b = bank(); altered = copy.deepcopy(b)
    altered['donor_mix'][:, m.POS] = b['donor_mix'][:, m.POS[::-1]]
    mix = np.array([.4, .1, .2, .1, .2]); day = pd.Timestamp('2025-01-01', tz='UTC')
    np.testing.assert_allclose(m.donor_weights(b, mix, day, budget='MG', conditional=True),
                               m.donor_weights(altered, mix, day, budget='MG', conditional=True), atol=1e-15)


def test_end_to_end_mg_hidden_monthly_split_cannot_change_any_prediction():
    frame = monthly(18)
    frame[list(m.M8_COLUMNS)] = [.25, .125, .0625, .125, .125, .125, .125, .0625]
    changed = frame.copy()
    changed[[m.M8_COLUMNS[i] for i in m.POS]] = [.0625, .0625, .125, .25]
    visible_a, visible_b = (m.group_monthly(x, 'MG') for x in (frame, changed))
    pd.testing.assert_frame_equal(visible_a, visible_b)
    days = pd.DatetimeIndex(['2025-01-01'], tz='UTC')
    law_a, law_b = (m.coarse_law(x, 'BPAT', days, budget='MG') for x in (visible_a, visible_b))
    for key in ('mix', 'draws', 'metadata'):
        np.testing.assert_array_equal(law_a[key], law_b[key])
    common = laws(1)['M8']; b = bank()
    a = m.predict(b, dict(M8=common, MG=law_a), days)['arms']
    c = m.predict(b, dict(M8=common, MG=law_b), days)['arms']
    for arm in m.ARMS:
        if arm.startswith('MG-'):
            for key in a[arm]:
                np.testing.assert_array_equal(a[arm][key], c[arm][key])
