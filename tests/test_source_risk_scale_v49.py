"""Synthetic exclusion, conditional-law parity and fixed H1/H2 sampling checks."""
import copy

import numpy as np
import pandas as pd
import pytest

from scripts import source_risk_scale_v49 as model
from scripts import conditional_law_v29 as old
from scripts import monthly_transfer_v25 as sampling
from transcif.research.fuel_transport_v7 import FuelTrajectoryBank, recompose_paths


PAIR = ('BPAT', 'CISO')
N = len(model.DAYS)


class ExcludedRow(dict):
    def __getitem__(self, key):
        if key != 'region':
            raise AssertionError('Excluded monthly row was accessed beyond region')
        return super().__getitem__(key)


class Poison:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Bank numerics read before exclusions')


def bank(excluded=PAIR, zero=True):
    rng = np.random.default_rng(4901)
    sources = [s for s in model.SOURCES if s not in excluded]
    regions, days, mixes, q = [], [], [], []
    for i, region in enumerate(sources):
        for month in (1, 2):
            mix = rng.dirichlet(np.ones(8))
            if zero and i == 0 and month == 1:
                mix[model.POS] = 0
                mix[model.ZERO] = .25
            shape = rng.uniform(.1, 2, (24, 8))
            shape /= shape.mean(0)
            shape[:, mix == 0] = 1
            regions.append(region)
            days.append(pd.Timestamp(year=2024, month=month, day=25 if month == 1 else 1, tz='UTC').value)
            mixes.append(mix); q.append(shape)
    mix = np.stack(mixes)
    q = np.ascontiguousarray(np.stack(q).transpose(0, 2, 1)).transpose(0, 2, 1)
    return dict(excluded=np.array(excluded), source_regions=np.array(sources), regions=np.array(regions),
        days=np.array(days, np.int64), months=np.array([pd.Timestamp(d, tz='UTC').month for d in days], np.int64),
        row_ids=np.arange(len(days)), cutoff=np.array(model.BANK_CUTOFF.value, np.int64),
        donor_mix=mix, monthmean=mix*100, support=mix > 0, q=q, raw_cif=recompose_paths(mix, q))


def monthly(excluded=PAIR):
    rng = np.random.default_rng(4902)
    rows = []
    for source in model.SOURCES:
        if source in excluded:
            rows.append(ExcludedRow(region=source))
            continue
        for start in pd.date_range('2023-01-01', periods=4, freq='MS', tz='UTC'):
            mix = rng.dirichlet(np.ones(8))
            end = start+pd.offsets.MonthBegin(1)
            rows.append(dict(region=source, period_start=start.isoformat(), period_end=end.isoformat(),
                             available_at=(end+pd.Timedelta(days=10)).isoformat(),
                             **dict(zip(model.coarse.SHARES, map(str, mix)))))
    return rows


@pytest.fixture(scope='module')
def context():
    b = bank()
    for value in b.values():
        value.setflags(write=False)
    return model.fit_scale(b, monthly(), PAIR)


def test_pair_bank_validation_and_metadata_first():
    model.validate_pair_bank(bank(), PAIR)
    b = bank(); b['q'] = Poison()
    with pytest.raises(ValueError):
        model.validate_pair_bank(b, ('CISO', 'BPAT'))
    with pytest.raises(ValueError):
        model.validate_pair_bank(bank(('BPAT',)), ('BPAT',))
    with pytest.raises(ValueError):
        model.fit_scale(bank(('BPAT',)), monthly(('BPAT',)), ('BPAT',))


@pytest.mark.parametrize('kind', ['source_order', 'foreign_region', 'duplicate', 'day', 'months', 'row_ids',
                                  'cutoff', 'support', 'q_prior', 'q_normalization', 'raw', 'monthmean',
                                  'nan', 'complex', 'raw_reconstruction', 'negative_q', 'shape', 'support_dtype'])
def test_invalid_bank_rejected(kind):
    b = bank()
    if kind == 'source_order': b['source_regions'] = b['source_regions'][::-1]
    elif kind == 'foreign_region': b['regions'][0] = 'BPAT'
    elif kind == 'duplicate': b['days'][1] = b['days'][0]; b['months'][1] = b['months'][0]
    elif kind == 'day': b['days'][0] = pd.Timestamp('2024-01-24', tz='UTC').value
    elif kind == 'months': b['months'][0] = 6
    elif kind == 'row_ids': b['row_ids'] = b['row_ids'][::-1]
    elif kind == 'cutoff': b['cutoff'] = np.array(model.BANK_CUTOFF.value+1)
    elif kind == 'support': b['support'][0, 0] = True
    elif kind == 'q_prior': b['q'][0, 0, 0] = 2
    elif kind == 'q_normalization': b['q'][1, :, 0] *= 1.1
    elif kind == 'raw': b['raw_cif'][1, 0] = 1001
    elif kind == 'monthmean': b['monthmean'][1, 0] += 2
    elif kind == 'nan': b['q'][0, 0, 0] = np.nan
    elif kind == 'complex': b['donor_mix'] = b['donor_mix'].astype(complex)
    elif kind == 'raw_reconstruction': b['raw_cif'][1, 0] += 1
    elif kind == 'negative_q': b['q'][1, 0, 0] = -1
    elif kind == 'shape': b['q'] = b['q'][:, :-1]
    else: b['support'] = b['support'].astype(float)
    with pytest.raises(ValueError):
        model.validate_pair_bank(b, PAIR)


def test_five_source_final_four_source_oof_and_zero_bank_response(context):
    assert context.audit['fit_count'] == 6
    allowed = set(model.SOURCES)-set(PAIR)
    assert set(context.final.metadata['training_regions']) == allowed
    assert len(context.folds) == 5
    for held, fold in context.folds.items():
        assert set(fold.metadata['training_regions']) == allowed-{held}
        assert all(k.split('/')[0] not in (*PAIR, held) for k in fold.metadata['training_keys'])
        columns = [i for i, k in enumerate(context.final.metadata['training_keys']) if k.startswith(held+'/')]
        np.testing.assert_array_equal(context.oof_weights[context.table['regions'] == held][:, columns], 0)
    zero = context.table['zero_pos_mask']
    assert zero.any()
    np.testing.assert_array_equal(context.oof_residual[zero], 0)
    assert np.isfinite(context.oof_scale).all() and np.all(context.oof_scale >= 1e-6)
    assert context.audit['excluded_pair'] == list(PAIR)


def test_six_source_fit_parity_with_original_v29():
    excluded = ('BPAT',)
    b, rows = bank(excluded), monthly(excluded)
    # Original V29 skips outer without touching its fields, just as this adapter.
    original = old.fit_leaf_controls(b, rows, outer='BPAT')
    actual = model._fit_scale(b, rows, excluded)
    for name in ('keys', 'X', 'ratios', 'regions', 'bank_month_ids', 'zero_pos_mask'):
        np.testing.assert_array_equal(actual.table[name], original.table[name])
    for name in ('oof_mean', 'oof_scale', 'oof_residual', 'oof_covariance3', 'oof_leaf_ids', 'oof_weights'):
        np.testing.assert_array_equal(getattr(actual, name), getattr(original, name))
    assert actual.final.metadata['training_keys'] == original.final.metadata['training_keys']
    np.testing.assert_array_equal(actual.final.weights, original.final.weights)
    probe = np.random.default_rng(49).dirichlet(np.ones(5), 8)
    for name, expected in original.final.conditional(probe).items():
        np.testing.assert_array_equal(actual.final.conditional(probe)[name], expected)


@pytest.fixture(scope='module')
def prediction(context):
    mix = np.tile([.5, .125, .125, .125, .125], (N, 1))
    draws = np.broadcast_to(mix[:, None], (N, 128, 5)).copy()
    draws[:, :4] = [0., .25, .25, .25, .25]
    return mix, draws, model.predict(context, mix, draws, model.DAYS, 'BPAT')


def test_fixed_stream_fallback_and_independent_first_day_formula(context, prediction):
    mix, draws, result = prediction
    day = model.DAYS[0]
    uniforms = FuelTrajectoryBank._rng(17023, day).random((128, 8))
    w = sampling.donor_weights(context.bank, mix[0], day, budget='MG', conditional=True)
    ids = sampling._sample(w, uniforms[:, 0])
    original, q, fallback = sampling._fine_draws(context.bank, draws[0], ids, w, uniforms, 'MG')
    effective = np.where(fallback >= 0, fallback, ids)
    mapping = context.table['bank_month_ids'][effective]
    query = context.final.conditional(draws[0])
    before = query['mean']+(query['scale']/context.oof_scale[mapping])[:, None]*context.oof_residual[mapping]
    fine = original.copy(); fine[:, model.POS] = draws[0, :, :1]*model.prior.project(before)
    anchor = context.bank['raw_cif'][ids]+((fine-context.bank['donor_mix'][ids])@model.EF)[:, None]
    for name, expected in [('donor_ids', ids), ('fallback_ids', fallback), ('fine', fine),
                           ('unprojected_ratios', before), ('A', anchor), ('R', recompose_paths(fine, q))]:
        np.testing.assert_allclose(result[name][0], expected, atol=1e-12, rtol=0)
    assert np.any(result['fallback_ids'] >= 0)
    np.testing.assert_array_equal(result['U'], result['fine'] @ model.EF)
    np.testing.assert_array_equal(result['P'], draws[..., 0]*(result['query_mean'] @ model.EF[model.POS]))
    np.testing.assert_allclose(result['fine'][..., model.POS].sum(-1), draws[..., 0], atol=1e-12)
    np.testing.assert_array_equal(result['fine'][..., model.ZERO], draws[..., 1:])
    # Additive donor residuals can survive a zero query POS; do not repair them.
    assert np.any(abs(result['A'][:, :4]) > 1e-9)
    np.testing.assert_array_equal(result['U'][:, :4], 0)
    np.testing.assert_array_equal(result['P'][:, :4], 0)
    np.testing.assert_array_equal(result['R'][:, :4], 0)
    assert result['A'].shape == result['R'].shape == (N, 128, 24)
    assert result['audit']['excluded_pair'] == list(PAIR) and result['audit']['held_region'] == 'BPAT'
    assert result['audit']['days'] == N == 342
    assert result['audit']['start'] == '2024-01-25T00:00:00+00:00'
    assert result['audit']['H1_query_days'] == 158 and result['audit']['H2_query_days'] == 184
    assert result['audit']['H1_future_other_source_bank_allowed'] is True
    assert result['audit']['H1_temporal_bank_holdout'] is False


def test_all_h1_h2_days_match_scale_formula_and_do_not_mutate_bank(context, prediction):
    mix, draws, result = prediction
    saved = {name: value.copy() for name, value in context.bank.items()}
    conditional = context.final.conditional(draws.reshape(-1, 5))
    mean = conditional['mean'].reshape(N, 128, 4)
    scale = conditional['scale'].reshape(N, 128)
    np.testing.assert_array_equal(result['query_mean'], mean)
    np.testing.assert_array_equal(result['query_scale'], scale)
    for t, day in enumerate(model.DAYS):
        uniform = FuelTrajectoryBank._rng(17023, day).random((128, 8))
        weight = sampling.donor_weights(context.bank, mix[t], day, budget='MG', conditional=True)
        ids = sampling._sample(weight, uniform[:, 0])
        original, q, fallback = sampling._fine_draws(context.bank, draws[t], ids, weight, uniform, 'MG')
        selected = np.where(fallback >= 0, fallback, ids)
        months = context.table['bank_month_ids'][selected]
        before = mean[t]+(scale[t]/context.oof_scale[months])[:, None]*context.oof_residual[months]
        fine = original.copy(); fine[:, model.POS] = draws[t, :, :1]*model.prior.project(before)
        np.testing.assert_array_equal(result['donor_ids'][t], ids)
        np.testing.assert_array_equal(result['fallback_ids'][t], fallback)
        np.testing.assert_array_equal(result['effective_month_ids'][t], months)
        np.testing.assert_allclose(result['fine'][t], fine, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result['R'][t], recompose_paths(fine, q), atol=1e-12, rtol=0)
        anchor = context.bank['raw_cif'][ids]+((fine-context.bank['donor_mix'][ids])@model.EF)[:, None]
        np.testing.assert_allclose(result['A'][t], anchor, atol=1e-12, rtol=0)
    for name, value in saved.items():
        np.testing.assert_array_equal(context.bank[name], value)


@pytest.mark.parametrize('kind', ['region', 'dates', 'naive', 'order', 'draws', 'point', 'complex'])
def test_prediction_guards_before_sampling(context, kind):
    mix = np.tile([.5, .125, .125, .125, .125], (N, 1))
    draws = np.broadcast_to(mix[:, None], (N, 128, 5)).copy()
    days, region = model.DAYS, 'BPAT'
    if kind == 'region': region = 'ERCO'
    elif kind == 'dates': days = model.DAYS[:-1]
    elif kind == 'naive': days = model.DAYS.tz_localize(None)
    elif kind == 'order': days = model.DAYS[::-1]
    elif kind == 'draws': draws[0, 0, 0] = np.nan
    elif kind == 'point': mix[0, 0] = -.1
    else: draws = draws.astype(complex)
    with pytest.raises(ValueError):
        model.predict(context, mix, draws, days, region)


def test_invalid_source_scale_fails_without_fallback(context):
    broken = copy.copy(context)
    broken.oof_scale = context.oof_scale.copy()
    broken.oof_scale[0] = 0
    mix = np.full((N, 5), .2)
    with pytest.raises(ValueError):
        model.predict(broken, mix, np.broadcast_to(mix[:, None], (N, 128, 5)), model.DAYS, 'BPAT')
