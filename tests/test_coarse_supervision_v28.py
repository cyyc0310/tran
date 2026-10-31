"""Synthetic only: supervision-window closure and fixed bank residual sampling."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import coarse_supervision_v28 as m
from scripts import group_regression_v27 as old
from transcif.research.fuel_transport_v7 import recompose_paths


def coarse_rows():
    rng = np.random.default_rng(2801); rows = []
    for region in m.SOURCES:
        for start in pd.date_range('2021-12-01', '2025-01-01', freq='MS', tz='UTC'):
            end = start+pd.offsets.MonthBegin(1)
            mix = rng.dirichlet(np.ones(8))
            rows.append(dict(region=region, period_start=start.isoformat(), period_end=end.isoformat(),
                available_at=(end+pd.Timedelta(days=30)).isoformat(), **dict(zip(m.SHARES, map(str, mix)))))
    return rows


def bank_stream():
    rng = np.random.default_rng(2802)
    regions = np.repeat(m.SOURCES[1:], 2)
    dates = np.tile(pd.DatetimeIndex(['2024-01-25', '2024-02-01'], tz='UTC').asi8, 6)
    mix = rng.dirichlet(np.ones(8), 12)
    q = rng.uniform(.1, 2, (12, 24, 8)); q /= q.mean(1, keepdims=True)
    q = np.ascontiguousarray(q.transpose(0, 2, 1)).transpose(0, 2, 1)
    bank = dict(q=q, donor_mix=mix, raw_cif=recompose_paths(mix, q), support=np.ones((12, 8), bool), regions=regions, days=dates)
    origins = pd.date_range('2025-01-01', periods=2, tz='UTC')
    draws = rng.dirichlet(np.ones(5), (2, 128)); ids = rng.integers(0, 12, (2, 128), dtype=np.int64)
    fine, raw = [], []
    for t in range(2):
        ratio = mix[ids[t]][:, m.POS]/mix[ids[t]][:, m.POS].sum(1)[:, None]
        fd = np.zeros((128, 8)); fd[:, m.ZERO] = draws[t, :, 1:]; fd[:, m.POS] = draws[t, :, :1]*ratio
        fine.append(fd); raw.append(recompose_paths(fd, q[ids[t]].copy(order='K')))
    baseline = dict(raw=np.stack(raw), fine_draws=np.stack(fine), donor_ids=ids, fallback_ids=np.full((2, 128), -1, np.int64))
    return bank, baseline, draws, origins


@pytest.fixture(scope='module')
def experiment():
    bank, baseline, draws, dates = bank_stream()
    parent = old.fit_regressions(bank, outer='BPAT').predict(baseline, draws, dates)
    background = {k: np.stack([parent['arms'][a][k] for a in old.ARMS]) for k in m.FIELDS}
    background.update(arms=np.array(old.ARMS), origins=dates.asi8,
                      donor_ids=baseline['donor_ids'], fallback_ids=baseline['fallback_ids'])
    context = m.fit_heads(bank, coarse_rows(), outer='BPAT')
    return context, baseline, draws, dates, background


def test_windows_nested_cutoff_and_outer_future_numeric_poison():
    rows = coarse_rows()
    for row in rows:
        if row['region'] == 'BPAT':
            row['period_start'] = 'OUTER_TIME_POISON'
            row.update({k: 'OUTER_NUMERIC_POISON' for k in m.SHARES})
        elif row['period_start'] >= '2024-12-01':
            row.update({k: 'FUTURE_NUMERIC_POISON' for k in m.SHARES})
    selected = [m.select_coarse_rows(rows, outer='BPAT', window=w) for w in m.WINDOWS]
    assert [len(x['keys']) for x in selected] == [36, 180, 210]
    assert set(selected[0]['keys']) < set(selected[1]['keys']) < set(selected[2]['keys'])
    for x in selected:
        assert 'BPAT' not in x['regions']
        assert np.all(x['period_end'] <= m.CUTOFF.value) and np.all(x['available_at'] <= m.CUTOFF.value)
    assert all('/2024-12' not in key for key in selected[2]['keys'])


def test_deployment_release_boundary_is_inclusive_and_numeric_gate_is_first():
    rows = coarse_rows()
    row = next(r for r in rows if r['region'] == 'CISO' and r['period_start'].startswith('2024-11'))
    row['available_at'] = '2025-01-02T00:00:00Z'
    saved = {c: row[c] for c in m.SHARES}
    row.update({c: 'NOT_RELEASED_NUMERIC_POISON' for c in m.SHARES})
    absent = m.select_coarse_rows(rows, outer='BPAT', window='LR24')
    assert 'CISO/2024-11' not in absent['keys']
    row.update(saved); row['available_at'] = '2025-01-01T00:00:00Z'
    present = m.select_coarse_rows(rows, outer='BPAT', window='LR24')
    i = present['keys'].index('CISO/2024-11')
    assert present['available_at'][i] == m.CUTOFF.value


@pytest.mark.parametrize('defect', ['duplicate', 'share', 'period', 'release', 'region'])
def test_bad_eligible_source_rows_fail(defect):
    rows = coarse_rows()
    row = next(r for r in rows if r['region'] == 'CISO' and r['period_start'].startswith('2024-01'))
    if defect == 'duplicate':rows.append(row.copy())
    elif defect == 'share':row[m.SHARES[0]] = '-.1'
    elif defect == 'period':row['period_end'] = '2024-02-02T00:00:00+00:00'
    elif defect == 'release':row['available_at'] = '2024-01-01T00:00:00+00:00'
    else:row['region'] = 'UNSEEN'
    with pytest.raises(ValueError):m.select_coarse_rows(rows, outer='BPAT', window='S24')


def test_zero_pos_coarse_label_excluded_without_enlarging_bank_pool(experiment):
    rows = coarse_rows()
    row = next(r for r in rows if r['region'] == 'CISO' and r['period_start'].startswith('2024-01'))
    row.update(dict(zip(m.SHARES, map(str, [0, 0, 0, .25, .25, .25, .25, 0]))))
    history = m.select_coarse_rows(rows, outer='BPAT', window='S24')
    assert (~history['positive_mask']).sum() == 1
    assert 'CISO/2024-01' in history['audit']['zero_pos_keys']
    ctx = experiment[0]
    assert all(len(c.table['keys']) == 12 for c in ctx.contexts.values())
    assert [len(ctx.histories[w]['keys']) for w in m.WINDOWS] == [36, 180, 210]
    assert ctx.audit['residual_pool_expanded'] is ctx.audit['q_bank_expanded'] is False


def test_all_history_folds_exclude_teacher_months_and_weights_are_local(experiment):
    ctx = experiment[0]
    for window in m.WINDOWS:
        for name in ('Ridge', 'ET'):
            fit = ctx.audit['fit_windows'][window]['fits'][name]
            assert abs(fit['final']['weight_mean']-1) < 1e-15
            for fold in fit['folds']:
                held = fold['excluded_oof_region']
                assert held not in fold['training_regions'] and 'BPAT' not in fold['training_regions']
                assert all(not key.startswith(held+'/') for key in fold['training_keys'])
                assert all(key.startswith(held+'/2024-') for key in fold['residual_response_keys'])
                assert all(x <= m.CUTOFF.value for x in fold['available_at'])
            delta = ctx.contexts[window].residuals[name]
            predicted = np.array(fit['bank_oof_prediction_raw'])
            np.testing.assert_array_equal(delta, ctx.bank_table['ratios']-predicted)
    json.dumps(ctx.audit, allow_nan=False)


def test_held_coarse_label_perturbation_cannot_change_its_bank_oof_prediction(experiment):
    ctx = experiment[0]; rows = coarse_rows()
    for row in rows:
        if row['region'] == 'CISO':
            values = [row[m.SHARES[i]] for i in m.POS[::-1]]
            for i, v in zip(m.POS, values):row[m.SHARES[i]] = v
    changed = m.fit_heads(ctx.bank, rows, outer='BPAT')
    held = ctx.bank_table['regions'] == 'CISO'
    for window in m.WINDOWS:
        for name in ('Ridge', 'ET'):
            np.testing.assert_array_equal(ctx.contexts[window].residuals[name][held], changed.contexts[window].residuals[name][held])


def test_twenty_one_arms_exact_background_shared_stream_and_et_plugin(experiment):
    ctx, baseline, draws, dates, background = experiment
    out = ctx.predict(baseline, draws, dates, background)
    assert tuple(out['arms']) == m.ARMS and len(m.ARMS) == 21
    for arm, index in zip(m.ARMS[:3], [0, 2, 4]):
        for field in m.FIELDS:np.testing.assert_array_equal(out['arms'][arm][field], background[field][index])
    for window in m.WINDOWS:
        for name in ('Ridge', 'ET'):
            delta = ctx.contexts[window].residuals[name][out['effective_month_ids']]
            point = out[f'{window}_{name}_query_raw']
            a, b = [out['arms'][f'{window}-{name}-residual_{end}'] for end in ('anchor', 'recompose')]
            np.testing.assert_array_equal(a['unprojected_ratios'], point+delta)
            np.testing.assert_array_equal(a['fine_draws'], b['fine_draws'])
            p = out['arms'][f'{window}-{name}-plugin_recompose']
            np.testing.assert_array_equal(p['unprojected_ratios'], point)
            np.testing.assert_allclose(p['ratios'], old.project(point), atol=1e-15)
    for arm in out['arms'].values():
        assert arm['raw'].shape == (2, 128, 24) and np.isfinite(arm['raw']).all()
        np.testing.assert_allclose(arm['fine_draws'][..., m.POS].sum(-1), draws[..., 0], atol=1e-15)
        np.testing.assert_array_equal(arm['fine_draws'][..., m.ZERO], draws[..., 1:])
    assert out['head_source_EF_bounds'].shape == (3, 2)
    assert out['head_query_nearest_visible_distance'].shape == (3, 2, 128)
    assert not out['head_source_EF_outside'][:3].any()
    json.dumps(out['audit'], allow_nan=False)


@pytest.mark.parametrize('field', ['raw', 'fine_draws'])
def test_one_ulp_original_mg_stream_rejected(experiment, field):
    ctx, baseline, draws, dates, background = experiment
    bad = copy.deepcopy(baseline)
    bad[field][0, 0, 0] = np.nextafter(bad[field][0, 0, 0], np.inf)
    with pytest.raises(ValueError, match='Exact'):
        ctx.predict(bad, draws, dates, background)


def test_query_hidden_values_are_not_inputs(experiment):
    ctx, baseline, draws, dates, background = experiment
    a = ctx.predict({**baseline, 'local_eight_truth': np.ones((2, 8))/8}, draws, dates, background)
    b = ctx.predict({**baseline, 'local_eight_truth': np.zeros((2, 8))}, draws, dates, background)
    for arm in m.ARMS:
        for k in m.FIELDS:np.testing.assert_array_equal(a['arms'][arm][k], b['arms'][arm][k])


def test_predeployment_query_rejected_before_any_model_prediction(experiment, monkeypatch):
    ctx, baseline, draws, _, background = experiment
    dates = pd.DatetimeIndex(['2024-12-31', '2025-01-01'], tz='UTC')
    original = {**background, 'origins': dates.asi8}
    for context in ctx.contexts.values():
        for model in context.models.values():
            monkeypatch.setattr(model, 'predict', lambda *args, **kwargs: pytest.fail('Predeployment query reached model prediction'))
    with pytest.raises(ValueError, match='deployment cutoff'):
        ctx.predict(baseline, draws, dates, original)
