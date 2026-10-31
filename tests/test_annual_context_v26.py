"""Synthetic-only annual-release, matched-stream and support regressions."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import annual_context_v26 as m
from transcif.research.fuel_transport_v7 import recompose_paths


def annual(region, year, mix, release=None):
    return dict(region=region, year=str(year), schema='eight', unit='proxy_year',
        period_start=f'{year}-01-01T00:00:00Z', period_end=f'{year+1}-01-01T00:00:00Z',
        available_at=release or f'{year+1}-03-01T00:00:00Z',
        **{'lower_mass_'+fuel: str(value) for fuel, value in zip(m.FUELS, mix)})


def inputs(*, zero_query=False, borrow=False, zero_hour=False):
    rng = np.random.default_rng(26023)
    dates = pd.date_range('2025-01-01', periods=2, tz='UTC')
    mix = rng.dirichlet(np.ones(8), 6)
    q = rng.uniform(.1, 2., (6, 24, 8))
    support = np.ones((6, 8), bool)
    if borrow:
        mix[0] = [0., 0., 0., .25, .25, .25, .25, 0.]
        support[0, m.POS] = False
        q[0][:, m.POS] = 1.
    if zero_hour:
        q[:, 0, :] = 0.
    q = np.ascontiguousarray(q.transpose(0, 2, 1)).transpose(0, 2, 1)
    bank = dict(donor_mix=mix, q=q, support=support, raw_cif=recompose_paths(mix, q),
        regions=np.array(m.SOURCES[1:]), days=np.repeat(pd.Timestamp('2024-01-25', tz='UTC').value, 6))
    draws = rng.dirichlet(np.ones(5), (2, 128))
    if zero_query:
        draws[:] = [0., .25, .25, .25, .25]
    ids = rng.integers(0, 6, (2, 128), dtype=np.int64)
    if borrow:
        ids[:] = 0
    fallback = np.full((2, 128), -1, np.int64)
    fine, paths = [], []
    for t in range(2):
        sampled_q = q[ids[t]].copy(order='K')
        selected = ids[t].copy()
        if borrow and not zero_query:
            selected[:] = 1; fallback[t] = 1
            for particle in range(128):
                sampled_q[particle][:, m.POS] = q[1][:, m.POS]
        mass = mix[selected][:, m.POS].sum(1)
        ratio = np.divide(mix[selected][:, m.POS], mass[:, None], out=np.zeros((128, 4)), where=mass[:, None] > 0)
        fd = np.zeros((128, 8)); fd[:, m.ZERO] = draws[t, :, 1:]; fd[:, m.POS] = draws[t, :, :1]*ratio
        fine.append(fd); paths.append(recompose_paths(fd, sampled_q))
    baseline = dict(raw=np.stack(paths), fine_draws=np.stack(fine), donor_ids=ids, fallback_ids=fallback)
    target_mix = [.4, .05, .025, .125, .125, .125, .125, .025]
    source_mix = [.05, .3, .1, .125, .125, .125, .125, .05]
    rows = [annual('BPAT', 2023, target_mix)]+[annual(r, 2022, source_mix) for r in m.SOURCES[1:]]
    return bank, baseline, draws, dates, rows


def run(fixture):
    bank, baseline, draws, dates, rows = fixture
    return m.predict(bank, baseline, draws, dates, rows, region='BPAT')


def test_context_release_gate_before_numeric_and_latest_year():
    rows = [annual('BPAT', 2022, [.125]*8), annual('BPAT', 2023, [.125]*8)]
    for fuel in m.FUELS:
        rows[1]['lower_mass_'+fuel] = 'FUTURE_POISON'
    at = pd.Timestamp('2024-02-29', tz='UTC')
    context = m.select_annual_context(rows, 'BPAT', at)
    assert context.status == 'available' and context.year == 2022
    assert context.age_days >= context.release_age_days >= 0
    json.dumps(context.as_dict(), allow_nan=False)
    with pytest.raises(ValueError, match='could not convert'):
        m.select_annual_context(rows, 'BPAT', pd.Timestamp('2024-03-01', tz='UTC'))
    rows[1] = annual('BPAT', 2023, [.125]*8)
    assert m.select_annual_context(rows[::-1], 'BPAT', pd.Timestamp('2024-03-01', tz='UTC')).year == 2023


def test_duplicate_year_and_incomplete_calendar_rejected():
    row = annual('BPAT', 2022, [.125]*8)
    at = pd.Timestamp('2025-01-01', tz='UTC')
    with pytest.raises(ValueError, match='Duplicate'):
        m.select_annual_context([row, row.copy()], 'BPAT', at)
    bad = row.copy(); bad['period_start'] = '2022-02-01T00:00:00Z'
    with pytest.raises(ValueError, match='Complete calendar'):
        m.select_annual_context([bad], 'BPAT', at)
    with pytest.raises(ValueError, match='timezone'):
        m.select_annual_context([row], 'BPAT', pd.Timestamp('2025-01-01'))


def test_other_schema_and_region_not_parsed_and_zero_pos_is_explicit():
    ignored = dict(region='BPAT', schema='merged_positive_proxy', period_end='POISON')
    foreign = dict(region='CISO', schema='eight', period_end='POISON')
    at = pd.Timestamp('2025-01-01', tz='UTC')
    assert m.select_annual_context([ignored, foreign], 'BPAT', at).status == 'missing'
    rows = [annual('BPAT', 2022, [.125]*8), annual('BPAT', 2023, [0, 0, 0, .25, .25, .25, .25, 0])]
    context = m.select_annual_context(rows, 'BPAT', at)
    assert context.status == 'zero_pos' and context.year == 2023 and context.ratio is None


def test_five_arms_exact_baseline_group_mass_fixed_and_anchors():
    fixture = inputs(); bank, baseline, draws, dates, _ = fixture
    snapshot = copy.deepcopy(fixture); out = run(fixture)
    assert tuple(out['arms']) == m.ARMS
    np.testing.assert_array_equal(out['arms'][m.ARMS[0]]['raw'], baseline['raw'])
    np.testing.assert_array_equal(out['arms'][m.ARMS[0]]['fine_draws'], baseline['fine_draws'])
    np.testing.assert_array_equal(out['donor_ids'], baseline['donor_ids'])
    np.testing.assert_array_equal(out['fallback_ids'], baseline['fallback_ids'])
    target = np.array([.8, .1, .05, .05])
    for arm, arrays in out['arms'].items():
        np.testing.assert_allclose(arrays['fine_draws'][..., m.POS].sum(-1), draws[..., 0], atol=1e-15)
        np.testing.assert_array_equal(arrays['fine_draws'][..., m.ZERO], draws[..., 1:])
        assert arrays['unsupported_positive_energy_fraction'].shape == (2, 128)
        assert arrays['zero_total_mask'].shape == (2, 128, 24)
        if 'anchor' in arm:
            expected = bank['raw_cif'][baseline['donor_ids']] + ((arrays['fine_draws']-bank['donor_mix'][baseline['donor_ids']])@m.EF)[..., None]
            np.testing.assert_array_equal(arrays['raw'], expected)
    np.testing.assert_allclose(out['arms']['AY-fixed_recompose']['ratios'], np.broadcast_to(target, (2, 128, 4)), atol=1e-15)
    for before, after in zip(snapshot[:3], fixture[:3]):
        if isinstance(before, dict):
            for field in before:
                np.testing.assert_array_equal(before[field], after[field])
        else:
            np.testing.assert_array_equal(before, after)
    json.dumps(out['audit'], allow_nan=False)


def test_residual_formula_projection_and_donor_year_asof():
    fixture = inputs(); bank, baseline, _, _, rows = fixture
    source_future = annual('CISO', 2023, [.125]*8, release='2024-02-01T00:00:00Z')
    for fuel in m.FUELS:
        source_future['lower_mass_'+fuel] = 'FUTURE_SOURCE_POISON'
    rows.append(source_future)
    out = run(fixture)
    original = out['arms'][m.ARMS[0]]['ratios']
    source = np.array(out['audit']['donor_contexts'][0]['ratio'])
    target = np.array(out['audit']['target_contexts'][0]['ratio'])
    expected = m.project_simplex(target+original-source)
    np.testing.assert_allclose(out['arms']['AY-residual_recompose']['ratios'], expected, atol=1e-15)
    np.testing.assert_array_equal(out['residual_unprojected_ratios'], target+original-source)
    np.testing.assert_array_equal(out['residual_projection_l1'], abs(expected-(target+original-source)).sum(-1))
    np.testing.assert_array_equal(out['residual_projection_boundary_count'], (expected == 0).sum(-1))
    assert out['residual_projection_applied'].all()
    assert not out['source_year_unavailable'].any()
    assert all(c['year'] == 2022 for c in out['audit']['donor_contexts'])
    np.testing.assert_array_equal(out['donor_year'], 2022)
    assert np.any(expected == 0)
    for day in range(2):
        ids = baseline['donor_ids'][day]
        own_ratio = bank['donor_mix'][ids][:, m.POS]/bank['donor_mix'][ids][:, m.POS].sum(1)[:, None]
        np.testing.assert_allclose(original[day], own_ratio)


@pytest.mark.parametrize('zero_year', [False, True])
def test_missing_target_uses_original_ratios_and_recompose(zero_year):
    fixture = list(inputs()); fixture[4] = [r for r in fixture[4] if r['region'] != 'BPAT']
    if zero_year:
        fixture[4].append(annual('BPAT', 2023, [0, 0, 0, .25, .25, .25, .25, 0]))
    out = run(fixture)
    assert out['target_context_unavailable'].all()
    assert out['audit']['target_original_ratio_particles'] == 256
    np.testing.assert_array_equal(out['residual_projection_l1'], 0)
    assert not out['residual_projection_applied'].any()
    np.testing.assert_array_equal(out['residual_unprojected_ratios'], out['arms'][m.ARMS[0]]['ratios'])
    for arm in m.ARMS:
        np.testing.assert_array_equal(out['arms'][arm]['fine_draws'], fixture[1]['fine_draws'])
        np.testing.assert_array_equal(out['arms'][arm]['ratios'], out['arms'][m.ARMS[0]]['ratios'])
    for arm in ['AY-fixed_recompose', 'AY-residual_recompose']:
        np.testing.assert_allclose(out['arms'][arm]['raw'], fixture[1]['raw'], atol=1e-12)


@pytest.mark.parametrize('zero_year', [False, True])
def test_missing_source_year_is_fixed_not_original_mg(zero_year):
    fixture = list(inputs()); fixture[4] = [r for r in fixture[4] if r['region'] == 'BPAT']
    if zero_year:
        fixture[4] += [annual(r, 2022, [0, 0, 0, .25, .25, .25, .25, 0]) for r in m.SOURCES[1:]]
    out = run(fixture)
    assert out['source_year_unavailable'].all() and not out['target_context_unavailable'].any()
    assert out['audit']['residual_zero_delta_particles'] == 256
    np.testing.assert_array_equal(out['residual_projection_l1'], 0)
    assert not out['residual_projection_applied'].any()
    np.testing.assert_array_equal(out['residual_unprojected_ratios'], out['arms']['AY-fixed_recompose']['ratios'])
    for key in ['fine_draws', 'raw', 'ratios']:
        np.testing.assert_array_equal(out['arms']['AY-residual_recompose'][key], out['arms']['AY-fixed_recompose'][key])
    assert not np.array_equal(out['arms']['AY-fixed_recompose']['ratios'], out['arms'][m.ARMS[0]]['ratios'])


def test_fallback_j_controls_month_ratio_date_annual_support_jointly():
    fixture = list(inputs(borrow=True)); bank, baseline, draws, dates, rows = fixture
    bank['days'][1] = pd.Timestamp('2024-03-02', tz='UTC').value
    new_source = [.1, .2, .1, .125, .125, .125, .125, .1]
    rows.append(annual('ERCO', 2023, new_source))
    out = run(fixture)
    assert out['audit']['donor_contexts'][1]['year'] == 2023
    np.testing.assert_array_equal(out['effective_pos_ids'], 1)
    expected_original = bank['donor_mix'][1, m.POS]/bank['donor_mix'][1, m.POS].sum()
    np.testing.assert_allclose(out['arms'][m.ARMS[0]]['ratios'], np.broadcast_to(expected_original, (2, 128, 4)))
    expected_residual = m.project_simplex(np.array([.8, .1, .05, .05])+expected_original-np.array([.2, .4, .2, .2]))
    np.testing.assert_allclose(out['arms']['AY-residual_recompose']['ratios'], np.broadcast_to(expected_residual, (2, 128, 4)), atol=1e-15)
    assert not out['arms']['AY-fixed_recompose']['unsupported_positive_energy_fraction'].any()


def test_annual_new_support_uses_flat_q_one_and_records_energy_fraction():
    fixture = list(inputs()); bank, baseline, draws, dates, rows = fixture
    bank['donor_mix'][:, 0] = 0
    bank['donor_mix'] /= bank['donor_mix'].sum(1)[:, None]
    bank['support'][:, 0] = False; bank['q'][:, :, 0] = 1
    bank['raw_cif'] = recompose_paths(bank['donor_mix'], bank['q'])
    fine, raw = [], []
    for day in range(2):
        ids = baseline['donor_ids'][day]
        ratio = bank['donor_mix'][ids][:, m.POS]/bank['donor_mix'][ids][:, m.POS].sum(1)[:, None]
        fd = np.zeros((128, 8)); fd[:, m.ZERO] = draws[day, :, 1:]; fd[:, m.POS] = draws[day, :, :1]*ratio
        fine.append(fd); raw.append(recompose_paths(fd, bank['q'][ids]))
    baseline['fine_draws'], baseline['raw'] = np.stack(fine), np.stack(raw)
    out = run(fixture)
    fixed = out['arms']['AY-fixed_recompose']
    assert np.all(fixed['fine_draws'][..., 0] > 0)
    assert np.all(fixed['unsupported_positive_energy_fraction'] > 0)
    for day in range(2):
        ids = baseline['donor_ids'][day]
        energy = fixed['fine_draws'][day][:, None, m.POS]*bank['q'][ids][:, :, m.POS]
        expected = energy[:, :, 0].sum(1)/energy.sum((1, 2))
        np.testing.assert_allclose(fixed['unsupported_positive_energy_fraction'][day], expected, atol=1e-15)
    assert not out['arms'][m.ARMS[0]]['unsupported_positive_energy_fraction'].any()


@pytest.mark.parametrize('borrow', [False, True])
def test_zero_query_pos_legal_with_year_or_without_source_support(borrow):
    fixture = inputs(zero_query=True, borrow=borrow)
    out = run(fixture)
    for arrays in out['arms'].values():
        np.testing.assert_array_equal(arrays['fine_draws'][..., m.POS], 0)
        np.testing.assert_array_equal(arrays['unsupported_positive_energy_fraction'], 0)
        assert np.isfinite(arrays['raw']).all()
    if borrow:
        assert out['donor_month_zero'].all()


def test_zero_total_physical_diagnostic_and_baseline_identity():
    fixture = inputs(zero_hour=True)
    out = run(fixture)
    for arm, arrays in out['arms'].items():
        assert arrays['zero_total_mask'][:, :, 0].all()
        assert np.isfinite(arrays['raw']).all()
        if 'anchor' not in arm:
            np.testing.assert_allclose(arrays['raw'][:, :, 0], arrays['fine_draws']@m.EF, atol=1e-12)


@pytest.mark.parametrize('defect', ['raw', 'fine', 'fallback', 'float_ids', 'held', 'unsupported_q'])
def test_invalid_or_misaligned_stream_fails(defect):
    fixture = list(inputs()); bank, baseline, draws, _, _ = fixture
    if defect == 'raw': baseline['raw'][0, 0, 0] += .01
    elif defect == 'fine': baseline['fine_draws'][0, 0, 0] += .01
    elif defect == 'fallback': baseline['fallback_ids'][0, 0] = 1
    elif defect == 'float_ids': baseline['donor_ids'] = baseline['donor_ids'].astype(float)
    elif defect == 'held': bank['regions'][0] = 'BPAT'
    else:
        bank['support'][0, 0] = False
        bank['donor_mix'][0, 0] = 0; bank['donor_mix'][0] /= bank['donor_mix'][0].sum()
    with pytest.raises(ValueError):
        run(fixture)


def test_simplex_projection_matches_independent_active_sets():
    rng = np.random.default_rng(2601)
    values = rng.normal(0, 3, (60, 4))
    result = m.project_simplex(values)
    for x, y in zip(values, result):
        active = y > 0
        shift = (x[active].sum()-1)/active.sum()
        np.testing.assert_allclose(y, np.maximum(x-shift, 0), atol=2e-15)
    np.testing.assert_allclose(result.sum(1), 1, atol=1e-15)
    assert np.all(result >= 0)


def test_mixed_stride_parent_original_baseline_is_returned_exactly():
    fixture = inputs(borrow=True)
    assert fixture[0]['q'].strides == (1536, 8, 192)
    out = run(fixture)
    np.testing.assert_array_equal(out['arms'][m.ARMS[0]]['raw'], fixture[1]['raw'])
    assert out['audit']['baseline_recomposition_max_abs_error'] == 0


@pytest.mark.parametrize('field', ['raw', 'fine_draws'])
def test_original_stream_one_ulp_corruption_fails_exact_gate(field):
    fixture = inputs()
    fixture[1][field][0, 0, 0] = np.nextafter(fixture[1][field][0, 0, 0], np.inf)
    with pytest.raises(ValueError, match='identity|identical MG'):
        run(fixture)
