"""Synthetic contribution identities and source-level persistence summaries."""
import numpy as np
import pytest

from scripts import residual_persistence_v39 as m


def data():
    c = np.array([[20., 40., 0., 10.], [10., 30., 0., 20.]])
    g = np.array([[12., 20., 0., 4.], [2., 15., 0., 8.]])
    u = np.array([[.4, .4, 3., .2], [.5, .3, 2., .3]])
    return g, c, u


def test_carbon_identity_true_minus_pred_and_contribution_definition():
    g, c, u = data(); copies = [a.copy() for a in (g, c, u)]
    r = m.residuals(g, c, u)
    np.testing.assert_allclose(r['x'], g/c.sum(1)[:, None])
    np.testing.assert_allclose(r['xhat'], (c/c.sum(1)[:, None])*u)
    np.testing.assert_allclose(r['r'], r['x']-r['xhat'])
    np.testing.assert_allclose(r['d'], r['r'].sum(1))
    np.testing.assert_allclose(r['k'], 600*r['r'][:, 0]+400*r['r'][:, 2])
    ef_true = (g/g.sum(1)[:, None])@m.EF
    predicted = c*u
    ef_pred = (predicted/predicted.sum(1)[:, None])@m.EF
    np.testing.assert_allclose(r['EF_true'], ef_true)
    np.testing.assert_allclose(r['EF_pred'], ef_pred)
    np.testing.assert_allclose(r['EF_error_true_minus_pred'], ef_true-ef_pred)
    np.testing.assert_allclose(r['identity_rhs'], ef_true-ef_pred, atol=1e-12)
    assert not np.allclose(r['identity_rhs'], -(ef_true-ef_pred))
    # Oil has no defined G/C_oil rate, but its capacity-normalized contribution is valid zero.
    np.testing.assert_array_equal(r['x'][:, 2], 0)
    np.testing.assert_array_equal(r['xhat'][:, 2], 0)
    for a, copy in zip((g, c, u), copies): np.testing.assert_array_equal(a, copy)


def test_common_generation_scale_and_joint_mw_scale_properties():
    g, c, u = data(); r = m.residuals(g, c, u)
    scaled = m.residuals(100*g, 100*c, u)
    for key in r: np.testing.assert_allclose(r[key], scaled[key], atol=1e-12)
    exact = m.residuals(c*u, c, u)
    for key in ('r', 'd', 'k', 'EF_error_true_minus_pred', 'identity_rhs'):
        np.testing.assert_allclose(exact[key], 0, atol=1e-12)


def test_source_rates_above_one_are_not_clipped():
    g, c, u = data(); u[0, 0] = 2.
    result = m.residuals(g, c, u)
    assert result['xhat'][0, 0] == pytest.approx(2*c[0, 0]/c[0].sum())
    assert result['xhat'][0, 0] > c[0, 0]/c[0].sum()


@pytest.mark.parametrize('kind', ['cap_zero', 'true_zero', 'pred_zero', 'cap_conflict'])
def test_undefined_denominators_and_zero_capacity_contradiction_fail(kind):
    g, c, u = data()
    if kind == 'cap_zero': c[0] = 0; g[0] = 0
    elif kind == 'true_zero': g[0] = 0
    elif kind == 'pred_zero': u[0] = 0
    else: g[0, 2] = 1
    with pytest.raises(ValueError): m.residuals(g, c, u)


@pytest.mark.parametrize('which,kind', [(0, 'negative'), (1, 'negative'), (2, 'negative'),
    (0, 'complex'), (1, 'nan'), (2, 'inf'), (0, 'shape'), (1, 'shape'), (2, 'shape')])
def test_residual_invalid_inputs(which, kind):
    values = list(data())
    if kind == 'shape': values[which] = values[which][:, :3]
    else:
        if kind == 'complex': values[which] = values[which].astype(complex)
        values[which].flat[0] = {'negative': -1, 'complex': 1j, 'nan': np.nan, 'inf': np.inf}[kind]
    with pytest.raises(ValueError): m.residuals(*values)


def records():
    rows = []
    for outer in m.SOURCES:
        for index, source in enumerate(s for s in m.SOURCES if s != outer):
            early = float(index-2)
            for period, day, ef, d in [('early', '2023-06-01', early, early),
                                       ('late', '2023-07-01', 2*early+1, -early)]:
                rows.append(dict(outer=outer, source=source, key=f'{source}-{period}',
                    period_start=day, d=d, k=3*ef, EF_error_true_minus_pred=ef))
    return rows


def test_pairs_are_source_means_and_outer_points_not_pooled_months():
    rows = records(); rows.append(dict(rows[0], key='extra', EF_error_true_minus_pred=4.))
    result = m.summarize_pairs(rows)
    assert len(result['pairs']) == 42 and len(result['outers']) == 7
    pair = result['pairs'][0]
    assert pair['outer'] == m.SOURCES[0] and pair['source'] == m.SOURCES[1]
    assert pair['early_count'] == 2 and pair['late_count'] == 1
    assert pair['early_EFbias'] == 1 and pair['late_EFbias'] == -3
    other = result['outers'][1]
    assert other['EFbias_pearson'] == pytest.approx(1.)
    assert other['early_d_late_EFbias_pearson'] == pytest.approx(1.)
    assert other['EFbias_pearson_status'] == 'defined'
    assert other['EFbias_pearson_n'] == 6
    assert not other['EFbias_pearson_x_constant'] and not other['EFbias_pearson_y_constant']
    early_scale = np.linalg.norm(np.arange(6)-np.arange(6).mean())
    assert other['EFbias_pearson_x_centered_l2'] == pytest.approx(early_scale)
    assert other['EFbias_pearson_y_centered_l2'] == pytest.approx(2*early_scale)
    assert other['same_sign_count'] == 5 and other['zero_pairs'] == 1
    early = np.arange(6)-2; late = 2*early+1
    assert other['carry_EF_MAE'] == pytest.approx(np.mean(np.abs(late-early)))
    assert other['zero_EF_MAE'] == pytest.approx(np.mean(np.abs(late)))
    reverse = m.summarize_pairs(rows[::-1])
    assert reverse == result


def test_constant_correlations_are_null_and_zero_signs_are_explicit():
    rows = records()
    for row in rows:
        row['EF_error_true_minus_pred'] = 0.
        row['d'] = 1.
    result = m.summarize_pairs(rows)
    for outer in result['outers']:
        assert outer['EFbias_pearson'] is None
        assert outer['early_d_late_EFbias_pearson'] is None
        assert outer['EFbias_pearson_status'] == 'undefined_constant'
        assert outer['early_d_late_EFbias_pearson_status'] == 'undefined_constant'
        assert outer['same_sign_count'] == 0 and outer['zero_pairs'] == 6
        assert outer['EFbias_pearson_x_constant'] and outer['EFbias_pearson_y_constant']
        assert outer['EFbias_pearson_x_centered_l2'] == outer['EFbias_pearson_y_centered_l2'] == 0
        assert outer['carry_EF_MAE'] == outer['zero_EF_MAE'] == 0


def test_one_constant_half_is_undefined_and_monthly_inputs_remain_unchanged():
    rows = records()
    for row in rows:
        if row['period_start'] == '2023-07-01': row['EF_error_true_minus_pred'] = 4.
    copies = [row.copy() for row in rows]
    result = m.summarize_pairs(rows)
    assert rows == copies
    for outer in result['outers']:
        assert outer['EFbias_pearson'] is None
        assert outer['early_d_late_EFbias_pearson'] is None


def test_timezone_split_uses_utc_and_signed_association_is_retained():
    rows = records()
    for row in rows:
        if row['period_start'] == '2023-06-01':
            row['period_start'] = '2023-07-01T00:30:00+01:00'  # UTC still early.
            row['d'] *= -1
    result = m.summarize_pairs(rows)
    assert all(o['early_d_late_EFbias_pearson'] == pytest.approx(-1.) for o in result['outers'])


@pytest.mark.parametrize('kind', ['missing_outer', 'missing_source', 'missing_half', 'duplicate',
    'self_source', 'nan', 'complex', 'naive', 'missing_field'])
def test_summary_rejects_incomplete_or_invalid_inputs(kind):
    rows = records()
    if kind == 'missing_outer': rows = [r for r in rows if r['outer'] != m.SOURCES[0]]
    elif kind == 'missing_source': rows = rows[2:]
    elif kind == 'missing_half': rows = rows[1:]
    elif kind == 'duplicate': rows.append(rows[0].copy())
    elif kind == 'self_source': rows[0]['source'] = rows[0]['outer']
    elif kind == 'nan': rows[0]['d'] = np.nan
    elif kind == 'complex': rows[0]['k'] = 1j
    elif kind == 'naive': rows[0]['period_start'] = '2023-06-01T00:00:00'
    else: del rows[0]['d']
    with pytest.raises(ValueError): m.summarize_pairs(rows)
