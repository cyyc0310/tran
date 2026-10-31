"""Synthetic-only hourly level/shape bridge checks; no archive or truth reads."""
import numpy as np
import pytest

from scripts import hourly_bridge_v46 as model


def inputs():
    rng = np.random.default_rng(4601)
    mg = rng.dirichlet(np.ones(5), size=(6, 128))
    mg[:, :2] = [0., .25, .25, .25, .25]
    mean = rng.dirichlet(np.ones(4), size=(6, 128))
    ratio = rng.dirichlet(np.ones(4), size=(6, 128))
    fine = np.empty((6, 128, 8))
    fine[..., model.POS] = mg[..., :1] * ratio
    fine[..., model.ZERO] = mg[..., 1:]
    u = fine @ model.EF
    hour = np.sin(np.arange(24) * 2 * np.pi / 24)
    raw = u[..., None] + rng.normal(0, 50, (6, 128, 1)) + 130 * hour
    levels = mg[None, ..., 0] * rng.uniform(400, 1000, (45, 6, 128))
    return [raw, fine, mean, mg, levels, raw + 17, raw - 23]


def arm(result, name):
    return result['raw'][model.ARMS.index(name)]


def test_registry_and_information_budgets():
    assert len(model.LEVEL_ARMS) == 45 and len(set(model.LEVEL_ARMS)) == 45
    assert model.LEVEL_ARMS[:4] == ('Legacy-Size-Ridge', 'Legacy-Size-ET',
                                   'Legacy-SizeCap-Ridge', 'Legacy-SizeCap-ET')
    assert model.LEVEL_ARMS[12:15] == ('Capacity-OP', 'CoalOilUpper',
                                      'Mix-Router-DirectEFET-w0.25')
    assert model.LEVEL_ARMS[15] == 'Gain-SizeCapET-EF-Mean'
    assert model.LEVEL_ARMS[-1] == 'Gain-DirectEFET-Joint-HistoryForecastCap-ET'
    assert len(model.ARMS) == len(set(model.ARMS)) == 99
    assert tuple(model.BUDGET_BY_ARM) == model.ARMS
    assert list(model.BUDGET_BY_ARM.values()).count('B0') == 10
    assert list(model.BUDGET_BY_ARM.values()).count('B1') == 88
    assert model.BUDGET_BY_ARM['M8-reference'] == 'M8'
    assert all(model.BUDGET_BY_ARM[f'{mode}-{s}'] == 'B1'
               for mode in ('CenterOnly', 'FlatKeep') for s in model.CONTROLS)


def test_literal_references_immutability_and_reconstruction():
    values = inputs()
    saved = [x.copy() for x in values]
    result = model.build(*values)
    assert result['raw'].shape == (99, 6, 128, 24)
    for i, source in enumerate((values[0], values[5], values[6])):
        np.testing.assert_array_equal(result['raw'][i], source)
    for a, b in zip(values, saved):
        np.testing.assert_array_equal(a, b)
    assert all(np.asarray(v).shape == (96, 6) for v in result['diagnostics'].values())
    assert all(v <= 1e-9 for v in result['audit']['invariant_max_abs'].values())
    assert result['raw'].min() < 0  # the pure bridge must not clip
    result['raw'][0, 0, 0, 0] = 99999
    np.testing.assert_array_equal(values[0], saved[0])


def test_replace_removes_and_keep_retains_inherited_level_variation():
    values = inputs()
    values[3][...] = [1., 0., 0., 0., 0.]
    values[2][...] = [.25, .25, .25, .25]
    t = np.linspace(0, 1, 128)
    values[1][...] = 0
    values[1][..., 0] = t
    values[1][..., 1] = 1 - t
    u = values[1] @ model.EF
    shape = np.linspace(-30, 30, 24)
    values[0] = u[..., None] + shape
    values[4][...] = 650
    result = model.build(*values)
    replace = arm(result, 'Replace-CoalOilUpper')
    keep = arm(result, 'Keep-CoalOilUpper')
    np.testing.assert_allclose(replace, np.broadcast_to(650 + shape, replace.shape), atol=1e-12)
    assert np.std(keep.mean(-1), axis=1).min() > 100
    np.testing.assert_allclose(keep - replace,
                              np.broadcast_to((u - 650)[..., None], keep.shape), atol=1e-12)


def test_center_matches_keep_mean_but_not_spread_and_flat_preserves_daily_distribution():
    values = inputs()
    result = model.build(*values)
    for level in model.CONTROLS:
        keep = arm(result, 'Keep-' + level)
        center = arm(result, 'CenterOnly-' + level)
        flat = arm(result, 'FlatKeep-' + level)
        np.testing.assert_allclose(center.mean(1), keep.mean(1), atol=1e-10)
        np.testing.assert_allclose(flat.mean(-1), keep.mean(-1), atol=1e-10)
        assert np.max(abs(center.std(1) - keep.std(1))) > 1
        assert np.max(flat.std(-1)) < 1e-10
        np.testing.assert_allclose(flat.mean(-1).std(1), keep.mean(-1).std(1), atol=1e-10)


def test_single_path_shape_preservation_does_not_preserve_quantile_shape():
    values = inputs()
    values[3][...] = [1., 0., 0., 0., 0.]
    values[2][...] = [.25, .25, .25, .25]
    values[1][...] = 0
    values[1][..., model.POS] = .25
    h = np.linspace(-80, 80, 24)
    values[0] = np.broadcast_to(650 + np.r_[np.ones(64), -np.ones(64)][None, :, None] * h,
                                (6, 128, 24)).copy()
    values[4][...] = 650
    values[4][..., :64] = 450
    values[4][..., 64:] = 850
    result = model.build(*values)
    new = arm(result, 'Replace-CoalOilUpper')
    np.testing.assert_allclose(new - new.mean(-1, keepdims=True),
                               values[0] - values[0].mean(-1, keepdims=True), atol=1e-10)
    oldq, newq = (np.quantile(v, .05, axis=1) for v in (values[0], new))
    assert np.max(abs((oldq - oldq.mean(-1, keepdims=True)) -
                      (newq - newq.mean(-1, keepdims=True)))) > 50


def test_zero_pos_counts_and_diagnostic_means_are_not_repaired():
    values = inputs()
    values[0][:, :2] = 12
    result = model.build(*values)
    d = result['diagnostics']
    index = model.ARMS[3:].index('Replace-CoalOilUpper')
    assert np.all(d['zero_mg_pos_draw_count'][index] == 2)
    assert np.all(d['zero_mg_pos_nonzero_raw_hour_count'][index] == 48)
    assert np.all(d['zero_mg_pos_positive_clipped_hour_count'][index] == 48)
    np.testing.assert_allclose(d['new_level_mean'][index], values[4][13].mean(-1))
    np.testing.assert_allclose(d['raw_hour_mean'][index],
                              arm(result, 'Replace-CoalOilUpper').mean((1, 2)))
    np.testing.assert_allclose(d['inherited_level_residual_mean'][index],
                              (result['old_scale_level'] - result['old_point_level']).mean(-1))


@pytest.mark.parametrize('index', range(7))
@pytest.mark.parametrize('poison', [np.nan, np.inf, 1j])
def test_nonfinite_and_complex_fail(index, poison):
    values = inputs()
    values[index] = values[index].astype(complex if poison == 1j else float)
    values[index].flat[0] = poison
    with pytest.raises(ValueError):
        model.build(*values)


@pytest.mark.parametrize('index', range(7))
def test_shapes_fail(index):
    values = inputs()
    values[index] = values[index][:-1]
    with pytest.raises(ValueError):
        model.build(*values)


@pytest.mark.parametrize('kind', ['fine_negative', 'mean_sum', 'mg_sum', 'mg_zero_mismatch',
                                  'mg_pos_mismatch', 'level_low', 'level_high'])
def test_domain_and_composition_alignment_fail(kind):
    values = inputs()
    if kind == 'fine_negative':
        values[1][0, 4, 0] = -.01
    elif kind == 'mean_sum':
        values[2][0, 4] *= .5
    elif kind == 'mg_sum':
        values[3][0, 4] *= .5
    elif kind == 'mg_zero_mismatch':
        values[1][0, 4, [3, 4]] = values[1][0, 4, [4, 3]]
    elif kind == 'mg_pos_mismatch':
        values[1][0, 4, 0] += .01
        values[1][0, 4, 3] -= .01
    else:
        values[4][0, 0, 4] = values[3][0, 4, 0] * (399 if kind == 'level_low' else 1001)
    with pytest.raises(ValueError):
        model.build(*values)
