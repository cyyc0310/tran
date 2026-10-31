"""Synthetic fixed recipe and particle-mixing tests; no data or fitting."""
import numpy as np
import pytest

from scripts import capacity_routing_v36 as m


def compositions():
    train = np.array([[.2, .4, .1, .3], [.4, .3, .2, .1], [.3, .2, .3, .2]])
    query = np.array([[.3, .3, .2, .2], [.45, .25, .1, .2], [.2, .2, .35, .25],
                      [.45, .1, .35, .1], [.25, .45, .15, .15], [.15, .35, .15, .35]])
    return train, query


def test_ten_fixed_recipes_distinguish_direction_scalar_and_full_range():
    train, query = compositions()
    result = m.weights(train, query)
    assert m.RECIPES == ('Global0', 'Global0.25', 'Global0.5', 'Global0.75', 'Global1',
                         'Range4', 'CoalUpper', 'OilUpper', 'CoalOilUpper', 'AnchorEFUpper')
    assert result.shape == (10, 6)
    np.testing.assert_array_equal(result[:5], np.repeat(np.arange(5)[:, None]/4, 6, axis=1))
    np.testing.assert_array_equal(result[5:], [[0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0]])
    assert set(np.unique(result[5:])) <= {0., 1.}


def test_extrema_tolerance_is_fixed_and_comparisons_are_strict():
    train, _ = compositions(); query = np.repeat(train[1:2], 6, axis=0)
    for j, delta in enumerate([0., .5e-12, 1e-12, 2e-12, -1e-12, -2e-12]):
        query[j, 0] += delta; query[j, 1] -= delta
    result = m.weights(train, query)
    np.testing.assert_array_equal(result[6], [0, 0, 0, 1, 0, 0])


def test_training_duplicates_order_and_query_order_do_not_change_rules():
    train, query = compositions(); original = (train.copy(), query.copy())
    expected = m.weights(train, query)
    np.testing.assert_array_equal(expected, m.weights(np.vstack([train[::-1], train]), query))
    np.testing.assert_array_equal(expected[:, ::-1], m.weights(train, query[::-1]))
    np.testing.assert_array_equal(train, original[0]); np.testing.assert_array_equal(query, original[1])


@pytest.mark.parametrize('which,kind', [('train', 'empty'), ('train', 'shape'), ('train', 'nan'),
                                     ('query', 'shape'), ('query', 'negative'), ('query', 'sum'), ('query', 'inf')])
def test_invalid_compositions_are_rejected(which, kind):
    train, query = compositions(); value = train.copy() if which == 'train' else query.copy()
    if kind == 'empty': value = value[:0]
    elif kind == 'shape': value = value[:, :3] if which == 'train' else value[:5]
    elif kind == 'sum': value[0, 0] += .1
    else: value[0, 0] = {'nan': np.nan, 'negative': -.1, 'inf': np.inf}[kind]
    with pytest.raises(ValueError):
        m.weights(value if which == 'train' else train, value if which == 'query' else query)


def arrays():
    pos = np.linspace(.1, .8, 128)
    mg = np.tile(np.c_[pos, np.repeat(((1-pos)/4)[:, None], 4, axis=1)][None], (6, 1, 1))
    base = np.tile(np.linspace(450, 750, 128), (6, 1)); anchor = np.full((6, 128), 600.)
    return m.weights(*compositions()), base, anchor, mg


def test_particle_blend_formula_and_every_binary_endpoint_is_exact():
    w, base, anchor, mg = arrays(); original = [x.copy() for x in (w, base, anchor, mg)]
    result = m.blend(w, base, anchor, mg)
    assert set(result) == {'EF_draws', 'CIF_draws'}
    assert result['EF_draws'].shape == result['CIF_draws'].shape == (10, 6, 128)
    np.testing.assert_allclose(result['EF_draws'], (1-w[..., None])*base+w[..., None]*anchor)
    np.testing.assert_array_equal(result['CIF_draws'], result['EF_draws']*mg[None, ..., 0])
    for i in range(10):
        np.testing.assert_array_equal(result['EF_draws'][i, w[i] == 0], base[w[i] == 0])
        np.testing.assert_array_equal(result['EF_draws'][i, w[i] == 1], anchor[w[i] == 1])
    for a, b in zip((w, base, anchor, mg), original): np.testing.assert_array_equal(a, b)


def test_median_is_taken_after_mixing_not_mixed_from_point_medians():
    pos = np.r_[np.repeat(.2, 50), np.repeat(.4, 50), np.repeat(.8, 28)]
    mg = np.tile(np.c_[pos, np.repeat(((1-pos)/4)[:, None], 4, axis=1)][None], (6, 1, 1))
    base = np.tile(np.r_[np.repeat(1000., 50), np.repeat(400., 78)], (6, 1))
    anchor = np.full((6, 128), 600.)
    result = m.blend(np.full((10, 6), .5), base, anchor, mg)
    actual = np.median(result['CIF_draws'][0, 0])
    wrong = .5*np.median(base[0]*pos)+.5*np.median(anchor[0]*pos)
    assert actual == 200 and wrong == 220


@pytest.mark.parametrize('which,kind', [('weight', 'shape'), ('weight', 'negative'), ('weight', 'above'),
                                     ('base', 'nan'), ('anchor', 'shape'), ('anchor', 'vary'),
                                     ('mg', 'shape'), ('mg', 'negative'), ('mg', 'sum')])
def test_blend_rejects_bad_shapes_values_and_nonconstant_anchor(which, kind):
    w, base, anchor, mg = arrays(); values = dict(weight=w, base=base, anchor=anchor, mg=mg)
    value = values[which].copy()
    if kind == 'shape': value = value[:-1]
    elif kind == 'vary': value[0, 0] += 1
    else: value.flat[0] = {'negative': -.1, 'above': 1.1, 'nan': np.nan, 'sum': .99}[kind]
    values[which] = value
    with pytest.raises(ValueError): m.blend(values['weight'], values['base'], values['anchor'], values['mg'])
