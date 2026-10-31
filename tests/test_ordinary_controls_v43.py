"""Synthetic endpoint, routing and particle alignment tests; zero fitting."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import ordinary_controls_v43 as m
from scripts import capacity_routing_v36 as old


EXPECTED_PAIRS = (
    ('Pool-SizeCapET', 'Legacy-SizeCap-ET', 'Expanded-SizeCap-ET'),
    ('Pool-DirectEFET', 'Legacy-DirectEF-ET', 'Expanded-DirectEF-ET'),
    ('Capacity-SizeCapET', 'Expanded-Size-ET', 'Expanded-SizeCap-ET'),
    ('Capacity-DirectEFET', 'Expanded-Size-ET', 'Expanded-DirectEF-ET'),
    ('StrongB0-SizeCapET', 'Legacy-Size-Ridge', 'Expanded-SizeCap-ET'),
    ('StrongB0-DirectEFET', 'Legacy-Size-Ridge', 'Expanded-DirectEF-ET'),
    ('Anchor-SizeCapET', 'Capacity-OP', 'Expanded-SizeCap-ET'),
    ('Anchor-DirectEFET', 'Capacity-OP', 'Expanded-DirectEF-ET'),
    ('Router-SizeCapET', 'CoalOilUpper', 'Expanded-SizeCap-ET'),
    ('Router-DirectEFET', 'CoalOilUpper', 'Expanded-DirectEF-ET'))


def arrays():
    pos = np.linspace(.1, .8, 128)
    mg = np.tile(np.c_[pos, np.repeat(((1-pos)/4)[:, None], 4, axis=1)][None], (6, 1, 1))
    ef = np.stack([np.tile(420+20*i+np.linspace(0, 5, 128), (6, 1)) for i in range(12)])
    anchor = np.full((6, 128), 700.)
    q = np.array([[.2, .4, .1, .3], [.4, .3, .2, .1], [.3, .2, .3, .2]])
    query = np.array([[.3, .3, .2, .2], [.45, .25, .1, .2], [.2, .2, .35, .25],
                      [.45, .1, .35, .1], [.25, .45, .15, .15], [.15, .35, .15, .35]])
    return ef, ef*mg[None, ..., 0], mg, anchor, q, np.vstack([q, query]), query


def test_exact_pair_order_weight_direction_and_particle_formulas():
    args = arrays(); result = m.build(*args)
    assert m.PAIRS == EXPECTED_PAIRS and m.WEIGHTS == (.25, .5, .75)
    assert m.PARENT_ARMS == tuple(f'{scope}-{family}-{model}' for scope in ('Legacy', 'Expanded')
        for family in ('Size', 'SizeCap', 'DirectEF') for model in ('Ridge', 'ET'))
    assert m.ARMS == tuple(f'Mix-{p}-w{w}' for p, _, _ in EXPECTED_PAIRS for w in m.WEIGHTS)+tuple(
        'ExpandedRange-'+r for r in old.RECIPES[5:])
    assert result['EF_draws'].shape == result['CIF_draws'].shape == (35, 6, 128)
    assert result['weights'].shape == (35, 6)
    assert len(result['audit']['recipes']) == 35 and len(result['endpoints']) == 14
    for pair, (_, left, right) in enumerate(EXPECTED_PAIRS):
        for wi, w in enumerate(m.WEIGHTS):
            i = 3*pair+wi
            expected = (1-w)*result['endpoint_EF'][left]+w*result['endpoint_EF'][right]
            np.testing.assert_array_equal(result['EF_draws'][i], expected)
            np.testing.assert_array_equal(result['weights'][i], w)
    np.testing.assert_array_equal(result['CIF_draws'][:30], result['EF_draws'][:30]*args[2][None, ..., 0])


def test_original_router_and_new_ranges_reuse_frozen_rules_with_exact_endpoints():
    args = list(arrays())
    # A tolerated archived arithmetic difference must survive binary endpoint copies.
    args[1][0, 0, 0] += 1e-10
    result = m.build(*args)
    old_w = old.weights(args[4], args[6]); new_w = old.weights(args[5], args[6])
    base = result['endpoint_EF']['Legacy-Size-Ridge']; base_cif = args[1][0]
    anchor_cif = args[3]*args[2][..., 0]
    np.testing.assert_array_equal(result['endpoint_EF']['CoalOilUpper'], np.where(old_w[8, :, None] == 1, args[3], base))
    np.testing.assert_array_equal(result['endpoint_CIF']['CoalOilUpper'], np.where(old_w[8, :, None] == 1, anchor_cif, base_cif))
    np.testing.assert_array_equal(result['weights'][30:], new_w[5:])
    assert np.any(old_w[5:] != new_w[5:])
    for i, w in enumerate(new_w[5:], 30):
        np.testing.assert_array_equal(result['EF_draws'][i], np.where(w[:, None] == 1, args[3], base))
        np.testing.assert_array_equal(result['CIF_draws'][i], np.where(w[:, None] == 1, anchor_cif, base_cif))
    for i, name in enumerate(m.PARENT_ARMS):
        np.testing.assert_array_equal(result['endpoint_EF'][name], args[0][i])
        np.testing.assert_array_equal(result['endpoint_CIF'][name], args[1][i])


def test_median_after_mixing_is_not_a_mixture_of_medians():
    args = list(arrays())
    pos = np.r_[np.repeat(.2, 50), np.repeat(.4, 50), np.repeat(.8, 28)]
    args[2] = np.tile(np.c_[pos, np.repeat(((1-pos)/4)[:, None], 4, axis=1)][None], (6, 1, 1))
    args[0][0] = np.tile(np.r_[np.repeat(1000., 50), np.repeat(400., 78)], (6, 1))
    args[0][m.PARENT_ARMS.index('Expanded-SizeCap-ET')] = 600.
    args[1] = args[0]*args[2][None, ..., 0]
    result = m.build(*args)
    i = m.ARMS.index('Mix-StrongB0-SizeCapET-w0.5')
    assert np.median(result['CIF_draws'][i, 0]) == 200
    right = m.PARENT_ARMS.index('Expanded-SizeCap-ET')
    wrong = .5*np.median(args[1][0, 0])+.5*np.median(args[1][right, 0])
    assert wrong == 220


def test_inputs_are_unchanged_and_endpoints_are_copies():
    args = arrays(); before = deepcopy(args); result = m.build(*args)
    for a, b in zip(args, before): np.testing.assert_array_equal(a, b)
    result['endpoint_EF']['Legacy-Size-Ridge'][0, 0] = 0
    np.testing.assert_array_equal(args[0], before[0])


def test_expanded_ranges_are_monotone_and_both_binary_branches_copy_exactly():
    args = list(arrays()); args[5] = np.vstack([args[4], args[6][1:2]])
    result = m.build(*args)
    old_w = result['audit']['old_weights'][5:]
    new_w = result['weights'][30:]
    assert (new_w <= old_w).all() and (new_w < old_w).any()
    assert {0., 1.} == set(np.unique(new_w))
    for i, w in enumerate(new_w, 30):
        np.testing.assert_array_equal(result['EF_draws'][i, w == 0], args[0][0, w == 0])
        np.testing.assert_array_equal(result['CIF_draws'][i, w == 0], args[1][0, w == 0])
        np.testing.assert_array_equal(result['EF_draws'][i, w == 1], args[3][w == 1])
        np.testing.assert_array_equal(result['CIF_draws'][i, w == 1], (args[3]*args[2][..., 0])[w == 1])


def test_bound_tolerance_validates_without_clipping_any_endpoint():
    args = list(arrays()); args[0][0, 0, 0] = 400.-5e-10
    args[3][:] = 1000.+5e-10; args[1] = args[0]*args[2][None, ..., 0]
    result = m.build(*args)
    assert result['endpoint_EF']['Legacy-Size-Ridge'][0, 0] < 400.
    assert result['endpoint_EF']['Capacity-OP'][0, 0] > 1000.


@pytest.mark.parametrize('index,kind', [(0, 'shape'), (0, 'complex'), (0, 'nan'), (0, 'low'),
    (0, 'high'), (1, 'alignment'), (1, 'permutation'), (2, 'negative'), (2, 'sum'),
    (2, 'shape'), (3, 'vary'), (3, 'high'), (4, 'empty'), (4, 'shape'),
    (4, 'negative'), (5, 'sum'), (6, 'shape'), (6, 'complex')])
def test_invalid_inputs_fail_closed(index, kind):
    args = list(arrays())
    if kind == 'shape': args[index] = args[index][:-1] if index not in (4, 5) else args[index][:, :3]
    elif kind == 'empty': args[index] = args[index][:0]
    elif kind == 'permutation': args[index] = args[index][..., ::-1]
    elif kind == 'vary': args[index][0, 0] += .1
    else:
        if kind == 'complex': args[index] = args[index].astype(complex)
        args[index].flat[0] = {'complex': 1j, 'nan': np.nan, 'low': 399., 'high': 1001.,
                              'alignment': 1., 'negative': -.1, 'sum': .99}[kind]
    with pytest.raises(ValueError): m.build(*args)
