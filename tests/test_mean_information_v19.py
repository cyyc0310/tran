"""Analytic cases for the source historical mean-identification diagnostic."""
from pathlib import Path
import sys
import json

import numpy as np
import pytest
from scipy.optimize import linprog

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_mean_information_v19 as model


def test_uniform_generation_identifies_mean_but_not_tail():
    flat = model.block_bounds(np.ones(6), np.full(6, 500.))
    spiky = model.block_bounds(np.ones(6), np.array([1000., 1000., 1000., 0., 0., 0.]))
    assert flat == spiky
    assert flat['quadratic_cv_lo'] == pytest.approx(500., abs=1e-8)
    assert flat['quadratic_cv_hi'] == pytest.approx(500., abs=1e-8)
    assert flat['full_g_lo'] == pytest.approx(500., abs=1e-8)
    assert flat['full_g_hi'] == pytest.approx(500., abs=1e-8)


def test_fractional_knapsack_matches_independent_lp():
    rng = np.random.default_rng(1926)
    for n in (6, 24):
        for _ in range(20):
            g = rng.uniform(.01, 20, n)
            mu = rng.uniform(0, 1000)
            actual = model.full_g_mean_bounds(g, mu)
            weights = g/g.sum()
            low = linprog(np.ones(n)/n, A_eq=weights[None], b_eq=[mu], bounds=(0, 1000), method='highs')
            high = linprog(-np.ones(n)/n, A_eq=weights[None], b_eq=[mu], bounds=(0, 1000), method='highs')
            assert low.success and high.success
            np.testing.assert_allclose(actual, [low.fun, -high.fun], rtol=0, atol=1e-8)


@pytest.mark.parametrize('mu', [0., 1000.])
def test_supported_endpoint_requires_constant_carbon(mu):
    for g in (np.arange(1., 7.), *(np.random.default_rng(seed).lognormal(0, 8, 24) for seed in (0, 1))):
        result = model.block_bounds(g, np.full(len(g), mu))
        for key in ('finite', 'simple_cv', 'quadratic_cv', 'full_g'):
            assert result[key+'_lo'] == pytest.approx(mu)
            assert result[key+'_hi'] == pytest.approx(mu)


def test_all_nested_bounds_contain_truth_on_unequal_generation():
    rng = np.random.default_rng(1927)
    for _ in range(100):
        g = rng.lognormal(0, 3, 24)
        c = rng.uniform(0, 1000, 24)
        result = model.block_bounds(g, c)
        assert result['mu_equal'] == pytest.approx(c.mean())
        assert result['weighted_minus_equal'] == pytest.approx(np.average(c, weights=g)-c.mean())


def test_invalid_information_rejected():
    with pytest.raises(ValueError):
        model.full_g_mean_bounds(np.array([0., 1.]), 500.)
    with pytest.raises(ValueError):
        model.block_bounds(np.ones(6), np.ones(5))


def test_rounded_energy_endpoint_is_not_an_exact_endpoint_certificate():
    result = model.block_bounds(np.array([2.**60, 1.]), np.array([1000., 0.]))
    assert result['mu_energy'] == 1000.
    assert result['mu_equal'] == 500.
    assert result['rounded_endpoint_without_certificate_count'] == 1
    assert result['mu_energy_lo'] < 1000.
    for key in ('finite', 'simple_cv', 'quadratic_cv', 'full_g'):
        assert result[key+'_lo'] <= 500.+1e-8
        assert result[key+'_hi'] >= 500.-1e-8


def test_check_rejects_retargeted_source_even_with_matching_rewritten_hashes(tmp_path, monkeypatch):
    parent = tmp_path/'parent'
    output = tmp_path/'output'
    original = tmp_path/'original'
    redirected = tmp_path/'redirected'
    for folder in (parent, output, original, redirected):
        folder.mkdir()
    (tmp_path/'script.py').write_text('synthetic code\n')
    for region in model.SOURCES:
        for folder in (original, redirected):
            (folder/f'{region}_hourly.csv').write_text('synthetic bytes only\n')
    source_hashes = {str(p): model.sha(p) for p in original.iterdir()}
    (parent/'freeze.json').write_text(json.dumps(dict(source=str(original), input_hashes=source_hashes)))
    (parent/'source_completed.json').write_text(json.dumps(dict(freeze_sha256=model.sha(parent/'freeze.json'),
            target_values_opened=False, unseen_values_opened=False)))
    monkeypatch.setattr(model, 'ROOT', tmp_path)
    monkeypatch.setattr(model, 'PARENT', parent)
    monkeypatch.setattr(model, 'OUT', output)
    monkeypatch.setattr(model, 'CODE', ('script.py',))
    model.prepare()
    model.check()
    frozen = json.loads((output/'freeze.json').read_text())
    frozen['source'] = str(redirected)
    for region in model.SOURCES:
        frozen['input_hashes'].pop(str(original/f'{region}_hourly.csv'))
        path = redirected/f'{region}_hourly.csv'
        frozen['input_hashes'][str(path)] = model.sha(path)
    (output/'freeze.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='V18 parent source identity'):
        model.check()
