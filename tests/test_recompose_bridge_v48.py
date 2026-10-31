"""Synthetic formula and budget checks for the alternate hourly base."""
import numpy as np
import pytest
from scripts import recompose_bridge_v48 as m


@pytest.fixture(scope='module')
def inputs():
    rng = np.random.default_rng(48)
    mg = rng.dirichlet(np.ones(5), size=(181, 128))
    ratio = rng.dirichlet(np.ones(4), size=(181, 128))
    fine = np.zeros((181, 128, 8))
    fine[..., m.prior.ZERO] = mg[..., 1:]
    fine[..., m.prior.POS] = mg[..., :1] * ratio
    levels = np.tile((fine @ m.EF)[None], (15, 1, 1))
    r = rng.uniform(-20, 800, (181, 128, 24))
    return dict(recompose_raw=r, anchor_raw=r*.6+10, recompose_fine=fine, anchor_fine=fine.copy(),
                query_mean=ratio, MG=mg, levels=levels)


def test_registry_budgets_and_only_new_raw(inputs):
    assert len(m.ARMS) == 138 and len(set(m.ARMS)) == 138
    assert len(m.NEW_ARMS) == 75 and len(m.LEVEL_ARMS) == 15
    assert {b: list(m.BUDGET_BY_ARM.values()).count(b) for b in ('B0', 'B1', 'M8')} == {'B0': 38, 'B1': 99, 'M8': 1}
    result = m.build(**inputs)
    assert result['raw'].shape == (75, 181, 128, 24)
    assert result['audit']['diagnostic_arms'] == list(m.NEW_ARMS)
    assert result['audit']['temporary_reference_slots_discarded']
    r, s = inputs['recompose_raw'], inputs['levels']
    u = inputs['recompose_fine'] @ m.EF
    p = inputs['MG'][..., 0] * (inputs['query_mean'] @ m.EF[m.prior.POS])
    for i in range(15):
        keep = r - p[..., None] + s[i, ..., None]
        assert np.allclose(result['raw'][i], r - u[..., None] + s[i, ..., None], atol=1e-9, rtol=0)
        assert np.allclose(result['raw'][15+i], keep, atol=1e-9, rtol=0)
        assert np.allclose(result['raw'][30+i], r + (s[i]-p).mean(-1)[:, None, None], atol=1e-9, rtol=0)
        assert np.allclose(result['raw'][45+i], np.repeat(keep.mean(-1, keepdims=True), 24, -1), atol=1e-9, rtol=0)
        old_keep = inputs['anchor_raw'] - p[..., None] + s[i, ..., None]
        aligned = result['raw'][60+i]
        assert np.allclose(aligned.mean(-1), old_keep.mean(-1), atol=1e-9, rtol=0)
        assert np.allclose(aligned-aligned.mean(-1, keepdims=True), r-r.mean(-1, keepdims=True), atol=1e-9, rtol=0)
    assert max(result['audit']['invariant_max_abs'].values()) <= 1e-9
    assert all(v.shape == (75, 181) for v in result['diagnostics'].values())


def test_pairing_rejected_before_path_construction(inputs, monkeypatch):
    bad = inputs['anchor_fine'].copy(); bad[0, 0, 0] += .01
    monkeypatch.setattr(m.prior, 'build_paths', lambda *_: pytest.fail('constructed paths before pairing check'))
    with pytest.raises(ValueError, match='pairing'):
        m.build(**(inputs | {'anchor_fine': bad}))
