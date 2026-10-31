"""DRAFT/PAUSED legacy branch: pure synthetic helpers and fail-closed stage guards."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scripts import native_anchor_v23 as m
from scripts import run_native_anchor_v23 as r
from scripts.local_sketch_v17 import PointModel
from scripts.native_center_v18 import build


def parent(n=3):
    rng = np.random.default_rng(23023);answer = {}
    for phase in ('train', 'query'):
        answer[phase + '_anchors'] = np.stack([np.full(n, 300. + i * 1e-13) for i in range(4)])
        answer[phase + '_x__native_only'] = rng.normal(size=(n, 54))
        for name, width in zip(m.INPUTS[1:], m.WIDTHS[1:]):
            answer[phase + '_x__' + name] = rng.normal(size=(n, width))
    return answer


def raw():
    return np.random.default_rng(23).uniform(0, 1000, (3, 128, 24))


def reproduction_fixture():
    n = raw();rng = np.random.default_rng(4);centers = rng.uniform(100, 900, (5, 3, 24))
    ids = np.arange(3 * 128).reshape(3, 128);references = {}
    for packet in ('T8', 'G5'):
        point = centers[m.INPUTS.index('packet_only_' + packet)]
        old = build(n, ids, point, np.full(3, 500.), np.full((3, 4), 500.) if packet == 'T8' else None, packet)
        references[packet] = dict(center=point.copy(), methods=old['methods'], alphas=old['alphas'], raw=old['raw'])
    return n, centers, m.deltas_from_centers(n, centers), references


def test_five_inputs_keep_native_anchor_and_each_packets_own_exact_anchor():
    p = parent();x, a = m.input_arrays(p, 'train')
    assert tuple(x) == m.INPUTS
    assert tuple(v.shape[1] for v in x.values()) == m.WIDTHS
    np.testing.assert_array_equal(x['native_plus_D2_anchor'][:, :54], p['train_x__native_only'])
    np.testing.assert_array_equal(x['native_plus_D2_anchor'][:, 54], p['train_anchors'][0])
    np.testing.assert_array_equal(a['packet_only_T8'], p['train_anchors'][2])
    np.testing.assert_array_equal(a['joint_G5'], p['train_anchors'][3])
    assert not np.array_equal(a['packet_only_T8'], a['native_plus_D2_anchor'])
    p['train_x__packet_only_T8'] = p['train_x__packet_only_T8'][:, :-1]
    with pytest.raises(ValueError, match='Five complete'):
        m.input_arrays(p, 'train')


def test_same_anchor_model_exactly_matches_frozen_v17_and_clips_absolute_point():
    rng = np.random.default_rng(230);x = rng.normal(size=(30, 7));y = rng.uniform(0, 1000, (30, 24))
    a = rng.uniform(0, 1000, 30);regions = np.array(['CISO'] * 10 + ['MISO'] * 20)
    days = pd.date_range('2024-01-25', periods=30, tz='UTC')
    fitted = m.AnchorModel().fit(x, y, a, regions, days, exclude='BPAT')
    old = PointModel('ET').fit(x, y, a, regions, days, exclude='BPAT')
    np.testing.assert_array_equal(fitted.predict(x[:3], a[:3]), old.predict(x[:3], a[:3]))
    np.testing.assert_array_equal(fitted.scaler.mean_, old.scaler.mean_)
    saturated = m.AnchorModel().fit(x, np.full((30, 24), 1000.), np.zeros(30), regions, days, exclude='BPAT')
    assert np.all(saturated.predict(x[:3], np.full(3, 500.)) == 1000)
    with pytest.raises(ValueError, match='held region'):
        m.AnchorModel().fit(x, y, a, regions, days, exclude='CISO')


def test_exact_v17_centers_and_all12_v18_paths_reproduced_with_shared_alpha0():
    n, centers, deltas, refs = reproduction_fixture()
    assert len(m.arms()) == 31 and len(set(m.arms())) == 31
    assert m.exact_reproduction(n, centers, deltas, refs) == dict(packet_centers=2, projection_alpha_paths=12)
    identity, point = m.reconstruct(n, deltas, m.arms()[0])
    np.testing.assert_array_equal(identity, n)
    np.testing.assert_array_equal(point, n.mean(1))
    assert not np.shares_memory(identity, n)


def test_reproduction_does_not_tolerate_even_one_ulp_center_or_path_change():
    n, centers, deltas, refs = reproduction_fixture()
    refs['T8']['center'][0, 0] = np.nextafter(refs['T8']['center'][0, 0], np.inf)
    with pytest.raises(ValueError, match='Exact V17'):
        m.exact_reproduction(n, centers, deltas, refs)
    n, centers, deltas, refs = reproduction_fixture()
    index = list(zip(refs['G5']['methods'], refs['G5']['alphas'])).index(('ET_full', 1.))
    refs['G5']['raw'][index, 0, 0, 0] = np.nextafter(refs['G5']['raw'][index, 0, 0, 0], np.inf)
    with pytest.raises(ValueError, match='Exact V18'):
        m.exact_reproduction(n, centers, deltas, refs)


def test_center_minus_native_is_not_native_offset_residual_model_prediction():
    n, centers, deltas, _ = reproduction_fixture()
    np.testing.assert_array_equal(deltas, centers - n.mean(1)[None])
    diagnostic = m.diagnostics(n, deltas)
    assert diagnostic['requested_center'].shape == (31, 3, 24)
    i = m.arms().index(('joint_T8', 'full', 1.))
    np.testing.assert_allclose(diagnostic['requested_center'][i], centers[m.INPUTS.index('joint_T8')], rtol=0, atol=1e-12)
    assert diagnostic['native_noise_max_abs_error'].max() < 1e-9


@pytest.mark.parametrize('digest', [None, 'x' * 64, 'A' * 64, '0' * 63])
def test_explicit_digest_required_before_any_parent_read(monkeypatch, digest):
    monkeypatch.setattr(r, 'read', lambda _: pytest.fail('Invalid digest reached parent reader'))
    with pytest.raises(ValueError, match='Explicit64hex'):
        r.parent_inputs(digest)


def test_absent_or_pending_parent_cannot_enter_training(tmp_path, monkeypatch):
    path = tmp_path / 'parent.json';monkeypatch.setattr(r, 'MANIFEST', path)
    with pytest.raises(ValueError, match='Completed audited'):
        r.parent_inputs('0' * 64)
    path.write_text(json.dumps({'numerical_audit_status': 'PENDING'}))
    monkeypatch.setattr(r.prior, 'check', lambda: pytest.fail('Pending parent reached training inputs'))
    with pytest.raises(ValueError, match='must be PASS'):
        r.parent_inputs(hashlib.sha256(path.read_bytes()).hexdigest())


def test_paused_partial_attempt_and_missing_completegrid_fail_before_fine_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path);monkeypatch.setattr(r, 'check', lambda: {})
    (tmp_path / 'freeze.json').write_text('{}');(tmp_path / 'predictions').mkdir()
    monkeypatch.setattr(r, 'parent_archive', lambda _: pytest.fail('Partial attempt reached parent arrays'))
    with pytest.raises(RuntimeError, match='DRAFT/PAUSED'):
        r.predict()
    seal = dict(entries=[], complete_grid=True, archive_count=7, raw_arm_count=31, curve_count=155,
                prediction_reads_any_new_fine=False, target_values_opened=False, exact_reproduction_required=True)
    monkeypatch.setattr(r, 'read', lambda _: seal)
    monkeypatch.setattr(r.prior.sketch, 'source_days', lambda *args: pytest.fail('Missing seal reached2025fine'))
    with pytest.raises(ValueError, match='Exact7 archives'):
        r.check_predictions(seal)


@pytest.mark.parametrize('stage', ['prepare', 'predict', 'score'])
def test_every_formal_entry_is_unconditionally_paused_before_any_io(monkeypatch, stage):
    def forbidden(*args, **kwargs):
        pytest.fail('Paused formal stage attempted I/O')
    for name in ('check', 'read', 'sha', 'parent_inputs', 'parent_archive'):
        monkeypatch.setattr(r, name, forbidden)
    monkeypatch.setattr(r.Path, 'open', forbidden)
    monkeypatch.setattr(r.np, 'load', forbidden)
    monkeypatch.setattr(r.pd, 'read_csv', forbidden)
    with pytest.raises(RuntimeError, match='public monthly/yearly fuel statistics only'):
        getattr(r, stage)(None) if stage == 'prepare' else getattr(r, stage)()


def metric_values(crps=100.):
    return dict(crps=crps, empirical_coverage90=.92, empirical_width90=200., empirical_interval_score90=240.,
                empirical_lower_tail_miss=.04, empirical_upper_tail_miss=.04, point_level_mae=20., point_shape_mae=30.)


def selection_tables():
    rows, refs = [], []
    for region in m.SOURCES:
        for name, projection, alpha in m.arms():
            for factor in m.FACTORS:
                values = metric_values()
                if name == 'joint_T8':
                    values.update(crps=90., empirical_width90=190., empirical_interval_score90=230.)
                rows.append(dict(region=region, input=name, projection=projection, alpha=alpha, factor=factor, **values))
        for packet in ('D2', 'D8', 'G5', 'daily_fuel_with_prequential_residual_law'):
            refs.append(dict(region=region, packet=packet, reference_family='V16_additive' if packet.startswith('daily') else 'V17_direct',
                method='ET_global', alpha=np.nan, factor=1.25, response_offset='day_anchor', native_input_available=False, **metric_values()))
    return pd.DataFrame(rows), pd.DataFrame(refs)


def test155curve_selection_keeps_nan_alpha_backgrounds_and_exact_matched_controls():
    regional, refs = selection_tables()
    assert len(regional) == 1085 and len(r.prior.macro_scores(regional)) == 155
    picked = r.selection(regional, refs)
    assert picked['source_budget_gate']
    assert picked['background_selections']['V16_additive']['alpha'] is None
    assert len(r.matched_contrasts(regional)) == 420
    assert picked['custom_method'] is False and picked['target_test_authorized'] is False
    refs.loc[refs.packet.eq('D8'), 'crps'] = 80.
    assert r.selection(regional, refs)['source_budget_gate'] is False
    regional.loc[regional.input.eq('joint_T8'), 'empirical_coverage90'] = .96
    assert not r.selection(regional, refs)['selections']['joint_T8']['passed']
