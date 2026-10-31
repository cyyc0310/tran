"""Synthetic same-offset features, native identity and immutable phase guards."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scripts import native_refit_v21 as m
from scripts import run_native_refit_v21 as r


def raw(n=2):
    return np.random.default_rng(21).uniform(10, 990, (n, 128, 24))


def packets(days):
    result = {}
    for p in m.PACKETS:
        x = np.zeros((len(days), m.PACKET_WIDTHS[p]))
        x[:, -10:-4] = np.stack([m.calendar(d) for d in days]);result[p] = x
    return result


def test_native_and_joint_feature_schema_keeps_calendar_once():
    days = pd.date_range('2024-01-25', periods=2, tz='UTC');n = raw()
    x = m.input_features(n, days, packets(days))
    assert len(x) == 9 and x['native_only'].shape == (2, 54)
    np.testing.assert_array_equal(x['native_only'][:, :24], n.mean(1))
    np.testing.assert_array_equal(x['native_only'][:, 24:48], n.std(1))
    for p in m.PACKETS:
        assert x['packet_only_' + p].shape[1] == m.PACKET_WIDTHS[p]
        assert x['joint_' + p].shape[1] == 48 + m.PACKET_WIDTHS[p]
        np.testing.assert_array_equal(x['joint_' + p][:, 48:], x['packet_only_' + p])
    broken = packets(days);broken['D2'][0, -10] += .01
    with pytest.raises(ValueError, match='calendar differs'):
        m.input_features(n, days, broken)
    broken = packets(days);broken['D8'] = broken['D8'][:, 1:]
    with pytest.raises(ValueError, match='packet-feature cohort'):
        m.input_features(n, days, broken)


def test_shared_alpha0_is_exact_and_all55_arms_reconstruct_without_noise_change():
    n = raw();delta = np.random.default_rng(2).normal(0, 40, (9, 2, 24))
    assert len(m.arms()) == 55 and len(set(m.arms())) == 55
    identity, requested = m.reconstruct(n, delta, m.arms()[0])
    assert np.array_equal(identity, n) and not np.shares_memory(identity, n)
    np.testing.assert_array_equal(requested, n.mean(1))
    for arm in m.arms()[1:]:
        shifted, point = m.reconstruct(n, delta, arm)
        projected = m.project(delta[m.INPUTS.index(arm[0])], arm[1])
        np.testing.assert_array_equal(shifted, n + arm[2] * projected[:, None])
        np.testing.assert_allclose(point, shifted.mean(1), rtol=0, atol=1e-10)
        np.testing.assert_allclose(shifted - shifted.mean(1)[:, None], n - n.mean(1)[:, None], rtol=0, atol=1e-10)


def test_day_block_full_projections_and_clip_realized_center_width():
    d = np.arange(24.)[None]
    assert np.all(m.project(d, 'day') == 11.5)
    np.testing.assert_array_equal(m.project(d, 'block')[0, ::6], [2.5, 8.5, 14.5, 20.5])
    np.testing.assert_array_equal(m.project(d, 'full'), d)
    n = np.broadcast_to(np.linspace(900, 999, 128)[None, :, None], (1, 128, 24)).copy()
    deltas = np.full((9, 1, 24), 200.)
    diagnostic = m.diagnostics(n, deltas)
    i = m.arms().index(('native_only', 'full', 1.))
    assert diagnostic['requested_center'][i].min() > 1000
    assert np.all(diagnostic['realized_center'][i] == 1000)
    assert diagnostic['new_clipping_rate'][i, 0] == 1
    assert diagnostic['unclipped_width90'][i, 0] > 0
    assert diagnostic['initial_clipped_width90'][i, 0] == 0


def test_residual_model_same_response_region_weights_and_unclipped_delta():
    days = pd.date_range('2024-01-25', periods=30, tz='UTC')
    labels = np.array(['CISO'] * 10 + ['MISO'] * 20)
    x = np.arange(60.).reshape(30, 2);response = np.full((30, 24), 1100.)
    fitted = m.ResidualModel().fit(x, response, labels, days, excluded='BPAT')
    weights = m.balanced(labels)
    np.testing.assert_allclose(fitted.scaler.mean_, np.average(x, axis=0, weights=weights))
    np.testing.assert_allclose(fitted.predict(x[:2]), 1100.)  # No absolute-center clipping of residuals.
    assert fitted.model.n_estimators == 150 and fitted.model.min_samples_leaf == 10
    assert fitted.model.random_state == 17023 and fitted.model.n_jobs == 1
    with pytest.raises(ValueError, match='excluding held'):
        m.ResidualModel().fit(x, response, labels, days, excluded='CISO')
    with pytest.raises(ValueError, match='H1 proper'):
        m.ResidualModel().fit(x, response, labels, days + pd.Timedelta(days=365), excluded='BPAT')


def test_dataset_uses_pair_excluded_H1_native_offset_for_all_nine_inputs(monkeypatch):
    context = r.DataContext.__new__(r.DataContext)
    day = pd.Timestamp('2024-01-25', tz='UTC');a = np.zeros((24, 8));a[:, 0] = a[:, 4] = 1
    context.fine = {region: [(day, a)] for region in r.SOURCES}
    context.entries = {('BPAT', region, 'H1_retrospective'): {'path': region} for region in r.SOURCES if region != 'BPAT'}
    def teacher(path):
        value = 100 + r.SOURCES.index(path)
        return dict(origins=r.parent_api.model.H1_DAYS.asi8,
                    raw=np.broadcast_to(float(value), (158, 128, 24)))
    monkeypatch.setattr(r, 'load_arrays', teacher)
    context.packet_features = lambda labels, days: (packets(days), {}, np.full((4, len(days)), 500.))
    x, data, _ = context.dataset('BPAT')
    assert set(x) == set(m.INPUTS) and 'BPAT' not in data['train_regions']
    np.testing.assert_array_equal(data['train_response'], data['train_truth'] - data['train_native_mean'])
    assert np.all(data['train_truth'] == 500)
    assert not np.array_equal(data['train_response'], data['train_truth'] - 500)
    assert set(data['train_teacher_paths']) == set(r.SOURCES) - {'BPAT'}


@pytest.mark.parametrize('digest', [None, 'x' * 64, 'A' * 64, '0' * 63])
def test_parent_digest_guard_before_any_metadata_or_source_read(monkeypatch, digest):
    monkeypatch.setattr(r, 'read', lambda _: pytest.fail('Invalid digest reached reader'))
    with pytest.raises(ValueError, match='Explicit64hex'):
        r.parent_inputs(digest)


def test_incomplete_parent_rejected_and_pending_receipt_cannot_replace_final_pass(tmp_path, monkeypatch):
    manifest = tmp_path / 'parent.json';monkeypatch.setattr(r, 'MANIFEST', manifest)
    with pytest.raises(ValueError, match='Completed audited R1'):
        r.parent_inputs('0' * 64)
    manifest.write_text(json.dumps({'numerical_audit_status': 'PENDING_INDEPENDENT_AUDIT'}))
    monkeypatch.setattr(r.parent_api, 'check', lambda: pytest.fail('Pending parent reached original inputs'))
    with pytest.raises(ValueError, match='must be PASS'):
        r.parent_inputs(hashlib.sha256(manifest.read_bytes()).hexdigest())


def test_partial_predict_refused_before_H1_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path);monkeypatch.setattr(r, 'check', lambda: {})
    (tmp_path / 'freeze.json').write_text('{}');(tmp_path / 'predictions').mkdir()
    monkeypatch.setattr(r, 'DataContext', lambda _: pytest.fail('Partial prediction reached H1 reader'))
    with pytest.raises(FileExistsError, match='Preserve prediction'):
        r.predict()


def test_incomplete7_grid_rejected_before_training_or_2025_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path);monkeypatch.setattr(r, 'check', lambda: {})
    seal = dict(entries=[], archive_count=7, raw_arm_count=55, curve_count=275, complete_grid=True,
        prediction_reads_2025_fine_labels=False, source_H2_fine_labels_read=False, target_values_opened=False)
    monkeypatch.setattr(r, 'read', lambda _: seal)
    monkeypatch.setattr(r.sketch, 'source_days', lambda *args: pytest.fail('Incomplete seal reached fine labels'))
    with pytest.raises(ValueError, match='Exact7compact'):
        r.score()


def scores():
    rows = []
    for region in r.SOURCES:
        for name, projection, alpha in m.arms():
            for factor in m.FACTORS:
                values = dict(crps=100., empirical_coverage90=.92, empirical_width90=200.,
                    empirical_interval_score90=240., empirical_lower_tail_miss=.04,
                    empirical_upper_tail_miss=.04, point_level_mae=20., point_shape_mae=30.)
                if name == 'joint_T8':
                    values.update(crps=90., empirical_width90=190., empirical_interval_score90=230.)
                rows.append(dict(region=region, input=name, projection=projection, alpha=alpha, factor=factor, **values))
    return pd.DataFrame(rows)


def test_full275_curve_selection_and_explicit_matched_contrasts():
    table = scores();macro = r.macro_scores(table)
    assert len(macro) == 275 and len(table) == 1925
    choice = r.selection(table)
    assert choice['source_budget_gate'] and choice['packet_selections']['T8']['input'] == 'joint_T8'
    assert choice['custom_method'] is False and choice['target_test_authorized'] is False
    comparisons = r.matched_contrasts(table)
    assert len(comparisons) == 1260
    assert set(comparisons.comparison) == {'joint_minus_packet', 'T8_minus_G5'}
    assert np.all(comparisons[comparisons.packet.eq('T8')]['crps_joint_minus_packet'] == -10)
    assert np.all(comparisons[comparisons.packet.eq('joint_T8_minus_G5')]['crps_T8_minus_G5'] == -10)
    # A native-only refit that is stronger than T8 must block the claimed packet gain.
    table.loc[table.input.eq('native_only'), 'crps'] = 80.
    assert r.selection(table)['source_budget_gate'] is False


def test_source_reader_H1_only_filters_H2_and_2025_poison(tmp_path):
    path = tmp_path / 'BPAT_hourly.csv'
    table = pd.DataFrame(np.ones((24, 8)), columns=['generation_' + f for f in r.sketch.model.FUELS])
    table['time'] = pd.date_range('2024-01-25', periods=24, freq='h', tz='UTC')
    table['region'], table['valid'] = 'BPAT', True
    future = table.iloc[[0]].copy();future['time'] = '2024-07-01T00:00:00+00:00'
    for key in ['generation_' + f for f in r.sketch.model.FUELS]:
        future[key] = 'H2_DO_NOT_PARSE'
    later = future.copy();later['time'] = '2025-01-01T00:00:00+00:00'
    pd.concat([table, future, later]).to_csv(path, index=False)
    read = r.sketch.source_days(path, '2024-01-25', '2024-07-01')
    assert len(read) == 1 and read[0][1].shape == (24, 8)
