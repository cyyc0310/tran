"""Synthetic identity, information, clipping and seal tests; no real pilot inputs."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from scripts import native_center_v18 as m
from scripts import run_native_center_v18 as r


def paths(days=2):
    generator = np.random.default_rng(18)
    native = generator.uniform(100, 700, (days, 128, 24))
    ids = np.tile(np.arange(128), (days, 1))
    return native, ids


def test_fixed_grid_has_35_archives_and_260_curves_with_repeated_zero_controls():
    assert len(r.registered_grid()) == 35
    assert sum(len(m.arms(p)) for p in m.PACKETS) == 51
    assert (1+sum(len(m.arms(p)) for p in m.PACKETS))*len(m.FACTORS) == 260
    assert all(sum(a == 0 for _, a in m.arms(p)) == (5 if p == 'T8' else 4) for p in m.PACKETS)
    assert all('raw_block' not in [name for name, _ in m.arms(p)] for p in ('D2', 'D8', 'G5'))


@pytest.mark.parametrize('resolution,rank', [('day', 1), ('block', 4), ('full', 24)])
def test_orthogonal_projection_preserves_mean_idempotence_and_residual_orthogonality(resolution, rank):
    delta = np.random.default_rng(4).normal(size=(3, 24))
    projected = m.project(delta, resolution)
    np.testing.assert_allclose(projected.mean(1), delta.mean(1), atol=1e-15)
    np.testing.assert_allclose(m.project(projected, resolution), projected, atol=1e-15)
    np.testing.assert_allclose((projected*(delta-projected)).sum(1), 0, atol=1e-14)
    if rank == 4:
        assert np.unique(projected[0]).size == 4
    if rank == 24:
        np.testing.assert_array_equal(projected, delta)


def test_every_alpha_zero_is_exact_native_and_all_nonzero_paths_keep_native_noise():
    native, ids = paths()
    fitted = np.tile(np.linspace(250, 650, 24), (2, 1))
    for packet in m.PACKETS:
        archive = m.build(native, ids, fitted, np.full(2, 420.),
                          np.tile([200., 350., 500., 650.], (2, 1)) if packet == 'T8' else None, packet)
        np.testing.assert_array_equal(archive['donor_ids'], ids)
        for i, (method, alpha) in enumerate(m.arms(packet)):
            if alpha == 0:
                np.testing.assert_array_equal(archive['raw'][i], native)
                assert not np.shares_memory(archive['raw'][i], native)
            actual = archive['raw'][i]-archive['raw'][i, :, :1]
            expected = native-native[:, :1]
            np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
            assert archive['native_noise_max_abs_error'][i].max() < 1e-9
    np.testing.assert_array_equal(ids, np.tile(np.arange(128), (2, 1)))


def test_projection_targets_and_dayraw_do_not_replace_native_hourly_shape():
    native, ids = paths(1);center = np.arange(24)[None]*10+100
    archive = m.build(native, ids, center, np.array([450.]), None, 'D2')
    mean = native.mean(1)
    def requested(method):
        return archive['requested_center'][m.arms('D2').index((method, 1.))]
    np.testing.assert_allclose(requested('ET_full'), center, atol=1e-12)
    np.testing.assert_allclose(requested('ET_day').mean(1), center.mean(1), atol=1e-12)
    np.testing.assert_allclose(requested('ET_day')-requested('ET_day').mean(1, keepdims=True),
                               mean-mean.mean(1, keepdims=True), atol=1e-12)
    np.testing.assert_allclose(requested('ET_block').reshape(1, 4, 6).mean(2), center.reshape(1, 4, 6).mean(2))
    np.testing.assert_allclose(requested('raw_day').mean(1), 450., atol=1e-12)


def test_ET_point_reading_rejects_risk_model_disagreement_and_ignores_biased_rawmean():
    centers = np.full((4, 2, 24), 333.)
    archive = dict(models=np.array(['ridge_global', 'ET_global', 'ET_hetero', 'ET_QRF']),
                   centers=centers, raw=np.full((4, 2, 128, 24), 600.))
    np.testing.assert_array_equal(m.extract_et_center(archive), 333.)
    centers[3, 0, 0] += 1e-12
    with pytest.raises(ValueError, match='identical'):
        m.extract_et_center(archive)


def test_clipping_changes_realized_center_and_width_without_being_a_new_risk_law():
    native = np.tile(np.linspace(0, 1000, 128)[None, :, None], (1, 1, 24))
    raw = m.shifted_paths(native, np.full((1, 24), 400.), 1.)
    diagnostics = m.summarize_paths(raw, native, np.full((1, 24), 900.))
    assert diagnostics['new_clipping_rate'][0] > .39
    assert diagnostics['unclipped_center'][0, 0] == pytest.approx(900.)
    assert diagnostics['realized_center'][0, 0] < 900.
    uncut = np.sort(raw, axis=1)[:, 121]-np.sort(raw, axis=1)[:, 6]
    point, clipped, rate = m.scaled_paths(raw, 1.)
    cut = np.sort(clipped, axis=1)[:, 121]-np.sort(clipped, axis=1)[:, 6]
    assert np.all(cut < uncut)
    np.testing.assert_array_equal(point, diagnostics['realized_center'])
    assert rate[0] == 0
    _, _, expanded_rate = m.scaled_paths(raw, 2.)
    assert expanded_rate[0] > 0


def test_block_H_is_not_fabricated_for_other_packets_and_bad_native_is_rejected():
    native, ids = paths(1)
    with pytest.raises(ValueError, match='not available'):
        m.build(native, ids, native.mean(1), np.array([400.]), np.full((1, 4), 400.), 'G5')
    with pytest.raises(ValueError, match='four finite'):
        m.build(native, ids, native.mean(1), np.array([400.]), None, 'T8')
    native[0, 0, 0] = -1
    with pytest.raises(ValueError, match='Bounded native'):
        m.shifted_paths(native, np.zeros((1, 24)), 0.)


def test_context_uses_explicit_EWMA_schema_and_lawful_packet_replay(tmp_path, monkeypatch):
    from scripts import local_sketch_v17 as old
    folder = tmp_path/'predictions';folder.mkdir();monkeypatch.setattr(r, 'PARENT', tmp_path)
    blocks = np.array([100., 300., 600., 800.])
    packet_values = {'D2': np.array([1.8, 4.]), 'T8': np.column_stack((blocks/1000, np.ones(4))).ravel()}
    histories = {}
    for packet, values in packet_values.items():
        table = pd.DataFrame([dict(region='BPAT', day='2024-12-01T00:00:00Z',
                                  **{f'v{i}': v for i, v in enumerate(values)})])
        histories[packet, 'BPAT', 7] = old.PacketHistory(table, packet, 7)
    features = np.stack([histories['T8', 'BPAT', 7].features(day)[0] for day in r.DAYS])
    path = folder/'source-BPAT-7-T8.npz'
    arrays = dict(origins=r.DAYS.asi8, features=features, anchors=np.full(181, 450.),
                  centers=np.full((4, 181, 24), 400.), models=np.array(old.MODELS))
    np.savez_compressed(path, **arrays)
    _, _, block, audit = r.context('BPAT', 'T8', histories)
    np.testing.assert_allclose(block, np.tile(blocks, (181, 1)), atol=1e-12)
    assert audit['common_anchor_max_abs_difference'] < 1e-9
    # Swap latest and EWMA block entries while leaving the schema shape valid.
    damaged = features.copy();damaged[:, 11] += 1
    np.savez_compressed(path, **{**arrays, 'features': damaged})
    with pytest.raises(ValueError, match='lawful released-packet replay'):
        r.context('BPAT', 'T8', histories)


def test_incomplete35seal_fails_before_source_truth_or_reference_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    (tmp_path/'freeze.json').write_text('{}')
    monkeypatch.setattr(r, 'check', lambda: {'source': 'UNREAD'})
    touched = []
    monkeypatch.setattr(r.prior, 'source_days', lambda *args: touched.append('truth'))
    monkeypatch.setattr(r, 'reference_curves', lambda: touched.append('reference'))
    r.write(tmp_path/'prediction_seal.json', dict(freeze_sha256=r.sha(tmp_path/'freeze.json'),
            entries=[], complete_grid=True, source2025fine_labels_read=False, target_values_opened=False))
    with pytest.raises(ValueError, match='Exact35'):
        r.score()
    assert not touched


def test_wrong_registered_grid_path_fails_before_any_archive_load(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    entries = [dict(region=source, packet=packet, path='/unregistered/archive.npz')
               for source, packet in sorted(r.registered_grid())]
    with pytest.raises(ValueError, match='Unregistered center prediction path'):
        r.check_predictions(dict(entries=entries, complete_grid=True,
                                source2025fine_labels_read=False, target_values_opened=False))


def regional_fixture():
    rows = []
    for packet, value in [('native', 30.), ('D2', 28.), ('D8', 29.), ('G5', 27.), ('T8', 25.)]:
        for source in m.SOURCES:
            row = dict(region=source, packet=packet, method='native' if packet == 'native' else 'ET_day',
                       alpha=0. if packet == 'native' else 1., factor=1.25, **{k: 0. for k in r.METRICS})
            row.update(crps=value, empirical_coverage90=.92, empirical_width90=100., empirical_interval_score90=value+100.)
            rows.append(row)
    return pd.DataFrame(rows)


def test_source_selection_requires_all_packet_controls_and_keeps_claims_false():
    regional = regional_fixture();choice = r.selection(regional)
    assert choice['source_information_gate'] and choice['strongest_limited_packet']['packet'] == 'G5'
    assert not choice['custom_method'] and not choice['target_test_authorized'] and not choice['H_timing_causal_claim']
    json.dumps(choice, allow_nan=False)
    regional.loc[regional.packet.eq('D2'), 'crps'] = 24.
    assert not r.selection(regional)['source_information_gate']
    regional.loc[regional.packet.eq('T8') & regional.region.eq('BPAT'), 'empirical_coverage90'] = .84
    assert not r.selection(regional)['selections']['T8']['passed']


def test_matched_contrasts_are_same_projection_alpha_factor_only():
    rows = []
    for region in m.SOURCES:
        for packet in ('T8', 'G5'):
            for method, alpha in m.arms(packet):
                for factor in m.FACTORS:
                    rows.append(dict(region=region, packet=packet, method=method, alpha=alpha, factor=factor,
                                     **{k: 1. if packet == 'T8' else 2. for k in r.METRICS}))
    matched = r.matched_contrasts(pd.DataFrame(rows))
    assert len(matched) == 420 and 'raw_block' not in set(matched.method)
    np.testing.assert_array_equal(matched['crps_T8_minus_G5'], -1.)


def test_reference_curves_preserve_every_V16_additive_and_V17_ET_risk_setting(tmp_path, monkeypatch):
    p17, p16 = tmp_path/'old17', tmp_path/'old16';p17.mkdir();p16.mkdir()
    monkeypatch.setattr(r, 'PARENT', p17);monkeypatch.setattr(r, 'V16', p16)
    rows = []
    for region in m.SOURCES:
        for packet in ('native', *m.PACKETS):
            for name in (('native',) if packet == 'native' else ('ET_global', 'ET_hetero', 'ET_QRF')):
                for factor in m.FACTORS:
                    rows.append(dict(region=region, packet=packet, model=name, lag=0 if packet == 'native' else 7,
                                     factor=factor, **{k: .5 for k in r.METRICS}))
    pd.DataFrame(rows).to_csv(p17/'region_metrics.csv', index=False)
    rows = []
    for region in m.SOURCES:
        for forecaster in ('latest', 'weekday', 'ewma', 'seasonal_kalman'):
            for factor in m.FACTORS:
                rows.append(dict(region=region, lag=7, forecaster=forecaster, arm='additive', factor=factor,
                                 **{k: .5 for k in r.METRICS}))
    pd.DataFrame(rows).to_csv(p16/'region_metrics.csv', index=False)
    result = r.reference_curves()
    assert len(result) == 595
    assert not result[result.reference_family.eq('V17_direct') & result.packet.ne('native')].native_input_available.any()
    damaged = pd.DataFrame(rows).iloc[:-1];damaged.to_csv(p16/'region_metrics.csv', index=False)
    with pytest.raises(ValueError, match='Complete sealed'):
        r.reference_curves()


def test_old_style_exclusive_seals_and_hash_mutation_fail_closed(tmp_path):
    path = tmp_path/'receipt.json';r.write(path, {'v': 1});digest = r.sha(path)
    with pytest.raises(FileExistsError):
        r.write(path, {'v': 2})
    path.write_text('{}')
    with pytest.raises(ValueError, match='Changed frozen bytes'):
        r.verify({str(path): digest})


def test_frozen_source_registry_and_input_bytes_cannot_be_retargeted(tmp_path, monkeypatch):
    import hashlib
    out, parent, source = tmp_path/'new', tmp_path/'ancestor', tmp_path/'registered_source'
    out.mkdir();parent.mkdir();source.mkdir()
    monkeypatch.setattr(r, 'OUT', out);monkeypatch.setattr(r, 'PARENT', parent)
    manifest = tmp_path/'old_manifest.json';manifest.write_text('{}')
    monkeypatch.setattr(r, 'V16_MANIFEST', manifest);monkeypatch.setattr(r, 'V16_MANIFEST_SHA', r.sha(manifest))
    old_code = {}
    for i in range(54):
        path = tmp_path/f'old_code_{i}.py';path.write_text('# frozen synthetic ancestor\n')
        old_code[str(path)] = r.sha(path)
    ancestor = dict(code_hashes=old_code, source=str(source))
    r.write(parent/'freeze.json', ancestor)
    env = {'synthetic': 'only'};monkeypatch.setattr(r.prior, 'environment', lambda: env)
    inputs = {str(manifest): r.sha(manifest), str(parent/'freeze.json'): r.sha(parent/'freeze.json')}
    for region in m.SOURCES:
        path = source/f'{region}_hourly.csv';path.write_text('DO_NOT_PARSE_SYNTHETIC')
        inputs[str(path)] = r.sha(path)
    frozen = {**r.config(), 'source': str(source), 'input_hashes': inputs,
              'code_hashes': {**old_code, **{str(r.ROOT/p): r.sha(r.ROOT/p) for p in r.CODE}},
              'environment': env, 'environment_sha256': hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()}
    r.write(out/'freeze.json', frozen)
    assert r.check()['source'] == str(source)
    (out/'freeze.json').write_text(json.dumps({**frozen, 'source': str(tmp_path/'other_source')}))
    with pytest.raises(ValueError, match='Changed frozen center pilot contract'):
        r.check()
    (out/'freeze.json').write_text(json.dumps(frozen))
    (source/'BPAT_hourly.csv').write_text('changed input bytes')
    with pytest.raises(ValueError, match='Changed frozen bytes'):
        r.check()


def test_corrupted_ancestor_ET_center_cannot_pass_projection_archive_verification(tmp_path, monkeypatch):
    folder = tmp_path/'predictions';folder.mkdir();monkeypatch.setattr(r, 'OUT', tmp_path)
    native, ids = paths(181)
    monkeypatch.setattr(r, 'native_input', lambda region: (native, ids))
    genuine = native.mean(1)
    wrong = np.clip(genuine+10, 0, 1000)
    arrays = m.build(native, ids, wrong, np.full(181, 400.), None, 'D2')
    path = folder/'source-BPAT-D2.npz'
    np.savez_compressed(path, origins=r.DAYS.asi8, **arrays)
    entries = [dict(region=source, packet=packet, path=str(folder/f'source-{source}-{packet}.npz'))
               for source, packet in sorted(r.registered_grid())]
    entry = next(e for e in entries if (e['region'], e['packet']) == ('BPAT', 'D2'))
    entry['sha256'] = r.sha(path)
    entries.remove(entry);entries.insert(0, entry)
    monkeypatch.setattr(r, 'archived_context', lambda region, packet: (genuine, None, np.full(181, 400.), None))
    with pytest.raises(ValueError, match='Registered projection/alpha grid or center diagnostics changed'):
        r.check_predictions(dict(entries=entries, complete_grid=True,
                                source2025fine_labels_read=False, target_values_opened=False))


def test_legal_full_emitting_raw_ratio_roundoff_does_not_abort_or_change_anchor():
    from scripts.local_sketch_v17 import channels
    value = 85478795.0378618
    anchor = float(channels(np.array([value, value]), 'D2')[-1])
    assert 1000 < anchor < 1000+1e-9
    native, ids = paths(1)
    archive = m.build(native, ids, native.mean(1), np.array([anchor]), np.full((1, 4), anchor), 'T8')
    np.testing.assert_array_equal(archive['day_anchor'], anchor)
    assert np.isfinite(archive['raw']).all()
    with pytest.raises(ValueError, match='daily anchors'):
        m.build(native, ids, native.mean(1), np.array([1000.01]), None, 'D2')
