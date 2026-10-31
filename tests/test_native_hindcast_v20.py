"""Only synthetic source banks/vintages; no formal V20 generation or H2 labels."""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts import native_hindcast_v20 as m
from scripts import run_native_hindcast_v20 as r


def frame(region, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.uniform(.1, 20, (48, 8))
    result = pd.DataFrame(a, columns=m.GENERATION)
    result['time'] = pd.date_range('2024-01-25', periods=48, freq='h', tz='UTC')
    result['valid'], result['region'] = True, region
    return result


def source_frames():
    return {region: frame(region, i+1) for i, region in enumerate(m.SOURCES)}


def law(split):
    days = m.split_days(split)
    mix = np.tile(np.full(8, 1/8), (len(days), 1))
    draws = np.broadcast_to(mix[:, None], (len(days), 128, 8)).copy()
    return dict(coarse_mix=mix, coarse_draws=draws, coarse_metadata=np.zeros((len(days), 4)),
                coarse_max_period_end=np.full(len(days), -1, dtype=np.int64),
                coarse_max_available_at=np.full(len(days), -1, dtype=np.int64))


def test_source_loader_filters_future_before_any_generation_numeric_conversion(tmp_path):
    good = frame('BPAT')
    future = good.iloc[[0]].copy();future['time'] = '2024-07-01T00:00:00Z'
    for column in m.GENERATION:
        future[column] = 'H2_FINE_MUST_NOT_BE_READ'
    later = future.copy();later['time'] = '2025-01-01T00:00:00Z'
    path = tmp_path/'BPAT_hourly.csv';pd.concat([good, future, later]).to_csv(path, index=False)
    read = m.proper_frame(path, 'BPAT')
    np.testing.assert_array_equal(read[m.GENERATION], good[m.GENERATION])
    assert read.time.min() == m.START and read.time.max() < m.CUTOFF
    with pytest.raises(ValueError, match='Registered source'):
        m.proper_frame(tmp_path/'PACE_hourly.csv', 'PACE')


def test_invalid_proper_day_does_not_turn_missing_fuel_into_a_donor(tmp_path):
    table = frame('BPAT');table.loc[0, 'valid'] = False
    for column in m.GENERATION:
        table[column] = table[column].astype(object)
        table.loc[:23, column] = 'INVALID_DAY_DO_NOT_CONVERT'
    path = tmp_path/'BPAT_hourly.csv';table.to_csv(path, index=False)
    read = m.proper_frame(path, 'BPAT')
    assert len(read) == 24 and read.time.min() == pd.Timestamp('2024-01-26', tz='UTC')


def monthly_row(region='BPAT', start='2023-12-01', end='2024-01-01', available='2024-01-15'):
    return dict(region=region, period_start=start+'T00:00:00Z', period_end=end+'T00:00:00Z',
                available_at=available+'T00:00:00Z', **{k: 1/8 for k in m.COLUMNS})


def test_source_coarse_gate_discards_other_regions_and_unreleased_future_before_float(tmp_path):
    rows = [monthly_row(), monthly_row(start='2024-12-01', end='2025-01-01', available='2025-01-15'),
            monthly_row(region='PACE')]
    for row in rows[1:]:
        for key in m.COLUMNS:
            row[key] = 'UNAUTHORIZED_OR_UNRELEASED'
    path = tmp_path/'monthly.csv';pd.DataFrame(rows).to_csv(path, index=False)
    result = m.monthly_prefix(path)
    assert len(result) == 1 and result.region.tolist() == ['BPAT']
    np.testing.assert_array_equal(result[list(m.COLUMNS)].to_numpy(), 1/8)


def test_exact91_prediction_grid_distinguishes_H1_retrospective_H2_hindcast_query():
    grid = r.expected_grid()
    assert len(grid) == 91
    assert sum(split == 'query_copy' for _, _, split in grid) == 7
    assert sum(split == 'H1_retrospective' for _, _, split in grid) == 42
    assert sum(split == 'H2_hindcast' for _, _, split in grid) == 42
    assert len(set(m.exclusions(a, b) for a in m.SOURCES for b in m.SOURCES if a != b)) == 21
    with pytest.raises(ValueError):
        m.exclusions('BPAT', 'BPAT')
    with pytest.raises(ValueError):
        m.split_days('H1_hindcast')


def test_bank_strict_double_exclusion_and_all_H1_row_normalization_provenance():
    frames = source_frames();pair = ('BPAT', 'CISO');subset = tuple(s for s in m.SOURCES if s not in pair)
    bank = m.native._bank(frames, subset)
    arrays = m.bank_arrays(bank, pair)
    assert len(bank.days_) == 10 and set(arrays['regions']) == set(subset)
    assert arrays['days'].max() < m.CUTOFF.value
    restored = m.restore_bank(arrays, pair)
    np.testing.assert_array_equal(restored.q_, bank.q_)
    with pytest.raises(ValueError, match='pair-excluded'):
        m.bank_arrays(bank, ('BPAT', 'ERCO'))
    altered = {**arrays, 'days': arrays['days'].copy()};altered['days'][0] = m.CUTOFF.value
    with pytest.raises(ValueError, match='H1 region-days'):
        m.restore_bank(altered, pair)
    altered = {**arrays, 'monthmean': arrays['monthmean'].copy()};altered['monthmean'][0, 0] += 1
    with pytest.raises(ValueError):
        m.restore_bank(altered, pair)


@pytest.mark.parametrize('split', ['H1_retrospective', 'H2_hindcast'])
def test_old_sampler_ids_and_ratio_paths_reproduce_with_explicit_temporal_semantics(split):
    frames = source_frames();pair = ('BPAT', 'CISO');subset = tuple(s for s in m.SOURCES if s not in pair)
    bank = m.native._bank(frames, subset);coarse = law(split);dates = m.split_days(split)
    result = m.teachers(bank, coarse, split)
    expected, _ = m.native._indices(bank, coarse['coarse_mix'], dates, np.array(subset))
    np.testing.assert_array_equal(result['donor_ids'], expected)
    raw = np.stack([m.recompose_paths(draw, bank.q_[ids]) for draw, ids in zip(coarse['coarse_draws'], expected)])
    np.testing.assert_array_equal(result['raw'], raw)
    np.testing.assert_array_equal(result['center'], raw.mean(1))
    assert result['chronological_hindcast'].item() is (split == 'H2_hindcast')
    assert result['within_proper_future_donors'].item() is (split == 'H1_retrospective')
    if split == 'H2_hindcast':
        assert np.all(result['future_donor_rows'] == 0) and np.all(result['sampled_future_donors'] == 0)
    else:
        assert result['future_donor_rows'][0] == len(bank.days_)
        assert result['sampled_future_donors'][0] == 128


def test_H1_uses_lower_level_geographic_sampler_without_silently_bypassing_predict_guard():
    frames = source_frames();subset = tuple(s for s in m.SOURCES if s not in ('BPAT', 'CISO'))
    bank = m.native._bank(frames, subset)
    with pytest.raises(ValueError, match='follow the source fitting cutoff'):
        bank.predict(np.full((1, 8), 1/8), m.H1_DAYS[:1])
    assert m.teachers(bank, law('H1_retrospective'), 'H1_retrospective')['chronological_hindcast'].item() is False


def test_excluded_H1_labels_cannot_change_teacher_bank_or_raw_paths():
    frames = source_frames();subset = tuple(s for s in m.SOURCES if s not in ('BPAT', 'CISO'))
    before = m.teachers(m.native._bank(frames, subset), law('H2_hindcast'), 'H2_hindcast')
    for region in ('BPAT', 'CISO'):
        frames[region].loc[:, m.GENERATION] *= 1000
    after = m.teachers(m.native._bank(frames, subset), law('H2_hindcast'), 'H2_hindcast')
    np.testing.assert_array_equal(before['raw'], after['raw'])


def test_all_H2_calendar_days_are_predicted_without_an_H2_fine_validity_mask():
    bank = m.native._bank(source_frames(), tuple(s for s in m.SOURCES if s not in ('ERCO', 'ISNE')))
    result = m.teachers(bank, law('H2_hindcast'), 'H2_hindcast')
    assert result['raw'].shape == (184, 128, 24)
    assert result['origins'][0] == m.CUTOFF.value
    assert result['origins'][-1] == pd.Timestamp('2024-12-31', tz='UTC').value


def test_incomplete_grid_fails_before_loading_any_source_input(monkeypatch):
    monkeypatch.setattr(r, 'check', lambda: pytest.fail('Incomplete seal reached input reader'))
    with pytest.raises(ValueError, match='Exact91'):
        r.check_predictions(dict(entries=[], banks=[], complete_grid=True,
            source_H2_fine_labels_read=False, source_2025_fine_labels_read=False,
            target_values_opened=False, new_model_fit=False, scores_computed=False))


def test_query_copy_is_exact_raw_and_ids_and_has_six_region_row_provenance(tmp_path, monkeypatch):
    parent = tmp_path/'parent';(parent/'predictions').mkdir(parents=True);monkeypatch.setattr(r, 'PARENT', parent)
    frames = source_frames();rows = m.query_rows(frames, 'BPAT')
    rng = np.random.default_rng(4);raw = rng.uniform(0, 1000, (181, 128, 24))
    ids = rng.integers(0, len(rows['bank_days']), (181, 128));source = parent/'predictions/source-BPAT-native.npz'
    np.savez_compressed(source, origins=m.QUERY_DAYS.asi8, raw=raw, donor_ids=ids)
    result = r.native_copy('BPAT', frames)
    np.testing.assert_array_equal(result['raw'], raw);np.testing.assert_array_equal(result['donor_ids'], ids)
    np.testing.assert_array_equal(result['center'], raw.mean(1))
    assert set(result['bank_regions']) == set(m.SOURCES)-{'BPAT'}
    assert not np.shares_memory(result['raw'], raw)
    np.savez_compressed(source, origins=m.QUERY_DAYS.asi8, raw=raw, donor_ids=ids+9999)
    with pytest.raises(ValueError, match='donor range'):
        r.native_copy('BPAT', frames)


def test_query_copy_rejects_partial_2025_calendar_grid(tmp_path, monkeypatch):
    parent = tmp_path/'parent';(parent/'predictions').mkdir(parents=True);monkeypatch.setattr(r, 'PARENT', parent)
    np.savez_compressed(parent/'predictions/source-BPAT-native.npz', origins=m.QUERY_DAYS.asi8[:-1],
                        raw=np.zeros((180, 128, 24)), donor_ids=np.zeros((180, 128), dtype=int))
    with pytest.raises(ValueError, match='Invalid inherited'):
        r.native_copy('BPAT', source_frames())


def test_prediction_refuses_existing_partial_directory_before_source_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path);monkeypatch.setattr(r, 'check', lambda: {})
    (tmp_path/'freeze.json').write_text('{}');(tmp_path/'banks').mkdir()
    monkeypatch.setattr(m, 'proper_frame', lambda *args: pytest.fail('Partial attempt reached source parser'))
    with pytest.raises(FileExistsError, match='Preserve prediction'):
        r.predict()


def test_exclusive_write_and_frozen_byte_mutation_guard(tmp_path):
    path = tmp_path/'seal.json';r.write(path, {'synthetic': True});digest = r.sha(path)
    with pytest.raises(FileExistsError):
        r.write(path, {'synthetic': False})
    path.write_text('{}')
    with pytest.raises(ValueError, match='Changed frozen bytes'):
        r.verify({str(path): digest})


def test_coarse_vintages_cannot_change_an_earlier_prediction_prefix():
    table = pd.DataFrame([monthly_row(), monthly_row(start='2024-08-01', end='2024-09-01', available='2024-09-15')])
    for column in ('period_start', 'period_end', 'available_at'):
        table[column] = pd.to_datetime(table[column], utc=True)
    before = m.coarse_arrays(table, 'BPAT', 'H2_hindcast')
    changed = table.copy();changed.loc[1, list(m.COLUMNS)] = 0.;changed.loc[1, m.COLUMNS[0]] = 1.
    after = m.coarse_arrays(changed, 'BPAT', 'H2_hindcast')
    early = m.H2_DAYS < pd.Timestamp('2024-09-15', tz='UTC')
    np.testing.assert_array_equal(before['coarse_mix'][early], after['coarse_mix'][early])
    np.testing.assert_array_equal(before['coarse_draws'][early], after['coarse_draws'][early])
    assert np.all(before['coarse_max_available_at'] <= m.H2_DAYS.asi8)
    assert np.all(before['coarse_max_period_end'] <= m.H2_DAYS.asi8)
    assert before['coarse_metadata'][0, 0] == 1 and before['coarse_metadata'][-1, 0] == 2
    with pytest.raises(ValueError, match='registered source'):
        m.coarse_arrays(table, 'PACE', 'H2_hindcast')


def test_completion_rejects_false_information_claim_even_when_parent_hashes_match(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path);monkeypatch.setattr(r, 'check', lambda: {})
    (tmp_path/'freeze.json').write_text('{}');digest = r.sha(tmp_path/'freeze.json')
    r.write(tmp_path/'prediction_seal.json', {'freeze_sha256': digest})
    r.write(tmp_path/'completed.json', dict(version=20, freeze_sha256=digest,
            prediction_seal_sha256=r.sha(tmp_path/'prediction_seal.json'), predictions=91, banks=21,
            source_H2_fine_labels_read=True, source_2025_fine_labels_read=False,
            target_values_opened=False, new_model_fit=False, scores_computed=False))
    monkeypatch.setattr(r, 'check_predictions', lambda *args: pytest.fail('Invalid completion reached archives'))
    with pytest.raises(ValueError, match='information boundary'):
        r.verify_completed()
