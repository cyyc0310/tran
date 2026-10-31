"""Synthetic-only verification recovery tests; no real stage or source reads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts import verify_native_hindcast_v20r1 as r


def source_frames():
    frames = {}
    for i, region in enumerate(r.model.SOURCES):
        values = np.random.default_rng(i + 4).uniform(.1, 100, (48, 8))
        frame = pd.DataFrame(values, columns=r.model.GENERATION)
        frame['time'] = pd.date_range('2024-01-25', periods=48, freq='h', tz='UTC')
        frame['valid'], frame['region'] = True, region
        frames[region] = frame
    return frames


def bank():
    pair = ('BPAT', 'CISO')
    return pair, r.model.native._bank(source_frames(), tuple(s for s in r.model.SOURCES if s not in pair))


def law():
    n = len(r.model.H1_DAYS)
    rng = np.random.default_rng(14)
    return dict(coarse_mix=rng.dirichlet(np.ones(8), n),
                coarse_draws=rng.dirichlet(np.ones(8), (n, 128)),
                coarse_metadata=np.zeros((n, 4)), coarse_max_period_end=np.full(n, -1, dtype=np.int64),
                coarse_max_available_at=np.full(n, -1, dtype=np.int64))


def test_npz_layout_regression_preserves_values_but_requires_original_fresh_layout(tmp_path):
    pair, fresh = bank()
    path = tmp_path / 'bank.npz'
    np.savez_compressed(path, **r.model.bank_arrays(fresh, pair))
    saved = r.load_arrays(path)
    assert np.array_equal(saved['q'], fresh.q_)
    assert fresh.q_.strides[-2:] == (8, 192)
    assert saved['q'].flags.c_contiguous and saved['q'].strides[-2:] == (64, 8)
    coarse = law()
    original = r.model.teachers(fresh, coarse, 'H1_retrospective')
    old_replay = r.model.teachers(r.model.restore_bank(saved, pair), coarse, 'H1_retrospective')
    assert np.array_equal(original['donor_ids'], old_replay['donor_ids'])
    assert not np.array_equal(original['raw'], old_replay['raw'])
    verified = r.verified_fresh_bank(saved, fresh, pair)
    assert verified is fresh
    fixed = r.model.teachers(verified, coarse, 'H1_retrospective')
    r.exact_fields(original, fixed, 'corrected replay')
    np.savez_compressed(tmp_path / 'prediction.npz', **original)
    r.exact_fields(r.load_arrays(tmp_path / 'prediction.npz'), fixed, 'saved original')


@pytest.mark.parametrize('field', ['raw', 'donor_ids'])
def test_wrong_prediction_values_or_sampler_ids_are_rejected(field):
    expected = dict(raw=np.zeros((1, 128, 24)), donor_ids=np.zeros((1, 128), dtype=np.int64))
    actual = {k: v.copy() for k, v in expected.items()}
    actual[field].flat[0] += 1
    with pytest.raises(ValueError, match='exact field mismatch: ' + field):
        r.exact_fields(actual, expected, 'synthetic corrupted archive')


def test_no_tolerance_even_when_bank_normalization_guard_would_accept_tiny_delta():
    pair, fresh = bank()
    arrays = {key: value.copy() for key, value in r.model.bank_arrays(fresh, pair).items()}
    arrays['raw_cif'][0, 0] += 1e-11
    r.model.restore_bank(arrays, pair)  # Its legacy numerical guard is intentionally insufficient for equality.
    with pytest.raises(ValueError, match='exact field mismatch: raw_cif'):
        r.verified_fresh_bank(arrays, fresh, pair)


def test_exact_field_registry_dtype_and_nan_checks():
    with pytest.raises(ValueError, match='field registry'):
        r.exact_fields({'raw': np.zeros(1)}, {}, 'registry')
    with pytest.raises(ValueError, match='exact field mismatch'):
        r.exact_fields({'ids': np.zeros(1, dtype=np.int32)}, {'ids': np.zeros(1, dtype=np.int64)}, 'dtype')
    with pytest.raises(ValueError, match='exact field mismatch'):
        r.exact_fields({'raw': np.array([np.nan])}, {'raw': np.array([np.nan])}, 'nan')


def test_aliases_register91_predictions21_banks_and_keep_original_paths():
    entries, banks = r.aliases()
    assert len(entries) == 91 and len(banks) == 21
    assert {(e['outer'], e['teacher'], e['split']) for e in entries} == r.prior.expected_grid()
    assert len({e['path'] for e in entries + banks}) == 112
    assert all(Path(e['path']).parent == r.FAILED / 'predictions' for e in entries)
    assert all(Path(e['path']).parent == r.FAILED / 'banks' for e in banks)
    assert sum(e.get('chronological_hindcast', False) for e in entries) == 42
    assert sum(e.get('within_proper_future_donors', False) for e in entries) == 42


def test_parent_with_completed_marker_is_not_the_pinned_failed_recovery_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'FAILED', tmp_path)
    (tmp_path / 'completed.json').write_text('{}')
    monkeypatch.setattr(r.prior, 'check', lambda: pytest.fail('Incorrect parent reached inherited reader'))
    with pytest.raises(ValueError, match='pinned failed, uncompleted'):
        r.failed_parent()


def test_arbitrary_incomplete_parent_cannot_bypass_failure_pins(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'FAILED', tmp_path)
    (tmp_path / 'freeze.json').write_text('{}')
    monkeypatch.setattr(r.prior, 'check', lambda: pytest.fail('Unpinned parent reached source checker'))
    with pytest.raises(ValueError, match='Changed frozen bytes'):
        r.failed_parent()


def test_all_three_prelaunch_reports_required(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    for name in r.REVIEWS[:2]:
        (tmp_path / name).write_text('Independent APPROVE')
    with pytest.raises(ValueError, match='All three'):
        r.check_review_files()
    (tmp_path / r.REVIEWS[2]).write_text('Scientific PASS')
    r.check_review_files()


def test_partial_r1_freeze_refuses_prepare_before_parent_or_source_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    (tmp_path / 'freeze.json').write_text('{}')
    monkeypatch.setattr(r, 'failed_parent', lambda: pytest.fail('Partial R1 reached parent reader'))
    with pytest.raises(FileExistsError, match='Preserve complete or partial'):
        r.prepare()


def test_changed_freeze_during_verification_prevents_any_completion_write(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    (tmp_path / 'freeze.json').write_text('{}')
    frozen = {'entries': [], 'banks': []}
    monkeypatch.setattr(r, 'check', lambda: frozen)
    def changed(_):
        (tmp_path / 'freeze.json').write_text('{"during_run":true}')
        return {}
    monkeypatch.setattr(r, 'verify_values', changed)
    with pytest.raises(ValueError, match='Freeze changed'):
        r.run_verification()
    assert not (tmp_path / 'completed.json').exists()
    assert not (tmp_path / 'verification_seal.json').exists()


def test_existing_verification_receipt_refuses_rerun_before_array_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    (tmp_path / 'freeze.json').write_text('{}')
    (tmp_path / 'verification_seal.json').write_text('{}')
    monkeypatch.setattr(r, 'check', lambda: {})
    monkeypatch.setattr(r, 'verify_values', lambda _: pytest.fail('Existing receipt reached arrays'))
    with pytest.raises(FileExistsError, match='Preserve verification'):
        r.run_verification()


def test_receipts_disclose_original_exit1_and_pending_independent_audit():
    contract = r.contract()
    assert contract['producer_attempt_status'] == 'failed_exit_1'
    assert contract['generation_attempt_success'] is False
    assert contract['new_predictions_generated'] is False
    assert contract['numerical_audit_status'] == 'PENDING_INDEPENDENT_AUDIT'
    assert contract['recovery'] == 'verification_only_fresh_bank_layout'
