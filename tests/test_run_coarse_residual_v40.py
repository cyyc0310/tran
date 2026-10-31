"""Information access and complete prediction sealing on synthetic fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import run_coarse_residual_v40 as run


def source_records(outer='BPAT'):
    fits, pairs = [], []
    for source in run.SOURCES:
        if source == outer:
            continue
        keys, regions = [], []
        for s in run.SOURCES:
            if s in (source, outer):
                continue
            n = 24 if s == 'FPL' else 26
            kk = [f'{s}/{at:%Y-%m}' for at in pd.date_range('2022-05-01', periods=n, freq='MS')]
            keys.extend(kk); regions.extend([s]*n)
        fits.append(dict(outer=outer, held_source=source, training_keys=keys, training_regions=regions,
            profiles={'month': np.full((12, 4), .5)}))
        pairs.append(dict(outer=outer, source=source, late_count='12', late_EFbias='20'))
    return fits, pairs


def test_each_source_label_matches_its_profile_and_only_late_label_is_read():
    class LateOnly(dict):
        def __getitem__(self, key):
            if key.startswith('early_'):
                raise AssertionError('Early fine-label feature access')
            return super().__getitem__(key)
    fits, pairs = source_records()
    names, curves, y = run.components(fits[::-1], [LateOnly(p) for p in pairs[::-1]], 'BPAT')
    assert names == [s for s in run.SOURCES if s != 'BPAT']
    assert curves.shape == (6, 12, 4)
    np.testing.assert_array_equal(y, np.full(6, 20.))


def test_outer_metadata_filter_precedes_numbers():
    fits, pairs = source_records()
    class Forbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Other-outer curve converted')
    fits.append(dict(outer='CISO', held_source='BPAT', profiles={'month': Forbidden()}))
    pairs.append(dict(outer='CISO', source='BPAT', late_EFbias=Forbidden()))
    assert len(run.components(fits, pairs, 'BPAT')[0]) == 6


def test_inner_operator_must_exclude_both_sources():
    fits, pairs = source_records()
    fits[0]['training_regions'][0] = 'BPAT'
    with pytest.raises(ValueError, match='five-source'):
        run.components(fits, pairs, 'BPAT')


def test_exactly_six_labels_and_late12_required():
    fits, pairs = source_records()
    with pytest.raises(ValueError, match='six matching'):
        run.components(fits, pairs[:-1], 'BPAT')
    pairs[0]['late_count'] = '11'
    with pytest.raises(ValueError, match='late12'):
        run.components(fits, pairs, 'BPAT')


def sealed(tmp_path, monkeypatch, bad_grid=False):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    folder = tmp_path / 'predictions'; folder.mkdir()
    run.write(tmp_path / 'freeze.json', {})
    freeze = run.sha(tmp_path / 'freeze.json')
    run.write(tmp_path / 'prediction_started.json', dict(freeze_sha256=freeze))
    entries = []
    for source in run.SOURCES:
        p = folder / (source+'.npz')
        keys = np.array([f'{source}/{at:%Y-%m}' for at in run.DATES])
        if bad_grid and source == run.SOURCES[-1]:keys[0] = 'CLOSED/2025-01'
        raw = np.full((5, 6, 6), 500.); ef = raw.mean(-1)
        np.savez(p, arms=np.array(run.model.ARMS), keys=keys, origins=run.DATES.asi8,
            raw_component_EF=raw, component_EF=raw, clipmask=np.zeros_like(raw, dtype=bool), ensemble_EF=ef,
            EF_draws=np.broadcast_to(ef[..., None], (5, 6, 128)), CIF_draws=np.full((5, 6, 128), 200.))
        run.write(p.with_suffix('.json'), {})
        entries.append(dict(region=source, path=str(p), sha256=run.sha(p), audit_path=str(p.with_suffix('.json')),
            audit_sha256=run.sha(p.with_suffix('.json'))))
    run.write(tmp_path / 'prediction_seal.json', dict(freeze_sha256=freeze,
        prediction_started_sha256=run.sha(tmp_path / 'prediction_started.json'), entries=entries))


def test_prediction_grid_is_checked_before_monthly_score_access(tmp_path, monkeypatch):
    sealed(tmp_path, monkeypatch, bad_grid=True)
    monkeypatch.setattr(run, 'check', lambda: {})
    def forbidden(*args):raise AssertionError('Scoring data accessed before seal validation')
    monkeypatch.setattr(run.common, 'monthly_rows', forbidden)
    with pytest.raises(ValueError, match='query grid'):
        run.score()


def test_prediction_start_receipt_is_bound(tmp_path, monkeypatch):
    sealed(tmp_path, monkeypatch)
    assert len(run.predictions()['entries']) == 7
    (tmp_path / 'prediction_started.json').write_text('{"freeze_sha256":"wrong"}')
    with pytest.raises(ValueError, match='source seal'):
        run.predictions()
