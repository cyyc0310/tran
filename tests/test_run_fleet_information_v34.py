"""V34 execution boundaries and predeclared information-value decision tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_fleet_information_v34 as runner


def region_scores():
    records = []
    for region in runner.model.SOURCES:
        for arm in runner.model.ARMS:
            error = 10. if arm in runner.model.ARMS[:2] else 8.
            records.append(dict(region=region, arm=arm, cif_abs_error=error,
                cif_sq_error=error ** 2, cif_signed_error=-error,
                ef_abs_error=2 * error, ef_sq_error=4 * error ** 2,
                ef_signed_error=-2 * error, cif_rmse=error, ef_rmse=2 * error))
    return pd.DataFrame(records)


def test_all_arms_and_comparisons_are_fixed():
    assert len(runner.model.ARMS) == 11
    pairs = runner.matched_pairs()
    assert len(pairs) == len(set(pairs)) == 24
    assert pairs[:2] == [('Capacity-OP', 'Size-Ridge'), ('Capacity-OP', 'Size-ET')]
    assert ('DirectEF-ET', 'SizeCap-ET') in pairs


def test_signal_requires_both_baselines_and_preserves_all_arms():
    frame = region_scores()
    macro, choice = runner.summarize(frame)
    assert len(macro) == 11
    assert choice['B1_signal_arms'] == list(runner.model.ARMS[2:])
    frame.loc[frame.arm.eq('Size-ET'), 'cif_abs_error'] = 7.
    _, choice = runner.summarize(frame)
    assert not choice['B1_signal_arms']
    assert not choice['new_holdout_authorized']
    assert not choice['mechanism_breakthrough']


@pytest.mark.parametrize('source', ['MISO', 'ERCO'])
def test_critical_source_regression_blocks_signal_even_below_ten_percent(source):
    frame = region_scores()
    frame.loc[frame.region.eq(source) & frame.arm.eq('Capacity-OP'), 'cif_abs_error'] = 10.1
    _, choice = runner.summarize(frame)
    assert 'Capacity-OP' not in choice['B1_signal_arms']
    assert 'Capacity-Operable' in choice['B1_signal_arms']


def test_any_source_ten_percent_guard_and_macro_five_percent_guard():
    frame = region_scores()
    frame.loc[frame.region.eq('BPAT') & frame.arm.eq('Capacity-OP'), 'cif_abs_error'] = 11.01
    frame.loc[frame.arm.eq('Capacity-Operable'), 'cif_abs_error'] = 9.6
    _, choice = runner.summarize(frame)
    assert 'Capacity-OP' not in choice['B1_signal_arms']
    assert 'Capacity-Operable' not in choice['B1_signal_arms']


def test_macro_rmse_is_mean_of_region_rmse_not_sqrt_of_mean_mse():
    frame = region_scores()
    subset = frame.arm.eq('Size-Ridge') & frame.region.eq('BPAT')
    frame.loc[subset, ['cif_rmse', 'cif_sq_error']] = [20., 400.]
    macro, _ = runner.summarize(frame)
    row = macro.set_index('arm').loc['Size-Ridge']
    assert row.cif_rmse == pytest.approx(80. / 7.)
    assert row.cif_rmse != pytest.approx(np.sqrt(row.cif_sq_error))


def test_query_loader_reads_only_whitelisted_members(monkeypatch):
    accessed = []

    class Archive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __getitem__(self, key):
            accessed.append(key)
            assert key in runner.QUERY_MEMBERS
            return np.asarray([1.])

    monkeypatch.setattr(runner.np, 'load', lambda *args, **kwargs: Archive())
    result = runner.load_query('not-opened.npz')
    assert tuple(accessed) == runner.QUERY_MEMBERS
    assert set(result) == set(runner.QUERY_MEMBERS)
    assert 'M8_draws' not in result


def test_prepare_preserves_existing_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    monkeypatch.setattr(runner, 'expected_contract', lambda: pytest.fail('Do not inspect new inputs'))
    with pytest.raises(FileExistsError):
        runner.prepare()


def test_prediction_failure_retains_marker_and_refuses_retry(tmp_path, monkeypatch):
    out = tmp_path / 'attempt'
    out.mkdir()
    (out / 'freeze.json').write_text('{}')
    monkeypatch.setattr(runner, 'OUT', out)
    monkeypatch.setattr(runner, 'check', lambda: dict(capacity_paths={'op': 'a', 'operable': 'b'}))
    monkeypatch.setattr(runner, 'capacity_rows', lambda path: (_ for _ in ()).throw(ValueError('bad capacity')))
    with pytest.raises(ValueError, match='bad capacity'):
        runner.predict()
    marker = out / 'prediction_started.json'
    before = marker.read_bytes()
    assert not (out / 'prediction_seal.json').exists()
    with pytest.raises(FileExistsError):
        runner.predict()
    assert marker.read_bytes() == before


def test_score_rejects_unsealed_predictions_before_any_label_access(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    monkeypatch.setattr(runner, 'check', lambda: {})

    def reject():
        raise ValueError('incomplete forecast seal')

    monkeypatch.setattr(runner, 'predictions', reject)
    monkeypatch.setattr(runner.common, 'monthly_rows', lambda _: pytest.fail('Labels accessed too early'))
    with pytest.raises(ValueError, match='incomplete forecast seal'):
        runner.score()


def test_empty_parent_pin_fails_closed_before_manifest_read(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'PARENTS', {tmp_path / 'absent.json': ''})
    monkeypatch.setattr(runner, 'read', lambda _: pytest.fail('Unpinned parent opened'))
    with pytest.raises(ValueError, match='Fixed independently audited'):
        runner.expected_contract()


def test_within_parent_conflicting_hashes_are_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    path.write_text('{}')
    manifest = dict(status='VERIFICATION_COMPLETE', independent_actual_audit_status='PASS',
        producer_code_hashes={'same': 'aaa'}, input_hashes={'same': 'bbb'},
        auditor_code_hashes={}, artifact_hashes={})
    monkeypatch.setattr(runner, 'PARENTS', {path: runner.sha(path)})
    monkeypatch.setattr(runner, 'read', lambda _: manifest)
    with pytest.raises(ValueError, match='Conflicting within-parent'):
        runner.expected_contract()


def test_synthetic_score_preserves_common_support_and_source_macro(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'freeze.json').write_text('{}')
    (out / 'prediction_seal.json').write_text('{}')
    (out / 'prediction_started.json').write_text('{}')
    labels, support, entries = {}, [], []
    counts = dict(zip(runner.model.SOURCES, [6, 6, 6, 6, 1, 5, 5]))
    for source_index, source in enumerate(runner.model.SOURCES):
        keys = [f'{source}/{at:%Y-%m}' for at in runner.DATES]
        for index, key in enumerate(keys):
            scored = index < counts[source]
            support.append(dict(region=source, key=key, status='scored' if scored else 'missing'))
            if scored:
                labels[key] = dict(cif=200., ef=500.)
        path = out / f'{source}.npz'
        np.savez(path, keys=keys,
                 CIF_point=np.full((11, 6), 200. + source_index),
                 EF_point=np.full((11, 6), 500. + source_index))
        entries.append(dict(region=source, path=str(path)))
    support = pd.DataFrame(support)
    support_path = tmp_path / 'support.csv'
    support.to_csv(support_path, index=False)
    monkeypatch.setattr(runner, 'OUT', out)
    monkeypatch.setattr(runner, 'check', lambda: dict(support_path=str(support_path)))
    events = []

    def predictions():
        events.append('sealed')
        return dict(entries=entries)

    def rows(_):
        assert events == ['sealed']
        events.append('labels')
        return []

    monkeypatch.setattr(runner, 'predictions', predictions)
    monkeypatch.setattr(runner.common, 'monthly_rows', rows)
    monkeypatch.setattr(runner.v31, 'scoring_labels', lambda _: (labels, support))
    runner.score()
    case = pd.read_csv(out / 'case_metrics.csv')
    macro = pd.read_csv(out / 'macro_metrics.csv')
    assert len(case) == 385
    assert len(pd.read_csv(out / 'matched_contrasts.csv')) == 168
    assert (macro.cif_abs_error == 3.).all()  # Equal region weight, not 35-month pooling.
    assert (out / 'label_support.csv').read_bytes() == support_path.read_bytes()
    assert not runner.read(out / 'source_completed.json')['mechanism_breakthrough']
