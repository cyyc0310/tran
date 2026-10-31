"""Synthetic-only posthoc error decomposition and sealed-output I/O checks."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_source_risk_v49 as analysis


def errors(n=2):
    truth = np.full((n, 24), 500.)
    level = np.resize(np.array([3., -4.]), n)
    shape = np.tile([-2., 2.], 12)
    error = np.broadcast_to(level[None, :, None]+shape, (8, n, 24)).copy()
    point = truth[None]-error
    return point, truth, error, error.mean(-1), error-error.mean(-1, keepdims=True)


def test_orthogonal_decomposition_has_known_closed_values():
    d = analysis.decompose(*errors())
    np.testing.assert_array_equal(d['hourly_mse'][0], [13., 20.])
    np.testing.assert_array_equal(d['level_mse'][0], [9., 16.])
    np.testing.assert_array_equal(d['shape_mse'][0], [4., 4.])
    np.testing.assert_array_equal(d['signed_level_bias'][0], [3., -4.])
    np.testing.assert_array_equal(d['mean_abs_level_error'][0], [3., 4.])
    np.testing.assert_array_equal(d['shape_mae'][0], [2., 2.])
    assert d['identity_max_abs'] == 0


@pytest.mark.parametrize('kind', ['shape', 'nonorthogonal', 'level', 'hourly', 'complex', 'nan'])
def test_bad_error_arrays_fail_independent_truth_check(kind):
    values = list(errors())
    if kind == 'shape': values[4] = values[4][..., :-1]
    elif kind == 'nonorthogonal': values[4] += 1
    elif kind == 'level': values[3] += 1
    elif kind == 'hourly': values[2] += 1
    elif kind == 'complex': values[0] = values[0].astype(complex)
    else: values[1][0, 0] = np.nan
    with pytest.raises(ValueError):
        analysis.decompose(*values)


def task_table():
    rows = []
    for held_id, held in enumerate(analysis.SOURCES):
        for outer_id, outer in enumerate(analysis.SOURCES):
            if outer == held:
                continue
            # Unequal held counts must not induce a day-weighted macro.
            days = 1 if held_id == 0 else 100
            level = float(held_id+outer_id+1)
            for panel in analysis.PANELS:
                for arm, budget in zip(analysis.ARMS, analysis.BUDGETS):
                    rows.append(dict(outer=outer, held=held, panel=panel, arm=arm, budget=budget,
                        valid_days=days, hourly_mse=level**2+4, level_mse=level**2, shape_mse=4.,
                        signed_level_bias=level, mean_abs_level_error=level, shape_mae=2.,
                        identity_max_abs=0.))
    return pd.DataFrame(rows)


def test_equal_held_and_outer_means_then_rmse_and_fraction():
    tasks = task_table()
    held, macro = analysis.aggregate(tasks)
    assert len(held) == 112 and len(macro) == 16
    subset = tasks[(tasks.panel == analysis.PANELS[0]) & (tasks.arm == analysis.ARMS[0])]
    expected = subset.groupby('held').hourly_mse.mean().mean()
    value = macro.iloc[0]
    assert value.hourly_mse == pytest.approx(expected)
    assert value.hourly_rmse == pytest.approx(np.sqrt(expected))
    assert abs(value.hourly_rmse-subset.groupby('held').hourly_mse.mean().pow(.5).mean()) > .01
    assert value.level_mse_fraction == pytest.approx(value.level_mse/value.hourly_mse)
    assert abs(value.hourly_mse-np.average(subset.hourly_mse, weights=subset.valid_days)) > 1
    with pytest.raises(ValueError):
        analysis.aggregate(tasks.iloc[1:])


def test_zero_error_fraction_is_explicit_null():
    tasks = task_table()
    for name in analysis.MEAN_METRICS:
        tasks[name] = 0.
    held, macro = analysis.aggregate(tasks)
    assert macro.level_mse_fraction.isna().all() and held.level_mse_fraction.isna().all()
    assert np.all(macro.hourly_rmse == 0)
    point = np.full((8, 2, 24), 500.)
    d = analysis.decompose(point, point[0], np.zeros_like(point), np.zeros((8, 2)), np.zeros_like(point))
    np.testing.assert_array_equal(d['hourly_mse'], 0)


def save_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def fixture_source(root, zero=False):
    root.mkdir()
    for folder in ('risk', 'labels'):
        (root/folder).mkdir()
    for name in ('freeze.json', 'prediction_seal.json', 'label_started.json'):
        save_json(root/name, {})
    origins = analysis.DAYS.asi8[[0, 158]]
    labels, entries, outer_tasks = [], [], {s: [] for s in analysis.SOURCES}
    for held in analysis.SOURCES:
        keys = np.array([f'{held}/2024-01-25', f'{held}/2024-07-01'])
        p = root/'labels'/f'{held}.npz'
        np.savez(p, origins=origins, truth=np.full((2, 24), 500.), positions=[0, 158], keys=keys,
                 panels=np.array(analysis.PANELS))
        labels.append(dict(region=held, path=str(p), sha256=analysis.sha(p), days=2))
        for outer in analysis.SOURCES:
            if outer == held:
                continue
            p = root/'risk'/f'{outer}__{held}.npz'
            point, truth, hourly, level, shape = errors()
            if zero:
                point[:] = truth
                hourly[:] = 0; level[:] = 0; shape[:] = 0
            np.savez(p, origins=origins, arms=np.array(analysis.ARMS), budgets=np.array(analysis.BUDGETS),
                point=point, hourly_error=hourly, level_error=level, shape_error=shape,
                keys=keys, panels=np.array(analysis.PANELS))
            entries.append(dict(outer=outer, held=held, pair=sorted((outer, held)), label_region=held,
                risk_path=str(p), risk_sha256=analysis.sha(p), source_label_path=str(root/'labels'/f'{held}.npz'),
                valid_days=2))
            outer_tasks[outer].append(str(p))
    index = dict(entries=entries, labels=labels, outer_tasks=outer_tasks)
    save_json(root/'task_index.json', index)
    save_json(root/'summary.json', dict(risk_models_fitted=False, source_2025_truth_read=False,
        directed_tasks=42, bases=8, source_label_days=14, directed_valid_days=84, base_day_examples=672))
    for name in analysis.EXTRA_TABLES:
        (root/name).write_text('metadata_only\n')
    outputs = [*root.glob('risk/*.npz'), *root.glob('labels/*.npz'), root/'task_index.json', root/'summary.json',
               *(root/name for name in analysis.EXTRA_TABLES)]
    save_json(root/'completed.json', dict(status='COMPLETE', source_2025_truth_read=False,
        target_values_opened=False, unseen_values_opened=False,
        freeze_sha256=analysis.sha(root/'freeze.json'), prediction_seal_sha256=analysis.sha(root/'prediction_seal.json'),
        label_started_sha256=analysis.sha(root/'label_started.json'), output_hashes={str(p): analysis.sha(p) for p in outputs}))
    return root


def test_synthetic_completed_outputs_only_and_exclusive_destination(tmp_path):
    source = fixture_source(tmp_path/'source')
    output = tmp_path/'diagnostics'
    analysis.run(source, output)
    summary = json.loads((output/'summary.json').read_text())
    assert summary['new_fits'] == 0 and summary['posthoc'] is True
    assert summary['rows'] == dict(task_panel=672, held_panel=112, macro=16)
    assert summary['correlations_computed'] is False
    macro = pd.read_csv(output/'macro.csv')
    assert np.allclose(macro.loc[macro.panel == analysis.PANELS[0], 'level_mse_fraction'], 9/13)
    assert np.allclose(macro.loc[macro.panel == analysis.PANELS[1], 'level_mse_fraction'], 16/20)
    frozen = json.loads((output/'freeze.json').read_text())
    assert frozen['input_hashes'][str(source/'completed.json')] == analysis.sha(source/'completed.json')
    assert analysis.sha(analysis.Path(analysis.__file__)) in frozen['code_hashes'].values()
    with pytest.raises(FileExistsError):
        analysis.run(tmp_path/'nonexistent', output)


def test_corrupt_hash_precedes_npz_loading(tmp_path, monkeypatch):
    source = fixture_source(tmp_path/'source')
    (source/'risk'/'BPAT__CISO.npz').write_bytes(b'changed')
    monkeypatch.setattr(np, 'load', lambda *_args, **_kwargs: pytest.fail('arrays opened before hashes passed'))
    with pytest.raises(ValueError, match='hash'):
        analysis.run(source, tmp_path/'diagnostics')
    assert not (tmp_path/'diagnostics').exists()


def test_zero_fraction_is_json_null_and_empty_csv(tmp_path):
    source = fixture_source(tmp_path/'source', zero=True)
    output = tmp_path/'diagnostics'
    analysis.run(source, output)
    summary = json.loads((output/'summary.json').read_text())
    assert all(row['level_mse_fraction'] is None for row in summary['macro_results'])
    assert pd.read_csv(output/'macro.csv').level_mse_fraction.isna().all()
    assert 'NaN' not in (output/'summary.json').read_text()


def test_current_task_index_risk_hash_link_is_independently_checked(tmp_path, monkeypatch):
    source = fixture_source(tmp_path/'source')
    index_path = source/'task_index.json'
    index = json.loads(index_path.read_text())
    index['entries'][0]['risk_sha256'] = 'f'*64
    save_json(index_path, index)
    completed = json.loads((source/'completed.json').read_text())
    completed['output_hashes'][str(index_path)] = analysis.sha(index_path)
    save_json(source/'completed.json', completed)
    monkeypatch.setattr(np, 'load', lambda *_args, **_kwargs: pytest.fail('arrays opened before index links passed'))
    with pytest.raises(ValueError, match='risk hash'):
        analysis.run(source, tmp_path/'diagnostics')
