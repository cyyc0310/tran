"""Prediction/score isolation and unchanged reference-grid tests for V37."""
import copy

import numpy as np
import pandas as pd
import pytest

from scripts import run_utilization_decoder_v37 as runner


@pytest.mark.parametrize('entry', ['probability_theta', 'probability_design', 'objective_design', 'training_features'])
def test_complex_inputs_rejected_before_float_conversion(entry):
    theta = np.zeros((1, 3))
    design = np.ones((1, 1))
    with pytest.raises(ValueError, match='Real'):
        if entry == 'probability_theta':
            runner.model.probabilities(theta + 1j, design)
        elif entry == 'probability_design':
            runner.model.probabilities(theta, design + 1j)
        elif entry == 'objective_design':
            runner.model.objective(theta.ravel(), design + 1j, np.ones((1, 4)) / 4, np.ones(1))
        else:
            runner.model.fit_predict(np.ones((1, 36)) + 1j, np.ones((1, 4)) / 4, np.ones(1),
                np.ones((1, 4)) / 4, np.ones((6, 128, 36)), np.ones((6, 4)) / 4, ['toy'])


def test_prepare_refuses_existing_attempt_before_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    monkeypatch.setattr(runner, 'contract', lambda: pytest.fail('No input access'))
    with pytest.raises(FileExistsError):
        runner.prepare()


def test_score_refuses_unsealed_grid_before_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    monkeypatch.setattr(runner, 'check', lambda: {})
    monkeypatch.setattr(runner, 'predictions', lambda: (_ for _ in ()).throw(ValueError('unsealed')))
    monkeypatch.setattr(runner.common, 'monthly_rows', lambda _: pytest.fail('Labels accessed early'))
    with pytest.raises(ValueError, match='unsealed'):
        runner.score()


def test_optimizer_failure_keeps_marker_and_blocks_retry(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'freeze.json').write_text('{}')
    audit_path = tmp_path / 'source.json'
    fit = dict(arm='SizeCap-Ridge', training_keys=['CISO/2022-01'], training_regions=['CISO'],
               training_X=[[0.] * 36], training_y=[[0., 1., 0., 0.]], sample_weights=[1.],
               training_period_start_ns=[1], training_period_end_ns=[2], training_available_ns=[3])
    peer = copy.deepcopy(fit)
    peer['arm'] = 'SizeCap-ET'
    runner.write(audit_path, dict(fits=[fit, peer]))
    frozen = dict(entries=[dict(region='BPAT', path='unopened', audit_path=str(audit_path))])
    monkeypatch.setattr(runner, 'OUT', out)
    monkeypatch.setattr(runner, 'check', lambda: frozen)
    monkeypatch.setattr(runner.v36, 'load_archive', lambda *args: dict(query_feature_SizeCap=np.zeros((6, 128, 36))))
    monkeypatch.setattr(runner.v36, 'source_inputs', lambda *args: (np.ones((1, 4)) / 4, np.ones((6, 4)) / 4, [], {}))
    monkeypatch.setattr(runner.model, 'fit_predict', lambda *args: (_ for _ in ()).throw(ValueError('synthetic optimizer failure')))
    with pytest.raises(ValueError, match='synthetic optimizer failure'):
        runner.predict()
    receipt = (out / 'prediction_started.json').read_bytes()
    assert not (out / 'prediction_seal.json').exists()
    with pytest.raises(FileExistsError):
        runner.predict()
    assert (out / 'prediction_started.json').read_bytes() == receipt


def test_score_copies_full_reference_grid_and_adds_same_support(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    out.mkdir()
    for name in ('freeze.json', 'prediction_started.json', 'prediction_seal.json'):
        (out / name).write_text('{}')
    counts = dict(zip(runner.SOURCES, (6, 6, 6, 6, 1, 5, 5)))
    labels, support, baseline, entries = {}, [], [], []
    for source in runner.SOURCES:
        keys = [f'{source}/{at:%Y-%m}' for at in runner.DATES]
        for j, key in enumerate(keys):
            present = j < counts[source]
            support.append(dict(region=source, key=key, status='scored' if present else 'missing'))
            if not present:
                continue
            labels[key] = dict(cif=120., ef=600., ef_defined=True)
            for arm in runner.v36.model.RECIPES:
                for mode in ('mean', 'median'):
                    baseline.append(dict(region=source, key=key, recipe=arm, mode=mode,
                        cif_point=140., ef_point=620., cif_bias=20., cif_abs_error=20., cif_sq_error=400.,
                        ef_bias=20., ef_abs_error=20., ef_sq_error=400.))
        path = tmp_path / f'{source}.npz'
        np.savez(path, keys=keys, CIF_draws=np.full((3, 6, 128), 130.), EF_draws=np.full((3, 6, 128), 650.))
        entries.append(dict(region=source, path=str(path)))
    baseline_path, support_path = tmp_path / 'baseline.csv', tmp_path / 'support.csv'
    pd.DataFrame(baseline).to_csv(baseline_path, index=False)
    support = pd.DataFrame(support)
    support.to_csv(support_path, index=False)
    frozen = dict(baseline_case_path=str(baseline_path), support_path=str(support_path))
    monkeypatch.setattr(runner, 'OUT', out)
    monkeypatch.setattr(runner, 'check', lambda: frozen)
    monkeypatch.setattr(runner, 'predictions', lambda: dict(entries=entries))
    monkeypatch.setattr(runner.common, 'monthly_rows', lambda _: [])
    monkeypatch.setattr(runner.v31, 'scoring_labels', lambda _: (labels, support))
    runner.score()
    case = pd.read_csv(out / 'case_metrics.csv', float_precision='round_trip')
    original = pd.read_csv(baseline_path, float_precision='round_trip').rename(columns={'recipe': 'arm'})
    assert case.iloc[:700].reset_index(drop=True).equals(original)
    assert len(case) == 910
    assert len(pd.read_csv(out / 'region_metrics.csv')) == 182
    assert len(pd.read_csv(out / 'macro_metrics.csv')) == 26
    assert len(pd.read_csv(out / 'leave_one_source_out_macro.csv')) == 182
    macro = pd.read_csv(out / 'macro_metrics.csv')
    assert (macro[macro.arm.isin(runner.model.ARMS)].cif_abs_error == 10.).all()
    assert (out / 'label_support.csv').read_bytes() == support_path.read_bytes()
    assert not runner.read(out / 'source_completed.json')['new_holdout_authorized']
