"""Selection leakage, paired-support and fail-closed V11 analysis tests."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_weather_recomposition_v11 import (
    ARMS, METRICS, REQUIRED_CHECKS, build_contrasts, holdout_decision,
    paired, performance_checks, source_crps_best, take,
)


def days(value=10.):
    return pd.DataFrame([
        dict(region=region, origin=f'2026-{month:02d}-{day:02d}',
             **{metric: value for metric in METRICS})
        for region, n in [('A', 2), ('B', 1)]
        for month in range(1, 7) for day in range(1, n + 1)
    ])


def grid():
    selections, frames = {}, []
    configs = [(b, m) for b in ('B', 'W') for m in ARMS]
    configs += [('B', 'native_recompose'), ('B', 'simple_anchor'),
                ('W', 'global_Brisk'), ('W', 'hetero_Brisk')]
    for budget, model in configs:
        factor = 1. if budget == 'B' else 2.
        selections[f'{budget}--{model}'] = dict(configuration=dict(
            budget=budget, model=model, factor=factor), status='passed')
        for f in (1., 2.):
            # Deliberately different best target factor; source choice must win.
            frames.append(days(20. if f == 1. else 1.).assign(
                lag=7, budget=budget, model=model, factor=f))
    return pd.concat(frames, ignore_index=True), selections


def test_synchronous_month_pair_is_equal_region_and_matches_full_support():
    left = days()
    left.loc[left.region.eq('A'), METRICS] = 2.
    left.loc[left.region.eq('B'), METRICS] = 8.
    right = left.copy()
    right[METRICS] = 0.
    result = paired(left, right, n_bootstrap=50)
    overall = result[result.label.eq('overall')]
    np.testing.assert_allclose(overall.primary_mean, 5.)
    np.testing.assert_allclose(overall.ci95_low, 5.)
    np.testing.assert_allclose(overall.ci95_high, 5.)
    with pytest.raises(ValueError, match='identical paired'):
        paired(left, right.iloc[:-1], n_bootstrap=50)
    with pytest.raises(ValueError, match='unique region/day'):
        paired(pd.concat([left, left.iloc[:1]]), right, n_bootstrap=50)
    with pytest.raises(ValueError, match='Six fixed H1'):
        paired(left.iloc[:1], right.iloc[:1], n_bootstrap=50)


def test_fixed_source_factor_and_counterfactuals_never_use_target_optimum():
    table, selections = grid()
    primary = selections['W--donor']['configuration']
    result = build_contrasts(table, selections, primary, n_bootstrap=20)
    fixed = result[result.comparison.str.startswith('fixed_')]
    np.testing.assert_allclose(fixed.primary_factor, 1.)
    np.testing.assert_allclose(fixed.reference_factor, 1.)
    np.testing.assert_allclose(fixed.difference, 0.)
    selected = result[result.comparison.eq('own_selected_W_minus_B_donor')]
    np.testing.assert_allclose(selected.primary_factor, 2.)
    np.testing.assert_allclose(selected.reference_factor, 1.)
    np.testing.assert_allclose(selected.difference, -19.)
    assert set(result.comparison[result.comparison.str.startswith('primary_minus_')]) == {
        'primary_minus_native', 'primary_minus_simple', 'primary_minus_B_same_arm'}
    missing = table[~(table.budget.eq('W') & table.model.eq('donor') & table.factor.eq(1.))]
    with pytest.raises(ValueError, match='Missing fixed'):
        build_contrasts(missing, selections, primary, n_bootstrap=20)


def test_counterfactual_mismatched_days_fail_even_when_means_match():
    table, selections = grid()
    index = table[table.budget.eq('W') & table.model.eq('global_Brisk') &
                  table.factor.eq(1.)].index[0]
    with pytest.raises(ValueError, match='identical paired'):
        build_contrasts(table.drop(index), selections,
                        selections['W--donor']['configuration'], n_bootstrap=20)


def test_weather_proxy_never_opens_unseen_regions_and_missing_checks_fail():
    checks = {key: True for key in REQUIRED_CHECKS}
    passed = holdout_decision(checks)
    assert passed['performance_gate_passed']
    assert not passed['new_holdout_triggered']
    assert not passed['confirmation_eligible_track']
    assert not passed['custom_innovation_gate_passed']
    assert not passed['quantile_strong_baseline_completed']
    assert not holdout_decision({})['performance_gate_passed']
    omitted = dict(checks)
    omitted.pop('native_width_not_increased')
    assert holdout_decision(omitted)['missing_checks'] == ['native_width_not_increased']
    assert not holdout_decision(omitted)['performance_gate_passed']
    for key in REQUIRED_CHECKS:
        assert not holdout_decision({**checks, key: False})['performance_gate_passed']


def test_primary_gate_cannot_pass_without_complete_references():
    table, selections = grid()
    primary = selections['W--donor']['configuration']
    contrasts = build_contrasts(table, selections, primary, n_bootstrap=20)
    with pytest.raises(ValueError, match='Missing primary gate metrics'):
        performance_checks(take(table, primary), {'status': 'passed'},
                           contrasts[~contrasts.comparison.eq('primary_minus_native')])


def test_pure_crps_source_diagnostic_uses_equal_region_weights_only():
    table, _ = grid()
    winner = source_crps_best(table)
    assert winner == dict(budget='B', model='donor', factor=2.)
    with pytest.raises(ValueError, match='Missing fixed'):
        take(table, dict(budget='B', model='donor', factor=3.))


def test_nonfinite_paired_metric_is_rejected():
    left, right = days(), days()
    left.loc[0, 'level_crps'] = np.inf
    with pytest.raises(ValueError, match='Nonfinite paired'):
        paired(left, right, n_bootstrap=20)


def test_analysis_verifies_both_phases_and_writes_bound_artifacts(tmp_path, monkeypatch):
    import scripts
    from scripts import analyze_weather_recomposition_v11 as analysis

    run, output = tmp_path / 'run', tmp_path / 'analysis'
    run.mkdir()
    table, wrapped = grid()
    selections = {key: {**record['configuration'], 'status': record['status']}
                  for key, record in wrapped.items()
                  if not key.endswith(('global_Brisk', 'hetero_Brisk'))}
    choices = dict(selections=selections, primary=selections['W--donor'],
                   baseline=selections['B--donor'], native=selections['B--native_recompose'],
                   simple=selections['B--simple_anchor'])
    (run / 'source_choices.json').write_text(json.dumps(choices))
    (run / 'prefit_freeze.json').write_text('{}')
    for stage in ('source', 'target'):
        folder = run / f'{stage}_scores'
        folder.mkdir()
        scores = table.assign(stage=stage, semester='H1')
        for name in ('day_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv'):
            scores.to_csv(folder / name, index=False)
        (folder / 'score_seal.json').write_text('{}')
        for suffix in ('completed', 'prediction_seal'):
            (run / f'{stage}_{suffix}.json').write_text('{}')
    calls = []
    fake = SimpleNamespace(METRICS=METRICS, verify_phase=lambda path, stage: calls.append(stage))
    monkeypatch.setattr(scripts, 'run_weather_recomposition_v11', fake, raising=False)
    original = analysis.build_contrasts
    monkeypatch.setattr(analysis, 'build_contrasts',
                        lambda *args, **kwargs: original(*args, n_bootstrap=20))
    decision = analysis.analyze(run, output)
    assert calls == ['source', 'target']
    assert not decision['new_holdout_triggered']
    metadata = json.loads((output / 'analysis_metadata.json').read_text())
    assert metadata['verified_phases'] == ['source', 'target']
    assert not metadata['target_selection_performed']
    assert set(metadata['output_hashes']) == {
        'selected_region_metrics.csv', 'selected_macro_metrics.csv',
        'paired_month_bootstrap.csv', 'holdout_trigger.json'}
    for name, digest in metadata['output_hashes'].items():
        assert analysis.sha(output / name) == digest
    selected = pd.read_csv(output / 'selected_macro_metrics.csv')
    assert {'baseline', 'native', 'simple', 'primary',
            'source_pure_crps_best_diagnostic'}.issubset(selected.selection_role)
    with pytest.raises(FileExistsError):
        analysis.analyze(run, output)
