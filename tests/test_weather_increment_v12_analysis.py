"""Source-only choices, two-round closure and fail-closed V12 evidence gates."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_weather_increment_v12 as analysis
from scripts.analyze_weather_recomposition_v11 import METRICS, paired, take


def days(value=10.):
    table = pd.DataFrame([
        dict(region=region, origin=f'2026-{month:02d}-{day:02d}',
             **{metric: value for metric in METRICS})
        for region, count in [('PACE', 2), ('LGEE', 1)]
        for month in range(1, 7) for day in range(1, count+1)
    ])
    table['empirical_coverage90'] = .92
    return table


def grid(round_number):
    from scripts.run_weather_increment_v12 import configurations
    frames, selections = [], {}
    for c in configurations(round_number):
        b, m = c['budget'], c['model']
        record = dict(budget=b, model=m, factor=1. if b == 'B' else 2., status='passed',
                      source_metrics=dict(violation=0., crps=10.))
        if m == 'qrf_25':
            record['source_metrics']['crps'] = 8.
        if m == 'qrf_50':
            record.update(status='failed', source_metrics=dict(violation=.02, crps=.01))
        selections[f'{b}--{m}'] = record
        for factor in (1., 2.):
            frames.append(days(20. if factor == 1 else 1.).assign(
                stage='target', semester='H1', lag=7, budget=b, model=m, factor=factor))
    primary = selections['W--donor_a050' if round_number == 1 else 'W--qrf_25']
    choices = dict(selections=selections, primary=primary,
                   baseline=selections['B--global_total'],
                   native=selections['B--native_recompose'], simple=selections['B--simple_anchor'],
                   diagnostic_crps_best=selections['W--qrf_0'])
    if round_number == 1:
        choices['locked_alpha'] = .5
    return pd.concat(frames, ignore_index=True), choices


def test_native_comparisons_keep_source_factor_even_when_target_prefers_another():
    table, choice = grid(1)
    contrasts = analysis.build_contrasts(table, choice['selections'], choice['primary'], 1, n_bootstrap=10)
    fixed = contrasts[contrasts.comparison.str.startswith('fixed_native_factor')]
    np.testing.assert_array_equal(fixed.primary_factor, 1.)
    np.testing.assert_array_equal(fixed.reference_factor, 1.)
    np.testing.assert_allclose(fixed.difference, 0.)
    own = contrasts[contrasts.comparison.str.startswith('own_selected') & contrasts.metric.eq('crps')]
    np.testing.assert_array_equal(own.primary_factor, 2.)
    np.testing.assert_array_equal(own.reference_factor, 1.)
    np.testing.assert_allclose(own.difference, -19.)
    assert {'primary_minus_W_global_total', 'primary_minus_W_hetero_total',
            'primary_minus_W_qrf_0'}.issubset(set(contrasts.comparison))


def test_round2_matches_B_factors_and_excludes_primary_self_reference():
    table, choice = grid(2)
    contrasts = analysis.build_contrasts(table, choice['selections'], choice['primary'], 2,
                                        baseline=choice['baseline'], n_bootstrap=10)
    fixed = contrasts[contrasts.comparison.str.startswith('fixed_B_factor')]
    np.testing.assert_array_equal(fixed.primary_factor, 1.)
    np.testing.assert_array_equal(fixed.reference_factor, 1.)
    assert 'primary_minus_W_qrf_25' not in set(contrasts.comparison)
    baseline = contrasts[contrasts.comparison.eq('primary_minus_Bbaseline')]
    assert set(baseline.reference_model) == {'global_total'}
    assert set(baseline.reference_budget) == {'B'}
    best = contrasts[contrasts.comparison.eq('Wdonor_minus_source_best_Wqrf')]
    assert set(best.reference_model) == {'qrf_25'}
    shrink = contrasts[contrasts.comparison.str.startswith('fixed_W_qrf0_factor')]
    np.testing.assert_array_equal(shrink.primary_factor, 2.)
    np.testing.assert_array_equal(shrink.reference_factor, 2.)


def test_mismatched_scoring_days_fail_instead_of_silent_inner_join():
    table, choice = grid(1)
    index = table[table.model.eq('donor_a025') & table.factor.eq(1.)].index[0]
    with pytest.raises(ValueError, match='identical paired'):
        analysis.build_contrasts(table.drop(index), choice['selections'], choice['primary'], 1,
                                 n_bootstrap=10)
    with pytest.raises(ValueError, match='unique region/day'):
        paired(pd.concat([days(), days().iloc[:1]]), days(), n_bootstrap=10)


def test_source_best_quantile_uses_gate_before_CRPS_and_failed_violation_before_CRPS():
    _, choice = grid(2)
    selections = deepcopy(choice['selections'])
    assert analysis.source_best_quantile(selections)['model'] == 'qrf_25'
    selections['W--qrf_0']['source_metrics']['crps'] = 100.
    selections['W--qrf_25'].update(status='failed', source_metrics=dict(violation=.01, crps=0.))
    assert analysis.source_best_quantile(selections)['model'] == 'qrf_0'
    selections['W--qrf_0'].update(status='failed', source_metrics=dict(violation=.03, crps=100.))
    assert analysis.source_best_quantile(selections)['model'] == 'qrf_25'
    selections['W--qrf_25']['source_metrics']['crps'] = 1000.
    assert analysis.source_best_quantile(selections)['model'] == 'qrf_25'
    selections['W--qrf_25']['source_metrics']['violation'] = np.nan
    with pytest.raises(ValueError, match='ranking requires'):
        analysis.source_best_quantile(selections)


def test_quantile_reference_does_not_follow_target_best():
    table, choice = grid(2)
    table.loc[table.model.eq('qrf_50'), 'crps'] = 0.
    contrasts = analysis.build_contrasts(table, choice['selections'], choice['primary'], 2,
                                        baseline=choice['baseline'], n_bootstrap=10)
    best = contrasts[contrasts.comparison.eq('Wdonor_minus_source_best_Wqrf')]
    assert set(best.reference_model) == {'qrf_25'}


def test_locked_strength_and_prior_round_identity_are_required(tmp_path):
    r1, r2 = tmp_path/'r1', tmp_path/'r2'
    f1 = dict(round=1)
    f2 = dict(round=2, prior_round=str(r1), alpha=.5, target_scores_used_for_round_lock=False)
    choice = dict(locked_alpha=.5)
    analysis.validate_round_lock(r1, r2, f1, f2, choice)
    for change in (dict(alpha=1.), dict(prior_round=str(tmp_path/'other')),
                   dict(target_scores_used_for_round_lock=True), dict(round=1)):
        with pytest.raises(ValueError, match='source-only strength lock'):
            analysis.validate_round_lock(r1, r2, f1, {**f2, **change}, choice)


def test_practical_and_strong_donor_are_separate_and_never_open_new_regions():
    practical = {key: True for key in analysis.PRACTICAL_REQUIRED_CHECKS}
    donor = {key: True for key in analysis.DONOR_REQUIRED_CHECKS}
    result = analysis.holdout_decision(practical, donor)
    assert result['practical_gain_gate_passed'] and result['strong_donor_gate_passed']
    assert not result['custom_innovation_gate_passed']
    assert not result['new_holdout_triggered']
    assert not result['confirmation_eligible_track']
    assert result['quantile_strong_baseline_completed']
    result = analysis.holdout_decision(practical, {**donor, 'Whetero_crps_at_least_5pct_lower': False})
    assert result['practical_gain_gate_passed'] and not result['strong_donor_gate_passed']
    assert not analysis.holdout_decision({}, {})['performance_gate_passed']
    for name in analysis.PRACTICAL_REQUIRED_CHECKS:
        missing = {key: True for key in analysis.PRACTICAL_REQUIRED_CHECKS if key != name}
        result = analysis.holdout_decision(missing, donor)
        assert not result['practical_gain_gate_passed']
        assert result['practical_gain']['missing_checks'] == [name]


def test_performance_checks_missing_or_nonfinite_evidence_fails_closed():
    table, choice = grid(2)
    contrasts = analysis.build_contrasts(table, choice['selections'], choice['primary'], 2,
                                        baseline=choice['baseline'], n_bootstrap=10)
    args = dict(prefix='primary_minus_', references=analysis.PRACTICAL_REFERENCES)
    checks = analysis.performance_checks(take(table, choice['primary']), choice['primary'], contrasts, **args)
    assert checks['source_gate_passed']
    assert checks['target_worst_region_coverage_at_least_85pct']
    with pytest.raises(ValueError, match='Missing gate metrics'):
        analysis.performance_checks(take(table, choice['primary']), choice['primary'],
            contrasts[~contrasts.comparison.eq('primary_minus_Bbaseline')], **args)
    bad = contrasts.copy()
    bad.loc[bad.comparison.eq('primary_minus_native') & bad.metric.eq('crps'), 'ci95_high'] = np.nan
    with pytest.raises(ValueError, match='Nonfinite gate metrics'):
        analysis.performance_checks(take(table, choice['primary']), choice['primary'], bad, **args)


def test_full_analysis_verifies_four_phases_and_binds_immutable_outputs(tmp_path, monkeypatch):
    from scripts import run_weather_increment_v12 as runner
    rounds = {i: tmp_path/f'round{i}' for i in (1, 2)}
    freezes, seals, calls = {}, {}, []
    for number, path in rounds.items():
        path.mkdir()
        table, choice = grid(number)
        (path/'source_choices.json').write_text(json.dumps(choice))
        frozen = dict(round=number, prior_round=str(rounds[1]) if number == 2 else None,
                      alpha=.5 if number == 2 else 1., target_scores_used_for_round_lock=False,
                      code_hashes={}, input_hashes={})
        freezes[path] = frozen
        (path/'prefit_freeze.json').write_text(json.dumps(frozen))
        for stage in ('source', 'target'):
            folder = path/f'{stage}_scores'
            folder.mkdir()
            scores = table.assign(stage=stage)
            if stage == 'source':
                scores['origin'] = scores.origin.str.replace('2026', '2025')
            for name in ('day_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv'):
                scores.to_csv(folder/name, index=False)
            (folder/'score_seal.json').write_text('{}')
            seal = dict(entries=[], other_round_source=str(rounds[3-number]))
            seals[path, stage] = seal
            (path/f'{stage}_prediction_seal.json').write_text(json.dumps(seal))
            (path/f'{stage}_completed.json').write_text('{}')
    def verify_phase(path, stage):
        calls.append((path, stage))
        return seals[path, stage]
    monkeypatch.setattr(runner, 'verify_phase', verify_phase)
    monkeypatch.setattr(runner, 'verify', lambda path: freezes[path])
    protocol = tmp_path/'protocol.md'
    protocol.write_text('frozen test protocol')
    monkeypatch.setattr(runner, 'PROTOCOL', protocol)
    original = analysis.build_contrasts
    monkeypatch.setattr(analysis, 'build_contrasts', lambda *args, **kwargs:
                        original(*args, **{**kwargs, 'n_bootstrap': 10}))
    original_paired = analysis.paired
    monkeypatch.setattr(analysis, 'paired', lambda *args, **kwargs:
                        original_paired(*args, **{**kwargs, 'n_bootstrap': 10}))
    output = tmp_path/'analysis'
    result = analysis.analyze(rounds[1], rounds[2], output)
    assert calls == [(rounds[i], stage) for i in (1, 2) for stage in ('source', 'target')]
    assert not result['new_holdout_triggered']
    metadata = json.loads((output/'analysis_metadata.json').read_text())
    assert metadata['adaptive_rounds'] and not metadata['independent_confirmation']
    assert metadata['locked_alpha'] == .5
    assert metadata['source_best_Wqrf']['model'] == 'qrf_25'
    assert len(metadata['verified_phases']) == 4
    for name, digest in metadata['output_hashes'].items():
        assert analysis.sha(output/name) == digest
    selected_rows = pd.read_csv(output/'selected_macro_metrics.csv')
    assert set(selected_rows['round']) == {1, 2}
    assert {'primary', 'baseline', 'native', 'simple', 'diagnostic_crps_best',
            'Wdonor', 'source_best_Wqrf'}.issubset(set(selected_rows.selection_role))
    comparisons = pd.read_csv(output/'paired_month_bootstrap.csv')
    cross = comparisons[comparisons.comparison.eq('round2_primary_minus_round1_primary')]
    assert cross.adaptive_round_comparison.all()
    assert set(cross.stage) == {'source', 'target'}
    with pytest.raises(FileExistsError):
        analysis.analyze(rounds[1], rounds[2], output)

