"""V14 source policy locks, standard/physical references and fail-closed gates."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_weather_coarse_v14 as a
from scripts.analyze_weather_recomposition_v11 import METRICS, take


def days(value=10.):
    frame = pd.DataFrame([dict(region=r, origin=f'2026-{m:02d}-01',
        **{metric: value for metric in METRICS}) for r in ('PACE', 'LGEE') for m in range(1, 7)])
    frame['empirical_coverage90'] = .92
    return frame


def grid(number):
    from scripts.run_weather_coarse_v14 import configurations
    frames, selections = [], {}
    for c in configurations(number):
        budget, model = c['budget'], c['model']
        factor = 1. if budget == 'B' or model.startswith('level_') else 2.
        crps = 8. if model == 'qrf_0' else (9. if model == 'physical_rank' else 10.)
        record = dict(budget=budget, model=model, factor=factor, status='passed',
                      source_metrics=dict(violation=0., crps=crps))
        if model == 'qrf_50':
            record.update(status='failed', source_metrics=dict(violation=.02, crps=.01))
        selections[f'{budget}--{model}'] = record
        for f in (1., 2.):
            frames.append(days(20. if f == 1 else 1.).assign(
                stage='target', semester='H1', lag=7, budget=budget, model=model, factor=f))
    return pd.concat(frames, ignore_index=True), dict(selections=selections,
        primary=selections['W--gate_forest' if number == 1 else 'W--qrf_0'],
        baseline=selections['B--native_recompose'], native=selections['B--native_recompose'],
        simple=selections['B--simple_anchor'], diagnostic_crps_best=selections['W--fixed_donor'],
        locked_policy='gate_forest')


def test_weather_policy_contrasts_use_source_factors_not_target_optimum():
    table, c = grid(1)
    result = a.build_contrasts(table, c['selections'], c['primary'], 1, n_bootstrap=10)
    fixed = result[result.comparison.str.startswith('fixed_native_factor')]
    assert set(fixed.primary_factor) == {1.}
    assert set(fixed.reference_factor) == {1.}
    assert set(fixed.difference) == {0.}
    own = result[result.comparison.eq('own_selected_W_gate_forest_minus_native') & result.metric.eq('crps')]
    np.testing.assert_allclose(own.difference, -19.)
    assert {'primary_minus_W_fixed_donor', 'primary_minus_W_constant_half'}.issubset(set(result.comparison))


def test_physical_level_rank_and_weather_effects_have_matched_source_factors():
    table, c = grid(2)
    result = a.build_contrasts(table, c['selections'], c['primary'], 2,
                               baseline=c['baseline'], n_bootstrap=10)
    sameB = result[result.comparison.str.startswith('fixed_B_factor')]
    assert set(sameB.primary_factor) == {1.} and set(sameB.reference_factor) == {1.}
    physical = result[result.comparison.eq('fixed_level_factor_physical_minus_level_W_rank')]
    assert set(physical.primary_factor) == {1.} and set(physical.reference_factor) == {1.}
    rank = result[result.comparison.eq('fixed_independent_factor_rank_minus_independent_W_physical')]
    assert set(rank.primary_factor) == {2.} and set(rank.reference_factor) == {2.}
    mechanism = result[result.comparison.eq('mechanism_minus_matched_level')]
    assert set(mechanism.primary_model) == {'physical_rank'}
    assert set(mechanism.reference_model) == {'level_rank'}
    assert 'primary_minus_Wstandard' not in set(result.comparison)
    assert 'primary_minus_W_qrf_0' not in set(result.comparison)


def test_source_reference_ranking_ignores_target_and_fails_without_source_evidence():
    table, c = grid(2)
    table.loc[table.model.eq('qrf_50'), 'crps'] = 0.
    assert a.source_best(c['selections'], 'W', a.STANDARD_FAMILIES)['model'] == 'qrf_0'
    records = deepcopy(c['selections'])
    for model in a.STANDARD_FAMILIES:
        records['W--'+model].update(status='failed', source_metrics=dict(violation=.03, crps=10.))
    records['W--qrf_50']['source_metrics']['violation'] = .01
    assert a.source_best(records, 'W', a.STANDARD_FAMILIES)['model'] == 'qrf_50'
    records['W--qrf_50']['source_metrics']['violation'] = np.nan
    with pytest.raises(ValueError, match='ranking requires'):
        a.source_best(records, 'W', a.STANDARD_FAMILIES)


def test_scoring_support_mismatch_and_missing_factor_fail_closed():
    table, c = grid(1)
    index = table[table.model.eq('gate_forest') & table.factor.eq(1.)].index[0]
    with pytest.raises(ValueError, match='identical paired'):
        a.build_contrasts(table.drop(index), c['selections'], c['primary'], 1, n_bootstrap=10)
    with pytest.raises(ValueError, match='Missing fixed'):
        take(table, dict(budget='B', model='native_recompose', factor=3.))


def test_policy_lock_requires_same_prior_source_choice_in_both_rounds(tmp_path):
    r1, r2 = tmp_path/'r1', tmp_path/'r2'
    _, c1 = grid(1)
    _, c2 = grid(2)
    f2 = dict(round=2, prior_round=str(r1), weather_policy='gate_forest',
              target_scores_used_for_round_lock=False)
    a.validate_round_lock(r1, r2, dict(round=1), f2, c1, c2)
    for change in (dict(weather_policy='fixed_donor'), dict(prior_round=str(r2)),
                   dict(target_scores_used_for_round_lock=True)):
        with pytest.raises(ValueError, match='source-only policy lock'):
            a.validate_round_lock(r1, r2, dict(round=1), {**f2, **change}, c1, c2)
    with pytest.raises(ValueError, match='source-only policy lock'):
        a.validate_round_lock(r1, r2, dict(round=1), f2, c1, {**c2, 'locked_policy':'gate_ridge'})


def test_practical_mechanism_self_reference_and_no_automatic_novelty():
    practical = {key: True for key in a.PRACTICAL_REQUIRED_CHECKS}
    mechanism = {key: True for key in a.MECHANISM_REQUIRED_CHECKS}
    result = a.holdout_decision(practical, mechanism)
    assert result['practical_gain_gate_passed'] and result['mechanism_gate_passed']
    assert not result['custom_innovation_gate_passed'] and not result['new_holdout_triggered']
    assert not result['confirmation_eligible_track']
    omitted = {key:value for key,value in practical.items() if not key.startswith('Wstandard_')}
    assert not a.holdout_decision(omitted, mechanism)['practical_gain_gate_passed']
    exempt = a.holdout_decision(omitted, mechanism, standard_self_reference=True)
    assert exempt['practical_gain_gate_passed']
    assert exempt['practical_gain']['self_reference_exemptions'] == ['Wstandard']
    assert not a.holdout_decision({}, {}, standard_self_reference=True)['performance_gate_passed']
    failed = a.holdout_decision(practical, {**mechanism, 'matched_level_width_not_increased':False})
    assert failed['practical_gain_gate_passed'] and not failed['mechanism_gate_passed']
    for key in a.MECHANISM_REQUIRED_CHECKS:
        checks = {name:True for name in a.MECHANISM_REQUIRED_CHECKS if name != key}
        assert not a.holdout_decision(practical, checks)['mechanism_gate_passed']


def test_cohort_repair_preserves_v13_analysis_decision_rules():
    from scripts import analyze_weather_coarse_v13 as previous
    practical = {key:True for key in a.PRACTICAL_REQUIRED_CHECKS}
    mechanism = {key:True for key in a.MECHANISM_REQUIRED_CHECKS}
    cases = [(practical, mechanism, False),
             ({**practical, 'target_worst_region_coverage_at_least_85pct':False}, mechanism, False),
             ({key:value for key,value in practical.items() if not key.startswith('Wstandard_')},
              {**mechanism, 'matched_level_crps_ci_upper_below_zero':False}, True),
             ({}, {}, False)]
    for p,m,self_reference in cases:
        assert a.holdout_decision(p,m,standard_self_reference=self_reference)==previous.holdout_decision(
            p,m,standard_self_reference=self_reference)


def test_missing_or_nonfinite_gate_metric_is_rejected():
    table, c = grid(2)
    result = a.build_contrasts(table, c['selections'], c['primary'], 2, baseline=c['baseline'], n_bootstrap=10)
    record = a.source_best(c['selections'], 'W', a.PHYSICAL_FAMILIES)
    args = dict(prefix='mechanism_minus_', references=a.MECHANISM_REFERENCES)
    assert a.performance_checks(take(table, record), record, result, **args)['source_gate_passed']
    with pytest.raises(ValueError, match='Missing gate metrics'):
        a.performance_checks(take(table, record), record,
            result[~result.comparison.eq('mechanism_minus_matched_level')], **args)
    result.loc[result.comparison.eq('mechanism_minus_native') & result.metric.eq('crps'), 'ci95_high'] = np.nan
    with pytest.raises(ValueError, match='Nonfinite gate'):
        a.performance_checks(take(table, record), record, result, **args)


def test_analysis_four_verified_phases_source_lock_self_case_and_bound_outputs(tmp_path, monkeypatch):
    from scripts import run_weather_coarse_v14 as runner
    rounds = {i:tmp_path/f'round{i}' for i in (1,2)}
    freezes, seals, calls = {}, {}, []
    for number,path in rounds.items():
        path.mkdir()
        table, choice = grid(number)
        (path/'source_choices.json').write_text(json.dumps(choice))
        frozen = dict(round=number, prior_round=str(rounds[1]) if number==2 else None,
            weather_policy='gate_forest' if number==2 else None,
            calibration_sources=['BPAT','CISO','ERCO','FPL','MISO','NYIS'],
            target_scores_used_for_round_lock=False, code_hashes={}, input_hashes={})
        freezes[path] = frozen
        (path/'prefit_freeze.json').write_text(json.dumps(frozen))
        for stage in ('source','target'):
            folder = path/f'{stage}_scores'
            folder.mkdir()
            scores = table.assign(stage=stage)
            if stage=='source':
                scores['origin'] = scores.origin.str.replace('2026','2025')
            for name in ('day_metrics.csv','region_metrics.csv','macro_metrics.csv'):
                scores.to_csv(folder/name,index=False)
            (folder/'score_seal.json').write_text('{}')
            seal = dict(entries=[],other_round_source=str(rounds[3-number]))
            seals[path,stage] = seal
            (path/f'{stage}_prediction_seal.json').write_text(json.dumps(seal))
            (path/f'{stage}_completed.json').write_text('{}')
    def verify_phase(path,stage):
        calls.append((path,stage))
        return seals[path,stage]
    monkeypatch.setattr(runner,'verify_phase',verify_phase)
    monkeypatch.setattr(runner,'verify',lambda path:freezes[path])
    protocol = tmp_path/'protocol.md'
    protocol.write_text('test frozen protocol')
    monkeypatch.setattr(runner,'PROTOCOL',protocol)
    original = a.paired
    monkeypatch.setattr(a,'paired',lambda *args,**kwargs:original(*args,**{**kwargs,'n_bootstrap':10}))
    output=tmp_path/'analysis'
    result=a.analyze(rounds[1],rounds[2],output)
    assert calls==[(rounds[i],stage) for i in (1,2) for stage in ('source','target')]
    assert not result['new_holdout_triggered']
    metadata=json.loads((output/'analysis_metadata.json').read_text())
    assert metadata['locked_policy']=='gate_forest'
    assert metadata['analysis_version']==14 and metadata['cohort_repair_only']
    assert metadata['calibration_sources']==['BPAT','CISO','ERCO','FPL','MISO','NYIS']
    assert metadata['practical_standard_self_reference']
    assert metadata['source_best_Wphysical']['model']=='physical_rank'
    assert metadata['source_best_Wstandard']['model']=='qrf_0'
    assert metadata['adaptive_rounds'] and not metadata['independent_confirmation']
    for name,digest in metadata['output_hashes'].items():
        assert a.sha(output/name)==digest
    selected=pd.read_csv(output/'selected_macro_metrics.csv')
    assert {'selected_donor','source_best_Wstandard','source_best_Wphysical','primary'}.issubset(set(selected.selection_role))
    comparisons=pd.read_csv(output/'paired_month_bootstrap.csv')
    cross=comparisons[comparisons.comparison.eq('round2_primary_minus_round1_primary')]
    assert cross.adaptive_round_comparison.all() and set(cross.stage)=={'source','target'}
    with pytest.raises(FileExistsError):
        a.analyze(rounds[1],rounds[2],output)

