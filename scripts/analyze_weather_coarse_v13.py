"""Source-fixed V13 weather-policy and physical-risk comparisons; no unseen opening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_weather_recomposition_v11 import METRICS, config, paired, sha, take

STANDARD_FAMILIES = ('global_total', 'hetero_total', 'location_scale', 'qrf_0', 'qrf_50')
PHYSICAL_FAMILIES = ('physical_independent', 'physical_rank')
RISK_FAMILIES = (*STANDARD_FAMILIES, 'level_independent', 'level_rank', *PHYSICAL_FAMILIES)
POLICIES = ('fixed_donor', 'constant_half', 'gate_ridge', 'gate_forest')
PRACTICAL_REFERENCES = ('native', 'simple', 'Bbaseline', 'Wstandard')
MECHANISM_REFERENCES = (*PRACTICAL_REFERENCES, 'matched_level')
COVERAGE_CHECKS = ('source_gate_passed', 'target_macro_coverage_at_least_90pct',
                   'target_worst_region_coverage_at_least_85pct')
REFERENCE_CHECKS = ('crps_at_least_5pct_lower', 'crps_ci_upper_below_zero',
                    'width_not_increased', 'interval_score_lower')
PRACTICAL_REQUIRED_CHECKS = (*COVERAGE_CHECKS, *tuple(
    f'{reference}_{check}' for reference in PRACTICAL_REFERENCES for check in REFERENCE_CHECKS))
MECHANISM_REQUIRED_CHECKS = (*COVERAGE_CHECKS, *tuple(
    f'{reference}_{check}' for reference in MECHANISM_REFERENCES for check in REFERENCE_CHECKS))


def selected(selections: dict[str, Any], budget: str, model: str) -> dict[str, Any]:
    """Read a source selection with an explicit key/configuration identity check."""
    record = dict(config(selections[f'{budget}--{model}']))
    if record.get('budget') != budget or record.get('model') != model:
        raise ValueError('Source selection key/configuration mismatch')
    if not np.isfinite(float(record['factor'])):
        raise ValueError('Nonfinite source factor')
    return record


def source_best(selections: dict[str, Any], budget: str,
                models: tuple[str, ...]) -> dict[str, Any]:
    """Choose source-fixed settings using source coverage then CRPS, never target scores."""
    records = [selected(selections, budget, model) for model in models]
    for record in records:
        scores = record.get('source_metrics', {})
        if (record.get('status') not in ('passed', 'failed') or
                not all(np.isfinite(float(scores.get(key, np.nan))) for key in ('violation', 'crps'))):
            raise ValueError('Source ranking requires recorded status, violation and CRPS')
        if (record['status'] == 'passed') != (scores['violation'] <= 1e-12):
            raise ValueError('Source status/violation mismatch')
    passed = [record for record in records if record['status'] == 'passed']
    return min(passed or records, key=lambda record: (
        0.0 if passed else record['source_metrics']['violation'],
        record['source_metrics']['crps'], record['budget'], record['model'], record['factor']))


def same_configuration(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare exact source configuration identities, including spread factor."""
    return all(config(left)[key] == config(right)[key] for key in ('budget', 'model', 'factor'))


def build_contrasts(table: pd.DataFrame, selections: dict[str, Any],
                    primary: dict[str, Any], round_number: int, *,
                    baseline: dict[str, Any] | None = None,
                    n_bootstrap: int = 5000) -> pd.DataFrame:
    """Compare frozen source settings; table values never determine reference choice."""
    if round_number not in (1, 2):
        raise ValueError('V13 round must be 1 or 2')
    primary = dict(config(primary))
    if primary['budget'] != 'W':
        raise ValueError('V13 primary must have W budget')
    rows = []

    def add(name: str, left: dict[str, Any], right: dict[str, Any], convention: str) -> None:
        rows.append(paired(take(table, left), take(table, right), n_bootstrap=n_bootstrap)
            .assign(comparison=name, factor_convention=convention,
                    primary_budget=left['budget'], primary_model=left['model'],
                    primary_factor=left['factor'], reference_budget=right['budget'],
                    reference_model=right['model'], reference_factor=right['factor']))

    native, simple = selected(selections, 'B', 'native_recompose'), selected(selections, 'B', 'simple_anchor')
    policy_models = POLICIES if round_number == 1 else ('fixed_donor', 'selected_donor')
    for model in policy_models:
        policy = selected(selections, 'W', model)
        add(f'fixed_native_factor_W_{model}_minus_native',
            {**policy, 'factor': native['factor']}, native, 'native_source_fixed')
        add(f'own_selected_W_{model}_minus_native', policy, native, 'each_source_selected')
        for reference_model in ('fixed_donor', 'constant_half') if round_number == 1 else ('fixed_donor',):
            if reference_model != model:
                reference = selected(selections, 'W', reference_model)
                add(f'fixed_reference_factor_{model}_minus_{reference_model}',
                    {**policy, 'factor': reference['factor']}, reference, 'reference_source_fixed')
                add(f'own_selected_{model}_minus_{reference_model}', policy, reference, 'each_source_selected')
    for name, reference in (('native', native), ('simple', simple)):
        add('primary_minus_' + name, primary, reference, 'each_source_selected')
    if round_number == 1:
        for model in POLICIES:
            reference = selected(selections, 'W', model)
            if not same_configuration(primary, reference):
                add('primary_minus_W_' + model, primary, reference, 'each_source_selected')
    else:
        if baseline is None or config(baseline)['budget'] != 'B':
            raise ValueError('Round 2 requires the source-selected B baseline')
        standard = source_best(selections, 'W', STANDARD_FAMILIES)
        mechanism = source_best(selections, 'W', PHYSICAL_FAMILIES)
        add('primary_minus_Bbaseline', primary, config(baseline), 'each_source_selected')
        if not same_configuration(primary, standard):
            add('primary_minus_Wstandard', primary, standard, 'each_source_selected')
        for model in RISK_FAMILIES:
            b, w = selected(selections, 'B', model), selected(selections, 'W', model)
            add(f'fixed_B_factor_W_minus_B_{model}', {**w, 'factor': b['factor']}, b, 'B_family_source_fixed')
            add(f'own_selected_W_minus_B_{model}', w, b, 'each_source_selected')
            if not same_configuration(primary, w):
                add('primary_minus_W_' + model, primary, w, 'each_source_selected')
        for budget in ('B', 'W'):
            for coupling in ('independent', 'rank'):
                physical, level = selected(selections, budget, 'physical_' + coupling), selected(selections, budget, 'level_' + coupling)
                add(f'fixed_level_factor_physical_minus_level_{budget}_{coupling}',
                    {**physical, 'factor': level['factor']}, level, 'level_source_fixed')
                add(f'own_selected_physical_minus_level_{budget}_{coupling}', physical, level, 'each_source_selected')
            for family in ('level', 'physical'):
                independent, rank = selected(selections, budget, family + '_independent'), selected(selections, budget, family + '_rank')
                add(f'fixed_independent_factor_rank_minus_independent_{budget}_{family}',
                    {**rank, 'factor': independent['factor']}, independent, 'independent_source_fixed')
                add(f'own_selected_rank_minus_independent_{budget}_{family}', rank, independent, 'each_source_selected')
            global_ref = selected(selections, budget, 'global_total')
            for model in ('hetero_total', 'location_scale'):
                conditional = selected(selections, budget, model)
                add(f'fixed_global_factor_{model}_minus_global_{budget}',
                    {**conditional, 'factor': global_ref['factor']}, global_ref, 'global_source_fixed')
            qrf0, qrf50 = selected(selections, budget, 'qrf_0'), selected(selections, budget, 'qrf_50')
            add(f'fixed_qrf0_factor_qrf50_minus_qrf0_{budget}',
                {**qrf50, 'factor': qrf0['factor']}, qrf0, 'qrf0_source_fixed')
        coupling = mechanism['model'].removeprefix('physical_')
        level_match = selected(selections, 'W', 'level_' + coupling)
        for name, reference in (('native', native), ('simple', simple), ('Bbaseline', config(baseline)),
                                ('Wstandard', standard), ('matched_level', level_match)):
            add('mechanism_minus_' + name, mechanism, reference, 'each_source_selected')
    return pd.concat(rows, ignore_index=True)


def performance_checks(rows: pd.DataFrame, record: dict[str, Any], contrasts: pd.DataFrame,
                       *, prefix: str, references: tuple[str, ...]) -> dict[str, bool]:
    """Require complete finite gate evidence; missing reference metrics are errors."""
    if rows.empty or not np.isfinite(rows[METRICS].to_numpy(float)).all():
        raise ValueError('Empty/nonfinite performance evidence')
    by_region = rows.groupby('region')[METRICS].mean()
    checks = dict(source_gate_passed=record.get('status') == 'passed',
        target_macro_coverage_at_least_90pct=bool(by_region.empirical_coverage90.mean() >= .9),
        target_worst_region_coverage_at_least_85pct=bool(by_region.empirical_coverage90.min() >= .85))
    for name in references:
        overall = contrasts[contrasts.comparison.eq(prefix + name) & contrasts.label.eq('overall')]
        if overall.metric.duplicated().any():
            raise ValueError('Duplicate gate metric')
        scores = overall.set_index('metric')
        if not set(METRICS).issubset(scores.index):
            raise ValueError('Missing gate metrics')
        if not np.isfinite(scores[['primary_mean', 'reference_mean', 'difference', 'ci95_high']].to_numpy(float)).all():
            raise ValueError('Nonfinite gate metrics')
        checks[name + '_crps_at_least_5pct_lower'] = bool(
            scores.loc['crps', 'primary_mean'] <= .95 * scores.loc['crps', 'reference_mean'])
        checks[name + '_crps_ci_upper_below_zero'] = bool(scores.loc['crps', 'ci95_high'] < 0)
        checks[name + '_width_not_increased'] = bool(scores.loc['empirical_width90', 'difference'] <= 0)
        checks[name + '_interval_score_lower'] = bool(scores.loc['empirical_interval_score90', 'difference'] < 0)
    return checks


def holdout_decision(practical_checks: dict[str, bool], mechanism_checks: dict[str, bool],
                     *, standard_self_reference: bool = False) -> dict[str, Any]:
    """Separate practical from physical evidence; never turn either into novelty permission."""
    def gate(checks: dict[str, bool], required: tuple[str, ...]) -> dict[str, Any]:
        missing = sorted(set(required) - set(checks))
        return dict(checks=checks, missing_checks=missing,
                    passed=not missing and all(checks[key] is True for key in required))
    practical_required = tuple(key for key in PRACTICAL_REQUIRED_CHECKS
                               if not (standard_self_reference and key.startswith('Wstandard_')))
    practical = gate(practical_checks, practical_required)
    mechanism = gate(mechanism_checks, MECHANISM_REQUIRED_CHECKS)
    practical['self_reference_exemptions'] = ['Wstandard'] if standard_self_reference else []
    return dict(practical_gain=practical, mechanism_gate=mechanism,
        practical_gain_gate_passed=practical['passed'], mechanism_gate_passed=mechanism['passed'],
        performance_gate_passed=practical['passed'],
        custom_innovation_gate_passed=False, custom_innovation_gate=False,
        new_holdout_triggered=False, confirmation_eligible_track=False,
        weather_proxy=True, quantile_strong_baseline_completed=True,
        target_selection_performed=False, target_training_performed=False,
        scope='Two adaptive source-development rounds; previously viewed targets; six-month intervals descriptive only',
        weather_contribution='Source-trained weather policies and matched B/W contrasts are evaluated separately',
        physical_contribution='Source-selected physical arm must beat standards and matched scalar propagation',
        rank_contribution='Fixed-factor rank-minus-independent contrasts are descriptive; not automatically implied by mechanism gate',
        weather_confirmation='Clear performance and a separately frozen weather-proxy confirmation contract are required',
        novelty='Gate regression, factor propagation, conditional distributions and empirical rank coupling are standard components')


def validate_round_lock(round1: Path, round2: Path, frozen1: dict[str, Any],
                        frozen2: dict[str, Any], choices1: dict[str, Any],
                        choices2: dict[str, Any]) -> None:
    """Require exactly this round 1 source policy, never target-driven policy changes."""
    policy = choices1.get('locked_policy')
    if (frozen1.get('round') != 1 or frozen2.get('round') != 2 or
            Path(frozen2.get('prior_round', '')).resolve() != round1.resolve() or
            policy not in POLICIES or policy != config(choices1['primary'])['model'] or
            frozen2.get('weather_policy') != policy or choices2.get('locked_policy') != policy or
            frozen2.get('target_scores_used_for_round_lock') is not False):
        raise ValueError('Round 2 detached from round 1 source-only policy lock')
    if round1.resolve() == round2.resolve():
        raise ValueError('Two distinct immutable rounds required')


def analyze(round1: Path, round2: Path, output: Path) -> dict[str, Any]:
    """Verify four phase closures and publish fresh source-fixed paired comparisons."""
    from scripts import run_weather_coarse_v13 as runner

    rounds = {1: Path(round1).resolve(), 2: Path(round2).resolve()}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if set(runner.METRICS) != set(METRICS):
        raise ValueError('Runner/analyzer metric contract mismatch')
    seals = {(number, stage): runner.verify_phase(path, stage)
             for number, path in rounds.items() for stage in ('source', 'target')}
    frozen = {number: runner.verify(path) for number, path in rounds.items()}
    choices = {number: json.loads((path / 'source_choices.json').read_text())
               for number, path in rounds.items()}
    validate_round_lock(rounds[1], rounds[2], frozen[1], frozen[2], choices[1], choices[2])
    for number, path in rounds.items():
        if Path(seals[number, 'target']['other_round_source']).resolve() != rounds[3-number]:
            raise ValueError('Target seal bound to a different peer source round')
        primary = config(choices[number]['primary'])
        eligible = {(c['budget'], c['model']) for c in runner.configurations(number) if c['primary_eligible']}
        if (primary['budget'], primary['model']) not in eligible:
            raise ValueError('Ineligible source primary')
    tables = {(number, stage): pd.read_csv(path / f'{stage}_scores/day_metrics.csv')
              for number, path in rounds.items() for stage in ('source', 'target')}
    regions, contrasts = [], []
    for number in rounds:
        choice = choices[number]
        records = [(key, config(record)) for key, record in choice['selections'].items()]
        records += [(role, config(choice[role])) for role in
                    ('primary', 'native', 'simple', 'baseline', 'diagnostic_crps_best')]
        if number == 2:
            records += [('selected_donor', selected(choice['selections'], 'W', 'selected_donor')),
                        ('source_best_Wstandard', source_best(choice['selections'], 'W', STANDARD_FAMILIES)),
                        ('source_best_Wphysical', source_best(choice['selections'], 'W', PHYSICAL_FAMILIES))]
        for stage in ('source', 'target'):
            table = tables[number, stage]
            for role, c in records:
                by_region = take(table, c).groupby('region')[METRICS].mean()
                for region, values in by_region.iterrows():
                    regions.append(dict(round=number, stage=stage, selection_role=role,
                        lag=7, **{key: c[key] for key in ('budget', 'model', 'factor')},
                        region=region, **values.to_dict()))
            contrasts.append(build_contrasts(table, choice['selections'], choice['primary'], number,
                baseline=choice['baseline']).assign(round=number, stage=stage, lag=7,
                                                   adaptive_round_comparison=False))
    for stage in ('source', 'target'):
        left, right = config(choices[2]['primary']), config(choices[1]['primary'])
        contrasts.append(paired(take(tables[2, stage], left), take(tables[1, stage], right))
            .assign(comparison='round2_primary_minus_round1_primary',
                    factor_convention='each_round_source_selected', round=2, stage=stage, lag=7,
                    primary_budget=left['budget'], primary_model=left['model'], primary_factor=left['factor'],
                    reference_budget=right['budget'], reference_model=right['model'], reference_factor=right['factor'],
                    adaptive_round_comparison=True))
    contrast_table = pd.concat(contrasts, ignore_index=True)
    target = contrast_table[contrast_table['round'].eq(2) & contrast_table.stage.eq('target')]
    primary = choices[2]['primary']
    standard = source_best(choices[2]['selections'], 'W', STANDARD_FAMILIES)
    mechanism = source_best(choices[2]['selections'], 'W', PHYSICAL_FAMILIES)
    standard_self = same_configuration(primary, standard)
    practical_references = tuple(name for name in PRACTICAL_REFERENCES if not (standard_self and name == 'Wstandard'))
    practical_checks = performance_checks(take(tables[2, 'target'], primary), primary, target,
        prefix='primary_minus_', references=practical_references)
    mechanism_checks = performance_checks(take(tables[2, 'target'], mechanism), mechanism, target,
        prefix='mechanism_minus_', references=MECHANISM_REFERENCES)
    trigger = holdout_decision(practical_checks, mechanism_checks, standard_self_reference=standard_self)
    inputs = {Path(__file__).resolve(), ROOT / 'scripts/analyze_weather_recomposition_v11.py',
              ROOT / 'scripts/run_weather_coarse_v13.py', runner.PROTOCOL}
    for number, path in rounds.items():
        inputs.update(Path(p) for key in ('code_hashes', 'input_hashes') for p in frozen[number][key])
        inputs.update(path / name for name in ('prefit_freeze.json', 'source_choices.json'))
        inputs.update(path / f'{stage}_{suffix}.json' for stage in ('source', 'target')
                      for suffix in ('completed', 'prediction_seal'))
        inputs.update(path / f'{stage}_scores' / name for stage in ('source', 'target')
                      for name in ('day_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv', 'score_seal.json'))
        inputs.update(path / entry['path'] for stage in ('source', 'target') for entry in seals[number, stage]['entries'])
    input_hashes = {str(path): sha(path) for path in sorted(inputs)}
    output.mkdir(parents=True)
    region_table = pd.DataFrame(regions)
    region_table.to_csv(output / 'selected_region_metrics.csv', index=False)
    region_table.groupby(['round', 'stage', 'selection_role', 'lag', 'budget', 'model', 'factor'])[
        METRICS].mean().reset_index().to_csv(output / 'selected_macro_metrics.csv', index=False)
    contrast_table.to_csv(output / 'paired_month_bootstrap.csv', index=False)
    (output / 'holdout_trigger.json').write_text(json.dumps(trigger, indent=2) + '\n')
    metadata = dict(rounds={str(key): str(value) for key, value in rounds.items()},
        primaries={str(key): choices[key]['primary'] for key in rounds},
        round2_selected_donor=selected(choices[2]['selections'], 'W', 'selected_donor'),
        source_best_Wstandard=standard, source_best_Wphysical=mechanism,
        practical_standard_self_reference=standard_self, locked_policy=choices[1]['locked_policy'],
        verified_phases=[f'round{number}_{stage}' for number, stage in seals],
        n_bootstrap=5000, seed=17023, n_month_blocks=6, adaptive_rounds=True,
        independent_confirmation=False, target_selection_performed=False,
        target_training_performed=False, weather_proxy=True, quantile_strong_baseline_completed=True,
        input_hashes=input_hashes,
        output_hashes={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return trigger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round1', type=Path, required=True)
    parser.add_argument('--round2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.round1, args.round2, args.output), indent=2))


if __name__ == '__main__':
    main()
