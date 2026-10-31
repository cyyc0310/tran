"""Source-fixed two-round V12 comparisons; adaptive development, no unseen opening."""
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

PRACTICAL_REFERENCES = ('native', 'simple', 'Bbaseline')
DONOR_REFERENCES = ('native', 'simple', 'Whetero', 'source_best_Wqrf')
COVERAGE_CHECKS = ('source_gate_passed', 'target_macro_coverage_at_least_90pct',
                   'target_worst_region_coverage_at_least_85pct')
REFERENCE_CHECKS = ('crps_at_least_5pct_lower', 'crps_ci_upper_below_zero',
                    'width_not_increased', 'interval_score_lower')
PRACTICAL_REQUIRED_CHECKS = (*COVERAGE_CHECKS, *tuple(
    f'{reference}_{check}' for reference in PRACTICAL_REFERENCES for check in REFERENCE_CHECKS))
DONOR_REQUIRED_CHECKS = (*COVERAGE_CHECKS, *tuple(
    f'{reference}_{check}' for reference in DONOR_REFERENCES for check in REFERENCE_CHECKS))


def selected(selections: dict[str, Any], budget: str, model: str) -> dict[str, Any]:
    """Read a source selection with an explicit key/configuration identity check."""
    record = dict(config(selections[f'{budget}--{model}']))
    if record.get('budget') != budget or record.get('model') != model:
        raise ValueError('Source selection key/configuration mismatch')
    if not np.isfinite(float(record['factor'])):
        raise ValueError('Nonfinite source factor')
    return record


def source_best_quantile(selections: dict[str, Any]) -> dict[str, Any]:
    """Choose among source-fixed W QRF settings using source coverage then CRPS."""
    records = [selected(selections, 'W', model) for model in ('qrf_0', 'qrf_25', 'qrf_50')]
    for record in records:
        scores = record.get('source_metrics', {})
        if (record.get('status') not in ('passed', 'failed') or
                not all(np.isfinite(float(scores.get(key, np.nan))) for key in ('violation', 'crps'))):
            raise ValueError('Source QRF ranking requires recorded status, violation and CRPS')
        if (record['status'] == 'passed') != (scores['violation'] <= 1e-12):
            raise ValueError('Source QRF status/violation mismatch')
    passed = [record for record in records if record['status'] == 'passed']
    return min(passed or records, key=lambda record: (
        0.0 if passed else record['source_metrics']['violation'],
        record['source_metrics']['crps'], record['budget'], record['model'], record['factor']))


def build_contrasts(table: pd.DataFrame, selections: dict[str, Any],
                    primary: dict[str, Any], round_number: int, *,
                    baseline: dict[str, Any] | None = None,
                    n_bootstrap: int = 5000) -> pd.DataFrame:
    """Compare frozen source settings; this function never ranks the score table."""
    if round_number not in (1, 2):
        raise ValueError('V12 round must be 1 or 2')
    primary = dict(config(primary))
    if primary['budget'] != 'W':
        raise ValueError('V12 primary must have W budget')
    rows = []

    def add(name: str, left: dict[str, Any], right: dict[str, Any], convention: str) -> None:
        rows.append(paired(take(table, left), take(table, right), n_bootstrap=n_bootstrap)
            .assign(comparison=name, factor_convention=convention,
                    primary_budget=left['budget'], primary_model=left['model'],
                    primary_factor=left['factor'], reference_budget=right['budget'],
                    reference_model=right['model'], reference_factor=right['factor']))

    native = selected(selections, 'B', 'native_recompose')
    simple = selected(selections, 'B', 'simple_anchor')
    donor_models = ('donor_a025', 'donor_a050', 'donor_a100') if round_number == 1 else ('donor',)
    for model in donor_models:
        donor = selected(selections, 'W', model)
        add(f'fixed_native_factor_W_{model}_minus_native',
            {**donor, 'factor': native['factor']}, native, 'native_source_fixed')
        add(f'own_selected_W_{model}_minus_native', donor, native, 'each_source_selected')
    for name, reference in (('native', native), ('simple', simple)):
        add('primary_minus_' + name, primary, reference, 'each_source_selected')
    refs = ('global_total', 'hetero_total', 'qrf_0') if round_number == 1 else (
        'donor', 'global_total', 'hetero_total', 'qrf_0', 'qrf_25', 'qrf_50')
    for model in refs:
        reference = selected(selections, 'W', model)
        if model != primary['model']:
            add('primary_minus_W_' + model, primary, reference, 'each_source_selected')
    if round_number == 2:
        if baseline is None or config(baseline)['budget'] != 'B':
            raise ValueError('Round 2 requires the source-selected B baseline')
        add('primary_minus_Bbaseline', primary, config(baseline), 'each_source_selected')
        for model in ('global_total', 'hetero_total', 'qrf_0', 'qrf_25', 'qrf_50'):
            b, w = selected(selections, 'B', model), selected(selections, 'W', model)
            add(f'fixed_B_factor_W_minus_B_{model}', {**w, 'factor': b['factor']}, b,
                'B_family_source_fixed')
            add(f'own_selected_W_minus_B_{model}', w, b, 'each_source_selected')
        qrf0 = selected(selections, 'W', 'qrf_0')
        for model in ('qrf_25', 'qrf_50'):
            mixed = selected(selections, 'W', model)
            add('fixed_W_qrf0_factor_' + model + '_minus_qrf_0',
                {**mixed, 'factor': qrf0['factor']}, qrf0, 'W_qrf0_source_fixed')
        global_w, hetero_w = selected(selections, 'W', 'global_total'), selected(selections, 'W', 'hetero_total')
        add('fixed_W_global_factor_hetero_minus_global',
            {**hetero_w, 'factor': global_w['factor']}, global_w, 'W_global_source_fixed')
        donor = selected(selections, 'W', 'donor')
        for name, reference in (('native', native), ('simple', simple),
                                ('Whetero', hetero_w), ('source_best_Wqrf', source_best_quantile(selections))):
            add('Wdonor_minus_' + name, donor, reference, 'each_source_selected')
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


def holdout_decision(practical_checks: dict[str, bool], donor_checks: dict[str, bool]) -> dict[str, Any]:
    """Separate practical and donor evidence; standard-method gains never open labels."""
    def gate(checks: dict[str, bool], required: tuple[str, ...]) -> dict[str, Any]:
        missing = sorted(set(required) - set(checks))
        return dict(checks=checks, missing_checks=missing,
                    passed=not missing and all(checks[key] is True for key in required))
    practical, donor = gate(practical_checks, PRACTICAL_REQUIRED_CHECKS), gate(donor_checks, DONOR_REQUIRED_CHECKS)
    return dict(practical_gain=practical, strong_donor_gate=donor,
        practical_gain_gate_passed=practical['passed'], strong_donor_gate_passed=donor['passed'],
        performance_gate_passed=practical['passed'],
        custom_innovation_gate_passed=False, custom_innovation_gate=False,
        new_holdout_triggered=False, confirmation_eligible_track=False,
        weather_proxy=True, quantile_strong_baseline_completed=True,
        target_selection_performed=False, target_training_performed=False,
        scope='Two adaptive source-development rounds; previously viewed targets; six-month intervals descriptive only',
        weather_contribution='Matched W-minus-B and W-donor-minus-native contrasts estimate weather contribution separately',
        standard_method_gain='A QRF/heteroscedastic primary can improve practical prediction without donor mechanism evidence',
        weather_confirmation='Clear performance and a separately frozen weather-proxy confirmation contract are required',
        novelty='Current differential transport, scale regression, QRF and distribution mixtures are standard methods')


def validate_round_lock(round1: Path, round2: Path, frozen1: dict[str, Any],
                        frozen2: dict[str, Any], choices1: dict[str, Any]) -> None:
    """Require round 2 to inherit exactly this round 1 source-only weather strength."""
    if (frozen1.get('round') != 1 or frozen2.get('round') != 2 or
            Path(frozen2.get('prior_round', '')).resolve() != round1.resolve() or
            choices1.get('locked_alpha') not in (.25, .5, 1.) or
            frozen2.get('alpha') != choices1.get('locked_alpha') or
            frozen2.get('target_scores_used_for_round_lock') is not False):
        raise ValueError('Round 2 detached from round 1 source-only strength lock')
    if round1.resolve() == round2.resolve():
        raise ValueError('Two distinct immutable rounds required')


def analyze(round1: Path, round2: Path, output: Path) -> dict[str, Any]:
    """Verify four phase closures and publish fresh source-fixed paired comparisons."""
    from scripts import run_weather_increment_v12 as runner

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
    validate_round_lock(rounds[1], rounds[2], frozen[1], frozen[2], choices[1])
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
            records += [('Wdonor', selected(choice['selections'], 'W', 'donor')),
                        ('source_best_Wqrf', source_best_quantile(choice['selections']))]
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
    primary, donor = choices[2]['primary'], selected(choices[2]['selections'], 'W', 'donor')
    practical_checks = performance_checks(take(tables[2, 'target'], primary), primary, target,
        prefix='primary_minus_', references=PRACTICAL_REFERENCES)
    donor_checks = performance_checks(take(tables[2, 'target'], donor), donor, target,
        prefix='Wdonor_minus_', references=DONOR_REFERENCES)
    trigger = holdout_decision(practical_checks, donor_checks)
    inputs = {Path(__file__).resolve(), ROOT / 'scripts/analyze_weather_recomposition_v11.py',
              ROOT / 'scripts/run_weather_increment_v12.py', runner.PROTOCOL}
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
        round2_donor=donor, source_best_Wqrf=source_best_quantile(choices[2]['selections']),
        locked_alpha=choices[1]['locked_alpha'],
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
