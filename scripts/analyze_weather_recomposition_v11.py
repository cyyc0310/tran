"""Source-fixed V11 weather and risk contrasts; no automatic unseen-region opening."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METRICS = ['crps', 'empirical_coverage90', 'empirical_width90',
           'empirical_interval_score90', 'empirical_lower_tail_miss',
           'empirical_upper_tail_miss', 'empirical_path_coverage90',
           'point_level_mae', 'point_shape_mae',
           'level_crps', 'shape_crps', 'level_coverage90', 'level_width90',
           'shape_coverage90', 'shape_width90', 'center_level_mae',
           'center_shape_mae', 'centered_shape_energy_score']
ARMS = ('donor', 'donor_no_shape_risk', 'global_total', 'hetero_total')
REFERENCES = ('native', 'simple', 'B_same_arm')
REQUIRED_CHECKS = (
    'source_gate_passed', 'target_macro_coverage_at_least_90pct',
    'target_worst_region_coverage_at_least_85pct',
    *tuple(f'{name}_{test}' for name in REFERENCES for test in (
        'crps_at_least_5pct_lower', 'crps_ci_upper_below_zero',
        'width_not_increased', 'interval_score_lower')),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def config(record: dict[str, Any]) -> dict[str, Any]:
    return record.get('configuration', record)


def take(table: pd.DataFrame, record: dict[str, Any], lag: int = 7) -> pd.DataFrame:
    c = config(record)
    mask = table.lag.eq(lag)
    for key in ('budget', 'model', 'factor'):
        mask &= table[key].eq(c[key])
    result = table.loc[mask].copy()
    if result.empty:
        raise ValueError(f'Missing fixed score configuration: {c}, lag={lag}')
    return result


def holdout_decision(checks: dict[str, bool], frozen: dict | None = None) -> dict[str, Any]:
    """The protocol gates evidence, not permission to open unseen labels."""
    missing = sorted(set(REQUIRED_CHECKS) - set(checks))
    passed = not missing and all(checks[key] is True for key in REQUIRED_CHECKS)
    return dict(checks=checks, missing_checks=missing,
        performance_gate_passed=bool(passed), confirmation_eligible_track=False,
        new_holdout_triggered=False, weather_proxy=True,
        quantile_strong_baseline_completed=False,
        custom_innovation_gate_passed=False,
        scope='Previously viewed regions; six-month intervals descriptive only',
        weather_confirmation='A separately frozen weather-proxy confirmation contract is required')


def paired(left: pd.DataFrame, right: pd.DataFrame, *, n_bootstrap: int = 5000,
           seed: int = 17023, metrics: list[str] | None = None) -> pd.DataFrame:
    """Synchronous calendar-month bootstrap, equal regions and days within region."""
    metrics = METRICS if metrics is None else list(metrics)
    if not metrics or n_bootstrap < 1:
        raise ValueError('Metrics and positive bootstrap count required')
    keys = ['region', 'origin']
    left, right = left.copy(), right.copy()
    for table in (left, right):
        table['origin'] = pd.to_datetime(table.origin, utc=True)
        if table[keys].isna().any().any():
            raise ValueError('Paired region/day keys cannot be missing')
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError('Paired configurations must have unique region/day')
    joined = left[keys + metrics].merge(right[keys + metrics], on=keys,
        how='outer', suffixes=('_p', '_r'), indicator=True, validate='one_to_one')
    if not joined._merge.eq('both').all() or joined.empty:
        raise ValueError('Configurations do not have identical paired day support')
    months_index = joined.origin.dt.tz_localize(None).dt.to_period('M')
    months = sorted(months_index.unique())
    if len(months) != 6 or sorted(joined.origin.dt.month.unique()) != list(range(1, 7)):
        raise ValueError('Six fixed H1 calendar blocks required')
    joined['month'] = months_index
    draws = np.random.default_rng(seed).multinomial(6, np.full(6, 1 / 6), n_bootstrap)
    rows = []
    for label in ['overall', *sorted(joined.region.unique())]:
        frame = joined if label == 'overall' else joined[joined.region.eq(label)]
        groups = sorted(frame.region.unique())
        for metric in metrics:
            points, references, boot = [], [], []
            for region in groups:
                part = frame[frame.region.eq(region)]
                values = part[[metric + '_p', metric + '_r']].to_numpy(float)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite paired metrics')
                delta = part[metric + '_p'] - part[metric + '_r']
                points.append(values[:, 0].mean())
                references.append(values[:, 1].mean())
                sums = delta.groupby(part.month).sum().reindex(months, fill_value=0).to_numpy()
                counts = part.groupby('month').size().reindex(months, fill_value=0).to_numpy()
                denominator = draws @ counts
                if np.any(denominator == 0):
                    raise ValueError('Bootstrap draw has no valid days')
                boot.append((draws @ sums) / denominator)
            low, high = np.quantile(np.mean(boot, axis=0), [.025, .975])
            rows.append(dict(label=label, metric=metric, primary_mean=float(np.mean(points)),
                reference_mean=float(np.mean(references)),
                difference=float(np.mean(points) - np.mean(references)),
                ci95_low=float(low), ci95_high=float(high), n_paired_days=len(frame),
                n_regions=len(groups), n_month_blocks=6, n_bootstrap=n_bootstrap))
    return pd.DataFrame(rows)


def source_crps_best(source: pd.DataFrame) -> dict[str, Any]:
    """Diagnostic only: equal-region CRPS, without the coverage gate."""
    cells = source.groupby(['budget', 'model', 'factor', 'region'])[METRICS].mean()
    macro = cells.groupby(['budget', 'model', 'factor'])[METRICS].mean().reset_index()
    return macro.sort_values(['crps', 'budget', 'model', 'factor']).iloc[0][
        ['budget', 'model', 'factor']].to_dict()


def build_contrasts(table: pd.DataFrame, selections: dict[str, Any],
                    primary: dict[str, Any], *, n_bootstrap: int = 5000) -> pd.DataFrame:
    """Factors always come from the source choices, never from this score table."""
    rows = []

    def add(name, left, right, convention):
        rows.append(paired(take(table, left), take(table, right), n_bootstrap=n_bootstrap)
            .assign(comparison=name, factor_convention=convention,
                    primary_budget=left['budget'], primary_model=left['model'],
                    primary_factor=left['factor'], reference_budget=right['budget'],
                    reference_model=right['model'], reference_factor=right['factor']))

    def selected(budget, model):
        c = dict(config(selections[f'{budget}--{model}']))
        if c['budget'] != budget or c['model'] != model:
            raise ValueError('Source selection key/configuration mismatch')
        return c

    for arm in ARMS:
        b, w = selected('B', arm), selected('W', arm)
        add(f'fixed_B_factor_W_minus_B_{arm}', {**w, 'factor': b['factor']}, b, 'B_source_fixed')
        add(f'own_selected_W_minus_B_{arm}', w, b, 'each_source_selected')
    for model in ('global_total', 'hetero_total'):
        b = selected('B', model)
        counter = 'global_Brisk' if model == 'global_total' else 'hetero_Brisk'
        cf = dict(budget='W', model=counter, factor=b['factor'])
        add(f'fixed_B_risk_center_{model}', cf, b, 'B_source_fixed')
        add(f'fixed_B_factor_reestimated_risk_minus_Brisk_{model}',
            dict(budget='W', model=model, factor=b['factor']), cf, 'B_source_fixed')
    donor = selected('B', 'donor')
    glob = selected('B', 'global_total')
    for budget in ('B', 'W'):
        add(f'fixed_B_donor_factor_shape_risk_deleted_{budget}',
            dict(budget=budget, model='donor_no_shape_risk', factor=donor['factor']),
            dict(budget=budget, model='donor', factor=donor['factor']), 'B_donor_source_fixed')
        add(f'fixed_B_global_factor_hetero_minus_global_{budget}',
            dict(budget=budget, model='hetero_total', factor=glob['factor']),
            dict(budget=budget, model='global_total', factor=glob['factor']), 'B_global_source_fixed')
        add(f'own_selected_hetero_minus_global_{budget}',
            selected(budget, 'hetero_total'), selected(budget, 'global_total'), 'each_source_selected')
    refs = dict(native=selected('B', 'native_recompose'),
                simple=selected('B', 'simple_anchor'),
                B_same_arm=selected('B', primary['model']))
    for name, reference in refs.items():
        add('primary_minus_' + name, primary, reference, 'each_source_selected')
    return pd.concat(rows, ignore_index=True)


def performance_checks(primary_rows: pd.DataFrame, primary_record: dict[str, Any],
                        contrasts: pd.DataFrame) -> dict[str, bool]:
    by_region = primary_rows.groupby('region')[METRICS].mean()
    checks = dict(source_gate_passed=primary_record.get('status') == 'passed',
        target_macro_coverage_at_least_90pct=bool(by_region.empirical_coverage90.mean() >= .9),
        target_worst_region_coverage_at_least_85pct=bool(by_region.empirical_coverage90.min() >= .85))
    for name in REFERENCES:
        overall = contrasts[contrasts.comparison.eq('primary_minus_' + name) &
                            contrasts.label.eq('overall')]
        if overall.metric.duplicated().any():
            raise ValueError('Duplicate primary gate metric')
        t = overall.set_index('metric')
        if not set(METRICS).issubset(t.index):
            raise ValueError('Missing primary gate metrics')
        checks[name + '_crps_at_least_5pct_lower'] = bool(t.loc['crps', 'primary_mean'] <= .95*t.loc['crps', 'reference_mean'])
        checks[name + '_crps_ci_upper_below_zero'] = bool(t.loc['crps', 'ci95_high'] < 0)
        checks[name + '_width_not_increased'] = bool(t.loc['empirical_width90', 'difference'] <= 0)
        checks[name + '_interval_score_lower'] = bool(t.loc['empirical_interval_score90', 'difference'] < 0)
    return checks


def analyze(run: Path, output: Path) -> dict[str, Any]:
    from scripts import run_weather_recomposition_v11 as runner
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    verifications = {stage: runner.verify_phase(run, stage) for stage in ('source', 'target')}
    if set(runner.METRICS) != set(METRICS):
        raise ValueError('Runner/analyzer metric contract mismatch')
    choices = json.loads((run / 'source_choices.json').read_text())
    selections = choices['selections']
    primary_record = choices['primary']
    primary = config(primary_record)
    if primary['budget'] != 'W' or primary['model'] not in ('donor', 'global_total', 'hetero_total'):
        raise ValueError('Primary must be a source-selected eligible W arm')
    tables = {stage: pd.read_csv(run / f'{stage}_scores/day_metrics.csv')
              for stage in ('source', 'target')}
    region_rows, contrasts = [], []
    pure_best = source_crps_best(tables['source'])
    records = [(key, config(record)) for key, record in selections.items()]
    records += [(role, config(choices[role])) for role in ('primary', 'baseline', 'native', 'simple')]
    records += [('source_pure_crps_best_diagnostic', pure_best)]
    for stage, table in tables.items():
        for role, c in records:
            selected = take(table, c)
            by_region = selected.groupby('region')[METRICS].mean()
            for region, values in by_region.iterrows():
                region_rows.append(dict(stage=stage, selection_role=role, lag=7,
                    **{k: c[k] for k in ('budget', 'model', 'factor')},
                    region=region, **values.to_dict()))
        contrasts.append(build_contrasts(table, selections, primary).assign(stage=stage, lag=7))
    contrast_table = pd.concat(contrasts, ignore_index=True)
    checks = performance_checks(take(tables['target'], primary), primary_record,
                                contrast_table[contrast_table.stage.eq('target')])
    trigger = holdout_decision(checks)
    output.mkdir(parents=True)
    regions = pd.DataFrame(region_rows)
    regions.to_csv(output / 'selected_region_metrics.csv', index=False)
    regions.groupby(['stage', 'selection_role', 'lag', 'budget', 'model', 'factor'])[
        METRICS].mean().reset_index().to_csv(output / 'selected_macro_metrics.csv', index=False)
    contrast_table.to_csv(output / 'paired_month_bootstrap.csv', index=False)
    (output / 'holdout_trigger.json').write_text(json.dumps(trigger, indent=2) + '\n')
    inputs = [run / 'source_choices.json', run / 'prefit_freeze.json', Path(__file__).resolve(),
              ROOT / 'scripts/run_weather_recomposition_v11.py',
              ROOT / 'docs/WEATHER_RECOMPOSITION_V11_PROTOCOL.md',
              *[run / f'{stage}_{suffix}.json' for stage in ('source', 'target')
                for suffix in ('completed', 'prediction_seal')],
              *[run / f'{stage}_scores' / name for stage in ('source', 'target')
                for name in ('day_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv', 'score_seal.json')]]
    metadata = dict(run=str(run), primary=primary, source_pure_crps_best=pure_best,
        verified_phases=list(verifications),
        n_bootstrap=5000, seed=17023, target_selection_performed=False,
        target_training_performed=False, weather_proxy=True,
        quantile_strong_baseline_completed=False,
        input_hashes={str(path): sha(path) for path in inputs},
        output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return trigger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.output), indent=2))
