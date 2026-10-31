"""Source-fixed information-value contrasts; seen-region development only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METRICS = ['crps', 'empirical_coverage90', 'empirical_width90',
           'empirical_interval_score90', 'point_level_mae', 'point_shape_mae']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def config(record):
    return record.get('configuration', record)


def take(table, record, lag=7):
    c = config(record)
    mask = table.lag.eq(lag)
    for key in ('budget', 'model', 'factor'):
        mask &= table[key].eq(c[key])
    result = table.loc[mask].copy()
    if result.empty:
        raise ValueError(f'Missing fixed score configuration: {c}, lag={lag}')
    return result


def holdout_decision(checks, frozen):
    eligible = frozen.get('weather_csv') is None
    return dict(checks=checks, confirmation_eligible_track=eligible,
        new_holdout_triggered=bool(eligible and all(checks.values())),
        scope='Previously viewed regions; six-month intervals descriptive only',
        weather_confirmation='Weather proxy track requires a separate confirmation contract')


def paired(left, right, *, n_bootstrap=5000, seed=17023):
    """Synchronous month blocks; equal region weights, day-weight within region."""
    keys = ['region', 'origin']
    left, right = left.copy(), right.copy()
    for table in (left, right):
        table['origin'] = pd.to_datetime(table.origin, utc=True)
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError('Paired configurations must have unique region/day')
    joined = left[keys + METRICS].merge(right[keys + METRICS], on=keys,
        how='outer', suffixes=('_p', '_r'), indicator=True, validate='one_to_one')
    if not joined._merge.eq('both').all() or joined.empty:
        raise ValueError('Configurations do not have identical paired day support')
    joined['month'] = pd.to_datetime(joined.origin, utc=True).dt.month
    months = sorted(joined.month.unique())
    if len(months) != 6:
        raise ValueError('Six fixed calendar blocks required')
    draws = np.random.default_rng(seed).multinomial(6, np.full(6, 1 / 6), n_bootstrap)
    rows = []
    for label in ['overall', *sorted(joined.region.unique())]:
        frame = joined if label == 'overall' else joined[joined.region.eq(label)]
        groups = sorted(frame.region.unique())
        for metric in METRICS:
            point, reference, boot = [], [], []
            for region in groups:
                part = frame[frame.region.eq(region)]
                delta = part[metric + '_p'] - part[metric + '_r']
                if not np.isfinite(delta).all():
                    raise ValueError('Nonfinite paired metrics')
                point.append(part[metric + '_p'].mean())
                reference.append(part[metric + '_r'].mean())
                sums = delta.groupby(part.month).sum().reindex(months, fill_value=0).to_numpy()
                counts = part.groupby('month').size().reindex(months, fill_value=0).to_numpy()
                denom = draws @ counts
                if np.any(denom == 0):
                    raise ValueError('Bootstrap draw has no valid days')
                boot.append((draws @ sums) / denom)
            low, high = np.quantile(np.mean(boot, axis=0), [.025, .975])
            rows.append(dict(label=label, metric=metric, primary_mean=float(np.mean(point)),
                reference_mean=float(np.mean(reference)), difference=float(np.mean(point) - np.mean(reference)),
                ci95_low=float(low), ci95_high=float(high), n_paired_days=len(frame),
                n_regions=len(groups), n_month_blocks=6, n_bootstrap=n_bootstrap))
    return pd.DataFrame(rows)


def analyze(run, output):
    from scripts import run_information_value_v10 as runner
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    runner.verify_phase(run, 'source')
    runner.verify_phase(run, 'target')
    choices = json.loads((run / 'source_choices.json').read_text())
    day = pd.read_csv(run / 'target_scores/day_metrics.csv')
    source = pd.read_csv(run / 'source_scores/day_metrics.csv')
    selections = choices['selections']
    primary = config(choices['primary'])
    rows, contrasts = [], []
    for lag in (7, 30):
        for budget, record in selections.items():
            c = {**config(record), 'budget': budget}
            for stage, table in [('source', source), ('target', day)]:
                selected = take(table, c, lag)
                by_region = selected.groupby('region')[METRICS].mean()
                for region, values in by_region.iterrows():
                    rows.append(dict(stage=stage, lag=lag, budget=budget,
                        model=c['model'], factor=c['factor'], region=region, **values.to_dict()))
            if budget != 'B':
                contrasts.append(paired(take(day, c, lag), take(day, {**config(selections['B']), 'budget': 'B'}, lag))
                    .assign(comparison='selected_' + budget + '_minus_selected_B', lag=lag))
        # Hold a model and its I0 source-selected spread fixed when adding info.
        for model in sorted(day.loc[day.budget.eq('B'), 'model'].unique()):
            selected = source[source.lag.eq(7) & source.budget.eq('B') & source.model.eq(model)]
            cells = selected.groupby(['factor', 'region'])[METRICS].mean()
            means = cells.groupby('factor')[METRICS].mean()
            worst = cells.empirical_coverage90.groupby('factor').min()
            means['violation'] = np.maximum(.9-means.empirical_coverage90, 0) + np.maximum(means.empirical_coverage90-.95, 0) + np.maximum(.85-worst, 0)
            means.loc[means.violation.le(1e-12), 'violation'] = 0.
            factor = float(means.reset_index().sort_values(['violation','crps','factor']).iloc[0].factor)
            base = dict(budget='B', model=model, factor=factor)
            for budget in selections:
                if budget != 'B':
                    contrasts.append(paired(take(day, {**base, 'budget': budget}, lag), take(day, base, lag))
                        .assign(comparison=f'same_{model}_{budget}_minus_B', lag=lag))
    # The primary and all historical references were fixed without target scores.
    references = [('B', {**config(selections['B']), 'budget':'B'}, 7)]
    background = day[day.budget.eq('V9_background')]
    for model, part in background.groupby('model'):
        if part.factor.nunique() != 1:
            raise ValueError('V9 reference must retain a single old source-fixed factor')
        references.append((model, dict(budget='V9_background', model=model, factor=float(part.factor.iloc[0])), 0))
    checks = {'source_gate_passed': choices['primary'].get('status') == 'passed'}
    p = take(day, primary, 7)
    preg = p.groupby('region')[METRICS].mean()
    checks['target_macro_coverage_at_least_90pct'] = bool(preg.empirical_coverage90.mean() >= .9)
    checks['target_worst_region_coverage_at_least_85pct'] = bool(preg.empirical_coverage90.min() >= .85)
    for name, reference, ref_lag in references:
        result = paired(p, take(day, reference, ref_lag))
        contrasts.append(result.assign(comparison='primary_minus_' + name, lag=7))
        t = result[result.label.eq('overall')].set_index('metric')
        checks[name + '_crps_at_least_5pct_lower'] = bool(t.loc['crps','primary_mean'] <= .95*t.loc['crps','reference_mean'])
        checks[name + '_crps_ci_upper_below_zero'] = bool(t.loc['crps','ci95_high'] < 0)
        checks[name + '_width_not_increased'] = bool(t.loc['empirical_width90','difference'] <= 0)
        checks[name + '_interval_score_lower'] = bool(t.loc['empirical_interval_score90','difference'] < 0)
    if len(references) != 3:
        raise ValueError('Both V9 dynamic and anchor references required')
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output/'selected_region_metrics.csv', index=False)
    summary = pd.DataFrame(rows).groupby(['stage','lag','budget','model','factor'])[METRICS].mean().reset_index()
    summary.to_csv(output/'selected_macro_metrics.csv', index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(output/'paired_month_bootstrap.csv', index=False)
    trigger = holdout_decision(checks, json.loads((run / 'prefit_freeze.json').read_text()))
    (output/'holdout_trigger.json').write_text(json.dumps(trigger, indent=2)+'\n')
    inputs = [run/'source_choices.json', run/'prefit_freeze.json', Path(__file__).resolve(),
              *[run/f'{s}_scores'/n for s in ('source','target') for n in ('day_metrics.csv','region_metrics.csv','macro_metrics.csv')]]
    meta = dict(run=str(run), primary=primary, n_bootstrap=5000, seed=17023,
        target_selection_performed=False, target_training_performed=False,
        input_hashes={str(p):sha(p) for p in inputs},
        output_hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'analysis_metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    return trigger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.output), indent=2))
