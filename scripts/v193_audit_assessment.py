"""Independent full-score/Winkler/cohort/selection and sampled CDF V193 audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd

from v151_hourly_law import sha
from transcif.research.weighted_metrics import distribution_metrics

SOURCE = ROOT / 'results/v193-benchmark-assessment-r3'
BASE = ROOT / 'results/research-prefix'
OUT = ROOT / 'results/v193-benchmark-assessment-audit-r2'
REGIONS = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
METHODS = ('prior', 'gaussian', 'diverse', 'level_template', 'ridge', 'extra_trees')
METRICS = ('mae', 'crps', 'coverage90', 'width90', 'winkler90')
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])


def check(actual, expected, tolerance=1e-8) -> float:
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('shape/nonfinite')
    error = float(np.max(abs(a - b))) if a.size else 0.
    if error > tolerance:
        raise ValueError(f'independent mismatch {error}')
    return error


def integral_crps(values: np.ndarray, weights: np.ndarray, y: float) -> float:
    """CDF squared integral, independent of the weighted pair-distance identity."""
    weights = np.asarray(weights, float);weights /= weights.sum()
    knots = np.unique(np.r_[values, y])
    midpoint = (knots[:-1] + knots[1:]) / 2
    cdf = np.asarray([weights[values <= x].sum() for x in midpoint])
    return float(np.dot(np.diff(knots), (cdf - (midpoint >= y)) ** 2))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = json.loads((SOURCE / 'summary.json').read_text())
    if summary['assertion_failures'] != 0:
        raise ValueError('producer failed')
    hashes = summary['input_hashes'].copy()
    for name, expected in summary['output_hashes'].items():
        if sha(SOURCE / name) != expected:
            raise ValueError('producer output hash')
    for p in SOURCE.iterdir():
        if p.is_file():
            hashes[str(p.relative_to(ROOT))] = sha(p)
    hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    r2_folder = ROOT / 'results/v193-benchmark-assessment-r2'
    r2_summary = json.loads((r2_folder / 'summary.json').read_text())
    hashes[str((r2_folder / 'summary.json').relative_to(ROOT))] = sha(r2_folder / 'summary.json')
    for filename in ('window_scores.csv', 'region_metrics.csv', 'macro.csv', 'selection.json'):
        path = r2_folder / filename
        if sha(path) != r2_summary['output_hashes'][filename]:
            raise ValueError('retired r2 primary output hash')
        hashes[str(path.relative_to(ROOT))] = sha(path)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('initial hash')
    seal = json.loads((SOURCE / 'prediction_seal.json').read_text())
    if sha(SOURCE / 'artifact_index.csv') != seal['prediction_index_sha256'] or sha(SOURCE / 'forecast_summaries.npz') != seal['prediction_summaries_sha256']:
        raise ValueError('forecast seal')
    index = pd.read_csv(SOURCE / 'artifact_index.csv');scores = pd.read_csv(SOURCE / 'window_scores.csv')
    keys = ['region', 'origin', 'method']
    if (len(index) != 15372 or not np.array_equal(index.artifact_id, np.arange(15372))
            or index.duplicated(keys).any() or not index[keys].equals(scores[keys])):
        raise ValueError('full forecast/scoring grid')
    with np.load(SOURCE / 'forecast_summaries.npz', allow_pickle=False) as p:
        mean, lower, upper = p['mean'], p['lower'], p['upper']
    if any(a.shape != (15372, 24) or not np.isfinite(a).all() for a in (mean, lower, upper)) or (lower > upper).any():
        raise ValueError('sealed numeric schema')
    max_old, max_winkler, max_cdf, max_cdf_hour, independent_cdf_hours = 0., 0., 0., 0., 0
    available_by_region = {}
    for region in REGIONS:
        labels = pd.read_csv(BASE / 'data' / f'{region}_hourly.csv')
        labels['time'] = pd.to_datetime(labels.time, utc=True)
        labels = labels.loc[labels.time.dt.year.eq(2024)]
        truth = {}
        for day, g in labels.groupby(labels.time.dt.floor('D')):
            g = g.sort_values('time')
            times = pd.DatetimeIndex(g.time)
            expected = pd.date_range(day, periods=24, freq='h')
            if len(g) == 24 and set(times) == set(expected) and g.valid.to_numpy(bool).all():
                generation = g[[f'generation_{f}' for f in ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')]].to_numpy(float)
                total = generation.sum(axis=1)
                if np.isfinite(generation).all() and (total > 0).all():
                    truth[day.isoformat()] = np.sum(generation * EF, axis=1) / total
        available_by_region[region] = len(truth)
        old = pd.read_csv(BASE / 'gate3' / f'{region}_other400/window_scores.csv')
        old = old.loc[old.method.isin(METHODS) & old.seed.isin([0, -1])].set_index(['method', 'origin'])
        if old.index.duplicated().any():
            raise ValueError('old comparison duplicates')
        # Deterministic first/last COMPLETE scored day, not best-performing day.
        selected = {min(truth), max(truth)}
        with np.load(BASE / 'gate3' / f'{region}_other400/predictions.npz', allow_pickle=False) as packed:
            for row in index.loc[index.region.eq(region)].itertuples(index=False):
                i = row.artifact_id;score = scores.iloc[i];y = truth.get(row.origin)
                if y is None:
                    if score.status != 'missing_label' or score.hours != 0:
                        raise ValueError('missing label cohort mismatch')
                    continue
                if score.status != 'scored' or score.hours != 24:
                    raise ValueError('valid label cohort mismatch')
                old_score = old.loc[(row.method, row.origin)]
                names = ['mae', 'rmse', 'crps', 'coverage90', 'width90', 'shape_mae', 'level_mae']
                max_old = max(max_old, check(score[names], old_score[names]))
                width = upper[i] - lower[i]
                penalties = np.where(y < lower[i], 20 * (lower[i] - y), 0) + np.where(y > upper[i], 20 * (y - upper[i]), 0)
                max_winkler = max(max_winkler, check([score.winkler90, score.predictive_width], [np.mean(width + penalties), np.mean(width)]))
                if row.origin in selected:
                    if row.prediction_kind == 'inference':
                        values, w = packed[row.source_key + '_pred'], packed[row.source_key + '_weights'].copy()
                    else:
                        values = np.minimum(1000., np.maximum(0., packed[row.source_key][None] + packed['residual_' + row.method]))
                        w = np.ones(len(values))
                    hour_scores = [integral_crps(values[:, h], w.copy(), float(y[h])) for h in range(24)]
                    pointwise = [distribution_metrics(values[:, h:h+1], w.copy(), np.array([y[h]]))['crps'] for h in range(24)]
                    max_cdf_hour = max(max_cdf_hour, check(hour_scores, pointwise))
                    max_cdf = max(max_cdf, check([np.mean(hour_scores)], [score.crps]))
                    independent_cdf_hours += 24
        print(f'audit {region}: {len(truth)} complete days', flush=True)
    if independent_cdf_hours != 2016 or sum(available_by_region.values()) != 2266:
        raise ValueError('independent check scope/count')
    # Every method sees precisely the same forecast keys and label mask.
    masks = []
    for method in METHODS:
        g = scores.loc[scores.method.eq(method)].sort_values(['region', 'origin'])
        mask = g[['region', 'origin', 'status', 'hours']].reset_index(drop=True)
        if masks and not mask.equals(masks[0]):
            raise ValueError('method label availability differs')
        masks.append(mask)
    selection = json.loads((SOURCE / 'selection.json').read_text())
    if len(selection) != 30 or len({(r['method'], r['width_limit']) for r in selection}) != 30:
        raise ValueError('selection key grid')
    for row in selection:
        f = scores.loc[scores.method.eq(row['method']) & scores.status.eq('scored')].copy()
        counts = f.groupby('region').hours.sum()
        f['original_weight'] = f.apply(lambda r: r.hours / counts[r.region] / 7, axis=1)
        take = f.predictive_width <= row['width_limit']
        mass = float(f.loc[take, 'original_weight'].sum())
        check([row['fixed_region_weight_acceptance_rate']], [mass])
        all_values = [float(np.dot(f.original_weight, f[m])) for m in METRICS]
        check([row['all'][m] for m in METRICS], all_values)
        if (row['accepted_scored_windows'] != int(take.sum()) or row['accepted_scored_hours'] != int(f.loc[take].hours.sum())
                or row['scored_windows'] != 2266 or row['forecast_windows'] != 2562 or row['missing_windows'] != 296):
            raise ValueError('selection full/accepted denominator')
        if mass == 0:
            if row['accepted_fixed_region_conditioned'] is not None or row['accepted_nonempty_region_macro'] is not None:
                raise ValueError('empty acceptance not None')
        else:
            weighted = [float(np.dot(f.loc[take, 'original_weight'], f.loc[take, m]) / mass) for m in METRICS]
            check([row['accepted_fixed_region_conditioned'][m] for m in METRICS], weighted)
            conditional = f.loc[take].groupby('region')[list(METRICS)].mean()
            check([row['accepted_nonempty_region_macro'][m] for m in METRICS], conditional.mean().to_numpy())
            if row['regions_with_accepted_labels'] != len(conditional):
                raise ValueError('conditional region denominator')
    # Independent complete region/macro reconstruction (including regional RMSE).
    rebuilt = []
    for (region, method), g in scores.loc[scores.status.eq('scored')].groupby(['region', 'method']):
        names = ['mae', 'rmse', 'crps', 'coverage90', 'width90', 'shape_mae', 'level_mae', 'winkler90']
        values = {m: float(g[m].mean()) for m in names};values['rmse'] = float(np.sqrt(np.mean(g.rmse ** 2)))
        rebuilt.append({'region': region, 'method': method, **values})
    regional = pd.DataFrame(rebuilt)
    for filename, expected, key in [('region_metrics.csv', regional, ['region', 'method']),
                                    ('macro.csv', regional.groupby('method', as_index=False)[names].mean(), ['method'])]:
        saved = pd.read_csv(SOURCE / filename).sort_values(key).reset_index(drop=True)
        expected = expected.sort_values(key).reset_index(drop=True)
        if not saved[key].equals(expected[key]):
            raise ValueError('aggregate key mismatch')
        check(saved[names], expected[names])
    monthly_rows = []
    month_scores = scores.loc[scores.status.eq('scored')].copy()
    month_scores['month'] = pd.to_datetime(month_scores.origin, utc=True).dt.month
    for (region, month, method), g in month_scores.groupby(['region', 'month', 'method']):
        values = {m: float(g[m].mean()) for m in names}
        values['rmse'] = float(np.sqrt(np.average(g.rmse ** 2, weights=g.hours)))
        monthly_rows.append({'region': region, 'month': month, 'method': method, **values})
    month_key = ['region', 'month', 'method']
    actual_month = pd.read_csv(SOURCE / 'region_month_metrics.csv').sort_values(month_key).reset_index(drop=True)
    expected_month = pd.DataFrame(monthly_rows).sort_values(month_key).reset_index(drop=True)
    if not actual_month[month_key].equals(expected_month[month_key]):
        raise ValueError('monthly aggregate keys')
    check(actual_month[names], expected_month[names])
    r2 = ROOT / 'results/v193-benchmark-assessment-r2'
    for filename in ('window_scores.csv', 'region_metrics.csv', 'macro.csv', 'selection.json'):
        if (SOURCE / filename).read_bytes() != (r2 / filename).read_bytes():
            raise ValueError('monthly fix changed primary output: ' + filename)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('final hash')
    OUT.mkdir()
    result = {'assertion_failures': 0, 'forecast_artifacts': 15372, 'scored_origin_region_days': 2266,
              'missing_origin_region_days': 296, 'scored_method_windows': 13596, 'missing_method_windows': 1776,
              'scored_days_by_region': available_by_region, 'old_metrics_rechecked_windows': 13596,
              'winkler_rechecked_windows': 13596, 'independent_CDF_hours': independent_cdf_hours,
              'max_old_metric_error': max_old, 'max_independent_winkler_error': max_winkler,
              'max_independent_CDF_window_average_error': max_cdf, 'max_independent_CDF_hour_error': max_cdf_hour, 'selection_rows_rebuilt': 30,
              'region_month_rows_rebuilt': len(expected_month), 'r2_primary_outputs_byte_identical': True,
              'models_fitted': False, 'scope': 'All scores/masks/Winkler/selection/aggregates; CDF integration only first/last complete day per region, all6 methods and24hours, not full independent CRPS recomputation.',
              'input_hashes': hashes, 'output_hashes': {}}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if not k.endswith('hashes')}, flush=True)


if __name__ == '__main__':
    main()
