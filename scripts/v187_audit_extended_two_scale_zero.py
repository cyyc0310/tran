"""Refit long-budget hurdle controls; audit source roles, leaves, and Brier losses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from extended_monthly_information import ExtendedMonthlyInformation
from extended_two_scale_zero import (METHODS, MONTH_METHODS, COMPONENTS, MONTH_HEADS, HURDLES, check_inputs)
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation, monthly_context
from monthly_risk_features import feature_matrices
from v151_hourly_law import sha
from v154_hourly_strong_baselines import balanced
from v170_fuel_marginal_diagnosis import FUELS, verify
from v171_two_scale_zero import METRICS, aggregate
from v179_audit_increment_support import compare_frames
from v187_extended_two_scale_zero import (ROOT, OUT as SOURCE, DIRECT, OLD, CONTEXT, COUNTS,
                                         make_predictor, training_arrays, monthly_prefix_rows, reproduce)

OUT = ROOT / 'results/v187-extended-two-scale-zero-audit-r1'


def leaf_tables(predictor) -> dict:
    """Independent empirical weighted means for each trained tree's leaf."""
    result = {}
    for name, models in predictor.models.items():
        for fuel, model in enumerate(models):
            if name in MONTH_HEADS:
                X = predictor.month_X[MONTH_HEADS[name]]
                target = predictor.month_zero[:, fuel].astype(float)
                regions = predictor.month_regions
            else:
                take = predictor.positive[:, fuel]
                X, target, regions = predictor.X[name][take], predictor.zero[take, fuel].astype(float), predictor.regions[take]
            weights = balanced(regions)
            leaves = model.apply(X)
            tables = []
            for t, estimator in enumerate(model.estimators_):
                mass = np.bincount(leaves[:, t], weights=weights, minlength=estimator.tree_.node_count)
                zero = np.bincount(leaves[:, t], weights=weights * target, minlength=estimator.tree_.node_count)
                tables.append(np.divide(zero, mass, out=np.zeros_like(zero), where=mass > 0))
            result[(name, fuel)] = tables
    return result


def check_kernels(predictor, q: int, monthly: dict, components: dict, tables: dict) -> float:
    features, contexts, extra = feature_matrices(predictor.short, predictor.long, q)
    context = contexts[1]
    monthly_X = {'base13': context['base'][None], 'history36': context['history'][None],
                 'history77': np.r_[context['history'], extra][None]}
    error = 0.
    for name, models in predictor.models.items():
        X = monthly_X[MONTH_HEADS[name]] if name in MONTH_HEADS else features['long43' if name == 'long_cond43' else 'long84']
        for f, model in enumerate(models):
            ids = model.apply(X)
            expected = sum(table[ids[:, t]] for t, table in enumerate(tables[(name, f)])) / 32
            actual = monthly[name][f] if name in MONTH_HEADS else components[name][:, f]
            error = max(error, float(np.max(abs(expected - actual))))
    if error > 1e-12:
        raise ValueError('independent empirical-leaf forecast differs')
    return error


def independent_scores(data, rec, hourly, monthly, components, fold: int) -> dict:
    q = int(rec.profile_id)
    row = data.index.iloc[q]
    states = data.states[int(row.offset):int(row.stop)]
    actual = (data.raw[int(row.offset):int(row.stop)] == 0).astype(float)
    local = data.local(q, 'asof_30d')
    output = {k: [] for k in COUNTS}

    def score(p, y):
        p, y = np.broadcast_arrays(np.asarray(p, float), np.asarray(y, float))
        return {'brier': float(np.mean(p * p - 2 * p * y + y)), 'mean_forecast': float(np.sum(p) / p.size),
                'observed_fraction': float(np.sum(y) / y.size),
                'calibration_bias': float((np.sum(p) - np.sum(y)) / p.size)}

    for f, fuel in enumerate(FUELS):
        identity = {'fold': fold, 'profile_id': q, 'region': rec.region, 'month': rec.month,
                    'hours': len(states), 'fuel': fuel, 'input_fuel_positive': bool(data.M[local, f] > 0),
                    'current_fuel_positive': bool(data.M[q, f] > 0)}
        for method, p in hourly.items():
            output['hourly'].append(identity | {'method': method, **score(p[states, f], actual[:, f])})
        for method, p in monthly.items():
            output['monthly'].append(identity | {'method': method, 'is_world_proxy': method in ('world_gaussian', 'conditional_activity_recomposition'),
                                                 **score(p[f], data.M[q, f] == 0)})
        if data.M[q, f] > 0:
            for method, p in components.items():
                output['components'].append(identity | {'method': method, **score(p[states, f], actual[:, f])})
    return output


def audit_fold(data, short, long, inventory, fold: int) -> dict:
    predictor = make_predictor(data, short, long, inventory, fold)
    folder = SOURCE / f'fold_{fold}'
    queries = predictor.frozen.queries
    pd.testing.assert_frame_equal(queries, pd.read_csv(folder / 'query_inputs.csv'), check_exact=True)
    if (predictor.fingerprint() != json.loads((folder / 'model_fingerprints.json').read_text())
            or predictor.provenance != json.loads((folder / 'training_provenance.json').read_text())):
        raise ValueError('model fingerprint/provenance differs')
    with np.load(folder / 'training_matrices.npz', allow_pickle=False) as old:
        expected = training_arrays(predictor)
        if set(old.files) != set(expected) or any(not np.array_equal(old[k], v) for k, v in expected.items()):
            raise ValueError('source matrix inventory differs')
    expected_ids = inventory.loc[inventory.fold.eq(fold) & inventory.budget.eq('long_all'), 'profile_id'].to_numpy(int)
    if not np.array_equal(expected_ids, predictor.month_ids):
        raise ValueError('monthly supervision differs from frozen long_all roster')
    eligible = [int(q) for q in long.index.profile_id if long.index.region.iloc[q] in predictor.frozen.fit
                and long.index.month.iloc[q][:4] in ('2022', '2023', '2024') and monthly_context(long, int(q)) is not None]
    if set(eligible) != set(expected_ids):
        raise ValueError('monthly long_all eligibility incomplete')
    compare_frames(pd.DataFrame(monthly_prefix_rows(predictor)), pd.read_csv(folder / 'monthly_training_prefix.csv'), ['profile_id'])
    for q, context in predictor.month_contexts.items():
        if (long.index.region.iloc[q] not in predictor.frozen.fit or q in context['prefix']
                or not (long.release.iloc[context['prefix']] <= long.dates.iloc[q]).all()
                or any(long.index.region.iloc[p] != long.index.region.iloc[q] for p in context['prefix'])):
            raise ValueError('monthly source role/prefix violation')
    for q, (_, contexts, _) in predictor.contexts.items():
        if not data.index.month.iloc[q].startswith('2024') or data.index.region.iloc[q] not in predictor.frozen.fit:
            raise ValueError('fine source role/year violation')
        for monthly, context in zip((short, long), contexts):
            if (q in context['prefix'] or not (monthly.release.iloc[context['prefix']] <= monthly.dates.iloc[q]).all()
                    or any(monthly.index.region.iloc[p] != data.index.region.iloc[q] for p in context['prefix'])):
                raise ValueError('fine source prefix violation')
    if (not np.array_equal(predictor.zero, data.raw[predictor.raw_ids] == 0)
            or not np.array_equal(predictor.positive, data.M[predictor.profiles] > 0)
            or not set(predictor.regions).issubset(predictor.frozen.fit)):
        raise ValueError('source zero or positive-month filter differs')
    tables = leaf_tables(predictor)
    for q in queries.profile_id.iloc[[0, -1]]:
        check_inputs(predictor, int(q))
    with np.load(folder / 'probabilities.npz', allow_pickle=False) as old:
        if not np.array_equal(old['profile_id'], queries.profile_id.to_numpy(int)):
            raise ValueError('query ID order differs')
        saved = {k: old[k].copy() for k in COUNTS}
    shapes = {'hourly': (len(queries), 21, 48, 8), 'monthly': (len(queries), 8, 8), 'components': (len(queries), 4, 48, 8)}
    if any(a.shape != shapes[k] or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1 for k, a in saved.items()):
        raise ValueError('probability axes/domain differ')
    rebuilt = {k: [] for k in COUNTS}
    contexts = []
    max_kernel = max_hurdle = 0.
    for j, rec in enumerate(queries.itertuples(index=False)):
        q = int(rec.profile_id)
        hourly, monthly, components, context = predictor.predict(q)
        contexts.append({'profile_id': q, 'local': context['local'], 'prefix': context['prefix'],
                         'base': context['base'].tolist(), 'history': context['history'].tolist()})
        for kind, predictions, names in [('hourly', hourly, METHODS), ('monthly', monthly, MONTH_METHODS), ('components', components, COMPONENTS)]:
            if any(not np.array_equal(predictions[name], saved[kind][j, m]) for m, name in enumerate(names)):
                raise ValueError('refitted complete query probabilities differ')
        max_kernel = max(max_kernel, check_kernels(predictor, q, monthly, components, tables))
        for method, (month, component) in HURDLES.items():
            independent = 1 - (1 - monthly[month][None]) * (1 - components[component])
            max_hurdle = max(max_hurdle, float(np.max(abs(independent - hourly[method]))))
        if rec.role == 'test':
            # Explicit current-positive task below is not a deployable filter.
            scored = independent_scores(data, rec, hourly, monthly, components, fold)
            for kind, rows in scored.items():
                rebuilt[kind].extend(rows)
    if contexts != json.loads((folder / 'contexts.json').read_text()) or max_hurdle > 1e-12:
        raise ValueError('query context or hurdle arithmetic differs')
    frames = {k: pd.DataFrame(v) for k, v in rebuilt.items()}
    errors = {}
    for kind, frame in frames.items():
        old = pd.read_csv(folder / f'{kind}_scores.csv')
        if not np.isfinite(old[list(METRICS)]).all().all():
            raise ValueError('nonfinite saved score')
        errors[kind] = compare_frames(frame, old, ['profile_id', 'method', 'fuel'])
    reproduce(frames, fold)
    print(f'audited V187 fold {fold}: {len(frames["hourly"])} hourly rows', flush=True)
    return {'fold': fold, 'forests_refitted': 40, 'query_forecasts_rebuilt': len(queries),
            **{k + '_rows': len(v) for k, v in frames.items()}, 'max_score_error': max(errors.values()),
            'max_empirical_leaf_error': max_kernel, 'max_hurdle_formula_error': max_hurdle}


def comparison_pairs(kind: str) -> list[tuple[str, str]]:
    if kind == 'hourly':
        pairs = [(m, r) for m in HURDLES for r in ('long43', 'long84', 'short43', 'world_gaussian',
                                                 'month_persistence_conditional', 'month_history_conditional')]
        for month in MONTH_HEADS:
            pairs += [(f'{month}_{a}', f'{month}_{b}') for a, b in [('long_cond43', 'clock'), ('long_cond84', 'clock'), ('long_cond84', 'long_cond43')]]
        for component in ('clock', 'long_cond43', 'long_cond84'):
            pairs += [(f'{a}_{component}', f'{b}_{component}') for a, b in [('long_month_history', 'long_month_base'),
                       ('long_month_volatility', 'long_month_base'), ('long_month_volatility', 'long_month_history')]]
        pairs += [(f'{month}_{component}', 'long_persistence_cond43' if component == 'long_cond43' else 'long_persistence_cond84')
                  for month in MONTH_HEADS for component in ('long_cond43', 'long_cond84')]
        return pairs
    if kind == 'monthly':
        return ([(m, r) for m in MONTH_HEADS for r in ('month_persistence', 'month_base', 'month_history')]
                + [(m, r) for m in MONTH_HEADS for r in MONTH_HEADS if m != r])
    return [('long_cond43', 'short_cond43'), ('long_cond84', 'short_cond43'),
            ('long_cond43', 'clock'), ('long_cond84', 'clock'), ('long_cond84', 'long_cond43')]


def comparisons(tables: dict) -> pd.DataFrame:
    rng = np.random.default_rng(187)
    records = []
    for kind, regional in tables.items():
        pairs = comparison_pairs(kind)
        if len(pairs) != {'hourly': 90, 'monthly': 15, 'components': 5}[kind] or len(pairs) != len(set(pairs)):
            raise ValueError('comparison pair inventory')
        for (cohort, fuel), table in regional.groupby(['cohort', 'fuel']):
            for method, reference in pairs:
                a, b = [table.loc[table.method.eq(name)].sort_values('region') for name in (method, reference)]
                if not len(a) or not np.array_equal(a.region, b.region) or a.region.duplicated().any():
                    raise ValueError('paired task region inventory')
                delta = a.brier.to_numpy() - b.brier.to_numpy()
                boot = delta[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
                baseline = float(b.brier.mean())
                records.append({'task': kind, 'cohort': cohort, 'fuel': fuel, 'method': method, 'reference': reference,
                    'regions': len(a), 'brier': float(a.brier.mean()), 'reference_brier': baseline,
                    'paired_brier_difference': float(delta.mean()), 'brier_reduction_percent': float(-100 * delta.mean() / baseline) if baseline > 0 else None,
                    'region_bootstrap_lo': float(np.quantile(boot, .025)), 'region_bootstrap_hi': float(np.quantile(boot, .975)),
                    'regions_better_brier': int((delta < -1e-12).sum())})
    if len(records) != 1760:
        raise ValueError('complete comparison denominator')
    return pd.DataFrame(records)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = dict(summary['input_hashes'])
    for name in ('summary.json', *summary['output_hashes']):
        path = SOURCE / name
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for path in (Path(__file__), ROOT / 'scripts/v179_audit_increment_support.py'):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input changed')
    data, short = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    long = ExtendedMonthlyInformation(ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
    inventory = pd.read_csv(CONTEXT / 'training_inventory.csv')
    with threadpool_limits(limits=2):
        checks = [audit_fold(data, short, long, inventory, fold) for fold in range(6)]
    regional_tables = {}
    for kind, count in COUNTS.items():
        rows = pd.concat([pd.read_csv(SOURCE / f'fold_{f}/{kind}_scores.csv') for f in range(6)], ignore_index=True)
        if len(rows) != count or (SOURCE / f'{kind}_scores.csv').read_text() != rows.to_csv(index=False):
            raise ValueError('root task score serialization/count differs')
        regional, macro = aggregate(rows)
        compare_frames(regional, pd.read_csv(SOURCE / f'{kind}_region_metrics.csv'), ['cohort', 'region', 'method', 'fuel'])
        compare_frames(macro, pd.read_csv(SOURCE / f'{kind}_macro_metrics.csv'), ['cohort', 'method', 'fuel'])
        regional_tables[kind] = regional
    comparison = comparisons(regional_tables)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input changed during run')
    OUT.mkdir()
    for name, frame in [('reconstruction', pd.DataFrame(checks)), ('comparisons', comparison)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'forests_refitted': 240, 'query_forecasts_rebuilt': 482,
              **{k + '_rows_rebuilt': sum(c[k + '_rows'] for c in checks) for k in COUNTS},
              'model_query_kernel_checks': 482 * 40, 'future_query_raw_guards': 12, 'comparisons': len(comparison),
              'max_score_error': max(c['max_score_error'] for c in checks),
              'max_empirical_leaf_error': max(c['max_empirical_leaf_error'] for c in checks),
              'max_hurdle_formula_error': max(c['max_hurdle_formula_error'] for c in checks),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Full 240-forest/source/long_all-prefix/probability/task rebuild. Independent empirical leaf means, '
                       'hurdle identity and Brier expansion; feature/tree helpers shared. Current-positive is only a conditional '
                       'evaluation task, not deployment selection. No new CIF or original-method claim.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
