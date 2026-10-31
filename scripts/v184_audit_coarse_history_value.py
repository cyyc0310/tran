"""Rebuild monthly information-value forecasts and audit OOF/leaf/loss arithmetic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from extended_monthly_information import ExtendedMonthlyInformation
from monthly_history_features import MonthlyInformation, monthly_context, month_number
from monthly_history_value_models import FORESTS, EPS, METHODS
from monthly_distribution_scoring import score, aggregate, COARSE_METRICS, FUEL_METRICS, NONCONSTANT
from v151_hourly_law import EF, sha
from v170_fuel_marginal_diagnosis import verify
from v179_audit_increment_support import bisection_projection, compare_frames
from v184_coarse_history_value import (ROOT, BASE, SOURCE as OLD, OUT as SOURCE, CONTEXT, BUDGETS,
                                       BASE_CONTROL, fit_fold, fingerprint, source_matrices, source_prefix_rows, build_all, mutation)

OUT = ROOT / 'results/v184-coarse-history-value-audit-r1'


def raw_weights(regions):
    count = {r: int((regions == r).sum()) for r in set(regions)}
    return np.asarray([1 / count[r] for r in regions])


def closed_linear(X, y, regions, query):
    raw = raw_weights(regions)
    p = raw / raw.sum()
    mean = (p[:, None] * X).sum(axis=0)
    scale = np.maximum(np.sqrt((p[:, None] * (X - mean)**2).sum(axis=0)), 1e-3)
    z = (X - mean) / scale
    zm, ym = p @ z, p @ y
    centered = z - zm
    W = p * len(X)
    coef = np.linalg.solve(centered.T @ (W[:, None] * centered) + np.eye(X.shape[1]),
                           centered.T @ (W[:, None] * (y - ym)))
    intercept = ym - zm @ coef
    return ((query - mean) / scale) @ coef + intercept, (mean, scale, coef, intercept)


def local_point(data, q, method):
    origin = data.dates.iloc[q]
    ids = data.groups[data.index.region.iloc[q]]
    prefix = ids[(data.release.iloc[ids] <= origin).to_numpy()]
    prefix = prefix[np.argsort(data.dates.iloc[prefix].to_numpy())]
    S = data.shares[int(prefix[-1])]
    if method == 'persistence_residual':
        return S.copy()
    if method == 'seasonal_residual':
        selected = [p for p in prefix if data.dates.iloc[p].month == origin.month and data.dates.iloc[p].year == origin.year - 1]
        return data.shares[int(selected[-1])].copy() if selected else S.copy()
    ids = prefix[-12:]
    if len(ids) < 4:
        return S.copy()
    last = month_number(data.dates.iloc[int(ids[-1])])
    x = np.asarray([month_number(data.dates.iloc[int(p)]) - last for p in ids], float)
    y = data.shares[ids]
    slope = (x - x.mean()) @ (y - y.mean(axis=0)) / (np.sum((x - x.mean())**2) + 12.)
    value = y.mean(axis=0) + (month_number(origin) - last - x.mean()) * slope
    return bisection_projection(value[None])[0]


def audit_training(model, folder, budget):
    stored = np.load(folder / f'{budget}_training_matrices.npz', allow_pickle=False)
    matrices = source_matrices(model)
    if set(stored.files) != set(matrices) or any(not np.array_equal(stored[k], v) for k, v in matrices.items()):
        raise ValueError('complete source matrices not reproduced')
    compare_frames(pd.DataFrame(source_prefix_rows(model)), pd.read_csv(folder / f'{budget}_training_prefix.csv'), ['profile_id'])
    data = model.data
    if set(model.regions) != model.proper:
        raise ValueError('proper source coverage missing')
    for q, context in model.contexts.items():
        if (data.index.region.iloc[q] not in model.proper or q in context['prefix']
                or not (data.release.iloc[context['prefix']] <= data.dates.iloc[q]).all()):
            raise ValueError('source proper/prefix leakage')
    raw = raw_weights(model.regions)
    p = raw / raw.sum()
    delta = model.current - model.previous
    delta[abs(delta) <= EPS] = 0
    scalar = np.asarray([data.shares[q] @ EF - data.shares[i] @ EF for q, i in zip(model.ids, model.inputs)])
    magnitude = np.asarray([max(1., abs(data.shares[q] @ EF), abs(data.shares[i] @ EF)) for q, i in zip(model.ids, model.inputs)])
    scalar[abs(scalar) <= EPS * magnitude] = 0
    targets, error = {}, float(max(abs(model.prior - p)))
    for name, y in [('scalar8', np.repeat(scalar[:, None] / 1000, 8, axis=1)), ('delta8', delta)]:
        mean = (p[:, None] * y).sum(axis=0)
        scale = np.maximum(np.sqrt((p[:, None] * (y - mean)**2).sum(axis=0)), 1e-3)
        targets[name] = (y - mean) / scale
        error = max(error, max(abs(mean - model.transforms[name]['mean'])), max(abs(scale - model.transforms[name]['scale'])))
    _, full = closed_linear(model.X['base'], delta, model.regions, model.X['base'])
    for a, b in zip(full, (model.linear[0], model.linear[1], model.linear[2].coef_.T, model.linear[2].intercept_)):
        error = max(error, float(np.max(abs(a - b))))
    ordered = sorted(model.proper, key=lambda r: hashlib.sha256(('v184/coarse/' + r).encode()).hexdigest())
    folds = {r: j % 5 for j, r in enumerate(ordered)}
    if folds != model.region_fold:
        raise ValueError('OOF source region folds differ')
    linear_error, anchor_error = np.empty_like(model.current), np.empty(len(model.ids))
    for fold in range(5):
        test = np.asarray([folds[r] == fold for r in model.regions])
        predicted, _ = closed_linear(model.X['base'][~test], delta[~test], model.regions[~test], model.X['base'][test])
        linear_error[test] = model.current[test] - bisection_projection(model.previous[test] + predicted)
        # A separate fit checks the held-source role and sigma calculation;
        # scikit-learn's tree-growing implementation is shared, not independent.
        fit = ExtraTreesRegressor(n_estimators=32, min_samples_leaf=4, max_features=1.,
                                   random_state=20261028, n_jobs=2).fit(model.X['base'][~test], scalar[~test],
                                                                     sample_weight=raw_weights(model.regions[~test]))
        fit.n_jobs = 1
        value = np.clip(model.previous[test] @ EF + fit.predict(model.X['base'][test]), 0, 1000)
        for j, S in enumerate(model.previous[test]):
            if len(np.unique(EF[S > 0])) == 1:
                value[j] = S @ EF
        anchor_error[test] = value - model.current[test] @ EF
    linear_error[abs(linear_error) <= EPS] = 0
    error = max(error, float(np.max(abs(linear_error - model.residuals['pooled_linear_residual']))),
                abs(max(1., float(np.sqrt(p @ anchor_error**2))) - model.sigma))
    residuals = {'persistence_residual': delta.copy(), 'pooled_linear_residual': linear_error}
    for method in ('seasonal_residual', 'trend_residual'):
        values = model.current - np.asarray([local_point(data, int(q), method) for q in model.ids])
        values[abs(values) <= EPS] = 0
        residuals[method] = values
        error = max(error, float(np.max(abs(values - model.residuals[method]))))
    if error > 1e-8:
        raise ValueError(f'independent training/OOF arithmetic differs: {error}')
    leaves = {name: fit.apply(model.X[FORESTS[name][0]]) for name, fit in model.models.items()}
    return {'raw': raw, 'prior': p, 'delta': delta, 'targets': targets, 'leaves': leaves,
            'residuals': residuals, 'linear': full, 'error': error}


def independent_law_checks(model, q, laws, trained):
    context = monthly_context(model.data, q)
    S = model.data.shares[context['local']]
    support = bisection_projection(S + trained['delta'])
    logp = np.log(trained['prior']) - .5 * ((np.sqrt(model.previous) - np.sqrt(S))**2).sum(axis=1) / .09
    pH = np.exp(np.maximum(logp - max(logp), -700)); pH /= pH.sum()
    error = 0.
    for method, (values, weights) in laws.items():
        if method in trained['residuals']:
            if method == 'pooled_linear_residual':
                mean, scale, coef, intercept = trained['linear']
                point = bisection_projection((S + ((context['base'] - mean) / scale) @ coef + intercept)[None])[0]
            else:
                point = local_point(model.data, q, method)
            expected = bisection_projection(point + trained['residuals'][method])
            expected_weights = trained['prior']
        elif method == 'gaussian_increment':
            A = S @ EF if len(np.unique(EF[S > 0])) == 1 else float(np.clip(S @ EF + model.anchor_model.predict(context['base'][None])[0], 0, 1000))
            logits = np.log(pH) - .5 * ((support @ EF - A) / model.sigma)**2
            expected_weights = np.exp(np.maximum(logits - max(logits), -700)); expected_weights /= expected_weights.sum()
            expected = support
        else:
            feature, response = FORESTS[method]
            signature = model.models[method].apply(context[feature][None])[0]
            p = np.zeros(len(model.ids))
            for t, node in enumerate(signature):
                mask = trained['leaves'][method][:, t] == node
                p[mask] += trained['raw'][mask] / trained['raw'][mask].sum() / 32
            predicted_target = model.models[method].predict(context[feature][None])[0]
            error = max(error, float(np.max(abs(p @ trained['targets'][response] - predicted_target))))
            expected_weights = .95 * p + .05 * pH
            expected = support
        error = max(error, float(np.max(abs(values - expected))), float(max(abs(weights - expected_weights))))
    if error > 1e-8:
        raise ValueError(f'independent query law differs: {error}')
    return error


def independent_losses(values, p, truth, cache):
    key = values.shape, hashlib.sha256(values.tobytes()).digest()
    if key not in cache:
        difference = values[:, None] - values[None]
        projected = np.zeros_like(values)
        for ids in ([3, 4, 5, 6], [1, 7]):
            projected[:, ids] = values[:, ids] - values[:, ids].mean(axis=1, keepdims=True)
        cache[key] = (np.linalg.norm(difference, axis=-1), abs(difference), projected,
                      np.linalg.norm(projected[:, None] - projected[None], axis=-1))
    D, absolute, projected, DP = cache[key]
    projected_truth = np.zeros(8)
    for ids in ([3, 4, 5, 6], [1, 7]):
        projected_truth[ids] = truth[ids] - truth[ids].mean()
    marginal = p @ abs(values - truth) - .5 * np.einsum('i,ijf,j->f', p, absolute, p)
    zero = p @ (values == 0)
    anchor = values @ EF; target = truth @ EF
    coarse = {'energy_score': float(p @ np.linalg.norm(values - truth, axis=1) - .5 * p @ D @ p),
              'same_EF_energy_score': float(p @ np.linalg.norm(projected - projected_truth, axis=1) - .5 * p @ DP @ p),
              'anchor_crps': float(p @ abs(anchor - target) - .5 * p @ abs(anchor[:, None] - anchor[None]) @ p),
              'share_crps_mean': float(marginal.mean())}
    fuels = [{'crps': float(marginal[f]), 'predicted_zero': float(zero[f]),
              'zero_brier': float((zero[f] - (truth[f] == 0))**2)} for f in range(8)]
    return coarse, fuels


def audit_fold(short, long, inventory, fold):
    folder = SOURCE / f'fold_{fold}'
    models, queries = fit_fold(short, long, inventory, fold)
    metadata = json.loads((folder / 'model_fingerprints.json').read_text())
    if metadata != {b: fingerprint(m) for b, m in models.items()}:
        raise ValueError('complete model refit fingerprint differs')
    trained = {b: audit_training(m, folder, b) for b, m in models.items()}
    for q in queries.profile_id.iloc[[0, -1]]:
        mutation(models, short, int(q))
    coarse = pd.read_csv(folder / 'coarse_month_metrics.csv').set_index(['profile_id', 'method'])
    fuels = pd.read_csv(folder / 'coarse_fuel_metrics.csv').set_index(['profile_id', 'method', 'fuel_index'])
    index = pd.read_csv(folder / 'prediction_index.csv')
    packed = np.load(folder / 'coarse_predictions.npz', allow_pickle=False)
    values, probabilities = packed['values'], packed['probabilities']
    if (len(coarse) != len(queries) * 28 or len(fuels) != len(queries) * 224 or len(index) != len(coarse)
            or coarse.index.duplicated().any() or fuels.index.duplicated().any()
            or index.duplicated(['profile_id', 'method']).any() or index.coarse_offset.iloc[0] != 0
            or not np.array_equal(index.coarse_offset.to_numpy()[1:], index.coarse_stop.to_numpy()[:-1])
            or (index.coarse_stop <= index.coarse_offset).any() or index.coarse_stop.iloc[-1] != len(values)
            or len(values) != len(probabilities) or values.shape[1:] != (8,) or not np.isfinite(values).all()
            or not np.isfinite(probabilities).all() or (probabilities <= 0).any()):
        raise ValueError('complete packed prediction inventory')
    index = index.set_index(['profile_id', 'method'])
    count = fuel_count = 0
    score_error = law_error = loss_error = 0.
    for number, rec in enumerate(queries.itertuples(index=False), start=1):
        q = int(rec.profile_id)
        laws = build_all(models, short, q)
        for budget, model in models.items():
            selected = {name: laws[f'{budget}__{name}'] for name in METHODS}
            law_error = max(law_error, independent_law_checks(model, q, selected, trained[budget]))
        actual = short.shares[q].copy()
        cache, independent_cache = {}, {}
        for method, (v, p) in laws.items():
            saved = index.loc[(q, method)]
            if (saved.fold, saved.region, saved.month) != (fold, rec.region, rec.month):
                raise ValueError('packed prediction identity')
            a, b = int(saved.coarse_offset), int(saved.coarse_stop)
            if not np.array_equal(values[a:b], v) or not np.array_equal(probabilities[a:b], p):
                raise ValueError('packed law values/probabilities not exactly reproduced')
            metrics, metrics_fuel = score(v, p, actual, cache)
            independent, independent_fuel = independent_losses(v, p, actual, independent_cache)
            old = coarse.loc[(q, method)]
            if (old.fold, old.region, old.month) != (fold, rec.region, rec.month) or not np.isfinite(old[list(COARSE_METRICS)].to_numpy(float)).all():
                raise ValueError('coarse score identity/finiteness')
            score_error = max(score_error, max(abs(metrics[m] - old[m]) for m in COARSE_METRICS))
            loss_error = max(loss_error, max(abs(vv - old[m]) for m, vv in independent.items()))
            count += 1
            for r in metrics_fuel:
                f = r['fuel_index']; old = fuels.loc[(q, method, f)]
                if (old.fold, old.region, old.month) != (fold, rec.region, rec.month) or not np.isfinite(old[list(FUEL_METRICS)].to_numpy(float)).all():
                    raise ValueError('fuel score identity/finiteness')
                score_error = max(score_error, max(abs(float(r[m]) - float(old[m])) for m in FUEL_METRICS))
                loss_error = max(loss_error, max(abs(vv - old[m]) for m, vv in independent_fuel[f].items()))
                fuel_count += 1
        if number % 10 == 0 or number == len(queries):
            print(f'monthly audit fold {fold}: {number}/{len(queries)}', flush=True)
    if score_error > 1e-8 or loss_error > 1e-8:
        raise ValueError('complete monthly score/loss reconstruction differs')
    return {'fold': fold, 'coarse_rows_rebuilt': count, 'fuel_rows_rebuilt': fuel_count,
            'packed_laws_exactly_rebuilt': count, 'max_score_error': score_error, 'max_independent_loss_error': loss_error,
            'max_independent_law_error': law_error, 'max_independent_training_error': max(t['error'] for t in trained.values())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = summary['input_hashes'] | {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                                      str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json'),
                                      'scripts/v179_audit_increment_support.py': sha(ROOT / 'scripts/v179_audit_increment_support.py')}
    for name in summary['output_hashes']:
        hashes[str((SOURCE / name).relative_to(ROOT))] = sha(SOURCE / name)
    external = [('world_gaussian', OLD), ('allmonth_gaussian', OLD),
                ('increment_world_gaussian', ROOT / 'results/v179-increment-support-r1'),
                ('increment_forest_history_scalar8', ROOT / 'results/v179-increment-support-r1')]
    for _, folder in external:
        verify(folder)
        for name in ('summary.json', 'coarse_month_metrics.csv', 'coarse_fuel_metrics.csv'):
            p = folder / name; hashes[str(p.relative_to(ROOT))] = sha(p)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit inputs changed')
    coarse = pd.read_csv(SOURCE / 'coarse_month_metrics.csv')
    fuels = pd.read_csv(SOURCE / 'coarse_fuel_metrics.csv')
    for name, table, count, keys in [('coarse_month_metrics', coarse, 6748, ['profile_id', 'method']),
                                      ('coarse_fuel_metrics', fuels, 53984, ['profile_id', 'method', 'fuel_index'])]:
        blocks = pd.concat([pd.read_csv(SOURCE / f'fold_{k}/{name}.csv') for k in range(6)], ignore_index=True)
        if len(table) != count or table.duplicated(keys).any() or (SOURCE / f'{name}.csv').read_text() != blocks.to_csv(index=False):
            raise ValueError('complete root score inventory/serialization')
    regions, macro, fuel_regions, fuel_macro = aggregate(coarse, fuels)
    for name, table, keys in [('coarse_region_metrics', regions, ['region', 'method']), ('coarse_macro_metrics', macro, ['cohort', 'method']),
                               ('coarse_fuel_region_metrics', fuel_regions, ['region', 'method', 'fuel_index']),
                               ('coarse_fuel_macro_metrics', fuel_macro, ['cohort', 'method', 'fuel_index'])]:
        compare_frames(table, pd.read_csv(SOURCE / f'{name}.csv'), keys)
    short = MonthlyInformation(ROOT)
    long = ExtendedMonthlyInformation(ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
    inventory = pd.read_csv(CONTEXT / 'training_inventory.csv')
    with threadpool_limits(limits=2):
        checks = [audit_fold(short, long, inventory, fold) for fold in range(6)]
    if sum(r['coarse_rows_rebuilt'] for r in checks) != 6748 or sum(r['fuel_rows_rebuilt'] for r in checks) != 53984:
        raise ValueError('complete audit scope')
    keys = set(coarse.profile_id.unique())
    tables = [regions]
    for method, folder in external:
        frame = pd.read_csv(folder / 'coarse_month_metrics.csv')
        frame = frame.loc[frame.method.eq(method)]
        if set(frame.profile_id) != keys or frame.profile_id.duplicated().any():
            raise ValueError('external coarse query cohort mismatch')
        tables.append(frame.groupby(['region', 'method'])[list(COARSE_METRICS[:-1])].mean().reset_index())
    table = pd.concat(tables, ignore_index=True)
    pairs = []
    for budget in BUDGETS:
        pairs += [(f'{budget}__{a}', f'{budget}__{b}', 'same_budget') for a in METHODS for b in METHODS if a != b]
    pairs += [(f'{a}__{m}', f'{b}__{m}', 'information_effect') for a, b in [('long_fixed', 'short'), ('long_all', 'long_fixed'), ('long_all', 'short')] for m in METHODS]
    pairs += [(f'{budget}__{m}', ref, 'point_reference' if ref == 'coarse_persistence' else 'short_external_reference')
              for budget in BUDGETS for m in METHODS for ref in ['coarse_persistence', *[name for name, _ in external]]]
    if len(pairs) != 378:
        raise ValueError('comparison pair denominator')
    comparisons = []
    rng = np.random.default_rng(184)
    for cohort in ('all29', 'nonconstant24'):
        selected = table if cohort == 'all29' else table.loc[~table.region.isin(NONCONSTANT)]
        for method, ref, kind in pairs:
            a, b = [selected.loc[selected.method.eq(m)].sort_values('region') for m in (method, ref)]
            if not np.array_equal(a.region, b.region) or a.region.duplicated().any():
                raise ValueError('paired comparison region mismatch')
            resample = rng.integers(0, len(a), size=(10000, len(a)))
            row = {'cohort': cohort, 'method': method, 'reference': ref, 'comparison': kind, 'regions': len(a),
                   'anchor_coverage_difference': float(a.anchor_coverage.mean() - b.anchor_coverage.mean()),
                   'anchor_width_difference': float(a.anchor_width.mean() - b.anchor_width.mean()),
                   'anchor_width_change_percent': float(100 * (a.anchor_width.mean() / b.anchor_width.mean() - 1)) if b.anchor_width.mean() > 0 else None}
            for metric in ('energy_score', 'same_EF_energy_score', 'anchor_crps', 'share_crps_mean'):
                delta = a[metric].to_numpy() - b[metric].to_numpy()
                boot = delta[resample].mean(axis=1)
                row |= {metric + '_difference': float(delta.mean()), metric + '_reduction_percent': float(100 * (1 - a[metric].mean() / b[metric].mean())),
                        metric + '_bootstrap_lo': float(np.quantile(boot, .025)), metric + '_bootstrap_hi': float(np.quantile(boot, .975)),
                        metric + '_regions_better': int((delta < -1e-12).sum())}
            comparisons.append(row)
    quality = []
    index = short.index
    for q in sorted(keys):
        local = short.local(int(q), 'asof_30d')
        complete = []
        for profile in (q, local):
            date = short.dates.iloc[profile]
            complete.append(int(index.hours.iloc[profile]) == int(((date + pd.offsets.MonthBegin(1)) - date) / pd.Timedelta(hours=1)))
        quality.append({'profile_id': q, 'both_hour_counts_match': all(complete)})
    groups = coarse.merge(pd.DataFrame(quality), on='profile_id', validate='many_to_one')
    strata = groups.groupby(['both_hour_counts_match', 'region', 'method'])[list(COARSE_METRICS)].mean().reset_index()
    strata_macro = strata.groupby(['both_hour_counts_match', 'method'])[list(COARSE_METRICS)].mean().reset_index()
    if len(comparisons) != 756 or any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('final comparison scope or input changed')
    OUT.mkdir(parents=True)
    for name, frame in [('reconstruction', pd.DataFrame(checks)), ('comparisons', pd.DataFrame(comparisons)), ('coarse_quality_strata', strata_macro)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'coarse_rows_rebuilt': sum(c['coarse_rows_rebuilt'] for c in checks),
              'fuel_rows_rebuilt': sum(c['fuel_rows_rebuilt'] for c in checks), 'packed_laws_exactly_rebuilt': sum(c['packed_laws_exactly_rebuilt'] for c in checks),
              'model_budget_fold_groups_refitted': 18, 'independent_linear_OOF_groups': 90, 'separate_anchor_OOF_forest_refits': 90,
              'future_query_mutation_cases': 12, 'future_budget_groups': 36, 'future_method_law_guards': 324,
              'max_score_error': max(c['max_score_error'] for c in checks), 'max_independent_loss_error': max(c['max_independent_loss_error'] for c in checks),
              'max_independent_law_error': max(c['max_independent_law_error'] for c in checks),
              'max_independent_training_error': max(c['max_independent_training_error'] for c in checks), 'comparisons': len(comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Full model/source/prefix/packed law/score rebuild. Independent closed-form ridge/trend, proper response normalization, empirical leaves, bisection supports, pairwise losses; tree-growing and exact quantile/scoring helpers shared. Coarse-only exploratory source development, no hourly or originality result.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
