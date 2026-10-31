"""Independent V192 selection, linear algebra, leaf means and score audit.

Forest growth and the producer feature constructor are shared for refits. Prefixes,
feature arithmetic, ridge/VAR normal equations, Kalman update, leaf aggregation,
truth and losses are independently checked. No additional regions/times are read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v151_hourly_law import EF, sha
from v192_reallocated_kappa import ROOT, ALLOCATIONS, METHODS, ORIGINS, Context, Predictor

SOURCE = ROOT / 'results/v192-reallocated-kappa-r1'
OUT = ROOT / 'results/v192-reallocated-kappa-audit-r2'


def close(actual: np.ndarray, expected: np.ndarray, tolerance: float, label: str) -> float:
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'{label}: shape/nonfinite')
    error = float(np.max(abs(a - b))) if a.size else 0.
    if error > tolerance:
        raise ValueError(f'{label}: {error} > {tolerance}')
    return error


def independent_feature(context: Context, q: int) -> tuple[list[int], np.ndarray]:
    data = context.data
    region, date = data.index.region.iloc[q], data.dates.iloc[q]
    ids = [int(p) for p in data.groups[region] if data.release.iloc[p] <= date]
    ids.sort(key=lambda p: data.dates.iloc[p])
    local = ids[-1]
    S = data.shares[local]
    a = float(np.dot(S, EF))
    base = np.r_[S, np.log1p(data.M[local].sum()), (date - data.dates.iloc[local]).days / 365,
                 np.sin(2 * np.pi * date.month / 12), np.cos(2 * np.pi * date.month / 12), a / 1000]
    slope, logslope, gap = np.zeros(8), 0., 0.
    if len(ids) > 1:
        previous = ids[-2]
        x, y = data.dates.iloc[local], data.dates.iloc[previous]
        gap = (x.year - y.year) * 12 + x.month - y.month
        slope = (S - data.shares[previous]) / gap
        logslope = (np.log1p(data.M[local].sum()) - np.log1p(data.M[previous].sum())) / gap
    errors = []
    for p in ids:
        available = [int(j) for j in data.groups[region] if data.release.iloc[j] <= data.dates.iloc[p]]
        if available:
            last = max(available, key=lambda j: data.dates.iloc[j])
            errors.append(float(data.shares[p] @ EF) - float(data.shares[last] @ EF))
    errors = np.asarray(errors[-6:])
    mix = data.M[ids].sum(axis=0);mix /= mix.sum()
    response = data.shares[ids] @ EF
    tail = [np.std(response) / 1000, np.log1p(len(ids)),
            float(np.sqrt(np.mean(errors ** 2))) / 1000 if len(errors) else 0.,
            float(np.mean(errors)) / 1000 if len(errors) else 0., np.log1p(len(errors))]
    return ids, np.r_[base, slope, logslope, gap / 12, mix, tail]


def independent_selection(context: Context, target: str, origin: str, allocation: str) -> list[dict]:
    rows = []
    for region in context.roster:
        if region == target:
            continue
        candidates = context.index.loc[context.index.region.eq(region)
                                       & context.index.month.str.startswith('2024')].copy()
        candidates = candidates.loc[candidates.profile_id.map(lambda p: context.data.release.iloc[p] <= pd.Timestamp(origin))]
        candidates = candidates.sort_values('month')
        if allocation == 'recent':
            candidates = candidates.tail(2)
        count, extra = divmod(context.budgets[region], len(candidates))
        for i, row in enumerate(candidates.itertuples(index=False)):
            bank = np.arange(int(row.offset), int(row.stop))
            draw = np.random.default_rng(np.random.SeedSequence([192, int(row.profile_id)])).permutation(bank)
            ids = sorted(int(p) for p in draw[:count + (i < extra)])
            timestamps = pd.to_datetime(context.seconds[ids], unit='s', utc=True)
            if not ((timestamps < pd.Timestamp(origin, tz='UTC')).all()
                    and (timestamps.strftime('%Y-%m-01') == row.month).all()):
                raise ValueError('independent raw calendar/cutoff')
            rows.append({'region': region, 'profile_id': int(row.profile_id), 'raw_ids': ids,
                         'prefix': independent_feature(context, int(row.profile_id))[0]})
    return rows


def ridge_predict(X: np.ndarray, y: np.ndarray, w: np.ndarray, query: np.ndarray) -> np.ndarray:
    center = np.average(X, axis=0, weights=w)
    scale = np.sqrt(np.average((X - center) ** 2, axis=0, weights=w))
    scale[scale < 1e-12] = 1.
    Z = (X - center) / scale
    wn = w / np.mean(w)
    zm = np.average(Z, axis=0, weights=wn)
    ym = np.average(y, weights=wn)
    z = Z - zm
    coef = np.linalg.solve(z.T @ (wn[:, None] * z) + np.eye(X.shape[1]), z.T @ (wn * (y - ym)))
    return ((query - center) / scale - zm) @ coef + ym


def leaf_predict(model, X: np.ndarray, y: np.ndarray, w: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Exact empirical weighted leaf means, independent of forest.predict."""
    total = np.zeros(len(query))
    for tree in model.estimators_:
        leaves, wanted = tree.apply(X), tree.apply(query)
        for i, leaf in enumerate(wanted):
            take = leaves == leaf
            if not take.any():
                raise ValueError('empty trained leaf')
            total[i] += np.sum(w[take] * y[take]) / np.sum(w[take]) / len(model.estimators_)
    return total


def dynamic_prediction(model: Predictor, q: int, name: str) -> tuple[float, float]:
    dyn = model.dynamic[name]
    states = np.column_stack([model.a, model.kappa]) / 1000
    pairs = dyn.pairs
    X, Y = states[pairs[:, 0]], states[pairs[:, 1]]
    regions = model.regions[pairs[:, 0]]
    names, counts = np.unique(regions, return_counts=True)
    w = np.asarray([1 / len(names) / counts[np.where(names == r)[0][0]] for r in regions])
    Z = (regions[:, None] == names).astype(float)
    Z -= np.sum(w[:, None] * Z, axis=0)
    design = np.column_stack([np.ones(len(X)), X, Z]) if name == 'dynamic_random' else np.column_stack([np.ones(len(X)), X])
    penalty = np.diag(np.r_[0., np.ones(design.shape[1] - 1)])
    D = np.vstack([np.sqrt(w / np.mean(w))[:, None] * design, penalty])
    rhs = np.vstack([np.sqrt(w / np.mean(w))[:, None] * Y, np.zeros((len(penalty), 2))])
    coefficient = np.linalg.lstsq(D, rhs, rcond=None)[0]
    close(coefficient, dyn.coefficient, 1e-10, 'independent augmented least squares')
    U, s, V = np.linalg.svd(coefficient[1:3].T, full_matrices=False)
    A = U @ np.diag(np.minimum(s, .98)) @ V
    b = coefficient[0]
    residual = Y - X @ A.T - (design @ coefficient - X @ coefficient[1:3])
    Q = sum(wi * np.outer(e, e) for wi, e in zip(w, residual)) + np.eye(2) * 1e-8
    close(A, dyn.A, 1e-10, 'A');close(Q, dyn.Q, 1e-10, 'Q')
    context, data = model.context, model.context.data
    region, origin = data.index.region.iloc[q], data.dates.iloc[q]
    ids = [int(p) for p in data.groups[region] if p < data.canonical_count
           and data.index.month.iloc[p].startswith('2024') and data.release.iloc[p] <= origin]
    ids.sort(key=lambda p: data.dates.iloc[p])
    first = ids[0]
    k0 = ridge_predict(model.X, model.kappa, model.w,
                       np.asarray([context.feature(first)['history']]))[0]
    x = np.array([float(data.shares[first] @ EF), k0]) / 1000
    training_kappa = ridge_predict(model.X, model.kappa, model.w, model.X)
    initial_kappa_variance = float(model.w @ ((model.kappa - training_kappa) / 1000) ** 2)
    P = np.diag([max(float(np.var(states[:, 0])), 1e-8), max(initial_kappa_variance, 1e-8)])
    close(P, dyn.P0, 1e-10, 'independent initial covariance')
    previous = data.dates.iloc[first]
    for p in ids[1:]:
        date = data.dates.iloc[p]
        gap = (date.year - previous.year) * 12 + date.month - previous.month
        for _ in range(gap):
            x = A @ x + b;P = A @ P @ A.T + Q
        observation = float(data.shares[p] @ EF) / 1000
        # Scalar conditional Gaussian identities, separate from producer vector update.
        new_k = x[1] + P[1, 0] / P[0, 0] * (observation - x[0])
        variance_k = P[1, 1] - P[1, 0] * P[0, 1] / P[0, 0]
        if variance_k < -1e-12:
            raise ValueError('negative posterior variance')
        x = np.array([observation, new_k]);P = np.diag([0., variance_k])
        previous = date
    for _ in range((origin.year - previous.year) * 12 + origin.month - previous.month):
        x = A @ x + b
    return float(1000 * x[0]), float(1000 * x[1])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = json.loads((SOURCE / 'summary.json').read_text())
    hashes = summary['input_hashes'].copy()
    hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    for p in SOURCE.rglob('*'):
        if p.is_file():
            hashes[str(p.relative_to(ROOT))] = sha(p)
    for key, value in summary['input_hashes'].items():
        if sha(ROOT / key) != value:
            raise ValueError('producer input hash')
    for key, value in summary['output_hashes'].items():
        if sha(SOURCE / key) != value:
            raise ValueError('producer output hash')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    context = Context()
    if context.roster != sorted(context.roster) or len(context.roster) != 19:
        raise ValueError('roster')
    results, max_score_error, max_prediction_error = [], 0., 0.
    source_profiles, source_hours = 0, 0
    for origin in ORIGINS:
        for target in context.roster:
            folder = SOURCE / f'{origin}_{target}'
            block = json.loads((folder / 'summary.json').read_text())
            if block['manifest'] != manifest or block['target'] != target or block['origin'] != origin:
                raise ValueError('block/root identity')
            for file, digest in block['output_hashes'].items():
                if sha(folder / file) != digest:
                    raise ValueError('block output hash')
            prefixes = json.loads((folder / 'prefixes.json').read_text())
            for key, expected in prefixes.items():
                q = int(key)
                prefix, independent_X = independent_feature(context, q)
                if prefix != expected:
                    raise ValueError('independent prefix mismatch')
                close(independent_X, context.feature(q)['history'], 1e-10, 'independent monthly features')
            prediction_rows = json.loads((folder / 'predictions.json').read_text())
            saved = pd.read_csv(folder / 'scores.csv')
            q = context.profile(target, origin)
            if len(saved) != 16 or saved.duplicated(['allocation', 'method']).any():
                raise ValueError('block score key grid')
            for allocation in ALLOCATIONS:
                selected = independent_selection(context, target, origin, allocation)
                if selected != json.loads((folder / f'{allocation}_source.json').read_text()):
                    raise ValueError('independent selected samples/prefix')
                model = Predictor(context, target, origin, allocation)
                independent_coarse = []
                for region in context.roster:
                    if region == target:
                        continue
                    for p in context.data.groups[region]:
                        if context.data.release.iloc[p] <= pd.Timestamp(origin):
                            known = [j for j in context.data.groups[region]
                                     if context.data.release.iloc[j] <= context.data.dates.iloc[p]]
                            if known:
                                independent_coarse.append(int(p))
                if independent_coarse != list(model.coarse_ids):
                    raise ValueError('independent coarse source list/labels')
                source_profiles += len(selected);source_hours += sum(len(r['raw_ids']) for r in selected)
                with np.load(folder / f'{allocation}_matrices.npz', allow_pickle=False) as packed:
                    arrays = model.arrays()
                    if set(packed.files) != set(arrays):
                        raise ValueError('matrix schema')
                    for name, value in arrays.items():
                        if not np.array_equal(packed[name], value):
                            raise ValueError(f'refit matrix {name}')
                # Independent source responses use elementwise sums, not BLAS g@EF.
                for rec, a, mu in zip(selected, model.a, model.mu):
                    g = context.raw[np.asarray(rec['raw_ids'])]
                    a2 = float(np.sum(context.data.shares[rec['profile_id']] * EF))
                    mu2 = float(np.mean(np.sum(g * EF, axis=1) / np.sum(g, axis=1)))
                    close([a, mu], [a2, mu2], 1e-9, 'source response')
                current = model.predict()
                expected = [r for r in prediction_rows if r['allocation'] == allocation]
                if current != expected:
                    raise ValueError('all-method refit predictions not exact')
                X = np.asarray([context.feature(q)['history']])
                direct_r = float(ridge_predict(model.X, model.mu, model.w, X)[0])
                kappa_r = float(ridge_predict(model.X, model.kappa, model.w, X)[0])
                direct_et = float(leaf_predict(model.direct_et, model.X, model.mu, model.w, X)[0])
                kappa_et = float(leaf_predict(model.kappa_et, model.X, model.kappa, model.w, X)[0])
                regions = context.data.index.region.to_numpy()[model.coarse_ids]
                names, counts = np.unique(regions, return_counts=True)
                cw = np.asarray([1 / len(names) / counts[np.where(names == r)[0][0]] for r in regions])
                a_hat = float(leaf_predict(model.coarse_model, model.coarse_X, model.coarse_y, cw, X)[0])
                independent = {'direct_ridge36': (a_hat, a_hat - direct_r),
                               'direct_et36': (a_hat, a_hat - direct_et),
                               'decomp_ridge36': (a_hat, kappa_r), 'decomp_et36': (a_hat, kappa_et)}
                independent.update({n: dynamic_prediction(model, q, n) for n in ('dynamic_pooled', 'dynamic_random')})
                prefix, _ = independent_feature(context, q)
                last = prefix[-1]
                independent['anchor'] = (float(context.data.shares[last] @ EF), 0.)
                annual = [p for p in prefix if context.data.dates.iloc[p].year == pd.Timestamp(origin).year - 1
                          and context.data.dates.iloc[p].month == pd.Timestamp(origin).month]
                independent['seasonal'] = (float(context.data.shares[annual[0]] @ EF), 0.) if annual else independent['anchor']
                for rec in current:
                    a2, k2 = independent[rec['method']]
                    max_prediction_error = max(max_prediction_error, close(
                        [rec['predicted_a'], rec['predicted_kappa'], rec['predicted_mu']],
                        [a2, k2, float(np.clip(a2 - k2, 0, 1000))], 1e-7, 'independent prediction'))
            # Read held labels only after all prediction refits/independent checks.
            raw = context.index.iloc[q]
            g = context.raw[int(raw.offset):int(raw.stop)]
            mu = float(np.mean(np.sum(g * EF, axis=1) / np.sum(g, axis=1)))
            a = float(np.sum(context.data.shares[q] * EF))
            prediction_columns = ['target', 'origin', 'allocation', 'profile_id', 'method',
                                  'predicted_a', 'predicted_kappa', 'mu_unclipped', 'predicted_mu', 'common_coarse_a']
            if set(prediction_rows[0]) != set(prediction_columns):
                raise ValueError('prediction JSON schema')
            frame = pd.DataFrame(prediction_rows, columns=prediction_columns)
            for key, value in {'actual_mu': mu, 'actual_a': a, 'actual_kappa': a - mu,
                               'valid_hours': len(g)}.items():
                frame[key] = value
            frame['mu_ae'] = abs(frame.predicted_mu - mu);frame['mu_se'] = (frame.predicted_mu - mu) ** 2
            frame['kappa_ae'] = abs(frame.predicted_kappa - (a - mu));frame['a_ae'] = abs(frame.predicted_a - a)
            keys = ['allocation', 'method']
            frame = frame.sort_values(keys).reset_index(drop=True);saved = saved.sort_values(keys).reset_index(drop=True)
            if list(frame.columns) != list(saved.columns):
                raise ValueError('score schema')
            for col in frame.columns:
                if pd.api.types.is_numeric_dtype(frame[col]):
                    max_score_error = max(max_score_error, close(frame[col], saved[col], 1e-7, 'score ' + col))
                elif not frame[col].equals(saved[col]):
                    raise ValueError('score identities')
            results.append(frame)
        print(f'audit {origin} 19 holdouts complete', flush=True)
    scores = pd.concat(results, ignore_index=True)
    keys = ['target', 'origin', 'allocation', 'method']
    root = pd.read_csv(SOURCE / 'scores.csv').sort_values(keys).reset_index(drop=True)
    rebuilt = scores.sort_values(keys).reset_index(drop=True)
    if len(scores) != 912 or scores.duplicated(keys).any() or not root[keys].equals(rebuilt[keys]):
        raise ValueError('full root query keys')
    for c in rebuilt.columns:
        if pd.api.types.is_numeric_dtype(rebuilt[c]):
            close(root[c], rebuilt[c], 1e-7, 'root scores')
    metrics = ['mu_ae', 'mu_se', 'kappa_ae', 'a_ae']
    regional = rebuilt.groupby(['allocation', 'method', 'target'])[metrics].mean()
    macro = regional.groupby(['allocation', 'method']).mean().reset_index()
    macro['mu_rmse'] = np.sqrt(macro.mu_se)
    period = rebuilt.groupby(['allocation', 'method', 'origin'])[metrics].mean().reset_index()
    period['mu_rmse'] = np.sqrt(period.mu_se)
    for name, expected, key in [('macro', macro, ['allocation', 'method']), ('by_origin', period, ['allocation', 'method', 'origin'])]:
        saved = pd.read_csv(SOURCE / f'{name}.csv').sort_values(key).reset_index(drop=True)
        expected = expected.sort_values(key).reset_index(drop=True)
        if not saved[key].equals(expected[key]):
            raise ValueError('aggregate keys')
        close(saved.drop(columns=key), expected.drop(columns=key), 1e-7, name)
    rng = np.random.default_rng(192);boot = rng.integers(0, 19, size=(2000, 19))
    comparisons = pd.read_csv(SOURCE / 'comparisons.csv')
    if len(comparisons) != 32 or comparisons.duplicated(['allocation', 'method', 'reference']).any():
        raise ValueError('comparison grid')
    for row in comparisons.itertuples(index=False):
        new = regional.loc[(row.allocation, row.method)].sort_index().mu_ae.to_numpy()
        old = regional.loc[(row.allocation, row.reference)].sort_index().mu_ae.to_numpy()
        delta = new - old;lo, hi = np.quantile(delta[boot].mean(axis=1), [.025, .975])
        close([row.mae_delta, row.relative_mae_reduction, row.regions_improved, row.descriptive_ci_low, row.descriptive_ci_high],
              [delta.mean(), 1 - new.mean() / old.mean(), (delta < 0).sum(), lo, hi], 1e-7, 'paired region bootstrap')
        if row.regions != 19 or row.queries != 57:
            raise ValueError('comparison denominator')
    for gate in summary['probability_extension_triggers']:
        method = gate['method']
        p = comparisons.loc[comparisons.allocation.eq('spread') & comparisons.method.eq(method)
                            & comparisons.reference.isin(['direct_ridge36', 'direct_et36'])]
        strong = bool(len(p) == 2 and p.relative_mae_reduction.ge(.05).all()
                      and p.regions_improved.ge(12).all() and p.descriptive_ci_high.lt(0).all())
        temporal = all(float(period.loc[period.allocation.eq('spread') & period.method.eq(method) & period.origin.eq(o), 'mu_ae'].iloc[0])
                       < min(float(period.loc[period.allocation.eq('spread') & period.method.eq(ref) & period.origin.eq(o), 'mu_ae'].iloc[0])
                             for ref in ('direct_ridge36', 'direct_et36')) for o in ORIGINS[1:])
        reduction = float(1 - regional.loc[('spread', method)].mu_ae.mean() / regional.loc[('recent', method)].mu_ae.mean())
        if gate['strong_direct_checks'] != strong or gate['May_June_each_beats_both_direct'] != temporal:
            raise ValueError('trigger component')
        close([gate['spread_vs_recent_mae_reduction']], [reduction], 1e-10, 'allocation trigger')
        if gate['probability_extension_trigger'] != bool(strong and temporal and reduction >= .02):
            raise ValueError('probability extension decision')
    if any(sha(ROOT / key) != value for key, value in hashes.items()):
        raise ValueError('audit final input hash')
    OUT.mkdir()
    info = {'assertion_failures': 0, 'source_profile_records_rebuilt': source_profiles,
            'sampled_source_hour_occurrences_rebuilt': source_hours, 'holdout_queries': 57,
            'score_records_rebuilt': len(scores), 'comparisons_rebuilt': 32,
            'max_score_error': max_score_error, 'max_independent_prediction_error': max_prediction_error,
            'refit_target_origin_allocations': 114, 'input_hashes': hashes, 'output_hashes': {},
            'scope': 'Source-internal retrospective forward development; shared tree growth, independently checked weighted leaf means/linear algebra/prefixes/losses; no hourly CIF distribution scoring.'}
    (OUT / 'summary.json').write_text(json.dumps(info, indent=2) + '\n')
    print({k: v for k, v in info.items() if not k.endswith('hashes')}, flush=True)


if __name__ == '__main__':
    main()
