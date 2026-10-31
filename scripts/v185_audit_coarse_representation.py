"""Rebuild the response/support factorial and check its arithmetic independently."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from coarse_representation_controls import NAMES, LEVEL, SUPPORTS
from extended_monthly_information import ExtendedMonthlyInformation
from monthly_history_features import monthly_context
from monthly_distribution_scoring import score, aggregate, COARSE_METRICS, FUEL_METRICS, NONCONSTANT
from v151_hourly_law import EF, sha
from v170_fuel_marginal_diagnosis import verify
from v179_audit_increment_support import bisection_projection, compare_frames
from v184_coarse_history_value import source_matrices, source_prefix_rows
from v184_audit_coarse_history_value import independent_losses, raw_weights
from v185_coarse_representation import ROOT, OUT as SOURCE, CONTEXT, checked_hashes, fit_fold, compare_old

OUT = ROOT / 'results/v185-coarse-representation-audit-r1'


def independent_laws(model, q, laws):
    """Recompute raw region weights, empirical leaves, and bisection supports."""
    b = model.base
    raw = raw_weights(b.regions)
    prior = raw / raw.sum()
    mean = (prior[:, None] * b.current).sum(axis=0)
    scale = np.maximum(np.sqrt((prior[:, None] * (b.current - mean)**2).sum(axis=0)), 1e-3)
    target = (b.current - mean) / scale
    error = max(float(np.max(abs(mean - model.mean))), float(np.max(abs(scale - model.scale))),
                float(np.max(abs(target - model.target))))
    context = monthly_context(b.data, q)
    S = b.data.shares[context['local']]
    supports = {'increment': bisection_projection(S + b.delta), 'absolute': b.current}
    logp = np.log(prior) - ((np.sqrt(b.previous) - np.sqrt(S))**2).sum(axis=1) / .18
    pH = np.exp(np.maximum(logp - logp.max(), -700)); pH /= pH.sum()
    probabilities = {'hellinger': pH}
    for name, fit in model.models.items():
        feature = model.features[name]
        training_leaves = model.leaves[name]
        query_leaves = fit.apply(context[feature][None])[0]
        p = np.zeros(len(b.ids))
        for tree, leaf in enumerate(query_leaves):
            mask = training_leaves[:, tree] == leaf
            p[mask] += raw[mask] / np.sum(raw[mask]) / len(fit.estimators_)
        probabilities[name] = .95 * p + .05 * pH
        if name in LEVEL:
            # The forest's point prediction must equal the same unsmoothed leaf mixture.
            error = max(error, float(np.max(abs(p @ target - fit.predict(context[feature][None])[0]))))
    if len(np.unique(EF[S > 0])) == 1:
        anchor = S @ EF
    else:
        anchor = float(np.clip(S @ EF + b.anchor_model.predict(context['base'][None])[0], 0, 1000))
    for support, values in supports.items():
        logits = np.log(pH) - .5 * ((values @ EF - anchor) / b.sigma)**2
        gaussian = np.exp(np.maximum(logits - logits.max(), -700)); gaussian /= gaussian.sum()
        for name in NAMES:
            expected = gaussian if name == 'gaussian' else probabilities[name]
            v, p = laws[f'{support}__{name}']
            error = max(error, float(np.max(abs(v - values))), float(np.max(abs(p - expected))))
    if error > 1e-10:
        raise ValueError(f'independent law/response arithmetic differs: {error}')
    return error


def audit_fold(data, inventory, fold):
    folder = SOURCE / f'fold_{fold}'
    verify(folder)
    model, queries = fit_fold(data, inventory, fold)
    if model.metadata() != json.loads((folder / 'model_fingerprints.json').read_text()):
        raise ValueError('model refit fingerprint differs')
    stored = np.load(folder / 'training_matrices.npz', allow_pickle=False)
    matrices = source_matrices(model.base) | {'level_target': model.target, 'level_mean': model.mean, 'level_scale': model.scale}
    if set(stored.files) != set(matrices) or any(not np.array_equal(stored[k], v) for k, v in matrices.items()):
        raise ValueError('source matrices not rebuilt')
    compare_frames(pd.DataFrame(source_prefix_rows(model.base)), pd.read_csv(folder / 'training_prefix.csv'), ['profile_id'])
    for q in queries.profile_id.iloc[[0, -1]]:
        model.future_guard(int(q))
    coarse = pd.read_csv(folder / 'coarse_month_metrics.csv')
    fuels = pd.read_csv(folder / 'coarse_fuel_metrics.csv')
    if (not np.isfinite(coarse[list(COARSE_METRICS)].to_numpy(float)).all()
            or not np.isfinite(fuels[list(FUEL_METRICS)].to_numpy(float)).all()):
        raise ValueError('nonfinite saved score metric')
    compare_old(coarse, fuels, fold)
    ci = coarse.set_index(['profile_id', 'method'])
    fi = fuels.set_index(['profile_id', 'method', 'fuel_index'])
    index = pd.read_csv(folder / 'prediction_index.csv')
    packed = np.load(folder / 'coarse_predictions.npz', allow_pickle=False)
    values, probabilities = packed['values'], packed['probabilities']
    if (len(coarse) != len(queries) * 16 or len(fuels) != len(queries) * 128 or len(index) != len(coarse)
            or ci.index.duplicated().any() or fi.index.duplicated().any() or index.duplicated(['profile_id', 'method']).any()
            or index.coarse_offset.iloc[0] != 0 or not np.array_equal(index.coarse_offset.to_numpy()[1:], index.coarse_stop.to_numpy()[:-1])
            or (index.coarse_stop <= index.coarse_offset).any() or index.coarse_stop.iloc[-1] != len(values)
            or values.shape != (len(probabilities), 8) or not np.isfinite(values).all()
            or not np.isfinite(probabilities).all() or (probabilities <= 0).any()):
        raise ValueError('complete packed prediction inventory differs')
    index = index.set_index(['profile_id', 'method'])
    maximum = {'score': 0., 'independent_loss': 0., 'independent_law': 0.}
    for number, rec in enumerate(queries.itertuples(index=False), start=1):
        q = int(rec.profile_id)
        laws = model.build(q)
        maximum['independent_law'] = max(maximum['independent_law'], independent_laws(model, q, laws))
        truth = data.shares[q].copy()
        cache, independent_cache = {}, {}
        for name, (v, p) in laws.items():
            packed_row = index.loc[(q, name)]
            a, z = int(packed_row.coarse_offset), int(packed_row.coarse_stop)
            if (packed_row.region != rec.region or packed_row.month != rec.month or int(packed_row.fold) != fold
                    or not np.array_equal(v, values[a:z]) or not np.array_equal(p, probabilities[a:z])):
                raise ValueError('packed law/identity differs')
            c, f = score(v, p, truth, cache)
            independent, independent_fuels = independent_losses(v, p, truth, independent_cache)
            saved = ci.loc[(q, name)]
            if saved.region != rec.region or saved.month != rec.month or int(saved.fold) != fold:
                raise ValueError('coarse identity differs')
            maximum['score'] = max(maximum['score'], max(abs(c[m] - saved[m]) for m in COARSE_METRICS))
            maximum['independent_loss'] = max(maximum['independent_loss'], max(abs(c[m] - loss) for m, loss in independent.items()))
            for fuel, metrics in enumerate(f):
                saved_fuel = fi.loc[(q, name, fuel)]
                if saved_fuel.region != rec.region or saved_fuel.month != rec.month or int(saved_fuel.fold) != fold:
                    raise ValueError('fuel identity differs')
                maximum['score'] = max(maximum['score'], max(abs(float(metrics[m]) - float(saved_fuel[m])) for m in FUEL_METRICS))
                maximum['independent_loss'] = max(maximum['independent_loss'], max(abs(metrics[m] - loss) for m, loss in independent_fuels[fuel].items()))
        if number % 10 == 0 or number == len(queries):
            print(f'V185 audit fold {fold}: {number}/{len(queries)}', flush=True)
    if max(maximum.values()) > 1e-8:
        raise ValueError('score rebuild or independent arithmetic differs')
    return {'fold': fold, 'coarse_rows': len(coarse), 'fuel_rows': len(fuels), **maximum}


def comparisons(regions):
    """Descriptive paired-region comparisons fixed in the protocol."""
    rows = []
    metrics = ('energy_score', 'same_EF_energy_score', 'anchor_crps', 'share_crps_mean')
    pairs = [(f'{s}__{a}', f'{s}__{b}', 'same_support') for s in SUPPORTS for a in NAMES for b in NAMES if a != b]
    pairs += [(f'absolute__{a}', f'increment__{a}', 'support_control') for a in NAMES]
    if len(pairs) != 120:
        raise ValueError('comparison method count')
    for cohort in ('all29', 'nonconstant24'):
        frame = regions if cohort == 'all29' else regions.loc[~regions.region.isin(NONCONSTANT)]
        roster = sorted(frame.region.unique())
        if len(roster) != (29 if cohort == 'all29' else 24):
            raise ValueError('comparison region count')
        bootstrap = np.random.default_rng(185).integers(0, len(roster), size=(10000, len(roster)))
        for method, reference, kind in pairs:
            a = frame.loc[frame.method.eq(method)].set_index('region').loc[roster]
            b = frame.loc[frame.method.eq(reference)].set_index('region').loc[roster]
            row = {'cohort': cohort, 'method': method, 'reference': reference, 'comparison': kind, 'regions': len(roster),
                   'anchor_coverage_difference': float(a.anchor_coverage.mean() - b.anchor_coverage.mean()),
                   'anchor_width_change_percent': float(100 * (a.anchor_width.mean() / b.anchor_width.mean() - 1))}
            for metric in metrics:
                d = (a[metric] - b[metric]).to_numpy()
                lo, hi = np.quantile(d[bootstrap].mean(axis=1), [.025, .975])
                row |= {metric + '_difference': float(d.mean()), metric + '_reduction_percent': float(-100 * d.mean() / b[metric].mean()),
                        metric + '_bootstrap_lo': float(lo), metric + '_bootstrap_hi': float(hi), metric + '_regions_better': int((d < 0).sum())}
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    upstream = verify(SOURCE)
    hashes = dict(upstream['input_hashes'])
    hashes |= {str((SOURCE / p).relative_to(ROOT)): h for p, h in upstream['output_hashes'].items()}
    for p in (SOURCE / 'summary.json', Path(__file__), ROOT / 'scripts/v184_audit_coarse_history_value.py',
              ROOT / 'scripts/v179_audit_increment_support.py'):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    checked_hashes()
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input provenance differs')
    data = ExtendedMonthlyInformation(ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
    inventory = pd.read_csv(CONTEXT / 'training_inventory.csv')
    with threadpool_limits(limits=2):
        records = [audit_fold(data, inventory, fold) for fold in range(6)]
    coarse = pd.concat([pd.read_csv(SOURCE / f'fold_{k}/coarse_month_metrics.csv') for k in range(6)], ignore_index=True)
    fuels = pd.concat([pd.read_csv(SOURCE / f'fold_{k}/coarse_fuel_metrics.csv') for k in range(6)], ignore_index=True)
    for name, table in [('coarse_month_metrics', coarse), ('coarse_fuel_metrics', fuels)]:
        if (SOURCE / f'{name}.csv').read_text() != table.to_csv(index=False):
            raise ValueError('root producer text binding differs')
    tables = dict(zip(('coarse_region_metrics', 'coarse_macro_metrics', 'coarse_fuel_region_metrics', 'coarse_fuel_macro_metrics'), aggregate(coarse, fuels)))
    for name, table in tables.items():
        keys = [k for k in ('cohort', 'region', 'method', 'fuel_index') if k in table]
        compare_frames(table, pd.read_csv(SOURCE / f'{name}.csv'), keys)
    comparison = comparisons(tables['coarse_region_metrics'])
    if len(comparison) != 240 or sum(r['coarse_rows'] for r in records) != 3856 or sum(r['fuel_rows'] for r in records) != 30848:
        raise ValueError('audit denominator differs')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input changed')
    OUT.mkdir()
    pd.DataFrame(records).to_csv(OUT / 'reconstruction.csv', index=False)
    comparison.to_csv(OUT / 'comparisons.csv', index=False)
    result = {'assertion_failures': 0, 'coarse_rows_rebuilt': 3856, 'fuel_rows_rebuilt': 30848, 'packed_laws_rebuilt': 3856,
              'refitted_fold_groups': 6, 'level_forests_refitted': 12, 'future_query_cases': 12, 'future_method_guards': 192,
              'reference_coarse_rows': 1205, 'reference_fuel_rows': 9640, 'comparisons': len(comparison),
              **{'max_' + key + '_error': max(r[key] for r in records) for key in ('score', 'independent_loss', 'independent_law')},
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Full coarse-only rebuild; independent level normalization, empirical leaves, bisection supports and pairwise losses. Tree growing and exact quantiles shared. No hourly or original-method claim.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
