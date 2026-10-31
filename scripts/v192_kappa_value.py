"""Frozen V192 preflight or source-internal LOO monthly-mean development study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from v151_hourly_law import EF, sha
from v192_reallocated_kappa import ROOT, ALLOCATIONS, METHODS, ORIGINS, Context, Predictor

DOC = ROOT / 'docs/V192_REALLOCATED_KAPPA_PROTOCOL.md'
OUT = ROOT / 'results/v192-reallocated-kappa-r1'
PRE = ROOT / 'results/v192-reallocated-kappa-preflight-r1'


def dependencies() -> list[Path]:
    return [Path(__file__), DOC, ROOT / 'scripts/v192_reallocated_kappa.py',
            ROOT / 'scripts/monthly_history_features.py', ROOT / 'scripts/extended_monthly_information.py',
            ROOT / 'scripts/v151_hourly_law.py', ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv',
            ROOT / 'results/v159-fuel-activity-source-pilot-r1/source_sampling.csv',
            ROOT / 'results/v144-canonical-profile-cache-r1/profiles.npz',
            ROOT / 'results/v144-canonical-profile-cache-r1/profile_index.csv',
            ROOT / 'results/v153-calendar-times-r1/timestamps.npz',
            ROOT / 'results/v188-forward-zero-risk-r1/fold_0/training_matrices.npz',
            ROOT / 'results/v188-forward-zero-risk-r1/fold_0/training_provenance.json']


def input_hashes() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha(p) for p in dependencies()}


def verify_hashes(hashes: dict[str, str]) -> None:
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('frozen input changed')


def source_rows(model: Predictor) -> list[dict]:
    return [{'region': r['region'], 'profile_id': int(r['profile_id']),
             'raw_ids': [int(p) for p in r['raw_ids']],
             'prefix': [int(p) for p in r['prefix']]} for r in model.records]


def preflight(context: Context, hashes: dict[str, str]) -> None:
    if PRE.exists():
        raise FileExistsError(PRE)
    checks, cases = [], []
    for origin in ORIGINS:
        for target in (context.roster[0], context.roster[-1]):
            for allocation in ALLOCATIONS:
                original = Predictor(context, target, origin, allocation)
                baseline = pd.DataFrame(original.predict())
                serialized = json.loads(json.dumps(source_rows(original)))
                for old, new in zip(original.records, serialized):
                    if (list(old['raw_ids']) != new['raw_ids'] or old['prefix'] != new['prefix']
                            or new['region'] == target):
                        raise ValueError('source serializer/role')
                    for p in new['prefix']:
                        if context.data.release.iloc[p] > context.data.dates.iloc[new['profile_id']]:
                            raise ValueError('serialized prefix time')
                # Intervention reads no held truth: replace unavailable M with constants.
                future = (context.data.release > pd.Timestamp(origin)).to_numpy()
                M, S = context.data.M.copy(), context.data.shares.copy()
                try:
                    context.data.M[future] = np.arange(1, 9) * 12345.
                    context.data.shares[future] = np.arange(1, 9) / 36.
                    context.clear_features()
                    fresh = pd.DataFrame(Predictor(context, target, origin, allocation).predict())
                    pd.testing.assert_frame_equal(baseline, fresh, check_exact=True)
                finally:
                    context.data.M[:], context.data.shares[:] = M, S
                    context.clear_features()
                # Poison all held-region fine rows; fitting must never access them.
                raw = context.raw
                if raw is None:
                    raise ValueError('source bank not loaded')
                context.raw = raw.copy()
                try:
                    for rec in context.index.loc[context.index.region.eq(target)].itertuples(index=False):
                        context.raw[int(rec.offset):int(rec.stop)] = 123.
                    fresh = pd.DataFrame(Predictor(context, target, origin, allocation).predict())
                    pd.testing.assert_frame_equal(baseline, fresh, check_exact=True)
                finally:
                    context.raw = raw
                # Coarse support/model is identical for the two allocation modes.
                checks.append({'origin': origin, 'target': target, 'allocation': allocation,
                               'source_profiles': len(original.records),
                               'source_hours': sum(len(r['raw_ids']) for r in original.records),
                               'future_M_fresh_refit_exact': True, 'held_region_raw_fresh_refit_exact': True})
                cases.extend(baseline.to_dict('records'))
            print(f'preflight {origin} {target} complete', flush=True)
    table = pd.DataFrame(cases)
    for _, group in table.groupby(['target', 'origin', 'method']):
        if group.common_coarse_a.nunique() != 1:
            raise ValueError('coarse predictor allocation negative control')
        if (group.method.iloc[0] in ('anchor', 'seasonal') or group.origin.iloc[0] == ORIGINS[0]) and group.predicted_mu.nunique() != 1:
            raise ValueError('simple baseline allocation negative control')
    verify_hashes(hashes)
    PRE.mkdir()
    pd.DataFrame(checks).to_csv(PRE / 'checks.csv', index=False)
    table.to_csv(PRE / 'predictions_no_truth.csv', index=False)
    summary = {'assertion_failures': 0, 'query_cases': 6, 'allocation_cases': 12,
               'prediction_records': 96, 'fresh_future_M_refits': 12,
               'fresh_held_region_raw_refits': 12, 'held_truth_scored': False,
               'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in PRE.iterdir()}}
    (PRE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('preflight complete, no held truth scoring', flush=True)


def aggregate(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ['mu_ae', 'mu_se', 'kappa_ae', 'a_ae']
    regional = scores.groupby(['allocation', 'method', 'target'], as_index=False)[metrics].mean()
    macro = regional.groupby(['allocation', 'method'], as_index=False)[metrics].mean()
    macro['mu_rmse'] = np.sqrt(macro.mu_se)
    period = scores.groupby(['allocation', 'method', 'origin'], as_index=False)[metrics].mean()
    period['mu_rmse'] = np.sqrt(period.mu_se)
    return macro, period


def comparisons(scores: pd.DataFrame) -> pd.DataFrame:
    regional = scores.groupby(['allocation', 'method', 'target']).mu_ae.mean()
    rng = np.random.default_rng(192)
    boot = rng.integers(0, 19, size=(2000, 19))
    rows = []
    for allocation in ALLOCATIONS:
        for method in METHODS[4:]:
            for reference in ('direct_ridge36', 'direct_et36', 'anchor', 'seasonal'):
                new = regional.loc[(allocation, method)].sort_index()
                old = regional.loc[(allocation, reference)].sort_index()
                if not new.index.equals(old.index) or len(new) != 19:
                    raise ValueError('comparison cohort')
                delta = (new - old).to_numpy()
                ci = np.quantile(delta[boot].mean(axis=1), [.025, .975])
                rows.append({'allocation': allocation, 'method': method, 'reference': reference,
                             'regions': 19, 'queries': 57, 'mae_delta': float(delta.mean()),
                             'relative_mae_reduction': float(1 - new.mean() / old.mean()),
                             'regions_improved': int((delta < 0).sum()),
                             'descriptive_ci_low': float(ci[0]), 'descriptive_ci_high': float(ci[1])})
    return pd.DataFrame(rows)


def triggers(scores: pd.DataFrame, paired: pd.DataFrame) -> list[dict]:
    regional = scores.groupby(['allocation', 'method', 'target']).mu_ae.mean()
    answer = []
    for method in METHODS[4:]:
        p = paired.loc[paired.allocation.eq('spread') & paired.method.eq(method)
                       & paired.reference.isin(['direct_ridge36', 'direct_et36'])]
        head = bool(len(p) == 2 and p.relative_mae_reduction.ge(.05).all()
                    and p.regions_improved.ge(12).all() and p.descriptive_ci_high.lt(0).all())
        periods = True
        for origin in ORIGINS[1:]:
            g = scores.loc[scores.allocation.eq('spread') & scores.origin.eq(origin)].groupby('method').mu_ae.mean()
            periods &= bool(g[method] < min(g['direct_ridge36'], g['direct_et36']))
        gain = float(1 - regional.loc[('spread', method)].mean() / regional.loc[('recent', method)].mean())
        answer.append({'method': method, 'strong_direct_checks': head,
                       'May_June_each_beats_both_direct': periods,
                       'spread_vs_recent_mae_reduction': gain,
                       'probability_extension_trigger': bool(head and periods and gain >= .02)})
    return answer


def run(context: Context, hashes: dict[str, str]) -> None:
    pre = json.loads((PRE / 'summary.json').read_text())
    if pre['input_hashes'] != hashes or not pre['held_truth_scored'] is False:
        raise ValueError('preflight is not current/no-truth')
    OUT.mkdir(exist_ok=True)
    manifest = {'input_hashes': hashes, 'preflight_summary_sha256': sha(PRE / 'summary.json')}
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('resume manifest changed')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    all_scores = []
    for origin in ORIGINS:
        for target in context.roster:
            name = f'{origin}_{target}'
            final = OUT / name
            if final.exists():
                block = json.loads((final / 'summary.json').read_text())
                if block['manifest'] != manifest:
                    raise ValueError('resume block inputs')
                for file, digest in block['output_hashes'].items():
                    if sha(final / file) != digest:
                        raise ValueError('resume block corrupt')
                all_scores.append(pd.read_csv(final / 'scores.csv'))
                continue
            models, predictions = {}, []
            for allocation in ALLOCATIONS:
                model = Predictor(context, target, origin, allocation)
                models[allocation] = model
                predictions.extend(model.predict())
            # Seal every method and both allocation predictions before target labels.
            frozen = json.dumps(predictions, sort_keys=True)
            q = context.profile(target, origin)
            row = context.index.iloc[q]
            if context.raw is None:
                raise ValueError('fine bank not initialized')
            g = context.raw[int(row.offset):int(row.stop)]
            if (g.sum(axis=1) <= 0).any() or not np.isfinite(g).all():
                raise ValueError('query positive-generation truth')
            mu = float(np.mean((g @ EF) / g.sum(axis=1)))
            a = float(context.data.shares[q] @ EF)
            truth = {'actual_mu': mu, 'actual_a': a, 'actual_kappa': a - mu, 'valid_hours': len(g)}
            scores = pd.DataFrame(predictions)
            for key, value in truth.items():
                scores[key] = value
            scores['mu_ae'] = abs(scores.predicted_mu - mu)
            scores['mu_se'] = (scores.predicted_mu - mu) ** 2
            scores['kappa_ae'] = abs(scores.predicted_kappa - (a - mu))
            scores['a_ae'] = abs(scores.predicted_a - a)
            if json.dumps(predictions, sort_keys=True) != frozen or len(scores) != 16:
                raise ValueError('prediction seal/count')
            temp = Path(tempfile.mkdtemp(prefix='.v192-', dir=OUT))
            scores.to_csv(temp / 'scores.csv', index=False)
            (temp / 'predictions.json').write_text(frozen + '\n')
            prefixes = {str(p): [int(x) for x in context.feature(int(p))['prefix']]
                        for model in models.values() for p in np.r_[model.profile_ids, model.coarse_ids, q]}
            (temp / 'prefixes.json').write_text(json.dumps(prefixes, sort_keys=True) + '\n')
            for allocation, model in models.items():
                np.savez_compressed(temp / f'{allocation}_matrices.npz', **model.arrays())
                (temp / f'{allocation}_source.json').write_text(json.dumps(source_rows(model)) + '\n')
            block = {'manifest': manifest, 'target': target, 'origin': origin, 'prediction_records': 16,
                     'query_prefix': [int(p) for p in context.feature(q)['prefix']],
                     'output_hashes': {p.name: sha(p) for p in temp.iterdir()}}
            (temp / 'summary.json').write_text(json.dumps(block, indent=2) + '\n')
            os.rename(temp, final)
            all_scores.append(scores)
            print(f'{origin} {target} complete, 16 predictions scored', flush=True)
    scores = pd.concat(all_scores, ignore_index=True)
    if len(scores) != 912 or scores.duplicated(['target', 'origin', 'allocation', 'method']).any():
        raise ValueError('full score grid')
    macro, period = aggregate(scores)
    paired = comparisons(scores)
    gate = triggers(scores, paired)
    verify_hashes(hashes)
    for name, frame in [('scores', scores), ('macro', macro), ('by_origin', period), ('comparisons', paired)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    summary = {'assertion_failures': 0, 'queries': 57, 'regions': 19, 'origins': list(ORIGINS),
               'score_records': len(scores), 'methods_per_allocation': 8, 'allocation_modes': list(ALLOCATIONS),
               'probability_extension_triggers': gate, 'new_CIF_distribution_scores': False,
               'original_goal_complete': False, 'input_hashes': hashes,
               'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
                                 if p.is_file() and p != OUT / 'summary.json'}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('all 57 cases complete', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    hashes = input_hashes()
    context = Context()
    if args.preflight:
        preflight(context, hashes)
    else:
        run(context, hashes)


if __name__ == '__main__':
    main()
