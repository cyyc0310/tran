"""Rebuild portfolio sources/laws and compare complete same-budget CIF scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation
from portfolio_activity import PortfolioPrediction, METHODS, REFERENCES, SETTINGS, MATCHED
from v151_hourly_law import EF, sha
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v173_moment_protected_risk import METRICS, aggregate
from v177_portfolio_value import BASE, INVENTORY, OUT as SOURCE, ROOT, check_mutation

OUT = ROOT / 'results/v177-portfolio-audit-r1'
NONCONSTANT = ['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA']


def audit_source_bank(predictor):
    """Independent raw-index intersection and fuel formulas for every world."""
    data, frozen = predictor.frozen.data, predictor.frozen
    worst = 0.
    checked = 0
    for world, proof in zip(predictor.worlds, predictor.source_proofs):
        qs, locals_ = world['profiles'], world['inputs']
        maps = []
        for q in qs:
            region = data.index.region.iloc[q]
            source = frozen.activity.sources[region]
            picks = source['allowed'][source['profiles'] == q]
            maps.append({int(predictor.seconds[i]): int(i) for i in picks})
        common = sorted(set(maps[0]) & set(maps[1]))
        raw = np.asarray([[maps[0][t], maps[1][t]] for t in common], np.int64)
        if (len(common) != proof['shared_hours'] or hashlib.sha256(raw.tobytes()).hexdigest() != proof['raw_sha256']
                or any(data.local(q, 'asof_30d') != p for q, p in zip(qs, locals_))):
            raise ValueError('independent source intersection or inputs differ')
        g = data.raw[raw]
        states = data.states[raw[:, 0]]
        T = data.index.hours.iloc[qs[0]]
        past = data.M[locals_] / T
        volumes = past.sum(axis=1)
        aggregate_S = past.sum(axis=0) / volumes.sum()
        B = .99 * aggregate_S + .01 * frozen.prior
        beta = volumes / volumes.sum()
        current_S = data.M[qs].sum(axis=0) / data.M[qs].sum()
        if (max(abs(world['past'] - aggregate_S)) > 1e-14
                or max(abs(world['current'] - current_S)) > 1e-14):
            raise ValueError('independent source coarse composition differs')
        expected = {'pair_sync': g.sum(axis=1) / volumes.sum() / B}
        for seed in (1, 2):
            order = np.arange(len(common))
            for state in sorted(set(states.tolist())):
                ids = np.flatnonzero(states == state)
                key = f'v177/perm/{",".join(world["parents"])}/{world["month"]}/{state}/{seed}'
                number = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
                order[ids] = np.random.default_rng(number).permutation(ids)
                if not np.array_equal(np.sort(order[ids]), ids):
                    raise ValueError('independent permutation not within calendar cell')
            if hashlib.sha256(np.asarray(order, np.int64).tobytes()).hexdigest() != proof[f'permutation{seed}_sha256']:
                raise ValueError('independent permutation hash differs')
            expected[f'pair_permute{seed}'] = (g[:, 0] + g[order, 1]) / volumes.sum() / B
        for h in range(24):
            take = np.flatnonzero(states % 24 == h)
            n = len(take)
            for mode in ('pair_sync', 'pair_permute1', 'pair_permute2', 'pair_mixture', 'selected_parent'):
                actual, tags, weights = world['cells'][mode][h]
                if mode in expected:
                    wanted = expected[mode][take]
                    tag = np.zeros(n, int)
                    weight = np.full(n, 1 / n)
                else:
                    parent = []
                    for j in range(2):
                        carrier = B if mode == 'pair_mixture' else .99 * past[j] / volumes[j] + .01 * frozen.prior
                        parent.append(g[take, j] / volumes[j] / carrier)
                    wanted = np.concatenate(parent)
                    tag = np.zeros(2 * n, int) if mode == 'pair_mixture' else np.repeat([0, 1], n)
                    weight = (np.r_[np.full(n, beta[0] / n), np.full(n, beta[1] / n)]
                              if mode == 'pair_mixture' else np.full(2 * n, 1 / n))
                error = float(np.max(abs(actual - wanted) / np.maximum(1., abs(wanted))))
                worst = max(worst, error)
                if error > 1e-12 or not np.array_equal(tags, tag) or max(abs(weights - weight)) > 1e-14:
                    raise ValueError('independent portfolio fuel/probability formula differs')
                checked += 1
    return checked, worst


def independent_weights(bank, S, B, forecast, gaussian):
    divergence = .5 * np.sum((np.sqrt(bank.past) - np.sqrt(S))**2, axis=1)
    transformed = bank.current * B / bank.source_B
    anchors = transformed @ EF / transformed.sum(axis=1)
    logits = np.log(bank.prior) - divergence / .09
    if gaussian:
        logits -= .5 * ((anchors - forecast['anchor']) / forecast['sigma'])**2
    probability = np.exp(np.clip(logits - max(logits), -700, 0))
    return probability / probability.sum()


def audit_fold(data, monthly, fold):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    predictor = PortfolioPrediction(frozen, monthly, INVENTORY, ROOT).prepare(fold)
    source_cells, source_error = audit_source_bank(predictor)
    block = SOURCE / f'fold_{fold}'
    saved_summary = verify(block)
    if predictor.scale_error != saved_summary['max_proportional_activity_error']:
        raise ValueError('scaling negative control differs')
    proof = pd.read_csv(block / 'source_worlds.csv')
    rebuilt = pd.DataFrame(predictor.source_proofs)
    pair = rebuilt.merge(proof, on=['fold', 'parents', 'month'], validate='one_to_one', suffixes=('_a', '_b'))
    if len(pair) != len(proof) or len(pair) != len(rebuilt):
        raise ValueError('source world inventory differs')
    for field in ['profiles', 'input_profiles', 'shared_hours', 'clock_cells', 'raw_sha256',
                  'permutation1_sha256', 'permutation2_sha256']:
        if not np.array_equal(pair[field + '_a'], pair[field + '_b']):
            raise ValueError('source world proof differs')
    if not np.isfinite(pair.matched_moment_error_b).all() or max(abs(pair.matched_moment_error_a - pair.matched_moment_error_b)) > 1e-12:
        raise ValueError('source moment proof differs')
    rows = pd.read_csv(block / 'test_month_metrics.csv').set_index(['profile_id', 'method', 'radius_policy', 'calibration'])
    contexts = pd.read_csv(block / 'contexts.csv').set_index('profile_id')
    details = pd.read_csv(block / 'recalibration_maps.csv').set_index(['profile_id', 'method'])
    radii = pd.read_csv(block / 'calibration_radii.csv').set_index(['method', 'calibration'])
    if (len(details) != len(frozen.queries) * 12 or details.index.duplicated().any()
            or len(radii) != 28 or radii.index.duplicated().any() or not np.isfinite(radii.radius).all()
            or set(contexts.index) != set(frozen.queries.profile_id) or contexts.index.duplicated().any()):
        raise ValueError('query diagnostic inventory')
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    errors = {name: {} for name in METHODS}
    checked = updated = physical = retained = 0
    max_score = max_radius = max_diagnostic = max_fuel = 0.
    for role in ('calibration', 'test'):
        for rec in frozen.queries.loc[frozen.queries.role.eq(role)].itertuples(index=False):
            q = int(rec.profile_id)
            laws, forecast, new_details = predictor.build(q)
            forecast2, S, B, donors, prior, anchors = predictor.physical.inputs(q)
            old_context = contexts.loc[q]
            if (old_context.role, old_context.region, old_context.month, old_context.fold) != (role, rec.region, rec.month, fold):
                raise ValueError('context identity')
            for key in ('raw_anchor', 'anchor', 'past_anchor', 'sigma', 'local'):
                error = abs(float(old_context[key]) - float(forecast[key]))
                if not np.isfinite(error) or forecast2[key] != forecast[key]:
                    raise ValueError('forecast context differs')
                max_diagnostic = max(max_diagnostic, error)
            logits = np.log(prior) - .5 * ((anchors - forecast['anchor']) / forecast['sigma'])**2
            gaussian = np.exp(np.maximum(logits - max(logits), -700))
            gaussian /= gaussian.sum()
            prepared = [frozen.activity.donor(int(d)) for d in donors]
            base_means = {reference: np.asarray([sum(w * a[h].mean(axis=0) for w, a in zip(weights, prepared)) * B
                                                for h in range(24)])
                          for reference, weights in zip(REFERENCES, (prior, gaussian))}
            geometry = predictor.geometry(int(frozen.month[q])) | {'expanded_original': predictor.expanded_geometry()}
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)] % 24
            occupancy = np.bincount(clock, minlength=24) / len(clock)
            fuel_by_method = {}
            for detail in new_details:
                method = detail['method']
                old = details.loc[(q, method)]
                if (old.role, old.region, old.month, old.fold) != (role, rec.region, rec.month, fold):
                    raise ValueError('query detail identity')
                if json.loads(old.worlds) != json.loads(detail['worlds']):
                    raise ValueError('selected portfolio worlds differ')
                for key in ('weights', 'fuel_mean'):
                    a, b = np.asarray(json.loads(detail[key])), np.asarray(json.loads(old[key]))
                    if a.shape != b.shape or not np.isfinite(b).all():
                        raise ValueError('query probability/fuel dimensions or finiteness')
                    max_diagnostic = max(max_diagnostic, float(max(abs(a - b).ravel())))
                for key in ('generator_energy_anchor', 'matched_moment_error'):
                    if not np.isfinite(old[key]):
                        raise ValueError('nonfinite query diagnostic')
                    max_diagnostic = max(max_diagnostic, abs(detail[key] - old[key]))
                reference, mode = SETTINGS[method]
                bank = geometry[mode]
                weights = independent_weights(bank, S, B, forecast, reference == REFERENCES[1])
                old_weights = np.asarray(json.loads(old.weights))
                if (abs(old_weights.sum() - 1) > 1e-12 or not (old_weights > 0).all()
                        or max(abs(weights - old_weights)) > 1e-12):
                    raise ValueError('independent context weights differ')
                virtual = []
                for activity, component, within in bank.cells:
                    virtual.append((weights[component] * within) @ activity * B)
                wanted = np.asarray(virtual) if mode == 'expanded_original' else .5 * (base_means[reference] + virtual)
                fuel = np.asarray(json.loads(old.fuel_mean))
                error = float(np.max(abs(wanted - fuel) / wanted.sum(axis=1, keepdims=True)))
                max_fuel = max(max_fuel, error)
                energy_anchor = float(occupancy @ (fuel @ EF) / (occupancy @ fuel.sum(axis=1)))
                if error > 2e-12 or abs(energy_anchor - old.generator_energy_anchor) > 1e-8:
                    raise ValueError('independent predictive fuel moments differ')
                fuel_by_method[method] = fuel
                for h, law in enumerate(laws[method]):
                    if (not np.isfinite(law.x).all() or not np.isfinite(law.w).all()
                            or not (law.w > 0).all() or min(law.x) < -1e-10 or max(law.x) > 1000 + 1e-10):
                        raise ValueError('predictive physical law invalid')
                    physical += 1
                    if mode != 'expanded_original':
                        base = laws[reference][h]
                        ids = np.searchsorted(law.x, base.x)
                        if (not np.array_equal(law.x[ids], base.x) or min(law.w[ids] - .5 * base.w) < -1e-14):
                            raise ValueError('half mixture lost original support or mass')
                        retained += 1
                updated += 1
            for prefix in ('conditional', 'gaussian'):
                mean = fuel_by_method[f'{prefix}_pair_sync']
                for mode in MATCHED:
                    if np.max(abs(mean - fuel_by_method[f'{prefix}_{mode}']) / mean.sum(axis=1, keepdims=True)) > 2e-12:
                        raise ValueError('matched predictive means changed')
            actual = data.truth(q)
            for method in METHODS:
                if role == 'calibration':
                    law = CalendarLaw(laws[method], clock)
                    distance = np.maximum.reduce([law.quantile(.05) - actual, actual - law.quantile(.95), np.zeros(len(actual))])
                    errors[method].setdefault(rec.region, []).append(distance)
                else:
                    for policy in (('recalibrated',) if method in REFERENCES else ('fixed', 'recalibrated')):
                        for calibration in ('pooled', 'max_region'):
                            reference = method if method in REFERENCES else SETTINGS[method][0]
                            radius = radius_map[(method if policy == 'recalibrated' else reference, calibration)]
                            old = rows.loc[(q, method, policy, calibration)]
                            if (old.region, old.month, old.fold, old.hours) != (rec.region, rec.month, fold, len(actual)):
                                raise ValueError('score identity')
                            scores = score_laws(laws[method], actual, clock, radius)
                            delta = np.asarray([scores[k] - old[k] for k in scores])
                            if not np.isfinite(delta).all() or not np.isfinite(old.radius):
                                raise ValueError('nonfinite score or radius')
                            max_score = max(max_score, float(max(abs(delta))))
                            max_radius = max(max_radius, abs(radius - old.radius))
                            checked += 1
        if role == 'calibration':
            radius_map = {}
            for method in METHODS:
                if set(errors[method]) != set(frozen.queries.loc[frozen.queries.role.eq(role), 'region']):
                    raise ValueError('calibration roster')
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[method])):
                    radius_map[(method, label)] = radius
                    max_radius = max(max_radius, abs(radius - radii.loc[(method, label), 'radius']))
    if (checked != len(rows) or updated != len(details) or max_score > 1e-7 or max_radius > 1e-8
            or max_diagnostic > 1e-8 or max_fuel > 2e-12):
        raise ValueError('complete portfolio reconstruction failed')
    print(f'portfolio audit fold {fold}: {checked} scores, {source_cells} source cells', flush=True)
    return {'fold': fold, 'rebuilt_score_rows': checked, 'detail_rows_rebuilt': updated,
            'source_worlds_rebuilt': len(predictor.worlds), 'source_cells_independently_checked': source_cells,
            'max_source_formula_error': source_error, 'predictive_physical_cells': physical,
            'original_half_support_checks': retained, 'max_score_error': max_score,
            'max_radius_error': max_radius, 'max_diagnostic_error': max_diagnostic, 'max_fuel_moment_error': max_fuel}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = summary['input_hashes'] | {str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json'),
                                      str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    external = {'positive_recomposition': BASE, 'scalar_forest': BASE,
                'activity_mmd_forest': ROOT / 'results/v160-carbon-cdf-source-r1',
                'forecast_composition': ROOT / 'results/v164-scenario-reweighting-r1',
                'world_entropy': ROOT / 'results/v164-scenario-reweighting-r1',
                'hetero_history_barycentric': ROOT / 'results/v165-coarse-distribution-lifting-r1',
                'gaussian_protect_fuel8_direct20_a50': ROOT / 'results/v173-moment-protected-risk-r1',
                'gaussian_pit_conditional43_a50': ROOT / 'results/v174-cif-recalibration-r1',
                'gaussian_crps_global_a50': ROOT / 'results/v175-cif-crps-map-r1'}
    for directory in set(external.values()):
        verify(directory)
        for name in ('summary.json', 'test_month_metrics.csv'):
            hashes[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input changed')
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    if (len(rows) != 12532 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any()
            or not np.isfinite(rows[list(METRICS) + ['radius', 'hours']]).all().all()):
        raise ValueError('complete score inventory')
    blocks = pd.concat([pd.read_csv(SOURCE / f'fold_{k}/test_month_metrics.csv') for k in range(6)], ignore_index=True)
    if (SOURCE / 'test_month_metrics.csv').read_text() != blocks.to_csv(index=False):
        raise ValueError('root producer serialization differs')
    regional, macro = aggregate(rows)
    for name, new, keys in [('region_metrics', regional, ['region', 'method', 'radius_policy', 'calibration']),
                            ('macro_metrics', macro, ['cohort', 'method', 'radius_policy', 'calibration'])]:
        old = pd.read_csv(SOURCE / f'{name}.csv')
        pair = new.merge(old, on=keys, validate='one_to_one', suffixes=('_a', '_b'))
        fields = list(METRICS) + (['worst_coverage', 'regions'] if name == 'macro_metrics' else [])
        if (len(pair) != len(old) or len(pair) != len(new) or not np.isfinite(old[fields]).all().all()
                or any(max(abs(pair[f + '_a'] - pair[f + '_b'])) > 1e-8 for f in fields)):
            raise ValueError('aggregate reconstruction failed')
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    with threadpool_limits(limits=2):
        checks = [audit_fold(data, monthly, fold) for fold in range(6)]
    tables = [regional]
    cohort_keys = set(rows[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None))
    for method, directory in external.items():
        frame = pd.read_csv(directory / 'test_month_metrics.csv')
        frame = frame.loc[frame.method.eq(method) & frame.budget.eq('asof_30d')]
        if 'radius_policy' in frame:
            frame = frame.loc[frame.radius_policy.eq('recalibrated')]
        if (set(frame[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None)) != cohort_keys
                or frame.duplicated(['region', 'month', 'calibration']).any()
                or not np.isfinite(frame[list(METRICS[:5])]).all().all()):
            raise ValueError('external reference cohort differs')
        tables.append(pd.DataFrame([{'region': key[0], 'method': method, 'calibration': key[1], 'radius_policy': 'recalibrated',
                                     **{m: float(np.average(group[m], weights=group.hours)) for m in METRICS[:5]}}
                                    for key, group in frame.groupby(['region', 'calibration'])]))
    table = pd.concat(tables, ignore_index=True)
    rng = np.random.default_rng(177)
    comparisons = []
    for rec in macro.loc[~macro.method.isin(REFERENCES)].itertuples(index=False):
        selected = table.loc[table.calibration.eq(rec.calibration)]
        if rec.cohort == 'nonconstant24':
            selected = selected.loc[~selected.region.isin(NONCONSTANT)]
        a = selected.loc[selected.method.eq(rec.method) & selected.radius_policy.eq(rec.radius_policy)].sort_values('region')
        reference, mode = SETTINGS[rec.method]
        prefix = 'conditional' if reference == REFERENCES[0] else 'gaussian'
        peers = [(name, 'recalibrated', 'strong_reference') for name in (*REFERENCES, *external)]
        if mode != 'expanded_original':
            peers.append((f'{prefix}_expanded_original', rec.radius_policy, 'matched_expanded_bank'))
        if mode == 'pair_sync':
            peers.extend((f'{prefix}_{peer}', rec.radius_policy, 'matched_pair_structure')
                         for peer in ('pair_permute1', 'pair_permute2', 'pair_mixture', 'selected_parent'))
        if mode == 'pair_mixture':
            peers.append((f'{prefix}_selected_parent', rec.radius_policy, 'matched_parent_context'))
        for peer, policy, comparison in peers:
            b = selected.loc[selected.method.eq(peer) & selected.radius_policy.eq(policy)].sort_values('region')
            if len(a) != rec.regions or not np.array_equal(a.region, b.region):
                raise ValueError('paired comparison inventory')
            difference = a.crps.to_numpy() - b.crps.to_numpy()
            boot = difference[rng.integers(len(a), size=(3000, len(a)))].mean(axis=1)
            comparisons.append({'cohort': rec.cohort, 'method': rec.method, 'radius_policy': rec.radius_policy,
                                'calibration': rec.calibration, 'comparison': comparison, 'reference': peer,
                                'reference_radius_policy': policy, 'paired_crps_difference': float(difference.mean()),
                                'region_bootstrap_lo': float(np.quantile(boot, .025)),
                                'region_bootstrap_hi': float(np.quantile(boot, .975)),
                                'regions_better_crps': int((difference < -1e-9).sum()),
                                'crps_reduction_percent': float(100 * (1 - rec.crps / b.crps.mean())),
                                'coverage': rec.coverage, 'worst_coverage': rec.worst_coverage, 'width': rec.width,
                                'width_change_percent': float(100 * (rec.width / b.width.mean() - 1)), 'winkler': rec.winkler,
                                'numeric_threshold_only': bool(rec.coverage >= .9 and rec.worst_coverage >= .85
                                                               and rec.crps <= .95 * b.crps.mean()
                                                               and rec.width <= b.width.mean() + 1e-9
                                                               and rec.winkler < b.winkler.mean())})
    if len(comparisons) != 1216:
        raise ValueError('comparison denominator')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit inputs changed during run')
    verify(SOURCE)
    OUT.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(OUT / 'reconstruction.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT / 'comparisons.csv', index=False)
    result = {'assertion_failures': 0, 'rebuilt_score_rows': sum(c['rebuilt_score_rows'] for c in checks),
              'query_details_rebuilt': sum(c['detail_rows_rebuilt'] for c in checks),
              'source_worlds_rebuilt': sum(c['source_worlds_rebuilt'] for c in checks),
              'independent_source_cells': sum(c['source_cells_independently_checked'] for c in checks),
              'predictive_physical_cells': sum(c['predictive_physical_cells'] for c in checks),
              'original_half_support_checks': sum(c['original_half_support_checks'] for c in checks),
              'future_month_mutation_cases': 12, 'max_score_error': max(c['max_score_error'] for c in checks),
              'max_independent_source_error': max(c['max_source_formula_error'] for c in checks),
              'max_fuel_moment_error': max(c['max_fuel_moment_error'] for c in checks),
              'comparisons': len(comparisons), 'numeric_threshold_passes': sum(c['numeric_threshold_only'] for c in comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Complete frozen-law/calibration/score reconstruction; independent source-index/fuel and query-weight/moment formulas checked. Shared scoring helpers; bootstrap only descriptive source-development evidence.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
