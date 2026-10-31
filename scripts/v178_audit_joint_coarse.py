"""Refit monthly world kernels and rebuild every fine/coarse evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from joint_coarse_worlds import JointCoarsePrediction, METHODS, REFERENCES, SETTINGS, RESPONSES, BACKOFF
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation, monthly_context
from v151_hourly_law import EF, sha
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v173_moment_protected_risk import METRICS, aggregate
from v178_joint_coarse_world import (BASE, OUT as SOURCE, ROOT, COARSE_METRICS, FUEL_METRICS,
                                     aggregate_coarse, check_mutation)

OUT = ROOT / 'results/v178-joint-coarse-audit-r1'
NONCONSTANT = ['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA']


def audit_training(predictor, block):
    models, monthly = predictor.month_models, predictor.monthly
    saved = json.loads((block / 'model_metadata.json').read_text())
    if models.metadata != saved:
        raise ValueError('coarse forest metadata/targets not reproduced')
    old_rows = pd.read_csv(block / 'training_worlds.csv')
    fresh = pd.DataFrame(models.source_rows())
    if len(old_rows) != len(fresh) or any(not np.array_equal(old_rows[c], fresh[c]) for c in fresh.columns):
        raise ValueError('monthly training source rows differ')
    for q, p in zip(models.ids, models.inputs):
        context = models.contexts[int(q)]
        if (monthly.index.region.iloc[q] not in models.proper or monthly.local(int(q), 'asof_30d') != p
                or not (monthly.release.iloc[context['prefix']] <= monthly.dates.iloc[q]).all()
                or int(q) in context['prefix']):
            raise ValueError('coarse training role/prefix leakage')
    count = {r: int((models.regions == r).sum()) for r in models.proper}
    weight = np.asarray([1 / count[r] for r in models.regions])
    delta = monthly.shares[models.ids] - monthly.shares[models.inputs]
    epsilon = 64 * np.finfo(float).eps
    delta[abs(delta) <= epsilon] = 0
    anchors = np.asarray([monthly.shares[q] @ EF for q in models.ids])
    previous = np.asarray([monthly.shares[p] @ EF for p in models.inputs])
    scalar = anchors - previous
    scalar[abs(scalar) <= epsilon * np.maximum.reduce([np.ones(len(scalar)), abs(anchors), abs(previous)])] = 0
    logratio = np.log1p(monthly.shares[models.ids] / predictor.source_B)
    raw = {'scalar': scalar[:, None] / 1000, 'delta': delta, 'logratio': logratio}
    normalized, max_error = {}, 0.
    for key, values in raw.items():
        mu = (weight[:, None] * values).sum(axis=0) / weight.sum()
        sd = np.maximum(np.sqrt((weight[:, None] * (values - mu)**2).sum(axis=0) / weight.sum()), 1e-3)
        normalized[key] = (values - mu) / sd
        meta = models.transforms[key]
        max_error = max(max_error, float(max(abs(mu - meta['mean']))), float(max(abs(sd - meta['scale']))))
    wanted = {'scalar8': np.repeat(normalized['scalar'], 8, axis=1),
              'delta8': normalized['delta'], 'logratio8': normalized['logratio']}
    for key, response in [('scalar', 'scalar_mmd80'), ('delta', 'joint_mmd80')]:
        values = normalized[key]
        distance = np.linalg.norm(values[:, None] - values[None], axis=-1)
        positive = distance[np.triu_indices(len(values), k=1)]
        positive = positive[positive > 0]
        bandwidth = float(np.median(positive)) if len(positive) else 1.
        stored = models.transforms[response]
        max_error = max(max_error, abs(bandwidth - stored['bandwidth']) / max(1., bandwidth))
        frequency = np.asarray(stored['frequency'])
        independent_frequency = np.random.default_rng(20261029).normal(size=frequency.shape) / stored['bandwidth']
        max_error = max(max_error, float(np.max(abs(frequency - independent_frequency) / np.maximum(1., abs(frequency)))))
        phase = values @ frequency
        wanted[response] = np.concatenate([np.cos(phase), np.sin(phase)], axis=1) / np.sqrt(40)
    for response in RESPONSES:
        max_error = max(max_error, float(np.max(abs(models.targets[response] - wanted[response]))))
    if max_error > 1e-10:
        raise ValueError('independent proper response transform differs')
    # Check every component of the shared fine geometry against allowed raw g.
    data = predictor.frozen.data
    fine_checks, fine_error = 0, 0.
    for j, (q, p) in enumerate(zip(models.ids, models.inputs)):
        region = data.index.region.iloc[q]
        source = predictor.frozen.activity.sources[region]
        pick = source['allowed'][source['profiles'] == q]
        if not len(pick):
            raise ValueError('coarse world outside sampled-hour bank')
        source_B = .99 * monthly.shares[p] + .01 * predictor.frozen.prior
        for h, (activity, component, within) in enumerate(predictor.geometry.cells):
            take = pick[data.states[pick] % 24 == h]
            if not len(take):
                take = pick
            expected = data.raw[take] / (monthly.M[p].sum() / data.index.hours.iloc[q]) / source_B
            mask = component == j
            error = float(np.max(abs(activity[mask] - expected) / np.maximum(1., abs(expected))))
            fine_error = max(fine_error, error)
            if error > 1e-12 or max(abs(within[mask] - 1 / len(take))) > 1e-14:
                raise ValueError('shared geometry used incorrect fine samples')
            fine_checks += 1
    return fine_checks, fine_error, max_error


def independent_support(predictor, profiles, B):
    monthly = predictor.monthly
    values = []
    for p in profiles:
        local = monthly.local(int(p), 'asof_30d')
        denominator = .99 * monthly.shares[local] + .01 * predictor.frozen.prior
        v = B * monthly.shares[int(p)] / denominator
        values.append(v / v.sum())
    return np.asarray(values)


def independent_auxiliary(predictor, q, details, forecast):
    """Broadcast norm/absolute-distance losses, not CDF scoring or scipy cdist."""
    monthly = predictor.monthly
    truth = monthly.shares[q]
    B = .99 * monthly.shares[forecast['local']] + .01 * predictor.frozen.prior
    distributions = [(d['method'], independent_support(predictor, json.loads(d['source_profiles']), B),
                       np.asarray(json.loads(d['weights']))) for d in details]
    distributions += [('coarse_persistence', monthly.shares[forecast['local']][None], np.ones(1)),
                      ('coarse_point_mix', forecast['shares'][None], np.ones(1))]
    cache, scores, fuel_scores = {}, {}, {}

    def project(v):
        result = np.zeros_like(v)
        for ids in ([3, 4, 5, 6], [1, 7]):
            result[..., ids] = v[..., ids] - np.mean(v[..., ids], axis=-1, keepdims=True)
        return result

    for method, values, w in distributions:
        key = values.tobytes()
        if key not in cache:
            differences = values[:, None] - values[None]
            p = project(values)
            cache[key] = (np.sqrt(np.sum(differences**2, axis=-1)),
                          np.linalg.norm(p[:, None] - p[None], axis=-1),
                          abs(differences), p)
        D, DP, absolute, p = cache[key]
        marginal = []
        for f in range(8):
            crps = float(w @ abs(values[:, f] - truth[f]) - .5 * w @ absolute[..., f] @ w)
            zero = float(w @ (values[:, f] == 0))
            fuel_scores[(method, f)] = {'crps': crps, 'predicted_zero': zero,
                                        'zero_brier': float((zero - (truth[f] == 0))**2)}
            marginal.append(crps)
        anchor_values = values @ EF
        anchor_truth = truth @ EF
        scores[method] = {'share_crps_mean': float(np.mean(marginal)),
                          'energy_score': float(w @ np.linalg.norm(values - truth, axis=1) - .5 * w @ D @ w),
                          'same_EF_energy_score': float(w @ np.linalg.norm(p - project(truth), axis=1) - .5 * w @ DP @ w),
                          'anchor_crps': float(w @ abs(anchor_values - anchor_truth)
                                               - .5 * w @ abs(anchor_values[:, None] - anchor_values[None]) @ w)}
    return scores, fuel_scores


def audit_fold(data, monthly, fold):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    predictor = JointCoarsePrediction(frozen, monthly)
    block = SOURCE / f'fold_{fold}'
    fine_checks, fine_error, transform_error = audit_training(predictor, block)
    models = predictor.month_models
    training_leaves = {key: m.apply(models.X['history' if key.startswith('history_') else 'base'])
                       for key, m in models.models.items()}
    counts = {r: int((models.regions == r).sum()) for r in models.proper}
    raw_weight = np.asarray([1 / counts[r] for r in models.regions])
    rows = pd.read_csv(block / 'test_month_metrics.csv').set_index(['profile_id', 'method', 'radius_policy', 'calibration'])
    contexts = pd.read_csv(block / 'contexts.csv').set_index('profile_id')
    saved = pd.read_csv(block / 'world_probabilities.csv').set_index(['profile_id', 'method'])
    radii = pd.read_csv(block / 'calibration_radii.csv').set_index(['method', 'calibration'])
    coarse = pd.read_csv(block / 'coarse_month_metrics.csv').set_index(['profile_id', 'method'])
    fuels = pd.read_csv(block / 'coarse_fuel_metrics.csv').set_index(['profile_id', 'method', 'fuel_index'])
    if (len(saved) != len(frozen.queries) * 14 or saved.index.duplicated().any()
            or len(radii) != 28 or radii.index.duplicated().any() or not np.isfinite(radii.radius).all()
            or set(contexts.index) != set(frozen.queries.profile_id) or contexts.index.duplicated().any()
            or coarse.index.duplicated().any() or fuels.index.duplicated().any()
            or len(coarse) != int(frozen.queries.role.eq('test').sum()) * 16
            or len(fuels) != int(frozen.queries.role.eq('test').sum()) * 128):
        raise ValueError('complete query/coarse diagnostic inventory')
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    errors = {m: {} for m in METHODS}
    checked = details_checked = physical = coarse_checked = fuel_checked = kernel_checked = 0
    max_score = max_radius = max_diagnostic = max_coarse = max_independent = max_kernel = 0.
    for role in ('calibration', 'test'):
        for rec in frozen.queries.loc[frozen.queries.role.eq(role)].itertuples(index=False):
            q = int(rec.profile_id)
            laws, forecast, details = predictor.build(q)
            old_context = contexts.loc[q]
            if (old_context.role, old_context.region, old_context.month, old_context.fold) != (role, rec.region, rec.month, fold):
                raise ValueError('context identity')
            for key in ('raw_anchor', 'anchor', 'past_anchor', 'sigma', 'local'):
                difference = abs(float(forecast[key]) - float(old_context[key]))
                if not np.isfinite(difference):
                    raise ValueError('nonfinite context')
                max_diagnostic = max(max_diagnostic, difference)
            context = monthly_context(monthly, q)
            S = monthly.shares[context['local']]
            B = .99 * S + .01 * frozen.prior
            # Independent empirical leaf mixtures, not the production leaf maps.
            leaves = {}
            for key, model in models.models.items():
                feature = 'history' if key.startswith('history_') else 'base'
                signature = model.apply(context[feature][None])[0]
                p = np.zeros(len(models.ids))
                for t, node in enumerate(signature):
                    mask = training_leaves[key][:, t] == node
                    p[mask] += raw_weight[mask] / raw_weight[mask].sum() / 32
                leaves[key] = p
                response = key[len(feature) + 1:]
                value = p @ models.targets[response]
                max_kernel = max(max_kernel, float(max(abs(value - model.predict(context[feature][None])[0]))))
                kernel_checked += 1
            source_prior = np.asarray([1 / len(frozen.fit) / len(frozen.bank[monthly.index.region.iloc[p]]) for p in models.ids])
            Hlog = np.log(source_prior) - .5 * ((np.sqrt(monthly.shares[models.inputs]) - np.sqrt(S))**2).sum(axis=1) / .09
            pH = np.exp(np.maximum(Hlog - max(Hlog), -700)); pH /= pH.sum()
            coarse_values = independent_support(predictor, models.ids, B)
            Glog = Hlog - .5 * ((coarse_values @ EF - forecast['anchor']) / forecast['sigma'])**2
            pG = np.exp(np.maximum(Glog - max(Glog), -700)); pG /= pG.sum()
            for detail in details:
                method = detail['method']
                old = saved.loc[(q, method)]
                if (old.role, old.region, old.month, old.fold) != (role, rec.region, rec.month, fold):
                    raise ValueError('world probability identity')
                ids = json.loads(detail['source_profiles'])
                if ids != json.loads(old.source_profiles):
                    raise ValueError('world identifiers differ')
                for key in ('weights', 'leaf_weights', 'coarse_mean'):
                    a, b = np.asarray(json.loads(detail[key])), np.asarray(json.loads(old[key]))
                    if a.shape != b.shape or not np.isfinite(b).all():
                        raise ValueError('world probability dimensions/finiteness')
                    if a.size:
                        max_diagnostic = max(max_diagnostic, float(np.max(abs(a - b))))
                for key in ('coarse_anchor_mean', 'effective_worlds'):
                    if not np.isfinite(old[key]):
                        raise ValueError('nonfinite coarse diagnostic')
                    max_diagnostic = max(max_diagnostic, abs(detail[key] - old[key]))
                w = np.asarray(json.loads(old.weights))
                if (w <= 0).any() or abs(w.sum() - 1) > 1e-12:
                    raise ValueError('positive probability support')
                if method in SETTINGS:
                    _, kind, key = SETTINGS[method]
                    expected = ((1 - BACKOFF) * leaves[key] + BACKOFF * pH) if kind == 'forest' else (pG if key == 'gaussian' else pH)
                    if ids != models.ids.tolist() or max(abs(w - expected)) > 1e-12:
                        raise ValueError('independent coarse world kernel differs')
                    if kind == 'forest' and max(abs(np.asarray(json.loads(old.leaf_weights)) - leaves[key])) > 1e-12:
                        raise ValueError('independent leaf probabilities differ')
                    for a, b in zip(laws[method], laws['allmonth_hellinger']):
                        if (not np.array_equal(a.x, b.x) or not np.isfinite(a.w).all() or not (a.w > 0).all()
                                or min(a.x) < -1e-10 or max(a.x) > 1000 + 1e-10):
                            raise ValueError('shared physical support changed')
                        physical += 1
                supported = independent_support(predictor, ids, B)
                if (max(abs(w @ supported - np.asarray(json.loads(old.coarse_mean)))) > 1e-12
                        or abs(w @ (supported @ EF) - old.coarse_anchor_mean) > 1e-8
                        or abs(1 / (w @ w) - old.effective_worlds) > 1e-8):
                    raise ValueError('independent coarse moments differ')
                details_checked += 1
            if role == 'test':
                new_coarse, new_fuels = predictor.score_coarse(q, details, forecast)
                independent, independent_fuel = independent_auxiliary(predictor, q, details, forecast)
                for record in new_coarse:
                    old = coarse.loc[(q, record['method'])]
                    if (old.role, old.region, old.month, old.fold) != (role, rec.region, rec.month, fold):
                        raise ValueError('coarse score identity')
                    for key in COARSE_METRICS:
                        if not np.isfinite(old[key]):
                            raise ValueError('nonfinite coarse score')
                        max_coarse = max(max_coarse, abs(record[key] - old[key]))
                    for key in ('current_hour_count_matches_calendar', 'input_hour_count_matches_calendar'):
                        if old[key] != record[key]:
                            raise ValueError('coarse quality metadata differs')
                    for key, value in independent[record['method']].items():
                        max_independent = max(max_independent, abs(value - old[key]))
                    coarse_checked += 1
                for record in new_fuels:
                    old = fuels.loc[(q, record['method'], record['fuel_index'])]
                    if (old.role, old.region, old.month, old.fold) != (role, rec.region, rec.month, fold):
                        raise ValueError('fuel score identity')
                    for key in FUEL_METRICS:
                        if not np.isfinite(old[key]):
                            raise ValueError('nonfinite fuel score')
                        max_coarse = max(max_coarse, abs(float(record[key]) - float(old[key])))
                    for key, value in independent_fuel[(record['method'], record['fuel_index'])].items():
                        max_independent = max(max_independent, abs(value - old[key]))
                    fuel_checked += 1
            actual = data.truth(q)
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)] % 24
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
                                raise ValueError('fine score identity')
                            new = score_laws(laws[method], actual, clock, radius)
                            difference = np.asarray([new[k] - old[k] for k in new])
                            if not np.isfinite(difference).all() or not np.isfinite(old.radius):
                                raise ValueError('nonfinite fine score/radius')
                            max_score = max(max_score, float(max(abs(difference))))
                            max_radius = max(max_radius, abs(radius - old.radius))
                            checked += 1
        if role == 'calibration':
            radius_map = {}
            for method in METHODS:
                if set(errors[method]) != set(frozen.queries.loc[frozen.queries.role.eq(role), 'region']):
                    raise ValueError('calibration region inventory')
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[method])):
                    radius_map[(method, label)] = radius
                    max_radius = max(max_radius, abs(radius - radii.loc[(method, label), 'radius']))
    if (checked != len(rows) or details_checked != len(saved) or coarse_checked != len(coarse) or fuel_checked != len(fuels)
            or max_score > 1e-7 or max_radius > 1e-8 or max_diagnostic > 1e-8
            or max_coarse > 1e-8 or max_independent > 1e-8 or max_kernel > 1e-12):
        raise ValueError('complete coarse/fine reconstruction failed')
    print(f'joint coarse audit fold {fold}: {checked} CIF/{coarse_checked} coarse/{fuel_checked} fuel scores', flush=True)
    return {'fold': fold, 'rebuilt_score_rows': checked, 'world_records_rebuilt': details_checked,
            'coarse_rows_rebuilt': coarse_checked, 'fuel_rows_rebuilt': fuel_checked,
            'monthly_forests_refitted': 10, 'independent_fine_source_cells': fine_checks,
            'max_source_geometry_error': fine_error, 'max_response_transform_error': transform_error,
            'independent_kernel_checks': kernel_checked, 'max_kernel_error': max_kernel,
            'physical_support_checks': physical, 'max_score_error': max_score,
            'max_radius_error': max_radius, 'max_diagnostic_error': max_diagnostic,
            'max_coarse_rebuild_error': max_coarse, 'max_independent_auxiliary_error': max_independent}


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
                'gaussian_crps_global_a50': ROOT / 'results/v175-cif-crps-map-r1',
                'gaussian_selected_parent': ROOT / 'results/v177-portfolio-value-r1'}
    for directory in set(external.values()):
        verify(directory)
        for name in ('summary.json', 'test_month_metrics.csv'):
            hashes[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input changed')
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    if (len(rows) != 12532 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any()
            or not np.isfinite(rows[list(METRICS) + ['radius', 'hours']]).all().all()):
        raise ValueError('complete fine inventory')
    fixed = rows.loc[rows.radius_policy.eq('fixed')]
    gaussian = rows.loc[rows.method.eq('world_gaussian'), ['profile_id', 'calibration', 'radius']]
    match = fixed.merge(gaussian, on=['profile_id', 'calibration'], validate='many_to_one', suffixes=('_new', '_old'))
    if len(match) != 5784 or max(abs(match.radius_new - match.radius_old)) > 1e-8:
        raise ValueError('common fixed Gaussian radius not matched')
    coarse = pd.read_csv(SOURCE / 'coarse_month_metrics.csv')
    fuels = pd.read_csv(SOURCE / 'coarse_fuel_metrics.csv')
    for name, frame, count, keys in [('test_month_metrics', rows, 12532, ['profile_id', 'method', 'radius_policy', 'calibration']),
                                     ('coarse_month_metrics', coarse, 3856, ['profile_id', 'method']),
                                     ('coarse_fuel_metrics', fuels, 30848, ['profile_id', 'method', 'fuel_index'])]:
        blocks = pd.concat([pd.read_csv(SOURCE / f'fold_{k}/{name}.csv') for k in range(6)], ignore_index=True)
        if len(frame) != count or frame.duplicated(keys).any() or (SOURCE / f'{name}.csv').read_text() != blocks.to_csv(index=False):
            raise ValueError('root producer serialization or inventory differs')
    regional, macro = aggregate(rows)
    cr, cm, fr, fm = aggregate_coarse(coarse, fuels)
    rebuilds = [('region_metrics', regional, ['region', 'method', 'radius_policy', 'calibration'], list(METRICS)),
                ('macro_metrics', macro, ['cohort', 'method', 'radius_policy', 'calibration'], list(METRICS) + ['regions', 'worst_coverage']),
                ('coarse_region_metrics', cr, ['region', 'method'], list(COARSE_METRICS)),
                ('coarse_macro_metrics', cm, ['cohort', 'method'], list(COARSE_METRICS) + ['regions', 'worst_anchor_coverage']),
                ('coarse_fuel_region_metrics', fr, ['region', 'method', 'fuel_index'], list(FUEL_METRICS)),
                ('coarse_fuel_macro_metrics', fm, ['cohort', 'method', 'fuel_index'], list(FUEL_METRICS) + ['regions'])]
    for name, new, keys, metrics in rebuilds:
        old = pd.read_csv(SOURCE / f'{name}.csv')
        pair = new.merge(old, on=keys, validate='one_to_one', suffixes=('_a', '_b'))
        if (len(pair) != len(old) or len(pair) != len(new) or not np.isfinite(old[metrics]).all().all()
                or any(max(abs(pair[m + '_a'] - pair[m + '_b'])) > 1e-8 for m in metrics)):
            raise ValueError('aggregate reconstruction differs')
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
            raise ValueError('external strong reference cohort mismatch')
        tables.append(pd.DataFrame([{'region': key[0], 'method': method, 'calibration': key[1], 'radius_policy': 'recalibrated',
                                     **{m: float(np.average(group[m], weights=group.hours)) for m in METRICS[:5]}}
                                    for key, group in frame.groupby(['region', 'calibration'])]))
    table = pd.concat(tables, ignore_index=True)
    rng = np.random.default_rng(178)
    comparisons = []
    for rec in macro.loc[~macro.method.isin(REFERENCES)].itertuples(index=False):
        selected = table.loc[table.calibration.eq(rec.calibration)]
        if rec.cohort == 'nonconstant24':
            selected = selected.loc[~selected.region.isin(NONCONSTANT)]
        a = selected.loc[selected.method.eq(rec.method) & selected.radius_policy.eq(rec.radius_policy)].sort_values('region')
        _, kind, key = SETTINGS[rec.method]
        peers = [(m, 'recalibrated', 'strong_reference') for m in (*REFERENCES, *external)]
        if kind == 'forest':
            feature, response = key.split('_', 1)
            peers.extend((m, rec.radius_policy, 'matched_full_bank') for m in ('allmonth_hellinger', 'allmonth_gaussian'))
            if response in ('delta8', 'logratio8'):
                peers.append((f'forest_{feature}_scalar8', rec.radius_policy, 'matched_response8'))
            if response == 'joint_mmd80':
                peers.append((f'forest_{feature}_scalar_mmd80', rec.radius_policy, 'matched_response80'))
                peers.append((f'forest_{feature}_delta8', rec.radius_policy, 'matched_joint_embedding'))
            if response == 'scalar_mmd80':
                peers.append((f'forest_{feature}_scalar8', rec.radius_policy, 'matched_scalar_embedding'))
            if response == 'logratio8':
                peers.append((f'forest_{feature}_delta8', rec.radius_policy, 'matched_coordinate'))
            if feature == 'history':
                peers.append((f'forest_base_{response}', rec.radius_policy, 'matched_history'))
        for peer, policy, comparison in peers:
            b = selected.loc[selected.method.eq(peer) & selected.radius_policy.eq(policy)].sort_values('region')
            if len(a) != rec.regions or not np.array_equal(a.region, b.region):
                raise ValueError('paired comparison regions differ')
            delta = a.crps.to_numpy() - b.crps.to_numpy()
            boot = delta[rng.integers(len(a), size=(3000, len(a)))].mean(axis=1)
            comparisons.append({'cohort': rec.cohort, 'method': rec.method, 'radius_policy': rec.radius_policy,
                                'calibration': rec.calibration, 'comparison': comparison, 'reference': peer,
                                'reference_radius_policy': policy, 'paired_crps_difference': float(delta.mean()),
                                'region_bootstrap_lo': float(np.quantile(boot, .025)),
                                'region_bootstrap_hi': float(np.quantile(boot, .975)),
                                'regions_better_crps': int((delta < -1e-9).sum()),
                                'crps_reduction_percent': float(100 * (1 - rec.crps / b.crps.mean())),
                                'coverage': rec.coverage, 'worst_coverage': rec.worst_coverage,
                                'width': rec.width, 'width_change_percent': float(100 * (rec.width / b.width.mean() - 1)),
                                'winkler': rec.winkler,
                                'numeric_threshold_only': bool(rec.coverage >= .9 and rec.worst_coverage >= .85
                                                               and rec.crps <= .95 * b.crps.mean()
                                                               and rec.width <= b.width.mean() + 1e-9
                                                               and rec.winkler < b.winkler.mean())})
    if len(comparisons) != 1448:
        raise ValueError('comparison denominator')
    coarse_pairs = {(c['cohort'], c['method'], c['reference']) for c in comparisons if c['reference'] in METHODS}
    coarse_pairs |= {(cohort, method, reference) for cohort in ('all29', 'nonconstant24') for method in METHODS
                     for reference in ('coarse_persistence', 'coarse_point_mix')}
    coarse_comparisons = []
    for cohort, method, reference in sorted(coarse_pairs):
        selected = cr if cohort == 'all29' else cr.loc[~cr.region.isin(NONCONSTANT)]
        a, b = [selected.loc[selected.method.eq(m)].sort_values('region') for m in (method, reference)]
        if not np.array_equal(a.region, b.region):
            raise ValueError('coarse paired regions differ')
        coarse_comparisons.append({'cohort': cohort, 'method': method, 'reference': reference,
                                   **{f'{m}_difference': float((a[m].to_numpy() - b[m].to_numpy()).mean()) for m in COARSE_METRICS}})
    if len(coarse_comparisons) != 178:
        raise ValueError('coarse comparison denominator')
    quality = []
    for label, mask in [('both_hour_counts_match', coarse.current_hour_count_matches_calendar & coarse.input_hour_count_matches_calendar),
                        ('other_hour_counts', ~(coarse.current_hour_count_matches_calendar & coarse.input_hour_count_matches_calendar))]:
        subset = coarse.loc[mask]
        local = subset.groupby(['region', 'method'])[list(COARSE_METRICS)].mean().reset_index()
        for method, group in local.groupby('method'):
            quality.append({'quality': label, 'method': method, 'regions': len(group),
                            'query_months': int(subset.loc[subset.method.eq(method)].shape[0]),
                            **{m: float(group[m].mean()) for m in COARSE_METRICS}})
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit inputs changed during run')
    verify(SOURCE)
    OUT.mkdir(parents=True)
    for name, frame in [('reconstruction', pd.DataFrame(checks)), ('comparisons', pd.DataFrame(comparisons)),
                        ('coarse_comparisons', pd.DataFrame(coarse_comparisons)), ('coarse_quality_strata', pd.DataFrame(quality))]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'rebuilt_score_rows': sum(c['rebuilt_score_rows'] for c in checks),
              'world_records_rebuilt': sum(c['world_records_rebuilt'] for c in checks),
              'coarse_rows_rebuilt': sum(c['coarse_rows_rebuilt'] for c in checks),
              'fuel_rows_rebuilt': sum(c['fuel_rows_rebuilt'] for c in checks),
              'monthly_forests_refitted': 60, 'independent_kernel_checks': sum(c['independent_kernel_checks'] for c in checks),
              'independent_fine_source_cells': sum(c['independent_fine_source_cells'] for c in checks),
              'physical_support_checks': sum(c['physical_support_checks'] for c in checks),
              'common_fixed_radius_rows': len(match), 'future_month_mutation_cases': 12,
              'max_score_error': max(c['max_score_error'] for c in checks),
              'max_independent_auxiliary_error': max(c['max_independent_auxiliary_error'] for c in checks),
              'comparisons': len(comparisons), 'coarse_comparisons': len(coarse_comparisons),
              'numeric_threshold_passes': sum(c['numeric_threshold_only'] for c in comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Complete frozen coarse-model/fine-law/calibration/score reconstruction. Independent response transforms, raw-source geometry, leaf mixtures and auxiliary loss formulas checked; shared quantile/fine-scoring helpers. Source-development comparisons only.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
