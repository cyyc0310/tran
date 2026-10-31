"""Source-only fixed-phase headroom, keeping every hourly law and radius unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fuel_activity_forest import ActivityDonors, carrier, source_activity
from hourly_fuel_data import HourlyFuelData
from phase_loss_tables import choose_phase, phase_scores, signed_phase
from v151_hourly_law import sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v162-fixed-phase-diagnostic-r1'
PROTOCOL = ROOT / 'docs/V162_FIXED_PHASE_DIAGNOSTIC_PROTOCOL.md'
METHODS = ('positive_recomposition', 'conditional_activity_recomposition')
METRICS = ('crps', 'mae', 'coverage', 'width', 'winkler', 'calendar_width', 'shuffle_crps')


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame([dict(zip(keys, key if isinstance(key, tuple) else (key,)))
                         | {'months': len(group), 'hours': int(group.hours.sum())}
                         | {m: float(np.average(group[m], weights=group.hours)) for m in METRICS}
                         for key, group in frame.groupby(keys)])


def run_block(data, samples, fold: int, budget: str, block: Path, manifest_digest: str):
    old_block = BASE / f'fold_{fold}_{budget}'
    meta = json.loads((old_block / 'summary.json').read_text())
    fit = sorted(meta['source_regions'])
    test = set(meta['test_regions'])
    if test & set(fit):
        raise ValueError('source role overlap')
    draw = samples.loc[samples.budget.eq(budget)].set_index('region')
    allowed = {r: np.asarray(json.loads(draw.loc[r, 'sampled_raw_hour_indices']), int) for r in fit}
    proper = {r: data.source(r, budget, allowed[r]) for r in fit}
    prior = np.mean([data.shares[proper[r][3]].mean(axis=0) for r in fit], axis=0) + 1e-6
    prior /= prior.sum()
    if not np.array_equal(prior, np.asarray(meta['carrier_prior'])):
        raise ValueError('carrier prior changed')
    sources = {r: source_activity(data, r, budget, allowed[r], prior) for r in fit}
    shape_bank, activity_bank = ActivityDonors(data, sources, 'shape'), ActivityDonors(data, sources)
    bank = {r: np.unique(sources[r]['profiles']) for r in fit}
    month_number = pd.to_datetime(data.index.month).dt.month.to_numpy()
    queries = pd.read_csv(old_block / 'query_inputs.csv')
    queries = queries.loc[queries.role.eq('test')]
    radii = pd.read_csv(old_block / 'calibration_radii.csv')
    outputs = []
    for query in queries.itertuples(index=False):
        q = int(query.profile_id)
        row = data.index.iloc[q]
        local = data.local(q, budget)
        if row.region not in test or row.region != query.region or not row.month.startswith('2024'):
            raise ValueError('query source identity/year')
        if local != query.input_profile_id:
            raise ValueError('query input vintage changed')
        S = data.shares[local]
        donors = []
        for r in fit:
            ids = bank[r]
            distance = np.abs(month_number[ids] - month_number[q])
            distance = np.minimum(distance, 12 - distance)
            donors.extend(ids[np.lexsort((ids, distance))[:2]])
        donors = np.asarray(donors, int)
        if len(donors) != 2 * len(fit) or row.region in set(data.index.region.iloc[donors]):
            raise ValueError('donor denominator/role')
        distance = .5 * ((np.sqrt(data.shares[donors]) - np.sqrt(S))**2).sum(axis=1)
        weights = np.exp(-distance / .3**2)
        weights /= weights.sum()
        shape, _ = shape_bank.laws(donors, weights, S)
        laws = {'positive_recomposition': shape}
        if budget == 'same_month':
            laws['conditional_activity_recomposition'] = shape
        else:
            input_ids = np.asarray([data.local(int(d), budget) for d in donors], int)
            distance = .5 * ((np.sqrt(data.shares[input_ids]) - np.sqrt(S))**2).sum(axis=1)
            weights = np.exp(-distance / .3**2)
            weights /= weights.sum()
            dynamic, _ = activity_bank.laws(donors, weights, carrier(S, budget, prior))
            laws['conditional_activity_recomposition'] = dynamic
        actual = data.truth(q)
        clock = data.states[int(row.offset):int(row.stop)] % 24
        for method in METHODS:
            for calibration in ('pooled', 'max_region'):
                selected = radii.loc[radii.method.eq(method) & radii.calibration.eq(calibration), 'radius']
                if len(selected) != 1:
                    raise ValueError('calibration radius identity')
                scores = phase_scores(laws[method], actual, clock, float(selected.iloc[0]))
                for phase in range(24):
                    outputs.append({'fold': fold, 'region': row.region, 'month': row.month,
                                    'profile_id': q, 'budget': budget, 'method': method,
                                    'calibration': calibration, 'phase': phase, 'signed_phase': signed_phase(phase),
                                    'hours': len(actual)} | {m: float(values[phase]) for m, values in scores.items()})
    rows = pd.DataFrame(outputs)
    if len(rows) != len(queries) * 2 * 2 * 24:
        raise ValueError('phase grid denominator')
    old = pd.read_csv(old_block / 'test_month_metrics.csv')
    old = old.loc[old.method.isin(METHODS)]
    keys = ['region', 'month', 'method', 'calibration']
    match = rows.loc[rows.phase.eq(0)].merge(old, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    if len(match) != len(queries) * 4:
        raise ValueError('phase zero denominator')
    errors = {m: float(np.max(np.abs(match[m + '_new'] - match[m + '_old'])))
              for m in ('crps', 'mae', 'coverage', 'width', 'winkler')}
    if max(errors.values()) > 1e-7:
        raise ValueError('phase zero failed to reproduce V159')
    for _, group in rows.groupby(['profile_id', 'method', 'calibration']):
        if np.ptp(group.calendar_width) > 1e-10:
            raise ValueError('calendar width invariance')
    staging = Path(tempfile.mkdtemp(prefix='.v162-block-', dir=OUT.parent))
    rows.to_csv(staging / 'phase_month_metrics.csv', index=False)
    queries.to_csv(staging / 'query_inputs.csv', index=False)
    summary = {'fold': fold, 'budget': budget, 'assertion_failures': 0,
               'phase_zero_records': len(match), 'phase_zero_max_errors': errors,
               'run_manifest_sha256': manifest_digest,
               'output_hashes': {p.name: sha(p) for p in sorted(staging.iterdir())}}
    (staging / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    staging.rename(block)
    print(f'completed phase fold {fold}/{budget}: {len(rows)} grid records, {len(match)} reproduced', flush=True)


def select_cases(rows: pd.DataFrame):
    region_phase = aggregate(rows, ['region', 'budget', 'method', 'calibration', 'phase'])
    selection = region_phase.loc[region_phase.calibration.eq('pooled')]
    policies = []
    for (region, budget, method), group in selection.groupby(['region', 'budget', 'method']):
        phase = choose_phase(group.phase.to_numpy(), group.crps.to_numpy())
        policies.append({'region': region, 'budget': budget, 'method': method,
                         'case': 'region_budget_method_oracle', 'phase': phase})
    shared = selection.loc[selection.method.eq('conditional_activity_recomposition')].groupby(['region', 'phase']).crps.mean().reset_index()
    for region, group in shared.groupby('region'):
        phase = choose_phase(group.phase.to_numpy(), group.crps.to_numpy())
        for budget in ('same_month', 'asof_30d'):
            for method in METHODS:
                policies.append({'region': region, 'budget': budget, 'method': method,
                                 'case': 'shared_region_oracle', 'phase': phase})
    month = pd.to_datetime(rows.month).dt.month
    first_half = rows.loc[(month <= 6) & rows.calibration.eq('pooled')]
    for region, budget, method in selection[['region', 'budget', 'method']].drop_duplicates().itertuples(index=False, name=None):
        group = first_half.loc[first_half.region.eq(region) & first_half.budget.eq(budget) & first_half.method.eq(method)]
        score = aggregate(group, ['phase']) if len(group) else None
        phase = choose_phase(score.phase.to_numpy(), score.crps.to_numpy()) if score is not None else 0
        policies.append({'region': region, 'budget': budget, 'method': method,
                         'case': 'H1_phase_H2', 'phase': phase,
                         'h1_months': int(group.month.nunique()), 'no_h1_fallback': len(group) == 0})
    policy_frame = pd.DataFrame(policies)
    selected = rows.merge(policy_frame, on=['region', 'budget', 'method', 'phase'], validate='many_to_many')
    keep = ~selected.case.eq('H1_phase_H2') | (pd.to_datetime(selected.month).dt.month >= 7)
    selected = selected.loc[keep].copy()
    unshifted = rows.loc[rows.phase.eq(0)].copy()
    unshifted['case'] = 'unshifted'
    second_half = unshifted.loc[pd.to_datetime(unshifted.month).dt.month >= 7].copy()
    second_half['case'] = 'H2_unshifted'
    selected = pd.concat([selected, unshifted, second_half], ignore_index=True)
    regional = aggregate(selected, ['case', 'region', 'budget', 'method', 'calibration'])
    macro = regional.groupby(['case', 'budget', 'method', 'calibration']).agg(
        crps=('crps', 'mean'), mae=('mae', 'mean'), coverage=('coverage', 'mean'),
        worst_coverage=('coverage', 'min'), width=('width', 'mean'), winkler=('winkler', 'mean'),
        calendar_width=('calendar_width', 'mean'), shuffle_crps=('shuffle_crps', 'mean'),
        regions=('region', 'size')).reset_index()
    upper = region_phase.groupby(['region', 'budget', 'method', 'calibration']).coverage.max().reset_index(name='maximum_fixed_phase_coverage')
    return region_phase, policy_frame, selected, regional, macro, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    upstream = json.loads((BASE / 'summary.json').read_text())
    if upstream['assertion_failures'] or upstream['completed_blocks'] != 12:
        raise ValueError('incomplete V159 archive')
    for name, digest in upstream['output_hashes'].items():
        if sha(BASE / name) != digest:
            raise ValueError('V159 output changed')
    hashes = dict(upstream['input_hashes'])
    for p in (Path(__file__), PROTOCOL, ROOT / 'scripts/phase_loss_tables.py',
              ROOT / 'scripts/fuel_activity_forest.py', ROOT / 'scripts/hourly_fuel_data.py', BASE / 'summary.json'):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    for name, digest in hashes.items():
        if sha(ROOT / name) != digest:
            raise ValueError('frozen dependency changed')
    manifest = json.dumps({'input_hashes': hashes, 'methods': list(METHODS), 'phase_count': 24},
                          indent=2, sort_keys=True, allow_nan=False) + '\n'
    if OUT.exists():
        if (OUT / 'run_manifest.json').read_text() != manifest:
            raise ValueError('resume manifest differs')
        if (OUT / 'summary.json').exists():
            raise RuntimeError('phase diagnostic already complete')
    else:
        OUT.mkdir(parents=True)
        (OUT / 'run_manifest.json').write_text(manifest)
    digest = sha(OUT / 'run_manifest.json')
    data = HourlyFuelData(ROOT)
    samples = pd.read_csv(BASE / 'source_sampling.csv')
    blocks = []
    with threadpool_limits(limits=2):
        for fold in range(6):
            for budget in ('same_month', 'asof_30d'):
                block = OUT / f'fold_{fold}_{budget}'
                if block.exists():
                    saved = json.loads((block / 'summary.json').read_text())
                    if saved['run_manifest_sha256'] != digest or saved['assertion_failures']:
                        raise ValueError('phase checkpoint identity')
                    for name, value in saved['output_hashes'].items():
                        if sha(block / name) != value:
                            raise ValueError('phase checkpoint changed')
                else:
                    run_block(data, samples, fold, budget, block, digest)
                blocks.append(block)
    rows = pd.concat([pd.read_csv(b / 'phase_month_metrics.csv') for b in blocks], ignore_index=True)
    keys = ['region', 'budget', 'month', 'method', 'calibration', 'phase']
    if len(rows) != 54528 or rows.duplicated(keys).any():
        raise ValueError('complete phase grid denominator')
    region_phase, policies, selected, regional, macro, upper = select_cases(rows)
    if any(sha(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError('frozen phase inputs changed')
    for name, frame in [('phase_month_metrics', rows), ('region_phase_metrics', region_phase),
                        ('phase_policies', policies), ('selected_month_metrics', selected),
                        ('region_metrics', regional), ('macro_metrics', macro), ('coverage_upper_limits', upper)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'status': 'SOURCE_ONLY_ORACLE_PHASE_DIAGNOSTIC', 'assertion_failures': 0,
              'completed_blocks': 12, 'phase_grid_rows': len(rows), 'phase_zero_records_reproduced': int(rows.phase.eq(0).sum()),
              'input_hashes': hashes,
              'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              'limitations': ['Label-informed fixed phase is diagnostic, not a deployable zero-history predictor.',
                              'H1 phase uses local hourly labels; H2 only checks temporal stability.',
                              'Uniform calendar width is invariant; valid-hour scored width may change.',
                              '2024 shared-time source development, fixed-EF production proxy and simulated release dates.']}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.calibration.eq('pooled')].to_string(index=False))


if __name__ == '__main__':
    main()
