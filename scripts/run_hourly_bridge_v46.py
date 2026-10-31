"""Zero-fit monthly-level bridge, with prediction sealing before scoped hourly scoring."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import hourly_bridge_v46 as model
from scripts import run_coarse_context_v30 as common
from scripts.run_local_sketch_v17 import source_days, METRICS as OLD_METRICS
from scripts.run_information_value_v10 import score_arrays
from scripts.native_center_v18 import scaled_paths

sha, read, write, now = common.sha, common.read, common.write, common.now
SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
DATES = pd.date_range('2025-01-01', periods=6, freq='MS', tz='UTC')
DAYS = pd.date_range('2025-01-01', '2025-06-30', freq='D', tz='UTC')
FACTORS = (.75, 1., 1.25, 1.5, 2.)
V29_ARMS = ('L24-ET-plugin_recompose', 'L24-ET-residual_anchor', 'L24-ET-residual_recompose',
            'Leaf-anchor', 'Leaf-recompose', 'Gaussian-anchor', 'Gaussian-recompose',
            'Scale-anchor', 'Scale-recompose', 'Scale-shuffle_recompose')
METRICS = (*OLD_METRICS, 'empirical_path_coverage90', 'final_spread_clipping_rate')
COPY_METRICS = (*OLD_METRICS, 'final_spread_clipping_rate')
OUT = ROOT / 'results/hourly-bridge-v46'
PARENT = ROOT / 'docs/evidence/gain-direction-v44-verification.json'
PARENT_SHA = '5d91faa3d8a25a94f8b6100bd1bd7fc462edb2c52254216b336e70f596f2f8fd'
CODE = ('scripts/hourly_bridge_v46.py', 'scripts/run_hourly_bridge_v46.py',
        'tests/test_hourly_bridge_v46.py', 'tests/test_run_hourly_bridge_v46.py',
        'docs/HOURLY_BRIDGE_V46_PROTOCOL.md')
HELPERS = ('run_coarse_context_v30', 'coarse_context_v30', 'run_local_sketch_v17',
           'local_sketch_v17', 'run_information_value_v10', 'native_center_v18')
REVIEWS = tuple(ROOT / f'results/hourly-bridge-v46-{kind}-review.md' for kind in ('code', 'math'))
OUTPUTS = ('day_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv',
           'leave_one_source_out_macro.csv', 'label_support.csv', 'summary.json')


def verify(mapping: dict) -> None:
    for path in mapping:
        if not Path(path).is_file() or Path(path).is_symlink():
            raise ValueError('Regular bound file required: ' + path)
    common.verify(mapping)


def inherited() -> dict:
    """Resolve manifest metadata; do not rehash historical data outside this stage."""
    verify({str(PARENT): PARENT_SHA})
    cursor = read(PARENT)
    if cursor['status'] != 'VERIFICATION_COMPLETE' or cursor['independent_formula_review_status'] != 'PASS':
        raise ValueError('Independently verified V44 required')
    result, seen = {}, set()
    while True:
        for kind in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
            for path, digest in cursor[kind].items():
                if path in result and result[path] != digest:
                    raise ValueError('Conflicting inherited metadata')
                result[path] = digest
        if 'parent_manifest' not in cursor:
            break
        path, digest = cursor['parent_manifest'], cursor['parent_manifest_sha256']
        if path in seen:
            raise ValueError('Cyclic inherited metadata')
        seen.add(path)
        verify({path: digest})
        cursor = read(path)
    return result


def support_metadata(table: pd.DataFrame) -> list[str]:
    """Lock dates using availability columns, never metric values."""
    if set(table) != {'region', 'origin', 'arm', 'factor'}:
        raise ValueError('Availability metadata only required')
    table = table.copy()
    table['origin'] = pd.to_datetime(table.origin, utc=True, errors='raise')
    table = table[table.origin.isin(DATES)]
    if not table.region.isin(SOURCES).all() or table.duplicated().any():
        raise ValueError('Unique registered availability grid required')
    expected = [f'{s}/{at:%Y-%m-%d}' for s in SOURCES for at in DATES
                if at.month != 1 and (s != 'ISNE' or at.month in (2, 5))]
    actual = []
    expected_grid = {(a, f) for a in V29_ARMS for f in FACTORS}
    for (region, at), group in table.groupby(['region', 'origin'], sort=False):
        if len(group) != 50 or set(zip(group.arm, group.factor)) != expected_grid:
            raise ValueError('Exact ten-arm five-factor support required')
        actual.append(f'{region}/{at:%Y-%m-%d}')
    if set(actual) != set(expected) or len(actual) != 32:
        raise ValueError('Predeclared 32-day support required')
    return expected


def contract() -> dict:
    closure = inherited()
    prior_path = str(ROOT / 'results/conditional-law-v29/freeze.json')
    if prior_path not in closure:
        raise ValueError('Bound V29 freeze required')
    verify({prior_path: closure[prior_path]})
    old = read(prior_path)
    if old['sources'] != list(SOURCES) or old['arms'] != list(V29_ARMS):
        raise ValueError('Exact source/cache registry required')
    baseline = str(ROOT / 'results/conditional-law-v29/day_metrics.csv')
    archives = {s: dict(
        hourly=str(ROOT / f'results/conditional-law-v29/predictions/{s}.npz'),
        m8=old['monthly_prediction_paths'][s],
        v42=str(ROOT / f'results/donor-budget-v42/predictions/{s}.npz'),
        capacity=str(ROOT / f'results/fleet-information-v34/predictions/{s}.npz'),
        mix=str(ROOT / f'results/ordinary-controls-v43/predictions/{s}.npz'),
        route=str(ROOT / f'results/ordinary-controls-v43/predictions/{s}.json'),
        gain=str(ROOT / f'results/gain-direction-v44/predictions/{s}.npz'),
        truth=str(Path(old['source']) / f'{s}_hourly.csv')) for s in SOURCES}
    aliases = [prior_path, baseline, *[p for a in archives.values() for p in a.values()],
               *[str(ROOT / 'scripts' / (h + '.py')) for h in HELPERS]]
    if any(p not in closure for p in aliases):
        raise ValueError('All direct aliases must have inherited hashes')
    inputs = {p: closure[p] for p in aliases} | {str(PARENT): PARENT_SHA}
    for path in REVIEWS:
        if not path.is_file() or path.is_symlink() or not path.read_text().strip():
            raise ValueError('Both final prelaunch reports must exist')
        inputs[str(path)] = sha(path)
    verify(inputs)
    support = support_metadata(pd.read_csv(baseline, usecols=['region', 'origin', 'arm', 'factor']))
    return dict(version=46, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE},
        input_hashes=inputs, environment=common.environment(), archives=archives,
        baseline_path=baseline, score_keys=support, sources=list(SOURCES), origins=DATES.asi8.tolist(),
        arms=list(model.ARMS), level_arms=list(model.LEVEL_ARMS), factors=list(FACTORS),
        budgets=model.BUDGET_BY_ARM, new_fits=0, new_fine_label_keys=False,
        target_values_opened=False, unseen_values_opened=False,
        cached_hourly_predictions_authorized=True, authorized_source_truth_days=32,
        independent_confirmation=False, method_innovation_claim=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V46 attempt')
    frozen = contract()
    OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared V46: 42 prediction queries, 32 predeclared score days, zero fits', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('Frozen V46 contract changed')
    return frozen


def indices(origins: np.ndarray, daily: bool) -> np.ndarray:
    expected = DAYS if daily else DATES
    if not np.array_equal(origins, expected.asi8):
        raise ValueError('Exact registered UTC dates required')
    return np.array([expected.get_loc(at) for at in DATES], int)


def monthly(path: str, source: str, mg: np.ndarray) -> dict:
    with np.load(path, allow_pickle=False) as z:
        indices(z['origins'], False)
        if (z['keys'].tolist() != [f'{source}/{at:%Y-%m}' for at in DATES]
                or not np.array_equal(z['MG_draws'], mg)):
            raise ValueError('Literal monthly key/MG coupling required')
        return {k: z[k].copy() for k in ('arms', 'CIF_draws')}


def load_for_prediction(entry: dict, source: str) -> dict:
    with np.load(entry['hourly'], allow_pickle=False) as z:
        take = indices(z['origins'], True)
        if z['arms'].tolist() != list(V29_ARMS):
            raise ValueError('Exact V29 arms required')
        a, r = V29_ARMS.index('Scale-anchor'), V29_ARMS.index('Scale-recompose')
        anchor, fine = z['raw'][a, take].copy(), z['fine_draws'][a, take].copy()
        recompose, mean = z['raw'][r, take].copy(), z['query_mean'][take].copy()
        mg = z['MG_draws'][take].copy()
    with np.load(entry['m8'], allow_pickle=False) as z:
        take = indices(z['origins'], True)
        if not np.array_equal(z['MG_draws'][take], mg):
            raise ValueError('Literal V25 MG coupling required')
        names = z['arms'].tolist()
        if names.count('M8-recompose_conditional') != 1:
            raise ValueError('Unique M8 reference required')
        m8 = z['raw'][names.index('M8-recompose_conditional'), take].copy()
    v42 = monthly(entry['v42'], source, mg)
    if v42['arms'].tolist() != list(model.LEVEL_ARMS[:12]):
        raise ValueError('Exact twelve V42 endpoints required')
    with np.load(entry['capacity'], allow_pickle=False) as z:
        indices(z['origins'], False)
        if (z['keys'].tolist() != [f'{source}/{at:%Y-%m}' for at in DATES]
                or not np.array_equal(z['MG_draws'], mg)):
            raise ValueError('Literal capacity query coupling required')
        names = z['arms'].tolist()
        if names.count('Capacity-OP') != 1:
            raise ValueError('Unique capacity anchor required')
        ef = z['EF_draws'][names.index('Capacity-OP')].copy()
        if ef.shape != (6, 128) or not np.array_equal(ef, np.broadcast_to(ef[:, :1], ef.shape)):
            raise ValueError('Particle-constant registered capacity anchor required')
        capacity = ef * mg[..., 0]
    audit = read(entry['route'])
    weights = np.asarray(audit['old_weights'], float)
    if weights.shape != (10, 6) or not np.isin(weights[8], (0., 1.)).all():
        raise ValueError('Exact binary CoalOilUpper router required')
    coil = np.where(weights[8, :, None] == 1., capacity, v42['CIF_draws'][0])
    mix, gain = monthly(entry['mix'], source, mg), monthly(entry['gain'], source, mg)
    if gain['arms'].tolist() != list(model.LEVEL_ARMS[15:]):
        raise ValueError('Exact thirty V44 gates required')
    names = mix['arms'].tolist()
    if names.count(model.LEVEL_ARMS[14]) != 1:
        raise ValueError('Unique fixed mix required')
    levels = np.concatenate((v42['CIF_draws'], capacity[None], coil[None],
                             mix['CIF_draws'][names.index(model.LEVEL_ARMS[14])][None], gain['CIF_draws']))
    return dict(scale_raw=anchor, scale_fine=fine, query_mean=mean, MG=mg, levels=levels,
                scale_recompose=recompose, m8_raw=m8)


def predict() -> None:
    frozen = check()
    if (OUT / 'prediction_started.json').exists() or (OUT / 'predictions').exists():
        raise FileExistsError('Preserve V46 prediction attempt')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    folder = OUT / 'predictions'
    folder.mkdir()
    entries, diagnostics = [], []
    for source in SOURCES:
        inputs = load_for_prediction(frozen['archives'][source], source)
        result = model.build(**inputs)
        path = folder / (source + '.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, arms=np.array(model.ARMS), level_arms=np.array(model.LEVEL_ARMS),
                origins=DATES.asi8, keys=np.array([f'{source}/{at:%Y-%m}' for at in DATES]),
                MG_draws=inputs['MG'], level_draws=inputs['levels'], raw=result['raw'],
                old_scale_level=result['old_scale_level'], old_point_level=result['old_point_level'])
        audit = path.with_suffix('.json')
        write(audit, dict(region=source, **result['audit']))
        for i, arm in enumerate(model.ARMS[3:]):
            for j, at in enumerate(DATES):
                diagnostics.append(dict(region=source, origin=at.isoformat(), arm=arm,
                    budget=model.BUDGET_BY_ARM[arm],
                    **{k: float(v[i, j]) for k, v in result['diagnostics'].items()}))
        entries.append(dict(region=source, path=str(path), sha256=sha(path),
                            audit_path=str(audit), audit_sha256=sha(audit)))
        print('Predicted', source, '99 arms, six month starts; zero fits', flush=True)
    if len(diagnostics) != 4032:
        raise ValueError('Complete raw diagnostic grid required')
    pd.DataFrame(diagnostics).to_csv(OUT / 'raw_diagnostics.csv', mode='x', index=False)
    check()
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), entries=entries,
        raw_diagnostics_sha256=sha(OUT / 'raw_diagnostics.csv'), complete_grid=True, source_truth_read=False))


def predictions() -> dict:
    seal = read(OUT / 'prediction_seal.json')
    if (not seal['complete_grid'] or seal['freeze_sha256'] != sha(OUT / 'freeze.json')
            or seal['prediction_started_sha256'] != sha(OUT / 'prediction_started.json')
            or [e['region'] for e in seal['entries']] != list(SOURCES)):
        raise ValueError('Complete prediction seal required')
    mapping = {str(OUT / 'raw_diagnostics.csv'): seal['raw_diagnostics_sha256']}
    for entry in seal['entries']:
        mapping.update({entry['path']: entry['sha256'], entry['audit_path']: entry['audit_sha256']})
    verify(mapping)
    result = {}
    for entry in seal['entries']:
        source = entry['region']
        with np.load(entry['path'], allow_pickle=False) as z:
            indices(z['origins'], False)
            if (z['arms'].tolist() != list(model.ARMS) or z['level_arms'].tolist() != list(model.LEVEL_ARMS)
                    or z['keys'].tolist() != [f'{source}/{at:%Y-%m}' for at in DATES]
                    or z['raw'].shape != (99, 6, 128, 24) or not np.isfinite(z['raw']).all()
                    or z['level_draws'].shape != (45, 6, 128)):
                raise ValueError('Complete sealed prediction grid required before truth')
            result[source] = z['raw'].copy()
    return result


def scoped_truth(entry: dict, source: str, keys: list[str]) -> tuple[list[int], np.ndarray]:
    positions, values = [], []
    for j, at in enumerate(DATES):
        if f'{source}/{at:%Y-%m-%d}' not in keys:
            continue
        rows = source_days(Path(entry['truth']), at.strftime('%Y-%m-%d'),
                           (at + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        if len(rows) != 1 or rows[0][0] != at or rows[0][1].shape != (24, 8):
            raise ValueError('Predeclared exact source day missing; preserve attempt')
        g = rows[0][1]
        values.append((g / g.sum(1)[:, None]) @ model.EF)
        positions.append(j)
    return positions, np.stack(values)


def restore_baselines(days: pd.DataFrame, original: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Verify old numeric scores, then copy their parsed numeric values literally."""
    keys = ['region', 'origin', 'arm', 'factor']
    old = original[original.arm.isin(model.ARMS[:2])].set_index(keys)
    selected = days.arm.isin(model.ARMS[:2])
    index = pd.MultiIndex.from_frame(days.loc[selected, keys])
    if len(index) != 320 or old.index.has_duplicates or not index.isin(old.index).all():
        raise ValueError('Exact 320 baseline rows required')
    values = old.loc[index, list(COPY_METRICS)].to_numpy(float)
    computed = days.loc[selected, list(COPY_METRICS)].to_numpy(float)
    error = float(abs(values - computed).max())
    if not np.isfinite(values).all() or error > 1e-9:
        raise ValueError('Original score reconstruction mismatch')
    result = days.copy()
    result.loc[selected, list(COPY_METRICS)] = values
    return result, error


def original_scores(path: str) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip')


def aggregate(days: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regions = days.groupby(['region', 'arm', 'factor', 'budget'], sort=False)[list(METRICS)].mean().reset_index()
    groups = ['arm', 'factor', 'budget']
    macro = regions.groupby(groups, sort=False)[list(METRICS)].mean().reset_index()
    worst = regions.groupby(groups, sort=False).empirical_coverage90.min().to_numpy()
    macro['worst_region_coverage90'] = worst
    baseline = macro[(macro.arm == 'Scale-anchor') & (macro.factor == 1.)].iloc[0]
    macro['primary_factor'] = macro.factor.eq(1.)
    macro['coverage_not_lower_width_not_higher_vs_anchor1'] = (
        (macro.empirical_coverage90 >= baseline.empirical_coverage90 - 1e-12)
        & (macro.empirical_width90 <= baseline.empirical_width90 + 1e-12))
    macro['nominal_and_efficient_vs_anchor1'] = (
        (macro.empirical_coverage90 >= .90) & (macro.worst_region_coverage90 >= .85)
        & (macro.empirical_width90 <= baseline.empirical_width90 + 1e-12)
        & (macro.crps <= baseline.crps + 1e-12)
        & (macro.empirical_interval_score90 <= baseline.empirical_interval_score90 + 1e-12))
    leave = []
    for source in SOURCES:
        remaining = regions[regions.region != source]
        table = remaining.groupby(groups, sort=False)[list(METRICS)].mean().reset_index()
        table.insert(0, 'excluded_region', source)
        leave.append(table)
    return regions, macro, pd.concat(leave, ignore_index=True)


def score() -> None:
    frozen = check()
    raw = predictions()  # Validate all seven archives before any truth is parsed.
    if (OUT / 'score_started.json').exists() or (OUT / 'completed.json').exists():
        raise FileExistsError('Preserve V46 score attempt')
    write(OUT / 'score_started.json', dict(created_utc=now(), prediction_seal_sha256=sha(OUT / 'prediction_seal.json'),
        authorized_score_keys=frozen['score_keys']))
    rows, support = [], []
    for source in SOURCES:
        positions, truth = scoped_truth(frozen['archives'][source], source, frozen['score_keys'])
        for j, at in enumerate(DATES):
            support.append(dict(region=source, origin=at.isoformat(), available=j in positions))
        for i, arm in enumerate(model.ARMS):
            for factor in FACTORS:
                point, ensemble, clipping = scaled_paths(raw[source][i, positions], factor)
                metrics = score_arrays(point, ensemble, truth)
                for k, j in enumerate(positions):
                    rows.append(dict(region=source, origin=DATES[j].isoformat(), arm=arm,
                        factor=factor, budget=model.BUDGET_BY_ARM[arm],
                        final_spread_clipping_rate=float(clipping[k]),
                        **{name: float(value[k]) for name, value in metrics.items()}))
        print('Scored', source, len(positions), 'predeclared source days', flush=True)
    days = pd.DataFrame(rows)
    old = original_scores(frozen['baseline_path'])
    days, error = restore_baselines(days, old)
    regions, macro, leave = aggregate(days)
    tables = dict(zip(OUTPUTS[:5], (days, regions, macro, leave, pd.DataFrame(support))))
    for (name, table), count in zip(tables.items(), (15840, 3465, 495, 3465, 42)):
        if len(table) != count or not np.isfinite(table.select_dtypes(include='number').to_numpy()).all():
            raise ValueError('Complete finite score tables required')
        table.to_csv(OUT / name, mode='x', index=False)
    write(OUT / 'summary.json', dict(arms=99, source_days=32, sources=7, prediction_queries=42,
        legacy_rows_copied_numeric_exact=320, legacy_max_recomputed_abs_error=error,
        new_fits=0, factors=list(FACTORS), primary_factor=1.,
        hourly_marginal_coverage_is_simultaneous_coverage=False,
        independent_confirmation=False, new_holdout_authorized=False, method_innovation_claim=False))
    check()
    predictions()
    write(OUT / 'completed.json', dict(created_utc=now(), status='COMPLETE',
        freeze_sha256=sha(OUT / 'freeze.json'), prediction_seal_sha256=sha(OUT / 'prediction_seal.json'),
        score_started_sha256=sha(OUT / 'score_started.json'),
        output_hashes={str(OUT / p): sha(OUT / p) for p in OUTPUTS},
        source_truth_days_read=32, target_values_opened=False, unseen_values_opened=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'predict', 'score'))
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score}[args.phase]()
