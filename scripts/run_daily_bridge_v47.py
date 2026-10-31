"""Daily replay of frozen ordinary heads; scoped development scoring after sealing."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import daily_bridge_v47 as model
from scripts import run_hourly_bridge_v46 as prior

sha, read, write, now, verify = prior.sha, prior.read, prior.write, prior.now, prior.verify
SOURCES, DAYS, MONTHS, FACTORS = prior.SOURCES, prior.DAYS, prior.DATES, prior.FACTORS
METRICS, COPY_METRICS = prior.METRICS, prior.COPY_METRICS
OUT = ROOT / 'results/daily-bridge-v47'
PARENT = ROOT / 'docs/evidence/hourly-bridge-v46-verification.json'
PARENT_SHA = 'e0597281ae5ce839df6588b0f7a7bb2af78cd8bb49d99ccb84ad2b7ccd78257d'
CODE = ('scripts/daily_bridge_v47.py', 'scripts/run_daily_bridge_v47.py',
        'tests/test_daily_bridge_v47.py', 'tests/test_run_daily_bridge_v47.py', 'docs/DAILY_BRIDGE_V47_PROTOCOL.md')
HELPERS = ('run_hourly_bridge_v46', 'hourly_bridge_v46', 'forward_context_v31', 'fleet_information_v34',
           'donor_inventory_v41',
           'coarse_context_v30', 'run_coarse_context_v30', 'run_local_sketch_v17', 'local_sketch_v17',
           'run_information_value_v10', 'native_center_v18')
REVIEWS = tuple(ROOT / f'results/daily-bridge-v47-{kind}-review.md' for kind in ('code', 'math'))
COUNTS = dict(zip(SOURCES, (176, 178, 180, 179, 89, 166, 177)))
OUTPUTS = ('day_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv',
           'leave_one_source_out_macro.csv', 'label_support.csv', 'summary.json')


def inherited() -> dict:
    verify({str(PARENT): PARENT_SHA})
    cursor = read(PARENT)
    if cursor['status'] != 'VERIFICATION_COMPLETE' or cursor['independent_formula_review_status'] != 'PASS':
        raise ValueError('Independent V46 verification required')
    result, seen = {}, set()
    while True:
        for kind in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
            for p, h in cursor[kind].items():
                if p in result and result[p] != h:
                    raise ValueError('Conflicting inherited metadata')
                result[p] = h
        if 'parent_manifest' not in cursor:
            break
        p, h = cursor['parent_manifest'], cursor['parent_manifest_sha256']
        if p in seen:
            raise ValueError('Cyclic inherited metadata')
        seen.add(p)
        verify({p: h})
        cursor = read(p)
    return result


def support_metadata(table: pd.DataFrame) -> list[str]:
    if set(table) != {'region', 'origin', 'arm', 'factor'}:
        raise ValueError('Availability columns only required')
    table = table.copy()
    table['origin'] = pd.to_datetime(table.origin, utc=True, errors='raise')
    if (not table.origin.isin(DAYS).all() or not table.region.isin(SOURCES).all()
            or table.duplicated().any()):
        raise ValueError('Unique registered daily availability required')
    grid = {(a, f) for a in prior.V29_ARMS for f in FACTORS}
    pairs = set()
    for (source, at), group in table.groupby(['region', 'origin'], sort=False):
        if len(group) != 50 or set(zip(group.arm, group.factor)) != grid:
            raise ValueError('Exact old ten-arm five-factor grid required')
        pairs.add((source, at))
    if {s: sum(source == s for source, _ in pairs) for s in SOURCES} != COUNTS:
        raise ValueError('Original 1145 daily support required')
    return [f'{s}/{at:%Y-%m-%d}' for s in SOURCES for at in DAYS if (s, at) in pairs]


def contract() -> dict:
    closure = inherited()
    old_path = str(ROOT / 'results/conditional-law-v29/freeze.json')
    if old_path not in closure:
        raise ValueError('Bound V29 aliases required')
    verify({old_path: closure[old_path]})
    old = read(old_path)
    baseline = str(ROOT / 'results/conditional-law-v29/day_metrics.csv')
    monthly = str(ROOT / 'results/heldout-2026h1-data/monthly.csv')
    capacity = str(ROOT / 'results/fleet-inventory-v33/capacity_op.csv')
    archives = {s: dict(
        hourly=str(ROOT / f'results/conditional-law-v29/predictions/{s}.npz'),
        m8=old['monthly_prediction_paths'][s],
        v30=str(ROOT / f'results/coarse-context-v30/predictions/{s}.json'),
        v34=str(ROOT / f'results/fleet-information-v34/predictions/{s}.json'),
        v42=str(ROOT / f'results/donor-budget-v42/predictions/{s}.json'),
        monthly31=str(ROOT / f'results/forward-context-v31/predictions/{s}.npz'),
        monthly34=str(ROOT / f'results/fleet-information-v34/predictions/{s}.npz'),
        monthly42=str(ROOT / f'results/donor-budget-v42/predictions/{s}.npz'),
        mix=str(ROOT / f'results/ordinary-controls-v43/predictions/{s}.npz'),
        route=str(ROOT / f'results/ordinary-controls-v43/predictions/{s}.json'),
        truth=str(Path(old['source']) / f'{s}_hourly.csv')) for s in SOURCES}
    aliases = [old_path, baseline, monthly, capacity, *[p for a in archives.values() for p in a.values()],
               *[str(ROOT / 'scripts' / (h + '.py')) for h in HELPERS]]
    if any(p not in closure for p in aliases):
        raise ValueError('Inherited direct input aliases required')
    inputs = {p: closure[p] for p in aliases} | {str(PARENT): PARENT_SHA}
    for path in REVIEWS:
        if not path.is_file() or path.is_symlink() or not path.read_text().strip():
            raise ValueError('Both final preflight reports must exist')
        inputs[str(path)] = sha(path)
    verify(inputs)
    keys = support_metadata(pd.read_csv(baseline, usecols=['region', 'origin', 'arm', 'factor']))
    return dict(version=47, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE}, input_hashes=inputs,
        environment=prior.common.environment(), archives=archives, baseline_path=baseline,
        monthly_path=monthly, capacity_path=capacity, score_keys=keys, sources=list(SOURCES),
        origins=DAYS.asi8.tolist(), arms=list(model.ARMS), level_arms=list(model.LEVEL_ARMS),
        factors=list(FACTORS), budgets=model.BUDGET_BY_ARM, ET_reconstruction_fits=42, Ridge_replays=42,
        new_training_labels=False, MG_retrained=False, source_truth_days_authorized=1145,
        target_values_opened=False, unseen_values_opened=False, method_innovation_claim=False,
        independent_confirmation=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V47 attempt')
    frozen = contract()
    OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared V47: 1267 daily queries, 1145 score days, 42 ET reconstructions', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('Frozen V47 contract changed')
    return frozen


def capacity_for(path: str, source: str) -> tuple[np.ndarray, float]:
    with open(path, newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['year'] == '2023' and r['region'] == source]
    if len(rows) != 1:
        raise ValueError('Unique year-minus-two OP row required')
    row = rows[0]
    values = np.array([float(row['capacity_' + f + '_mw']) for f in model.FUELS])
    pos = values[list(model.POS)].sum()
    if not np.isfinite(values).all() or np.any(values < 0) or pos <= 0:
        raise ValueError('Finite nonnegative registered inventory with positive POS required')
    return values[list(model.POS)] / pos, float(pos)


def records(entry: dict, source: str) -> list[dict]:
    audits = {name: read(entry[name]) for name in ('v30', 'v34', 'v42')}
    if (any(a['region'] != source for a in audits.values())
            or [len(audits[k]['fits']) for k in ('v30', 'v34', 'v42')] != [10, 7, 6]
            or audits['v30']['outer'] != source or audits['v42']['outer'] != source
            or len(audits['v42']['training_keys']) != (388 if source == 'FPL' else 386)
            or len(set(audits['v42']['training_regions'])) != 15):
        raise ValueError('Matching outer audit identity required')
    by = {name: {f['arm']: f for f in a['fits']} for name, a in audits.items()}
    if any(len(a['fits']) != len(by[name]) for name, a in audits.items()):
        raise ValueError('Unique frozen fit records required')
    chosen = []
    for name in model.LEVEL_ARMS[:12]:
        scope, arm = name.split('-', 1)
        origin = 'v42' if scope == 'Expanded' else 'v30' if arm.startswith('Size-') else 'v34'
        record = dict(by[origin][arm])
        if origin == 'v42':
            top = audits[origin]
            if record['training_regions'] != top['training_regions']:
                raise ValueError('Expanded record/top source identity required')
            for field in ('training_keys', 'training_period_start_ns', 'training_period_end_ns', 'training_available_ns'):
                record[field] = top[field]
        chosen.append(record)
    return chosen


def hourly_inputs(entry: dict) -> dict:
    with np.load(entry['hourly'], allow_pickle=False) as z:
        prior.indices(z['origins'], True)
        if z['arms'].tolist() != list(prior.V29_ARMS):
            raise ValueError('Exact inherited hourly registry required')
        a, b = prior.V29_ARMS.index('Scale-anchor'), prior.V29_ARMS.index('Scale-recompose')
        result = dict(scale_raw=z['raw'][a].copy(), scale_fine=z['fine_draws'][a].copy(),
            query_mean=z['query_mean'].copy(), MG=z['MG_draws'].copy(), scale_recompose=z['raw'][b].copy())
    with np.load(entry['m8'], allow_pickle=False) as z:
        prior.indices(z['origins'], True)
        if not np.array_equal(z['MG_draws'], result['MG']):
            raise ValueError('Literal daily V25 MG coupling required')
        names = z['arms'].tolist()
        if names.count('M8-recompose_conditional') != 1:
            raise ValueError('Unique M8 reference required')
        result['m8_raw'] = z['raw'][names.index('M8-recompose_conditional')].copy()
    return result


def month_start_replay(entry: dict, source: str, feature: dict, heads: dict,
                       levels: np.ndarray, cap_ef: np.ndarray, fit_records: list[dict]) -> float:
    take = np.array([DAYS.get_loc(at) for at in MONTHS])
    old = prior.monthly(entry['monthly42'], source, feature['MG_draws'][take])
    if old['arms'].tolist() != list(model.LEVEL_ARMS[:12]):
        raise ValueError('Exact twelve original ordinary arms required')
    errors = [float(abs(levels[:12, take] - old['CIF_draws']).max())]
    with np.load(entry['monthly42'], allow_pickle=False) as z:
        errors.extend((float(abs(heads['EF_draws'][:, take] - z['EF_draws']).max()),
                       float(abs(heads['projection_l1'][6:, take] - z['expanded_projection_l1']).max())))
    old_raw = []
    with np.load(entry['monthly31'], allow_pickle=False) as z:
        prior.indices(z['origins'], False)
        if not np.array_equal(z['MG_draws'], feature['MG_draws'][take]):
            raise ValueError('Literal V31 month-start MG coupling required')
        for arm in ('Size-Ridge', 'Size-ET'):
            old_raw.append(z['ratios_raw'][z['arms'].tolist().index(arm)].copy())
    with np.load(entry['monthly34'], allow_pickle=False) as z:
        prior.indices(z['origins'], False)
        if (not np.array_equal(z['MG_draws'], feature['MG_draws'][take])
                or z['keys'].tolist() != [f'{source}/{at:%Y-%m}' for at in MONTHS]):
            raise ValueError('Original month-start feature/MG alignment required')
        for family in ('Size', 'SizeCap', 'DirectEF'):
            if not np.array_equal(feature['features'][family][take], z['query_feature_' + family]):
                raise ValueError('Literal original month-start feature replay required')
        cap = z['EF_draws'][z['arms'].tolist().index('Capacity-OP')]
        errors.append(float(abs(cap - cap_ef[take]).max()))
        cap_cif = cap * feature['MG_draws'][take, :, 0]
        weights = np.asarray(read(entry['route'])['old_weights'], float)[8]
        old_coil = np.where(weights[:, None] == 1., cap_cif, old['CIF_draws'][0])
        errors.extend((float(abs(levels[12, take] - cap_cif).max()),
                       float(abs(levels[13, take] - old_coil).max())))
        for i, arm in enumerate(('Size-Ridge', 'Size-ET', 'SizeCap-Ridge', 'SizeCap-ET', 'DirectEF-Ridge', 'DirectEF-ET')):
            errors.append(float(abs(heads['projection_l1'][i, take] - z['projection_l1'][z['arms'].tolist().index(arm)]).max()))
        for arm in ('SizeCap-Ridge', 'SizeCap-ET'):
            old_raw.append(z['ratios_raw'][z['ratio_arms'].tolist().index(arm)].copy())
    old_raw.extend(np.asarray(f['raw_query_prediction'], float) for f in fit_records[4:])
    if len(old_raw) != 12 or len(heads['raw_predictions']) != 12:
        raise ValueError('All twelve original raw head replays required')
    for actual, expected in zip(heads['raw_predictions'], old_raw):
        if actual[take].shape != expected.shape:
            raise ValueError('Original raw head prediction shape changed')
        errors.append(float(abs(actual[take] - expected).max()))
    mixed = prior.monthly(entry['mix'], source, feature['MG_draws'][take])
    names = mixed['arms'].tolist()
    errors.append(float(abs(levels[14, take] - mixed['CIF_draws'][names.index(model.LEVEL_ARMS[14])]).max()))
    # Route endpoints and mixture are checked independently of the fitting library.
    for i in range(12):
        if not np.allclose(levels[i], heads['EF_draws'][i] * feature['MG_draws'][..., 0], atol=1e-9, rtol=0):
            raise ValueError('Daily CIF/EF identity required')
    error = max(errors)
    if error > 1e-9:
        raise ValueError('Fixed head month-start reproduction failed')
    return error


def predict() -> None:
    frozen = check()
    if (OUT / 'prediction_started.json').exists() or (OUT / 'predictions').exists():
        raise FileExistsError('Preserve V47 prediction attempt')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    folder = OUT / 'predictions'
    folder.mkdir()
    with open(frozen['monthly_path'], newline='') as stream:
        monthly_rows = list(csv.DictReader(stream))
    entries, diagnostics = [], []
    fits = 0
    for source in SOURCES:
        entry = frozen['archives'][source]
        inputs = hourly_inputs(entry)
        q, cpos = capacity_for(frozen['capacity_path'], source)
        feature = model.daily_features(monthly_rows, source, inputs['MG'], q, cpos)
        fit_records = records(entry, source)
        heads = model.replay_heads(fit_records, feature)
        fits += heads['audit']['fit_count']
        cap = np.full((181, 128), float(q @ model.EF[list(model.POS)]))
        w = np.asarray(read(entry['route'])['old_weights'], float)
        if w.shape != (10, 6) or not np.isin(w[8], (0., 1.)).all() or not (w[8] == w[8, 0]).all():
            raise ValueError('Fixed half-year binary CoalOil route required')
        coil = cap if w[8, 0] else heads['EF_draws'][0]
        mixed = .75 * coil + .25 * heads['EF_draws'][11]
        ef = np.concatenate((heads['EF_draws'], cap[None], coil[None], mixed[None]))
        levels = ef * inputs['MG'][None, ..., 0]
        replay = month_start_replay(entry, source, feature, heads, levels, cap, fit_records)
        result = model.build_paths(**inputs, levels=levels)
        path = folder / (source + '.npz')
        arrays = {k: v for k, v in feature.items() if isinstance(v, np.ndarray)}
        arrays.update({'query_feature_' + k: v for k, v in feature['features'].items()})
        arrays.update(raw=result['raw'], level_draws=levels, EF_draws=ef,
            head_projection_l1=heads['projection_l1'], head_EF_raw=heads['EF_raw'],
            arms=np.array(model.ARMS), level_arms=np.array(model.LEVEL_ARMS), origins=DAYS.asi8,
            keys=np.array([f'{source}/{at:%Y-%m-%d}' for at in DAYS]),
            old_scale_level=result['old_scale_level'], old_point_level=result['old_point_level'])
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        audit = path.with_suffix('.json')
        write(audit, dict(region=source, head_audit=heads['audit'], feature_audit=feature['audit'],
            month_start_replay_max_abs=replay, capacity_q=q, capacity_pos_mw=cpos,
            coal_oil_right_weight=float(w[8, 0]), **result['audit']))
        for i, arm in enumerate(model.ARMS[3:]):
            for j, at in enumerate(DAYS):
                diagnostics.append(dict(region=source, origin=at.isoformat(), arm=arm,
                    budget=model.BUDGET_BY_ARM[arm],
                    **{k: float(v[i, j]) for k, v in result['diagnostics'].items()}))
        entries.append(dict(region=source, path=str(path), sha256=sha(path), audit_path=str(audit), audit_sha256=sha(audit)))
        print('Predicted', source, '63 arms x181 days; 6 ET rebuilt; month-start error', replay, flush=True)
        del inputs, feature, heads, result, arrays
    if fits != 42 or len(diagnostics) != 76020:
        raise ValueError('Exact deterministic fit and diagnostic counts required')
    pd.DataFrame(diagnostics).to_csv(OUT / 'raw_diagnostics.csv', mode='x', index=False)
    check()
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), entries=entries,
        raw_diagnostics_sha256=sha(OUT / 'raw_diagnostics.csv'), complete_grid=True, ET_fits=fits, source_truth_read=False))


def predictions() -> list[dict]:
    """Validate all seven archives before truth; retain only metadata in memory."""
    seal = read(OUT / 'prediction_seal.json')
    if (not seal['complete_grid'] or seal['ET_fits'] != 42
            or seal['freeze_sha256'] != sha(OUT / 'freeze.json')
            or seal['prediction_started_sha256'] != sha(OUT / 'prediction_started.json')
            or [e['region'] for e in seal['entries']] != list(SOURCES)):
        raise ValueError('Complete sealed daily prediction cohort required')
    hashes = {str(OUT / 'raw_diagnostics.csv'): seal['raw_diagnostics_sha256']}
    for entry in seal['entries']:
        hashes.update({entry['path']: entry['sha256'], entry['audit_path']: entry['audit_sha256']})
    verify(hashes)
    for entry in seal['entries']:
        with np.load(entry['path'], allow_pickle=False) as z:
            prior.indices(z['origins'], True)
            raw = z['raw']
            if (raw.shape != (63, 181, 128, 24) or not np.isfinite(raw).all()
                    or z['arms'].tolist() != list(model.ARMS) or z['level_arms'].tolist() != list(model.LEVEL_ARMS)
                    or z['keys'].tolist() != [f"{entry['region']}/{at:%Y-%m-%d}" for at in DAYS]
                    or z['level_draws'].shape != (15, 181, 128)):
                raise ValueError('Complete finite daily archive required before truth')
    return seal['entries']


def scoped_truth(entry: dict, source: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = prior.source_days(Path(entry['truth']), '2025-01-01', '2025-07-01')
    actual = [f'{source}/{at:%Y-%m-%d}' for at, _ in rows]
    expected = [k for k in keys if k.startswith(source + '/')]
    if actual != expected:
        raise ValueError('Actual valid UTC day support differs; preserve attempt')
    positions = np.array([DAYS.get_loc(at) for at, _ in rows], int)
    truth = np.stack([(g / g.sum(1)[:, None]) @ model.EF for _, g in rows])
    return positions, truth


def restore_baselines(days: pd.DataFrame, original: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    keys = ['region', 'origin', 'arm', 'factor']
    old = original[original.arm.isin(model.ARMS[:2])].set_index(keys)
    take = days.arm.isin(model.ARMS[:2])
    index = pd.MultiIndex.from_frame(days.loc[take, keys])
    if len(index) != 11450 or old.index.has_duplicates or not index.isin(old.index).all():
        raise ValueError('Exact 11450 old baseline rows required')
    values = old.loc[index, list(COPY_METRICS)].to_numpy(float)
    error = float(abs(values - days.loc[take, list(COPY_METRICS)].to_numpy(float)).max())
    if not np.isfinite(values).all() or error > 1e-9:
        raise ValueError('Original daily scoring mismatch')
    answer = days.copy()
    answer.loc[take, list(COPY_METRICS)] = values
    return answer, error


def score() -> None:
    frozen = check()
    entries = predictions()
    if (OUT / 'score_started.json').exists() or (OUT / 'completed.json').exists():
        raise FileExistsError('Preserve V47 score attempt')
    write(OUT / 'score_started.json', dict(created_utc=now(), prediction_seal_sha256=sha(OUT / 'prediction_seal.json'),
        authorized_source_days=1145))
    rows, support = [], []
    for entry in entries:
        source = entry['region']
        with np.load(entry['path'], allow_pickle=False) as z:
            raw = z['raw']
        positions, truth = scoped_truth(frozen['archives'][source], source, frozen['score_keys'])
        for j, at in enumerate(DAYS):
            support.append(dict(region=source, origin=at.isoformat(), available=j in positions))
        for i, arm in enumerate(model.ARMS):
            for factor in FACTORS:
                point, ensemble, clip = prior.scaled_paths(raw[i, positions], factor)
                metrics = prior.score_arrays(point, ensemble, truth)
                for k, j in enumerate(positions):
                    rows.append(dict(region=source, origin=DAYS[j].isoformat(), arm=arm, factor=factor,
                        budget=model.BUDGET_BY_ARM[arm], final_spread_clipping_rate=float(clip[k]),
                        **{name: float(value[k]) for name, value in metrics.items()}))
        print('Scored', source, len(positions), 'original development days', flush=True)
        del raw
    day, error = restore_baselines(pd.DataFrame(rows), prior.original_scores(frozen['baseline_path']))
    regional, macro, leave = prior.aggregate(day)
    tables = dict(zip(OUTPUTS[:5], (day, regional, macro, leave, pd.DataFrame(support))))
    for (name, table), count in zip(tables.items(), (360675, 2205, 315, 2205, 1267)):
        if len(table) != count or not np.isfinite(table.select_dtypes(include='number').to_numpy()).all():
            raise ValueError('Complete finite daily score grid required')
        table.to_csv(OUT / name, mode='x', index=False)
    write(OUT / 'summary.json', dict(prediction_queries=1267, source_days=1145, arms=63,
        ET_reconstruction_fits=42, Ridge_replays=42, new_training_labels=False,
        legacy_rows_copied_numeric_exact=11450, legacy_recompute_max_abs=error,
        primary_factor=1., fixed_factors=list(FACTORS), method_innovation_claim=False,
        independent_confirmation=False, method_frozen=False, new_holdout_authorized=False))
    check()
    write(OUT / 'completed.json', dict(created_utc=now(), status='COMPLETE', freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_seal_sha256=sha(OUT / 'prediction_seal.json'), score_started_sha256=sha(OUT / 'score_started.json'),
        output_hashes={str(OUT / p): sha(OUT / p) for p in OUTPUTS}, source_truth_days_read=1145,
        target_values_opened=False, unseen_values_opened=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'predict', 'score'))
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score}[args.phase]()
