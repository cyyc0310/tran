"""Alternate hourly-base experiment; zero fits and complete inherited comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import recompose_bridge_v48 as model
from scripts import run_hourly_bridge_v46 as common

sha, read, write, now, verify = common.sha, common.read, common.write, common.now, common.verify
SOURCES, DAYS, FACTORS, METRICS = common.SOURCES, common.DAYS, common.FACTORS, common.METRICS
OUT = ROOT / 'results/recompose-bridge-v48'
PARENT = ROOT / 'docs/evidence/daily-bridge-v47-verification.json'
PARENT_SHA = '4e5a3cfd29b6c6db424f734e89bbe05f7e7ba253b7dea652b4a1b994075df87c'
CODE = ('scripts/recompose_bridge_v48.py', 'scripts/run_recompose_bridge_v48.py',
        'tests/test_recompose_bridge_v48.py', 'tests/test_run_recompose_bridge_v48.py',
        'docs/RECOMPOSE_BRIDGE_V48_PROTOCOL.md')
HELPERS = ('run_hourly_bridge_v46', 'hourly_bridge_v46', 'daily_bridge_v47', 'coarse_context_v30',
           'forward_context_v31', 'fleet_information_v34', 'donor_inventory_v41', 'run_coarse_context_v30',
           'run_local_sketch_v17', 'local_sketch_v17', 'run_information_value_v10', 'native_center_v18')
REVIEWS = tuple(ROOT / f'results/recompose-bridge-v48-{kind}-review-r1.md' for kind in ('code', 'math'))
OUTPUTS = ('day_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv',
           'leave_one_source_out_macro.csv', 'label_support.csv', 'summary.json')


def inherited() -> dict:
    verify({str(PARENT): PARENT_SHA})
    cursor = read(PARENT)
    if cursor['status'] != 'VERIFICATION_COMPLETE' or cursor['independent_formula_review_status'] != 'PASS':
        raise ValueError('Independent V47 PASS required')
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
        seen.add(p); verify({p: h}); cursor = read(p)
    return result


def support_metadata(table: pd.DataFrame, keys: list[str]) -> None:
    if set(table) != {'region', 'origin', 'available'} or len(table) != 1267:
        raise ValueError('Complete inherited availability grid required')
    table = table.copy()
    table['origin'] = pd.to_datetime(table.origin, utc=True, errors='raise')
    expected = {(s, at) for s in SOURCES for at in DAYS}
    if (table.duplicated(['region', 'origin']).any() or set(zip(table.region, table.origin)) != expected
            or any(type(value) not in (bool, np.bool_) for value in table.available)):
        raise ValueError('Unique exact UTC availability and boolean mask required')
    actual = {f'{row.region}/{row.origin:%Y-%m-%d}' for row in table.itertuples() if row.available}
    if len(keys) != 1145 or len(set(keys)) != 1145 or actual != set(keys):
        raise ValueError('Exact inherited 1145 score keys required')


def contract() -> dict:
    closure = inherited()
    old_path = str(ROOT / 'results/daily-bridge-v47/freeze.json')
    if old_path not in closure:
        raise ValueError('Bound V47 freeze alias required')
    verify({old_path: closure[old_path]})
    old = read(old_path)
    if (old['sources'] != list(SOURCES) or old['origins'] != DAYS.asi8.tolist()
            or old['arms'] != list(model.prior.ARMS) or old['level_arms'] != list(model.LEVEL_ARMS)):
        raise ValueError('Exact inherited dates and registries required')
    baseline = str(ROOT / 'results/daily-bridge-v47/day_metrics.csv')
    support = str(ROOT / 'results/daily-bridge-v47/label_support.csv')
    archives = {s: dict(hourly=old['archives'][s]['hourly'], truth=old['archives'][s]['truth'],
                       levels=str(ROOT / f'results/daily-bridge-v47/predictions/{s}.npz')) for s in SOURCES}
    aliases = [old_path, baseline, support, *[p for entry in archives.values() for p in entry.values()],
               *[str(ROOT / 'scripts' / (h + '.py')) for h in HELPERS]]
    if any(p not in closure for p in aliases):
        raise ValueError('Inherited hashes required for every direct dependency')
    inputs = {p: closure[p] for p in aliases} | {str(PARENT): PARENT_SHA}
    for p in REVIEWS:
        if not p.is_file() or p.is_symlink() or not p.read_text().strip():
            raise ValueError('Both final prelaunch reports must exist')
        inputs[str(p)] = sha(p)
    verify(inputs)
    support_metadata(pd.read_csv(support), old['score_keys'])
    return dict(version=48, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE}, input_hashes=inputs,
        environment=common.common.environment(), archives=archives, baseline_path=baseline, support_path=support,
        score_keys=old['score_keys'], sources=list(SOURCES), origins=DAYS.asi8.tolist(), arms=list(model.ARMS),
        new_arms=list(model.NEW_ARMS), level_arms=list(model.LEVEL_ARMS), factors=list(FACTORS),
        budgets=model.BUDGET_BY_ARM, new_fits=0, authorized_source_days=1145,
        target_values_opened=False, unseen_values_opened=False, method_innovation_claim=False,
        independent_confirmation=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V48 attempt')
    frozen = contract(); OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared V48: 75 new controls, 63 old arms, 1145 development days; zero fits', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('Frozen V48 contract changed')
    return frozen


def query(entry: dict, source: str) -> dict:
    with np.load(entry['levels'], allow_pickle=False) as z:
        common.indices(z['origins'], True)
        if (z['arms'].tolist() != list(model.prior.ARMS) or z['level_arms'].tolist() != list(model.LEVEL_ARMS)
                or z['keys'].tolist() != [f'{source}/{at:%Y-%m-%d}' for at in DAYS]):
            raise ValueError('Literal inherited daily grid and level registry required')
        mg, levels = z['MG_draws'].copy(), z['level_draws'].copy()
    with np.load(entry['hourly'], allow_pickle=False) as z:
        common.indices(z['origins'], True)
        if z['arms'].tolist() != list(common.V29_ARMS) or not np.array_equal(z['MG_draws'], mg):
            raise ValueError('Literal daily hourly/level MG coupling required')
        a, r = common.V29_ARMS.index('Scale-anchor'), common.V29_ARMS.index('Scale-recompose')
        return dict(recompose_raw=z['raw'][r].copy(), anchor_raw=z['raw'][a].copy(), recompose_fine=z['fine_draws'][r].copy(),
            anchor_fine=z['fine_draws'][a].copy(), query_mean=z['query_mean'].copy(), MG=mg, levels=levels)


def predict() -> None:
    frozen = check()
    if (OUT / 'prediction_started.json').exists() or (OUT / 'predictions').exists():
        raise FileExistsError('Preserve V48 prediction attempt')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    folder = OUT / 'predictions'; folder.mkdir()
    entries, diagnostics = [], []
    for source in SOURCES:
        inputs = query(frozen['archives'][source], source)
        result = model.build(**inputs)
        with np.load(frozen['archives'][source]['levels'], allow_pickle=False) as z:
            old_raw = z['raw']
            if (not np.array_equal(old_raw[0], inputs['anchor_raw'])
                    or not np.array_equal(old_raw[1], inputs['recompose_raw'])):
                raise ValueError('Literal inherited anchor and recompose references required')
            aligned_error = float(abs(result['raw'][60:].mean(-1) - old_raw[18:33].mean(-1)).max())
            if not np.isfinite(aligned_error) or aligned_error > 1e-9:
                raise ValueError('AlignedKeep must retain actual inherited Keep particle day means')
            del old_raw
        p = folder / (source + '.npz')
        with p.open('xb') as stream:
            np.savez_compressed(stream, raw=result['raw'], arms=np.array(model.NEW_ARMS),
                level_arms=np.array(model.LEVEL_ARMS), levels=inputs['levels'], MG_draws=inputs['MG'],
                origins=DAYS.asi8, keys=np.array([f'{source}/{at:%Y-%m-%d}' for at in DAYS]),
                old_scale_level=result['old_scale_level'], old_point_level=result['old_point_level'])
        audit = p.with_suffix('.json')
        write(audit, dict(region=source, inherited_level_path=frozen['archives'][source]['levels'],
                         aligned_inherited_Keep_mean_max_abs=aligned_error, **result['audit']))
        entries.append(dict(region=source, path=str(p), sha256=sha(p), audit_path=str(audit), audit_sha256=sha(audit)))
        for i, arm in enumerate(model.NEW_ARMS):
            for j, at in enumerate(DAYS):
                diagnostics.append(dict(region=source, origin=at.isoformat(), arm=arm, budget=model.BUDGET_BY_ARM[arm],
                    **{k: float(v[i, j]) for k, v in result['diagnostics'].items()}))
        print('Predicted', source, '75 recompose-base controls x181; zero fits', flush=True)
        del inputs, result
    if len(diagnostics) != 95025:
        raise ValueError('Complete new diagnostic grid required')
    pd.DataFrame(diagnostics).to_csv(OUT / 'raw_diagnostics.csv', mode='x', index=False)
    check()
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), complete_grid=True,
        raw_diagnostics_sha256=sha(OUT / 'raw_diagnostics.csv'), entries=entries, source_truth_read=False))


def predictions() -> list[dict]:
    seal = read(OUT / 'prediction_seal.json')
    if (not seal['complete_grid'] or seal['freeze_sha256'] != sha(OUT / 'freeze.json')
            or seal['prediction_started_sha256'] != sha(OUT / 'prediction_started.json')
            or [e['region'] for e in seal['entries']] != list(SOURCES)):
        raise ValueError('Complete prediction seal required before truth')
    hashes = {str(OUT / 'raw_diagnostics.csv'): seal['raw_diagnostics_sha256']}
    for e in seal['entries']:
        hashes.update({e['path']: e['sha256'], e['audit_path']: e['audit_sha256']})
    verify(hashes)
    for e in seal['entries']:
        with np.load(e['path'], allow_pickle=False) as z:
            common.indices(z['origins'], True)
            raw = z['raw']
            if (z['arms'].tolist() != list(model.NEW_ARMS) or z['level_arms'].tolist() != list(model.LEVEL_ARMS)
                    or raw.shape != (75, 181, 128, 24) or not np.isfinite(raw).all()
                    or z['keys'].tolist() != [f"{e['region']}/{at:%Y-%m-%d}" for at in DAYS]
                    or z['levels'].shape != (15, 181, 128)):
                raise ValueError('Complete finite sealed daily grid required')
    return seal['entries']


def truth(entry: dict, source: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = common.source_days(Path(entry['truth']), '2025-01-01', '2025-07-01')
    actual = [f'{source}/{at:%Y-%m-%d}' for at, _ in rows]
    if actual != [k for k in keys if k.startswith(source + '/')]:
        raise ValueError('Inherited daily support mismatch; preserve attempt')
    return np.array([DAYS.get_loc(at) for at, _ in rows], int), np.stack([
        (g / g.sum(1)[:, None]) @ model.EF for _, g in rows])


def combine(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    keys = ['region', 'origin', 'arm', 'factor']
    for table, arms, count in ((old, model.prior.ARMS, 360675), (new, model.NEW_ARMS, 429375)):
        if (len(table) != count or set(table.arm) != set(arms) or table.duplicated(keys).any()
                or not np.isfinite(table[list(METRICS)].to_numpy(float)).all()
                or any(b != model.BUDGET_BY_ARM[a] for a, b in zip(table.arm, table.budget))):
            raise ValueError('Complete finite disjoint inherited/new metric grids required')
        cohort = table[['region', 'origin']].drop_duplicates()
        if len(cohort) != 1145 or table.groupby(['region', 'origin']).size().ne(len(arms)*5).any():
            raise ValueError('Complete per-day inherited/new metric grid required')
        if set(table.factor) != set(FACTORS):
            raise ValueError('All five fixed factors required')
    if set(map(tuple, old[['region', 'origin']].to_numpy())) != set(map(tuple, new[['region', 'origin']].to_numpy())):
        raise ValueError('Literal inherited/new score cohort required')
    return pd.concat((old, new), ignore_index=True)


def score() -> None:
    frozen = check(); entries = predictions()
    if (OUT / 'score_started.json').exists() or (OUT / 'completed.json').exists():
        raise FileExistsError('Preserve V48 score attempt')
    write(OUT / 'score_started.json', dict(created_utc=now(), prediction_seal_sha256=sha(OUT / 'prediction_seal.json')))
    rows = []
    for e in entries:
        source = e['region']
        with np.load(e['path'], allow_pickle=False) as z:
            raw = z['raw']
        positions, y = truth(frozen['archives'][source], source, frozen['score_keys'])
        for i, arm in enumerate(model.NEW_ARMS):
            for factor in FACTORS:
                point, ensemble, clip = common.scaled_paths(raw[i, positions], factor)
                metrics = common.score_arrays(point, ensemble, y)
                rows.extend(dict(region=source, origin=DAYS[j].isoformat(), arm=arm, factor=factor,
                    budget=model.BUDGET_BY_ARM[arm], final_spread_clipping_rate=float(clip[k]),
                    **{name: float(value[k]) for name, value in metrics.items()}) for k, j in enumerate(positions))
        print('Scored', source, len(positions), 'original development days', flush=True)
        del raw
    old = common.original_scores(frozen['baseline_path'])
    days = combine(pd.DataFrame(rows), old)
    region, macro, leave = common.aggregate(days)
    support = pd.read_csv(frozen['support_path'])
    support_metadata(support, frozen['score_keys'])
    tables = dict(zip(OUTPUTS[:5], (days, region, macro, leave, support)))
    for (name, table), count in zip(tables.items(), (790050, 4830, 690, 4830, 1267)):
        if len(table) != count or not np.isfinite(table.select_dtypes(include='number').to_numpy()).all():
            raise ValueError('Complete finite score tables required')
        table.to_csv(OUT / name, mode='x', index=False)
    write(OUT / 'summary.json', dict(new_fits=0, new_arms=75, inherited_arms=63, arms=138,
        new_day_rows=429375, inherited_day_rows_numeric_exact=360675, source_days=1145,
        prediction_queries=1267, raw_diagnostics=95025, fixed_factors=list(FACTORS), primary_factor=1.,
        method_innovation_claim=False, independent_confirmation=False, method_frozen=False, new_holdout_authorized=False))
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
