"""Immutable source-only fit, disjoint external calibration and legacy-dev scoring."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import pandas as pd
from scripts import disjoint_calibration_v50 as helper
from scripts import disjoint_point_v50 as point
from scripts import risk_baselines_v50 as risk
from scripts import monthly_transfer_v25 as monthly
from scripts import fleet_inventory_v33 as inventory
from scripts import run_hourly_bridge_v46 as common

sha, read, now, verify = common.sha, common.read, common.now, common.verify
SOURCES, EXTERNAL = helper.LEGACY, helper.EXTERNAL
CAL_DAYS, TEST_DAYS, LAGS, RULES = helper.CAL_DAYS, helper.TEST_DAYS, helper.LAGS, helper.RULES
BASES, BUDGETS, METHODS, METRICS = helper.BASES, helper.BUDGETS, risk.METHODS, helper.METRICS
HISTORY_DAYS = pd.date_range('2024-01-25', '2024-12-31', tz='UTC').as_unit('ns')
OUT = ROOT/'results/disjoint-calibration-v50'
PARENT = ROOT/'docs/evidence/source-risk-v49-verification.json'
PARENT_SHA = '8269659f1f9e5d4d6c06fca3a9b3b74c2299d9c6c3fd2242db5bdd0475335530'
CODE = tuple('scripts/'+name+'.py' for name in ('disjoint_point_v50', 'risk_baselines_v50',
    'disjoint_calibration_v50', 'run_disjoint_calibration_v50')) + tuple('tests/test_'+name+'.py' for name in
    ('disjoint_point_v50', 'risk_baselines_v50', 'disjoint_calibration_v50', 'run_disjoint_calibration_v50')) + (
    'docs/DISJOINT_CALIBRATION_V50_PROTOCOL.md',)
HELPERS = ('scripts/monthly_transfer_v25.py', 'scripts/conditional_law_v29.py',
    'scripts/coarse_supervision_v28.py', 'scripts/group_regression_v27.py', 'scripts/coarse_context_v30.py',
    'scripts/forward_context_v31.py', 'scripts/fleet_information_v34.py', 'scripts/fleet_inventory_v33.py',
    'scripts/run_hourly_bridge_v46.py', 'scripts/run_coarse_context_v30.py', 'scripts/run_local_sketch_v17.py',
    'scripts/local_sketch_v17.py', 'scripts/run_information_value_v10.py', 'scripts/native_center_v18.py',
    'src/transcif/research/fuel_transport_v7.py', 'src/transcif/research/dynamic_v2.py', 'src/transcif/research/eia_proxy.py')
REVIEWS = tuple(ROOT/f'results/disjoint-calibration-v50-{name}-review.md' for name in ('code', 'math'))


def to_jsonable(value: object) -> object:
    """Preserve audit content; unsupported/nonfinite objects are never hidden."""
    if isinstance(value,np.ndarray): return to_jsonable(value.tolist())
    if isinstance(value,np.generic): return to_jsonable(value.item())
    if isinstance(value,dict): return {k:to_jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [to_jsonable(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): raise ValueError('Nonfinite audit value; explicit q flag required')
    return value


def write(path: Path, value: dict) -> None:
    payload=to_jsonable(value)
    with path.open('x') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def inherited() -> dict[str, str]:
    verify({str(PARENT): PARENT_SHA})
    cursor, hashes, seen = read(PARENT), {}, set()
    if (cursor['status'] != 'VERIFICATION_COMPLETE' or not any(cursor.get(k) == 'PASS' for k in
            ('independent_formula_review_status', 'independent_actual_audit_status'))):
        raise ValueError('Fixed independently verified V49 parent required')
    while True:
        for kind in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
            for path, digest in cursor[kind].items():
                if path in hashes and hashes[path] != digest: raise ValueError('Conflicting inherited hash metadata')
                hashes[path] = digest
        if 'parent_manifest' not in cursor: break
        path, digest = cursor['parent_manifest'], cursor['parent_manifest_sha256']
        if path in seen: raise ValueError('Cyclic inherited metadata')
        seen.add(path); verify({path: digest}); cursor = read(path)
    return hashes


def external_support(paths: dict[str, str]) -> tuple[dict, dict]:
    if set(paths) != set(EXTERNAL): raise ValueError('Exactly ten external source paths required')
    keys = {s: helper.metadata(paths[s], s) for s in EXTERNAL}
    counts = {str(lag): {} for lag in LAGS}
    for source in EXTERNAL:
        dates = pd.DatetimeIndex(pd.to_datetime([key.split('/')[1] for key in keys[source]], utc=True)).as_unit('ns')
        if not len(dates): raise ValueError('Each external source requires calendar support')
        for lag in LAGS:
            counts[str(lag)][source] = int(helper.eligible(dates, lag).sum())
            if not counts[str(lag)][source]: raise ValueError('Every lag requires all ten nonempty external sources')
    return keys, counts


def proper_tasks(entries: list[dict], outer: str) -> list[dict]:
    expected = {(o, h) for o in SOURCES for h in SOURCES if o != h}
    if (outer not in SOURCES or len(entries) != 42 or {(e['outer'], e['held']) for e in entries} != expected):
        raise ValueError('Complete unique42 source-only tasks required')
    for e in entries:
        if e['pair'] != sorted((e['outer'], e['held'])) or e['label_region'] != e['held']:
            raise ValueError('Pair exclusion and held-only label identity required')
    return [next(e for e in entries if e['outer'] == outer and e['held'] == h) for h in SOURCES if h != outer]


def external_report(closure: dict[str,str], report_path: str) -> tuple[dict,dict]:
    """Read only a parent-bound inventory; its paths/hashes cannot extend the pool."""
    if report_path not in closure: raise ValueError('Parent-bound external support report required')
    verify({report_path:closure[report_path]})
    report=read(report_path)
    records=report['records']
    if (report['fixed_donor_sources']!=list(EXTERNAL) or report['metadata_only'] is not True
            or len(records)!=10 or {e['region'] for e in records}!=set(EXTERNAL)):
        raise ValueError('Exact ten-source parent inventory required')
    paths,hashes={},{report_path:closure[report_path]}
    for e in records:
        folder='external-v4-data' if e['region'] in EXTERNAL[:6] else 'external-v5-data'
        expected=str(ROOT/'results'/folder/(e['region']+'_hourly.csv'))
        if (e['status']!='present_metadata_verified' or e['path']!=expected
                or e['path'] not in closure or closure[e['path']]!=e['input_sha256']):
            raise ValueError('Every external path/hash must match fixed parent input metadata')
        paths[e['region']]=e['path'];hashes[e['path']]=closure[e['path']]
    return paths,hashes


def contract() -> dict:
    if point.BASES5!=BASES or point.BUDGETS!=BUDGETS or risk.EXTRA_NAMES!=helper.EXTRA_NAMES:
        raise ValueError('Shared point/risk/helper budget and feature registries required')
    closure = inherited()
    freeze_paths = {name: str(ROOT/f'results/{name}/freeze.json') for name in
                    ('conditional-law-v29', 'daily-bridge-v47', 'recompose-bridge-v48', 'source-risk-v49')}
    index_path = str(ROOT/'results/source-risk-v49/task_index.json')
    aliases = [*freeze_paths.values(), index_path]
    if any(path not in closure for path in aliases): raise ValueError('Bound parent freeze/index aliases required')
    verify({path: closure[path] for path in aliases})
    v29, v47, v48, v49 = [read(freeze_paths[name]) for name in freeze_paths]
    index = read(index_path)
    if (any(value['sources'] != list(SOURCES) for value in (v29,v47,v48,v49))
            or v48['origins'] != TEST_DAYS.asi8.tolist() or v49['origins'] != HISTORY_DAYS.asi8.tolist()):
        raise ValueError('Exact inherited legacy development grid required')
    score_keys = v48['score_keys']
    expected_keys = {f'{s}/{at:%Y-%m-%d}' for s in SOURCES for at in TEST_DAYS}
    if len(score_keys) != 1145 or len(set(score_keys)) != 1145 or not set(score_keys) <= expected_keys:
        raise ValueError('Exact1145 prelocked legacy score keys required')
    proper = {s: proper_tasks(index['entries'], s) for s in SOURCES}
    h1_keys = {s: [k for k in v49['label_keys'][s] if k.split('/')[1] < '2024-07-01'] for s in SOURCES}
    for source, keys in h1_keys.items():
        allowed = {f'{source}/{at:%Y-%m-%d}' for at in HISTORY_DAYS if at < pd.Timestamp('2024-07-01',tz='UTC')}
        if not keys or len(keys)!=len(set(keys)) or not set(keys)<=allowed:
            raise ValueError('Each proper source requires unique legal H1 keys')
    archives = {s: dict(v29=str(ROOT/f'results/conditional-law-v29/predictions/{s}.npz'),
        v29_audit=str(ROOT/f'results/conditional-law-v29/predictions/{s}.json'),
        v25=v29['monthly_prediction_paths'][s], v47=str(ROOT/f'results/daily-bridge-v47/predictions/{s}.npz'),
        v48=str(ROOT/f'results/recompose-bridge-v48/predictions/{s}.npz'), truth=v48['archives'][s]['truth'],
        size=str(ROOT/f'results/coarse-context-v30/predictions/{s}.json'),
        cap=str(ROOT/f'results/fleet-information-v34/predictions/{s}.json')) for s in SOURCES}
    legacy_monthly = v29['monthly_path']
    external_monthly = [str(ROOT/f'results/external-v{v}-data/monthly.csv') for v in (4, 5)]
    cap_path = str(ROOT/'results/donor-inventory-v41/capacity_op.csv')
    aliases += [legacy_monthly, *external_monthly, cap_path, *v29['query_paths'].values(),
        *[p for paths in v29['banks_by_outer'].values() for p in paths],
        *[p for entry in archives.values() for p in entry.values()], *v49['source_paths'].values(),
        *[str(ROOT/name) for name in HELPERS]]
    for entry in index['entries']:
        aliases += [entry[name] for name in ('path', 'risk_path', 'source_label_path', 'input_path')]
        if (closure.get(entry['path']) != entry['sha256'] or closure.get(entry['risk_path']) != entry['risk_sha256']):
            raise ValueError('Task index must agree with fixed parent prediction/risk hashes')
    if any(path not in closure for path in aliases): raise ValueError('All inherited direct aliases must be hash-bound')
    hashes = {path: closure[path] for path in set(aliases)} | {str(PARENT): PARENT_SHA}
    report_path = str(ROOT/'results/source-risk-v49-external-support.json')
    external_paths,external_hashes=external_report(closure,report_path)
    hashes.update(external_hashes)
    for path in REVIEWS:
        if not path.is_file() or path.is_symlink() or not path.read_text().strip():
            raise ValueError('Two actual overall prelaunch review reports required')
        hashes[str(path)] = sha(path)
    verify(hashes)
    cal_keys, lag_counts = external_support(external_paths)
    return dict(version=50, code_hashes={str(ROOT/name): sha(ROOT/name) for name in CODE}, input_hashes=hashes,
        environment=common.common.environment(), legacy_monthly=legacy_monthly, external_monthly=external_monthly,
        capacity_path=cap_path, external_paths=external_paths, source_paths=v49['source_paths'], archives=archives,
        query_paths=v29['query_paths'], banks_by_outer=v29['banks_by_outer'], proper_tasks=proper,
        H1_keys=h1_keys, cal_keys=cal_keys, lag_counts=lag_counts, score_keys=score_keys,
        sources=list(SOURCES), external=list(EXTERNAL), bases=list(BASES), budgets=list(BUDGETS), methods=list(METHODS),
        lags=list(LAGS), rules=list(RULES), cal_origins=CAL_DAYS.asi8.tolist(), test_origins=TEST_DAYS.asi8.tolist(),
        risk_parameters=risk.PARAMETERS, feature_names=risk.FEATURE_NAMES, sklearn_fits=448,
        target_values_opened=False, unseen_values_opened=False, independent_confirmation=False,
        method_innovation_claim=False, actual_first_release_verified=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink(): raise FileExistsError('Preserve V50 attempt')
    frozen = contract(); OUT.mkdir()
    write(OUT/'freeze.json', dict(created_utc=now(), **frozen))


def check() -> dict:
    frozen, expected = read(OUT/'freeze.json'), contract()
    # JSON normalizes tuples to lists, including feature-name whitelists.
    normalized = json.loads(json.dumps(to_jsonable(expected),allow_nan=False))
    if set(frozen) != set(normalized) | {'created_utc'} or any(frozen[k] != v for k, v in normalized.items()):
        raise ValueError('Frozen V50 code/input/role contract changed')
    return frozen


def monthly_rows(frozen: dict) -> tuple[list[dict], list[dict]]:
    def selected(path: str, allowed: tuple[str, ...]) -> list[dict]:
        with open(path, newline='') as stream:
            return [row for row in csv.DictReader(stream) if row['region'] in allowed]
    legacy = selected(frozen['legacy_monthly'], SOURCES)
    query = legacy.copy()
    for path in frozen['external_monthly']: query.extend(selected(path, EXTERNAL))
    return legacy, query


def capacity(path: str, source: str, origin: pd.Timestamp) -> tuple[np.ndarray, float]:
    if source not in (*SOURCES, *EXTERNAL): raise ValueError('Registered inventory source required')
    vintage = inventory.inventory_for_origin(origin)
    if vintage['year'] != origin.year-2 or pd.Timestamp(vintage['available_at']) > origin:
        raise ValueError('Available year-minus-two inventory required')
    with open(path, newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['region'] == source and r['year'] == str(vintage['year'])]
    if len(rows) != 1: raise ValueError('Unique registered source/year OP inventory required')
    values = helper.real([float(rows[0]['capacity_'+f+'_mw']) for f in helper.FUELS], (8,), 'OP capacities')
    pos = values[point.POS].sum()
    if np.any(values < 0) or not np.isfinite(values.sum()) or not np.isfinite(pos) or pos <= 0:
        raise ValueError('Nonnegative inventory and strictly positive finite POS total required')
    if (not np.isclose(float(rows[0]['capacity_pos_mw']), pos, atol=1e-9, rtol=1e-12)
            or not np.isclose(float(rows[0]['capacity_total_mw']), values.sum(), atol=1e-9, rtol=1e-12)):
        raise ValueError('Inventory totals must reconcile')
    return values[point.POS]/pos, float(pos)


def bank_query(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name].copy() for name in ('bank_regions', 'bank_days', 'bank_row_ids', 'bank_cutoff')}


def bank6(frozen: dict, outer: str) -> dict:
    banks = []
    for path in frozen['banks_by_outer'][outer]:
        with np.load(path, allow_pickle=False) as z:
            excluded,sources,regions=z['excluded'].tolist(),z['source_regions'].tolist(),z['regions']
            if (len(excluded)!=2 or len(set(excluded))!=2 or not set(excluded)<=set(SOURCES)
                    or excluded!=sorted(excluded) or outer not in excluded
                    or sources!=[s for s in SOURCES if s not in excluded]
                    or regions.ndim!=1 or set(regions)!=set(sources)):
                raise ValueError('Original pair bank must exclude outer before numeric loading')
            banks.append({name: z[name].copy() for name in z.files})
    return monthly.union_bank(*banks, bank_query(frozen['query_paths'][outer]), outer=outer)


def center_records(entry: dict, outer: str) -> list[dict]:
    audits = {name: json.loads(Path(entry[name]).read_text(), parse_float=point.FloatToken, parse_int=point.IntToken)
              for name in ('size', 'cap')}
    return point.decode_records(audits, outer)


def parity(entry: dict, query: dict, result: dict, ctx: dict) -> dict:
    errors = {}
    def grid(archive: Mapping[str,np.ndarray], require_keys: bool = False) -> None:
        if (not np.array_equal(archive['origins'],TEST_DAYS.asi8)
                or (require_keys and archive['keys'].tolist()!=[f"{query['region']}/{at:%Y-%m-%d}" for at in TEST_DAYS])):
            raise ValueError('Exact original181UTC date/key grid required')
    def close(name: str, a: np.ndarray, b: np.ndarray, tolerance: float) -> None:
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Parity shape/finite failure: '+name)
        error = float(abs(a-b).max()); errors[name] = error
        if error > tolerance: raise ValueError('Full181 fixed function parity failure: '+name)
    with np.load(entry['v25'], allow_pickle=False) as z:
        grid(z)
        close('MG_mix', query['MG_mix'], z['MG_mix'], 1e-12)
        close('MG_metadata', query['MG_metadata'], z['MG_metadata'], 1e-12)
    with np.load(entry['v29'], allow_pickle=False) as z:
        grid(z)
        names = z['arms'].tolist(); ia, ir = names.index('Scale-anchor'), names.index('Scale-recompose')
        old_raw=z['raw'];old_fine=z['fine_draws'][ia].copy()
        old_mg=z['MG_draws'];old_mean=z['query_mean']
        close('MG_draws', result['MG_draws'], old_mg, 1e-12)
        for key in ('donor_ids', 'fallback_ids', 'effective_pos_ids', 'effective_month_ids'):
            if not np.array_equal(result[key], z[key]): raise ValueError('Original literal sampler IDs required')
        for key in ('query_mean', 'query_scale', 'source_scale', 'scale_ratio'):
            close(key, result[key], old_mean if key=='query_mean' else z[key], 1e-9)
        scale=ctx['scale']
        for attribute,field in (('oof_mean','bank_oof_mean'),('oof_scale','bank_oof_scale'),
                ('oof_residual','bank_oof_residuals'),('oof_covariance3','bank_oof_covariance3'),
                ('oof_weights','bank_oof_weights')):
            close(attribute,getattr(scale,attribute),z[field],1e-9)
        if (not np.array_equal(scale.oof_leaf_ids,z['bank_oof_leaf_ids'])
                or scale.final.metadata['training_keys']!=z['training_keys'].tolist()):
            raise ValueError('Literal Scale OOF leaves and supervised training keys required')
        for attribute,field in (('X','training_X'),('y','training_ratios'),('weights','training_weights')):
            close('Scale_'+attribute,getattr(scale.final,attribute),z[field],1e-9)
        for key, old in (('A', old_raw[ia]), ('R', old_raw[ir]), ('fine', old_fine),
                         ('U', old_fine@point.EF), ('P', old_mg[...,0]*(old_mean@point.EF[point.POS]))):
            close(key, result[key], old, 1e-9)
        del old_raw,old_fine,old_mg,old_mean
    with np.load(entry['v47'], allow_pickle=False) as z:
        grid(z,True)
        close('V47_anchor', result['raw'][0], z['raw'][z['arms'].tolist().index('Scale-anchor')], 1e-9)
        for family in ('Size', 'SizeCap'):
            close('features_'+family, query['features'][family], z['query_feature_'+family], 1e-12)
        close('center_raw_EF',result['center_raw_ratios']@point.EF[point.POS],z['head_EF_raw'][[1,2]],1e-9)
        close('center_EF',result['center_EF'],z['EF_draws'][[1,2]],1e-9)
        close('center_projection',abs(result['center_raw_ratios']-result['center_ratios']).sum(-1),
              z['head_projection_l1'][[1,2]],1e-9)
    with np.load(entry['v48'], allow_pickle=False) as z:
        grid(z,True)
        names = z['arms'].tolist()
        if len(names)!=75 or len(set(names))!=75: raise ValueError('V48 cache must be exactly75new arms, not138combined')
        old_raw=z['raw']
        for i, name in enumerate(point.CACHE_NAMES[1:], 1):
            close('V48_'+name, result['raw'][i], old_raw[names.index(name)], 1e-9)
        del old_raw
    return errors


def proper(frozen: dict, outer: str) -> dict:
    pieces = []
    all_entries = [e for s in SOURCES for e in frozen['proper_tasks'][s]]
    for e in proper_tasks(all_entries, outer):
        held = e['held']
        with np.load(e['source_label_path'], allow_pickle=False) as z:
            label_keys, positions = z['keys'].tolist(), z['positions'].copy()
            keep = np.array([key.split('/')[1] < '2024-07-01' for key in label_keys])
            keys = [key for key, take in zip(label_keys, keep) if take]
            if (keys != frozen['H1_keys'][held] or not keys or not np.array_equal(z['origins'], HISTORY_DAYS.asi8[positions])):
                raise ValueError('Fixed held-only H1 label metadata required')
            take = positions[keep]
        with np.load(e['path'], allow_pickle=False) as z:
            if (not np.array_equal(z['origins'], HISTORY_DAYS.asi8)
                    or z['keys'].tolist() != [f'{held}/{at:%Y-%m-%d}' for at in HISTORY_DAYS]):
                raise ValueError('Full342 cached teacher prediction dates required')
            ids = [z['arms'].tolist().index(name) for name in BASES]
            raw = z['raw'][ids][:, take].copy()
        with np.load(e['input_path'], allow_pickle=False) as z:
            if (not np.array_equal(z['origins'], HISTORY_DAYS.asi8)
                    or z['keys'].tolist() != [f'{held}/{at:%Y-%m-%d}' for at in HISTORY_DAYS]):
                raise ValueError('Same held input/teacher keys required')
            features = {name: z['feature_'+name][take].copy() for name in ('Size', 'SizeCap')}
            if not np.array_equal(features['Size'][..., :5], z['MG_draws'][take]):
                raise ValueError('Literal sealed proper MG feature pairing required')
        packet = helper.feature_packet(raw, features)
        with np.load(e['risk_path'], allow_pickle=False) as z:
            if z['keys'].tolist() != label_keys: raise ValueError('Risk/label metadata support differs')
            ids = [z['arms'].tolist().index(name) for name in BASES]
            if not np.allclose(packet['point'], z['point'][ids][:, keep], atol=1e-9, rtol=0):
                raise ValueError('Postclip proper point must reproduce cached risk point')
        # The accepted reader filters timestamps before any generation conversion.
        # Only six held sources reach this call; neither H2 nor outer is converted.
        rows = common.source_days(Path(frozen['source_paths'][held]), '2024-01-25', '2024-07-01')
        if [f'{held}/{at:%Y-%m-%d}' for at, _ in rows] != keys:
            raise ValueError('Fresh H1 window support must match frozen label metadata')
        y = np.stack([(g/g.sum(1)[:, None])@helper.EF for _, g in rows])
        if (not np.isfinite(y).all() or np.any(y < 0) or np.any(y > 1000)
                or np.any(packet['point'] < 0) or np.any(packet['point'] > 1000)):
            raise ValueError('Physical proper point/truth in[0,1000] required')
        pieces.append(dict(packet=packet, truth=y, keys=keys, regions=np.repeat(held, len(keys))))
    return dict(X=[np.concatenate([part['packet']['X'][i] for part in pieces]) for i in range(5)],
        feature_names=pieces[0]['packet']['feature_names'], truth=np.concatenate([part['truth'] for part in pieces]),
        keys=[key for part in pieces for key in part['keys']], regions=np.concatenate([part['regions'] for part in pieces]),
        **{name: np.concatenate([part['packet'][name] for part in pieces], axis=1)
           for name in ('point', 'std_hour', 'std_level', 'std_shape')})


def _save(path: Path, **arrays: np.ndarray) -> None:
    with path.open('xb') as stream: np.savez_compressed(stream, **arrays)


def _entry(path: Path, **metadata: object) -> dict:
    return dict(path=str(path), sha256=sha(path), **metadata)


def query_cache(frozen: dict, rows: list[dict]) -> dict:
    """Seventeen source-only input laws, independent of outer; arrays are read-only."""
    def readonly(value: object) -> None:
        if isinstance(value,np.ndarray): value.setflags(write=False)
        elif isinstance(value,dict):
            for child in value.values(): readonly(child)
    cache={}
    for source in (*SOURCES,*EXTERNAL):
        dates=TEST_DAYS if source in SOURCES else CAL_DAYS
        q,c=capacity(frozen['capacity_path'],source,dates[0])
        cache[source]=(point.query(rows,source,dates,q,c),point.query(rows,source,dates))
        for value in cache[source]: readonly(value)
    return cache


def _query(ctx: dict, source: str, cache: dict) -> tuple[dict, dict, float]:
    value,b0_query=cache[source]
    prediction = point.predict(ctx, value)
    b0 = point.predict(ctx, b0_query)
    error = float(abs(b0['raw']-prediction['raw'][:3]).max())
    if error > 1e-9 or not np.allclose(b0_query['MG_draws'], value['MG_draws'], atol=1e-12, rtol=0):
        raise ValueError('B0 path functions must be independent of capacity features')
    return value, prediction, error


def _save_prediction(outer: str, source: str, value: dict, prediction: dict, models: list,
                     b0_error: float, replay: dict | None = None) -> dict:
    packet = helper.feature_packet(prediction['raw'], value['features'])
    r, keys = np.repeat(source, len(value['origins'])), value['keys'].tolist()
    initial = [model.predict(packet['X'][i], packet['point'][i], packet['std_hour'][i],
                packet['std_level'][i], packet['std_shape'][i], r, keys) for i, model in enumerate(models)]
    path = OUT/'predictions'/f'{outer}__{source}.npz'
    arrays = {name: array for name, array in value.items() if isinstance(array, np.ndarray)}
    arrays.update({'query_feature_'+name: array for name, array in value['features'].items()})
    arrays.update(raw=prediction['raw'], arms=np.array(BASES), budgets=np.array(BUDGETS), methods=np.array(METHODS),
        outer=np.array(outer), region=np.array(source),
        **{name: np.stack([result[name] for result in initial]) for name in ('lo0', 'hi0', 'scales', 'centers')},
        **{name: packet[name] for name in ('point', 'std_hour', 'std_level', 'std_shape')})
    _save(path, **arrays)
    audit_path = path.with_suffix('.json')
    write(audit_path, dict(outer=outer, query_region=source, point_audit=prediction['audit'],
        feature_audit=value['audit'], risk_audits=[result['audit'] for result in initial],
        risk_diagnostics=[result['diagnostics'] for result in initial], B0_capacity_independence_max_abs=b0_error,
        full181_parity=replay, external_calibration_response_opened=False, source2025_response_opened=False))
    return _entry(path, outer=outer, region=source, role='target' if source == outer else 'calibration',
                  audit_path=str(audit_path), audit_sha256=sha(audit_path))


def predict() -> None:
    frozen = check()
    if (OUT/'prediction_started.json').exists() or (OUT/'predictions').exists():
        raise FileExistsError('Preserve V50 prediction attempt')
    write(OUT/'prediction_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT/'freeze.json')))
    for folder in ('predictions', 'proper', 'models'): (OUT/folder).mkdir()
    legacy_rows, query_rows = monthly_rows(frozen)
    queries=query_cache(frozen,query_rows)
    entries, proper_entries, fits = [], [], 0
    for outer in SOURCES:
        records=center_records(frozen['archives'][outer], outer)
        ctx = point.fit(bank6(frozen, outer), legacy_rows, records, outer)
        if ctx['audit']['fit_count'] != 9: raise ValueError('Nine original point fits required')
        for actual,original in zip(ctx['audit']['center_fits'],records):
            for name in ('training_keys','training_regions',*point.DATE_FIELDS):
                if not np.array_equal(actual[name],original[name]): raise ValueError('Original center cohort metadata differs')
            for name in ('training_X','training_y','sample_weights','scaler_mean','scaler_scale',
                         * (('feature_importances',) if original['model']=='ET' else ('coefficients','intercept'))):
                if not np.allclose(np.asarray(actual[name],float),np.asarray(original[name],float),atol=1e-9,rtol=0):
                    raise ValueError('Original center training function reproduction differs: '+name)
        value, prediction, b0_error = _query(ctx,outer,queries)
        replay = parity(frozen['archives'][outer], value, prediction,ctx)
        data = proper(frozen, outer)
        models = [risk.fit_baselines(data['X'][i], data['point'][i], data['truth'], data['std_hour'][i],
            data['std_level'][i], data['std_shape'][i], data['regions'], data['keys'], outer=outer,
            budget=BUDGETS[i], feature_names=data['feature_names'][i]) for i in range(5)]
        if any(m.audit['fit_count'] != 11 or m.audit['synthetic_parameters_override'] for m in models):
            raise ValueError('Five risk fits must use formal fixed11HGB profiles')
        fits += 9+sum(m.audit['fit_count'] for m in models)
        path = OUT/'proper'/f'{outer}.npz'
        _save(path, truth=data['truth'], keys=np.array(data['keys']), regions=data['regions'],
            **{name: data[name] for name in ('point', 'std_hour', 'std_level', 'std_shape')},
            **{'X_'+str(i): data['X'][i] for i in range(5)},
            **{'feature_names_'+str(i): np.array(data['feature_names'][i]) for i in range(5)})
        model_path = OUT/'models'/f'{outer}.pkl'
        with model_path.open('xb') as stream: pickle.dump(dict(point=ctx, risk=models), stream, protocol=5)
        audit_path = OUT/'proper'/f'{outer}.json'
        write(audit_path, dict(outer=outer, point_fit=ctx['audit'], risk_fits=[m.audit for m in models],
            proper_source_H1_labels_used=True, proper_H2_or_outer_labels_used=False,
            base_teacher_third_source_excluded_for_risk_mean_OOF=False))
        proper_entries.append(_entry(path, outer=outer, model_path=str(model_path), model_sha256=sha(model_path),
                                      audit_path=str(audit_path), audit_sha256=sha(audit_path)))
        entries.append(_save_prediction(outer, outer, value, prediction, models, b0_error, replay))
        for source in EXTERNAL:
            value, prediction, b0_error = _query(ctx,source,queries)
            entries.append(_save_prediction(outer, source, value, prediction, models, b0_error))
        print('Predicted', outer, '11 query archives,64 fits; external responses unopened', flush=True)
    if fits != 448 or len(entries) != 77: raise ValueError('Exact448fits and77archives required')
    check()
    write(OUT/'prediction_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT/'freeze.json'),
        prediction_started_sha256=sha(OUT/'prediction_started.json'), entries=entries, proper=proper_entries,
        complete_grid=True, sklearn_fits=fits, proper_source_H1_labels_used=True,
        external_calibration_response_opened=False, source2025_response_opened=False))


def predictions() -> list[dict]:
    check(); seal = read(OUT/'prediction_seal.json')
    entries, proper_entries = seal.get('entries', []), seal.get('proper', [])
    expected = {(o, s) for o in SOURCES for s in (o, *EXTERNAL)}
    if (seal.get('complete_grid') is not True or len(entries) != 77
            or {(e['outer'], e['region']) for e in entries} != expected
            or len(proper_entries) != 7 or {e['outer'] for e in proper_entries} != set(SOURCES)):
        raise ValueError('Complete77 predictions and7proper/model bundles required before responses')
    if (seal['freeze_sha256'] != sha(OUT/'freeze.json')
            or seal['prediction_started_sha256'] != sha(OUT/'prediction_started.json')
            or seal['sklearn_fits'] != 448 or seal['external_calibration_response_opened'] is not False
            or seal['source2025_response_opened'] is not False):
        raise ValueError('Prediction chain and closed response interfaces required')
    hashes = {}
    for e in entries:
        expected_path = OUT/'predictions'/f"{e['outer']}__{e['region']}.npz"
        if e['path'] != str(expected_path) or e['audit_path'] != str(expected_path.with_suffix('.json')):
            raise ValueError('Registered prediction output paths required')
        hashes.update({e['path']:e['sha256'], e['audit_path']:e['audit_sha256']})
    for e in proper_entries:
        if (e['path'] != str(OUT/'proper'/f"{e['outer']}.npz")
                or e['model_path'] != str(OUT/'models'/f"{e['outer']}.pkl")
                or e['audit_path'] != str(OUT/'proper'/f"{e['outer']}.json")):
            raise ValueError('Registered self-generated proper/model output paths required')
        hashes.update({e['path']:e['sha256'], e['audit_path']:e['audit_sha256'], e['model_path']:e['model_sha256']})
    for folder,expected_names in (
            ('predictions',{f'{o}__{s}.{suffix}' for o,s in expected for suffix in ('npz','json')}),
            ('proper',{f'{s}.{suffix}' for s in SOURCES for suffix in ('npz','json')}),
            ('models',{f'{s}.pkl' for s in SOURCES})):
        if {p.name for p in (OUT/folder).iterdir()}!=expected_names:
            raise ValueError('Exact registered prediction/proper/model folder contents required')
    verify(hashes)
    for e in entries:
        dates = TEST_DAYS if e['region'] == e['outer'] else CAL_DAYS
        n = len(dates)
        with np.load(e['path'], allow_pickle=False) as z:
            if (e['role']!=('target' if e['region']==e['outer'] else 'calibration')
                    or z['arms'].tolist() != list(BASES) or z['budgets'].tolist() != list(BUDGETS)
                    or z['methods'].tolist() != list(METHODS) or not np.array_equal(z['origins'], dates.asi8)
                    or z['keys'].tolist() != [f"{e['region']}/{at:%Y-%m-%d}" for at in dates]
                    or z['outer'].item() != e['outer'] or z['region'].item() != e['region']):
                raise ValueError('Complete registered query identity/grid required')
            for name, shape in (('raw',(5,n,128,24)), ('lo0',(5,6,n,24)), ('hi0',(5,6,n,24)),
                                ('scales',(5,6,n,24)), ('centers',(5,6,n,24)), ('point',(5,n,24)),
                                ('std_hour',(5,n,24)), ('std_level',(5,n)), ('std_shape',(5,n))):
                helper.real(z[name], shape, 'sealed '+name)
            if np.any(z['scales'] <= 0) or np.any(z['lo0'] > z['hi0']):
                raise ValueError('Positive scales and ordered initial bounds required')
            if np.any(z['point']<0) or np.any(z['point']>1000): raise ValueError('Postclip sealed point must be physical')
    return entries


def quantile_record(value: dict, outer: str, base: int, method: int, lag: int) -> dict:
    record = dict(value)
    infinite = bool(np.isinf(record['q']))
    record.update(q=None if infinite else float(record['q']), is_infinite=infinite,q_is_infinite=infinite,
                  outer=outer, base=BASES[base], budget=BUDGETS[base], method=METHODS[method], lag=lag)
    return record


def calibrate() -> None:
    frozen = check(); entries = predictions()
    if (OUT/'calibration_started.json').exists() or (OUT/'calibration').exists():
        raise FileExistsError('Preserve V50 calibration attempt')
    write(OUT/'calibration_started.json', dict(created_utc=now(), prediction_seal_sha256=sha(OUT/'prediction_seal.json')))
    for folder in ('calibration', 'external_labels'): (OUT/folder).mkdir()
    labels, label_entries = {}, []
    for source in EXTERNAL:
        positions, y = helper.external_truth(frozen['external_paths'][source], source, frozen['cal_keys'][source])
        labels[source] = positions, y
        path = OUT/'external_labels'/f'{source}.npz'
        _save(path, origins=CAL_DAYS.asi8[positions], keys=np.array(frozen['cal_keys'][source]), truth=y, positions=positions)
        label_entries.append(_entry(path, region=source))
    lookup = {(e['outer'],e['region']):e for e in entries}
    records, final_entries = [], []
    for outer in SOURCES:
        source_scores = {}
        for source in EXTERNAL:
            positions, truth = labels[source]
            with np.load(lookup[outer,source]['path'], allow_pickle=False) as z:
                lo, hi, s = (np.take(z[name], positions, axis=2) for name in ('lo0','hi0','scales'))
                source_scores[source] = np.stack([risk.scores(lo[i],hi[i],s[i],truth) for i in range(5)])
        with np.load(lookup[outer,outer]['path'], allow_pickle=False) as z:
            target = {name:z[name].copy() for name in ('lo0','hi0','scales')}
        lower, upper = np.empty((5,6,3,2,181,24)), np.empty((5,6,3,2,181,24))
        for li, lag in enumerate(LAGS):
            selected, regions = [], []
            for source in EXTERNAL:
                keep = helper.eligible(CAL_DAYS[labels[source][0]],lag)
                if not keep.any(): raise ValueError('Every formal lag requires all ten calibration sources')
                selected.append(source_scores[source][:,:,keep])
                regions.extend([source]*int(keep.sum()))
            score = np.concatenate(selected,axis=2)
            for b in range(5):
                for m in range(6):
                    for ri,rule in enumerate(RULES):
                        value = risk.calibrate(score[b,m],regions,rule)
                        if value['K'] != 10: raise ValueError('Formal calibration must use exactly10external sources')
                        if value['source_counts']!=frozen['lag_counts'][str(lag)]:
                            raise ValueError('Calibration source counts must match prelocked lag metadata')
                        record = quantile_record(value,outer,b,m,lag); records.append(record)
                        final = risk.final_bounds(target['lo0'][b,m],target['hi0'][b,m],target['scales'][b,m],value['q'])
                        lower[b,m,li,ri],upper[b,m,li,ri] = final['lo'],final['hi']
        path = OUT/'calibration'/f'{outer}.npz'
        _save(path,lo=lower,hi=upper,origins=TEST_DAYS.asi8,keys=np.array([f'{outer}/{at:%Y-%m-%d}' for at in TEST_DAYS]),
              bases=np.array(BASES),methods=np.array(METHODS),lags=np.array(LAGS),rules=np.array(RULES),budgets=np.array(BUDGETS))
        final_entries.append(_entry(path,outer=outer))
        print('Calibrated',outer,'all180fixed configurations;2025responses unopened',flush=True)
    if len(records) != 1260: raise ValueError('Exact1260quantile records required')
    write(OUT/'calibration.json',dict(records=records,risk_refits=0,source2025_response_opened=False))
    table = pd.DataFrame([{**r,'source_counts':json.dumps(r['source_counts'],sort_keys=True)} for r in records])
    table.to_csv(OUT/'calibration.csv',index=False,mode='x')
    check()
    write(OUT/'calibration_seal.json',dict(created_utc=now(),freeze_sha256=sha(OUT/'freeze.json'),
        prediction_seal_sha256=sha(OUT/'prediction_seal.json'),
        calibration_started_sha256=sha(OUT/'calibration_started.json'),entries=final_entries,labels=label_entries,
        quantile_json_sha256=sha(OUT/'calibration.json'),quantile_csv_sha256=sha(OUT/'calibration.csv'),
        complete_grid=True,risk_refits=0,source2025_response_opened=False))


def calibration() -> list[dict]:
    predictions(); seal = read(OUT/'calibration_seal.json')
    if (seal.get('complete_grid') is not True or len(seal.get('entries',[])) != 7
            or {e['outer'] for e in seal['entries']} != set(SOURCES) or len(seal.get('labels',[])) != 10
            or {e['region'] for e in seal['labels']} != set(EXTERNAL) or seal['risk_refits'] != 0
            or seal['source2025_response_opened'] is not False
            or seal['freeze_sha256'] != sha(OUT/'freeze.json')
            or seal['prediction_seal_sha256'] != sha(OUT/'prediction_seal.json')
            or seal['calibration_started_sha256'] != sha(OUT/'calibration_started.json')):
        raise ValueError('Complete external calibration and final bounds seal required before2025truth')
    hashes = {str(OUT/'calibration.json'):seal['quantile_json_sha256'],str(OUT/'calibration.csv'):seal['quantile_csv_sha256']}
    for e in seal['entries']+seal['labels']:
        expected = OUT/'calibration'/f"{e['outer']}.npz" if 'outer' in e else OUT/'external_labels'/f"{e['region']}.npz"
        if e['path'] != str(expected): raise ValueError('Registered sealed response/finalbounds output paths required')
        hashes[e['path']] = e['sha256']
    verify(hashes)
    records = read(OUT/'calibration.json')['records']
    expected={(o,b,m,lag,rule) for o in SOURCES for b in BASES for m in METHODS for lag in LAGS for rule in RULES}
    if len(records)!=1260 or {(r['outer'],r['base'],r['method'],r['lag'],r['rule']) for r in records}!=expected:
        raise ValueError('Complete1260unique fixed quantile registry required')
    frozen=read(OUT/'freeze.json')
    for record in records:
        if (record['budget']!=BUDGETS[BASES.index(record['base'])] or record['K']!=10
                or record['source_counts']!=frozen['lag_counts'][str(record['lag'])]
                or abs(record['finite_level']-(.9 if record['rule']=='equal_region' else .99))>1e-12
                or record['q_is_infinite']!=record['is_infinite']
                or record['is_infinite']!=(record['q'] is None)
                or (record['q'] is not None and (not np.isfinite(record['q']) or record['q']<0))):
            raise ValueError('Quantile budget/count/level/finite encoding differs from fixed calibration')
    for e in seal['entries']:
        with np.load(e['path'],allow_pickle=False) as z:
            if (z['bases'].tolist()!=list(BASES) or z['methods'].tolist()!=list(METHODS)
                    or z['budgets'].tolist()!=list(BUDGETS)
                    or z['lags'].tolist()!=list(LAGS) or z['rules'].tolist()!=list(RULES)
                    or not np.array_equal(z['origins'],TEST_DAYS.asi8)
                    or z['keys'].tolist()!=[f"{e['outer']}/{at:%Y-%m-%d}" for at in TEST_DAYS]):
                raise ValueError('Exact final configuration/key grid required')
            lo,hi=(helper.real(z[name],(5,6,3,2,181,24),'final '+name) for name in ('lo','hi'))
            if np.any(lo<0) or np.any(hi>1000) or np.any(lo>hi): raise ValueError('Finite physical ordered final intervals required')
    return seal['entries']


def tables(day: pd.DataFrame, configurations: list[str]) -> tuple[pd.DataFrame,...]:
    region=day.groupby(['region',*configurations],sort=False)[list(METRICS)].mean().reset_index()
    macro=region.groupby(configurations,sort=False)[list(METRICS)].mean().reset_index()
    leave=[]
    for source in SOURCES:
        table=region[region.region!=source].groupby(configurations,sort=False)[list(METRICS)].mean().reset_index()
        table.insert(0,'excluded_region',source);leave.append(table)
    return region,macro,pd.concat(leave,ignore_index=True)


def score() -> None:
    frozen=check(); finals=calibration()
    if (OUT/'score_started.json').exists() or (OUT/'completed.json').exists():
        raise FileExistsError('Preserve V50 score attempt')
    write(OUT/'score_started.json',dict(created_utc=now(),calibration_seal_sha256=sha(OUT/'calibration_seal.json')))
    rows,raw_rows,support=[],[],[]
    for e in finals:
        source=e['outer']
        available=common.source_days(Path(frozen['archives'][source]['truth']),'2025-01-01','2025-07-01')
        keys=[f'{source}/{at:%Y-%m-%d}' for at,_ in available]
        expected=[k for k in frozen['score_keys'] if k.startswith(source+'/')]
        if keys!=expected: raise ValueError('Actual2025source support must equal fixed V48keys; do not drop days')
        take=np.array([TEST_DAYS.get_loc(at) for at,_ in available],int)
        y=np.stack([(g/g.sum(1)[:,None])@helper.EF for _,g in available])
        with np.load(e['path'],allow_pickle=False) as z:
            metrics=helper.interval_metrics(np.take(z['lo'],take,axis=4),np.take(z['hi'],take,axis=4),y)
        for b,base in enumerate(BASES):
            for m,method in enumerate(METHODS):
                for li,lag in enumerate(LAGS):
                    for ri,rule in enumerate(RULES):
                        rows.extend(dict(region=source,origin=at.isoformat(),base=base,budget=BUDGETS[b],method=method,
                            lag=lag,rule=rule,**{k:float(v[b,m,li,ri,j]) for k,v in metrics.items()}) for j,(at,_) in enumerate(available))
        with np.load(OUT/'predictions'/f'{source}__{source}.npz',allow_pickle=False) as z: raw=z['raw'][:,take].copy()
        for factor in helper.FACTORS:
            lo,hi=helper.raw_intervals(raw,factor);metrics=helper.interval_metrics(lo,hi,y)
            raw_rows.extend(dict(region=source,origin=at.isoformat(),base=base,budget=BUDGETS[b],factor=factor,
                **{k:float(v[b,j]) for k,v in metrics.items()}) for b,base in enumerate(BASES) for j,(at,_) in enumerate(available))
        support.extend(dict(region=source,origin=at.isoformat(),available=f'{source}/{at:%Y-%m-%d}' in set(keys)) for at in TEST_DAYS)
    day,raw_day=pd.DataFrame(rows),pd.DataFrame(raw_rows)
    regional,macro,leave=tables(day,['base','budget','method','lag','rule'])
    raw_region,raw_macro,_=tables(raw_day,['base','budget','factor'])
    outputs=[('day_metrics.csv',day,206100),('region_metrics.csv',regional,1260),('macro_metrics.csv',macro,180),
        ('leave_one_source_out_macro.csv',leave,1260),('raw_day_metrics.csv',raw_day,28625),
        ('raw_region_metrics.csv',raw_region,175),('raw_macro_metrics.csv',raw_macro,25),('label_support.csv',pd.DataFrame(support),1267)]
    for name,table,count in outputs:
        if len(table)!=count or not np.isfinite(table.select_dtypes(include='number').to_numpy()).all():
            raise ValueError('Fixed finite V50score/aggregation counts required: '+name)
        table.to_csv(OUT/name,index=False,mode='x')
    write(OUT/'summary.json',dict(sklearn_fits=448,point_ET_fits=56,point_Ridge_fits=7,risk_HGB_fits=385,
        risk_calibration_refits=0,external_calibration_sources=10,source2025_score_days=1145,configurations=180,
        raw_reference_curves=25,interval_CRPS_fabricated=False,previously_viewed_development=True,
        independent_confirmation=False,method_innovation_claim=False,actual_first_release_verified=False,
        risk_mean_OOF_is_base_teacher_third_source_exclusion=False,
        target_values_opened=False,unseen_values_opened=False,publication_lags=list(LAGS)))
    check()
    write(OUT/'completed.json',dict(created_utc=now(),status='COMPLETE',freeze_sha256=sha(OUT/'freeze.json'),
        prediction_seal_sha256=sha(OUT/'prediction_seal.json'),calibration_seal_sha256=sha(OUT/'calibration_seal.json'),
        score_started_sha256=sha(OUT/'score_started.json'),
        output_hashes={str(OUT/name):sha(OUT/name) for name in [*(x[0] for x in outputs),'summary.json']},
        target_values_opened=False,unseen_values_opened=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','predict','calibrate','score'))
    args=parser.parse_args()
    {'prepare':prepare,'predict':predict,'calibrate':calibrate,'score':score}[args.phase]()
