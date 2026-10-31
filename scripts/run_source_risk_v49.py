"""Build pair-excluded historical hourly risk examples, without 2025 scoring."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import source_risk_centers_v49 as centers
from scripts import source_risk_scale_v49 as scale
from scripts import monthly_transfer_v25 as monthly
from scripts import fleet_inventory_v33 as inventory
from scripts import run_hourly_bridge_v46 as common

sha, read, write, now, verify = common.sha, common.read, common.write, common.now, common.verify
SOURCES, DAYS, ARMS, BUDGETS = centers.SOURCES, centers.DAYS, centers.ARMS, centers.BUDGETS
PAIRS = tuple(itertools.combinations(SOURCES, 2))
H1_COUNTS = dict(zip(SOURCES, (157, 158, 158, 153, 158, 158, 158)))
H2_COUNTS = dict(zip(SOURCES, (180, 181, 69, 180, 26, 182, 182)))
COUNTS = {s: H1_COUNTS[s]+H2_COUNTS[s] for s in SOURCES}
OUT = ROOT / 'results/source-risk-v49'
PARENT = ROOT / 'docs/evidence/recompose-bridge-v48-verification.json'
PARENT_SHA = '2790acfbf96e24b37d1baa2f45a6431ccf247a8e8f71c482d0eab114cea59b64'
CODE = ('scripts/source_risk_scale_v49.py', 'scripts/source_risk_centers_v49.py', 'scripts/run_source_risk_v49.py',
        'tests/test_source_risk_scale_v49.py', 'tests/test_source_risk_centers_v49.py',
        'tests/test_run_source_risk_v49.py', 'docs/SOURCE_RISK_V49_PROTOCOL.md')
HELPERS = ('scripts/monthly_transfer_v25.py', 'scripts/conditional_law_v29.py', 'scripts/coarse_supervision_v28.py',
           'scripts/group_regression_v27.py', 'scripts/coarse_context_v30.py', 'scripts/forward_context_v31.py',
           'scripts/fleet_information_v34.py', 'scripts/fleet_inventory_v33.py', 'scripts/run_hourly_bridge_v46.py',
           'scripts/run_coarse_context_v30.py', 'scripts/run_local_sketch_v17.py', 'scripts/local_sketch_v17.py',
           'scripts/run_information_value_v10.py', 'scripts/native_center_v18.py',
           'src/transcif/research/fuel_transport_v7.py', 'src/transcif/research/dynamic_v2.py',
           'src/transcif/research/eia_proxy.py')
REVIEWS = tuple(ROOT / f'results/source-risk-v49-{kind}-review-r1.md' for kind in ('code', 'math'))
DIAGNOSTICS = (*common.METRICS, 'mc_daily_mean_std_raw', 'mc_centered_shape_rms_raw')


class FloatToken(str):
    """Retain a JSON floating token without converting training numerics."""


class IntToken(str):
    """Retain a JSON integer token, distinct from a floating parameter."""


def inherited() -> dict:
    verify({str(PARENT): PARENT_SHA})
    cursor = read(PARENT)
    if cursor['status'] != 'VERIFICATION_COMPLETE' or cursor['independent_formula_review_status'] != 'PASS':
        raise ValueError('Independent V48 PASS required')
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
        if p in seen: raise ValueError('Cyclic inherited metadata')
        seen.add(p); verify({p: h}); cursor = read(p)
    return result


def availability(path: str, source: str) -> list[str]:
    """Only timestamp, region and validity are converted during preparation."""
    with open(path, newline='') as stream:
        rows = [{k: row[k] for k in ('time', 'region', 'valid')} for row in csv.DictReader(stream)
                if '2024-01-25' <= row['time'] < '2025-01-01']
    if not rows: raise ValueError('Source has no historical risk metadata')
    table = pd.DataFrame(rows)
    dates = pd.DatetimeIndex(pd.to_datetime(table.time, utc=True, errors='raise')).as_unit('ns')
    if (dates.has_duplicates or not dates.equals(dates.floor('h')) or not table.region.eq(source).all()
            or not set(table.valid) <= {'True', 'False'}):
        raise ValueError('Unique source UTC hour metadata required')
    answer = []
    for at in DAYS:
        take = np.flatnonzero(dates.floor('D') == at)
        if len(take) != 24 or not table.iloc[take].valid.eq('True').all(): continue
        if not np.array_equal(np.sort(dates[take].asi8), pd.date_range(at, periods=24, freq='h').asi8):
            raise ValueError('Exact UTC daily grid required')
        answer.append(f'{source}/{at:%Y-%m-%d}')
    if (len(answer) != COUNTS[source] or sum(k[-5:-3] <= '06' for k in answer) != H1_COUNTS[source]):
        raise ValueError('Predeclared source support count changed')
    return answer


def contract() -> dict:
    closure = inherited()
    registry = str(ROOT / 'results/native-hindcast-v20r1/freeze.json')
    if registry not in closure: raise ValueError('Inherited bank metadata registry required')
    verify({registry: closure[registry]})
    lookup = {tuple(e['excluded']): e['path'] for e in read(registry)['banks']}
    if set(lookup) != set(PAIRS) or len(lookup) != 21: raise ValueError('Exact 21 pair banks required')
    monthly_path = str(ROOT / 'results/heldout-2026h1-data/monthly.csv')
    capacity_path = str(ROOT / 'results/fleet-inventory-v33/capacity_op.csv')
    sources = {s: str(ROOT / f'results/heldout-2026h1-data/{s}_hourly.csv') for s in SOURCES}
    heads = {s: dict(size=str(ROOT / f'results/coarse-context-v30/predictions/{s}.json'),
                     cap=str(ROOT / f'results/fleet-information-v34/predictions/{s}.json')) for s in SOURCES}
    aliases = [registry, monthly_path, capacity_path, *lookup.values(), *sources.values(),
               *[p for entry in heads.values() for p in entry.values()], *[str(ROOT / p) for p in HELPERS]]
    if any(p not in closure for p in aliases): raise ValueError('Bound direct dependency aliases required')
    inputs = {p: closure[p] for p in aliases} | {str(PARENT): PARENT_SHA}
    for p in REVIEWS:
        if not p.is_file() or p.is_symlink() or not p.read_text().strip(): raise ValueError('Both final prelaunch reports required')
        inputs[str(p)] = sha(p)
    verify(inputs)
    keys = {s: availability(p, s) for s, p in sources.items()}
    return dict(version=49, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE}, input_hashes=inputs,
        environment=common.common.environment(), sources=list(SOURCES), source_paths=sources,
        banks=[dict(pair=list(pair), path=lookup[pair]) for pair in PAIRS], head_paths=heads,
        monthly_path=monthly_path, capacity_path=capacity_path, label_keys=keys, origins=DAYS.asi8.tolist(),
        arms=list(ARMS), budgets=list(BUDGETS), sklearn_fits=168, Scale_ET_fits=126, center_ET_fits=21,
        center_Ridge_fits=21, risk_models_fitted=False, source_truth_days=2100,
        target_values_opened=False, unseen_values_opened=False, source_2025_truth_read=False,
        independent_confirmation=False, method_innovation_claim=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink(): raise FileExistsError('Preserve V49 attempt')
    frozen = contract(); OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared V49: 21 pair teachers, 42 tasks, 2100 valid source days; no 2025 labels', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('Frozen V49 contract changed')
    return frozen


def visible_months(rows: list[dict], source: str) -> pd.DataFrame:
    chosen, seen = [], set()
    for row in rows:
        if row['region'] != source: continue
        start, end, release = (centers.coarse._utc(row[k]) for k in ('period_start', 'period_end', 'available_at'))
        if start >= DAYS[-1] or end > DAYS[-1] or release > DAYS[-1]: continue
        if start.day != 1 or start != start.normalize() or end != start+pd.offsets.MonthBegin(1) or release < end:
            raise ValueError('Valid released whole monthly coarse record required')
        if start in seen: raise ValueError('Unique local coarse vintage required')
        seen.add(start)
        value = centers.coarse._visible(row)
        chosen.append(dict(region=source, period_start=start, period_end=end, available_at=release,
                           **dict(zip(monthly.MG_COLUMNS, value))))
    if not chosen: raise ValueError('Eligible local coarse history required')
    return pd.DataFrame(chosen)


def capacity(path: str, source: str) -> tuple[np.ndarray, float]:
    release = inventory.inventory_for_origin(DAYS[0].to_pydatetime())
    if release['year'] != 2022 or pd.Timestamp(release['available_at']) > DAYS[0]:
        raise ValueError('Legal year-minus-two historical capacity required')
    with open(path, newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['region'] == source and r['year'] == '2022']
    if len(rows) != 1: raise ValueError('Unique historical source capacity row required')
    c = np.array([float(rows[0]['capacity_'+fuel+'_mw']) for fuel in centers.coarse.FUELS])
    pos = c[centers.POS].sum()
    if not np.isfinite(c).all() or np.any(c < 0) or pos <= 0: raise ValueError('Positive finite POS capacity required')
    return c[centers.POS]/pos, float(pos)


def center_records(paths: dict, pair: tuple[str, str]) -> list[dict]:
    """Decode numeric audit tokens as text; convert only metadata before pair masking."""
    answer = []
    for name, arm in zip(('size', 'cap'), centers.HEADS):
        audit = json.loads(Path(paths[name]).read_text(), parse_float=FloatToken, parse_int=IntToken)
        if audit['region'] != pair[0]: raise ValueError('Canonical first-source audit required')
        selected = [f for f in audit['fits'] if f['arm'] == arm]
        if len(selected) != 1: raise ValueError('Unique fixed center fit required')
        record = dict(selected[0]); method = arm.split('-')[1]
        params = record['parameters']; expected = centers.forward.PARAMETERS[method]
        if set(params) != set(expected): raise ValueError('Original parameter keys required')
        converted = {}
        for k, value in expected.items():
            raw = params[k]
            if isinstance(value, bool):
                if type(raw) is not bool: raise ValueError('Boolean model parameter required')
                converted[k] = raw
            else:
                token_type = FloatToken if type(value) is float else IntToken
                if type(raw) is not token_type: raise ValueError('Original numeric parameter token type required')
                converted[k] = type(value)(raw)
        record['parameters'] = converted
        for field in centers.DATE_FIELDS: record[field] = [int(v) for v in record[field]]
        answer.append(record)
    return answer


def bank(path: str, pair: tuple[str, str]) -> dict:
    with np.load(path, allow_pickle=False) as z:
        excluded, source_regions, regions = z['excluded'], z['source_regions'], z['regions']
        expected = tuple(s for s in SOURCES if s not in pair)
        if (not np.array_equal(excluded, pair) or not np.array_equal(source_regions, expected)
                or set(regions) != set(expected)):
            raise ValueError('Pair provenance before bank fine fields required')
        result = {k: z[k].copy(order='K') for k in ('excluded', 'source_regions', 'regions', 'days', 'row_ids',
                    'cutoff', 'months', 'q', 'support', 'raw_cif', 'donor_mix', 'monthmean')}
    scale.validate_pair_bank(result, pair)
    return result


def tasks_for_outer(entries: list[dict], outer: str) -> list[dict]:
    if outer not in SOURCES or len(entries) != 42: raise ValueError('Registered outer and all 42 directed tasks required')
    expected = {(o, h) for o in SOURCES for h in SOURCES if o != h}
    actual = {(e['outer'], e['held']) for e in entries}
    if actual != expected or len(actual) != len(entries): raise ValueError('Complete unique directed task registry required')
    for e in entries:
        if (tuple(e['pair']) != tuple(sorted((e['outer'], e['held'])))
                or e['label_region'] != e['held']):
            raise ValueError('Task exclusion pair and held-only label identity required')
    selected = [e for e in entries if e['outer'] == outer]
    if len(selected) != 6 or {e['held'] for e in selected} != set(SOURCES)-{outer}:
        raise ValueError('Exactly six nonreverse held-label tasks required')
    return selected


def predict() -> None:
    frozen = check()
    if (OUT / 'prediction_started.json').exists() or (OUT / 'predictions').exists():
        raise FileExistsError('Preserve V49 prediction attempt')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    for folder in ('inputs', 'pair_models', 'predictions'): (OUT / folder).mkdir()
    with open(frozen['monthly_path'], newline='') as stream: rows = list(csv.DictReader(stream))
    source_inputs, queries, laws = [], {}, {}
    for source in SOURCES:
        law = monthly.coarse_law(visible_months(rows, source), source, DAYS, budget='MG', samples=128, seed=17023)
        q, c = capacity(frozen['capacity_path'], source)
        query = centers.features(rows, source, law['draws'], q, c)
        arrays = {k: v for k, v in query.items() if isinstance(v, np.ndarray)}
        arrays.update({'feature_' + k: v for k, v in query['features'].items()})
        arrays.update(MG_mix=law['mix'], MG_metadata=law['metadata'], origins=DAYS.asi8,
                      keys=np.array([f'{source}/{at:%Y-%m-%d}' for at in DAYS]), capacity_q=q, capacity_pos_mw=c)
        path = OUT / 'inputs' / (source + '.npz')
        with path.open('xb') as f: np.savez_compressed(f, **arrays)
        audit = path.with_suffix('.json')
        write(audit, dict(law['audit'], region=source, capacity_year=2022,
            capacity_available_at=inventory.inventory_for_origin(DAYS[0].to_pydatetime())['available_at'],
            query_local_hidden_fine_read=False, authentic_online_hindcast=False))
        source_inputs.append(dict(region=source, path=str(path), sha256=sha(path), audit_path=str(audit), audit_sha256=sha(audit)))
        queries[source], laws[source] = query, law
    entries, models, total_fits = [], [], 0
    for b in frozen['banks']:
        pair = tuple(b['pair']); name = '__'.join(pair)
        fitted_scale = scale.fit_scale(bank(b['path'], pair), rows, pair)
        fitted_centers = centers.fit_centers(center_records(frozen['head_paths'][pair[0]], pair), pair)
        fit_count = fitted_scale.audit['fit_count'] + fitted_centers['audit']['fit_count']
        if fit_count != 8: raise ValueError('Six conditional ETs plus two center fits per pair required')
        total_fits += fit_count
        model_path = OUT / 'pair_models' / (name + '.json')
        write(model_path, dict(pair=pair, bank_path=b['path'], fits=fit_count,
            scale_audit=fitted_scale.audit, center_audit=fitted_centers['audit']))
        models.append(dict(pair=pair, path=str(model_path), sha256=sha(model_path), fit_count=fit_count))
        for held in pair:
            outer = pair[1] if held == pair[0] else pair[0]
            law, query = laws[held], queries[held]
            s = scale.predict(fitted_scale, law['mix'], law['draws'], DAYS, held)
            h = centers.predict_centers(fitted_centers, query)
            paths = centers.bases(s['A'], s['R'], s['fine'], s['query_mean'], law['draws'], h['CIF_draws'])
            if (not np.array_equal(paths['raw'][0], s['A']) or not np.array_equal(paths['raw'][1], s['R'])
                    or not np.array_equal(paths['U'], s['U']) or not np.array_equal(paths['P'], s['P'])):
                raise ValueError('Literal shared Scale stream and scalar level pairing required')
            arrays = {k: v for k, v in s.items() if isinstance(v, np.ndarray) and k not in ('A', 'R')}
            arrays.update(raw=paths['raw'], center_levels=h['CIF_draws'], center_EF=h['EF_draws'],
                center_raw_ratios=h['raw_ratios'], center_ratios=h['ratios'], MG_draws=law['draws'],
                origins=DAYS.asi8, keys=np.array([f'{held}/{at:%Y-%m-%d}' for at in DAYS]), arms=np.array(ARMS),
                budgets=np.array(BUDGETS), panels=np.where(DAYS.month <= 6, 'H1_geo_backcast', 'H2_after_bank'), excluded_pair=np.array(pair), outer=np.array(outer), held=np.array(held))
            path = OUT / 'predictions' / (outer + '__' + held + '.npz')
            with path.open('xb') as f: np.savez_compressed(f, **arrays)
            audit = path.with_suffix('.json')
            write(audit, dict(outer=outer, held=held, pair=pair, label_region=held, prediction_audit=s['audit'],
                             model_path=str(model_path), source_input_path=str(OUT / 'inputs' / (held + '.npz'))))
            entries.append(dict(outer=outer, held=held, pair=list(pair), label_region=held, path=str(path),
                sha256=sha(path), audit_path=str(audit), audit_sha256=sha(audit), model_path=str(model_path)))
        print('Predicted pair', name, 'both directions, 8 fits; labels unopened', flush=True)
        del fitted_scale, fitted_centers, s, h, paths, arrays
    if total_fits != 168 or len(models) != 21 or len(entries) != 42: raise ValueError('Complete pair training grid required')
    for outer in SOURCES: tasks_for_outer(entries, outer)
    check()
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), entries=entries, source_inputs=source_inputs,
        pair_models=models, sklearn_fits=total_fits, complete_grid=True, risk_label_interface_opened=False,
        H1_training_banks_read_taskwise_excluded=True, source_H2_truth_read=False))


def predictions() -> dict:
    seal = read(OUT / 'prediction_seal.json')
    if (not seal['complete_grid'] or seal['sklearn_fits'] != 168 or len(seal['pair_models']) != 21
            or seal['freeze_sha256'] != sha(OUT / 'freeze.json')
            or seal['prediction_started_sha256'] != sha(OUT / 'prediction_started.json')
            or [e['region'] for e in seal['source_inputs']] != list(SOURCES)):
        raise ValueError('Complete pair prediction seal required before labels')
    hashes = {}
    for e in seal['entries'] + seal['source_inputs']:
        hashes.update({e['path']: e['sha256'], e['audit_path']: e['audit_sha256']})
    hashes.update({e['path']: e['sha256'] for e in seal['pair_models']})
    verify(hashes)
    for outer in SOURCES: tasks_for_outer(seal['entries'], outer)
    for e in seal['entries']:
        with np.load(e['path'], allow_pickle=False) as z:
            raw = z['raw']
            if (z['arms'].tolist() != list(ARMS) or z['budgets'].tolist() != list(BUDGETS)
                    or not np.array_equal(z['origins'], DAYS.asi8) or z['keys'].tolist() != [f"{e['held']}/{at:%Y-%m-%d}" for at in DAYS]
                    or z['excluded_pair'].tolist() != e['pair'] or z['outer'].item() != e['outer'] or z['held'].item() != e['held']
                    or raw.shape != (8, 342, 128, 24) or not np.isfinite(raw).all()
                    or z['center_levels'].shape != (2, 342, 128)):
                raise ValueError('All 42 finite complete prediction grids required before labels')
    return seal


def label() -> None:
    frozen = check(); seal = predictions()
    if (OUT / 'label_started.json').exists() or (OUT / 'completed.json').exists():
        raise FileExistsError('Preserve V49 labeling attempt')
    write(OUT / 'label_started.json', dict(created_utc=now(), prediction_seal_sha256=sha(OUT / 'prediction_seal.json')))
    (OUT / 'labels').mkdir(); (OUT / 'risk').mkdir()
    source_labels, label_entries, support = {}, [], []
    for source in SOURCES:
        rows = common.source_days(Path(frozen['source_paths'][source]), '2024-01-25', '2025-01-01')
        keys = [f'{source}/{at:%Y-%m-%d}' for at, _ in rows]
        if keys != frozen['label_keys'][source]: raise ValueError('Actual source-label support mismatch; preserve attempt')
        positions = np.array([DAYS.get_loc(at) for at, _ in rows], int)
        y = np.stack([(g/g.sum(1)[:, None]) @ centers.EF for _, g in rows])
        path = OUT / 'labels' / (source + '.npz')
        with path.open('xb') as f: np.savez_compressed(f, origins=DAYS.asi8[positions], truth=y, positions=positions, keys=np.array(keys),
                panels=np.where(DAYS[positions].month <= 6, 'H1_geo_backcast', 'H2_after_bank'))
        label_entries.append(dict(region=source, path=str(path), sha256=sha(path), days=len(positions),
                                  actual_first_release_verified=False))
        source_labels[source] = positions, y
        support.extend(dict(region=source, origin=at.isoformat(), available=i in positions)
                       for i, at in enumerate(DAYS))
    entries, scores = [], []
    for e in seal['entries']:
        outer, held = e['outer'], e['held']; positions, y = source_labels[held]
        with np.load(e['path'], allow_pickle=False) as z: raw = z['raw'][:, positions]
        point = np.clip(raw, 0, 1000).mean(2)
        error = centers.errors(point, y)
        path = OUT / 'risk' / (outer + '__' + held + '.npz')
        with path.open('xb') as f:
            np.savez_compressed(f, origins=DAYS.asi8[positions], arms=np.array(ARMS), budgets=np.array(BUDGETS),
                point=point, hourly_error=error['hourly_error'], level_error=error['level_error'],
                shape_error=error['shape_error'], keys=np.array(frozen['label_keys'][held]),
                panels=np.where(DAYS[positions].month <= 6, 'H1_geo_backcast', 'H2_after_bank'))
        entry = dict(e, risk_path=str(path), risk_sha256=sha(path),
            source_label_path=str(OUT / 'labels' / (held + '.npz')), input_path=str(OUT / 'inputs' / (held + '.npz')),
            valid_days=len(positions), error_reconstruction_max_abs=error['reconstruction_max_abs'])
        entries.append(entry)
        for i, (arm, budget) in enumerate(zip(ARMS, BUDGETS)):
            p, ensemble, clip = common.scaled_paths(raw[i], 1.)
            metrics = common.score_arrays(p, ensemble, y)
            centered = raw[i]-raw[i].mean(-1, keepdims=True)
            daily_std = raw[i].mean(-1).std(-1)
            shape_rms = np.sqrt(centered.var(1).mean(-1))
            for j, index in enumerate(positions):
                scores.append(dict(outer=outer, held=held, origin=DAYS[index].isoformat(), arm=arm, budget=budget,
                    panel='H1_geo_backcast' if DAYS[index].month <= 6 else 'H2_after_bank',
                    final_spread_clipping_rate=float(clip[j]), mc_daily_mean_std_raw=float(daily_std[j]),
                    mc_centered_shape_rms_raw=float(shape_rms[j]),
                    **{k: float(v[j]) for k, v in metrics.items()}))
        print('Labeled task', outer, '<-', held, len(positions), 'days', flush=True)
    outer_index = {outer: [e['risk_path'] for e in tasks_for_outer(entries, outer)] for outer in SOURCES}
    write(OUT / 'task_index.json', dict(entries=entries, labels=label_entries, outer_tasks=outer_index,
        reverse_tasks_may_contain_outer_labels=True, third_source_validation_provided=False,
        hourly_label_release_policy=None, actual_first_release_verified=False,
        requires_release_policy_before_risk_fit=True))
    day = pd.DataFrame(scores)
    regional = day.groupby(['outer', 'held', 'panel', 'arm', 'budget'], sort=False)[list(DIAGNOSTICS)].mean().reset_index()
    macro = regional.groupby(['outer', 'panel', 'arm', 'budget'], sort=False)[list(DIAGNOSTICS)].mean().reset_index()
    available = pd.DataFrame(support)
    available['month'] = pd.to_datetime(available.origin, utc=True).dt.month
    monthly_support = available.groupby(['region', 'month'], sort=False).available.sum().reset_index(name='valid_days')
    for name, table, count in [('source_case_metrics.csv', day, 100800), ('source_region_metrics.csv', regional, 672),
            ('source_outer_metrics.csv', macro, 112), ('label_support.csv', available, 2394), ('source_month_support.csv', monthly_support, 84)]:
        if len(table) != count or not np.isfinite(table.select_dtypes(include='number').to_numpy()).all():
            raise ValueError('Complete finite risk dataset diagnostics required')
        table.to_csv(OUT / name, mode='x', index=False)
    if sum(e['days'] for e in label_entries) != 2100 or sum(e['valid_days'] for e in entries) != 12600:
        raise ValueError('Prelocked 2100 unique and 12600 directed source days required')
    write(OUT / 'summary.json', dict(sklearn_fits=168, Scale_ET_fits=126, center_ET_fits=21, center_Ridge_fits=21,
        pair_models=21, directed_tasks=42, prediction_days=14364, source_label_days=2100, directed_valid_days=12600,
        base_day_examples=100800, bases=8, risk_models_fitted=False, proper_calibration_split_chosen=False,
        hourly_label_first_release_verified=False, requires_release_policy_before_risk_fit=True,
        source_2025_truth_read=False, method_innovation_claim=False, independent_confirmation=False))
    check()
    outputs = sorted(p for folder in ('labels', 'risk') for p in (OUT / folder).glob('*.npz'))
    outputs += [OUT / name for name in ('task_index.json', 'source_case_metrics.csv', 'source_region_metrics.csv',
                                      'source_outer_metrics.csv', 'label_support.csv', 'source_month_support.csv', 'summary.json')]
    write(OUT / 'completed.json', dict(created_utc=now(), status='COMPLETE', freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_seal_sha256=sha(OUT / 'prediction_seal.json'), label_started_sha256=sha(OUT / 'label_started.json'),
        output_hashes={str(p): sha(p) for p in outputs}, source_2025_truth_read=False,
        target_values_opened=False, unseen_values_opened=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'predict', 'label'))
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'label': label}[args.phase]()
