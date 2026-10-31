"""Nested source gain teachers and fixed gates with legal own-history MG replay."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import gain_direction_v44 as model
from scripts import run_ordinary_controls_v43 as prior

sha, read, write, now, verify = prior.sha, prior.read, prior.write, prior.now, prior.verify
SOURCES, DATES = prior.SOURCES, prior.DATES
OUT = ROOT / 'results/gain-direction-v44'
PARENT = ROOT / 'docs/evidence/ordinary-controls-v43-verification.json'
PARENT_SHA = 'a98874cbccf7e1f9d916de4bce15bc27a1abe79d4bdcb86f505c7b2369644ab6'
CODE = ('scripts/gain_direction_v44.py', 'scripts/run_gain_direction_v44.py',
        'tests/test_gain_direction_v44.py', 'tests/test_run_gain_direction_v44.py', 'docs/GAIN_DIRECTION_V44_PROTOCOL.md')
HELPERS = ('monthly_transfer_v25', 'coarse_context_v30', 'forward_context_v31', 'run_coarse_context_v30',
           'run_forward_context_v31', 'fleet_information_v34', 'capacity_routing_v36',
           'conditional_diagnostics_v35', 'donor_inventory_v41', 'donor_budget_v42',
           'ordinary_controls_v43', 'run_ordinary_controls_v43')
REVIEWS = tuple(ROOT / f'results/gain-direction-v44-{kind}-review.md' for kind in ('code', 'math'))
OUTPUTS = ('case_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv', 'leave_one_source_out_macro.csv',
           'label_support.csv', 'gain_case_metrics.csv', 'gain_region_metrics.csv', 'gain_macro_metrics.csv', 'summary.json')
MONTHLY_GROUPS = dict(legacy=SOURCES, v4=('SWPP', 'AVA', 'BANC', 'JEA', 'SC', 'NWMT'),
                      v5=('SCL', 'WACM', 'SEC', 'SCEG'))
VIEWS = ('HistoryForecast', 'HistoryForecastCap')


def contract() -> dict:
    verify({str(PARENT): PARENT_SHA})
    cursor = read(PARENT)
    if cursor['status'] != 'VERIFICATION_COMPLETE' or cursor['independent_formula_review_status'] != 'PASS':
        raise ValueError('Verified V43 required')
    closure, seen = {}, set()
    while True:
        for kind in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
            for path, h in cursor[kind].items():
                if path in closure and closure[path] != h:
                    raise ValueError('Conflicting inherited metadata')
                closure[path] = h
        if 'parent_manifest' not in cursor:
            break
        path, h = cursor['parent_manifest'], cursor['parent_manifest_sha256']
        if path in seen:
            raise ValueError('Cyclic inherited metadata')
        seen.add(path); verify({path: h}); cursor = read(path)
    paths = dict(monthly_path=str(ROOT / 'results/heldout-2026h1-data/monthly.csv'),
        baseline_case_path=str(ROOT / 'results/ordinary-controls-v43/case_metrics.csv'),
        support_path=str(ROOT / 'results/ordinary-controls-v43/label_support.csv'),
        cohort_path=str(ROOT / 'results/donor-budget-v42/cohort.json'),
        capacity_path=str(ROOT / 'results/donor-inventory-v41/capacity_op.csv'))
    monthly = dict(legacy=paths['monthly_path'], v4=str(ROOT / 'results/external-v4-data/monthly.csv'),
                   v5=str(ROOT / 'results/external-v5-data/monthly.csv'))
    archives = {s: dict(path=str(ROOT / f'results/donor-budget-v42/predictions/{s}.npz'),
        audit_path=str(ROOT / f'results/donor-budget-v42/predictions/{s}.json'),
        original_path=str(ROOT / f'results/fleet-information-v34/predictions/{s}.npz'),
        route_path=str(ROOT / f'results/ordinary-controls-v43/predictions/{s}.json')) for s in SOURCES}
    aliases = [*paths.values(), *monthly.values(), *[p for e in archives.values() for p in e.values()],
               *[str(ROOT / 'scripts' / (name+'.py')) for name in HELPERS]]
    if any(p not in closure for p in aliases):
        raise ValueError('Inherited direct dependency required')
    inputs = {p: closure[p] for p in aliases} | {str(PARENT): PARENT_SHA}
    for p in REVIEWS:
        if not p.is_file() or p.is_symlink() or not p.read_text().strip():
            raise ValueError('Both final prelaunch reviews must exist')
        inputs[str(p)] = sha(p)
    verify(inputs)
    return dict(version=44, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE}, input_hashes=inputs,
        environment=prior.common.environment(), **paths, monthly_paths=monthly, archives=archives,
        sources=list(SOURCES), arms=list(model.ARMS), fine_head_fits=315, meta_sklearn_fits=168, mean_estimates=42,
        source_MG_calls=412, query_MG_retrained=False, new_fine_label_keys=False, whole_prediction_budget='B1',
        method_innovation_claim=False, hourly_values_read=False, target_values_opened=False, unseen_values_opened=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V44 attempt')
    frozen = contract(); OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared V44', len(frozen['input_hashes']), 'direct dependencies', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('Frozen V44 contract changed')
    return frozen


def monthly_rows(paths: dict) -> list[dict]:
    rows = []
    for group, path in paths.items():
        with open(path, newline='') as stream:
            rows.extend(row for row in csv.DictReader(stream) if row['region'] in MONTHLY_GROUPS[group])
    return rows


def source_data(audit: dict, cohort: dict, outer: str) -> dict:
    expected = [key for key in cohort['keys'] if key.split('/')[0] != outer]
    keys, regions = audit['training_keys'], audit['training_regions']
    if (outer not in SOURCES or audit['region'] != outer or audit['outer'] != outer or keys != expected or
            len(keys) != (388 if outer == 'FPL' else 386) or len(regions) != len(keys) or
            any(key.split('/')[0] != s or s == outer for key, s in zip(keys, regions)) or len(set(regions)) != 15):
        raise ValueError('Excluded-outer metadata required before fine values')
    for field in ('period_start_ns', 'period_end_ns', 'available_ns'):
        dates = [at for key, at in zip(cohort['keys'], cohort[field]) if key.split('/')[0] != outer]
        if audit['training_'+field] != dates:
            raise ValueError('Original source dates required')
    n = len(keys)
    arrays = dict(X=np.asarray(audit['training_Size'], float), Q=np.asarray(audit['training_q'], float),
                  Cpos=np.asarray(audit['training_pos_capacity'], float), y=np.asarray(audit['training_y'], float))
    for name, shape in (('X', (n, 31)), ('Q', (n, 4)), ('Cpos', (n,)), ('y', (n, 4))):
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError('Complete finite original source arrays required')
    if np.any(arrays['Cpos'] <= 0):
        raise ValueError('Positive source POS capacity required')
    return dict(keys=keys, regions=np.asarray(regions), **arrays)


def teachers() -> None:
    frozen = check()
    if (OUT / 'teacher_started.json').exists() or (OUT / 'teachers').exists():
        raise FileExistsError('Preserve V44 teacher attempt')
    write(OUT / 'teacher_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    folder = OUT / 'teachers'; folder.mkdir()
    cohort = read(frozen['cohort_path']); keys = cohort['keys']
    if len(keys) != 412 or len(set(keys)) != 412:
        raise ValueError('Original 412 unique source keys required')
    mg = model.source_mg(monthly_rows(frozen['monthly_paths']), keys)
    with (OUT / 'source_MG.npz').open('xb') as stream:
        np.savez_compressed(stream, keys=np.asarray(keys), draws=mg['draws'], points=mg['points'])
    pd.DataFrame(mg['auditrows']).to_csv(OUT / 'source_MG_audit.csv', mode='x', index=False)
    lookup = {key: i for i, key in enumerate(keys)}; entries = []
    for outer in SOURCES:
        data = source_data(read(frozen['archives'][outer]['audit_path']), cohort, outer)
        indices = np.array([lookup[key] for key in data['keys']]); draws = mg['draws'][indices]
        n = len(indices); ef = np.empty((3, n, 128)); cif = np.empty_like(ef); raw = np.empty_like(ef)
        old_weights = np.empty(n); assigned = np.zeros(n, bool); folds = []
        for inner in dict.fromkeys(data['regions']):
            result = model.fit_inner(data['X'], data['Q'], data['Cpos'], data['y'], data['regions'], draws,
                                     outer=outer, inner=str(inner))
            ids = result['held_indices']
            if assigned[ids].any():
                raise ValueError('Inner source duplicated')
            ef[:, ids], cif[:, ids], raw[:, ids] = result['EF_draws'], result['CIF_draws'], result['model_EF']
            old_weights[ids] = result['old_coil_weights']; assigned[ids] = True
            folds.append(dict(inner=str(inner), held_indices=ids, **result['audit']))
        if not assigned.all() or len(folds) != 15:
            raise ValueError('Complete fifteen inner sources required')
        truth_ef = data['y'] @ np.array([1000., 400., 800., 400.])
        truth_cif = data['X'][:, 0] * truth_ef
        gains = model.gain_targets(ef, cif, truth_ef, truth_cif)
        features = model.meta_features(data['X'], draws, data['Q'], data['Cpos'])
        path = folder / (outer+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, keys=np.asarray(data['keys']), regions=data['regions'], source_MG_indices=indices,
                EF_draws=ef, CIF_draws=cif, model_EF=raw, old_coil_weights=old_weights,
                true_EF=truth_ef, true_CIF=truth_cif, gains=gains, **features)
        audit_path = path.with_suffix('.json')
        write(audit_path, dict(region=outer, training_keys=data['keys'], training_regions=data['regions'],
            inner_folds=folds, fine_head_fit_count=45, source_current_group_used_as_query=False,
            historical_fine_head_training_snapshot='2025-01-01'))
        entries.append(dict(region=outer, path=str(path), sha256=sha(path), audit_path=str(audit_path), audit_sha256=sha(audit_path)))
        print('Teachers', outer, '45 fine-head fits;', n, 'source months', flush=True)
    check()
    write(OUT / 'teacher_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        teacher_started_sha256=sha(OUT / 'teacher_started.json'), source_MG_sha256=sha(OUT / 'source_MG.npz'),
        source_MG_audit_sha256=sha(OUT / 'source_MG_audit.csv'), entries=entries))


def teachers_sealed() -> dict:
    seal = read(OUT / 'teacher_seal.json')
    if (seal['freeze_sha256'] != sha(OUT / 'freeze.json') or
            seal['teacher_started_sha256'] != sha(OUT / 'teacher_started.json') or
            read(OUT / 'teacher_started.json')['freeze_sha256'] != seal['freeze_sha256'] or
            seal['source_MG_sha256'] != sha(OUT / 'source_MG.npz') or
            seal['source_MG_audit_sha256'] != sha(OUT / 'source_MG_audit.csv')):
        raise ValueError('Complete source teacher seal required')
    cohort = read(read(OUT / 'freeze.json')['cohort_path'])
    with np.load(OUT / 'source_MG.npz', allow_pickle=False) as a:
        if (a['keys'].tolist() != cohort['keys'] or a['draws'].shape != (412, 128, 5) or a['points'].shape != (412, 5) or
                any(not np.isfinite(a[k]).all() or np.any(a[k] < 0) or
                    not np.allclose(a[k].sum(-1), 1., atol=1e-12, rtol=0) for k in ('draws', 'points'))):
            raise ValueError('Complete legal source MG grid required')
    return validate_entries(seal, 'teachers', teacher=True)


def validate_entries(seal: dict, folder_name: str, *, teacher: bool = False) -> dict:
    folder = OUT / folder_name
    if ([e['region'] for e in seal['entries']] != list(SOURCES) or
            {p.name for p in folder.iterdir()} != {f'{s}.{ext}' for s in SOURCES for ext in ('npz', 'json')}):
        raise ValueError('Exact seven sealed source pairs required')
    cohort = read(read(OUT / 'freeze.json')['cohort_path']) if teacher else None
    if teacher:
        with np.load(OUT / 'source_MG.npz', allow_pickle=False) as mg_archive:
            source_mg = mg_archive['draws'].copy()
        lookup = {key: i for i, key in enumerate(cohort['keys'])}
    for e in seal['entries']:
        path = folder / (e['region']+'.npz')
        if e['path'] != str(path) or e['audit_path'] != str(path.with_suffix('.json')):
            raise ValueError('Exact sealed aliases required')
        verify({e['path']: e['sha256'], e['audit_path']: e['audit_sha256']})
        with np.load(path, allow_pickle=False) as a:
            if teacher:
                expected = [key for key in cohort['keys'] if key.split('/')[0] != e['region']]
                n = len(expected)
                if a['keys'].tolist() != expected or a['regions'].tolist() != [k.split('/')[0] for k in expected]:
                    raise ValueError('Excluded-outer teacher keys required')
                expected_indices = np.asarray([lookup[key] for key in expected])
                if not np.array_equal(a['source_MG_indices'], expected_indices):
                    raise ValueError('Exact shared source MG indices required')
                shapes = dict(EF_draws=(3, n, 128), CIF_draws=(3, n, 128), model_EF=(3, n, 128),
                    old_coil_weights=(n,), true_EF=(n,), true_CIF=(n,), gains=(2, 3, n),
                    HistoryForecast=(n, 28), HistoryForecastCap=(n, 33))
            else:
                n = 6
                if (a['arms'].tolist() != list(model.ARMS) or not np.array_equal(a['origins'], DATES.asi8) or
                        a['keys'].tolist() != [f"{e['region']}/{at:%Y-%m}" for at in DATES]):
                    raise ValueError('Exact query prediction grid required')
                shapes = dict(EF_draws=(30, 6, 128), CIF_draws=(30, 6, 128), MG_draws=(6, 128, 5),
                              gain_prediction=(30, 6), actions=(30, 6))
            for name, shape in shapes.items():
                if a[name].shape != shape or not np.isfinite(a[name]).all():
                    raise ValueError('Finite complete sealed arrays required')
            if np.any(a['EF_draws'] < 400-1e-9) or np.any(a['EF_draws'] > 1000+1e-9):
                raise ValueError('EF must remain within fixed factor bounds')
            if teacher and not np.allclose(a['CIF_draws'], a['EF_draws']*source_mg[expected_indices][None, ..., 0], atol=1e-9, rtol=0):
                raise ValueError('Teacher CIF must use identical shared source MG')
            if teacher and not np.allclose(a['gains'], model.gain_targets(a['EF_draws'], a['CIF_draws'], a['true_EF'], a['true_CIF']), atol=1e-9, rtol=0):
                raise ValueError('Teacher gain targets must use the fixed mean loss formulas')
            if not teacher and (not np.array_equal(a['actions'], a['gain_prediction'] > 0) or
                    np.any(a['MG_draws'] < 0) or not np.allclose(a['MG_draws'].sum(-1), 1., atol=1e-12, rtol=0) or
                    not np.allclose(a['CIF_draws'], a['EF_draws']*a['MG_draws'][None, ..., 0], atol=1e-9, rtol=0)):
                raise ValueError('Fixed threshold and aligned MG multiplication required')
    return seal


def query_data(frozen: dict, outer: str) -> dict:
    entry = frozen['archives'][outer]
    a, anchor, q = prior.archive(entry, outer)
    with np.load(entry['original_path'], allow_pickle=False) as original:
        raw = original['query_feature_SizeCap'].copy()
    caps = []
    with open(frozen['capacity_path'], newline='') as stream:
        for row in csv.DictReader(stream):
            if row['region'] == outer and int(row['year']) == 2023:
                caps.append(float(row['capacity_pos_mw']))
    if len(caps) != 1 or not np.isfinite(caps[0]) or caps[0] <= 0:
        raise ValueError('Single original positive query POS inventory required')
    features = model.meta_features(raw[..., :31], a['MG_draws'], q, np.full(6, caps[0]))
    old_w = np.asarray(read(entry['route_path'])['old_weights'])[8]
    base_id = a['arms'].tolist().index('Legacy-Size-Ridge')
    ef = [np.where(old_w[:, None] == 1, anchor, a['EF_draws'][base_id])]
    cif = [np.where(old_w[:, None] == 1, anchor*a['MG_draws'][..., 0], a['CIF_draws'][base_id])]
    for arm in ('Expanded-SizeCap-ET', 'Expanded-DirectEF-ET'):
        i = a['arms'].tolist().index(arm); ef.append(a['EF_draws'][i]); cif.append(a['CIF_draws'][i])
    return dict(keys=a['keys'], MG_draws=a['MG_draws'], EF_draws=np.stack(ef), CIF_draws=np.stack(cif), features=features)


def predict() -> None:
    frozen, teacher = check(), teachers_sealed()
    if (OUT / 'prediction_started.json').exists() or (OUT / 'predictions').exists():
        raise FileExistsError('Preserve V44 prediction attempt')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        teacher_seal_sha256=sha(OUT / 'teacher_seal.json')))
    folder = OUT / 'predictions'; folder.mkdir(); entries, routes = [], []
    for entry in teacher['entries']:
        outer = entry['region']; query = query_data(frozen, outer)
        with np.load(entry['path'], allow_pickle=False) as a:
            result = model.fit_meta({name: a[name].copy() for name in VIEWS}, a['gains'], a['regions'],
                                    query['features'], outer=outer)
        actions = result['actions']; ef, cif = [], []
        for i in range(30):
            candidate = i//15+1
            ef.append(np.where(actions[i, :, None], query['EF_draws'][candidate], query['EF_draws'][0]))
            cif.append(np.where(actions[i, :, None], query['CIF_draws'][candidate], query['CIF_draws'][0]))
            routes.extend(dict(region=outer, key=str(key), arm=model.ARMS[i], candidate=candidate,
                target=model.TARGETS[(i%15)//5], gain_prediction=float(result['gain_prediction'][i, j]),
                action=bool(actions[i, j])) for j, key in enumerate(query['keys']))
        path = folder / (outer+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, arms=np.asarray(model.ARMS), keys=query['keys'], origins=DATES.asi8,
                MG_draws=query['MG_draws'], EF_draws=np.stack(ef), CIF_draws=np.stack(cif),
                gain_prediction=result['gain_prediction'], actions=actions)
        audit_path = path.with_suffix('.json'); write(audit_path, dict(region=outer, **result['audit']))
        entries.append(dict(region=outer, path=str(path), sha256=sha(path), audit_path=str(audit_path), audit_sha256=sha(audit_path)))
        print('Meta', outer, '24 fits and 6 Mean estimates', flush=True)
    pd.DataFrame(routes).to_csv(OUT / 'query_routes.csv', mode='x', index=False)
    check(); teachers_sealed()
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        teacher_seal_sha256=sha(OUT / 'teacher_seal.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), query_routes_sha256=sha(OUT / 'query_routes.csv'), entries=entries))


def predictions() -> dict:
    seal = read(OUT / 'prediction_seal.json'); teachers_sealed()
    if (seal['freeze_sha256'] != sha(OUT / 'freeze.json') or
            seal['teacher_seal_sha256'] != sha(OUT / 'teacher_seal.json') or
            seal['prediction_started_sha256'] != sha(OUT / 'prediction_started.json') or
            seal['query_routes_sha256'] != sha(OUT / 'query_routes.csv') or
            read(OUT / 'prediction_started.json')['freeze_sha256'] != seal['freeze_sha256'] or
            read(OUT / 'prediction_started.json')['teacher_seal_sha256'] != seal['teacher_seal_sha256']):
        raise ValueError('Complete query prediction seal required')
    return validate_entries(seal, 'predictions')


def gain_metrics(pred: float, truth: float, action: bool) -> dict:
    error = pred-truth
    return dict(predicted_gain=pred, true_gain=truth, gain_bias=error, gain_abs_error=abs(error),
        gain_sq_error=error*error, action=float(action), true_tie=float(truth == 0),
        noninferior_action=float((action and truth >= 0) or (not action and truth <= 0)),
        regret=max(truth, 0)-float(action)*truth)


def score() -> None:
    if any((OUT / name).exists() for name in (*OUTPUTS, 'score_started.json', 'completed.json')):
        raise FileExistsError('Preserve V44 score attempt')
    frozen, seal = check(), predictions()
    write(OUT / 'score_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_seal_sha256=sha(OUT / 'prediction_seal.json')))
    labels, support = prior.labels_api.scoring_labels(prior.common.monthly_rows(frozen))
    if len(labels) != 35 or support.to_csv(index=False) != Path(frozen['support_path']).read_text():
        raise ValueError('Original 35 labels and 42 support rows required')
    baseline = pd.read_csv(frozen['baseline_case_path'], float_precision='round_trip')
    if len(baseline) != 4970 or set(baseline.key) != set(labels):
        raise ValueError('Original full strong references required')
    records, gain_records = [], []
    for entry in seal['entries']:
        outer = entry['region']; query = query_data(frozen, outer)
        with np.load(entry['path'], allow_pickle=False) as a:
            for j, key in enumerate(a['keys']):
                if key not in labels:
                    continue
                truth = labels[key]
                true_gains = model.gain_targets(query['EF_draws'][:, j:j+1], query['CIF_draws'][:, j:j+1],
                    np.array([truth['ef']]), np.array([truth['cif']]))
                for i, arm in enumerate(model.ARMS):
                    points = prior.point_case(a['CIF_draws'][i, j], a['EF_draws'][i, j], truth['cif'], truth['ef'])
                    records.extend(dict(region=outer, key=str(key), arm=arm, mode=mode, **row) for mode, row in points.items())
                    gain_records.append(dict(region=outer, key=str(key), arm=arm,
                        **gain_metrics(float(a['gain_prediction'][i, j]), float(true_gains[i//15, (i%15)//5, 0]), bool(a['actions'][i, j]))))
    case = pd.concat([baseline, pd.DataFrame(records)], ignore_index=True)
    if len(case) != 7070 or not case.iloc[:4970].equals(baseline) or case.duplicated(['region', 'key', 'arm', 'mode']).any():
        raise ValueError('Exact 4970 reference plus 2100 new cases required')
    region, macro, leave = prior.aggregate(case)
    gain_case = pd.DataFrame(gain_records)
    gain_region = gain_case.groupby(['region', 'arm']).mean(numeric_only=True).reset_index()
    gain_region['gain_rmse'] = np.sqrt(gain_region.gain_sq_error)
    gain_macro = gain_region.groupby('arm').mean(numeric_only=True).reset_index()
    if (len(region), len(macro), len(leave), len(gain_case), len(gain_region), len(gain_macro)) != (1414, 202, 1414, 1050, 210, 30):
        raise ValueError('Complete endpoint and gain diagnostic tables required')
    tables = dict(case_metrics=case, region_metrics=region, macro_metrics=macro, leave_one_source_out_macro=leave,
        label_support=support, gain_case_metrics=gain_case, gain_region_metrics=gain_region, gain_macro_metrics=gain_macro)
    for name, frame in tables.items():
        frame.to_csv(OUT / (name+'.csv'), index=False, mode='x')
    summary = dict(fine_head_fits=315, meta_sklearn_fits=168, mean_estimates=42, source_MG_calls=412,
        query_MG_retrained=False, new_arms=30, original_reference_rows_exact=4970,
        rows={name: len(frame) for name, frame in tables.items()},
        whole_prediction_budget='B1', historical_teacher_snapshot='2025-01-01',
        authentic_historical_fine_head_hindcast=False, method_innovation_claim=False,
        new_holdout_authorized=False, CIF_interval_coverage_evaluated=False, hourly_coverage_evaluated=False)
    write(OUT / 'summary.json', summary); check(); predictions()
    write(OUT / 'completed.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_seal_sha256=sha(OUT / 'prediction_seal.json'), score_started_sha256=sha(OUT / 'score_started.json'),
        output_hashes={str(OUT / name): sha(OUT / name) for name in OUTPUTS}, **summary))
    print('Scored V44 aligned mean gain diagnostics; no holdout opened', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'teachers', 'predict', 'score'), required=True)
    args = parser.parse_args(); {'prepare': prepare, 'teachers': teachers, 'predict': predict, 'score': score}[args.stage]()
