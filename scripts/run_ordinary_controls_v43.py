"""Fixed zero-fit mixtures and empirical range controls from sealed V42 arrays."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import ordinary_controls_v43 as model
from scripts import run_coarse_context_v30 as common
from scripts import run_forward_context_v31 as labels_api
from scripts.conditional_diagnostics_v35 import point_case

sha, read, write, now = common.sha, common.read, common.write, common.now
SOURCES, DATES = labels_api.model.SOURCES, labels_api.DATES
OUT = ROOT / 'results/ordinary-controls-v43'
PARENT = ROOT / 'docs/evidence/donor-budget-v42-verification.json'
PARENT_SHA = '059b97ed4cc7013f6634ca053f31560276dda20ca2b55917c099cdd48316bc67'
CODE = ('scripts/ordinary_controls_v43.py', 'scripts/run_ordinary_controls_v43.py',
        'tests/test_ordinary_controls_v43.py', 'tests/test_run_ordinary_controls_v43.py',
        'docs/ORDINARY_CONTROLS_V43_PROTOCOL.md')
HELPERS = ('coarse_context_v30', 'forward_context_v31', 'run_coarse_context_v30',
           'run_forward_context_v31', 'conditional_diagnostics_v35', 'capacity_routing_v36')
REVIEWS = tuple(ROOT / f'results/ordinary-controls-v43-{kind}-review.md' for kind in ('code', 'math'))
OUTPUTS = ('case_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv',
           'leave_one_source_out_macro.csv', 'label_support.csv', 'summary.json')


def verify(mapping: dict) -> None:
    for path in mapping:
        if not Path(path).is_file() or Path(path).is_symlink():
            raise ValueError('Regular bound file required')
    common.verify(mapping)


def inherited() -> dict:
    """Resolve hashes from manifest metadata; hash only direct dependencies later."""
    verify({str(PARENT): PARENT_SHA})
    cursor = read(PARENT)
    if cursor['status'] != 'VERIFICATION_COMPLETE' or cursor['independent_formula_review_status'] != 'PASS':
        raise ValueError('Independent V42 PASS required')
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


def contract() -> dict:
    closure = inherited()
    paths = dict(monthly_path=str(ROOT / 'results/heldout-2026h1-data/monthly.csv'),
        baseline_case_path=str(ROOT / 'results/donor-budget-v42/case_metrics.csv'),
        support_path=str(ROOT / 'results/donor-budget-v42/label_support.csv'),
        cohort_path=str(ROOT / 'results/donor-budget-v42/cohort.json'))
    archives = {source: dict(path=str(ROOT / f'results/donor-budget-v42/predictions/{source}.npz'),
        audit_path=str(ROOT / f'results/donor-budget-v42/predictions/{source}.json'),
        original_path=str(ROOT / f'results/fleet-information-v34/predictions/{source}.npz')) for source in SOURCES}
    aliases = [*paths.values(), *[p for entry in archives.values() for p in entry.values()],
               *[str(ROOT / 'scripts' / (name+'.py')) for name in HELPERS]]
    if any(path not in closure for path in aliases):
        raise ValueError('Inherited dependency aliases required')
    inputs = {path: closure[path] for path in aliases} | {str(PARENT): PARENT_SHA}
    for path in REVIEWS:
        if not path.is_file() or path.is_symlink() or not path.read_text().strip():
            raise ValueError('Both final prelaunch reports must exist')
        inputs[str(path)] = sha(path)
    verify(inputs)
    return dict(version=43, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE}, input_hashes=inputs,
        environment=common.environment(), **paths, archives=archives, sources=list(SOURCES), arms=list(model.ARMS),
        new_fits=0, new_fine_label_keys=False, MG_retrained=False, method_innovation_claim=False,
        hourly_values_read=False, target_values_opened=False, unseen_values_opened=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V43 attempt')
    frozen = contract()
    OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared', len(frozen['input_hashes']), 'direct dependencies; zero fits', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != value for k, value in expected.items()):
        raise ValueError('Frozen V43 contract changed')
    return frozen


def training_q(audit: dict, cohort: dict, outer: str) -> tuple[np.ndarray, np.ndarray]:
    keys = audit['training_keys']
    regions = audit['training_regions']
    expected = [key for key in cohort['keys'] if key.split('/')[0] != outer]
    if (outer not in SOURCES or audit['region'] != outer or audit['outer'] != outer or keys != expected or
            len(keys) != (388 if outer == 'FPL' else 386) or len(regions) != len(keys) or
            any(key.split('/')[0] != source or source == outer for key, source in zip(keys, regions)) or
            len(set(regions)) != 15):
        raise ValueError('Exact excluded-outer training metadata required before q values')
    for field in ('period_start_ns', 'period_end_ns', 'available_ns'):
        expected_dates = [at for key, at in zip(cohort['keys'], cohort[field]) if key.split('/')[0] != outer]
        if audit['training_'+field] != expected_dates:
            raise ValueError('Training dates changed')
    q = np.asarray(audit['training_q'], float)
    if (q.shape != (len(keys), 4) or not np.isfinite(q).all() or np.any(q < 0) or
            not np.allclose(q.sum(-1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Finite simplex training q required')
    old = q[np.array([source in SOURCES for source in regions])]
    if len(old) != (156 if outer == 'FPL' else 154):
        raise ValueError('Original six-source q cohort required')
    return old.copy(), q.copy()


def archive(entry: dict, outer: str) -> tuple[dict, np.ndarray, np.ndarray]:
    with np.load(entry['path'], allow_pickle=False) as data:
        a = {name: data[name].copy() for name in ('arms', 'keys', 'origins', 'MG_draws', 'EF_draws', 'CIF_draws')}
    if (a['arms'].tolist() != list(model.PARENT_ARMS) or
            a['keys'].tolist() != [f'{outer}/{at:%Y-%m}' for at in DATES] or
            not np.array_equal(a['origins'], DATES.asi8)):
        raise ValueError('Exact inherited V42 query grid required')
    with np.load(entry['original_path'], allow_pickle=False) as old:
        if (old['keys'].tolist() != a['keys'].tolist() or not np.array_equal(old['origins'], a['origins']) or
                not np.array_equal(old['MG_draws'], a['MG_draws'])):
            raise ValueError('Literal original MG/query alignment required')
        names = old['arms'].tolist()
        if len(names) != 11 or len(set(names)) != 11 or 'Capacity-OP' not in names:
            raise ValueError('Original capacity anchor required')
        anchor = old['EF_draws'][names.index('Capacity-OP')].copy()
        raw = old['query_feature_SizeCap'].copy()
    if raw.shape != (6, 128, 36) or not np.isfinite(raw).all() or not np.array_equal(raw[..., :5], a['MG_draws']):
        raise ValueError('Original raw capacity query features required')
    q = raw[..., 31:35]
    if not np.array_equal(q, np.broadcast_to(q[:, :1], q.shape)):
        raise ValueError('Query q must be particle-constant')
    query = q[:, 0].copy()
    if (anchor.shape != (6, 128) or not np.isfinite(anchor).all() or
            not np.allclose(anchor, (400+600*query[:, 0]+400*query[:, 2])[:, None], atol=1e-9, rtol=0)):
        raise ValueError('Original capacity anchor/q consistency required')
    return a, anchor, query


def predict() -> None:
    frozen = check()
    if (OUT / 'prediction_started.json').exists() or (OUT / 'predictions').exists():
        raise FileExistsError('Preserve V43 prediction attempt')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    folder = OUT / 'predictions'; folder.mkdir()
    cohort = read(frozen['cohort_path'])
    if (len(cohort['keys']) != 412 or len(set(cohort['keys'])) != 412 or
            sum(key.split('/')[0] in SOURCES for key in cohort['keys']) != 180):
        raise ValueError('Original V42 cohort metadata required')
    entries, routes, limits = [], [], []
    for outer in SOURCES:
        old_q, new_q = training_q(read(frozen['archives'][outer]['audit_path']), cohort, outer)
        a, anchor, query = archive(frozen['archives'][outer], outer)
        result = model.build(a['EF_draws'], a['CIF_draws'], a['MG_draws'], anchor, old_q, new_q, query)
        if np.any(result['audit']['expanded_weights'][5:] > result['audit']['old_weights'][5:]):
            raise ValueError('Expanded extrema cannot add range-fallback triggers')
        path = folder / (outer+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, arms=np.array(model.ARMS), keys=a['keys'], origins=a['origins'],
                MG_draws=a['MG_draws'], EF_draws=result['EF_draws'], CIF_draws=result['CIF_draws'], weights=result['weights'])
        audit_path = path.with_suffix('.json')
        write(audit_path, dict(region=outer, old_training_cases=len(old_q), expanded_training_cases=len(new_q),
            training_q_subset_of_expanded=True, query_q=query, **result['audit']))
        for i, arm in enumerate(model.ARMS):
            routes.extend(dict(region=outer, key=str(key), arm=arm, right_weight=float(result['weights'][i, j]))
                          for j, key in enumerate(a['keys']))
        limit = dict(region=outer, old_cases=len(old_q), expanded_cases=len(new_q))
        for k, fuel in enumerate(('coal', 'gas', 'oil', 'other')):
            for name, q in (('old', old_q), ('expanded', new_q)):
                limit[name+'_'+fuel+'_min'] = float(q[:, k].min())
                limit[name+'_'+fuel+'_max'] = float(q[:, k].max())
        limits.append(limit)
        entries.append(dict(region=outer, path=str(path), sha256=sha(path), audit_path=str(audit_path), audit_sha256=sha(audit_path)))
        print('Predicted', outer, '35 controls; zero fits', flush=True)
    pd.DataFrame(routes).to_csv(OUT / 'route_table.csv', mode='x', index=False)
    pd.DataFrame(limits).to_csv(OUT / 'training_limits.csv', mode='x', index=False)
    if len(routes) != 1470:
        raise ValueError('Complete route grid required')
    check()
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), route_table_sha256=sha(OUT / 'route_table.csv'),
        training_limits_sha256=sha(OUT / 'training_limits.csv'), entries=entries))


def predictions() -> dict:
    seal = read(OUT / 'prediction_seal.json')
    if (seal['freeze_sha256'] != sha(OUT / 'freeze.json') or
            read(OUT / 'prediction_started.json')['freeze_sha256'] != seal['freeze_sha256'] or
            seal['prediction_started_sha256'] != sha(OUT / 'prediction_started.json') or
            seal['route_table_sha256'] != sha(OUT / 'route_table.csv') or
            seal['training_limits_sha256'] != sha(OUT / 'training_limits.csv') or
            [entry['region'] for entry in seal['entries']] != list(SOURCES)):
        raise ValueError('Complete prediction/routes seal required')
    folder = OUT / 'predictions'
    if {p.name for p in folder.iterdir()} != {f'{s}.{ext}' for s in SOURCES for ext in ('npz', 'json')}:
        raise ValueError('Exact seven prediction/audit pairs required')
    for entry in seal['entries']:
        path = folder / (entry['region']+'.npz')
        if entry['path'] != str(path) or entry['audit_path'] != str(path.with_suffix('.json')):
            raise ValueError('Exact sealed source aliases required')
        verify({entry['path']: entry['sha256'], entry['audit_path']: entry['audit_sha256']})
        with np.load(path, allow_pickle=False) as a:
            if (a['arms'].tolist() != list(model.ARMS) or not np.array_equal(a['origins'], DATES.asi8) or
                    a['keys'].tolist() != [f"{entry['region']}/{at:%Y-%m}" for at in DATES]):
                raise ValueError('Exact new prediction grid required')
            for name, shape in (('EF_draws', (35, 6, 128)), ('CIF_draws', (35, 6, 128)),
                                ('MG_draws', (6, 128, 5)), ('weights', (35, 6))):
                if a[name].shape != shape or not np.isfinite(a[name]).all():
                    raise ValueError('Finite complete arrays required')
            if (np.any(a['EF_draws'] < 400-1e-9) or np.any(a['EF_draws'] > 1000+1e-9) or
                    np.any(a['MG_draws'] < 0) or not np.allclose(a['MG_draws'].sum(-1), 1., atol=1e-12, rtol=0) or
                    np.any(a['weights'] < 0) or np.any(a['weights'] > 1) or
                    not np.allclose(a['CIF_draws'], a['EF_draws']*a['MG_draws'][None, ..., 0], atol=1e-9, rtol=0)):
                raise ValueError('Bounded EF and preserved MG multiplication required')
    return seal


def aggregate(case: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    region = case.groupby(['region', 'arm', 'mode']).mean(numeric_only=True).reset_index()
    for kind in ('cif', 'ef'):
        region[kind+'_rmse'] = np.sqrt(region[kind+'_sq_error'])
    counts = case.groupby(['region', 'arm', 'mode']).size().rename('scored_months').reset_index()
    region = region.merge(counts, on=['region', 'arm', 'mode'], validate='one_to_one')
    macro = region.drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index()
    leave = pd.concat([region[region.region.ne(s)].drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index().assign(omitted_source=s) for s in SOURCES], ignore_index=True)
    return region, macro, leave


def score() -> None:
    if any((OUT / name).exists() for name in (*OUTPUTS, 'score_started.json', 'completed.json')):
        raise FileExistsError('Preserve V43 score attempt')
    frozen, seal = check(), predictions()
    write(OUT / 'score_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_seal_sha256=sha(OUT / 'prediction_seal.json')))
    labels, support = labels_api.scoring_labels(common.monthly_rows(frozen))
    if len(labels) != 35 or support.to_csv(index=False) != Path(frozen['support_path']).read_text():
        raise ValueError('Original 35 score labels and 42 support rows required')
    baseline = pd.read_csv(frozen['baseline_case_path'], float_precision='round_trip')
    if len(baseline) != 2520 or set(baseline.key) != set(labels):
        raise ValueError('Original V42 comparison support required')
    records = []
    for entry in seal['entries']:
        with np.load(entry['path'], allow_pickle=False) as a:
            for j, key in enumerate(a['keys']):
                if key not in labels:
                    continue
                truth = labels[key]
                for i, arm in enumerate(model.ARMS):
                    points = point_case(a['CIF_draws'][i, j], a['EF_draws'][i, j], truth['cif'], truth['ef'])
                    records.extend(dict(region=entry['region'], key=str(key), arm=arm, mode=mode, **row) for mode, row in points.items())
    case = pd.concat([baseline, pd.DataFrame(records)], ignore_index=True)
    if len(case) != 4970 or not case.iloc[:2520].equals(baseline) or case.duplicated(['region', 'key', 'arm', 'mode']).any():
        raise ValueError('Original 2520 references and 2450 fixed new cases required')
    region, macro, leave = aggregate(case)
    if (len(region), len(macro), len(leave)) != (994, 142, 994):
        raise ValueError('Complete control/reference aggregates required')
    for name, frame in dict(case_metrics=case, region_metrics=region, macro_metrics=macro,
                           leave_one_source_out_macro=leave, label_support=support).items():
        frame.to_csv(OUT / (name+'.csv'), mode='x', index=False)
    summary = dict(new_fits=0, new_arms=35, source_months=35, query_origins=42, baseline_rows_exact=2520,
        rows=dict(case_metrics=4970, region_metrics=994, macro_metrics=142, leave_one_source_out_macro=994,
                  label_support=42, route_table=1470, training_limits=7),
        ordinary_controls_only=True, method_innovation_claim=False, new_holdout_authorized=False,
        MG_retrained=False, CIF_interval_coverage_evaluated=False, hourly_coverage_evaluated=False)
    write(OUT / 'summary.json', summary)
    check(); predictions()
    write(OUT / 'completed.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_seal_sha256=sha(OUT / 'prediction_seal.json'),
        score_started_sha256=sha(OUT / 'score_started.json'),
        output_hashes={str(OUT / name): sha(OUT / name) for name in OUTPUTS}, **summary))
    print('Scored 35 ordinary controls; no new fit or holdout', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'predict', 'score'), required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score}[args.stage]()
