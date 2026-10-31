"""Fixed old-versus-expanded donor controls, with preserved MG and source weights."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import donor_budget_v42 as model
from scripts import coarse_context_v30 as v30
from scripts import fleet_information_v34 as v34
from scripts import fleet_inventory_v33 as inventory
from scripts import run_donor_inventory_v41 as parent

v40 = parent.parent
v38 = v40.parent.parent
common = v40.common
sha, read, write, verify, now = parent.sha, parent.read, parent.write, parent.verify, parent.now
SOURCES, DATES = v38.SOURCES, v38.DATES
ARMS = tuple('Legacy-'+s for s in model.ARMS) + tuple('Expanded-'+s for s in model.ARMS)
OUT = ROOT / 'results/donor-budget-v42'
PARENT = ROOT / 'docs/evidence/donor-inventory-v41-verification.json'
PARENT_SHA = 'b796422578980f1855fa091ab11bc160890491b4ea46a9b305f61b48d1e85c9e'
CODE = ('scripts/donor_budget_v42.py', 'scripts/run_donor_budget_v42.py',
        'tests/test_donor_budget_v42.py', 'tests/test_run_donor_budget_v42.py', 'docs/DONOR_BUDGET_V42_PROTOCOL.md')
REVIEWS = tuple(ROOT / f'results/donor-budget-v42-{kind}-review.md' for kind in ('code', 'math'))
OUTPUTS = ('case_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv', 'leave_one_source_out_macro.csv',
           'label_support.csv', 'summary.json')


def contract() -> dict:
    verify({str(PARENT): PARENT_SHA})
    cursor = read(PARENT)
    if cursor['status'] != 'VERIFICATION_COMPLETE' or cursor['independent_formula_review_status'] != 'PASS':
        raise ValueError('Verified V41 required')
    closure, seen = {}, set()
    while True:
        for p, h in parent.parent.parent.mappings(cursor).items():
            if p in closure and closure[p] != h:
                raise ValueError('Conflicting inherited hash')
            closure[p] = h
        if 'parent_manifest' not in cursor:
            break
        p, h = cursor['parent_manifest'], cursor['parent_manifest_sha256']
        if p in seen:
            raise ValueError('Cyclic evidence')
        seen.add(p); verify({p: h}); cursor = read(p)
    verify(closure)
    monthly = dict(legacy=str(ROOT / 'results/heldout-2026h1-data/monthly.csv'),
        v4=str(ROOT / 'results/external-v4-data/monthly.csv'), v5=str(ROOT / 'results/external-v5-data/monthly.csv'))
    aliases = dict(capacity_path=str(ROOT / 'results/donor-inventory-v41/capacity_op.csv'),
        baseline_case_path=str(ROOT / 'results/coarse-residual-v40/case_metrics.csv'),
        support_path=str(ROOT / 'results/coarse-residual-v40/label_support.csv'))
    archives = {s: dict(path=str(ROOT / f'results/fleet-information-v34/predictions/{s}.npz'),
        audit_path=str(ROOT / f'results/fleet-information-v34/predictions/{s}.json')) for s in SOURCES}
    required = [monthly['legacy'], *aliases.values(), *[e[k] for e in archives.values() for k in ('path', 'audit_path')]]
    if any(p not in closure for p in required):
        raise ValueError('Bound legacy query, cohort and reference aliases required')
    inputs = {p: closure[p] for p in required} | {str(PARENT): PARENT_SHA}
    # New files contain only fixed already-viewed donor tables; their scope is projected below.
    for p in (monthly['v4'], monthly['v5'], *map(str, REVIEWS)):
        if not Path(p).is_file() or Path(p).is_symlink():
            raise ValueError('Regular new source input required')
        inputs[p] = sha(Path(p))
    if any(not p.read_text().strip() for p in REVIEWS):
        raise ValueError('Two independent prelaunch reviews required')
    verify(inputs)
    return dict(version=42, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE}, input_hashes=inputs,
        environment=common.environment(), monthly_paths=monthly, monthly_path=monthly['legacy'], **aliases,
        archives=archives, sources=list(SOURCES), candidate_pool=list(model.ALL_SOURCES), arms=list(ARMS),
        new_fits=42, MG_retrained=False, target_local_information_increased=False,
        old_training_weight_mass=dict(default=154, FPL=156), new_fine_supervision_end_exclusive=v30.END.isoformat(),
        hourly_values_read=False, target_values_opened=False, unseen_values_opened=False, method_innovation_claim=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V42 attempt')
    frozen = contract(); OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared', len(frozen['input_hashes']), 'direct inputs', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('Frozen V42 inputs/code/environment changed')
    return frozen


def monthly_rows(paths: dict) -> list[dict]:
    groups = dict(legacy=set(model.LEGACY_SOURCES), v4=set(model.DONOR_SOURCES[:6]), v5=set(model.DONOR_SOURCES[6:]))
    rows = []
    for group, path in paths.items():
        with open(path, newline='') as stream:
            for row in csv.DictReader(stream):
                if row['region'] in groups[group]:
                    rows.append(row)
    return rows


def capacity_index(path: str) -> dict:
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    indexed = {}
    for row in rows:
        source = row['region']
        if source not in model.ALL_SOURCES:
            raise ValueError('Only fixed seventeen inventory regions allowed')
        year = int(row['year']); key = (year, source)
        if year not in inventory.SUPPORTED_YEARS or key in indexed:
            raise ValueError('Unique registered source-year inventory required')
        c = np.array([float(row['capacity_'+f+'_mw']) for f in v30.FUELS])
        pos = c[v30.POS].sum()
        if (not np.isfinite(c).all() or np.any(c < 0) or not np.isfinite(pos) or pos <= 0 or
                not np.isclose(pos, float(row['capacity_pos_mw']), atol=1e-9, rtol=1e-12)):
            raise ValueError('V42 requires positive finite POS inventory; no source filtering')
        indexed[key] = c[v30.POS]
    if set(indexed) != {(y, s) for y in inventory.SUPPORTED_YEARS for s in model.ALL_SOURCES}:
        raise ValueError('Complete 68-row source inventory required')
    return indexed


def training(rows: list[dict], cases: dict, caps: dict, outer: str) -> dict:
    ids = np.flatnonzero(cases['regions'] != outer)
    wanted = {cases['keys'][i] for i in ids}
    selected = {}
    for row in rows:
        source = row['region']
        if source == outer or source not in model.ALL_SOURCES:
            continue
        key = v30._key(source, v30._utc(row['period_start']))
        if key not in wanted:
            continue
        if key in selected:
            raise ValueError('Duplicate source response key')
        selected[key] = row
    if set(selected) != wanted:
        raise ValueError('Complete excluded-outer source responses required')
    y, q, pos, diagnostics = [], [], [], []
    for i in ids:
        key, source = cases['keys'][i], str(cases['regions'][i])
        row = selected[key]
        ratio = v30.response(row)  # Only after outer, cohort and release restrictions.
        start = pd.Timestamp(int(cases['period_start_ns'][i]), unit='ns', tz='UTC')
        end = pd.Timestamp(int(cases['period_end_ns'][i]), unit='ns', tz='UTC')
        vintage = inventory.inventory_for_origin(start)
        c = caps[(vintage['year'], source)]; total = c.sum()
        volume, coverage = float(row['total_mwh']), float(row['coverage'])
        if not np.isfinite(volume) or volume <= 0 or not np.isfinite(coverage) or not 0 <= coverage <= 1:
            raise ValueError('Finite positive source volume and valid quality diagnostic required')
        mw = volume / ((end-start)/pd.Timedelta(hours=1)) * cases['current_g'][i, 0]
        generation = mw*ratio
        d = dict(outer=outer, key=key, source=source, coverage=coverage, inventory_year=vintage['year'],
            current_POS_MW=mw, POS_capacity_MW=total, nominal_POS_rate=mw/total)
        for f, fuel in enumerate(('coal', 'gas', 'oil', 'other')):
            d.update({f'capacity_{fuel}_MW': c[f], f'generation_{fuel}_MW': generation[f],
                f'zero_capacity_positive_generation_{fuel}': bool(c[f] == 0 and generation[f] > 0)})
        diagnostics.append(d); y.append(ratio); q.append(c/total); pos.append(total)
    return dict(ids=ids, keys=[cases['keys'][i] for i in ids], regions=cases['regions'][ids],
        X=cases['X_by_family']['Size'][ids], y=np.stack(y), q=np.stack(q), pos_capacity=np.array(pos), diagnostics=diagnostics)


def legacy_regression(cases: dict, train: dict, audit: dict, outer: str) -> dict:
    fit = next(f for f in audit['fits'] if f['arm'] == 'SizeCap-Ridge')
    v34._training_metadata(fit, outer)
    lookup = {key: i for i, key in enumerate(train['keys'])}
    own_old = [k for k in train['keys'] if k.split('/')[0] in model.LEGACY_SOURCES]
    if own_old != fit['training_keys']:
        raise ValueError('Original old source cohort keys changed')
    at = np.array([lookup[k] for k in own_old])
    ids = train['ids'][at]
    if not np.array_equal(train['y'][at], np.asarray(fit['training_y'])):
        raise ValueError('Original source responses changed')
    for old, new in (('training_period_start_ns', 'period_start_ns'), ('training_period_end_ns', 'period_end_ns'),
                     ('training_available_ns', 'available_ns')):
        if not np.array_equal(np.asarray(fit[old]), cases[new][ids]):
            raise ValueError('Original training dates changed')
    x = train['X'][at]; old_x = np.asarray(fit['training_X'])[:, :31]
    if not np.allclose(x, old_x, atol=1e-12, rtol=0):
        raise ValueError('Original Size31 features changed')
    return dict(old_keys_exact=True, old_y_exact=True, old_dates_exact=True,
        old_cases=len(own_old), old_Size_max_abs_diff=float(np.max(np.abs(x-old_x))))


def archive(path: str, outer: str) -> dict:
    names = ('arms', 'keys', 'origins', 'MG_draws', 'EF_draws', 'CIF_draws', 'EF_point', 'CIF_point',
             'query_feature_Size', 'query_feature_SizeCap', 'query_feature_DirectEF')
    with np.load(path, allow_pickle=False) as data:
        a = {k: data[k].copy() for k in names}
    if (a['arms'].tolist() != list(v34.ARMS) or a['keys'].tolist() != [f'{outer}/{d:%Y-%m}' for d in DATES] or
            not np.array_equal(a['origins'], DATES.asi8) or a['MG_draws'].shape != (6, 128, 5)):
        raise ValueError('Exact inherited query grid required')
    for key, shape in (('EF_draws', (11, 6, 128)), ('CIF_draws', (11, 6, 128)),
                       ('query_feature_Size', (6, 128, 31)), ('query_feature_SizeCap', (6, 128, 36)),
                       ('query_feature_DirectEF', (6, 128, 36))):
        if a[key].shape != shape or not np.isfinite(a[key]).all():
            raise ValueError('Finite inherited predictions/features required')
    return a


def predict() -> None:
    frozen = check()
    if (OUT / 'prediction_started.json').exists() or (OUT / 'predictions').exists():
        raise FileExistsError('Preserve V42 prediction attempt')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    folder = OUT / 'predictions'; folder.mkdir()
    rows = monthly_rows(frozen['monthly_paths'])
    cases = model.build_cases(rows)
    old_keys = [k for k in cases['keys'] if k.split('/')[0] in model.LEGACY_SOURCES]
    if len(old_keys) != 180:
        raise ValueError('Original 180 source cases must be preserved')
    caps = capacity_index(frozen['capacity_path'])
    write(OUT / 'cohort.json', dict(keys=cases['keys'], regions=cases['regions'], period_start_ns=cases['period_start_ns'],
        period_end_ns=cases['period_end_ns'], available_ns=cases['available_ns'], audit=cases['audit'],
        source_counts={s: int(np.count_nonzero(cases['regions'] == s)) for s in model.ALL_SOURCES}, hidden_labels_in_file=False))
    entries = []; diagnostics = []
    for outer in SOURCES:
        old = archive(frozen['archives'][outer]['path'], outer)
        train = training(rows, cases, caps, outer)
        regression = legacy_regression(cases, train, read(frozen['archives'][outer]['audit_path']), outer)
        query = {f: old['query_feature_'+f] for f in ('Size', 'SizeCap', 'DirectEF')}
        result = model.fit_heads(train['X'], train['q'], train['pos_capacity'], train['y'], train['regions'], query, outer=outer)
        old_ids = [list(v34.ARMS).index(arm) for arm in model.ARMS]
        ef = np.concatenate([old['EF_draws'][old_ids], result['EF_draws']])
        cif = np.concatenate([old['CIF_draws'][old_ids], result['EF_draws']*old['MG_draws'][None, ..., 0]])
        mean_diff = max(float(np.max(np.abs(old['EF_draws'][old_ids].mean(-1)-old['EF_point'][old_ids]))),
                        float(np.max(np.abs(old['CIF_draws'][old_ids].mean(-1)-old['CIF_point'][old_ids]))))
        if mean_diff > 1e-9:
            raise ValueError('Inherited mean-point consistency failed')
        path = folder / (outer+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, arms=np.array(ARMS), keys=old['keys'], origins=DATES.asi8,
                EF_draws=ef, CIF_draws=cif, MG_draws=old['MG_draws'], expanded_raw_EF=result['EF_raw'],
                expanded_projection_l1=result['projection_l1'])
        audit_path = path.with_suffix('.json')
        write(audit_path, dict(region=outer, training_keys=train['keys'], training_regions=train['regions'],
            training_period_start_ns=cases['period_start_ns'][train['ids']], training_period_end_ns=cases['period_end_ns'][train['ids']],
            training_available_ns=cases['available_ns'][train['ids']], training_Size=train['X'], training_y=train['y'],
            training_q=train['q'], training_pos_capacity=train['pos_capacity'], sample_weights=result['sample_weights'],
            legacy_regression=regression, legacy_mean_max_abs_diff=mean_diff, **result['audit']))
        diagnostics.extend(train['diagnostics'])
        entries.append(dict(region=outer, path=str(path), sha256=sha(path), audit_path=str(audit_path), audit_sha256=sha(audit_path)))
        print('Predicted', outer, 'new fits 6; training cases', len(train['keys']), 'sources', len(set(train['regions'])), flush=True)
    pd.DataFrame(diagnostics).to_csv(OUT / 'training_scope_diagnostics.csv', index=False, mode='x')
    check()
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), cohort_sha256=sha(OUT / 'cohort.json'),
        training_scope_diagnostics_sha256=sha(OUT / 'training_scope_diagnostics.csv'), entries=entries))


def predictions() -> dict:
    seal, started = read(OUT / 'prediction_seal.json'), read(OUT / 'prediction_started.json')
    if (seal['freeze_sha256'] != sha(OUT / 'freeze.json') or started['freeze_sha256'] != seal['freeze_sha256'] or
            seal['prediction_started_sha256'] != sha(OUT / 'prediction_started.json') or
            seal['cohort_sha256'] != sha(OUT / 'cohort.json') or
            seal['training_scope_diagnostics_sha256'] != sha(OUT / 'training_scope_diagnostics.csv') or
            [e['region'] for e in seal['entries']] != list(SOURCES)):
        raise ValueError('Complete V42 prediction/cohort seal required')
    folder = OUT / 'predictions'
    if {p.name for p in folder.iterdir()} != {f'{s}.{ext}' for s in SOURCES for ext in ('npz', 'json')}:
        raise ValueError('Exact prediction file set required')
    for e in seal['entries']:
        path = folder / (e['region']+'.npz')
        if e['path'] != str(path) or e['audit_path'] != str(path.with_suffix('.json')):
            raise ValueError('Exact source aliases required')
        verify({e['path']: e['sha256'], e['audit_path']: e['audit_sha256']})
        with np.load(path, allow_pickle=False) as a:
            if (a['arms'].tolist() != list(ARMS) or a['keys'].tolist() != [f"{e['region']}/{d:%Y-%m}" for d in DATES] or
                    not np.array_equal(a['origins'], DATES.asi8)):
                raise ValueError('Exact forward query grid required')
            for key, shape in (('EF_draws', (12, 6, 128)), ('CIF_draws', (12, 6, 128)), ('MG_draws', (6, 128, 5)),
                               ('expanded_raw_EF', (6, 6, 128)), ('expanded_projection_l1', (6, 6, 128))):
                if a[key].shape != shape or not np.isfinite(a[key]).all():
                    raise ValueError('Finite complete prediction arrays required')
            if not np.allclose(a['CIF_draws'], a['EF_draws']*a['MG_draws'][None, ..., 0], atol=1e-9, rtol=0):
                raise ValueError('Preserved MG multiplication required')
    return seal


def score() -> None:
    if any((OUT / name).exists() for name in (*OUTPUTS, 'completed.json')):
        raise FileExistsError('Preserve V42 score attempt')
    frozen, seal = check(), predictions()
    labels, support = v38.parent.v31.scoring_labels(common.monthly_rows(frozen))
    if len(labels) != 35 or support.to_csv(index=False) != Path(frozen['support_path']).read_text():
        raise ValueError('Original 35 defined-EF monthly support required')
    baseline = pd.read_csv(frozen['baseline_case_path'], float_precision='round_trip')
    if len(baseline) != 1680 or set(baseline.key) != set(labels):
        raise ValueError('Complete original V40 reference grid required')
    records = []
    for e in seal['entries']:
        with np.load(e['path'], allow_pickle=False) as a:
            for j, key in enumerate(a['keys']):
                if key not in labels:continue
                truth = labels[key]
                for i, arm in enumerate(ARMS):
                    points = v38.v36.point_case(a['CIF_draws'][i, j], a['EF_draws'][i, j], truth['cif'], truth['ef'])
                    records.extend(dict(region=e['region'], key=str(key), arm=arm, mode=mode, **r) for mode, r in points.items())
    case = pd.concat([baseline, pd.DataFrame(records)], ignore_index=True)
    if len(case) != 2520 or not case.iloc[:1680].equals(baseline) or case.duplicated(['region', 'key', 'arm', 'mode']).any():
        raise ValueError('Exact old references plus 840 fixed comparison cases required')
    region = v38.v36.aggregate(case, ['region', 'arm', 'mode'], ('cif', 'ef'))
    macro = region.drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index()
    leave = pd.concat([region[region.region.ne(s)].drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index().assign(omitted_source=s) for s in SOURCES], ignore_index=True)
    if (len(region), len(macro), len(leave)) != (504, 72, 504):
        raise ValueError('Complete legacy/expanded six-head aggregates required')
    for name, table in dict(case_metrics=case, region_metrics=region, macro_metrics=macro, leave_one_source_out_macro=leave, label_support=support).items():
        table.to_csv(OUT / (name+'.csv'), index=False, mode='x')
    summary = dict(new_fits=42, old_fits_retrained=0, source_months=35, query_origins=42, baseline_rows_exact=1680,
        rows=dict(case_metrics=2520, region_metrics=504, macro_metrics=72, leave_one_source_out_macro=504, label_support=42),
        source_pool_intervention_only=True, MG_retrained=False, method_innovation_claim=False,
        new_holdout_authorized=False, CIF_interval_coverage_evaluated=False, hourly_coverage_evaluated=False)
    write(OUT / 'summary.json', summary)
    check(); predictions()
    write(OUT / 'completed.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), prediction_seal_sha256=sha(OUT / 'prediction_seal.json'),
        output_hashes={str(OUT / name): sha(OUT / name) for name in OUTPUTS}, **summary))
    print('Scored six legacy and six expanded controls; no holdout opened', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'predict', 'score'), required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score}[args.stage]()
