"""Small source-only monthly inverse-profile falsification, with sealed predictions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import coarse_inverse_v38 as model
from scripts import coarse_context_v30 as v30
from scripts import run_utilization_decoder_v37 as parent

v36, common = parent.v36, parent.common
sha, read, write, verify, now = parent.sha, parent.read, parent.write, parent.verify, parent.now
SOURCES, DATES = parent.SOURCES, parent.DATES
PROFILES, KINDS = ('Constant', 'Month'), ('None', 'Common', 'FuelRidge')
ARMS = tuple(f'{p}-{k}' for p in PROFILES for k in KINDS)
OUT = ROOT / 'results/coarse-inverse-v38'
PARENT = ROOT / 'docs/evidence/utilization-decoder-v37-verification.json'
PARENT_SHA = '639d54297999aad035bedad5f9ba0795d9448af79be4ddda264637690aae62ad'
CODE = ('scripts/coarse_inverse_v38.py', 'scripts/run_coarse_inverse_v38.py',
        'tests/test_coarse_inverse_v38.py', 'tests/test_run_coarse_inverse_v38.py',
        'docs/COARSE_INVERSE_V38_PROTOCOL.md')
REVIEWS = tuple(ROOT / f'results/coarse-inverse-v38-{kind}-review.md' for kind in ('code', 'math'))
OUTPUTS = ('case_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv',
           'leave_one_source_out_macro.csv', 'label_support.csv', 'coarse_errors.csv',
           'profile_errors.csv', 'operator_sensitivity.csv', 'summary.json')


def contract() -> dict:
    verify({str(PARENT): PARENT_SHA})
    evidence = read(PARENT)
    if evidence['status'] != 'VERIFICATION_COMPLETE' or evidence['independent_formula_review_status'] != 'PASS':
        raise ValueError('Verified parent required')
    closure = {}
    for group in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
        for path, digest in evidence[group].items():
            if path in closure and closure[path] != digest:
                raise ValueError('Conflicting parent hash')
            closure[path] = digest
    verify(closure)
    old = read(ROOT / 'results/utilization-decoder-v37/freeze.json')
    capacity = str(ROOT / 'results/fleet-inventory-v33/capacity_op.csv')
    baseline = str(ROOT / 'results/utilization-decoder-v37/case_metrics.csv')
    aliases = [capacity, baseline, old['monthly_path'], old['support_path']]
    aliases += [e[k] for e in old['entries'] for k in ('path', 'audit_path')]
    if [e['region'] for e in old['entries']] != list(SOURCES) or any(p not in closure for p in aliases):
        raise ValueError('Exact inherited direct inputs required')
    inputs = {p: closure[p] for p in aliases}
    inputs[str(PARENT)] = PARENT_SHA
    for path in REVIEWS:
        if not path.is_file() or path.is_symlink() or not path.read_text().strip():
            raise ValueError('Two independent prelaunch reviews required')
        inputs[str(path)] = sha(path)
    return dict(version=38, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE},
        input_hashes=inputs, parent_closure_verified_files=len(closure), environment=common.environment(),
        entries=old['entries'], capacity_path=capacity, monthly_path=old['monthly_path'],
        baseline_case_path=baseline, support_path=old['support_path'], arms=list(ARMS),
        source_only=True, additional_source_response='same-key current total_mwh',
        new_fine_label_keys=False, hourly_values_read=False, unseen_values_opened=False,
        method_innovation_claim=False, new_holdout_authorized=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V38 attempt')
    frozen = contract()
    OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared', len(frozen['input_hashes']), 'direct inputs', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('Frozen V38 contract changed')
    return frozen


def monthly_index(rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        source = row['region']
        if source not in SOURCES:
            continue
        start = v30._utc(row['period_start'])
        if start < v30.START:
            continue
        end, release = v30._utc(row['period_end']), v30._utc(row['available_at'])
        if start != start.normalize() or start.day != 1 or end != start + pd.offsets.MonthBegin(1) or release < end:
            raise ValueError('Whole month and valid release required')
        key = v30._key(source, start)
        if key in result:
            raise ValueError('Duplicate monthly key')
        result[key] = (source, start, end, release, row)
    return result


def history(index: dict, source: str, origin: pd.Timestamp) -> list:
    eligible = [r for r in index.values() if r[0] == source and r[2] <= origin and r[3] <= origin]
    selected = sorted(eligible, key=lambda r: r[1])[-12:]
    if len(selected) != 12:
        raise ValueError('Exactly twelve legal past monthly records required')
    return selected


def pos_mw(record: tuple) -> float:
    total = float(record[4]['total_mwh'])
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Finite positive monthly total required')
    return total / ((record[2] - record[1]) / pd.Timedelta(hours=1)) * v30._visible(record[4])[0]


def capacity(capacities: dict, source: str, at: pd.Timestamp) -> np.ndarray:
    vintage = v36.inventory.inventory_for_origin(at)
    return capacities[(vintage['year'], source)][v30.POS].copy()


def training(fit: dict, index: dict, capacities: dict, source: str) -> dict:
    v36.v34._training_metadata(fit, source)
    keys = fit['training_keys']
    if len(keys) != (156 if source == 'FPL' else 154):
        raise ValueError('Original training cohort required')
    records = [index[k] for k in keys]
    for r, s, start, end, release in zip(records, fit['training_regions'], fit['training_period_start_ns'],
                                       fit['training_period_end_ns'], fit['training_available_ns']):
        if (r[0] != s or r[0] == source or [r[1].value, r[2].value, r[3].value] != [start, end, release]):
            raise ValueError('Matching excluded-outer training metadata required')
    y = np.asarray(fit['training_y'], float)
    v30._simplex(y[0], 4)
    if y.shape != (len(keys), 4) or not np.isfinite(y).all() or (y < 0).any() or not np.allclose(y.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError('Frozen response simplex required')
    g = np.stack([pos_mw(r) * yy for r, yy in zip(records, y)])
    c = np.stack([capacity(capacities, r[0], r[1]) for r in records])
    return dict(generation=g, capacities=c, months=np.array([r[1].month for r in records]),
        weights=np.asarray(fit['sample_weights']), keys=keys)


def predict() -> None:
    frozen = check()
    folder = OUT / 'predictions'
    if folder.exists() or (OUT / 'prediction_started.json').exists():
        raise FileExistsError('Preserve V38 prediction attempt')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    folder.mkdir()
    index = monthly_index(common.monthly_rows(frozen))
    with open(frozen['capacity_path'], newline='') as stream:
        capacities = v36.v34._capacity_values(v36.v34._capacity_metadata(list(csv.DictReader(stream))))
    entries = []
    for entry in frozen['entries']:
        source = entry['region']
        archived, audit = v36.load_archive(entry['path'], source), read(entry['audit_path'])
        v36.source_inputs(audit, archived, source)
        fit = next(f for f in audit['fits'] if f['arm'] == 'SizeCap-Ridge')
        train = training(fit, index, capacities, source)
        profiles = model.fit_profiles(train['generation'], train['capacities'], train['months'], train['weights'], train['keys'])
        profiles['audit']['source_support_by_fuel'] = {
            fuel: {s: dict(count=int(np.count_nonzero(profiles['valid'][:, f] &
                            (np.asarray(fit['training_regions']) == s))),
                          weight=float(train['weights'][profiles['valid'][:, f] &
                            (np.asarray(fit['training_regions']) == s)].sum()))
                   for s in SOURCES if s != source}
            for f, fuel in enumerate(('coal', 'gas', 'oil', 'other'))}
        ef, ratios, coarse = np.empty((6, 6)), np.empty((6, 6, 4)), np.empty((6, 6))
        contexts, solves = [], []
        for j, origin in enumerate(DATES):
            past = history(index, source, origin)
            c = np.stack([capacity(capacities, source, r[1]) for r in past])
            y = np.array([pos_mw(r) for r in past])
            ct = capacity(capacities, source, origin)
            contexts.append(dict(origin=origin.isoformat(), keys=[v30._key(source, r[1]) for r in past],
                starts=[r[1].isoformat() for r in past], releases=[r[3].isoformat() for r in past],
                newest_month_age=origin.year*12+origin.month-past[-1][1].year*12-past[-1][1].month,
                history_span_months=past[-1][1].year*12+past[-1][1].month-past[0][1].year*12-past[0][1].month+1,
                history_capacity=c, query_capacity=ct, POS_MW=y))
            for pi, profile_name in enumerate(PROFILES):
                profile = profiles['global' if pi == 0 else 'month']
                u = model.profile_predict(profile, np.array([r[1].month for r in past]))
                B, z = c * u / c.sum(1)[:, None], y / c.sum(1)
                b = ct * model.profile_predict(profile, np.array([origin.month]))[0]
                diagnostics = model.operator_diagnostics(B)
                for ki, kind in enumerate(KINDS):
                    i = pi*3 + ki
                    solution = model.adapt(B, z, kind)
                    if kind == 'Common':
                        ef[i, j], ratios[i, j] = ef[pi*3, j], ratios[pi*3, j]
                    else:
                        decoded = model.decode(b, solution['gamma'])
                        ef[i, j], ratios[i, j] = decoded['EF'], decoded['ratios']
                    coarse[i, j] = b @ solution['gamma']
                    solves.append(dict(arm=ARMS[i], origin=origin.isoformat(), B=B, z=z, b=b,
                        diagnostics=diagnostics, **solution))
        cif = ef[..., None] * archived['MG_draws'][None, ..., 0]
        ef_draws = np.broadcast_to(ef[..., None], cif.shape).copy()
        path = folder / f'{source}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, arms=np.array(ARMS), keys=archived['keys'], origins=DATES.asi8,
                EF_draws=ef_draws, CIF_draws=cif, ratios=ratios, coarse_POS_MW=coarse)
        audit_path = path.with_suffix('.json')
        write(audit_path, dict(region=source, training_keys=train['keys'], training_generation_MW=train['generation'],
            training_capacity_MW=train['capacities'], training_months=train['months'], sample_weights=train['weights'],
            profiles=profiles, contexts=contexts, solves=solves))
        entries.append(dict(region=source, path=str(path), sha256=sha(path), audit_path=str(audit_path), audit_sha256=sha(audit_path)))
        print('Predicted', source, 'two profiles, 36 fixed solves', flush=True)
    check()
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), entries=entries))


def predictions() -> dict:
    seal = read(OUT / 'prediction_seal.json')
    marker = read(OUT / 'prediction_started.json')
    if (seal['prediction_started_sha256'] != sha(OUT / 'prediction_started.json') or
            marker['freeze_sha256'] != sha(OUT / 'freeze.json')):
        raise ValueError('Matching prediction start receipt required')
    if seal['freeze_sha256'] != sha(OUT / 'freeze.json') or [e['region'] for e in seal['entries']] != list(SOURCES):
        raise ValueError('All seven sealed predictions required')
    expected = {f'{s}.{suffix}' for s in SOURCES for suffix in ('npz', 'json')}
    if {p.name for p in (OUT / 'predictions').iterdir()} != expected:
        raise ValueError('Exact prediction file set required')
    for e in seal['entries']:
        path = OUT / 'predictions' / (e['region']+'.npz')
        if e['path'] != str(path) or e['audit_path'] != str(path.with_suffix('.json')):
            raise ValueError('Exact sealed prediction aliases required')
        verify({e['path']: e['sha256'], e['audit_path']: e['audit_sha256']})
        with np.load(e['path'], allow_pickle=False) as a:
            if (a['arms'].tolist() != list(ARMS) or
                    a['keys'].tolist() != [f"{e['region']}/{at:%Y-%m}" for at in DATES] or
                    not np.array_equal(a['origins'], DATES.asi8)):
                raise ValueError('Exact source-month query grid required')
            for name, shape in (('EF_draws', (6, 6, 128)), ('CIF_draws', (6, 6, 128)),
                                ('ratios', (6, 6, 4)), ('coarse_POS_MW', (6, 6))):
                if a[name].shape != shape or not np.isfinite(a[name]).all():
                    raise ValueError('Finite complete prediction shapes required')
            if (np.any(a['ratios'] < 0) or not np.allclose(a['ratios'].sum(-1), 1., atol=1e-12, rtol=0)
                    or np.any(a['coarse_POS_MW'] < 0)):
                raise ValueError('Prediction mass and ratio validity required')
            for pi in range(2):
                for name in ('EF_draws', 'CIF_draws', 'ratios'):
                    if not np.array_equal(a[name][pi*3], a[name][pi*3+1]):
                        raise ValueError('Common must leave EF/CIF and ratios unchanged')
    return seal


def score() -> None:
    if any((OUT / p).exists() for p in (*OUTPUTS, 'source_completed.json')):
        raise FileExistsError('Preserve V38 score attempt')
    frozen, seal = check(), predictions()
    raw = common.monthly_rows(frozen)
    labels, support = parent.v31.scoring_labels(raw)
    if len(labels) != 35 or support.to_csv(index=False) != Path(frozen['support_path']).read_text():
        raise ValueError('Original 35 scored month support required')
    baseline = pd.read_csv(frozen['baseline_case_path'], float_precision='round_trip')
    if len(baseline) != 910 or set(baseline.key) != set(labels):
        raise ValueError('Complete V37 reference grid required')
    records, coarse = [], []
    index = monthly_index(raw)
    # Only after the complete prediction seal, collect the original source-label pool.
    pool = {}
    for e in frozen['entries']:
        fit = next(f for f in read(e['audit_path'])['fits'] if f['arm'] == 'SizeCap-Ridge')
        for key, y in zip(fit['training_keys'], fit['training_y']):
            if key in pool and not np.array_equal(pool[key], y):
                raise ValueError('Inconsistent original source response pool')
            pool[key] = np.array(y)
    if len(pool) != 180:
        raise ValueError('Original 180-key pool required')
    with open(frozen['capacity_path'], newline='') as stream:
        capacities = v36.v34._capacity_values(v36.v34._capacity_metadata(list(csv.DictReader(stream))))
    errors, sensitivity = [], []
    for e in seal['entries']:
        source, audit = e['region'], read(e['audit_path'])
        with np.load(e['path'], allow_pickle=False) as a:
            if a['arms'].tolist() != list(ARMS) or a['CIF_draws'].shape != (6, 6, 128):
                raise ValueError('Fixed six-arm query grid required')
            for pi in range(2):
                if not np.array_equal(a['EF_draws'][pi*3], a['EF_draws'][pi*3+1]) or not np.array_equal(a['CIF_draws'][pi*3], a['CIF_draws'][pi*3+1]):
                    raise ValueError('Common must leave EF and CIF exactly unchanged')
            for j, key in enumerate(a['keys']):
                if key not in labels:
                    continue
                truth = labels[key]
                for i, arm in enumerate(ARMS):
                    points = v36.point_case(a['CIF_draws'][i, j], a['EF_draws'][i, j], truth['cif'], truth['ef'])
                    records.extend(dict(region=source, key=str(key), arm=arm, mode=mode, **r) for mode, r in points.items())
                    actual, predicted = pos_mw(index[str(key)]), float(a['coarse_POS_MW'][i, j])
                    coarse.append(dict(region=source, key=str(key), arm=arm, actual=actual, predicted=predicted,
                        bias=predicted-actual, abs_error=abs(predicted-actual)))
        for pi, name in enumerate(PROFILES):
            row_norms = []
            for key, yy in sorted(pool.items()):
                r = index[key]
                if r[0] != source:
                    continue
                c = capacity(capacities, source, r[1])
                g = pos_mw(r)*yy
                if np.any((c == 0) & (g > 0)):
                    raise ValueError('Positive fine generation without registered capacity: '+key)
                profile = np.array(audit['profiles']['global' if pi == 0 else 'month'])
                predicted = model.profile_predict(profile, np.array([r[1].month]))[0]
                operator_error = (g - c*predicted)/c.sum()
                norm = float(np.linalg.norm(operator_error))
                row_norms.append(norm)
                for f, fuel in enumerate(('coal', 'gas', 'oil', 'other')):
                    errors.append(dict(region=source, key=key, profile=name, fuel=fuel, capacity=c[f], generation=g[f],
                        rate_defined=bool(c[f] > 0), rate_true=g[f]/c[f] if c[f] > 0 else np.nan,
                        rate_pred=predicted[f], rate_error=(g[f]/c[f]-predicted[f]) if c[f] > 0 else np.nan,
                        operator_error=operator_error[f], operator_row_norm=norm))
            rms = float(np.sqrt(np.mean(np.square(row_norms))))
            for solve in audit['solves']:
                if solve['arm'] == name+'-None':
                    sv = np.array(solve['diagnostics']['singular_values'])
                    sensitivity.append(dict(region=source, origin=solve['origin'], profile=name,
                        min_sv_per_sqrt12=sv[-1]/np.sqrt(12), max_sv_per_sqrt12=sv[0]/np.sqrt(12),
                        rank=solve['diagnostics']['rank'], held_source_operator_row_RMS=rms,
                        diagnostic_only_not_uncertainty_bound=True))
    case = pd.concat([baseline, pd.DataFrame(records)], ignore_index=True)
    if len(case) != 1330 or not case.iloc[:910].equals(baseline) or case.duplicated(['region', 'key', 'arm', 'mode']).any():
        raise ValueError('Complete unchanged references and 420 new cases required')
    region = v36.aggregate(case, ['region', 'arm', 'mode'], ('cif', 'ef'))
    macro = region.drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index()
    leave = pd.concat([region[region.region.ne(s)].drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index().assign(omitted_source=s) for s in SOURCES], ignore_index=True)
    if (len(region), len(macro), len(leave), len(coarse), len(errors), len(sensitivity)) != (266, 38, 266, 210, 1440, 84):
        raise ValueError('Fixed result dimensions required')
    tables = dict(case_metrics=case, region_metrics=region, macro_metrics=macro,
        leave_one_source_out_macro=leave, label_support=support, coarse_errors=pd.DataFrame(coarse),
        profile_errors=pd.DataFrame(errors), operator_sensitivity=pd.DataFrame(sensitivity))
    for name, table in tables.items():
        table.to_csv(OUT / (name+'.csv'), index=False, mode='x')
    summary = dict(source_profile_fits=14, fuel_ridge_solves=84, common_solves=84, prediction_origins=42,
        table_rows={name: len(table) for name, table in tables.items()}, baseline_rows_exact=910,
        method_innovation_claim=False, CIF_interval_coverage_evaluated=False, hourly_coverage_evaluated=False,
        new_holdout_authorized=False, source_profile_error_is_certified_target_operator_error=False)
    write(OUT / 'summary.json', summary)
    check()
    predictions()
    write(OUT / 'source_completed.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_seal_sha256=sha(OUT / 'prediction_seal.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'),
        output_hashes={str(OUT / p): sha(OUT / p) for p in OUTPUTS}, **summary))
    print('Scored six profile/adaptation arms; all hourly and unknown labels remain closed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'predict', 'score'))
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score}[args.stage]()
