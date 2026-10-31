"""Five fixed low-dimensional source-bias controls on legal coarse histories."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import coarse_residual_v40 as model
from scripts import coarse_inverse_v38 as profile
from scripts import run_residual_persistence_v39 as parent

v38, common = parent.parent, parent.parent.common
sha, read, write, verify, now = parent.sha, parent.read, parent.write, parent.verify, parent.now
SOURCES, DATES = v38.SOURCES, v38.DATES
PSEUDO = pd.Timestamp('2023-07-01', tz='UTC')
OUT = ROOT / 'results/coarse-residual-v40'
PARENT = ROOT / 'docs/evidence/residual-persistence-v39-verification.json'
PARENT_SHA = 'd6dd3c74e2c3c094008dee1be78934f9ff2e6afc1cde6a5b764cbe83fe5ccb26'
PREFLIGHT = ROOT / 'results/coarse-residual-v40-metadata-preflight.json'
PREFLIGHT_SHA = '99c09880b9705cfa75da8ab0418b43c541172919f575fca93ff4b2338f4904cb'
CODE = ('scripts/coarse_residual_v40.py', 'scripts/run_coarse_residual_v40.py',
        'tests/test_coarse_residual_v40.py', 'tests/test_run_coarse_residual_v40.py',
        'docs/COARSE_RESIDUAL_V40_PROTOCOL.md')
REVIEWS = tuple(ROOT / f'results/coarse-residual-v40-{kind}-review.md' for kind in ('code', 'math'))
OUTPUTS = ('case_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv',
           'leave_one_source_out_macro.csv', 'label_support.csv', 'summary.json')


def contract() -> dict:
    verify({str(PARENT): PARENT_SHA, str(PREFLIGHT): PREFLIGHT_SHA})
    evidence = read(PARENT)
    if evidence['status'] != 'VERIFICATION_COMPLETE' or evidence['independent_formula_review_status'] != 'PASS':
        raise ValueError('Verified V39 evidence required')
    closure = {}
    cursor = evidence
    seen = set()
    while True:
        for p, digest in parent.mappings(cursor).items():
            if p in closure and closure[p] != digest:
                raise ValueError('Conflicting inherited mapping')
            closure[p] = digest
        if 'parent_manifest' not in cursor:
            break
        p, h = cursor['parent_manifest'], cursor['parent_manifest_sha256']
        if p in seen:
            raise ValueError('Cyclic inherited evidence')
        seen.add(p)
        verify({p: h})
        cursor = read(p)
    verify(closure)
    paths = dict(profiles_path=str(ROOT / 'results/residual-persistence-v39/profiles.json'),
        pairs_path=str(ROOT / 'results/residual-persistence-v39/source_pairs.csv'),
        monthly_path=str(ROOT / 'results/heldout-2026h1-data/monthly.csv'),
        capacity_path=str(ROOT / 'results/fleet-inventory-v33/capacity_op.csv'),
        baseline_case_path=str(ROOT / 'results/coarse-inverse-v38/case_metrics.csv'),
        support_path=str(ROOT / 'results/coarse-inverse-v38/label_support.csv'))
    archives = {s: str(ROOT / f'results/fleet-information-v34/predictions/{s}.npz') for s in SOURCES}
    aliases = [*paths.values(), *archives.values()]
    if any(p not in closure for p in aliases):
        raise ValueError('Inherited direct aliases required')
    inputs = {p: closure[p] for p in aliases} | {str(PARENT): PARENT_SHA, str(PREFLIGHT): PREFLIGHT_SHA}
    for p in REVIEWS:
        if not p.is_file() or p.is_symlink() or not p.read_text().strip():
            raise ValueError('Two independent prelaunch reviews required')
        inputs[str(p)] = sha(p)
    return dict(version=40, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE}, input_hashes=inputs,
        environment=common.environment(), sources=list(SOURCES), archives=archives, **paths,
        arms=list(model.ARMS), new_calibration_fits=28, new_profile_fits=0, calibration_regions_per_outer=6,
        pseudo_origin=PSEUDO.isoformat(), source_only=True, new_fine_label_keys=False,
        hourly_values_read=False, unseen_values_opened=False, method_innovation_claim=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V40 attempt')
    frozen = contract()
    OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared', len(frozen['input_hashes']), 'direct inputs', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('Frozen V40 contract changed')
    return frozen


def components(fits: list[dict], pairs: list[dict], outer: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Metadata exclusion precedes converting source curves and late labels."""
    selected = [f for f in fits if f['outer'] == outer]
    labels = [r for r in pairs if r['outer'] == outer]
    others = [s for s in SOURCES if s != outer]
    if (len(selected) != 6 or len(labels) != 6 or
            {f['held_source'] for f in selected} != set(others) or {r['source'] for r in labels} != set(others)):
        raise ValueError('Exactly six matching source/profile calibrations required')
    ordered, bias = [], []
    for source in others:
        f = next(f for f in selected if f['held_source'] == source)
        r = next(r for r in labels if r['source'] == source)
        keys, regions = f['training_keys'], f['training_regions']
        if (len(keys) not in (128, 130) or len(set(keys)) != len(keys) or len(regions) != len(keys) or
                set(regions) != set(SOURCES)-{outer, source} or
                any(k.split('/')[0] != s for k, s in zip(keys, regions)) or int(r['late_count']) != 12):
            raise ValueError('Inherited five-source operator and late12 label required')
        ordered.append(np.asarray(f['profiles']['month'], float))
        bias.append(float(r['late_EFbias']))
    curves, y = np.stack(ordered), np.array(bias)
    if curves.shape != (6, 12, 4) or not np.isfinite(curves).all() or np.any(curves < 0) or not np.isfinite(y).all():
        raise ValueError('Finite aligned source curves and EF-bias responses required')
    return others, curves, y


def context_features(index: dict, capacities: dict, source: str, origin: pd.Timestamp,
                     curve: np.ndarray) -> tuple[np.ndarray, dict]:
    records = v38.history(index, source, origin)
    c = np.stack([v38.capacity(capacities, source, r[1]) for r in records])
    g = np.array([v38.pos_mw(r) for r in records])
    months = np.array([r[1].month for r in records])
    u = profile.profile_predict(curve, months)
    values = model.features(c, g, u)
    audit = dict(source=source, origin=origin.isoformat(), keys=[v38.v30._key(source, r[1]) for r in records],
        starts=[r[1].isoformat() for r in records], ends=[r[2].isoformat() for r in records],
        releases=[r[3].isoformat() for r in records], months=months,
        newest_month_age=origin.year*12+origin.month-records[-1][1].year*12-records[-1][1].month,
        calendar_span_months=records[-1][1].year*12+records[-1][1].month-records[0][1].year*12-records[0][1].month+1,
        capacities=c, POS_MW=g, profile_U=u, **values['audit'])
    return values['X'], audit


def mg_archive(path: str, source: str) -> dict:
    with np.load(path, allow_pickle=False) as a:
        values = {k: a[k].copy() for k in ('keys', 'origins', 'MG_draws')}
    g = values['MG_draws']
    if (values['keys'].tolist() != [f'{source}/{at:%Y-%m}' for at in DATES] or
            not np.array_equal(values['origins'], DATES.asi8) or g.shape != (6, 128, 5) or
            not np.isfinite(g).all() or np.any(g < 0) or not np.allclose(g.sum(-1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Original complete MG source query grid required')
    return values


def predict() -> None:
    frozen = check()
    if (OUT / 'prediction_started.json').exists() or (OUT / 'predictions').exists():
        raise FileExistsError('Preserve V40 prediction attempt')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    folder = OUT / 'predictions'
    folder.mkdir()
    index = v38.monthly_index(common.monthly_rows(frozen))
    with open(frozen['capacity_path'], newline='') as stream:
        capacities = v38.v36.v34._capacity_values(v38.v36.v34._capacity_metadata(list(csv.DictReader(stream))))
    fits = read(frozen['profiles_path'])['fits']
    with open(frozen['pairs_path'], newline='') as stream:
        pairs = list(csv.DictReader(stream))
    if len(fits) != 42 or len(pairs) != 42:
        raise ValueError('Exact V39 nested source grid required')
    preflight = {r['source']: r for r in read(PREFLIGHT)['entries']}
    entries = []
    for source in SOURCES:
        held_sources, curves, bias = components(fits, pairs, source)
        calibration, source_audits = [], []
        for held, curve in zip(held_sources, curves):
            x, audit = context_features(index, capacities, held, PSEUDO, curve)
            if any(audit[k] != preflight[held][k] for k in ('keys', 'starts', 'ends', 'releases', 'newest_month_age', 'calendar_span_months')):
                raise ValueError('Exact source legal12 metadata preflight required')
            calibration.append(x); source_audits.append(audit)
        query, base = np.empty((6, 6, 2)), np.empty((6, 6))
        query_audits, query_caps = [], []
        for j, origin in enumerate(DATES):
            c = v38.capacity(capacities, source, origin)
            query_caps.append(c)
            audits = []
            for k, curve in enumerate(curves):
                query[j, k], audit = context_features(index, capacities, source, origin, curve)
                base[j, k] = profile.decode(c*profile.profile_predict(curve, np.array([origin.month]))[0], np.ones(4))['EF']
                audits.append(audit)
            query_audits.append(audits)
        result = model.fit_predict(np.stack(calibration), bias, query, base)
        archived = mg_archive(frozen['archives'][source], source)
        ef = result['ensemble_EF']
        cif = ef[..., None]*archived['MG_draws'][None, ..., 0]
        path = folder / (source+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, arms=np.array(model.ARMS), keys=archived['keys'], origins=DATES.asi8,
                EF_draws=np.broadcast_to(ef[..., None], cif.shape).copy(), CIF_draws=cif,
                raw_component_EF=result['raw_component_EF'], component_EF=result['component_EF'],
                clipmask=result['clipmask'], ensemble_EF=ef)
        audit_path = path.with_suffix('.json')
        write(audit_path, dict(region=source, calibration_sources=held_sources, profile_curves=curves,
            calibration_X=np.stack(calibration), calibration_bias_g=bias, query_X=query, query_base_EF=base,
            source_contexts=source_audits, query_contexts=query_audits, query_capacities=query_caps, **result['audit']))
        entries.append(dict(region=source, path=str(path), sha256=sha(path), audit_path=str(audit_path), audit_sha256=sha(audit_path)))
        print('Predicted', source, 'four fixed source calibrators', flush=True)
    check()
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), entries=entries))


def predictions() -> dict:
    seal, started = read(OUT / 'prediction_seal.json'), read(OUT / 'prediction_started.json')
    if (seal['freeze_sha256'] != sha(OUT / 'freeze.json') or started['freeze_sha256'] != seal['freeze_sha256'] or
            seal['prediction_started_sha256'] != sha(OUT / 'prediction_started.json') or
            [e['region'] for e in seal['entries']] != list(SOURCES)):
        raise ValueError('Complete source seal and start receipt required')
    folder = OUT / 'predictions'
    if {p.name for p in folder.iterdir()} != {f'{s}.{ext}' for s in SOURCES for ext in ('npz', 'json')}:
        raise ValueError('Exact sealed prediction file set required')
    for e in seal['entries']:
        path = folder / (e['region']+'.npz')
        if e['path'] != str(path) or e['audit_path'] != str(path.with_suffix('.json')):
            raise ValueError('Exact source prediction aliases required')
        verify({e['path']: e['sha256'], e['audit_path']: e['audit_sha256']})
        with np.load(path, allow_pickle=False) as a:
            if (a['arms'].tolist() != list(model.ARMS) or not np.array_equal(a['origins'], DATES.asi8) or
                    a['keys'].tolist() != [f"{e['region']}/{at:%Y-%m}" for at in DATES]):
                raise ValueError('Exact future query grid required')
            for name, shape in (('EF_draws', (5, 6, 128)), ('CIF_draws', (5, 6, 128)),
                                ('raw_component_EF', (5, 6, 6)), ('component_EF', (5, 6, 6)), ('ensemble_EF', (5, 6))):
                if a[name].shape != shape or not np.isfinite(a[name]).all():
                    raise ValueError('Finite complete future predictions required')
            if (a['clipmask'].dtype != np.dtype(bool) or a['clipmask'].shape != (5, 6, 6) or
                    not np.array_equal(a['component_EF'], np.clip(a['raw_component_EF'], 400., 1000.)) or
                    not np.array_equal(a['clipmask'], a['raw_component_EF'] != a['component_EF']) or
                    not np.array_equal(a['ensemble_EF'], a['component_EF'].mean(-1)) or
                    not np.array_equal(a['EF_draws'], np.broadcast_to(a['ensemble_EF'][..., None], (5, 6, 128)))):
                raise ValueError('Clip-before-ensemble and EF particles required')
    return seal


def score() -> None:
    if any((OUT / p).exists() for p in (*OUTPUTS, 'completed.json')):
        raise FileExistsError('Preserve V40 score attempt')
    frozen, seal = check(), predictions()
    labels, support = v38.parent.v31.scoring_labels(common.monthly_rows(frozen))
    if len(labels) != 35 or support.to_csv(index=False) != Path(frozen['support_path']).read_text():
        raise ValueError('Original defined-EF 35-month score support required')
    baseline = pd.read_csv(frozen['baseline_case_path'], float_precision='round_trip')
    if len(baseline) != 1330 or set(baseline.key) != set(labels) or baseline.duplicated(['region', 'key', 'arm', 'mode']).any():
        raise ValueError('Complete unchanged V38 references required')
    records = []
    for e in seal['entries']:
        with np.load(e['path'], allow_pickle=False) as a:
            for j, key in enumerate(a['keys']):
                if key not in labels:
                    continue
                truth = labels[key]
                for i, arm in enumerate(model.ARMS):
                    points = v38.v36.point_case(a['CIF_draws'][i, j], a['EF_draws'][i, j], truth['cif'], truth['ef'])
                    records.extend(dict(region=e['region'], key=str(key), arm=arm, mode=mode, **r) for mode, r in points.items())
    case = pd.concat([baseline, pd.DataFrame(records)], ignore_index=True)
    if len(case) != 1680 or not case.iloc[:1330].equals(baseline) or case.duplicated(['region', 'key', 'arm', 'mode']).any():
        raise ValueError('Exact old references plus 350 new rows required')
    region = v38.v36.aggregate(case, ['region', 'arm', 'mode'], ('cif', 'ef'))
    macro = region.drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index()
    leave = pd.concat([region[region.region.ne(s)].drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index().assign(omitted_source=s) for s in SOURCES], ignore_index=True)
    if (len(region), len(macro), len(leave)) != (336, 48, 336):
        raise ValueError('Complete five-arm aggregate comparison required')
    tables = dict(case_metrics=case, region_metrics=region, macro_metrics=macro, leave_one_source_out_macro=leave, label_support=support)
    for name, table in tables.items():
        table.to_csv(OUT / (name+'.csv'), mode='x', index=False)
    summary = dict(new_calibration_fits=28, new_profile_fits=0, calibration_regions_per_outer=6,
        source_months=35, query_origins=42, baseline_rows_exact=1330, table_rows={k: len(t) for k, t in tables.items()},
        method_innovation_claim=False, new_holdout_authorized=False, CIF_interval_coverage_evaluated=False, hourly_coverage_evaluated=False)
    write(OUT / 'summary.json', summary)
    check(); predictions()
    write(OUT / 'completed.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), prediction_seal_sha256=sha(OUT / 'prediction_seal.json'),
        output_hashes={str(OUT / p): sha(OUT / p) for p in OUTPUTS}, **summary))
    print('Scored five fixed source-bias controls on original 35 months', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'predict', 'score'), required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score}[args.stage]()
