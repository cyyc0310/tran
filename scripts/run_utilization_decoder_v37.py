"""Three source-only CCE decoder baselines on the sealed V36 comparison grid."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import utilization_decoder_v37 as model
from scripts import run_capacity_routing_v36 as v36

OUT = ROOT / 'results/utilization-decoder-v37'
PARENT = ROOT / 'docs/evidence/capacity-routing-v36-verification.json'
PARENT_SHA = 'a7d9466339a40a3db9ffd80747530553e35c3ec2d34d68187041f26789c80aee'
SOURCES, DATES = v36.SOURCES, v36.DATES
ARMS = tuple(v36.model.RECIPES) + tuple(model.ARMS)
CODE = ('scripts/utilization_decoder_v37.py', 'scripts/run_utilization_decoder_v37.py',
        'tests/test_utilization_decoder_v37.py', 'tests/test_run_utilization_decoder_v37.py',
        'docs/UTILIZATION_DECODER_V37_PROTOCOL.md')
REVIEWS = tuple(ROOT / f'results/utilization-decoder-v37-{name}-review-r1.md' for name in ('code', 'math'))
OUTPUTS = ('case_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv',
           'leave_one_source_out_macro.csv', 'label_support.csv', 'summary.json')
common, v31 = v36.common, v36.v31
sha, read, write, verify, now = v36.sha, v36.read, v36.write, v36.verify, v36.now


def contract() -> dict:
    verify({str(PARENT): PARENT_SHA})
    evidence = read(PARENT)
    if evidence.get('status') != 'VERIFICATION_COMPLETE' or evidence.get('independent_formula_review_status') != 'PASS':
        raise ValueError('Audited V36 baseline closure required')
    inputs = {str(PARENT): PARENT_SHA}
    for field in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
        for path, digest in evidence[field].items():
            if path in inputs and inputs[path] != digest:
                raise ValueError('Conflicting parent mapping')
            inputs[path] = digest
    verify(inputs)
    old = read(ROOT / 'results/capacity-routing-v36/freeze.json')
    aliases = [old['monthly_path'], *[entry[key] for entry in old['entries'] for key in ('path', 'audit_path')]]
    baseline = str(ROOT / 'results/capacity-routing-v36/case_metrics.csv')
    support = str(ROOT / 'results/capacity-routing-v36/label_support.csv')
    if [e['region'] for e in old['entries']] != list(SOURCES) or any(p not in inputs for p in [*aliases, baseline, support]):
        raise ValueError('Exact seven-source, baseline and support aliases required')
    for path in REVIEWS:
        if not path.is_file() or path.is_symlink() or not path.read_text().strip():
            raise ValueError('Two final independent code/math reviews required')
        inputs[str(path)] = sha(path)
    code = {str(ROOT / path): sha(ROOT / path) for path in CODE}
    verify(code)
    return dict(version=37, code_hashes=code, input_hashes=inputs, environment=common.environment(),
        entries=old['entries'], monthly_path=old['monthly_path'], baseline_case_path=baseline,
        support_path=support, sources=list(SOURCES), arms=list(model.ARMS), new_fits=21,
        source_only=True, new_training_labels_used=False, method_innovation_claim=False,
        hourly_values_read=False, target_values_opened=False, unseen_values_opened=False)


def prepare() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V37 attempt')
    frozen = contract()
    OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    print('Prepared', len(frozen['code_hashes']), 'code', len(frozen['input_hashes']), 'input hashes', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if (set(frozen) != set(expected) | {'created_utc'} or
            any(frozen[key] != value for key, value in expected.items()) or
            any(frozen[key] is not value for key, value in expected.items() if isinstance(value, bool))):
        raise ValueError('Frozen V37 inputs/code/environment changed')
    return frozen


def predict() -> None:
    frozen = check()
    if (OUT / 'predictions').exists() or (OUT / 'prediction_seal.json').exists():
        raise FileExistsError('Preserve V37 prediction attempt')
    initial = sha(OUT / 'freeze.json')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=initial))
    folder = OUT / 'predictions'
    folder.mkdir()
    entries = []
    for entry in frozen['entries']:
        source = entry['region']
        values, audit = v36.load_archive(entry['path'], source), read(entry['audit_path'])
        train_q, query_q, _, _ = v36.source_inputs(audit, values, source)
        fit = next(f for f in audit['fits'] if f['arm'] == 'SizeCap-Ridge')
        peer = next(f for f in audit['fits'] if f['arm'] == 'SizeCap-ET')
        fields = ('training_keys', 'training_regions', 'training_X', 'training_y', 'sample_weights',
                  'training_period_start_ns', 'training_period_end_ns', 'training_available_ns')
        if any(not np.array_equal(fit[field], peer[field]) for field in fields):
            raise ValueError('Same frozen training cohort/labels/weights required')
        result = model.fit_predict(np.asarray(fit['training_X']), np.asarray(fit['training_y']),
            np.asarray(fit['sample_weights']), train_q, values['query_feature_SizeCap'], query_q,
            list(fit['training_keys']))
        ef = result['EF_draws']
        cif = ef * values['MG_draws'][None, ..., 0]
        path = folder / f'{source}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, arms=np.asarray(model.ARMS), keys=values['keys'], origins=DATES.asi8,
                ratios=result['ratios'], EF_draws=ef, CIF_draws=cif, EF_mean=ef.mean(-1), CIF_mean=cif.mean(-1),
                EF_median=np.median(ef, axis=-1), CIF_median=np.median(cif, axis=-1))
        audit_path = path.with_suffix('.json')
        write(audit_path, dict(region=source, excluded_outer=source, parent_audit_path=entry['audit_path'],
            training_parent_arm='SizeCap-Ridge', feature_names=fit['feature_names'],
            training_metadata={field: fit[field] for field in fields if field not in ('training_X', 'training_y')},
            **result['audit']))
        entries.append(dict(region=source, path=str(path), sha256=sha(path), audit_path=str(audit_path), audit_sha256=sha(audit_path)))
        print('Predicted', source, 'new fits', len(result['audit']['fits']), flush=True)
    check()
    verify({str(OUT / 'freeze.json'): initial})
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=initial,
        prediction_started_sha256=sha(OUT / 'prediction_started.json'), entries=entries,
        new_fits=21, complete_grid=True, scoring_labels_read=False, target_values_opened=False,
        unseen_values_opened=False, hourly_values_read=False))


def predictions() -> dict:
    seal = read(OUT / 'prediction_seal.json')
    verify({str(OUT / 'prediction_started.json'): seal['prediction_started_sha256']})
    if (seal['freeze_sha256'] != sha(OUT / 'freeze.json') or seal['complete_grid'] is not True or
            read(OUT / 'prediction_started.json')['freeze_sha256'] != seal['freeze_sha256'] or seal['new_fits'] != 21 or
            any(seal[key] is not False for key in ('scoring_labels_read', 'target_values_opened', 'unseen_values_opened', 'hourly_values_read')) or
            [e['region'] for e in seal['entries']] != list(SOURCES)):
        raise ValueError('Complete seven-source prediction seal required before score labels')
    folder = OUT / 'predictions'
    if {p.name for p in folder.iterdir()} != {f'{s}.{suffix}' for s in SOURCES for suffix in ('npz', 'json')}:
        raise ValueError('Exact fourteen prediction files required')
    for entry in seal['entries']:
        path = folder / f"{entry['region']}.npz"
        if entry['path'] != str(path) or entry['audit_path'] != str(path.with_suffix('.json')):
            raise ValueError('Prediction aliases changed')
        verify({entry['path']: entry['sha256'], entry['audit_path']: entry['audit_sha256']})
        with np.load(path, allow_pickle=False) as a:
            if (a['arms'].tolist() != list(model.ARMS) or not np.array_equal(a['origins'], DATES.asi8) or
                    a['keys'].tolist() != [f"{entry['region']}/{at:%Y-%m}" for at in DATES] or
                    any(a[k].shape != (3, 6, 128) or not np.isfinite(a[k]).all() for k in ('EF_draws', 'CIF_draws')) or
                    any(a[k].shape != (3, 6) or not np.isfinite(a[k]).all() for k in ('EF_mean', 'CIF_mean', 'EF_median', 'CIF_median')) or
                    a['ratios'].shape != (3, 6, 128, 4) or not np.isfinite(a['ratios']).all() or
                    np.any(a['ratios'] < 0) or not np.allclose(a['ratios'].sum(-1), 1, atol=1e-12, rtol=0)):
                raise ValueError('Finite fixed three-head full prediction grid required')
    return seal


def score() -> None:
    if any((OUT / name).exists() for name in (*OUTPUTS, 'source_completed.json')):
        raise FileExistsError('Preserve V37 score attempt')
    frozen, seal = check(), predictions()
    labels, support = v31.scoring_labels(common.monthly_rows(frozen))
    if len(labels) != 35 or not all(y['ef_defined'] for y in labels.values()) or support.to_csv(index=False) != Path(frozen['support_path']).read_text():
        raise ValueError('Exact same defined-EF scoring support required')
    baseline = pd.read_csv(frozen['baseline_case_path'], float_precision='round_trip').rename(columns={'recipe': 'arm'})
    if (len(baseline) != 700 or set(baseline.arm) != set(v36.model.RECIPES) or set(baseline['mode']) != {'mean', 'median'} or
            set(baseline.key) != set(labels) or baseline.duplicated(['region', 'key', 'arm', 'mode']).any()):
        raise ValueError('Complete unchanged V36 comparison grid required')
    records = []
    for entry in seal['entries']:
        with np.load(entry['path'], allow_pickle=False) as a:
            for j, key in enumerate(a['keys']):
                if key not in labels:
                    continue
                truth = labels[key]
                for i, arm in enumerate(model.ARMS):
                    summaries = v36.point_case(a['CIF_draws'][i, j], a['EF_draws'][i, j], truth['cif'], truth['ef'])
                    records.extend(dict(region=entry['region'], key=str(key), arm=arm, mode=mode, **r) for mode, r in summaries.items())
    case = pd.concat([baseline, pd.DataFrame(records)], ignore_index=True)
    if len(case) != 910 or case.duplicated(['region', 'key', 'arm', 'mode']).any() or not case.iloc[:700].equals(baseline):
        raise ValueError('Same support and numerically unchanged baseline rows required')
    region = v36.aggregate(case, ['region', 'arm', 'mode'], ('cif', 'ef'))
    macro = region.drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index()
    leave = pd.concat([region[region.region.ne(s)].drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index().assign(omitted_source=s) for s in SOURCES], ignore_index=True)
    if (len(region), len(macro), len(leave)) != (182, 26, 182):
        raise ValueError('Exact complete comparison aggregates required')
    for name, table in (('case_metrics', case), ('region_metrics', region), ('macro_metrics', macro), ('leave_one_source_out_macro', leave), ('label_support', support)):
        table.to_csv(OUT / f'{name}.csv', mode='x', index=False)
    summary = dict(new_fits=21, source_months=35, arms=list(ARMS), baseline_case_rows_copied=700,
        table_rows=dict(case_metrics=910, region_metrics=182, macro_metrics=26, leave_one_source_out_macro=182, label_support=42),
        method_innovation_claim=False, new_holdout_authorized=False, CIF_interval_coverage_evaluated=False, hourly_coverage_evaluated=False)
    write(OUT / 'summary.json', summary)
    check()
    predictions()
    write(OUT / 'source_completed.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        prediction_seal_sha256=sha(OUT / 'prediction_seal.json'), prediction_started_sha256=sha(OUT / 'prediction_started.json'),
        output_hashes={str(OUT / name): sha(OUT / name) for name in OUTPUTS}, **summary))
    print('Scored three new decoders and ten fixed reference rules on 35 months', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'predict', 'score'))
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score}[args.stage]()
