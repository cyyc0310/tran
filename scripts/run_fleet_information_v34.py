"""Source-only information-value experiment on sealed monthly forecasts."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import fleet_information_v34 as model
from scripts import run_coarse_context_v30 as common
from scripts import run_forward_context_v31 as v31

OUT = ROOT / 'results/fleet-information-v34'
PARENTS = {
    ROOT / 'docs/evidence/forward-context-v31-verification.json':
        'f6544892fc7bf24256cadd5809b3b9fcdfde60c4e42bde9bfa13db41ea9b46df',
    ROOT / 'docs/evidence/fleet-inventory-v33-verification.json':
        'f3649828c812a19bcc81371c3f1960239fc6c8572962afd61c6bf7be127aebe8',
}
CODE = ('scripts/fleet_information_v34.py', 'scripts/run_fleet_information_v34.py',
        'tests/test_fleet_information_v34.py', 'tests/test_run_fleet_information_v34.py',
        'docs/FLEET_INFORMATION_V34_PROTOCOL.md')
REVIEWS = tuple(ROOT / f'results/fleet-information-v34-{name}-review.md'
                for name in ('code', 'python', 'science'))
DATES = v31.DATES
METRICS = ('cif_abs_error', 'cif_sq_error', 'cif_signed_error',
           'ef_abs_error', 'ef_sq_error', 'ef_signed_error')
OUTPUTS = ('case_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv',
           'matched_contrasts.csv', 'label_support.csv', 'source_choice.json')
QUERY_MEMBERS = ('origins', 'arms', 'query_feature_Size', 'MG_draws',
                 'CIF_point', 'EF_point', 'CIF_draws', 'EF_draws', 'projection_l1')
sha, read, write, now = common.sha, common.read, common.write, common.now


def verify(mapping: dict) -> None:
    for path in mapping:
        if not Path(path).is_file() or Path(path).is_symlink():
            raise ValueError('Regular non-symlink bound file required')
    common.verify(mapping)


def expected_contract() -> dict:
    inputs = {}
    for path, digest in PARENTS.items():
        if not digest or not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError('Fixed independently audited V31 and V33 parents required')
        manifest = read(path)
        if (manifest['status'] != 'VERIFICATION_COMPLETE' or
                manifest['independent_actual_audit_status'] != 'PASS'):
            raise ValueError('Parent independent actual PASS required')
        mappings = {str(path): digest}
        for field in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
            for alias, value in manifest[field].items():
                if alias in mappings and mappings[alias] != value:
                    raise ValueError('Conflicting within-parent mapping')
                mappings[alias] = value
        for alias, value in mappings.items():
            if alias in inputs and inputs[alias] != value:
                raise ValueError('Conflicting parent mapping')
            inputs[alias] = value
    verify(inputs)
    old = read(ROOT / 'results/forward-context-v31/freeze.json')
    seal = read(ROOT / 'results/forward-context-v31/prediction_seal.json')
    query_paths = {entry['region']: entry['path'] for entry in seal['entries']}
    capacities = {policy: str(ROOT / f'results/fleet-inventory-v33/capacity_{policy}.csv')
                  for policy in ('op', 'operable')}
    support_path = str(ROOT / 'results/forward-context-v31/label_support.csv')
    aliases = [*old['head_source_paths'].values(), *query_paths.values(),
               *capacities.values(), old['monthly_path'], support_path]
    if (set(query_paths) != set(model.SOURCES) or
            set(old['head_source_paths']) != set(model.SOURCES) or
            any(alias not in inputs for alias in aliases)):
        raise ValueError('Exact bound aliases required')
    for path in REVIEWS:
        if not path.is_file() or path.is_symlink() or not path.read_text().strip():
            raise ValueError('Three independent final prelaunch reviews required')
        inputs[str(path)] = sha(path)
    code_hashes = {str(ROOT / path): sha(ROOT / path) for path in CODE}
    verify(code_hashes)
    return dict(version=34, sources=list(model.SOURCES), arms=list(model.ARMS),
                origins=DATES.asi8.tolist(), samples=128, new_fits=49,
                head_source_paths=old['head_source_paths'], query_paths=query_paths,
                capacity_paths=capacities, monthly_path=old['monthly_path'],
                support_path=support_path, code_hashes=code_hashes,
                input_hashes=inputs, environment=common.environment(),
                information_budget='B1: B0 plus lagged registered fleet inventory proxy',
                baseline_budget='B0', source_only=True, new_training_labels_used=False,
                hourly_values_read=False, target_values_opened=False, unseen_values_opened=False,
                custom_method=False, mechanism_breakthrough=False,
                probability_coverage_evaluated=False, historical_first_release_bytes_certified=False)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError('Preserve V34 attempt')
    contract = expected_contract()
    OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **contract))
    print('Prepared', len(contract['code_hashes']), 'code', len(contract['input_hashes']),
          'input hashes', flush=True)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), expected_contract()
    if (set(frozen) != set(expected) | {'created_utc'} or
            any(frozen[key] != value for key, value in expected.items()) or
            any(frozen[key] is not value for key, value in expected.items() if isinstance(value, bool))):
        raise ValueError('Frozen V34 contract changed')
    return frozen


def load_query(path: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in QUERY_MEMBERS}


def capacity_rows(path: str) -> list[dict]:
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def predict() -> None:
    frozen = check()
    initial = sha(OUT / 'freeze.json')
    folder = OUT / 'predictions'
    if folder.exists() or (OUT / 'prediction_seal.json').exists():
        raise FileExistsError('Preserve V34 prediction attempt')
    write(OUT / 'prediction_started.json', dict(created_utc=now(), freeze_sha256=initial))
    folder.mkdir()
    capacities = {key: capacity_rows(path) for key, path in frozen['capacity_paths'].items()}
    entries = []
    for region in model.SOURCES:
        query = load_query(frozen['query_paths'][region])
        result = model.fit_predict(read(frozen['head_source_paths'][region]), query,
                                   capacities['op'], capacities['operable'], region=region)
        arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
        arrays.update(arms=np.asarray(model.ARMS), origins=DATES.asi8,
                      keys=np.asarray([f'{region}/{at:%Y-%m}' for at in DATES]))
        # The two B0 arms are a literal copy, not newly selected or refitted heads.
        for index, name in enumerate(model.ARMS[:2]):
            old = query['arms'].tolist().index(name)
            if any(not np.array_equal(arrays[key][index], query[key][old])
                   for key in ('CIF_point', 'EF_point', 'CIF_draws', 'EF_draws')):
                raise ValueError('Sealed B0 baseline changed')
        path = folder / f'{region}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        audit = path.with_suffix('.json')
        write(audit, result['audit'])
        entries.append(dict(region=region, path=str(path), sha256=sha(path),
                            audit_path=str(audit), audit_sha256=sha(audit)))
        print('Predicted', region, 'origins', len(DATES), 'new fits', result['audit']['fit_count'], flush=True)
    check()
    verify({str(OUT / 'freeze.json'): initial})
    write(OUT / 'prediction_seal.json', dict(created_utc=now(), freeze_sha256=initial,
          prediction_started_sha256=sha(OUT / 'prediction_started.json'),
          entries=entries, complete_grid=True, archives=7, arms=11, origins_per_source=6,
          new_fits=49, copied_B0_arms=2, source_only=True, labels_read_for_scoring=False,
          hourly_values_read=False, target_values_opened=False, unseen_values_opened=False))


def predictions() -> dict:
    seal = read(OUT / 'prediction_seal.json')
    verify({str(OUT / 'prediction_started.json'): seal['prediction_started_sha256']})
    marker = read(OUT / 'prediction_started.json')
    if marker['freeze_sha256'] != sha(OUT / 'freeze.json'):
        raise ValueError('Prediction start receipt does not bind this freeze')
    if (seal['freeze_sha256'] != sha(OUT / 'freeze.json') or seal['complete_grid'] is not True or
            seal['source_only'] is not True or
            seal['archives'] != 7 or seal['arms'] != 11 or seal['origins_per_source'] != 6 or
            seal['new_fits'] != 49 or seal['copied_B0_arms'] != 2 or
            any(seal[key] is not False for key in ('labels_read_for_scoring', 'hourly_values_read',
                                                  'target_values_opened', 'unseen_values_opened')) or
            [entry['region'] for entry in seal['entries']] != list(model.SOURCES)):
        raise ValueError('Complete forecast-only seal required before scoring labels')
    folder = OUT / 'predictions'
    if (not folder.is_dir() or {path.name for path in folder.iterdir()} !=
            {f'{region}.{suffix}' for region in model.SOURCES for suffix in ('npz', 'json')} or
            any(not path.is_file() or path.is_symlink() for path in folder.iterdir())):
        raise ValueError('Exact fourteen regular prediction files required')
    for entry in seal['entries']:
        path = folder / f"{entry['region']}.npz"
        if entry['path'] != str(path) or entry['audit_path'] != str(path.with_suffix('.json')):
            raise ValueError('Prediction aliases changed')
        verify({entry['path']: entry['sha256'], entry['audit_path']: entry['audit_sha256']})
        with np.load(path, allow_pickle=False) as archive:
            if (archive['arms'].tolist() != list(model.ARMS) or
                    not np.array_equal(archive['origins'], DATES.asi8) or
                    archive['keys'].tolist() != [f"{entry['region']}/{at:%Y-%m}" for at in DATES] or
                    any(archive[key].shape != (11, 6, 128) or not np.isfinite(archive[key]).all()
                        for key in ('CIF_draws', 'EF_draws')) or
                    any(archive[key].shape != (11, 6) or not np.isfinite(archive[key]).all()
                        for key in ('CIF_point', 'EF_point'))):
                raise ValueError('Fixed finite eleven-arm grid required')
    return seal


def matched_pairs() -> list[tuple[str, str]]:
    pairs = [(candidate, reference) for candidate in model.ARMS[2:] for reference in model.ARMS[:2]]
    pairs += [(f'{family}-{name}', f'SizeCap-{name}')
              for family in ('IntensiveCap', 'DirectEF') for name in ('Ridge', 'ET')]
    pairs += [('Utilization-OP', 'Capacity-OP'), ('Capacity-Operable', 'Capacity-OP')]
    if len(pairs) != 24 or len(set(pairs)) != 24:
        raise ValueError('Fixed twenty-four comparisons required')
    return pairs


def summarize(region: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    macro = region.groupby('arm')[[*METRICS, 'cif_rmse', 'ef_rmse']].mean().reset_index()
    signals = []
    for candidate, reference in matched_pairs()[:18]:
        left = region[region.arm.eq(candidate)].set_index('region').cif_abs_error.reindex(model.SOURCES)
        right = region[region.arm.eq(reference)].set_index('region').cif_abs_error.reindex(model.SOURCES)
        good = (right.mean() > 1e-12 and left.mean() <= .95 * right.mean() and
                int((left < right - 1e-12).sum()) >= 4 and
                bool((left <= 1.1 * right + 1e-12).all()) and
                all(left[key] <= right[key] + 1e-12 for key in ('MISO', 'ERCO')))
        signals.append(dict(candidate=candidate, reference=reference, signal=bool(good),
                            improved_regions=int((left < right - 1e-12).sum()),
                            macro_cif_mae_delta=float(left.mean() - right.mean())))
    qualified = [candidate for candidate in model.ARMS[2:]
                 if all(item['signal'] for item in signals if item['candidate'] == candidate)]
    return macro, dict(comparisons=signals, B1_signal_arms=qualified,
                       practical_information_signal=bool(qualified), custom_method=False,
                       mechanism_breakthrough=False, new_holdout_authorized=False,
                       probability_coverage_evaluated=False, hourly_stage_authorized=False)


def score() -> None:
    if any((OUT / name).exists() for name in (*OUTPUTS, 'source_completed.json')):
        raise FileExistsError('Preserve V34 score attempt')
    frozen = check()
    seal = predictions()  # No label parsing until the full seven-source seal passes.
    labels, support = v31.scoring_labels(common.monthly_rows(frozen))
    if support.to_csv(index=False) != Path(frozen['support_path']).read_text():
        raise ValueError('Exact V31 scoring support required')
    records = []
    for entry in seal['entries']:
        with np.load(entry['path'], allow_pickle=False) as archive:
            for month, key in enumerate(archive['keys']):
                if key not in labels:
                    continue
                truth = labels[key]
                for index, arm in enumerate(model.ARMS):
                    error = float(archive['CIF_point'][index, month] - truth['cif'])
                    ef_error = float(archive['EF_point'][index, month] - truth['ef'])
                    records.append(dict(region=entry['region'], key=str(key), arm=arm,
                        cif_abs_error=abs(error), cif_sq_error=error ** 2, cif_signed_error=error,
                        ef_abs_error=abs(ef_error), ef_sq_error=ef_error ** 2, ef_signed_error=ef_error))
    case = pd.DataFrame(records)
    if (len(labels) != 35 or len(case) != 385 or case.duplicated(['region', 'key', 'arm']).any() or
            set(case.region) != set(model.SOURCES) or not np.isfinite(case[list(METRICS)]).all().all()):
        raise ValueError('Complete common 35-month, eleven-arm score support required')
    region = case.groupby(['region', 'arm'])[list(METRICS)].mean().reset_index()
    region['cif_rmse'] = np.sqrt(region.cif_sq_error)
    region['ef_rmse'] = np.sqrt(region.ef_sq_error)
    counts = case.groupby(['region', 'arm']).size().rename('scored_months').reset_index()
    region = region.merge(counts, on=['region', 'arm'], validate='one_to_one')
    macro, choice = summarize(region)
    contrasts = []
    for candidate, reference in matched_pairs():
        paired = region[region.arm.eq(candidate)].merge(region[region.arm.eq(reference)],
            on='region', suffixes=('_c', '_r'), validate='one_to_one')
        for row in paired.to_dict('records'):
            contrasts.append(dict(region=row['region'], candidate=candidate, reference=reference,
                **{metric + '_delta': row[metric + '_c'] - row[metric + '_r'] for metric in METRICS}))
    if len(region) != 77 or len(macro) != 11 or len(contrasts) != 168:
        raise ValueError('Exact 77-region/11-macro/168-contrast grid required')
    for name, frame in (('case_metrics', case), ('region_metrics', region), ('macro_metrics', macro),
                        ('matched_contrasts', pd.DataFrame(contrasts)), ('label_support', support)):
        frame.to_csv(OUT / f'{name}.csv', mode='x', index=False)
    write(OUT / 'source_choice.json', choice)
    check()
    predictions()
    write(OUT / 'source_completed.json', dict(version=34, created_utc=now(),
        freeze_sha256=sha(OUT / 'freeze.json'), prediction_seal_sha256=sha(OUT / 'prediction_seal.json'),
        prediction_started_sha256=sha(OUT / 'prediction_started.json'),
        output_hashes={str(OUT / name): sha(OUT / name) for name in OUTPUTS},
        source_only=True, new_fits=49, scored_source_months=35, custom_method=False,
        mechanism_breakthrough=False, hourly_values_read=False, target_values_opened=False,
        unseen_values_opened=False, probability_coverage_evaluated=False))
    print('Scored 35 months, 11 arms;', 'B1 signal arms:', choice['B1_signal_arms'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'predict', 'score'))
    arguments = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score}[arguments.stage]()
