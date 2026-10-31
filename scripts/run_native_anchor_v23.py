"""DRAFT/PAUSED legacy daily-budget branch; all formal entry points fail closed.

The user's main scenario permits public monthly/yearly fuel statistics only.
This unfinished draft is retained for provenance, not approved or executable as
a formal experiment. There is no CLI switch to bypass the unconditional pause.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scripts import native_anchor_v23 as model
from scripts import run_native_refit_v21 as prior

sha, read, write, verify, now = prior.sha, prior.read, prior.write, prior.verify, prior.now
OUT = ROOT / 'results/native-anchor-v23'
PARENT = prior.OUT
MANIFEST = ROOT / 'docs/evidence/native-refit-v21-verification.json'
V17, V18 = prior.sketch.OUT, prior.center.OUT
CODE = ('scripts/native_anchor_v23.py', 'scripts/run_native_anchor_v23.py',
        'tests/test_native_anchor_v23.py', 'docs/NATIVE_ANCHOR_V23_PROTOCOL.md')
REVIEWS = ('code-review.md', 'python-review.md', 'design-review.md')
SOURCES, FACTORS, METRICS, DAYS = model.SOURCES, model.FACTORS, prior.METRICS, prior.DAYS
OUTPUTS = prior.SCORE_OUTPUTS
PAUSED_REASON = ('V23 DRAFT/PAUSED: legacy daily packet budget paused; the user main scenario '
                 'permits public monthly/yearly fuel statistics only. No formal stage is authorized.')


def _paused() -> None:
    raise RuntimeError(PAUSED_REASON)


def config() -> dict[str, Any]:
    return dict(version=23, sources=list(SOURCES), inputs=list(model.INPUTS), lag=7,
        projections=list(model.PROJECTIONS), alphas=list(model.ALPHAS), factors=list(FACTORS),
        trees=150, leaf=10, seed=17023, samples=128, archive_count=7, raw_arm_count=31, curve_count=155,
        response='fine_y_minus_own_packet_anchor', desired_center='clip(anchor+ETprediction,0,1000)',
        source_H2_fine_labels_read=False, prediction_reads_any_new_fine=False,
        target_values_opened=False, unseen_values_opened=False, target_test_authorized=False,
        custom_method=False, mechanism_breakthrough=False, independent_confirmation=False,
        new_risk_fit=False, full_probabilistic_control_completed=False)


def historical_paths() -> list[Path]:
    return [folder / 'predictions' / name for region in SOURCES for packet in ('T8', 'G5')
        for folder, name in ((V17, f'source-{region}-7-{packet}.npz'), (V18, f'source-{region}-{packet}.npz'))]


def parent_inputs(digest: str) -> tuple[dict, dict[str, str]]:
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Explicit64hex root-approved final V21 manifest SHA required')
    if not MANIFEST.is_file() or sha(MANIFEST) != digest:
        raise ValueError('Completed audited V21 final manifest required')
    evidence = read(MANIFEST)
    if evidence.get('numerical_audit_status') != 'PASS':
        raise ValueError('Final V21 numerical audit must be PASS')
    prior.manifest_hashes(MANIFEST)
    parent = prior.check();seal = read(PARENT / 'prediction_seal.json');done = read(PARENT / 'source_completed.json')
    if (done['version'] != 21 or done['freeze_sha256'] != sha(PARENT / 'freeze.json')
            or seal['freeze_sha256'] != done['freeze_sha256']
            or done['prediction_seal_sha256'] != sha(PARENT / 'prediction_seal.json')
            or seal['complete_grid'] is not True or len(seal['entries']) != 7
            or {e['region'] for e in seal['entries']} != set(SOURCES)
            or any(done[k] is not False for k in ('target_values_opened', 'unseen_values_opened',
                'custom_method', 'mechanism_breakthrough', 'full_probabilistic_control_completed'))):
        raise ValueError('Complete immutable source-only V21 parent required')
    verify(done['output_hashes'])
    paths = [MANIFEST, *(PARENT / n for n in ('freeze.json', 'prediction_seal.json', 'source_completed.json')),
        *(Path(e[key]) for e in seal['entries'] for key in ('path', 'audit_path')),
        PARENT / 'region_metrics.csv', PARENT / 'reference_curves.csv',
        PARENT / 'diagnostic-options.md', *historical_paths(), *(OUT / n for n in REVIEWS)]
    for name in REVIEWS:
        if not (OUT / name).is_file() or not (OUT / name).read_text().strip():
            raise ValueError('All three independent prelaunch reviews required')
    inputs = {**parent['input_hashes'], **{str(path): sha(path) for path in paths}}
    old_freeze = read(V18 / 'freeze.json')
    old_evidence = read(prior.parent_api.prior.MANIFEST)
    for path in paths:
        if path.parent == PARENT or path.parent == PARENT / 'predictions':
            if evidence['artifact_hashes'].get(str(path.relative_to(ROOT))) != inputs[str(path)]:
                raise ValueError('Parent metadata/archive/reference not in final manifest')
        if path.parent == V17 / 'predictions':
            if inputs[str(path)] != old_freeze['input_hashes'][str(path)]:
                raise ValueError('V17 desired-point reference differs from sealed V18 input')
        if path.parent == V18 / 'predictions':
            if inputs[str(path)] != old_evidence['artifact_hashes'][str(path.relative_to(ROOT))]:
                raise ValueError('V18 raw-path reproduction reference changed')
    if len(inputs) != 210:
        raise ValueError('Exact210 input closure required')
    return parent, inputs


def prepare(digest: str) -> None:
    _paused()
    if any((OUT / n).exists() for n in ('freeze.json', 'predictions', 'prediction_seal.json', 'source_completed.json')):
        raise FileExistsError('Preserve partial or completed V23 attempt')
    parent, inputs = parent_inputs(digest)
    code = {**parent['code_hashes'], **{str(ROOT / p): sha(ROOT / p) for p in CODE}}
    if len(code) != 73:
        raise ValueError('Exact73 code closure required')
    verify(code);verify(inputs)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'freeze.json', {**config(), 'created_utc': now(), 'source': parent['source'],
        'parent_manifest_sha256': digest, 'code_hashes': code, 'input_hashes': inputs,
        'environment': parent['environment'], 'environment_sha256': parent['environment_sha256']})


def check() -> dict:
    frozen = read(OUT / 'freeze.json')
    if any(frozen[k] != v for k, v in config().items()):
        raise ValueError('Changed anchor-offset contract')
    parent, inputs = parent_inputs(frozen['parent_manifest_sha256'])
    code = {**parent['code_hashes'], **{str(ROOT / p): sha(ROOT / p) for p in CODE}}
    if (len(code) != 73 or frozen['code_hashes'] != code or frozen['input_hashes'] != inputs
            or frozen['source'] != parent['source'] or frozen['environment'] != parent['environment']
            or frozen['environment_sha256'] != parent['environment_sha256']):
        raise ValueError('Changed code/input/environment closure')
    verify({**code, **inputs});return frozen


def parent_archive(region: str) -> tuple[dict, str, str]:
    seal = read(PARENT / 'prediction_seal.json')
    entry = next(e for e in seal['entries'] if e['region'] == region)
    expected = PARENT / 'predictions' / f'source-{region}.npz'
    if entry['path'] != str(expected) or entry['audit_path'] != str(expected.with_suffix('.json')):
        raise ValueError('Registered parent compact archive required')
    return prior.load_arrays(entry['path']), entry['path'], entry['audit_path']


def common_arrays(parent: dict) -> tuple[dict, dict, dict, dict, dict]:
    tx, ta = model.input_arrays(parent, 'train');qx, qa = model.input_arrays(parent, 'query')
    keys = ('raw', 'origins', 'donor_ids', 'train_regions', 'train_origins', 'train_teacher_paths',
            'train_truth', 'train_weights')
    arrays = {key: parent[key].copy() for key in keys}
    arrays.update(inputs=np.array(model.INPUTS), arm_inputs=np.array([a[0] for a in model.arms()]),
        arm_projections=np.array([a[1] for a in model.arms()]), arm_alphas=np.array([a[2] for a in model.arms()]))
    for name in model.INPUTS:
        arrays['train_x__' + name], arrays['query_x__' + name] = tx[name], qx[name]
        arrays['train_anchor__' + name], arrays['query_anchor__' + name] = ta[name], qa[name]
        arrays['train_response__' + name] = arrays['train_truth'] - ta[name][:, None]
    return arrays, tx, ta, qx, qa


def reproduction(region: str, arrays: dict) -> dict[str, int]:
    references = {}
    for packet in ('T8', 'G5'):
        with np.load(V17 / 'predictions' / f'source-{region}-7-{packet}.npz', allow_pickle=False) as old:
            point = prior.center.model.extract_et_center(old)
            if not np.array_equal(old['origins'], arrays['origins']):
                raise ValueError('Historical desired-point query grid changed')
        with np.load(V18 / 'predictions' / f'source-{region}-{packet}.npz', allow_pickle=False) as old:
            if (not np.array_equal(old['origins'], arrays['origins'])
                    or not np.array_equal(old['donor_ids'], arrays['donor_ids'])):
                raise ValueError('Historical raw-path native identity changed')
            references[packet] = dict(center=point, methods=old['methods'], alphas=old['alphas'], raw=old['raw'])
    return model.exact_reproduction(arrays['raw'], arrays['desired_centers'], arrays['deltas'], references)


def predict() -> None:
    _paused()
    check();initial = sha(OUT / 'freeze.json')
    if any((OUT / n).exists() for n in ('predictions', 'prediction_seal.json')):
        raise FileExistsError('Preserve prediction attempt')
    folder = OUT / 'predictions';folder.mkdir();entries = []
    for region in SOURCES:
        parent, parent_path, audit_source = parent_archive(region)
        arrays, tx, ta, qx, qa = common_arrays(parent);points = []
        dates = pd.DatetimeIndex(pd.to_datetime(arrays['train_origins'], utc=True))
        for name in model.INPUTS:
            fitted = model.AnchorModel().fit(tx[name], arrays['train_truth'], ta[name], arrays['train_regions'], dates, exclude=region)
            points.append(fitted.predict(qx[name], qa[name]))
            arrays['scaler_mean__' + name], arrays['scaler_scale__' + name] = fitted.scaler.mean_, fitted.scaler.scale_
        arrays['desired_centers'] = np.stack(points)
        arrays['deltas'] = model.deltas_from_centers(arrays['raw'], arrays['desired_centers'])
        arrays.update({'diagnostic__' + k: v for k, v in model.diagnostics(arrays['raw'], arrays['deltas']).items()})
        path = folder / f'source-{region}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        # Persist a failed fit for diagnosis but never seal/score if exact reproduction fails.
        reproduced = reproduction(region, arrays)
        audit_path = path.with_suffix('.json')
        write(audit_path, dict(region=region, parent_archive=parent_path, asof_audit_reference=audit_source,
            response='y_minus_own_packet_anchor', desired_center=config()['desired_center'],
            native_anchor_channel='D2', new_fine_read=False, new_risk_fit=False, exact_reproduction=reproduced,
            model=dict(trees=150, leaf=10, seed=17023, n_jobs=1)))
        entries.append(dict(region=region, path=str(path), sha256=sha(path), audit_path=str(audit_path), audit_sha256=sha(audit_path)))
        print('Predicted same-anchor controls', region, flush=True)
    check()
    if sha(OUT / 'freeze.json') != initial:
        raise ValueError('Freeze changed during anchor prediction')
    seal = dict(created_utc=now(), freeze_sha256=initial, entries=entries, complete_grid=True,
        archive_count=7, raw_arm_count=31, curve_count=155, prediction_reads_any_new_fine=False,
        target_values_opened=False, exact_reproduction_required=True)
    check_predictions(seal);check()
    if sha(OUT / 'freeze.json') != initial:
        raise ValueError('Freeze changed during reproduction verification')
    write(OUT / 'prediction_seal.json', seal)


def check_predictions(seal: dict) -> None:
    entries = seal['entries']
    if (len(entries) != 7 or {e['region'] for e in entries} != set(SOURCES) or seal['complete_grid'] is not True
            or seal['archive_count'] != 7 or seal['raw_arm_count'] != 31 or seal['curve_count'] != 155
            or seal['prediction_reads_any_new_fine'] is not False or seal['target_values_opened'] is not False
            or seal['exact_reproduction_required'] is not True):
        raise ValueError('Exact7 archives/31arms/155curves/reproduction before truth required')
    expected_paths = set()
    for entry in entries:
        path = (OUT / 'predictions' / f"source-{entry['region']}.npz").resolve()
        if Path(entry['path']).resolve() != path or Path(entry['audit_path']).resolve() != path.with_suffix('.json'):
            raise ValueError('Unregistered anchor archive/audit path')
        verify({str(path): entry['sha256'], str(path.with_suffix('.json')): entry['audit_sha256']})
        expected_paths.update((path, path.with_suffix('.json')))
    if {p.resolve() for p in (OUT / 'predictions').iterdir()} != expected_paths:
        raise ValueError('Unexpected prediction files')
    check()
    if seal['freeze_sha256'] != sha(OUT / 'freeze.json'):
        raise ValueError('Prediction freeze binding changed')
    for entry in entries:
        region = entry['region'];actual = prior.load_arrays(entry['path']);parent, _, _ = parent_archive(region)
        expected, tx, _, _, _ = common_arrays(parent)
        for name in model.INPUTS:
            scaler = StandardScaler().fit(tx[name], sample_weight=expected['train_weights'])
            expected['scaler_mean__' + name], expected['scaler_scale__' + name] = scaler.mean_, scaler.scale_
        points = actual['desired_centers']
        expected['desired_centers'] = points
        expected['deltas'] = model.deltas_from_centers(expected['raw'], points)
        expected.update({'diagnostic__' + k: v for k, v in model.diagnostics(expected['raw'], expected['deltas']).items()})
        prior.parent_api.exact_fields(actual, expected, entry['path'])
        if reproduction(region, actual) != dict(packet_centers=2, projection_alpha_paths=12):
            raise ValueError('Incomplete exact reproduction checks')


def reference_curves() -> pd.DataFrame:
    old = pd.read_csv(PARENT / 'region_metrics.csv')
    old['packet'] = old['input'].str.rsplit('_', n=1).str[-1]
    old['method'] = old['input'] + ':' + old['projection']
    old['reference_family'], old['response_offset'], old['native_input_available'] = 'V21_nativeoffset', 'native_mean', True
    inherited = pd.read_csv(PARENT / 'reference_curves.csv')
    columns = ['region', 'packet', 'reference_family', 'method', 'alpha', 'factor',
               'response_offset', 'native_input_available', *METRICS]
    if len(old) != 1925 or len(inherited) != 2415:
        raise ValueError('Full V21/V18/V17/V16 reference grids required')
    result = pd.concat((old[columns], inherited[columns]), ignore_index=True)
    if not np.isfinite(result[list(METRICS)]).all().all():
        raise ValueError('Finite completed reference metrics required')
    return result


def choose(part: pd.DataFrame, ties: list[str]) -> dict:
    good = part[part.violation <= 1e-12]
    row = (good if len(good) else part).sort_values(([] if len(good) else ['violation']) + ['crps', *ties]).iloc[0]
    return {**{k: None if pd.isna(v) else v for k, v in row.to_dict().items()}, 'passed': bool(row.violation <= 1e-12)}


def selection(regional: pd.DataFrame, refs: pd.DataFrame) -> dict:
    macro = prior.macro_scores(regional)
    selected = {name: choose(macro[macro.input.eq(name)], ['input', 'projection', 'alpha', 'factor'])
                for name in ('native', *model.INPUTS)}
    keys = ['reference_family', 'packet', 'method', 'alpha', 'factor', 'response_offset', 'native_input_available']
    background = refs.groupby(keys, dropna=False)[list(METRICS)].mean()
    background['worst_coverage'] = refs.groupby(keys, dropna=False).empirical_coverage90.min()
    background['violation'] = (np.maximum(.9 - background.empirical_coverage90, 0)
        + np.maximum(background.empirical_coverage90 - .95, 0) + np.maximum(.85 - background.worst_coverage, 0))
    background = background.reset_index()
    picked_refs = {p: choose(background[background.packet.eq(p)], keys) for p in ('D2', 'D8', 'G5')}
    picked_refs['V16_additive'] = choose(background[background.reference_family.eq('V16_additive')], keys)
    main = selected['joint_T8']
    own_names = ('packet_only_T8', 'native_plus_D2_anchor', 'joint_G5', 'native')
    comparisons = [selected[n] for n in own_names] + list(picked_refs.values())
    gate = bool(main['passed'] and all(main['crps'] <= .95 * row['crps']
        and main['empirical_width90'] <= row['empirical_width90']
        and main['empirical_interval_score90'] < row['empirical_interval_score90'] for row in comparisons))
    return dict(created_utc=now(), selections=selected, background_selections=picked_refs,
        source_budget_gate=gate, source_budget_reference_inputs=list(own_names) + ['background_D2', 'background_D8', 'background_G5', 'V16_additive'],
        custom_method=False, mechanism_breakthrough=False, independent_confirmation=False,
        target_test_authorized=False, new_uncertainty_mechanism=False, full_probabilistic_control_completed=False,
        H_timing_causal_claim=False)


def matched_contrasts(regional: pd.DataFrame) -> pd.DataFrame:
    rows = [];keys = ['region', 'projection', 'alpha', 'factor']
    for packet in ('T8', 'G5'):
        left = regional[regional.input.eq('joint_' + packet)].drop(columns='input')
        right = regional[regional.input.eq('packet_only_' + packet)].drop(columns='input')
        merged = left.merge(right, on=keys, suffixes=('_joint', '_packet'), validate='one_to_one')
        if len(merged) != 210:
            raise ValueError('Complete same-anchor joint-minus-packet contrasts required')
        merged['packet'] = packet
        for metric in METRICS:
            merged[metric + '_joint_minus_packet'] = merged[metric + '_joint'] - merged[metric + '_packet']
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def score() -> None:
    _paused()
    if any((OUT / n).exists() for n in ('source_completed.json', *OUTPUTS)):
        raise FileExistsError('Preserve partial or completed score outputs')
    frozen = check();seal = read(OUT / 'prediction_seal.json');check_predictions(seal)
    start = {str(OUT / n): sha(OUT / n) for n in ('freeze.json', 'prediction_seal.json')};rows = []
    for entry in seal['entries']:
        region = entry['region'];arrays = prior.load_arrays(entry['path'])
        fine = prior.sketch.source_days(Path(frozen['source']) / f'{region}_hourly.csv', '2025-01-01', '2025-07-01')
        truth = {d.value: (a / a.sum(1)[:, None]) @ prior.sketch.model.EF for d, a in fine}
        keep = np.isin(arrays['origins'], list(truth));origins = arrays['origins'][keep]
        y = np.stack([truth[int(t)] for t in origins])
        boundary = ((arrays['raw'][keep] <= 0) | (arrays['raw'][keep] >= 1000)).mean((1, 2))
        for i, arm in enumerate(model.arms()):
            raw, _ = model.reconstruct(arrays['raw'], arrays['deltas'], arm)
            for factor in FACTORS:
                point, paths, final_clip = model.scaled_paths(raw[keep], factor)
                metrics = prior.sketch.score_arrays(point, paths, y)
                rows.extend(dict(region=region, origin=pd.Timestamp(int(t), tz='UTC').isoformat(),
                    input=arm[0], projection=arm[1], alpha=arm[2], factor=factor,
                    unclipped_width90=float(arrays['diagnostic__unclipped_width90'][i, keep][j]),
                    initial_clipped_width90=float(arrays['diagnostic__initial_clipped_width90'][i, keep][j]),
                    new_clipping_rate=float(arrays['diagnostic__new_clipping_rate'][i, keep][j]),
                    native_boundary_rate=float(boundary[j]), final_spread_clipping_rate=float(final_clip[j]),
                    **{k: float(v[j]) for k, v in metrics.items() if k in METRICS}) for j, t in enumerate(origins))
        print('Scored same-anchor controls', region, flush=True)
    day = pd.DataFrame(rows);keys = ['region', 'input', 'projection', 'alpha', 'factor']
    if (len(day) != 177475 or len(day[['region', 'origin']].drop_duplicates()) != 1145
            or day.duplicated(keys + ['origin']).any() or not np.isfinite(day.select_dtypes(include='number')).all().all()):
        raise ValueError('Exact177475row/1145day cohort required')
    if any(len(part) != 1145 for _, part in day.groupby(keys[1:])):
        raise ValueError('Every155curve must retain all1145source days')
    regional = day.groupby(keys)[list(METRICS)].mean().reset_index()
    if len(regional) != 1085:
        raise ValueError('Complete1085region grid required')
    refs = reference_curves()
    old_native = refs[refs.reference_family.eq('V21_nativeoffset') & refs.method.eq('native:none')]
    joined = regional[regional.input.eq('native')].merge(old_native, on=['region', 'factor'], suffixes=('_new', '_old'), validate='one_to_one')
    if len(joined) != 35 or any(not np.allclose(joined[k + '_new'], joined[k + '_old'], rtol=0, atol=1e-9) for k in METRICS):
        raise ValueError('Frozen native metric identity failed')
    tables = dict(day_metrics=day, region_metrics=regional, macro_metrics=prior.macro_scores(regional),
                  matched_input_contrasts=matched_contrasts(regional), reference_curves=refs)
    for name, table in tables.items():
        table.to_csv(OUT / f'{name}.csv', index=False, mode='x')
    write(OUT / 'source_choice.json', selection(regional, refs))
    check();check_predictions(seal);verify(start)
    write(OUT / 'source_completed.json', dict(created_utc=now(), version=23,
        freeze_sha256=start[str(OUT / 'freeze.json')], prediction_seal_sha256=start[str(OUT / 'prediction_seal.json')],
        output_hashes={str(OUT / n): sha(OUT / n) for n in OUTPUTS}, target_values_opened=False,
        unseen_values_opened=False, custom_method=False, mechanism_breakthrough=False,
        full_probabilistic_control_completed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'predict', 'score'), required=True)
    parser.add_argument('--parent-sha256', help='Root-approved final V21 PASS manifest; prepare only')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.parent_sha256)
    elif args.parent_sha256 is not None:
        parser.error('--parent-sha256 only applies to prepare')
    else:
        {'predict': predict, 'score': score}[args.stage]()
