"""Source-only same-offset native/packet residual refit; full seal before scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scripts import native_refit_v21 as model
from scripts import verify_native_hindcast_v20r1 as parent_api
from scripts import run_native_center_v18 as center

sketch = center.prior
sha, read, write, verify, now = sketch.sha, sketch.read, sketch.write, sketch.verify, sketch.now
OUT = ROOT / 'results/native-refit-v21'
PARENT = parent_api.OUT
MANIFEST = ROOT / 'docs/evidence/native-hindcast-v20r1-verification.json'
CODE = ('scripts/native_refit_v21.py', 'scripts/run_native_refit_v21.py',
        'tests/test_native_refit_v21.py', 'docs/NATIVE_REFIT_V21_PROTOCOL.md')
REVIEWS = ('code-review.md', 'python-review.md', 'design-review.md')
SOURCES, PACKETS, FACTORS, METRICS = model.SOURCES, model.PACKETS, model.FACTORS, sketch.METRICS
DAYS = pd.date_range('2025-01-01', '2025-06-30', tz='UTC')
SCORE_OUTPUTS = ('day_metrics.csv', 'region_metrics.csv', 'macro_metrics.csv',
                 'matched_input_contrasts.csv', 'reference_curves.csv', 'source_choice.json')


def config() -> dict[str, Any]:
    return dict(version=21, sources=list(SOURCES), inputs=list(model.INPUTS), packets=list(PACKETS),
        lag=7, projections=list(model.PROJECTIONS), alphas=list(model.ALPHAS), factors=list(FACTORS),
        seed=model.SEED, trees=model.TREES, leaf=model.LEAF, samples=128, archive_count=7,
        raw_arm_count=55, curve_count=275, proper_start='2024-01-25', proper_stop='2024-07-01',
        response='fine_y_minus_pair_excluded_H1_native_mean', H1_teacher_chronological=False,
        target_values_opened=False, unseen_values_opened=False, target_test_authorized=False,
        prediction_reads_2025_fine_labels=False, source_H2_fine_labels_read=False,
        custom_method=False, mechanism_breakthrough=False, independent_confirmation=False,
        new_risk_fit=False, full_probabilistic_control_completed=False)


def manifest_hashes(path: Path) -> dict[str, str]:
    evidence = read(path);hashes = {}
    for group in ('frozen_code_hashes', 'frozen_input_hashes', 'artifact_hashes'):
        for name, digest in evidence[group].items():
            hashes[str(Path(name) if Path(name).is_absolute() else ROOT / name)] = digest
    verify(hashes)
    return hashes


def parent_inputs(parent_sha256: str) -> tuple[dict, dict[str, str]]:
    """Hash/metadata gates only. PENDING R1 receipts alone cannot admit a child."""
    if not isinstance(parent_sha256, str) or not re.fullmatch('[0-9a-f]{64}', parent_sha256):
        raise ValueError('Explicit64hex approved R1 final-manifest digest required')
    if not MANIFEST.is_file() or sha(MANIFEST) != parent_sha256:
        raise ValueError('Completed audited R1 final manifest required')
    evidence = read(MANIFEST)
    if evidence.get('numerical_audit_status') != 'PASS':
        raise ValueError('Final R1 manifest numerical_audit_status must be PASS')
    manifest_hashes(MANIFEST)
    parent = parent_api.check()  # Keeps failed originals and PENDING receipts intact.
    seal, done = read(PARENT / 'verification_seal.json'), read(PARENT / 'completed.json')
    if (any(obj[k] != value for obj in (seal, done) for k, value in parent_api.contract().items())
            or any(obj['freeze_sha256'] != sha(PARENT / 'freeze.json') for obj in (seal, done))
            or done['verification_seal_sha256'] != sha(PARENT / 'verification_seal.json')
            or seal['entries'] != parent['entries'] or seal['banks'] != parent['banks']
            or seal['exact_counts'] != dict(query_copy=7, H1_retrospective=42, H2_hindcast=42)
            or any(obj['complete_grid'] is not True for obj in (seal, done))):
        raise ValueError('Complete immutable R1 aliases and verification receipts required')
    for name in ('freeze.json', 'verification_seal.json', 'completed.json'):
        path = PARENT / name
        if evidence['artifact_hashes'].get(str(path.relative_to(ROOT))) != sha(path):
            raise ValueError('R1 receipt not bound by final PASS manifest')
    # Complete V18 evidence binds V17 packet/data seals and all inherited references.
    verify({str(parent_api.prior.MANIFEST): parent_api.prior.MANIFEST_SHA})
    manifest_hashes(parent_api.prior.MANIFEST)
    old = read(center.OUT / 'freeze.json')
    old_done = read(center.OUT / 'source_completed.json')
    paths = [MANIFEST, *(PARENT / n for n in ('freeze.json', 'verification_seal.json', 'completed.json')),
        *(sketch.OUT / n for n in ('freeze.json', 'data_seal.json', 'prediction_seal.json', 'source_completed.json')),
        *(sketch.OUT / 'packets' / f'{p}.csv' for p in PACKETS),
        center.OUT / 'region_metrics.csv', center.OUT / 'reference_curves.csv',
        *(OUT / name for name in REVIEWS)]
    for name in REVIEWS:
        if not (OUT / name).is_file() or not (OUT / name).read_text().strip():
            raise ValueError('All three independent prelaunch reviews required')
    inputs = {**parent['input_hashes'], **{str(path): sha(path) for path in paths}}
    for path in paths:
        if path.parent == sketch.OUT or path.parent == sketch.OUT / 'packets':
            if inputs[str(path)] != old['input_hashes'][str(path)]:
                raise ValueError('V17 packet/reference metadata differs from sealed V18 input')
        if path.parent == center.OUT:
            if inputs[str(path)] != old_done['output_hashes'][str(path)]:
                raise ValueError('Reference curve not bound by completed V18')
    if len(inputs) != 158:
        raise ValueError('Exact158 input closure required')
    return parent, inputs


def prepare(parent_sha256: str) -> None:
    if any((OUT / n).exists() for n in ('freeze.json', 'predictions', 'prediction_seal.json', 'source_completed.json')):
        raise FileExistsError('Preserve complete or partial V21 attempt')
    parent, inputs = parent_inputs(parent_sha256)
    code = {**parent['code_hashes'], **{str(ROOT / name): sha(ROOT / name) for name in CODE}}
    if len(code) != 69:
        raise ValueError('Exact69 code closure required')
    verify(code);verify(inputs)
    environment = sketch.environment()
    frozen = {**config(), 'created_utc': now(), 'source': parent['source'],
        'parent_manifest_sha256': parent_sha256, 'code_hashes': code, 'input_hashes': inputs,
        'environment': environment,
        'environment_sha256': hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()}
    OUT.mkdir(parents=True, exist_ok=True);write(OUT / 'freeze.json', frozen)


def check() -> dict:
    frozen = read(OUT / 'freeze.json')
    if any(frozen[k] != value for k, value in config().items()):
        raise ValueError('Changed V21 residual-refit contract')
    parent, inputs = parent_inputs(frozen['parent_manifest_sha256'])
    code = {**parent['code_hashes'], **{str(ROOT / name): sha(ROOT / name) for name in CODE}}
    env = sketch.environment()
    if (frozen['source'] != parent['source'] or frozen['input_hashes'] != inputs
            or frozen['code_hashes'] != code or len(code) != 69 or frozen['environment'] != env
            or frozen['environment_sha256'] != hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()):
        raise ValueError('Changed input/code/environment closure')
    verify({**code, **inputs})
    return frozen


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class DataContext:
    def __init__(self, frozen: dict) -> None:
        self.parent = read(PARENT / 'freeze.json')
        self.entries = {(e['outer'], e['teacher'], e['split']): e for e in self.parent['entries']}
        self.fine = {r: sketch.source_days(Path(frozen['source']) / f'{r}_hourly.csv',
                         '2024-01-25', '2024-07-01') for r in SOURCES}
        if sum(map(len, self.fine.values())) != 1100 or any(not rows for rows in self.fine.values()):
            raise ValueError('Exact shared1100valid H1 proper days required')
        self.histories = {}
        for packet, n in sketch.PACKETS.items():
            table = pd.read_csv(sketch.OUT / 'packets' / f'{packet}.csv', dtype={f'v{i}': object for i in range(n)})
            for region in SOURCES:
                self.histories[packet, region, 7] = sketch.model.PacketHistory(table[table.region.eq(region)], packet, 7)

    def packet_features(self, labels: np.ndarray, days: pd.DatetimeIndex) -> tuple[dict, dict, np.ndarray]:
        features, audits, anchors = {}, {}, []
        for packet in PACKETS:
            rows = [sketch.checked_features(self.histories, packet, r, 7, day) for r, day in zip(labels, days)]
            features[packet] = np.stack([row[0] for row in rows]);anchors.append([row[1] for row in rows])
            audits[packet] = [dict(region=str(r), origin=day.isoformat(), anchor_difference=row[3], **row[2])
                              for r, day, row in zip(labels, days, rows)]
        return features, audits, np.array(anchors)

    def dataset(self, held: str) -> tuple[dict, dict, dict]:
        raw_parts, truth, labels, dates, paths = [], [], [], [], []
        for region in SOURCES:
            if region == held:
                continue
            entry = self.entries[held, region, 'H1_retrospective']
            arrays = load_arrays(entry['path'])
            origins = arrays['origins'];days = pd.DatetimeIndex([d for d, _ in self.fine[region]])
            indices = np.searchsorted(origins, days.asi8)
            if (not np.array_equal(origins, parent_api.model.H1_DAYS.asi8)
                    or not np.array_equal(origins[indices], days.asi8)):
                raise ValueError('H1 teacher must align the common valid response cohort')
            raw_parts.append(arrays['raw'][indices])
            truth.extend([(a / a.sum(1)[:, None]) @ sketch.model.EF for _, a in self.fine[region]])
            labels.extend([region] * len(days));dates.extend(days);paths.extend([entry['path']] * len(days))
        raw = np.concatenate(raw_parts);labels = np.asarray(labels, dtype='U');dates = pd.DatetimeIndex(dates)
        packets, audit, anchors = self.packet_features(labels, dates)
        features = model.input_features(raw, dates, packets)
        _, mean, sd = model.native_features(raw, dates);y = np.stack(truth)
        provenance = dict(train_regions=labels, train_origins=dates.as_unit('ns').asi8,
            train_teacher_paths=np.asarray(paths, dtype='U'), train_native_mean=mean, train_native_sd=sd,
            train_truth=y, train_response=y - mean, train_anchors=anchors,
            train_weights=model.balanced(labels) * len(labels))
        return features, provenance, audit

    def query(self, held: str) -> tuple[dict, dict, dict]:
        entry = self.entries[held, None, 'query_copy'];z = load_arrays(entry['path'])
        if not np.array_equal(z['origins'], DAYS.asi8):
            raise ValueError('Exact181day inherited query grid required')
        packets, audit, anchors = self.packet_features(np.repeat(held, len(DAYS)), DAYS)
        features = model.input_features(z['raw'], DAYS, packets)
        if not np.array_equal(z['center'], z['raw'].mean(1)):
            raise ValueError('Inherited native center must be particle mean')
        return features, dict(origins=z['origins'], raw=z['raw'], donor_ids=z['donor_ids'], query_anchors=anchors), audit


def arm_arrays() -> dict[str, np.ndarray]:
    grid = model.arms()
    return dict(inputs=np.array(model.INPUTS), arm_inputs=np.array([a[0] for a in grid]),
        arm_projections=np.array([a[1] for a in grid]), arm_alphas=np.array([a[2] for a in grid]))


def common_arrays(context: DataContext, held: str) -> tuple[dict, dict, dict, dict]:
    tx, provenance, taudit = context.dataset(held);qx, query, qaudit = context.query(held)
    arrays = {**query, **provenance, **arm_arrays()}
    for name in model.INPUTS:
        arrays['train_x__' + name], arrays['query_x__' + name] = tx[name], qx[name]
    return arrays, tx, qx, dict(training=taudit, query=qaudit)


def predict() -> None:
    frozen = check();initial = sha(OUT / 'freeze.json')
    if any((OUT / name).exists() for name in ('predictions', 'prediction_seal.json')):
        raise FileExistsError('Preserve prediction attempt')
    context = DataContext(frozen);folder = OUT / 'predictions';folder.mkdir();entries = []
    for held in SOURCES:
        arrays, tx, qx, audit = common_arrays(context, held)
        labels, dates = arrays['train_regions'], pd.DatetimeIndex(pd.to_datetime(arrays['train_origins'], utc=True))
        deltas = []
        for name in model.INPUTS:
            fitted = model.ResidualModel().fit(tx[name], arrays['train_response'], labels, dates, excluded=held)
            deltas.append(fitted.predict(qx[name]))
            arrays['scaler_mean__' + name] = fitted.scaler.mean_
            arrays['scaler_scale__' + name] = fitted.scaler.scale_
        arrays['deltas'] = np.stack(deltas)
        arrays.update({'diagnostic__' + k: v for k, v in model.diagnostics(arrays['raw'], arrays['deltas']).items()})
        path = folder / f'source-{held}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        audit_path = path.with_suffix('.json')
        counts = {r: int(np.count_nonzero(labels == r)) for r in SOURCES}
        write(audit_path, dict(region=held, training_counts=counts, excluded=held,
            model=dict(trees=model.TREES, leaf=model.LEAF, seed=model.SEED, n_jobs=1),
            response=config()['response'], H1_teacher_chronological=False,
            source_H2_fine_labels_read=False, prediction_reads_2025_fine_labels=False, **audit))
        entries.append(dict(region=held, path=str(path), sha256=sha(path), audit_path=str(audit_path), audit_sha256=sha(audit_path)))
        print('Predicted residual controls', held, flush=True)
    check()
    if sha(OUT / 'freeze.json') != initial:
        raise ValueError('Freeze changed during prediction')
    seal = dict(created_utc=now(), freeze_sha256=initial, entries=entries, complete_grid=True,
        archive_count=7, raw_arm_count=55, curve_count=275, prediction_reads_2025_fine_labels=False,
        source_H2_fine_labels_read=False, target_values_opened=False)
    check_predictions(seal)
    check()
    if sha(OUT / 'freeze.json') != initial:
        raise ValueError('Freeze changed during prediction verification')
    write(OUT / 'prediction_seal.json', seal)


def check_predictions(seal: dict) -> None:
    entries = seal['entries']
    if (len(entries) != 7 or {e['region'] for e in entries} != set(SOURCES)
            or seal['archive_count'] != 7 or seal['raw_arm_count'] != 55 or seal['curve_count'] != 275
            or seal['complete_grid'] is not True or any(seal[k] is not False for k in
                ('prediction_reads_2025_fine_labels', 'source_H2_fine_labels_read', 'target_values_opened'))):
        raise ValueError('Exact7compact archives/55arms/275curves required before truth')
    expected_paths = set()
    for entry in entries:
        path = (OUT / 'predictions' / f"source-{entry['region']}.npz").resolve()
        if Path(entry['path']).resolve() != path or Path(entry['audit_path']).resolve() != path.with_suffix('.json'):
            raise ValueError('Unregistered compact archive/audit path')
        verify({str(path): entry['sha256'], str(path.with_suffix('.json')): entry['audit_sha256']})
        expected_paths.update((path, path.with_suffix('.json')))
    if {p.resolve() for p in (OUT / 'predictions').iterdir()} != expected_paths:
        raise ValueError('Unexpected compact prediction-directory files')
    frozen = check()
    if seal['freeze_sha256'] != sha(OUT / 'freeze.json'):
        raise ValueError('Prediction seal freeze binding changed')
    context = DataContext(frozen)
    for entry in entries:
        held = entry['region'];actual = load_arrays(entry['path'])
        expected, tx, _, _ = common_arrays(context, held)
        for name in model.INPUTS:
            scaler = StandardScaler().fit(tx[name], sample_weight=expected['train_weights'])
            expected['scaler_mean__' + name], expected['scaler_scale__' + name] = scaler.mean_, scaler.scale_
        deltas = actual['deltas']
        if deltas.shape != (9, 181, 24) or not np.isfinite(deltas).all():
            raise ValueError('Nine finite residual predictions required')
        expected['deltas'] = deltas
        expected.update({'diagnostic__' + k: v for k, v in model.diagnostics(expected['raw'], deltas).items()})
        parent_api.exact_fields(actual, expected, entry['path'])


def macro_scores(regional: pd.DataFrame) -> pd.DataFrame:
    keys = ['input', 'projection', 'alpha', 'factor']
    macro = regional.groupby(keys)[list(METRICS)].mean()
    macro['worst_coverage'] = regional.groupby(keys).empirical_coverage90.min()
    macro['violation'] = (np.maximum(.9 - macro.empirical_coverage90, 0)
        + np.maximum(macro.empirical_coverage90 - .95, 0) + np.maximum(.85 - macro.worst_coverage, 0))
    return macro.reset_index()


def selection(regional: pd.DataFrame) -> dict:
    macro = macro_scores(regional)
    def choose(part: pd.DataFrame) -> dict:
        passed = part[part.violation <= 1e-12]
        keys = ([] if len(passed) else ['violation']) + ['crps', 'input', 'projection', 'alpha', 'factor']
        row = (passed if len(passed) else part).sort_values(keys).iloc[0]
        return {**row.to_dict(), 'passed': bool(row.violation <= 1e-12)}
    picked = {name: choose(macro[macro.input.eq(name)]) for name in ('native', *model.INPUTS)}
    packets = {p: choose(macro[macro.input.isin(('packet_only_' + p, 'joint_' + p))]) for p in PACKETS}
    main = packets['T8'];refs = [picked['native'], picked['native_only'], *(packets[p] for p in ('D2', 'D8', 'G5'))]
    gate = bool(main['passed'] and all(main['crps'] <= .95 * r['crps']
        and main['empirical_width90'] <= r['empirical_width90']
        and main['empirical_interval_score90'] < r['empirical_interval_score90'] for r in refs))
    return dict(created_utc=now(), selections=picked, packet_selections=packets,
        source_budget_gate=gate, source_budget_reference_inputs=['native', 'native_only', 'D2', 'D8', 'G5'],
        custom_method=False, mechanism_breakthrough=False, target_test_authorized=False,
        independent_confirmation=False, new_uncertainty_mechanism=False,
        same_offset_native_feature_control_completed=True, full_probabilistic_control_completed=False,
        H_timing_causal_claim=False)


def matched_contrasts(regional: pd.DataFrame) -> pd.DataFrame:
    rows = [];keys = ['region', 'projection', 'alpha', 'factor']
    for packet in PACKETS:
        left = regional[regional.input.eq('joint_' + packet)].drop(columns='input')
        right = regional[regional.input.eq('packet_only_' + packet)].drop(columns='input')
        merged = left.merge(right, on=keys, suffixes=('_joint', '_packet'), validate='one_to_one')
        if len(merged) != 7 * 3 * 2 * 5:
            raise ValueError('Complete matched joint-minus-packet grid required')
        merged['packet'] = packet;merged['comparison'] = 'joint_minus_packet'
        for metric in METRICS:
            merged[metric + '_joint_minus_packet'] = merged[metric + '_joint'] - merged[metric + '_packet']
        rows.append(merged)
    # Same input form/projection/alpha/factor T8-G5 is descriptive, not causal.
    for prefix in ('joint_', 'packet_only_'):
        left = regional[regional.input.eq(prefix + 'T8')].drop(columns='input')
        right = regional[regional.input.eq(prefix + 'G5')].drop(columns='input')
        merged = left.merge(right, on=keys, suffixes=('_T8', '_G5'), validate='one_to_one')
        if len(merged) != 7 * 3 * 2 * 5:
            raise ValueError('Complete matched T8-minus-G5 grid required')
        merged['packet'] = prefix + 'T8_minus_G5';merged['comparison'] = 'T8_minus_G5'
        for metric in METRICS:
            merged[metric + '_T8_minus_G5'] = merged[metric + '_T8'] - merged[metric + '_G5']
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def reference_curves() -> pd.DataFrame:
    old = pd.read_csv(center.OUT / 'region_metrics.csv')
    old['reference_family'] = 'V18_center_background';old['response_offset'] = 'day_anchor'
    inherited = pd.read_csv(center.OUT / 'reference_curves.csv')
    inherited['response_offset'] = 'V17_day_anchor_or_V16_prequential'
    inherited['method'] = inherited['reference_model'];inherited['alpha'] = np.nan
    old['native_input_available'] = True
    columns = ['region', 'packet', 'reference_family', 'method', 'alpha', 'factor',
               'response_offset', 'native_input_available', *METRICS]
    if len(old) != 7 * 260 or len(inherited) != 7 * 17 * 5:
        raise ValueError('Complete V18/V17/V16 background reference curves required')
    result = pd.concat([old[columns], inherited[columns]], ignore_index=True)
    if not np.isfinite(result[list(METRICS)]).all().all():
        raise ValueError('Nonfinite background reference scores')
    return result


def score() -> None:
    if any((OUT / name).exists() for name in ('source_completed.json', *SCORE_OUTPUTS)):
        raise FileExistsError('Preserve partial or completed scores')
    frozen = check();seal = read(OUT / 'prediction_seal.json')
    check_predictions(seal)  # Full compact reconstruction and all seven hashes before 2025 fine.
    start = {str(OUT / n): sha(OUT / n) for n in ('freeze.json', 'prediction_seal.json')};rows = []
    for entry in seal['entries']:
        region = entry['region'];arrays = load_arrays(entry['path'])
        fine = sketch.source_days(Path(frozen['source']) / f'{region}_hourly.csv', '2025-01-01', '2025-07-01')
        truth = {d.value: (a / a.sum(1)[:, None]) @ sketch.model.EF for d, a in fine}
        keep = np.isin(arrays['origins'], list(truth));origins = arrays['origins'][keep]
        y = np.stack([truth[int(t)] for t in origins])
        boundary = ((arrays['raw'][keep] <= 0) | (arrays['raw'][keep] >= 1000)).mean((1, 2))
        for i, arm in enumerate(model.arms()):
            raw, _ = model.reconstruct(arrays['raw'], arrays['deltas'], arm);raw = raw[keep]
            for factor in FACTORS:
                point, paths, final_clip = model.scaled_paths(raw, factor)
                metrics = sketch.score_arrays(point, paths, y)
                rows.extend(dict(region=region, origin=pd.Timestamp(int(t), tz='UTC').isoformat(),
                    input=arm[0], projection=arm[1], alpha=arm[2], factor=factor,
                    unclipped_width90=float(arrays['diagnostic__unclipped_width90'][i, keep][j]),
                    initial_clipped_width90=float(arrays['diagnostic__initial_clipped_width90'][i, keep][j]),
                    new_clipping_rate=float(arrays['diagnostic__new_clipping_rate'][i, keep][j]),
                    native_boundary_rate=float(boundary[j]),
                    final_spread_clipping_rate=float(final_clip[j]),
                    **{k: float(v[j]) for k, v in metrics.items() if k in METRICS}) for j, t in enumerate(origins))
        print('Scored residual controls', region, flush=True)
    day = pd.DataFrame(rows);keys = ['region', 'input', 'projection', 'alpha', 'factor']
    if (len(day[['region', 'origin']].drop_duplicates()) != 1145 or day.duplicated(keys + ['origin']).any()
            or not np.isfinite(day.select_dtypes(include='number')).all().all()):
        raise ValueError('Changed1145day common source evaluation cohort')
    for _, part in day.groupby(keys[1:]):
        if len(part) != 1145:
            raise ValueError('Every275curve must retain all1145days')
    regional = day.groupby(keys)[list(METRICS)].mean().reset_index()
    if len(regional) != 7 * 275:
        raise ValueError('Complete275curve source grid required')
    refs = reference_curves()
    baseline = regional[regional.input.eq('native')].merge(
        refs[refs.reference_family.eq('V18_center_background') & refs.packet.eq('native')],
        on=['region', 'factor'], suffixes=('_new', '_old'), validate='one_to_one')
    if len(baseline) != 35 or any(not np.allclose(baseline[k + '_new'], baseline[k + '_old'], rtol=0, atol=1e-9) for k in METRICS):
        raise ValueError('All native scores must match frozen V18')
    tables = dict(day_metrics=day, region_metrics=regional, macro_metrics=macro_scores(regional),
                  matched_input_contrasts=matched_contrasts(regional), reference_curves=refs)
    for name, table in tables.items():
        table.to_csv(OUT / f'{name}.csv', index=False, mode='x')
    write(OUT / 'source_choice.json', selection(regional))
    check();check_predictions(seal);verify(start)
    write(OUT / 'source_completed.json', dict(created_utc=now(), version=21,
        freeze_sha256=start[str(OUT / 'freeze.json')], prediction_seal_sha256=start[str(OUT / 'prediction_seal.json')],
        output_hashes={str(OUT / name): sha(OUT / name) for name in SCORE_OUTPUTS},
        target_values_opened=False, unseen_values_opened=False, custom_method=False,
        mechanism_breakthrough=False, full_probabilistic_control_completed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'predict', 'score'), required=True)
    parser.add_argument('--parent-sha256', help='Root-approved final R1 PASS manifest; prepare only')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.parent_sha256)
    elif args.parent_sha256 is not None:
        parser.error('--parent-sha256 may only be supplied to prepare')
    else:
        {'predict': predict, 'score': score}[args.stage]()
