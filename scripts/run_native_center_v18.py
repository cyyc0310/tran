"""Source-only sealed center-translation pilot on immutable V17 predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import pandas as pd
from scripts import native_center_v18 as model
from scripts import run_local_sketch_v17 as prior

SOURCES, PACKETS, FACTORS = model.SOURCES, model.PACKETS, model.FACTORS
METRICS = prior.METRICS
OUT = ROOT/'results/native-center-v18'
PARENT = ROOT/'results/local-sketch-v17'
V16 = ROOT/'results/daily-context-v16'
V16_MANIFEST = ROOT/'docs/evidence/context-v16-verification.json'
V16_MANIFEST_SHA = '03b38cd84d39390fa02947ed31fcff6110942e8f9ab6afb78a8282c16cefedde'
CODE = ('scripts/native_center_v18.py', 'scripts/run_native_center_v18.py',
        'tests/test_native_center_v18.py', 'docs/NATIVE_CENTER_V18_PROTOCOL.md')
sha, read, write, verify, now = prior.sha, prior.read, prior.write, prior.verify, prior.now
DAYS = pd.date_range('2025-01-01', '2025-06-30', tz='UTC')


def config() -> dict[str, Any]:
    return dict(version=18, sources=list(SOURCES), packets=list(PACKETS), lag=7,
                alphas=list(model.ALPHAS), factors=list(FACTORS), samples=128, archive_count=35,
                curve_count=260, custom_method=False, target_values_opened=False,
                unseen_values_opened=False, target_test_authorized=False, mechanism_breakthrough=False,
                prediction_reads_2025_fine_labels=False, new_risk_fit=False)


def ancestor_inputs() -> tuple[dict, dict[str, str]]:
    """Validate completed ancestor seals without reading reference-table numbers."""
    frozen, data = prior.check()
    seal, completed = read(PARENT/'prediction_seal.json'), read(PARENT/'source_completed.json')
    freeze_sha = sha(PARENT/'freeze.json')
    if (completed['version'] != 17 or completed['freeze_sha256'] != freeze_sha
            or seal['freeze_sha256'] != freeze_sha
            or completed['prediction_seal_sha256'] != sha(PARENT/'prediction_seal.json')
            or completed['target_values_opened'] is not False or completed['unseen_values_opened'] is not False):
        raise ValueError('Unchanged completed source-only V17 required')
    prior.check_predictions(seal)
    verify(completed['output_hashes'])
    region_path = str(PARENT/'region_metrics.csv')
    if region_path not in completed['output_hashes']:
        raise ValueError('V17 reference table must belong to completed outputs')
    paths = [PARENT/n for n in ('freeze.json', 'data_seal.json', 'prediction_seal.json', 'source_completed.json')]
    hashes = {**frozen['input_hashes'], **data['packet_hashes'], **completed['output_hashes'],
              **{str(p): sha(p) for p in paths}}
    for entry in seal['entries']:
        hashes[entry['path']] = entry['sha256']
        if 'audit_path' in entry:
            hashes[entry['audit_path']] = entry['audit_sha256']
    verify({str(V16_MANIFEST): V16_MANIFEST_SHA})
    evidence, v16done = read(V16_MANIFEST), read(V16/'source_completed.json')
    if (evidence['version'] != 16 or evidence['target_or_unseen_values_opened'] is not False
            or v16done['version'] != 16 or v16done['target_values_opened'] is not False
            or v16done['unseen_values_opened'] is not False):
        raise ValueError('Sealed source-only V16 references required')
    for path in (V16/'source_completed.json', V16/'region_metrics.csv', V16/'prediction_seal.json'):
        expected = evidence['artifact_hashes'][str(path.relative_to(ROOT))]
        verify({str(path): expected}); hashes[str(path)] = expected
    if (v16done['prediction_seal_sha256'] != hashes[str(V16/'prediction_seal.json')]
            or v16done['output_hashes'].get(str(V16/'region_metrics.csv')) != hashes[str(V16/'region_metrics.csv')]):
        raise ValueError('V16 completed parent/reference binding changed')
    hashes[str(V16_MANIFEST)] = V16_MANIFEST_SHA
    return frozen, hashes


def prepare() -> None:
    if (OUT/'freeze.json').exists() or (OUT/'predictions').exists():
        raise FileExistsError('Preserve complete or partial center experiment')
    ancestor, inputs = ancestor_inputs()
    code = {**ancestor['code_hashes'], **{str(ROOT/p): sha(ROOT/p) for p in CODE}}
    if len(code) != 58:
        raise ValueError('Exact inherited54 plus new4 code closure required')
    verify(code); verify(inputs)
    frozen = {**config(), 'created_utc': now(), 'source': ancestor['source'],
              'code_hashes': code, 'input_hashes': inputs,
              'environment': ancestor['environment'], 'environment_sha256': ancestor['environment_sha256']}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'freeze.json', frozen)


def check() -> dict:
    frozen = read(OUT/'freeze.json')
    ancestor = read(PARENT/'freeze.json')
    expected_code = {**ancestor['code_hashes'], **{str(ROOT/p): sha(ROOT/p) for p in CODE}}
    if (any(frozen[k] != v for k, v in config().items()) or len(frozen['code_hashes']) != 58
            or frozen['code_hashes'] != expected_code or frozen['source'] != ancestor['source']
            or frozen['input_hashes'].get(str(V16_MANIFEST)) != V16_MANIFEST_SHA
            or any(str(Path(frozen['source'])/f'{region}_hourly.csv') not in frozen['input_hashes'] for region in SOURCES)):
        raise ValueError('Changed frozen center pilot contract')
    import hashlib
    env = prior.environment()
    if (frozen['environment'] != env or frozen['environment_sha256'] !=
            hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()):
        raise ValueError('Changed execution environment')
    verify({**frozen['code_hashes'], **frozen['input_hashes']})
    return frozen


def registered_grid() -> set[tuple[str, str]]:
    return {(r, p) for r in SOURCES for p in ('native', *PACKETS)}


def native_input(region: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(PARENT/'predictions'/f'source-{region}-native.npz', allow_pickle=False) as z:
        if (not np.array_equal(z['origins'], DAYS.asi8) or z['models'].tolist() != ['native']
                or z['raw'].shape != (181, 128, 24) or not np.isfinite(z['raw']).all()
                or np.any(z['raw'] < 0) or np.any(z['raw'] > 1000)
                or z['donor_ids'].shape != (181, 128) or not np.issubdtype(z['donor_ids'].dtype, np.integer)
                or np.any(z['donor_ids'] < 0)):
            raise ValueError('Exact registered V17 native source arrays required')
        return z['raw'].copy(), z['donor_ids'].copy()


def archived_context(region: str, packet: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Read the registered point/anchor/EWMA layout from the sealed ancestor."""
    expected_channels = {'D2': 2, 'D8': 10, 'T8': 10, 'G5': 6}
    width = expected_channels[packet]
    with np.load(PARENT/'predictions'/f'source-{region}-7-{packet}.npz', allow_pickle=False) as z:
        center = model.extract_et_center(z)
        features, anchors = z['features'].copy(), z['anchors'].copy()
        if (not np.array_equal(z['origins'], DAYS.asi8) or center.shape != (181, 24)
                or features.shape != (181, 1+4*width+6+4) or anchors.shape != (181,)
                or not np.isfinite(features).all() or not np.isfinite(anchors).all()):
            raise ValueError('Registered complete V17 feature schema required')
    # x = [anchor, latest(C), raw-energy-EWMA(C), std(C), weekday(C), calendar(6), availability(4)].
    ewma = features[:, 1+width:1+2*width]
    if not np.allclose(ewma[:, -1], anchors, rtol=0, atol=1e-9):
        raise ValueError('Explicit EWMA schema does not contain the shared day anchor')
    return center, features, anchors, ewma[:, :4].copy() if packet == 'T8' else None


def context(region: str, packet: str, histories: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """Explicit V17 feature schema plus lawful packet replay; no unverified column."""
    center, features, anchors, block = archived_context(region, packet)
    replay = [prior.checked_features(histories, packet, region, 7, d) for d in DAYS]
    expected_features = np.stack([row[0] for row in replay])
    expected_anchors = np.array([row[1] for row in replay])
    if (not np.allclose(features, expected_features, rtol=0, atol=1e-9)
            or not np.allclose(anchors, expected_anchors, rtol=0, atol=1e-9)):
        raise ValueError('V17 features/anchors disagree with lawful released-packet replay')
    return center, anchors, block, dict(common_anchor_max_abs_difference=max(row[3] for row in replay),
        feature_replay_max_abs_difference=float(abs(features-expected_features).max()),
        query_availability=[row[2] for row in replay])


def predict() -> None:
    check(); start = {str(OUT/'freeze.json'): sha(OUT/'freeze.json')}
    if (OUT/'predictions').exists() or (OUT/'prediction_seal.json').exists():
        raise FileExistsError('Preserve prediction attempt')
    folder = OUT/'predictions'; folder.mkdir()
    histories = {}
    for packet, n in prior.PACKETS.items():
        table = pd.read_csv(PARENT/'packets'/f'{packet}.csv', dtype={f'v{i}': object for i in range(n)})
        for region in SOURCES:
            histories[packet, region, 7] = prior.model.PacketHistory(table[table.region.eq(region)], packet, 7)
    entries = []
    for region in SOURCES:
        native, ids = native_input(region)
        path = folder/f'source-{region}-native.npz'
        with path.open('xb') as f:
            np.savez_compressed(f, origins=DAYS.asi8, raw=native, donor_ids=ids)
        entries.append(dict(region=region, packet='native', path=str(path), sha256=sha(path)))
        common_anchor = None
        for packet in PACKETS:
            center, anchor, block, audit = context(region, packet, histories)
            if common_anchor is None:
                common_anchor = anchor.copy()
            if np.max(abs(anchor-common_anchor)) > 1e-9:
                raise ValueError('All V18 raw_day anchors must match D2')
            arrays = model.build(native, ids, center, anchor, block, packet)
            path = folder/f'source-{region}-{packet}.npz'
            with path.open('xb') as f:
                np.savez_compressed(f, origins=DAYS.asi8, **arrays)
            audit_path = path.with_suffix('.json')
            write(audit_path, dict(region=region, packet=packet, lag=7, methods=list(arrays['methods']),
                  alphas=arrays['alphas'].tolist(), prediction_reads_2025_fine_labels=False, **audit))
            entries.append(dict(region=region, packet=packet, path=str(path), sha256=sha(path),
                                audit_path=str(audit_path), audit_sha256=sha(audit_path)))
            print('Predicted', region, packet, flush=True)
    check(); verify(start)
    seal = dict(created_utc=now(), freeze_sha256=start[str(OUT/'freeze.json')], entries=entries,
                complete_grid=True, source2025fine_labels_read=False, target_values_opened=False)
    check_predictions(seal)
    write(OUT/'prediction_seal.json', seal)


def check_predictions(seal: dict) -> None:
    actual = [(e['region'], e['packet']) for e in seal['entries']]
    if (len(actual) != 35 or set(actual) != registered_grid() or seal['complete_grid'] is not True
            or seal['source2025fine_labels_read'] is not False or seal['target_values_opened'] is not False):
        raise ValueError('Exact35 prediction grid must be sealed before source truth')
    for entry in seal['entries']:
        region, packet = entry['region'], entry['packet']
        path = (OUT/'predictions'/f'source-{region}-{packet}.npz').resolve()
        if Path(entry['path']).resolve() != path:
            raise ValueError('Unregistered center prediction path')
        verify({str(path): entry['sha256']})
        native, ids = native_input(region)
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['origins'], DAYS.asi8) or not np.array_equal(z['donor_ids'], ids):
                raise ValueError('Common origins/native donor IDs changed')
            if packet == 'native':
                if set(z.files) != {'origins', 'raw', 'donor_ids'} or not np.array_equal(z['raw'], native):
                    raise ValueError('Exact native scenario identity failed')
                continue
            et, _, anchor, block = archived_context(region, packet)
            expected = model.build(native, ids, et, anchor, block, packet)
            if set(z.files) != {'origins', *expected} or any(not np.array_equal(z[k], v) for k, v in expected.items()):
                raise ValueError('Registered projection/alpha grid or center diagnostics changed')
        audit_path = path.with_suffix('.json')
        if Path(entry['audit_path']).resolve() != audit_path:
            raise ValueError('Unregistered prediction audit path')
        verify({str(audit_path): entry['audit_sha256']})


def macro_scores(regional: pd.DataFrame) -> pd.DataFrame:
    keys = ['packet', 'method', 'alpha', 'factor']
    macro = regional.groupby(keys)[list(METRICS)].mean()
    macro['worst_coverage'] = regional.groupby(keys).empirical_coverage90.min()
    macro['violation'] = (np.maximum(.9-macro.empirical_coverage90, 0)
                          +np.maximum(macro.empirical_coverage90-.95, 0)
                          +np.maximum(.85-macro.worst_coverage, 0))
    return macro.reset_index()


def selection(regional: pd.DataFrame) -> dict[str, Any]:
    macro = macro_scores(regional)
    def choose(part: pd.DataFrame) -> dict:
        passed = part[part.violation <= 1e-12]
        keys = ([] if len(passed) else ['violation'])+['crps', 'packet', 'method', 'alpha', 'factor']
        row = (passed if len(passed) else part).sort_values(keys).iloc[0]
        return {**row.to_dict(), 'passed': bool(row.violation <= 1e-12)}
    picked = {p: choose(macro[macro.packet.eq(p)]) for p in ('native', *PACKETS)}
    primary = picked['T8']
    references = [picked[p] for p in ('D2', 'D8', 'G5', 'native')]
    gain = bool(primary['passed'] and all(primary['crps'] <= .95*r['crps']
        and primary['empirical_width90'] <= r['empirical_width90']
        and primary['empirical_interval_score90'] < r['empirical_interval_score90'] for r in references))
    return dict(created_utc=now(), selections=picked,
        strongest_limited_packet=choose(macro[macro.packet.isin(('D2', 'D8', 'G5'))]),
        source_information_gate=gain, source_information_reference_packets=['D2', 'D8', 'G5', 'native'],
        primary_within_packet_comparisons='all ordinary projections, alphas and spread factors retained',
        custom_method=False, target_test_authorized=False, mechanism_breakthrough=False,
        H_timing_causal_claim=False, independent_confirmation=False,
        native_feature_refit_control_completed=False, new_uncertainty_mechanism=False)


def matched_contrasts(regional: pd.DataFrame) -> pd.DataFrame:
    keys = ['region', 'method', 'alpha', 'factor']
    left = regional[regional.packet.eq('T8')].drop(columns='packet')
    right = regional[regional.packet.eq('G5')].drop(columns='packet')
    joined = left.merge(right, on=keys, suffixes=('_T8', '_G5'), validate='one_to_one')
    if len(joined) != 7*4*3*5:
        raise ValueError('Complete same-projection/alpha/factor T8-G5 controls required')
    for metric in METRICS:
        joined[metric+'_T8_minus_G5'] = joined[metric+'_T8']-joined[metric+'_G5']
    return joined


def reference_curves() -> pd.DataFrame:
    """Inherited complete ordinary curves; their information sets are disclosed."""
    v17 = pd.read_csv(PARENT/'region_metrics.csv')
    v17 = v17[v17.lag.isin((0, 7)) & v17.model.isin(('native', 'ET_global', 'ET_hetero', 'ET_QRF'))].copy()
    v17['reference_family'] = 'V17_direct'; v17['reference_model'] = v17['model']
    v17['native_input_available'] = v17.packet.eq('native')
    v16 = pd.read_csv(V16/'region_metrics.csv')
    v16 = v16[v16.lag.eq(7) & v16.arm.eq('additive')].copy()
    v16['reference_family'] = 'V16_additive'; v16['reference_model'] = v16['forecaster']
    v16['packet'] = 'daily_fuel_with_prequential_residual_law'
    v16['native_input_available'] = True
    cols = ['region', 'packet', 'reference_family', 'reference_model', 'factor', 'native_input_available', *METRICS]
    result = pd.concat([v17[cols], v16[cols]], ignore_index=True)
    if (len(result) != 7*(1+4*3+4)*5 or not np.isfinite(result[list(METRICS)]).all().all()
            or not set(result.region) == set(SOURCES)
            or result.duplicated(cols[:5]).any()):
        raise ValueError('Complete sealed V16/V17 reference curves required')
    expected = {(r, p, name, f) for r in SOURCES for p in PACKETS
                for name in ('ET_global', 'ET_hetero', 'ET_QRF') for f in FACTORS}
    actual = {(row.region, row.packet, row.reference_model, row.factor)
              for row in result[result.reference_family.eq('V17_direct') & result.packet.ne('native')].itertuples()}
    if actual != expected:
        raise ValueError('Missing same-packet direct ET/hetero/QRF reference curve')
    if set(v16.forecaster) != {'latest', 'weekday', 'ewma', 'seasonal_kalman'}:
        raise ValueError('All four V16 additive ordinary forecasters required')
    expected_v16 = {(region, forecaster, factor) for region in SOURCES
                    for forecaster in ('latest', 'weekday', 'ewma', 'seasonal_kalman') for factor in FACTORS}
    if set(zip(v16.region, v16.forecaster, v16.factor)) != expected_v16:
        raise ValueError('Missing registered V16 additive reference setting')
    return result


def score() -> None:
    if any((OUT/name).exists() for name in ('source_completed.json', 'day_metrics.csv', 'region_metrics.csv')):
        raise FileExistsError('Preserve completed or partial scoring attempt')
    frozen = check()
    seal = read(OUT/'prediction_seal.json')
    if seal['freeze_sha256'] != sha(OUT/'freeze.json'):
        raise ValueError('Prediction freeze binding changed')
    check_predictions(seal)  # Complete grid and alpha-zero exactness before truth or reference numbers.
    start = {str(OUT/name): sha(OUT/name) for name in ('freeze.json', 'prediction_seal.json')}
    rows = []
    for region in SOURCES:
        observed = prior.source_days(Path(frozen['source'])/f'{region}_hourly.csv', '2025-01-01', '2025-07-01')
        truth = {day.value: (a/a.sum(1)[:, None])@prior.model.EF for day, a in observed}
        for entry in (e for e in seal['entries'] if e['region'] == region):
            with np.load(entry['path'], allow_pickle=False) as z:
                keep = np.isin(z['origins'], list(truth)); origins = z['origins'][keep]
                y = np.stack([truth[int(t)] for t in origins])
                native = entry['packet'] == 'native'
                grid = (('native', 0.),) if native else model.arms(entry['packet'])
                raw_arrays = z['raw'][None] if native else z['raw']
                for i, ((method, alpha), raw_all) in enumerate(zip(grid, raw_arrays)):
                    raw = raw_all[keep]
                    uncut_width = np.diff(np.sort(raw, axis=1)[:, [6, 121], :], axis=1)[:, 0].mean(1)
                    base_width = np.diff(np.sort(np.clip(raw, 0, 1000), axis=1)[:, [6, 121], :], axis=1)[:, 0].mean(1)
                    clip_rate = ((raw < 0) | (raw > 1000)).mean((1, 2))
                    boundary = ((raw <= 0) | (raw >= 1000)).mean((1, 2)) if native else z['native_boundary_rate'][keep]
                    for factor in FACTORS:
                        point, paths, final_clip = model.scaled_paths(raw, factor)
                        metrics = prior.score_arrays(point, paths, y)
                        rows.extend(dict(region=region, origin=pd.Timestamp(int(t), tz='UTC').isoformat(),
                            packet=entry['packet'], method=method, alpha=alpha, factor=factor,
                            unclipped_width90=float(uncut_width[j]), base_clipped_width90=float(base_width[j]),
                            native_boundary_rate=float(boundary[j]), new_clipping_rate=float(clip_rate[j]),
                            final_spread_clipping_rate=float(final_clip[j]),
                            **{k: float(v[j]) for k, v in metrics.items() if k in METRICS}) for j, t in enumerate(origins))
        print('Scored', region, flush=True)
    day = pd.DataFrame(rows); keys = ['region', 'packet', 'method', 'alpha', 'factor']
    if (len(day[['region', 'origin']].drop_duplicates()) != 1145 or day.duplicated(keys+['origin']).any()
            or not np.isfinite(day.select_dtypes(include='number')).all().all()):
        raise ValueError('Changed complete source evaluation cohort')
    for _, part in day.groupby(keys[1:]):
        if len(part) != 1145:
            raise ValueError('Every fixed V18 curve must use the same1145source region-days')
    regional = day.groupby(keys)[list(METRICS)].mean().reset_index()
    refs = reference_curves()
    native = regional[regional.packet.eq('native')]
    old_native = refs[refs.packet.eq('native')]
    merged = native.merge(old_native, on=['region', 'factor'], suffixes=('_new', '_old'), validate='one_to_one')
    if len(merged) != 35 or any(not np.allclose(merged[k+'_new'], merged[k+'_old'], rtol=0, atol=1e-9) for k in METRICS):
        raise ValueError('Native scores must match frozen V17 across all eight metrics')
    outputs = []
    tables = dict(day_metrics=day, region_metrics=regional, macro_metrics=macro_scores(regional),
                  matched_packet_contrasts=matched_contrasts(regional), reference_curves=refs)
    for name, table in tables.items():
        path = OUT/f'{name}.csv'; table.to_csv(path, index=False, mode='x'); outputs.append(path)
    choice = OUT/'source_choice.json'; write(choice, selection(regional)); outputs.append(choice)
    check(); check_predictions(seal); verify(start)
    write(OUT/'source_completed.json', dict(created_utc=now(), version=18,
          freeze_sha256=start[str(OUT/'freeze.json')], prediction_seal_sha256=start[str(OUT/'prediction_seal.json')],
          output_hashes={str(path): sha(path) for path in outputs}, target_values_opened=False,
          unseen_values_opened=False, custom_method=False, mechanism_breakthrough=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'predict', 'score'), required=True)
    {'prepare': prepare, 'predict': predict, 'score': score}[parser.parse_args().stage]()
