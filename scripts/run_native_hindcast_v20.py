"""Immutable source-only native hindcast data stage; no H2 fine labels or scores."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
from scripts import native_hindcast_v20 as model
from scripts import run_native_center_v18 as parent_api

sha, read, write, now = parent_api.sha, parent_api.read, parent_api.write, parent_api.now
OUT = ROOT/'results/native-hindcast-v20'
PARENT = ROOT/'results/native-center-v18'
MANIFEST = ROOT/'docs/evidence/native-center-v18-verification.json'
MANIFEST_SHA = 'efa95d7d79284d3157426fae9cd46785d75355fa0daee4fa479e1edc9571a3aa'
OLD = ROOT/'results/weather-coarse-v14-round1/prefit_freeze.json'
CODE = ('scripts/native_hindcast_v20.py', 'scripts/run_native_hindcast_v20.py',
        'tests/test_native_hindcast_v20.py', 'docs/NATIVE_HINDCAST_V20_PROTOCOL.md')
SOURCES = model.SOURCES


def verify(hashes: dict[str, str]) -> None:
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Changed frozen bytes: '+path)


def config() -> dict:
    return dict(version=20, sources=list(SOURCES), proper_start='2024-01-25', proper_end_exclusive='2024-07-01',
        hindcast_start='2024-07-01', hindcast_end='2024-12-31', samples=128, seed=17023,
        prediction_count=91, bank_count=21, new_model_fit=False, new_risk_fit=False, scores_computed=False,
        source_H2_fine_labels_read=False, source_2025_fine_labels_read=False, target_values_opened=False,
        unseen_values_opened=False, H1_teacher_chronological_hindcast=False,
        H1_teacher_within_proper_future_donors=True,
        publication_vintages_certified=False, target_authorized=False, mechanism_breakthrough=False)


def parent_metadata() -> tuple[dict, dict, dict]:
    verify({str(MANIFEST): MANIFEST_SHA})
    evidence, frozen, completed = read(MANIFEST), read(PARENT/'freeze.json'), read(PARENT/'source_completed.json')
    seal = read(PARENT/'prediction_seal.json')
    for name in ('freeze.json', 'prediction_seal.json', 'source_completed.json'):
        path = PARENT/name
        verify({str(path): evidence['artifact_hashes'][str(path.relative_to(ROOT))]})
    if (frozen['sources'] != list(SOURCES) or completed['version'] != 18
            or completed['freeze_sha256'] != sha(PARENT/'freeze.json')
            or seal['freeze_sha256'] != completed['freeze_sha256']
            or completed['prediction_seal_sha256'] != sha(PARENT/'prediction_seal.json')
            or completed['target_values_opened'] is not False or completed['unseen_values_opened'] is not False
            or seal['complete_grid'] is not True or len(seal['entries']) != 35):
        raise ValueError('Completed unchanged source-only V18 parent required')
    actual = [(e['region'], e['packet']) for e in seal['entries']]
    if set(actual) != {(r, p) for r in SOURCES for p in ('native', 'D2', 'D8', 'T8', 'G5')}:
        raise ValueError('Changed V18 parent prediction registry')
    verify({str(OLD): frozen['input_hashes'][str(OLD)]})
    old = read(OLD)
    if old['source'] != frozen['source']:
        raise ValueError('Coarse source directory differs from native parent')
    return frozen, evidence, old


def input_paths(source: str) -> set[str]:
    return ({str(MANIFEST), str(OLD), str(Path(source)/'monthly.csv'), str(Path(sys.executable).resolve())}
        | {str(PARENT/n) for n in ('freeze.json', 'prediction_seal.json', 'source_completed.json')}
        | {str(Path(source)/f'{r}_hourly.csv') for r in SOURCES}
        | {str(PARENT/'predictions'/f'source-{r}-native.npz') for r in SOURCES})


def prepare() -> None:
    if any((OUT/name).exists() for name in ('freeze.json', 'predictions', 'banks', 'prediction_seal.json')):
        raise FileExistsError('Preserve complete or partial V20 attempt')
    parent, evidence, old = parent_metadata()
    inputs = {}
    for path in input_paths(parent['source']):
        p = Path(path);digest = sha(p)
        if p.parent == Path(parent['source']):
            wanted = old['input_hashes'][path] if p.name == 'monthly.csv' else parent['input_hashes'][path]
            if digest != wanted:
                raise ValueError('Source inputs changed since sealed parent')
        if p.parent == PARENT/'predictions' and digest != evidence['artifact_hashes'][str(p.relative_to(ROOT))]:
            raise ValueError('Inherited native archive changed')
        inputs[path] = digest
    code = {**parent['code_hashes'], **{str(ROOT/p): sha(ROOT/p) for p in CODE}}
    if len(code) != 62 or len(inputs) != 21:
        raise ValueError('Exact62 code and21 input closure required')
    verify(code);verify(inputs)
    environment = parent_api.prior.environment()
    frozen = {**config(), 'created_utc': now(), 'source': parent['source'], 'code_hashes': code,
        'input_hashes': inputs, 'environment': environment,
        'environment_sha256': hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()}
    OUT.mkdir(parents=True, exist_ok=True);write(OUT/'freeze.json', frozen)


def check() -> dict:
    frozen = read(OUT/'freeze.json')
    if any(frozen[k] != v for k, v in config().items()) or len(frozen['code_hashes']) != 62:
        raise ValueError('Changed frozen hindcast contract')
    parent, evidence, old = parent_metadata()
    expected_code = {**parent['code_hashes'], **{str(ROOT/p): sha(ROOT/p) for p in CODE}}
    if (frozen['source'] != parent['source'] or frozen['code_hashes'] != expected_code
            or set(frozen['input_hashes']) != input_paths(parent['source'])):
        raise ValueError('Source registry or exact closure changed')
    for region in SOURCES:
        name = str(Path(parent['source'])/f'{region}_hourly.csv')
        if frozen['input_hashes'][name] != parent['input_hashes'][name]:
            raise ValueError('Source fine digest differs from parent')
        name = str(PARENT/'predictions'/f'source-{region}-native.npz')
        if frozen['input_hashes'][name] != evidence['artifact_hashes'][str(Path(name).relative_to(ROOT))]:
            raise ValueError('Source native digest differs from parent')
    name = str(Path(parent['source'])/'monthly.csv')
    if frozen['input_hashes'][name] != old['input_hashes'][name]:
        raise ValueError('Source coarse digest differs from sealed V14')
    environment = parent_api.prior.environment()
    if (environment != frozen['environment'] or frozen['environment_sha256'] !=
            hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()):
        raise ValueError('Execution environment changed')
    verify({**frozen['code_hashes'], **frozen['input_hashes']})
    return frozen


def bank_name(pair: tuple[str, str]) -> str:
    return 'bank-exclude-'+'-'.join(pair)+'.npz'


def native_copy(region: str, frames: dict) -> dict[str, np.ndarray]:
    path = PARENT/'predictions'/f'source-{region}-native.npz'
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'origins', 'raw', 'donor_ids'}:
            raise ValueError('Exact inherited native schema required')
        arrays = {k: z[k].copy() for k in z.files}
    rows = model.query_rows(frames, region)
    if (not np.array_equal(arrays['origins'], model.QUERY_DAYS.asi8) or arrays['raw'].shape != (181, 128, 24)
            or not np.isfinite(arrays['raw']).all() or np.any(arrays['raw'] < 0) or np.any(arrays['raw'] > 1000)
            or arrays['donor_ids'].shape != (181, 128) or not np.issubdtype(arrays['donor_ids'].dtype, np.integer)
            or np.any(arrays['donor_ids'] < 0) or np.any(arrays['donor_ids'] >= len(rows['bank_days']))):
        raise ValueError('Invalid inherited source native arrays or donor range')
    return {**arrays, 'center': arrays['raw'].mean(1), **rows}


def predict() -> None:
    frozen = check();initial = sha(OUT/'freeze.json')
    if any((OUT/name).exists() for name in ('predictions', 'banks', 'prediction_seal.json', 'completed.json')):
        raise FileExistsError('Preserve prediction attempt')
    (OUT/'predictions').mkdir();(OUT/'banks').mkdir()
    frames = {r: model.proper_frame(Path(frozen['source'])/f'{r}_hourly.csv', r) for r in SOURCES}
    monthly = model.monthly_prefix(Path(frozen['source'])/'monthly.csv')
    laws = {(r, split): model.coarse_arrays(monthly, r, split) for r in SOURCES
            for split in ('H1_retrospective', 'H2_hindcast')}
    banks, bank_entries, entries = {}, [], []
    for pair in itertools.combinations(SOURCES, 2):
        subset = tuple(r for r in SOURCES if r not in pair)
        bank = model.native._bank(frames, subset)
        arrays = model.bank_arrays(bank, pair);path = OUT/'banks'/bank_name(pair)
        with path.open('xb') as stream:np.savez_compressed(stream, **arrays)
        banks[pair] = bank
        bank_entries.append(dict(excluded=list(pair), path=str(path), sha256=sha(path), audit=bank.audit_))
    for outer in SOURCES:
        copied = native_copy(outer, frames);path = OUT/'predictions'/f'query-{outer}-native.npz'
        with path.open('xb') as stream:np.savez_compressed(stream, **copied)
        entries.append(dict(kind='query_copy', outer=outer, teacher=None, split='query_copy', path=str(path), sha256=sha(path),
            inherited_path=str(PARENT/'predictions'/f'source-{outer}-native.npz')))
        for teacher in SOURCES:
            if teacher == outer:continue
            pair = model.exclusions(outer, teacher)
            for split in ('H1_retrospective', 'H2_hindcast'):
                arrays = model.teachers(banks[pair], laws[teacher, split], split)
                path = OUT/'predictions'/f'{split}-outer-{outer}-teacher-{teacher}.npz'
                with path.open('xb') as stream:np.savez_compressed(stream, **arrays)
                entries.append(dict(kind='teacher', outer=outer, teacher=teacher, split=split, path=str(path), sha256=sha(path),
                    excluded=list(pair), bank_path=str(OUT/'banks'/bank_name(pair)),
                    chronological_hindcast=split == 'H2_hindcast', within_proper_future_donors=split == 'H1_retrospective'))
                print('Native teacher', split, outer, teacher, flush=True)
    check()
    if sha(OUT/'freeze.json') != initial:
        raise ValueError('Freeze changed during hindcast generation')
    seal = dict(created_utc=now(), freeze_sha256=initial, entries=entries, banks=bank_entries,
                complete_grid=True, source_H2_fine_labels_read=False, source_2025_fine_labels_read=False,
                target_values_opened=False, new_model_fit=False, scores_computed=False)
    check_predictions(seal)
    check()
    if sha(OUT/'freeze.json') != initial:
        raise ValueError('Freeze changed during full prediction verification')
    write(OUT/'prediction_seal.json', seal)
    write(OUT/'completed.json', dict(created_utc=now(), version=20, freeze_sha256=initial,
        prediction_seal_sha256=sha(OUT/'prediction_seal.json'), predictions=91, banks=21,
        source_H2_fine_labels_read=False, source_2025_fine_labels_read=False,
        target_values_opened=False, scores_computed=False, new_model_fit=False))


def expected_grid() -> set[tuple[str, str | None, str]]:
    return ({(r, None, 'query_copy') for r in SOURCES}
        | {(r, s, split) for r in SOURCES for s in SOURCES if r != s
           for split in ('H1_retrospective', 'H2_hindcast')})


def check_predictions(seal: dict) -> None:
    actual = [(e['outer'], e['teacher'], e['split']) for e in seal['entries']]
    pairs = [tuple(e['excluded']) for e in seal['banks']]
    if (len(actual) != 91 or set(actual) != expected_grid() or len(pairs) != 21
            or set(pairs) != set(itertools.combinations(SOURCES, 2)) or seal['complete_grid'] is not True
            or any(seal[k] is not False for k in ('source_H2_fine_labels_read', 'source_2025_fine_labels_read',
                                                'target_values_opened', 'new_model_fit', 'scores_computed'))):
        raise ValueError('Exact91 predictions and21 distinct banks required')
    hashes = {}
    for entry in seal['banks']:
        pair = tuple(entry['excluded']);path = (OUT/'banks'/bank_name(pair)).resolve()
        if Path(entry['path']).resolve() != path:
            raise ValueError('Unregistered bank path')
        hashes[str(path)] = entry['sha256']
    for entry in seal['entries']:
        outer, teacher, split = entry['outer'], entry['teacher'], entry['split']
        name = f'query-{outer}-native.npz' if teacher is None else f'{split}-outer-{outer}-teacher-{teacher}.npz'
        path = (OUT/'predictions'/name).resolve()
        if Path(entry['path']).resolve() != path or entry['kind'] != ('query_copy' if teacher is None else 'teacher'):
            raise ValueError('Unregistered prediction path or kind')
        if teacher is not None:
            if (entry['excluded'] != list(model.exclusions(outer, teacher))
                    or Path(entry['bank_path']).resolve() != (OUT/'banks'/bank_name(model.exclusions(outer, teacher))).resolve()
                    or entry['chronological_hindcast'] is not (split == 'H2_hindcast')
                    or entry['within_proper_future_donors'] is not (split == 'H1_retrospective')):
                raise ValueError('Teacher split semantics or double exclusion changed')
        elif entry['inherited_path'] != str(PARENT/'predictions'/f'source-{outer}-native.npz'):
            raise ValueError('Query copy must bind the registered parent source archive')
        hashes[str(path)] = entry['sha256']
    verify(hashes)
    frozen = check()
    if seal['freeze_sha256'] != sha(OUT/'freeze.json'):
        raise ValueError('Prediction seal freeze binding changed')
    frames = {r: model.proper_frame(Path(frozen['source'])/f'{r}_hourly.csv', r) for r in SOURCES}
    monthly = model.monthly_prefix(Path(frozen['source'])/'monthly.csv')
    laws = {(r, split): model.coarse_arrays(monthly, r, split) for r in SOURCES
            for split in ('H1_retrospective', 'H2_hindcast')}
    banks = {}
    for entry in seal['banks']:
        with np.load(entry['path'], allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        pair = tuple(entry['excluded']);banks[pair] = model.restore_bank(arrays, pair)
        # Match rows and all H1 values to an independently re-fit declared source subset.
        subset = tuple(r for r in SOURCES if r not in pair)
        reconstructed = model.native._bank(frames, subset)
        expected = model.bank_arrays(reconstructed, pair)
        if set(arrays) != set(expected) or any(not np.array_equal(arrays[k], v) for k, v in expected.items()):
            raise ValueError('Bank rows detached from registered H1 source data')
    for entry in seal['entries']:
        if entry['teacher'] is None:
            expected = native_copy(entry['outer'], frames)
        else:
            pair = model.exclusions(entry['outer'], entry['teacher'])
            expected = model.teachers(banks[pair], laws[entry['teacher'], entry['split']], entry['split'])
        with np.load(entry['path'], allow_pickle=False) as z:
            if set(z.files) != set(expected) or any(not np.array_equal(z[k], v) for k, v in expected.items()):
                raise ValueError('Prediction values, sampler IDs or exact parent-native identity changed')
    if {p.resolve() for p in (OUT/'predictions').iterdir()} != {Path(e['path']).resolve() for e in seal['entries']}:
        raise ValueError('Unexpected prediction-directory files')
    if {p.resolve() for p in (OUT/'banks').iterdir()} != {Path(e['path']).resolve() for e in seal['banks']}:
        raise ValueError('Unexpected bank-directory files')


def verify_completed() -> None:
    check();seal = read(OUT/'prediction_seal.json');done = read(OUT/'completed.json')
    if (done['freeze_sha256'] != sha(OUT/'freeze.json') or seal['freeze_sha256'] != done['freeze_sha256']
                or done['prediction_seal_sha256'] != sha(OUT/'prediction_seal.json')
                or done['version'] != 20 or done['predictions'] != 91 or done['banks'] != 21
                or any(done[k] is not False for k in ('source_H2_fine_labels_read', 'source_2025_fine_labels_read',
                                                    'target_values_opened', 'new_model_fit', 'scores_computed'))):
        raise ValueError('Completion binding or information boundary changed')
    check_predictions(seal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'predict', 'verify'), required=True)
    {'prepare': prepare, 'predict': predict, 'verify': verify_completed}[parser.parse_args().stage]()
