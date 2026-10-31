"""Verification-only recovery of V20's failed NPZ-layout replay; never generate."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from scripts import run_native_hindcast_v20 as prior

model = prior.model
sha, read, write, now = prior.sha, prior.read, prior.write, prior.now
OUT = ROOT / 'results/native-hindcast-v20r1'
FAILED = prior.OUT
CODE = ('scripts/verify_native_hindcast_v20r1.py', 'tests/test_native_hindcast_v20r1.py',
        'docs/NATIVE_HINDCAST_V20R1_PROTOCOL.md')
REVIEWS = ('code-review.md', 'python-review.md', 'design-review.md')
FAILED_PINS = {
    'freeze.json': '23e504b1e80310c1f5a6f2b95a5cdca37d48b427a0bf69a4dc2b0893d196b68c',
    'predict.log': 'ef8dec1d14814095bee5d955a90c8393f5110282e038aacd5bd7f428abb5e564',
    'diagnose_attempt1.py': 'ecdd5072255567b701ef486a7752e5b29a80d3861296ebf7ebdfc412d8d06d2e',
    'diagnosis-attempt1.json': '826f11ed4c81372c8ad73e9063f6bcb6fa56cbcd92f501f543dab992aca0e15a',
    'diagnosis-attempt1.md': '5b46f1efb59d3242f2e2d97ee9052d56ee0b888f468af4e37efd0d70c10d9bf2',
}


def contract() -> dict[str, Any]:
    return dict(version='20r1', producer_version=20, prediction_count=91, bank_count=21,
        producer_attempt_status='failed_exit_1', generation_attempt_success=False,
        recovery='verification_only_fresh_bank_layout', new_predictions_generated=False,
        source_H2_fine_labels_read=False, source_2025_fine_labels_read=False,
        target_values_opened=False, unseen_values_opened=False, new_model_fit=False,
        scores_computed=False, numerical_audit_status='PENDING_INDEPENDENT_AUDIT')


def aliases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Register old paths explicitly; do not copy or rename any NPZ."""
    entries, banks = [], []
    for pair in itertools.combinations(model.SOURCES, 2):
        banks.append(dict(excluded=list(pair), path=str(FAILED / 'banks' / prior.bank_name(pair))))
    for outer in model.SOURCES:
        entries.append(dict(outer=outer, teacher=None, split='query_copy', kind='query_copy',
            path=str(FAILED / 'predictions' / f'query-{outer}-native.npz'),
            inherited_path=str(prior.PARENT / 'predictions' / f'source-{outer}-native.npz')))
        for teacher in model.SOURCES:
            if teacher == outer:
                continue
            pair = model.exclusions(outer, teacher)
            for split in ('H1_retrospective', 'H2_hindcast'):
                entries.append(dict(outer=outer, teacher=teacher, split=split, kind='teacher',
                    path=str(FAILED / 'predictions' / f'{split}-outer-{outer}-teacher-{teacher}.npz'),
                    excluded=list(pair), bank_path=str(FAILED / 'banks' / prior.bank_name(pair)),
                    chronological_hindcast=split == 'H2_hindcast',
                    within_proper_future_donors=split == 'H1_retrospective'))
    return entries, banks


def failed_parent() -> dict[str, Any]:
    """Accept only this explicitly pinned failed recovery parent, not arbitrary partials."""
    if any((FAILED / name).exists() for name in ('completed.json', 'prediction_seal.json')):
        raise ValueError('Recovery requires the pinned failed, uncompleted V20 attempt')
    prior.verify({str(FAILED / name): digest for name, digest in FAILED_PINS.items()})
    frozen = prior.check()
    entries, banks = aliases()
    for folder, expected in (('predictions', entries), ('banks', banks)):
        if {p.resolve() for p in (FAILED / folder).iterdir()} != {Path(e['path']).resolve() for e in expected}:
            raise ValueError('Failed parent must have the exact91/21 complete archive inventory')
    diagnosis = read(FAILED / 'diagnosis-attempt1.json')
    if (diagnosis['status'] != 'DIAGNOSTIC_ONLY_FAILED_ATTEMPT'
            or diagnosis['first_failure_reproduced'] is not True
            or diagnosis['all_112_npz_hashes_unchanged'] is not True
            or diagnosis['freeze_sha256'] != FAILED_PINS['freeze.json']
            or diagnosis['log_sha256'] != FAILED_PINS['predict.log']):
        raise ValueError('Pinned diagnosed failure required')
    return frozen


def input_paths(parent: dict[str, Any]) -> set[str]:
    entries, banks = aliases()
    return (set(parent['input_hashes']) | {str(FAILED / name) for name in FAILED_PINS}
            | {e['path'] for e in entries + banks} | {str(OUT / name) for name in REVIEWS})


def check_review_files() -> None:
    # Independent reviewers own these reports; their exact bytes enter the freeze.
    for name in REVIEWS:
        path = OUT / name
        if not path.is_file() or not path.read_text().strip():
            raise ValueError('All three independent prelaunch review reports required')


def prepare() -> None:
    if any((OUT / name).exists() for name in ('freeze.json', 'verification_seal.json', 'completed.json')):
        raise FileExistsError('Preserve complete or partial R1 attempt')
    parent = failed_parent();check_review_files()
    inputs = {path: sha(path) for path in input_paths(parent)}
    code = {**parent['code_hashes'], **{str(ROOT / p): sha(ROOT / p) for p in CODE}}
    if len(code) != 65 or len(inputs) != 141:
        raise ValueError('Exact65 code and141 input closure required')
    prior.verify(inputs);prior.verify(code)
    prior.verify(parent['input_hashes'])
    environment = prior.parent_api.prior.environment()
    frozen = {**contract(), 'created_utc': now(), 'code_hashes': code, 'input_hashes': inputs,
        'source': parent['source'], 'failed_freeze_sha256': sha(FAILED / 'freeze.json'),
        'entries': aliases()[0], 'banks': aliases()[1], 'environment': environment,
        'environment_sha256': hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'freeze.json', frozen)


def check() -> dict[str, Any]:
    frozen = read(OUT / 'freeze.json')
    if any(frozen[k] != value for k, value in contract().items()):
        raise ValueError('Changed verification-only recovery contract')
    parent = failed_parent();check_review_files()
    expected_code = {**parent['code_hashes'], **{str(ROOT / p): sha(ROOT / p) for p in CODE}}
    entries, banks = aliases()
    if (len(frozen['code_hashes']) != 65 or len(frozen['input_hashes']) != 141
            or frozen['code_hashes'] != expected_code or set(frozen['input_hashes']) != input_paths(parent)
            or frozen['entries'] != entries or frozen['banks'] != banks
            or frozen['source'] != parent['source'] or frozen['failed_freeze_sha256'] != FAILED_PINS['freeze.json']):
        raise ValueError('Changed exact closure or archive alias registry')
    if any(frozen['input_hashes'][p] != digest for p, digest in parent['input_hashes'].items()):
        raise ValueError('Inherited input digests differ from failed parent')
    environment = prior.parent_api.prior.environment()
    if (environment != frozen['environment'] or frozen['environment_sha256'] !=
            hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()):
        raise ValueError('Changed execution environment')
    prior.verify(frozen['code_hashes']);prior.verify(frozen['input_hashes'])
    return frozen


def exact_fields(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], label: str) -> None:
    if set(actual) != set(expected):
        raise ValueError(f'{label}: field registry differs')
    for key, value in expected.items():
        if actual[key].dtype != value.dtype or not np.array_equal(actual[key], value):
            raise ValueError(f'{label}: exact field mismatch: {key}')


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def verified_fresh_bank(arrays: dict[str, np.ndarray], fresh: Any, pair: tuple[str, str]) -> Any:
    """Check stored values/provenance strictly; replay with original fitted layout."""
    model.restore_bank(arrays, pair)  # Keep all original normalization/support/schema guards.
    exact_fields(arrays, model.bank_arrays(fresh, pair), 'bank ' + '-'.join(pair))
    return fresh


def verify_values(frozen: dict[str, Any]) -> dict[str, Any]:
    """Only H1 fine inputs and legal source coarse prefixes are numerically parsed."""
    frames = {r: model.proper_frame(Path(frozen['source']) / f'{r}_hourly.csv', r) for r in model.SOURCES}
    monthly = model.monthly_prefix(Path(frozen['source']) / 'monthly.csv')
    laws = {(r, split): model.coarse_arrays(monthly, r, split) for r in model.SOURCES
            for split in ('H1_retrospective', 'H2_hindcast')}
    banks, layouts = {}, []
    for entry in frozen['banks']:
        pair = tuple(entry['excluded'])
        fresh = model.native._bank(frames, tuple(r for r in model.SOURCES if r not in pair))
        arrays = load_arrays(entry['path'])
        banks[pair] = verified_fresh_bank(arrays, fresh, pair)
        layouts.append(dict(excluded=list(pair), saved_q_strides=list(arrays['q'].strides),
                            fresh_q_strides=list(fresh.q_.strides), values_exact=True))
    counts = dict(query_copy=0, H1_retrospective=0, H2_hindcast=0)
    for entry in frozen['entries']:
        if entry['teacher'] is None:
            expected = prior.native_copy(entry['outer'], frames)
        else:
            pair = model.exclusions(entry['outer'], entry['teacher'])
            expected = model.teachers(banks[pair], laws[entry['teacher'], entry['split']], entry['split'])
        exact_fields(load_arrays(entry['path']), expected, entry['path'])
        counts[entry['split']] += 1
    if counts != dict(query_copy=7, H1_retrospective=42, H2_hindcast=42) or len(banks) != 21:
        raise ValueError('Incomplete exact array verification grid')
    return dict(exact_counts=counts, exact_bank_count=len(banks), bank_layouts=layouts,
                comparison='exact dtype and np.array_equal; no tolerance', aliases_only=True)


def run_verification() -> None:
    frozen = check();initial = sha(OUT / 'freeze.json')
    if any((OUT / name).exists() for name in ('verification_seal.json', 'completed.json')):
        raise FileExistsError('Preserve verification attempt outputs')
    result = verify_values(frozen)
    check()
    if sha(OUT / 'freeze.json') != initial:
        raise ValueError('Freeze changed during exact verification')
    seal = {**contract(), **result, 'created_utc': now(), 'freeze_sha256': initial,
        'entries': frozen['entries'], 'banks': frozen['banks'], 'complete_grid': True}
    write(OUT / 'verification_seal.json', seal)
    write(OUT / 'completed.json', {**contract(), 'created_utc': now(), 'freeze_sha256': initial,
        'verification_seal_sha256': sha(OUT / 'verification_seal.json'), 'complete_grid': True})


def verify_completed() -> None:
    frozen = check();initial = sha(OUT / 'freeze.json')
    seal = read(OUT / 'verification_seal.json');done = read(OUT / 'completed.json')
    if (any(obj[k] != value for obj in (seal, done) for k, value in contract().items())
            or any(obj['freeze_sha256'] != sha(OUT / 'freeze.json') for obj in (seal, done))
            or seal['entries'] != frozen['entries'] or seal['banks'] != frozen['banks']
            or any(obj['complete_grid'] is not True for obj in (seal, done))
            or done['verification_seal_sha256'] != sha(OUT / 'verification_seal.json')):
        raise ValueError('Changed R1 completion bindings or failure disclosure')
    result = verify_values(frozen)
    if any(seal[key] != value for key, value in result.items()):
        raise ValueError('Changed exact verification result')
    check()
    if sha(OUT / 'freeze.json') != initial:
        raise ValueError('Freeze changed during completed verification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'check'))
    {'prepare': prepare, 'verify': run_verification, 'check': verify_completed}[parser.parse_args().stage]()
