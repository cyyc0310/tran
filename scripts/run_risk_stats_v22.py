"""Post-outcome source exploration of exact HG/support tail information."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from fractions import Fraction as F
from scripts import risk_stats_v22 as model
from scripts import run_tail_count_v20 as prior

OUT = ROOT / 'results/risk-stats-v22'
PARENT = prior.OUT
MANIFEST = ROOT / 'docs/evidence/tail-count-v20-verification.json'
SOURCES, THRESHOLDS, EF = prior.SOURCES, prior.THRESHOLDS, prior.EF
BUDGETS = ('T8', 'T8_G2', 'T8_Gmin', 'T8_G24')
OUTPUTS = ('day_counts.csv', 'block_counts.csv', 'region_summary.csv', 'macro_summary.csv')
CODE = ('scripts/risk_stats_v22.py', 'scripts/run_risk_stats_v22.py',
        'tests/test_risk_stats_v22.py', 'docs/RISK_STATS_V22_PROTOCOL.md')
REVIEWS = ('code-review.md', 'python-review.md', 'design-review.md')
sha, read, write, now = prior.sha, prior.read, prior.write, prior.now


def contract() -> dict[str, Any]:
    return dict(version=22, sources=list(SOURCES), thresholds=list(THRESHOLDS), budgets=list(BUDGETS),
        support=1000, ef=list(EF), start='2024-07-01', end='2025-01-01', region_days=1000,
        day_rows=16000, block_rows=64000, region_rows=112, macro_rows=16,
        source_only=True, contemporaneous_oracle=True, decimal_fraction_semantics=True,
        post_outcome_source_exploration=True, independent_confirmation=False,
        forecast_gain=False, new_method=False, mechanism_breakthrough=False, target_authorized=False,
        target_values_opened=False, unseen_values_opened=False, source_2025_fine_labels_read=False)


def checked_parent(parent_sha256: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Check only hashes/metadata; no source numeric parsing before this gate."""
    if not isinstance(parent_sha256, str) or not re.fullmatch('[0-9a-f]{64}', parent_sha256):
        raise ValueError('Explicit64 lowercase-hex parent manifest SHA256 required')
    if not MANIFEST.is_file() or sha(MANIFEST) != parent_sha256:
        raise ValueError('Completed sealed Tail parent manifest required at approved digest')
    evidence = read(MANIFEST)
    for mapping in ('frozen_code_hashes', 'frozen_input_hashes', 'artifact_hashes'):
        for name, digest in evidence[mapping].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha(path) != digest:
                raise ValueError('Changed Tail manifest closure: ' + name)
    # Includes the fixed V19 evidence and its complete code/input/artifact closure.
    parent = prior.check()
    completed = read(PARENT / 'completed.json')
    expected_outputs = {str(PARENT / name): sha(PARENT / name) for name in OUTPUTS}
    if (completed['freeze_sha256'] != sha(PARENT / 'freeze.json')
            or completed['output_hashes'] != expected_outputs
            or completed['region_days'] != 1000 or completed['day_rows'] != 12000
            or completed['block_rows'] != 48000
            or any(completed[k] is not True for k in ('source_only', 'contemporaneous_oracle',
                                                      'decimal_fraction_semantics'))
            or any(completed[k] is not False for k in ('forecast_gain', 'mechanism_breakthrough',
                'target_authorized', 'target_values_opened', 'unseen_values_opened'))):
        raise ValueError('Complete source-only1000day Tail parent with four bound outputs required')
    for path in [PARENT / 'freeze.json', PARENT / 'completed.json', *(PARENT / name for name in OUTPUTS)]:
        if evidence['artifact_hashes'].get(str(path.relative_to(ROOT))) != sha(path):
            raise ValueError('Tail metadata/output is not bound by its final manifest')
    paths = [MANIFEST, PARENT / 'freeze.json', PARENT / 'completed.json',
             *(PARENT / name for name in OUTPUTS), Path(sys.executable).resolve(),
             *(Path(parent['source']) / f'{r}_hourly.csv' for r in SOURCES),
             *(OUT / name for name in REVIEWS)]
    for name in REVIEWS:
        if not (OUT / name).is_file() or not (OUT / name).read_text().strip():
            raise ValueError('Three independent prelaunch review reports required')
    inputs = {str(p): sha(p) for p in paths}
    for region in SOURCES:
        path = str(Path(parent['source']) / f'{region}_hourly.csv')
        if inputs[path] != parent['input_hashes'][path]:
            raise ValueError('Source digest differs from completed Tail parent')
    if len(inputs) != 18:
        raise ValueError('Exact18 direct input closure required')
    return parent, inputs


def prepare(parent_sha256: str) -> None:
    if any((OUT / name).exists() for name in ('freeze.json', 'completed.json', *OUTPUTS)):
        raise FileExistsError('Preserve complete or partial V22 attempt')
    parent, inputs = checked_parent(parent_sha256)
    code = {**parent['code_hashes'], **{str(ROOT / name): sha(ROOT / name) for name in CODE}}
    if len(code) != 8:
        raise ValueError('Exact8 direct producer/test/protocol code closure required')
    environment = prior.environment()
    frozen = {**contract(), 'created_utc': now(), 'source': parent['source'],
        'parent_manifest_sha256': parent_sha256, 'environment': environment,
        'code_hashes': code, 'input_hashes': inputs}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'freeze.json', frozen)


def check() -> dict[str, Any]:
    frozen = read(OUT / 'freeze.json')
    if any(frozen[key] != value for key, value in contract().items()):
        raise ValueError('Changed source-exploration information contract')
    parent, inputs = checked_parent(frozen['parent_manifest_sha256'])
    expected_code = {**parent['code_hashes'], **{str(ROOT / name): sha(ROOT / name) for name in CODE}}
    if (frozen['source'] != parent['source'] or frozen['input_hashes'] != inputs
            or frozen['code_hashes'] != expected_code or len(expected_code) != 8
            or frozen['environment'] != prior.environment()):
        raise ValueError('Changed source registry, code/input closure or environment')
    for path, digest in frozen['code_hashes'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Changed frozen V22 producer bytes')
    return frozen


def day_rows(region: str, day: str, fuel: list[list[F]]) -> tuple[list[dict], list[dict]]:
    if region not in SOURCES or len(fuel) != 24 or any(len(hour) != 8 for hour in fuel):
        raise ValueError('Registered source and complete24x8 exact fuel day required')
    if any(any(not isinstance(v, F) or v < 0 for v in hour) or sum(hour) <= 0 for hour in fuel):
        raise ValueError('Exact nonnegative fuels and strictly positive hourly G required')
    g = [sum(hour, F(0)) for hour in fuel]
    h = [sum((value * ef for value, ef in zip(hour, EF)), F(0)) for hour in fuel]
    daily, blocks = [], []
    for tau in THRESHOLDS:
        sets = {budget: [] for budget in BUDGETS}
        true_day = sum(energy > tau * power for energy, power in zip(h, g))
        for block, start in enumerate(range(0, 24, 6)):
            gs, hs = g[start:start + 6], h[start:start + 6]
            total, energy = sum(gs, F(0)), sum(hs, F(0))
            candidates = dict(T8=model.coarse_counts(6, total, energy, F(tau)),
                T8_G2=model.moment_counts(6, total, sum(v * v for v in gs), energy, F(tau)),
                T8_Gmin=model.minimum_g_counts(6, total, min(gs), energy, F(tau)),
                T8_G24=model.full_g_counts(gs, energy, F(tau)))
            if (not set(candidates['T8_G24']) <= set(candidates['T8_Gmin']) <= set(candidates['T8'])
                    or not set(candidates['T8_G24']) <= set(candidates['T8_G2']) <= set(candidates['T8'])
                    or (candidates['T8_Gmin'] == (0,)) != (candidates['T8_G24'] == (0,))):
                raise ValueError('HG/support information inclusions or exact K0 identity failed')
            actual = sum(e > tau * v for e, v in zip(hs, gs))
            for budget, counts in candidates.items():
                sets[budget].append(counts)
                blocks.append(dict(region=region, day=day, threshold=tau, budget=budget, block=block,
                    **model.describe(counts, actual)))
        combined = {budget: model.combine_blocks(parts) for budget, parts in sets.items()}
        if ((combined['T8_Gmin'] == (0,)) != (combined['T8_G24'] == (0,))
                or not set(combined['T8_G24']) <= set(combined['T8_Gmin']) <= set(combined['T8'])
                or not set(combined['T8_G24']) <= set(combined['T8_G2']) <= set(combined['T8'])):
            raise ValueError('Daily inclusions or exact K0 identity failed')
        for budget, counts in combined.items():
            daily.append(dict(region=region, day=day, threshold=tau, budget=budget,
                              **model.describe(counts, true_day)))
    return daily, blocks


def inherited_mask() -> set[tuple[str, str]]:
    table = pd.read_csv(PARENT / 'day_counts.csv', dtype={'counts': str})
    keys = ['region', 'day', 'threshold', 'budget']
    actual = set(table[keys].itertuples(index=False, name=None))
    mask = set(table[['region', 'day']].itertuples(index=False, name=None))
    if (len(table) != 12000 or len(mask) != 1000 or table[keys].duplicated().any()
            or {r for r, _ in mask} != set(SOURCES)
            or actual != {(r, d, t, b) for r, d in mask for t in THRESHOLDS for b in prior.BUDGETS}):
        raise ValueError('Exact inherited1000day/12000row parent grid required')
    return mask


def match_parent_curves(daily: pd.DataFrame, blocks: pd.DataFrame) -> None:
    """All three inherited budgets must be identical to the sealed parent curves."""
    for name, current, keys in (('day_counts.csv', daily, ['region', 'day', 'threshold', 'budget']),
        ('block_counts.csv', blocks, ['region', 'day', 'threshold', 'budget', 'block'])):
        parent = pd.read_csv(PARENT / name, dtype={'counts': str})
        child = current[current.budget.isin(prior.BUDGETS)]
        if set(parent.columns) != set(child.columns):
            raise ValueError('Inherited budget column registry changed')
        parent = parent.sort_values(keys).reset_index(drop=True)
        child = child[parent.columns].sort_values(keys).reset_index(drop=True)
        if not parent.equals(child):
            raise ValueError('Inherited exact parent count curves changed')


def run() -> None:
    if any((OUT / name).exists() for name in ('completed.json', *OUTPUTS)):
        raise FileExistsError('Preserve partial or completed V22 outputs')
    frozen = check();initial = sha(OUT / 'freeze.json')
    mask = inherited_mask()
    daily, blocks, observed = [], [], set()
    for region in SOURCES:
        for day, fuel in prior.source_days(Path(frozen['source']) / f'{region}_hourly.csv'):
            key = (region, day.isoformat())
            if key not in mask:
                raise ValueError('Source day outside inherited cohort')
            if key in observed:
                raise ValueError('Duplicate source day')
            observed.add(key)
            day_part, block_part = day_rows(region, key[1], fuel)
            daily.extend(day_part);blocks.extend(block_part)
        print('Counted', region, flush=True)
    if observed != mask:
        raise ValueError('Exact inherited1000day source mask required')
    day_table, block_table = pd.DataFrame(daily), pd.DataFrame(blocks)
    if len(day_table) != 16000 or len(block_table) != 64000:
        raise ValueError('Changed registered day/block grid')
    match_parent_curves(day_table, block_table)
    keys = ['region', 'threshold', 'budget']
    regional = day_table.groupby(keys)[list(prior.SUMMARY)].mean().reset_index()
    counts = day_table.groupby(keys).size().rename('valid_days').reset_index()
    regional = regional.merge(counts, on=keys, validate='one_to_one')
    macro = regional.groupby(['threshold', 'budget'])[list(prior.SUMMARY)].mean().reset_index()
    if len(regional) != 112 or len(macro) != 16:
        raise ValueError('Changed registered summary grid')
    outputs = {}
    for name, table in zip(OUTPUTS, (day_table, block_table, regional, macro)):
        path = OUT / name;table.to_csv(path, index=False, mode='x');outputs[str(path)] = sha(path)
    check()
    if sha(OUT / 'freeze.json') != initial:
        raise ValueError('Freeze changed during exact source diagnostic')
    write(OUT / 'completed.json', {**contract(), 'created_utc': now(), 'freeze_sha256': initial,
                                 'output_hashes': outputs})


def verify_completed() -> None:
    check();done = read(OUT / 'completed.json')
    if (any(done[k] != v for k, v in contract().items())
            or done['freeze_sha256'] != sha(OUT / 'freeze.json')
            or set(done['output_hashes']) != {str(OUT / name) for name in OUTPUTS}):
        raise ValueError('Changed V22 completion binding')
    for path, digest in done['output_hashes'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Changed completed V22 output')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run', 'check'), required=True)
    parser.add_argument('--parent-sha256', help='Root-approved completed Tail manifest SHA256; prepare only')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.parent_sha256)
    elif args.parent_sha256 is not None:
        parser.error('--parent-sha256 is locked in freeze and may only be supplied to prepare')
    else:
        {'run': run, 'check': verify_completed}[args.stage]()
