"""Source-only exact historical tail-count information diagnostic."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import tail_count_v20 as model

OUT = ROOT/'results/tail-count-v20'
PARENT = ROOT/'results/mean-information-v19'
MANIFEST = ROOT/'docs/evidence/risk-information-v19-verification.json'
MANIFEST_SHA = 'e2d30122f3daa245740efe902164a6e1367ec1fe906ebfdc589ff1944cc820fc'
SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
EF = (1000, 400, 800, 0, 0, 0, 0, 400)
THRESHOLDS = (100, 300, 600, 900)
BUDGETS = ('T8', 'T8_G2', 'T8_G24')
CODE = ('scripts/tail_count_v20.py', 'scripts/run_tail_count_v20.py',
        'tests/test_tail_count_v20.py', 'docs/TAIL_COUNT_V20_PROTOCOL.md')
SUMMARY = ('span', 'cardinality', 'point_identified', 'no_exceedance_certified',
           'any_exceedance_forced', 'actual_count')


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict:
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                packages={p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy')})


def ancestor_inputs() -> tuple[dict, dict]:
    if sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError('Fixed V19 evidence identity changed')
    evidence, parent, completed = read(MANIFEST), read(PARENT/'freeze.json'), read(PARENT/'completed.json')
    for mapping in ('frozen_code_hashes', 'frozen_input_hashes', 'artifact_hashes'):
        for name, digest in evidence[mapping].items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            if sha(path) != digest:
                raise ValueError('Changed V19 closure '+name)
    if (completed['freeze_sha256'] != sha(PARENT/'freeze.json') or completed['region_days'] != 1000
            or completed['target_values_opened'] is not False or completed['target_authorized'] is not False
            or completed['output_hashes'][str(PARENT/'day_bounds.csv')] != sha(PARENT/'day_bounds.csv')):
        raise ValueError('V19 completed cohort binding changed')
    paths = [MANIFEST, *(PARENT/n for n in ('freeze.json', 'completed.json', 'day_bounds.csv'))]
    paths += [Path(parent['source'])/f'{r}_hourly.csv' for r in SOURCES]
    inputs = {str(p): sha(p) for p in paths}
    for r in SOURCES:
        p = str(Path(parent['source'])/f'{r}_hourly.csv')
        if inputs[p] != parent['input_hashes'][p]:
            raise ValueError('Inherited source digest changed')
    return parent, inputs


def prepare() -> None:
    if (OUT/'freeze.json').exists():
        raise FileExistsError('Preserve existing freeze')
    parent, inputs = ancestor_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'freeze.json', dict(created_utc=now(), source=parent['source'], sources=list(SOURCES),
          thresholds=list(THRESHOLDS), budgets=list(BUDGETS), support=1000, ef=list(EF),
          start='2024-07-01', end='2025-01-01', decimal_fraction_semantics=True,
          source_only=True, contemporaneous_oracle=True, forecast_gain=False,
          mechanism_breakthrough=False, target_authorized=False, environment=environment(),
          code_hashes={str(ROOT/p): sha(ROOT/p) for p in CODE}, input_hashes=inputs))


def check() -> dict:
    frozen = read(OUT/'freeze.json')
    parent, inputs = ancestor_inputs()
    if (frozen['source'] != parent['source'] or frozen['sources'] != list(SOURCES)
            or frozen['thresholds'] != list(THRESHOLDS) or frozen['budgets'] != list(BUDGETS)
            or frozen['support'] != 1000 or frozen['ef'] != list(EF)
            or frozen['start'] != '2024-07-01' or frozen['end'] != '2025-01-01'
            or frozen['environment'] != environment() or frozen['input_hashes'] != inputs
            or frozen['decimal_fraction_semantics'] is not True
            or frozen['source_only'] is not True or frozen['contemporaneous_oracle'] is not True
            or any(frozen[k] is not False for k in ('forecast_gain', 'mechanism_breakthrough', 'target_authorized'))
            or set(frozen['code_hashes']) != {str(ROOT/p) for p in CODE}):
        raise ValueError('Frozen tail-count contract changed')
    for path, digest in frozen['code_hashes'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Changed tail-count producer bytes')
    return frozen


def source_days(path: Path) -> list[tuple[pd.Timestamp, list[list[F]]]]:
    region = path.stem.removesuffix('_hourly')
    if region not in SOURCES:
        raise ValueError('Only seven registered sources')
    with path.open(newline='') as stream:
        records = [row for row in csv.DictReader(stream) if '2024-07' <= row['time'] < '2025-01']
    groups = {}
    seen = set()
    for row in records:
        stamp = pd.Timestamp(row['time'])
        if (stamp.tzinfo is None or stamp != stamp.floor('h') or stamp in seen
                or row['region'] != region or row['valid'] not in ('True', 'False')):
            raise ValueError('Source time/identity/validity schema')
        seen.add(stamp)
        groups.setdefault(stamp.floor('D'), []).append((stamp, row))
    result = []
    for day, part in sorted(groups.items()):
        part.sort(key=lambda x: x[0])
        if len(part) != 24 or any(row['valid'] != 'True' for _, row in part):
            continue
        if [t for t, _ in part] != list(pd.date_range(day, periods=24, freq='h')):
            raise ValueError('Source complete day not24 consecutive UTC hours')
        a = [[F(row['generation_'+fuel]) for fuel in FUELS] for _, row in part]
        if any(any(v < 0 for v in hour) or sum(hour) <= 0 for hour in a):
            raise ValueError('Nonnegative supported source fuel energies required')
        result.append((day, a))
    return result


def run() -> None:
    names = ('day_counts', 'block_counts', 'region_summary', 'macro_summary')
    if any((OUT/n).exists() for n in ('completed.json', *(name+'.csv' for name in names))):
        raise FileExistsError('Preserve partial or completed tail-count attempt')
    frozen = check()
    freeze_digest = sha(OUT/'freeze.json')
    parent_days = pd.read_csv(PARENT/'day_bounds.csv', usecols=['region', 'day', 'packet'])
    mask = set(parent_days[parent_days.packet.eq('T8')][['region', 'day']].itertuples(index=False, name=None))
    daily, blocks, observed = [], [], set()
    for region in SOURCES:
        for day, fuel in source_days(Path(frozen['source'])/f'{region}_hourly.csv'):
            key = (region, day.isoformat())
            if key not in mask:
                raise ValueError('New tail diagnostic source day outside inherited cohort')
            observed.add(key)
            g = [sum(hour, F(0)) for hour in fuel]
            h = [sum((value*ef for value, ef in zip(hour, EF)), F(0)) for hour in fuel]
            for tau in THRESHOLDS:
                sets = {budget: [] for budget in BUDGETS}
                true_day = sum(energy > tau*power for energy, power in zip(h, g))
                for block, start in enumerate(range(0, 24, 6)):
                    gs, hs = g[start:start+6], h[start:start+6]
                    total, energy, square = sum(gs, F(0)), sum(hs, F(0)), sum((v*v for v in gs), F(0))
                    candidates = (model.coarse_counts(6, total, energy, F(tau)),
                                  model.moment_counts(6, total, square, energy, F(tau)),
                                  model.full_g_counts(gs, energy, F(tau)))
                    if not set(candidates[2]) <= set(candidates[1]) <= set(candidates[0]):
                        raise ValueError('Exact per-block information nesting failed')
                    actual = sum(e > tau*v for e, v in zip(hs, gs))
                    for budget, counts in zip(BUDGETS, candidates):
                        sets[budget].append(counts)
                        blocks.append(dict(region=region, day=day.isoformat(), threshold=tau,
                            budget=budget, block=block, **model.describe(counts, actual)))
                combined = {budget: model.combine_blocks(parts) for budget, parts in sets.items()}
                if not set(combined['T8_G24']) <= set(combined['T8_G2']) <= set(combined['T8']):
                    raise ValueError('Exact per-day information nesting failed')
                for budget, counts in combined.items():
                    daily.append(dict(region=region, day=day.isoformat(), threshold=tau,
                                      budget=budget, **model.describe(counts, true_day)))
        print('Counted', region, flush=True)
    if observed != mask or len(observed) != 1000:
        raise ValueError('Exact1000 source-region-day mask required')
    day_table, block_table = pd.DataFrame(daily), pd.DataFrame(blocks)
    if len(day_table) != 12000 or len(block_table) != 48000:
        raise ValueError('Registered day/block grid changed')
    keys = ['region', 'threshold', 'budget']
    regional = day_table.groupby(keys)[list(SUMMARY)].mean().reset_index()
    counts = day_table.groupby(keys).size().rename('valid_days').reset_index()
    regional = regional.merge(counts, on=keys, validate='one_to_one')
    macro = regional.groupby(['threshold', 'budget'])[list(SUMMARY)].mean().reset_index()
    if len(regional) != 84 or len(macro) != 12:
        raise ValueError('Registered summary grid changed')
    outputs = {}
    for name, table in zip(names, (day_table, block_table, regional, macro)):
        path = OUT/f'{name}.csv'
        table.to_csv(path, index=False, mode='x')
        outputs[str(path)] = sha(path)
    check()
    if sha(OUT/'freeze.json') != freeze_digest:
        raise ValueError('Freeze changed during tail-count diagnostic')
    write(OUT/'completed.json', dict(created_utc=now(), freeze_sha256=freeze_digest,
        output_hashes=outputs, region_days=1000, day_rows=12000, block_rows=48000,
        source_only=True, contemporaneous_oracle=True, decimal_fraction_semantics=True,
        forecast_gain=False, mechanism_breakthrough=False, target_authorized=False,
        target_values_opened=False, unseen_values_opened=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run'), required=True)
    {'prepare': prepare, 'run': run}[parser.parse_args().stage]()
