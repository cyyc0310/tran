"""Frozen historical-source mean identification diagnostic; no forecast outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from risk_information_witness_v19 import EF, FUELS, mean_outer_bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/mean-information-v19'
PARENT = ROOT/'results/native-center-v18'
SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
CODE = ('scripts/risk_information_witness_v19.py', 'tests/test_risk_information_witness_v19.py',
        'scripts/run_mean_information_v19.py', 'tests/test_mean_information_v19.py',
        'docs/MEAN_INFORMATION_V19_PROTOCOL.md')
B = 1000.


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict:
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                packages={k: importlib.metadata.version(k) for k in ('numpy', 'pandas', 'scipy')})


def prepare() -> None:
    if (OUT/'freeze.json').exists():
        raise FileExistsError('Preserve existing freeze')
    parent = json.loads((PARENT/'freeze.json').read_text())
    completed = json.loads((PARENT/'source_completed.json').read_text())
    if (completed['freeze_sha256'] != sha(PARENT/'freeze.json')
            or completed['target_values_opened'] is not False or completed['unseen_values_opened'] is not False):
        raise ValueError('Completed source-only V18 required')
    inputs = {str(PARENT/n): sha(PARENT/n) for n in ('freeze.json', 'source_completed.json')}
    for region in SOURCES:
        path = Path(parent['source'])/f'{region}_hourly.csv'
        digest = sha(path)
        if digest != parent['input_hashes'][str(path)]:
            raise ValueError('Source file changed since V18')
        inputs[str(path)] = digest
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'freeze.json', dict(created_utc=now(), source=parent['source'], sources=list(SOURCES),
          start='2024-07-01', end='2025-01-01', source_only=True, contemporaneous_oracle=True,
          budgets=['D2', 'T8'], support=B, ef=EF.tolist(), environment=environment(),
          code_hashes={str(ROOT/p): sha(ROOT/p) for p in CODE}, input_hashes=inputs,
          forecast_gain=False, mechanism_breakthrough=False, target_authorized=False))


def check() -> dict:
    frozen = json.loads((OUT/'freeze.json').read_text())
    if (frozen['sources'] != list(SOURCES) or frozen['start'] != '2024-07-01'
            or frozen['end'] != '2025-01-01' or frozen['support'] != B or frozen['ef'] != EF.tolist()
            or frozen['budgets'] != ['D2', 'T8'] or frozen['environment'] != environment()
            or frozen['source_only'] is not True or frozen['contemporaneous_oracle'] is not True
            or any(frozen[k] is not False for k in ('forecast_gain', 'mechanism_breakthrough', 'target_authorized'))
            or set(frozen['code_hashes']) != {str(ROOT/p) for p in CODE}):
        raise ValueError('Frozen identification contract changed')
    expected_inputs = {str(PARENT/n) for n in ('freeze.json', 'source_completed.json')}
    expected_inputs |= {str(Path(frozen['source'])/f'{r}_hourly.csv') for r in SOURCES}
    if set(frozen['input_hashes']) != expected_inputs:
        raise ValueError('Exact nine-input closure required')
    for path, digest in {**frozen['code_hashes'], **frozen['input_hashes']}.items():
        if sha(Path(path)) != digest:
            raise ValueError('Changed frozen bytes '+path)
    parent = json.loads((PARENT/'freeze.json').read_text())
    completed = json.loads((PARENT/'source_completed.json').read_text())
    if (frozen['source'] != parent['source'] or completed['freeze_sha256'] != sha(PARENT/'freeze.json')
            or completed['target_values_opened'] is not False or completed['unseen_values_opened'] is not False):
        raise ValueError('V18 parent source identity or source-only binding changed')
    for region in SOURCES:
        path = str(Path(frozen['source'])/f'{region}_hourly.csv')
        if frozen['input_hashes'][path] != parent['input_hashes'][path]:
            raise ValueError('Source digest no longer matches V18 parent')
    return frozen


def full_g_mean_bounds(generation: np.ndarray, mu_energy: float) -> tuple[float, float]:
    """Sharp ONLY for fixed positive G, one weighted mean and C in[0,1000]."""
    g = np.asarray(generation, float)
    if g.ndim != 1 or not len(g) or not np.isfinite(g).all() or (g <= 0).any() or not 0 <= mu_energy <= B:
        raise ValueError('Positive finite one-dimensional G and supported mean required')
    if mu_energy == 0 or mu_energy == B:
        return float(mu_energy), float(mu_energy)
    weights = g/g.sum()
    def endpoint(order: np.ndarray) -> float:
        remaining = mu_energy/B
        total = 0.
        for i in order:
            amount = min(1., max(0., remaining/weights[i]))
            total += amount
            remaining -= amount*weights[i]
        if abs(remaining) > 1e-12:
            raise ValueError('Unallocated carbon budget')
        return total*B/len(g)
    order = np.argsort(weights, kind='stable')
    return endpoint(order[::-1]), endpoint(order)


def block_bounds(g: np.ndarray, carbon: np.ndarray) -> dict:
    g, c = np.asarray(g, float), np.asarray(carbon, float)
    if (g.ndim != 1 or c.shape != g.shape or not len(g) or not np.isfinite(g).all()
            or not np.isfinite(c).all() or (g <= 0).any() or (c < 0).any() or (c > B).any()):
        raise ValueError('Matched positive G and supported C required')
    raw_mu = float(np.average(c, weights=g))
    if not -1e-9 <= raw_mu <= B+1e-9:
        raise ValueError('Weighted mean outside floating support guard')
    mu = float(np.clip(raw_mu, 0, B))
    cv = float(np.std(g/np.mean(g)))
    # A rounded weighted mean at0/B is not an exact endpoint certificate.
    # Positive sums/averages on supported C get a disclosed arithmetic margin;
    # only a verified all-zero/all-B raw path certifies that endpoint exactly.
    certified_endpoint = bool(np.all(c == 0) or np.all(c == B))
    if certified_endpoint:
        mu = 0. if np.all(c == 0) else B
    margin = 0. if certified_endpoint else B*(128*np.finfo(float).eps)
    mu_lo, mu_hi = max(0., mu-margin), min(B, mu+margin)
    finite = (mu_lo/len(g), (mu_hi+(len(g)-1)*B)/len(g))
    if certified_endpoint:
        finite = (mu, mu)
    radius = B*(cv/2)
    simple = (max(finite[0], mu_lo-radius), min(finite[1], mu_hi+radius))
    quad = (mean_outer_bounds(mu_lo, cv)[0], mean_outer_bounds(mu_hi, cv)[1])
    quadratic = (max(finite[0], quad[0]), min(finite[1], quad[1]))
    full = (full_g_mean_bounds(g, mu_lo)[0], full_g_mean_bounds(g, mu_hi)[1])
    intervals = dict(finite=finite, simple_cv=simple, quadratic_cv=quadratic, full_g=full)
    actual = float(c.mean())
    for lo, hi in intervals.values():
        if not lo-1e-8 <= actual <= hi+1e-8 or lo > hi+1e-8:
            raise ValueError('Identification bound failed to contain actual equal-hour mean')
    for outer, inner in zip(intervals.values(), list(intervals.values())[1:]):
        if inner[0] < outer[0]-1e-8 or inner[1] > outer[1]+1e-8 or inner[0] > inner[1]+1e-8:
            raise ValueError('Nested information sets violated')
    return dict(mu_energy=mu, mu_equal=actual, cv_generation=cv, weighted_minus_equal=mu-actual,
                mu_energy_lo=mu_lo, mu_energy_hi=mu_hi, aggregation_arithmetic_margin=margin,
                rounded_endpoint_without_certificate_count=int(mu in (0, B) and not certified_endpoint),
                weighted_mean_float_guard_count=int(mu != raw_mu),
                **{f'{name}_{side}': float(values[i]) for name, values in intervals.items()
                   for i, side in enumerate(('lo', 'hi'))})


def source_days(path: Path) -> list[tuple[pd.Timestamp, np.ndarray]]:
    if path.stem.removesuffix('_hourly') not in SOURCES:
        raise ValueError('Only seven registered sources')
    with path.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if '2024-07' <= r['time'] < '2025-01']
    table = pd.DataFrame(rows)
    table['time'] = pd.to_datetime(table.time, utc=True)
    if (table.time.duplicated().any() or not table.time.eq(table.time.dt.floor('h')).all()
            or not table.region.eq(path.stem.removesuffix('_hourly')).all() or not set(table.valid) <= {'True', 'False'}):
        raise ValueError('Source timestamp/identity/validity schema')
    result = []
    for day, part in table.groupby(table.time.dt.floor('D')):
        if len(part) != 24 or not part.valid.eq('True').all():
            continue
        part = part.sort_values('time')
        if not np.array_equal(pd.DatetimeIndex(part.time).asi8, pd.date_range(day, periods=24, freq='h').asi8):
            raise ValueError('UTC source day not complete')
        a = part[['generation_'+f for f in FUELS]].to_numpy(float)
        if not np.isfinite(a).all() or (a < 0).any() or (a.sum(1) <= 0).any():
            raise ValueError('Invalid eight-fuel source generation')
        result.append((day, a))
    if not result:
        raise ValueError('Registered source has no valid day')
    return result


def run() -> None:
    if any((OUT/p).exists() for p in ('day_bounds.csv', 'region_summary.csv', 'macro_summary.csv', 'completed.json')):
        raise FileExistsError('Preserve partial or completed attempt')
    frozen = check()
    freeze_hash = sha(OUT/'freeze.json')
    records = []
    for region in SOURCES:
        for day, a in source_days(Path(frozen['source'])/f'{region}_hourly.csv'):
            g = a.sum(1)
            raw_c = (a/g[:, None])@EF
            if (raw_c < -1e-9).any() or (raw_c > B+1e-9).any():
                raise ValueError('CIF support exceeds floating guard')
            c = np.clip(raw_c, 0, B)
            for packet, size in (('D2', 24), ('T8', 6)):
                blocks = [block_bounds(g[i:i+size], c[i:i+size]) for i in range(0, 24, size)]
                merged = {k: float(np.mean([b[k] for b in blocks])) for k in blocks[0]}
                # Diagnostic weighted mean for T8 is the equal-block average, not the daily energy mean.
                for key in ('weighted_mean_float_guard_count', 'rounded_endpoint_without_certificate_count'):
                    merged[key] = sum(b[key] for b in blocks)
                for name in ('finite', 'simple_cv', 'quadratic_cv', 'full_g'):
                    merged[name+'_width'] = merged[name+'_hi']-merged[name+'_lo']
                records.append(dict(region=region, day=day.isoformat(), packet=packet,
                                    hourly_support_float_guard_count=int(np.count_nonzero(c != raw_c)), **merged))
        print('Identified', region, flush=True)
    table = pd.DataFrame(records)
    if table.duplicated(['region', 'day', 'packet']).any():
        raise ValueError('Duplicated identification records')
    left = table[table.packet.eq('D2')][['region', 'day']].reset_index(drop=True)
    right = table[table.packet.eq('T8')][['region', 'day']].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError('Changed source-day mask across budgets')
    values = [k for k in table.select_dtypes(include='number').columns if not k.endswith('_count')]
    regional = table.groupby(['region', 'packet'])[values].mean().reset_index()
    counts = table.groupby(['region', 'packet']).size().rename('valid_days').reset_index()
    regional = regional.merge(counts, on=['region', 'packet'], validate='one_to_one')
    macro = regional.groupby('packet')[values].mean().reset_index()
    outputs = {}
    for name, data in (('day_bounds', table), ('region_summary', regional), ('macro_summary', macro)):
        path = OUT/f'{name}.csv'
        data.to_csv(path, index=False, mode='x')
        outputs[str(path)] = sha(path)
    check()
    if sha(OUT/'freeze.json') != freeze_hash:
        raise ValueError('Freeze changed during diagnostic')
    write(OUT/'completed.json', dict(created_utc=now(), freeze_sha256=freeze_hash, output_hashes=outputs,
          region_days=len(left), rows=len(table), target_values_opened=False, contemporaneous_oracle=True,
          forecast_gain=False, mechanism_breakthrough=False, target_authorized=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'run'))
    {'prepare': prepare, 'run': run}[parser.parse_args().stage]()
