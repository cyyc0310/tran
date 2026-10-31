"""Source-only lagged net-generation shape pilot on the exact V14 native bank.

Standard conditional analog/IPF controls, not a new calibrated model. Signed
net generation is a soft historical predictor, never the proxy denominator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import run_weather_increment_v12 as old
from transcif.research.dynamic_v2 import read_year
from transcif.research.fuel_transport_v7 import recompose_paths

SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
LAGS, WINDOWS, SAMPLES, SEED = (7, 30), (28, 56), 128, 17023
FACTORS = (.75, 1., 1.25, 1.5, 2.)
METHODS = ('native', 'row_only', 'analog_g', 'rake_mean_zero',
           'rake_mean_prior', 'rake_sample_prior', 'mixture_mean_prior')
PRIOR, BANDWIDTH, IPF_TOL, IPF_STEPS = .001, .3, 1e-9, 500
REFERENCE = ROOT / 'results/weather-coarse-v14-round1'
DEFAULT_OUT = ROOT / 'results/generation-context-v16'
OWN = ('scripts/generation_context_v16.py', 'tests/test_generation_context_v16.py',
       'docs/GENERATION_CONTEXT_V16_PROTOCOL.md')


def sha(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write(path: Path, value: dict[str, Any]) -> None:
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify(hashes: dict[str, str]) -> None:
    for path, expected in hashes.items():
        if sha(Path(path)) != expected:
            raise ValueError('Changed frozen bytes: ' + path)


def read_telemetry(path: Path, latest: pd.Timestamp) -> dict[str, pd.Series]:
    """Discard other regions and future rows before numeric G conversion."""
    values: dict[str, list] = {r: [] for r in SOURCES}
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            region = row['region']
            if region not in values:
                continue
            end = pd.Timestamp(row['hour_end'])
            if end.tzinfo is None:
                raise ValueError('Telemetry must use explicit UTC offsets')
            end = end.tz_convert('UTC')
            if end <= latest:
                values[region].append((end, row['g_mw']))
    result = {}
    for region, pairs in values.items():
        index = pd.DatetimeIndex([p[0] for p in pairs], tz='UTC')
        if index.has_duplicates or not index.equals(index.floor('h')):
            raise ValueError('Unique integral telemetry hour ends required')
        result[region] = pd.Series([p[1] for p in pairs], index=index, dtype=object).sort_index()
    return result


def history_shapes(series: pd.Series, origin: pd.Timestamp, lag: int,
                   window: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Positive complete UTC profiles ending by origin minus the assumed lag."""
    if lag not in LAGS or window not in WINDOWS or origin.tzinfo is None:
        raise ValueError('Registered lag/window and timezone-aware origin required')
    cutoff = origin - pd.Timedelta(days=lag)
    profiles, starts = [], []
    for day in pd.date_range(cutoff.normalize()-pd.Timedelta(days=window),
                             cutoff.normalize(), freq='D'):
        if day+pd.Timedelta(days=1) > cutoff:
            continue
        ends = pd.date_range(day+pd.Timedelta(hours=1), periods=24, freq='h')
        raw = series.reindex(ends)
        # Only this origin's legal window is parsed; absent/empty means missing.
        raw = raw.mask(raw.eq(''),np.nan)
        g = pd.to_numeric(raw, errors='raise').to_numpy(float)
        if not np.isfinite(g).all() or np.any(g <= 0):
            continue
        profiles.append(g/g.mean()); starts.append(day)
    same = [i for i, day in enumerate(starts) if day.dayofweek == origin.dayofweek]
    selected = same if same else list(range(len(profiles)))
    pool = np.stack([profiles[i] for i in selected]) if selected else np.empty((0, 24))
    return pool, dict(cutoff=cutoff.isoformat(), complete_positive_days=len(profiles),
        selected_days=len(selected), same_weekday_used=bool(same), missing_fallback=not selected,
        selected_day_starts=[starts[i].isoformat() for i in selected],
        max_selected_hour_end=(max(starts[i] for i in selected)+pd.Timedelta(days=1)).isoformat() if selected else None)


def row_only(a: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Negative control: common fuel scaling per hour cannot change CIF."""
    target = shape * a.sum((1, 2))[:, None]/24
    rows = a.sum(2)
    return a*np.divide(target, rows, out=np.ones_like(rows), where=rows > 0)[..., None]


def generation_cif(a: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Ratio of a physical generation matrix; explicit native zero-row fallback."""
    total = a.sum(-1)
    shares = np.divide(a,total[...,None],out=np.zeros_like(a),where=total[...,None]>0)
    return np.clip(np.where(total>0, shares@old.EF, fallback),old.EF.min(),old.EF.max())


def rake(a: np.ndarray, shape: np.ndarray, prior: float = PRIOR,
         max_steps: int = IPF_STEPS, tolerance: float = IPF_TOL) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """IPF row targets plus original particle fuel columns; failure returns native.

    The optional flat-row independent seed preserves fuel columns but creates
    positive cells. It is an explicit support relaxation, not physical dispatch.
    """
    a, shape = np.asarray(a, float), np.asarray(shape, float)
    if shape.ndim == 1:
        shape = np.broadcast_to(shape, (len(a), 24))
    if (a.ndim != 3 or a.shape[1:] != (24, 8) or shape.shape != (len(a), 24)
            or not np.isfinite(a).all() or np.any(a < 0) or np.any(a.sum((1, 2)) <= 0)
            or not np.isfinite(shape).all() or np.any(shape <= 0)
            or not np.allclose(shape.mean(1), 1., rtol=0, atol=1e-12)
            or not 0 <= prior <= 1 or max_steps < 1 or tolerance <= 0):
        raise ValueError('Nonnegative particles and finite positive mean-one row shapes required')
    columns, energy = a.sum(1), a.sum((1, 2))
    targets = shape*energy[:, None]/24
    seed = (1-prior)*a+prior*columns[:, None]/24
    x = seed.copy()
    for iteration in range(1, max_steps+1):
        rows = x.sum(2)
        x *= np.divide(targets, rows, out=np.zeros_like(rows), where=rows > 0)[..., None]
        sums = x.sum(1)
        x *= np.divide(columns, sums, out=np.zeros_like(sums), where=sums > 0)[:, None]
        error = np.maximum(np.abs(x.sum(2)-targets).max(1), np.abs(x.sum(1)-columns).max(1))/energy
        converged = error <= tolerance
        if converged.all():
            break
    iterations = np.full(len(a),iteration,dtype=np.int64)
    error = np.maximum(np.abs(x.sum(2)-targets).max(1), np.abs(x.sum(1)-columns).max(1))/energy
    # Test final iterates, not a transient convergence flag.
    converged = np.isfinite(x).all((1, 2)) & (error <= tolerance)
    output = np.where(converged[:, None, None], x, a)
    created = (a == 0) & (output > 0)
    return output, dict(converged=converged, iterations=iterations, attempted_margin_error=error,
        final_fuel_column_error=np.abs(output.sum(1)-columns).max(1)/energy,
        created_positive_cells=created.sum((1, 2)), created_positive_mass=(output*created).sum((1, 2))/energy)


def sample_donors(bank: Any, mix: np.ndarray, day: pd.Timestamp,
                  shape: np.ndarray | None = None) -> np.ndarray:
    """Native kernel, optionally multiplied by a fixed soft G-shape kernel."""
    weights, *_ = bank._weights(mix, day, 'recompose_joint', .3, np.ones(len(bank.days_), bool))
    if shape is not None:
        rows = (bank.q_*mix[None, None]).sum(2)
        level = rows.mean(1,keepdims=True)
        profiles = np.divide(rows,level,out=np.ones_like(rows),where=level>0)
        distance = np.square(np.log(np.maximum(profiles, 1e-12))-np.log(shape)).mean(1)
        # No row shape is identified for a zero-energy donor under this mix.
        distance = np.where(level[:,0]>0,distance,0.)
        log_weight = np.log(np.maximum(weights, np.finfo(float).tiny))-distance/BANDWIDTH**2
        weights = np.exp(log_weight-log_weight.max()); weights /= weights.sum()
    cdf = np.cumsum(weights); cdf[-1] = 1.
    u = bank._rng(SEED, day).random((SAMPLES, 8))[:, 0]
    if shape is not None:
        last = np.flatnonzero(weights>0)[-1]
        cdf[last:] = 1.
        return np.minimum(np.searchsorted(cdf,u,side='right'),last)
    return np.searchsorted(cdf, u, side='right')


def particles(bank: Any, mix: np.ndarray, draws: np.ndarray, day: pd.Timestamp,
              pool: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Seven fixed arms, each exactly 128 complete-day particles."""
    ids = sample_donors(bank, mix, day)
    q = bank.q_[ids]
    native = recompose_paths(draws, q)
    raw = np.repeat(native[None], len(METHODS), axis=0)
    audit: dict[str, Any] = dict(donor_indices=ids, analog_donor_indices=ids.copy(),
        mean_shape=np.ones(24), sampled_shape=np.ones((SAMPLES, 24)), rake={})
    if not len(pool):
        return raw, audit
    mean = pool.mean(0); mean /= mean.mean()
    u = bank._rng(SEED+160, day).random(SAMPLES)
    sampled = pool[np.minimum((u*len(pool)).astype(int), len(pool)-1)]
    audit.update(mean_shape=mean, sampled_shape=sampled)
    a = draws[:, None]*q
    raw[1] = generation_cif(row_only(a, mean), native)
    analog = sample_donors(bank, mix, day, mean)
    raw[2] = recompose_paths(draws, bank.q_[analog]); audit['analog_donor_indices'] = analog
    for index, target, prior in ((3, mean, 0.), (4, mean, PRIOR), (5, sampled, PRIOR)):
        positive = a.sum((1,2))>0
        adjusted = a.copy()
        evidence = dict(converged=np.zeros(SAMPLES,dtype=bool),iterations=np.zeros(SAMPLES,dtype=np.int64),
            attempted_margin_error=np.zeros(SAMPLES),final_fuel_column_error=np.zeros(SAMPLES),
            created_positive_cells=np.zeros(SAMPLES,dtype=np.int64),created_positive_mass=np.zeros(SAMPLES),
            zero_energy_fallback=~positive)
        if positive.any():
            row_target = target if target.ndim==1 else target[positive]
            adjusted[positive], valid = rake(a[positive], row_target, prior)
            for name,value in valid.items():
                evidence[name][positive] = value
        values = generation_cif(adjusted, native)
        # Degenerate row fallback must preserve the original native formula.
        raw[index] = np.where(evidence['converged'][:, None], values, native)
        audit['rake'][METHODS[index]] = evidence
    raw[6, 1::2] = raw[4, 1::2]  # 64 native +64 rake, never path averaging.
    if not np.allclose(raw[1], native, rtol=0, atol=1e-9):
        raise ValueError('Row-scaling negative control changed CIF')
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1000):
        raise ValueError('Invalid particle CIF')
    return raw, audit


def prepare(output: Path) -> dict[str, Any]:
    """Freeze only known-source dependencies; never open target arrays or labels."""
    output = Path(output).resolve()
    if (output/'freeze.json').exists() or (output/'predictions').exists():
        raise FileExistsError('Preserve existing/partial pilot output')
    prior = read(REFERENCE/'prefit_freeze.json')
    if prior['sources'] != list(SOURCES) or prior['proper_start'] != '2024-01-25':
        raise ValueError('Wrong V14 source registry/proper support')
    seal = read(REFERENCE/'source_prediction_seal.json')
    native, coarse = {}, {}
    manifest_path = ROOT/'docs/evidence/weather-coarse-v14-verification.json'
    evidence = read(manifest_path)
    paths = [REFERENCE/'prefit_freeze.json', REFERENCE/'source_prediction_seal.json',
             REFERENCE/'source_choices.json', REFERENCE/'source_completed.json',
             manifest_path, Path(prior['data'])/'total_generation.csv']
    for region in SOURCES:
        entries = [e for e in seal['entries'] if (e['region'], e['budget'], e['model']) == (region, 'B', 'native_recompose')]
        if len(entries) != 1:
            raise ValueError('Unique V14 source native required')
        path = REFERENCE/entries[0]['path']
        verify({str(path): entries[0]['sha256']})
        native[region], coarse[region] = str(path), prior['coarse_paths'][f'source/{region}']
        paths += [path, Path(coarse[region]), Path(prior['source'])/f'{region}_hourly.csv']
    verify({str(p):evidence['artifact_hashes'][str(p.relative_to(ROOT))]
            for p in paths if p.is_relative_to(REFERENCE)})
    verify({str(p): prior['input_hashes'][str(p)] for p in paths if str(p) in prior['input_hashes']})
    verify(prior['code_hashes'])
    code = {**prior['code_hashes'], **{str(ROOT/p): sha(ROOT/p) for p in OWN}}
    frozen = dict(version=16, branch='B_generation_context_source_pilot', created_utc=now(),
        source=prior['source'], telemetry=str(Path(prior['data'])/'total_generation.csv'),
        native_paths=native, coarse_paths=coarse, sources=list(SOURCES), lags=list(LAGS),
        windows=list(WINDOWS), methods=list(METHODS), factors=list(FACTORS), samples=SAMPLES,
        prior=PRIOR, ipf_tolerance=IPF_TOL, ipf_steps=IPF_STEPS, analog_bandwidth=BANDWIDTH,
        seed=SEED, new_H2_calibration=False, target_opened=False, unseen_opened=False,
        native_reference_factor=read(REFERENCE/'source_choices.json')['native']['factor'],
        code_hashes=code, input_hashes={str(p): sha(p) for p in paths}, environment=old.v10.environment())
    output.mkdir(parents=True, exist_ok=True)
    write(output/'freeze.json', frozen)
    (output/'predictions').mkdir()
    return frozen


def predict(output: Path) -> None:
    frozen = prepare(output)
    freeze_sha = sha(output/'freeze.json')
    verify({**frozen['code_hashes'], **frozen['input_hashes']})
    frames = {}
    for region in SOURCES:
        frame = read_year(Path(frozen['source'])/f'{region}_hourly.csv', 2024)
        times = pd.to_datetime(frame.time, utc=True)
        frames[region] = frame.loc[(times >= pd.Timestamp('2024-01-25', tz='UTC')) &
                                   (times < pd.Timestamp('2024-07-01', tz='UTC'))].copy()
    telemetry = read_telemetry(Path(frozen['telemetry']), pd.Timestamp('2025-06-23', tz='UTC'))
    entries = []
    for region in SOURCES:
        bank = old._bank(frames, tuple(r for r in SOURCES if r != region))
        with np.load(frozen['native_paths'][region], allow_pickle=False) as z:
            baseline = {k: z[k].copy() for k in ('origins', 'raw_ensemble', 'donor_indices', 'point', 'center')}
        days = pd.DatetimeIndex(pd.to_datetime(baseline['origins'], utc=True))
        if not days.equals(pd.date_range('2025-01-01', '2025-06-30', tz='UTC')):
            raise ValueError('Common181forecast days required')
        with np.load(frozen['coarse_paths'][region], allow_pickle=False) as z:
            pos = pd.Index(z['origins']).get_indexer(baseline['origins'])
            if np.any(pos < 0):
                raise ValueError('Missing coarse origin')
            mixes, draws = z['mix'][pos], z['draws'][pos]
        for lag in LAGS:
            for window in WINDOWS:
                raws, donors, analogs, means, sampled, availability, diagnostics = [], [], [], [], [], [], []
                for i, day in enumerate(days):
                    pool, record = history_shapes(telemetry[region], day, lag, window)
                    values, audit = particles(bank, mixes[i], draws[i], day, pool)
                    if (not np.array_equal(audit['donor_indices'], baseline['donor_indices'][i]) or
                            not np.array_equal(values[0], baseline['raw_ensemble'][i])):
                        raise ValueError('Native regeneration not exactly V14')
                    raws.append(values); donors.append(audit['donor_indices']); analogs.append(audit['analog_donor_indices'])
                    means.append(audit['mean_shape']); sampled.append(audit['sampled_shape']); availability.append(record)
                    diagnostics.append({m: {k: np.asarray(v).tolist() for k, v in e.items()} for m, e in audit['rake'].items()})
                raw = np.stack(raws, axis=1)
                center = raw.mean(2); center[0] = baseline['center']
                if not np.array_equal(np.clip(raw[0],0,1000).mean(1), baseline['point']):
                    raise ValueError('Native clipped point identity failed')
                path = output/'predictions'/f'source-{region}-lag{lag}-window{window}.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, origins=baseline['origins'], methods=np.array(METHODS),
                        raw_ensemble=raw, center=center, donor_indices=np.asarray(donors),
                        analog_donor_indices=np.asarray(analogs), mean_generation_shape=np.asarray(means),
                        sampled_generation_shape=np.asarray(sampled), coarse_mix=mixes, coarse_draws=draws)
                audit_path = path.with_suffix('.json')
                write(audit_path, dict(region=region, lag=lag, window=window, training_regions=list(bank.source_regions_),
                    telemetry_definition='signed BA net generation, soft historical shape only',
                    availability=availability, raking=diagnostics, evaluation_labels_read=False))
                entries.append(dict(region=region, lag=lag, window=window, path=str(path), sha256=sha(path),
                                    audit_path=str(audit_path), audit_sha256=sha(audit_path)))
                print(f'Predicted {region} lag{lag} window{window}', flush=True)
    verify({**frozen['code_hashes'], **frozen['input_hashes'], str(output/'freeze.json'):freeze_sha})
    write(output/'prediction_seal.json', dict(created_utc=now(), freeze_sha256=freeze_sha,
        entries=entries, source_evaluation_labels_read=False, target_opened=False))


def verify_predictions(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed before evaluation labels if any grid/archive/byte changed."""
    frozen, seal = read(output/'freeze.json'), read(output/'prediction_seal.json')
    verify({**frozen['code_hashes'], **frozen['input_hashes']})
    if (seal['freeze_sha256'] != sha(output/'freeze.json') or frozen['methods'] != list(METHODS)
            or frozen['sources'] != list(SOURCES) or frozen['lags'] != list(LAGS)
            or frozen['windows'] != list(WINDOWS) or frozen['factors'] != list(FACTORS)
            or frozen['samples'] != SAMPLES or seal['source_evaluation_labels_read'] is not False
            or frozen['prior'] != PRIOR or frozen['ipf_tolerance'] != IPF_TOL
            or frozen['ipf_steps'] != IPF_STEPS or frozen['analog_bandwidth'] != BANDWIDTH
            or frozen['seed'] != SEED or frozen['environment'] != old.v10.environment()
            or frozen['new_H2_calibration'] is not False or frozen['target_opened'] is not False
            or frozen['unseen_opened'] is not False):
        raise ValueError('Changed source pilot contract')
    expected = {(r,l,w) for r in SOURCES for l in LAGS for w in WINDOWS}
    if len(seal['entries']) != len(expected) or {(e['region'],e['lag'],e['window']) for e in seal['entries']} != expected:
        raise ValueError('Complete28archive grid required before source labels')
    for e in seal['entries']:
        verify({e['path']:e['sha256'],e['audit_path']:e['audit_sha256']})
        if not Path(e['path']).resolve().is_relative_to((output/'predictions').resolve()):
            raise ValueError('Archive outside immutable output')
        with np.load(e['path'], allow_pickle=False) as z:
            if (z['raw_ensemble'].shape != (len(METHODS),181,SAMPLES,24)
                    or z['methods'].tolist() != list(METHODS) or not np.isfinite(z['raw_ensemble']).all()):
                raise ValueError('Invalid pilot archive')
    return frozen, seal


def score(output: Path) -> None:
    if (output/'completed.json').exists() or (output/'day_metrics.csv').exists():
        raise FileExistsError('Preserve completed/partial scores')
    frozen, seal = verify_predictions(output)
    parents = {str(output/n):sha(output/n) for n in ('freeze.json','prediction_seal.json')}
    rows = []
    for region in SOURCES:
        truth_days, truth = old.v10._truth_lookup(Path(frozen['source']), 'source', region)
        for entry in (e for e in seal['entries'] if e['region'] == region):
            with np.load(entry['path'], allow_pickle=False) as z:
                keep, y = old.v10._match_truth(z['origins'],truth_days,truth)
                for j, method in enumerate(METHODS):
                    ensemble = np.clip(z['raw_ensemble'][j,keep],0,1000)
                    point = ensemble.mean(1)
                    for factor in FACTORS:
                        scaled = np.clip(point[:,None]+factor*(ensemble-point[:,None]),0,1000)
                        metrics = {**old.v10.score_arrays(point,scaled,y),
                                   **old._extra_metrics(point,scaled,y,z['center'][j,keep])}
                        rows.extend(dict(region=region,origin=pd.Timestamp(int(day),tz='UTC').isoformat(),
                            lag=entry['lag'],window=entry['window'],method=method,factor=factor,
                            **{k:float(v[i]) for k,v in metrics.items()}) for i,day in enumerate(z['origins'][keep]))
    day = pd.DataFrame(rows)
    keys = ['region','origin','lag','window','method','factor']
    if day.duplicated(keys).any() or not np.isfinite(day[list(old.METRICS)]).all().all():
        raise ValueError('Invalid score rows')
    for lag in LAGS:
        for window in WINDOWS:
            subset = day.loc[day.lag.eq(lag)&day.window.eq(window)]
            if len(subset[['region','origin']].drop_duplicates()) != 1145:
                raise ValueError('Changed native1145validday support')
    region_table = day.groupby(['region','lag','window','method','factor'])[list(old.METRICS)].mean().reset_index()
    macro = region_table.groupby(['lag','window','method','factor'])[list(old.METRICS)].mean().reset_index()
    worst = region_table.groupby(['lag','window','method','factor']).empirical_coverage90.min().rename('worst_coverage').reset_index()
    macro = macro.merge(worst,validate='one_to_one')
    outputs = []
    for name, table in [('day_metrics',day),('region_metrics',region_table),('macro_metrics',macro)]:
        path = output/f'{name}.csv'; table.to_csv(path,index=False,mode='x'); outputs.append(path)
    verify_predictions(output); verify(parents)
    write(output/'completed.json',dict(created_utc=now(),version=16,branch=frozen['branch'],
        prediction_seal_sha256=sha(output/'prediction_seal.json'),freeze_sha256=parents[str(output/'freeze.json')],
        output_hashes={str(p):sha(p) for p in outputs},new_H2_calibration=False,
        target_opened=False,unseen_opened=False,coverage_guarantee=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('predict','score'),required=True)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    (predict if args.phase=='predict' else score)(output)


if __name__ == '__main__':
    main()
