"""Frozen source-only V11 weather shape transport and total-risk comparison.

The 2024H2 calibration errors include coarse forecast error. Total-risk arms
therefore use a deterministic central composition, never coarse particles.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import run_information_value_v10 as v10
from scripts.weather_shapes_v11 import ShapeModel, ScaleModel, transport, residual_paths, split_profiles
from transcif.research.dynamic_v2 import EF, read_year, daily_profiles
from transcif.research.fuel_transport_v7 import FuelTrajectoryBank, recompose_paths

SOURCES, TARGETS = v10.SOURCES, v10.TARGETS
SEED, N_SAMPLES, FACTORS = 17023, 128, (.75, 1., 1.25, 1.5, 2.)
PROTOCOL = ROOT / 'docs/WEATHER_RECOMPOSITION_V11_PROTOCOL.md'
V9_ROOT = v10.V9_ROOT
sha, read, write, now = v10.sha, v10.read, v10.write, v10.now
DAY_KEYS, REGION_KEYS = v10.DAY_KEYS, v10.REGION_KEYS
METRICS = (*v10.METRICS, 'level_crps', 'shape_crps', 'level_coverage90',
           'level_width90', 'shape_coverage90', 'shape_width90', 'center_level_mae', 'center_shape_mae', 'centered_shape_energy_score')


def configurations() -> list[dict[str, Any]]:
    result = [dict(budget=b, model=m, primary_eligible=b == 'W' and m in
                   ('donor', 'global_total', 'hetero_total'))
              for b in ('B', 'W') for m in
              ('donor', 'donor_no_shape_risk', 'global_total', 'hetero_total')]
    return result + [dict(budget=b, model=m, primary_eligible=False) for b, m in
                     [('B', 'native_recompose'), ('B', 'simple_anchor'),
                      ('W', 'global_Brisk'), ('W', 'hetero_Brisk')]]


def allowed_sources(excluded: str | None = None) -> tuple[str, ...]:
    if excluded is not None and excluded not in SOURCES:
        raise ValueError('Outer exclusion must be a registered source')
    return tuple(r for r in SOURCES if r != excluded)


def _code_paths():
    return sorted(set([Path(__file__).resolve(), PROTOCOL,
                       ROOT / 'scripts/weather_shapes_v11.py',
                       ROOT / 'scripts/analyze_weather_recomposition_v11.py',
                       ROOT / 'scripts/run_information_value_v10.py',
                       ROOT / 'scripts/run_coarse_dynamics_v9.py',
                       ROOT / 'scripts/risk_paths_v9.py', *v10._code_paths()]))


def _coarse_paths():
    paths = {}
    for stage, regions in [('source', SOURCES), ('target', TARGETS)]:
        seal = read(V9_ROOT / f'{stage}_prediction_seal.json')
        for region in regions:
            entries = [e for e in seal['coarse_entries'] if e['region'] == region
                       and int(e['delay']) == 0 and e['method'] == 'harmonic_ridge']
            if len(entries) != 1:
                raise ValueError('Missing unique archived harmonic coarse prediction')
            entry = entries[0]
            path = V9_ROOT / entry['path']
            if sha(path) != entry['sha256']:
                raise ValueError('Archived harmonic coarse bytes changed')
            paths[f'{stage}/{region}'] = str(path.resolve())
    return paths


def freeze(output: Path, data: Path, *, weather_csv: Path | None = None,
           points_json: Path | None = None) -> dict[str, Any]:
    output, data = Path(output).resolve(), Path(data).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh immutable V11 output directory')
    if weather_csv is None or points_json is None:
        raise ValueError('V11 requires the frozen weather proxy and point registry')
    # Reuse the complete V10 provenance validation in an isolated temporary
    # directory, then extend its manifest before publishing the V11 freeze.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='v11-provenance-') as temp:
        provenance = v10.freeze(Path(temp) / 'validation', data,
                                weather_csv=weather_csv, points_json=points_json)
    coarse = _coarse_paths()
    paths = [*map(Path, coarse.values()), V9_ROOT / 'prefit_freeze.json',
             V9_ROOT / 'source_prediction_seal.json', V9_ROOT / 'target_prediction_seal.json']
    provenance['input_hashes'].update({str(p): sha(p) for p in paths})
    provenance.update(created_utc=now(), version=11, budgets=['B', 'W'], lags=[7],
                      models=sorted({c['model'] for c in configurations()}),
                      configurations=configurations(), proper_start='2024-01-25',
                      calibration_period='2024H2 source inner LOSO end-to-end forecast errors',
                      target_scope='Previously viewed 2026H1 development regions',
                      coarse_paths=coarse, code_hashes={str(p): sha(p) for p in _code_paths()})
    output.mkdir(parents=True)
    (output / 'predictions').mkdir()
    write(output / 'prefit_freeze.json', provenance)
    return provenance


def verify(output: Path) -> dict[str, Any]:
    frozen = read(Path(output) / 'prefit_freeze.json')
    for path, digest in {**frozen['input_hashes'], **frozen['code_hashes']}.items():
        if sha(Path(path)) != digest:
            raise ValueError('Frozen V11 bytes changed: ' + path)
    if (frozen['version'] != 11 or frozen['configurations'] != configurations()
            or frozen['sources'] != list(SOURCES) or frozen['targets'] != list(TARGETS)
            or frozen['proper_start'] != '2024-01-25' or frozen['samples'] != N_SAMPLES
            or frozen['factors'] != list(FACTORS) or frozen['seed'] != SEED
            or frozen['environment'] != v10.environment()):
        raise ValueError('V11 frozen grid/environment changed')
    return frozen


def _coarse(frozen, stage, region, days):
    path = Path(frozen['coarse_paths'][f'{stage}/{region}'])
    if sha(path) != frozen['input_hashes'][str(path)]:
        raise ValueError('Coarse archive changed')
    with np.load(path, allow_pickle=False) as a:
        ids = pd.Index(a['origins']).get_indexer(days.as_unit('ns').asi8)
        if (ids < 0).any():
            raise ValueError('Coarse archive lacks evaluation days')
        mix, draws = a['mix'][ids], a['draws'][ids]
    if (mix.shape != (len(days), 8) or draws.shape != (len(days), N_SAMPLES, 8)
            or not np.isfinite(draws).all() or np.any(draws < 0)
            or not np.allclose(draws.sum(-1), 1) or not np.allclose(mix.sum(-1), 1)):
        raise ValueError('Invalid archived coarse particles')
    return mix, draws


def _load_training(frozen, stage):
    source, target = Path(frozen['source']), Path(frozen['target'])
    sm, tm = pd.read_csv(source / 'monthly.csv'), pd.read_csv(target / 'monthly.csv')
    telemetry = pd.read_csv(Path(frozen['data']) / 'total_generation.csv')
    weather, points = v10._weather(Path(frozen['weather_csv'])), v10._points(Path(frozen['points_json']))
    frames, cache = {}, {}
    for region in SOURCES:
        frame = read_year(source / f'{region}_hourly.csv', 2024)
        dates = pd.to_datetime(frame.time, utc=True)
        frames[region] = frame.loc[(dates >= pd.Timestamp('2024-01-25', tz='UTC')) &
                                   (dates < pd.Timestamp('2024-07-01', tz='UTC'))].copy()
        bank = FuelTrajectoryBank([region]).fit({region: frames[region]})
        days, truth = daily_profiles(frame)
        mask = (days >= pd.Timestamp('2024-07-01', tz='UTC')) & (days < pd.Timestamp('2025-01-01', tz='UTC'))
        for split, ds in [('proper', bank.days_), ('calibration', days[mask])]:
            record = v10._feature_record(sm, telemetry, weather, points, region, ds, 7)
            if split == 'proper':
                record.update(q=bank.q_, support=bank.fuel_support_)
            else:
                record['truth'] = truth[mask]
            cache[region, split] = record
    for region in SOURCES if stage == 'source' else TARGETS:
        cache[region, 'evaluation'] = v10._feature_record(
            sm if stage == 'source' else tm, telemetry, weather, points, region,
            v10._evaluation_days(stage, True), 7)
    return frames, cache


class ShapeCache:
    """Fit each source subset once per budget, retaining predictions, not trees."""
    def __init__(self, records: dict, subsets: list[tuple[str, ...]],
                 queries: dict[tuple[str, ...], list[str]]) -> None:
        self.records, self.requests, self.predictions, self.audit = records, {}, {}, {}
        for subset in subsets:
            subset = tuple(sorted(subset))
            for r in subset:
                smaller = tuple(x for x in subset if x != r)
                self.request(smaller, r, 'proper')
                self.request(smaller, r, 'calibration')
                for donor in smaller:
                    self.request(tuple(x for x in smaller if x != donor), donor, 'proper')
            for r in queries[subset]:
                self.request(subset, r, 'evaluation')
    def request(self, subset: tuple[str, ...], region: str, split: str) -> None:
        self.requests.setdefault(tuple(sorted(subset)), set()).add((region, split))
    def get(self, subset: tuple[str, ...], budget: str, region: str,
            split: str) -> np.ndarray:
        subset = tuple(sorted(subset)); key = subset, budget
        if key not in self.predictions:
            rows = [self.records[r, 'proper'] for r in subset]
            model = ShapeModel().fit(
                np.concatenate([r['features'][budget] for r in rows]),
                np.concatenate([r['q'] for r in rows]),
                np.concatenate([np.repeat(r, len(self.records[r, 'proper']['days'])) for r in subset]),
                np.concatenate([r['support'] for r in rows]),
                days=pd.DatetimeIndex(np.concatenate([r['days'].to_numpy() for r in rows])))
            self.predictions[key] = {k: model.predict(self.records[k]['features'][budget])
                                     for k in sorted(self.requests[subset])}
            self.audit['|'.join(subset) + ':' + budget] = model.audit_
            print('V11 shape subset', ','.join(subset), budget, flush=True)
        return self.predictions[key][region, split]


def _bank(frames, subset):
    subset = tuple(sorted(subset))
    if not set(subset) <= set(SOURCES) or len(subset) < 2:
        raise ValueError('Donor bank requires permitted source regions')
    bank = FuelTrajectoryBank(subset).fit({r: frames[r] for r in subset})
    if bank.days_.min() < pd.Timestamp('2024-01-25', tz='UTC'):
        raise ValueError('B/W proper cohort must start January 25')
    return bank


def _donor_predictions(bank, shapes, budget):
    subset = bank.source_regions_
    return np.concatenate([shapes.get(tuple(r for r in subset if r != donor), budget,
                                      donor, 'proper') for donor in subset])


def center_paths(bank: FuelTrajectoryBank, mixes: np.ndarray, days: pd.DatetimeIndex,
                 query_shapes: np.ndarray, donor_predictions: np.ndarray,
                 diagnostics: dict[str, Any] | None = None) -> np.ndarray:
    """Full-bank corrected mean q, followed by the ratio (not mean ratio)."""
    result = []
    projection_sum = negative_count = projection_count = 0.
    amplitude_error = 0.
    amplitudes, donor_shapes = split_profiles(bank.q_)
    valid = np.broadcast_to((bank.fuel_support_ & (amplitudes > 0))[:, None, :], bank.q_.shape)
    for mix, day, shape in zip(mixes, days, query_shapes):
        weights, *_ = bank._weights(mix, day, 'recompose_joint', .3, np.ones(len(bank.days_), bool))
        q = transport(bank.q_, shape, donor_predictions, bank.fuel_support_)
        if diagnostics is not None:
            before = shape + donor_shapes - donor_predictions
            after = np.divide(q, amplitudes[:, None, :], out=np.ones_like(q), where=amplitudes[:, None, :] > 0)
            projection_sum += float(np.abs(after-before)[valid].sum())
            negative_count += int((before[valid] < 0).sum())
            projection_count += int(valid.sum())
            amplitude_error = max(amplitude_error, float(np.abs(q.mean(1)-bank.q_.mean(1)).max()))
        result.append(recompose_paths(mix, np.einsum('d,dhf->hf', weights, q)))
    if diagnostics is not None:
        diagnostics.update(mean_absolute_projection_change=projection_sum / max(projection_count, 1),
                           fraction_negative_before_projection=negative_count / max(projection_count, 1),
                           projection_supported_active_values=int(projection_count),
                           fullbank_q_mean_max_error=amplitude_error)
    return np.stack(result)


def _calibration(frames, records, shapes, subset, budget):
    errors, features, regions, days = [], [], [], []
    audits = []
    for region in sorted(subset):
        inner = tuple(r for r in subset if r != region)
        bank = _bank(frames, inner)
        record = records[region, 'calibration']
        center = center_paths(bank, record['features']['B'][:, :8], record['days'],
                              shapes.get(inner, budget, region, 'calibration'),
                              _donor_predictions(bank, shapes, budget))
        errors.append(record['truth'] - center)
        features.append(record['features'][budget])
        regions.extend([region] * len(center)); days.extend(record['days'])
        audits.append(dict(calibration_region=region, training_regions=list(inner),
                           donor_regions=list(bank.source_regions_)))
    errors, features, regions, days = np.concatenate(errors), np.concatenate(features), np.array(regions), pd.DatetimeIndex(days)
    scales = np.empty((len(errors), 2))
    for region in sorted(subset):
        keep = regions != region
        model = ScaleModel().fit(features[keep], errors[keep], regions[keep],
                                 exclude_region=region, days=days[keep])
        scales[~keep] = model.predict(features[~keep])
    final = ScaleModel().fit(features, errors, regions, days=days)
    return dict(errors=errors, features=features, regions=regions, days=days,
                scales=scales, model=final, audit=audits)


def _equal_region_weights(regions):
    return np.array([1 / (len(set(regions)) * np.sum(regions == r)) for r in regions])


def _indices(bank, mixes, days, cal_regions):
    donor, residual = [], []
    cdf_r = np.cumsum(_equal_region_weights(cal_regions)); cdf_r[-1] = 1
    for mix, day in zip(mixes, days):
        weights, *_ = bank._weights(mix, day, 'recompose_joint', .3, np.ones(len(bank.days_), bool))
        cdf = np.cumsum(weights); cdf[-1] = 1
        u = bank._rng(SEED, day).random((N_SAMPLES, 8))
        donor.append(np.searchsorted(cdf, u[:, 0], side='right'))
        residual.append(np.searchsorted(cdf_r, u[:, 1], side='right'))
    return np.array(donor), np.array(residual)


def _predict_region(frames, records, shapes, subset, region, mixes, draws, calibrations):
    bank, record = _bank(frames, subset), records[region, 'evaluation']
    days = record['days']
    donor_ids, residual_ids = _indices(bank, mixes, days, calibrations['B']['regions'])
    result = {}
    projection_audits = {}
    sampled_q = bank.q_[donor_ids]
    for budget in ('B', 'W'):
        query = shapes.get(subset, budget, region, 'evaluation')
        donor_pred = _donor_predictions(bank, shapes, budget)
        projection_audit = {}
        center = center_paths(bank, mixes, days, query, donor_pred, projection_audit)
        projection_audits[budget] = projection_audit
        for mode, carry in [('donor', True), ('donor_no_shape_risk', False)]:
            corrected = transport(sampled_q, query[:, None], donor_pred[donor_ids],
                                  bank.fuel_support_[donor_ids], keep_shape_residual=carry)
            raw = np.stack([recompose_paths(draws[i], corrected[i]) for i in range(len(days))])
            result[budget, mode] = dict(raw=raw, center=center, amplitude_error=float(np.max(np.abs(corrected.mean(2)-sampled_q.mean(2)))))
        for mode in ('global_total', 'hetero_total'):
            cal = calibrations[budget]
            scale_args = {} if mode == 'global_total' else dict(
                scales=cal['model'].predict(record['features'][budget]), calibration_scales=cal['scales'])
            raw = residual_paths(center, cal['errors'], residual_ids, **scale_args)
            result[budget, mode] = dict(raw=raw, center=center)
        if budget == 'W':
            cal = calibrations['B']
            for mode in ('global_Brisk', 'hetero_Brisk'):
                args = {} if mode == 'global_Brisk' else dict(
                    scales=cal['model'].predict(record['features']['B']), calibration_scales=cal['scales'])
                result[budget, mode] = dict(raw=residual_paths(center, cal['errors'], residual_ids, **args), center=center)
    native = np.stack([recompose_paths(draws[i], sampled_q[i]) for i in range(len(days))])
    simple = bank.raw_cif_[donor_ids] + ((draws - bank.donor_mix_[donor_ids]) @ EF)[..., None]
    for mode, raw in [('native_recompose', native), ('simple_anchor', simple)]:
        result['B', mode] = dict(raw=raw, center=raw.mean(1))
    audit = dict(training_regions=list(subset), donor_regions=list(bank.source_regions_),
                 excluded_evaluation_region=region if region in SOURCES else None,
                 proper_start='2024-01-25', source_calibration_regions=list(subset),
                 projection_by_budget=projection_audits,
                 calibration=[*calibrations['B']['audit']],
                 calibration_includes_coarse_error=True,
                 total_risk_uses_coarse_particles=False,
                 donor_rng='V9 joint inverse CDF, seed+UTC origin, column0; residual column1',
                 donor_amplitudes_preserved=True,
                 cif_daily_mean_preserved=False, target_hourly_used=False)
    return result, donor_ids, residual_ids, audit


def _grid(stage):
    return {(r, c['budget'], c['model']) for r in (SOURCES if stage == 'source' else TARGETS)
            for c in configurations()}


def check_entries(output: Path, frozen: dict[str, Any], stage: str,
                  seal: dict[str, Any]) -> None:
    entries = seal['entries']
    actual = {(e['region'], e['budget'], e['model']) for e in entries}
    if len(entries) != len(_grid(stage)) or actual != _grid(stage):
        raise ValueError('V11 prediction grid incomplete/duplicated')
    days = v10._evaluation_days(stage, True).as_unit('ns').asi8
    for e in entries:
        expected = allowed_sources(e['region'] if stage == 'source' else None)
        if set(e['audit']['training_regions']) != set(expected) or set(e['audit']['donor_regions']) != set(expected):
            raise ValueError('Outer region leaked into fit or donor bank')
        path = Path(output) / e['path']
        if sha(path) != e['sha256']:
            raise ValueError('V11 prediction bytes changed')
        with np.load(path, allow_pickle=False) as a:
            if (not np.array_equal(a['origins'], days) or a['ensemble'].shape != (len(days), N_SAMPLES, 24)
                    or a['point'].shape != (len(days), 24) or a['center'].shape != (len(days), 24)
                    or not np.isfinite(a['raw_ensemble']).all() or not np.isfinite(a['center']).all()
                    or not np.array_equal(a['ensemble'], np.clip(a['raw_ensemble'], 0, 1000))
                    or not np.allclose(a['point'], a['ensemble'].mean(1))):
                raise ValueError('Invalid V11 prediction archive')


def shape_energy_score(paths: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Multivariate energy score for centered paths, normalized L2 / sqrt(24)."""
    paths, truth = np.asarray(paths, float), np.asarray(truth, float)
    if paths.ndim != 3 or paths.shape[2] != 24 or truth.shape != (len(paths), 24):
        raise ValueError('Shape energy score needs days by samples by 24 paths')
    output = np.empty(len(paths))
    for start in range(0, len(paths), 32):
        x, y = paths[start:start+32], truth[start:start+32]
        first = np.sqrt(np.mean((x-y[:,None])**2, axis=2)).mean(1)
        norms = np.sum(x*x, axis=2)
        square = norms[:,:,None] + norms[:,None,:] - 2*np.einsum('dsh,dth->dst', x, x)
        distance = np.sqrt(np.maximum(square, 0)/24)
        output[start:start+32] = first - .5*distance.mean((1,2))
    return output


def _extra_metrics(point, ensemble, truth, center):
    def crps(draw, target):
        s = draw.shape[1]
        ordered = np.sort(draw, axis=1)
        coeff = 2 * np.arange(s) - s + 1
        return (np.abs(draw - target[:, None]).mean(1) -
                np.einsum('s,ds...->d...', coeff, ordered) / s**2).reshape(len(draw), -1).mean(1)
    level, target_level = ensemble.mean(2), truth.mean(1)
    shape, target_shape = ensemble - level[:, :, None], truth - target_level[:, None]
    lo, hi = int(np.ceil(.05 * ensemble.shape[1]))-1, int(np.ceil(.95 * ensemble.shape[1]))-1
    output = dict(level_crps=crps(level, target_level), shape_crps=crps(shape, target_shape),
                  center_level_mae=np.abs(center.mean(1)-truth.mean(1)),
                  center_shape_mae=np.abs((center-center.mean(1)[:,None])-target_shape).mean(1),
                  centered_shape_energy_score=shape_energy_score(shape, target_shape))
    for name, draw, target in [('level', level, target_level), ('shape', shape, target_shape)]:
        ordered = np.sort(draw, axis=1); lower, upper = ordered[:, lo], ordered[:, hi]
        output[name+'_coverage90'] = ((target >= lower) & (target <= upper)).reshape(len(draw), -1).mean(1)
        output[name+'_width90'] = (upper-lower).reshape(len(draw), -1).mean(1)
    return output


def score_phase(output: Path, frozen: dict[str, Any], stage: str,
                seal: dict[str, Any]) -> pd.DataFrame:
    output = Path(output)
    if not (output / f'{stage}_prediction_seal.json').is_file() or read(output / f'{stage}_prediction_seal.json') != seal:
        raise ValueError('Complete prediction seal required before evaluation labels')
    check_entries(output, frozen, stage, seal)
    rows = []
    data = Path(frozen[stage])
    for region in SOURCES if stage == 'source' else TARGETS:
        truth_days, truth = v10._truth_lookup(data, stage, region)
        for e in [e for e in seal['entries'] if e['region'] == region]:
            with np.load(output / e['path'], allow_pickle=False) as a:
                keep, aligned = v10._match_truth(a['origins'], truth_days, truth)
                point, ensemble, center = a['point'][keep], a['ensemble'][keep], a['center'][keep]
                for factor in FACTORS:
                    scaled = np.clip(ensemble.mean(1, keepdims=True)+factor*(ensemble-ensemble.mean(1, keepdims=True)), 0, 1000)
                    metrics = {**v10.score_arrays(point, scaled, aligned), **_extra_metrics(point, scaled, aligned, center)}
                    rows.extend(dict(stage=stage, region=region, origin=pd.Timestamp(int(day), tz='UTC').isoformat(),
                                     lag=7, budget=e['budget'], model=e['model'], factor=factor, semester='H1',
                                     **{k: float(v[i]) for k, v in metrics.items()})
                                for i, day in enumerate(a['origins'][keep]))
    day = pd.DataFrame(rows)
    if day.empty or day.duplicated(list(DAY_KEYS)).any() or not np.isfinite(day[list(METRICS)]).all().all():
        raise ValueError('Invalid V11 scores')
    folder = output / f'{stage}_scores'; folder.mkdir()
    day.to_csv(folder / 'day_metrics.csv', index=False)
    grouped = day.groupby(list(REGION_KEYS), sort=True)
    region = grouped[list(METRICS)].mean().reset_index(); region['days'] = grouped.size().to_numpy()
    region.to_csv(folder / 'region_metrics.csv', index=False)
    region.groupby(['stage', 'lag', 'budget', 'model', 'factor', 'semester'])[list(METRICS)].mean().reset_index().to_csv(folder / 'macro_metrics.csv', index=False)
    write(folder / 'score_seal.json', {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
    return pd.read_csv(folder / 'region_metrics.csv')  # canonical serialized values for selection


def selection(table: pd.DataFrame) -> dict[str, Any]:
    table = table.loc[table.stage.eq('source') & table.semester.eq('H1') & table.lag.eq(7)]
    expected = {(r, c['budget'], c['model'], f) for r in SOURCES for c in configurations() for f in FACTORS}
    actual = {(r.region, r.budget, r.model, float(r.factor)) for _, r in table.iterrows()}
    if len(table) != len(expected) or actual != expected:
        raise ValueError('Incomplete source selection grid')
    grouped = table.groupby(['budget', 'model', 'factor'], sort=True)
    ranks = grouped[list(METRICS)].mean().reset_index()
    ranks['worst_coverage'] = grouped.empirical_coverage90.min().to_numpy()
    ranks['violation'] = (np.maximum(.90-ranks.empirical_coverage90, 0) + np.maximum(ranks.empirical_coverage90-.95, 0) + np.maximum(.85-ranks.worst_coverage, 0))
    def choose(part):
        passed = part.loc[part.violation <= 1e-12]
        row = (passed if len(passed) else part).sort_values(([] if len(passed) else ['violation'])+['crps', 'budget', 'model', 'factor']).iloc[0]
        return dict(budget=str(row.budget), model=str(row.model), factor=float(row.factor),
                    status='passed' if row.violation <= 1e-12 else 'failed',
                    source_metrics={k: float(row[k]) for k in (*METRICS, 'worst_coverage', 'violation')})
    selections = {f"{c['budget']}--{c['model']}": choose(ranks.loc[ranks.budget.eq(c['budget']) & ranks.model.eq(c['model'])])
                  for c in configurations() if c['model'] not in ('global_Brisk', 'hetero_Brisk')}
    return dict(selections=selections, baseline=choose(ranks.loc[ranks.budget.eq('B') & ranks.model.isin(['donor', 'global_total', 'hetero_total'])]),
                primary=choose(ranks.loc[ranks.budget.eq('W') & ranks.model.isin(['donor', 'global_total', 'hetero_total'])]),
                native=selections['B--native_recompose'], simple=selections['B--simple_anchor'])


def validate_choices(output: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    output = Path(output); choice = read(output / 'source_choices.json')
    expected = selection(pd.read_csv(output / 'source_scores/region_metrics.csv'))
    if (any(choice.get(k) != v for k, v in expected.items()) or choice.get('target_scores_used') is not False
            or choice.get('source_score_seal_sha256') != sha(output / 'source_scores/score_seal.json')
            or choice.get('source_prediction_seal_sha256') != sha(output / 'source_prediction_seal.json')):
        raise ValueError('V11 source choice does not reproduce canonical source scores')
    return choice


def check_score_grid(output: Path, frozen: dict[str, Any], stage: str) -> None:
    folder = Path(output) / f'{stage}_scores'
    day, region, macro = [pd.read_csv(folder / name) for name in ('day_metrics.csv','region_metrics.csv','macro_metrics.csv')]
    expected = {(stage, r, 7, b, m, f, 'H1') for r, b, m in _grid(stage) for f in FACTORS}
    actual = {tuple(row[k] for k in REGION_KEYS) for _, row in region.iterrows()}
    if len(region) != len(expected) or actual != expected or day.duplicated(list(DAY_KEYS)).any():
        raise ValueError('V11 score grid incomplete/duplicated')
    for r in SOURCES if stage == 'source' else TARGETS:
        groups = list(day.loc[day.region.eq(r)].groupby(['budget','model','factor']))
        support = set(groups[0][1].origin)
        if not support or any(set(g.origin) != support for _, g in groups):
            raise ValueError('V11 unequal scoring support')
    for table, keys, calculated in [
        (region, list(REGION_KEYS), day.groupby(list(REGION_KEYS))[list(METRICS)].mean().reset_index()),
        (macro, ['stage','lag','budget','model','factor','semester'], region.groupby(['stage','lag','budget','model','factor','semester'])[list(METRICS)].mean().reset_index())]:
        joined = table.merge(calculated, on=keys, validate='one_to_one', suffixes=('_saved','_computed'))
        if len(joined) != len(table) or not np.allclose(joined[[m+'_saved' for m in METRICS]], joined[[m+'_computed' for m in METRICS]], atol=1e-10):
            raise ValueError('V11 score aggregation mismatch')


def verify_phase(output: Path, stage: str) -> dict[str, Any]:
    if stage not in ('source', 'target'):
        raise ValueError('Unknown V11 phase')
    output = Path(output); frozen = verify(output)
    done, seal = read(output / f'{stage}_completed.json'), read(output / f'{stage}_prediction_seal.json')
    for field, path in [('prefit_freeze_sha256','prefit_freeze.json'), ('prediction_seal_sha256',f'{stage}_prediction_seal.json'),
                        ('score_seal_sha256',f'{stage}_scores/score_seal.json'), ('source_choices_sha256','source_choices.json')]:
        if done.get(field) != sha(output / path):
            raise ValueError('V11 completion detached from '+field)
    if (seal.get('evaluation_hourly_labels_read_preseal') is not False or seal.get('complete_grid') is not True
            or done.get('target_training_performed') is not False or done.get('target_selection_performed') is not False):
        raise ValueError('V11 information boundary changed')
    check_entries(output, frozen, stage, seal)
    folder = output / f'{stage}_scores'; scoreseal = read(folder / 'score_seal.json')
    if set(scoreseal) != {'day_metrics.csv','region_metrics.csv','macro_metrics.csv'}:
        raise ValueError('V11 score manifest incomplete')
    for name, digest in scoreseal.items():
        if sha(folder / name) != digest:
            raise ValueError('V11 score bytes changed')
    check_score_grid(output, frozen, stage); choice = validate_choices(output, frozen)
    if stage == 'target':
        verify_phase(output, 'source')
        if seal.get('source_choices_sha256') != sha(output / 'source_choices.json') or pd.Timestamp(choice['created_utc']) > pd.Timestamp(seal['created_utc']):
            raise ValueError('Target predictions precede frozen source choice')
    return seal


def run_phase(output: Path, data: Path, stage: str, *,
              weather_csv: Path | None = None, points_json: Path | None = None) -> None:
    if stage not in ('source', 'target'):
        raise ValueError('Unknown V11 phase')
    output, data = Path(output).resolve(), Path(data).resolve()
    if not output.exists():
        if stage != 'source':
            raise ValueError('Source phase must precede target')
        freeze(output, data, weather_csv=weather_csv, points_json=points_json)
    frozen = verify(output)
    if Path(frozen['data']) != data or (weather_csv is not None and Path(weather_csv).resolve() != Path(frozen['weather_csv'])) or (points_json is not None and Path(points_json).resolve() != Path(frozen['points_json'])):
        raise ValueError('Input paths differ from freeze')
    if stage == 'target':
        verify_phase(output, 'source')
    if any((output / f'{stage}_{s}').exists() for s in ['prediction_seal.json','scores','completed.json']):
        raise FileExistsError('V11 stage immutable')
    frames, records = _load_training(frozen, stage)
    queries = {tuple(sorted(allowed_sources(r))): [r] for r in SOURCES} if stage == 'source' else {tuple(sorted(SOURCES)): list(TARGETS)}
    shapes = ShapeCache(records, list(queries), queries)
    entries = []
    for subset, regions in queries.items():
        calibrations = {b: _calibration(frames, records, shapes, subset, b) for b in ('B','W')}
        for region in regions:
            record = records[region, 'evaluation']
            mixes, draws = _coarse(frozen, stage, region, record['days'])
            results, donors, residuals, audit = _predict_region(frames, records, shapes, subset, region, mixes, draws, calibrations)
            for (budget, mode), prediction in results.items():
                path = output / 'predictions' / f'{stage}-{region}-{budget}-{mode}.npz'
                if path.exists():
                    raise FileExistsError(path)
                raw = prediction['raw']; ensemble = np.clip(raw, 0, 1000)
                np.savez_compressed(path, origins=record['days'].as_unit('ns').asi8,
                                    point=ensemble.mean(1), ensemble=ensemble, center=prediction['center'],
                                    raw_ensemble=raw, donor_indices=donors, residual_indices=residuals)
                entries.append(dict(stage=stage, region=region, lag=7, budget=budget, model=mode,
                                    path=str(path.relative_to(output)), sha256=sha(path), audit={**audit,
                                    'donor_q_mean_max_error': prediction.get('amplitude_error', 0.),
                                    'raw_clip_fraction': float(np.mean(raw != ensemble)),
                                    'clip_point_shift_mae': float(np.abs(raw.mean(1)-ensemble.mean(1)).mean())}))
            print('Predicted V11', stage, region, flush=True)
    verify(output)
    seal = dict(created_utc=now(), entries=entries, complete_grid=True,
                evaluation_hourly_labels_read_preseal=False, source_2024_labels_only_preseal=True,
                target_training_performed=False, target_selection_performed=False,
                source_choices_sha256=sha(output / 'source_choices.json') if stage == 'target' else None,
                shape_fit_audits=shapes.audit)
    check_entries(output, frozen, stage, seal)
    write(output / f'{stage}_prediction_seal.json', seal)
    table = score_phase(output, frozen, stage, seal)
    if stage == 'source':
        write(output / 'source_choices.json', dict(created_utc=now(), **selection(table),
              target_scores_used=False, source_h2_used_for_selection=False,
              source_prediction_seal_sha256=sha(output / 'source_prediction_seal.json'),
              source_score_seal_sha256=sha(output / 'source_scores/score_seal.json')))
    write(output / f'{stage}_completed.json', dict(created_utc=now(),
          prefit_freeze_sha256=sha(output / 'prefit_freeze.json'),
          prediction_seal_sha256=sha(output / f'{stage}_prediction_seal.json'),
          score_seal_sha256=sha(output / f'{stage}_scores/score_seal.json'),
          source_choices_sha256=sha(output / 'source_choices.json'),
          target_training_performed=False, target_selection_performed=False))
    verify_phase(output, stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['source','target'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'results/information-value-v10-data')
    parser.add_argument('--weather-csv', type=Path)
    parser.add_argument('--points-json', type=Path)
    args = parser.parse_args()
    run_phase(args.output, args.data, args.phase, weather_csv=args.weather_csv, points_json=args.points_json)


if __name__ == '__main__':
    main()
